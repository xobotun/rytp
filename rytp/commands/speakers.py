"""The speakers command group (contracts §5, design §6, §10).

Every operation is defined once here and both surfaces are generated from
it, so the CLI and the TUI cannot drift. Handlers hold no SQL — that is
`rytp.diarize.store` — and no interaction — that is
`rytp.diarize.mapper`. What is left is argument shaping and rendering.

`speakers.map` is the exception that proves the rule: it is `cli_only`,
because its job is to *open* the interactive surface and a surface cannot
sensibly launch itself. Everything it then does is a `MappingSession`,
which the TUI test drives directly.
"""

from __future__ import annotations

from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register, resolve_video_id
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import diarizer_rows, load_diarizer
from rytp.diarize.embed import embedder_rows, load_embedder
from rytp.diarize.link import engines_in_play, suggest_for_label, suggest_for_video
from rytp.diarize.mapper import format_ms
from rytp.diarize.pipeline import (
    diarize_video,
    diarizer_name,
    embed_video_speakers,
    embedder_name,
    wav_for,
)
from rytp.models import NotFoundError, RytpError

_EMPTY = "—"


# -- the roster ------------------------------------------------------------


def speakers_add(
    db: Database, *, label: str, aliases: str = "", notes: str | None = None
) -> CommandResult:
    """Add one person to the global roster."""
    parsed = store.parse_aliases(aliases)
    with db.transaction():
        speaker_id, created = store.add_speaker(db, label, aliases=parsed, notes=notes)
    if not created:
        return CommandResult(
            message=f"{label} is already speaker {speaker_id}; "
            f"use `speakers alias` to add aliases"
        )
    collisions = store.find_alias_collisions(
        db, parsed, exclude_speaker_id=speaker_id
    )
    warning = ""
    if collisions:
        taken = ", ".join(sorted({alias for alias, _sid in collisions}))
        warning = f" (warning: {taken} also resolves to another speaker)"
    return CommandResult(message=f"added {label} as speaker {speaker_id}{warning}")


def speakers_list(db: Database) -> CommandResult:
    """Show the roster and how much of the corpus each person accounts for."""
    rows = tuple(
        (
            str(row.speaker_id),
            row.label,
            ", ".join(row.aliases) or _EMPTY,
            str(row.n_videos),
            str(row.n_labels),
        )
        for row in store.roster_rows(db)
    )
    return CommandResult(
        columns=("id", "person", "aliases", "videos", "labels"),
        rows=rows,
        message=None if rows else "the roster is empty; add someone with `speakers add`",
    )


def speakers_alias(db: Database, *, speaker: str, aliases: str) -> CommandResult:
    """Add aliases to a person already on the roster."""
    person = store.resolve_speaker(db, speaker)
    parsed = store.parse_aliases(aliases)
    if not parsed:
        raise RytpError("give at least one alias, comma-separated")
    with db.transaction():
        merged = store.add_aliases(db, person.id, parsed)
    return CommandResult(message=f"{person.label} now answers to {', '.join(merged)}")


def speakers_remove(db: Database, *, speaker: str, yes: bool = False) -> CommandResult:
    """Delete a person from the roster (contracts §5).

    Not `speakers unlink`. That detaches one label of one video; this
    deletes the person, and every label naming them goes back to being a
    voice nobody named. The labels survive — `video_speakers` rows are
    `ON DELETE SET NULL` — so the knowledge that a video had four distinct
    speakers outlives the roster entry.

    No files are touched, so contracts §5 asks for neither `--dry-run` nor
    `--yes`. It does silently undo mapping work, though, so past
    `C.SPEAKER_REMOVE_CONFIRM_LINKS` links it asks first.
    """
    person = store.resolve_speaker(db, speaker)
    labels = store.linked_labels(db, person.id)
    if len(labels) >= C.SPEAKER_REMOVE_CONFIRM_LINKS and not yes:
        videos = len({row.video_id for row in labels})
        raise RytpError(
            f"{person.label} is named on {len(labels)} labels across {videos} "
            f"videos; pass --yes to undo that mapping work "
            f"(to detach just one label, use `speakers unlink`)"
        )
    with db.transaction():
        unlinked = store.remove_speaker(db, person.id)
    plural = "" if unlinked == 1 else "s"
    return CommandResult(
        message=f"removed {person.label} from the roster; {unlinked} local "
        f"label{plural} kept, now unnamed"
    )


# -- labels and linking ----------------------------------------------------


def speakers_labels(db: Database, *, video: str) -> CommandResult:
    """Show one video's diarizer labels and who each of them is."""
    video_id = resolve_video_id(db, video)
    rows = store.label_rows(db, video_id)
    if not rows:
        return CommandResult(
            message=f"video {video_id} has no labels; diarize it first "
            f"(`rytp speakers diarize {video_id}`)"
        )
    return CommandResult(
        columns=("label", "words", "speech", "person", "voice", "diarizer"),
        rows=tuple(
            (
                row.local_label,
                str(row.n_words),
                format_ms(row.speech_ms),
                row.speaker_label or _EMPTY,
                "yes" if row.has_embedding else "no",
                row.engine,
            )
            for row in rows
        ),
        message=f"{sum(1 for r in rows if r.speaker_id is None)} still unnamed",
    )


def speakers_link(
    db: Database, *, video: str, label: str, speaker: str
) -> CommandResult:
    """Say which real person one of a video's local labels is."""
    video_id = resolve_video_id(db, video)
    row = store.find_label(db, video_id, label)
    person = store.resolve_speaker(db, speaker)
    with db.transaction():
        store.link_video_speaker(db, row.video_speaker_id, person.id)
    return CommandResult(
        message=f"video {video_id} {label} is {person.label} "
        f"({row.n_words} words follow along)"
    )


def speakers_unlink(db: Database, *, video: str, label: str) -> CommandResult:
    """Undo a link, leaving the local label as a voice nobody named."""
    video_id = resolve_video_id(db, video)
    row = store.find_label(db, video_id, label)
    with db.transaction():
        store.link_video_speaker(db, row.video_speaker_id, None)
    return CommandResult(message=f"video {video_id} {label} is nobody again")


# -- the two long ones -----------------------------------------------------


def speakers_diarize(
    db: Database, *, video: str, diarizer: str = "", embedder: str = ""
) -> CommandResult:
    """Work out who spoke when, and label this video's words.

    Opt-in per video: the largest GPU cost in the project (design §6), so
    nothing runs this on your behalf.
    """
    video_id = resolve_video_id(db, video)
    wav = wav_for(db, video_id)
    name = diarizer_name(db, diarizer)
    outcome = diarize_video(db, video_id, wav_path=wav, diarizer=load_diarizer(db, name))
    message = outcome.message()

    wanted = embedder_name(db, embedder)
    if wanted:
        embedded = embed_video_speakers(
            db, video_id, wav_path=wav, embedder=load_embedder(db, wanted)
        )
        message = f"{message}; {embedded.message()}"
    return CommandResult(message=message)


def speakers_embed(db: Database, *, video: str, embedder: str = "") -> CommandResult:
    """Give this video's voices their fingerprints, for cross-video suggestions.

    Runnable on its own, and cheap: the speech is reconstructed from the
    already-labelled words rather than by diarizing again.
    """
    video_id = resolve_video_id(db, video)
    wanted = embedder_name(db, embedder)
    if not wanted:
        raise RytpError(
            "no embedder configured; pass --embedder or set the "
            f"{C.SETTINGS_EMBEDDER} setting"
        )
    if not store.label_rows(db, video_id):
        raise NotFoundError(f"video {video_id} has no labels; diarize it first")
    outcome = embed_video_speakers(
        db, video_id, wav_path=wav_for(db, video_id), embedder=load_embedder(db, wanted)
    )
    return CommandResult(message=outcome.message())


# -- the queue -------------------------------------------------------------


def speakers_enqueue(db: Database, *, video: str, priority: int = 0) -> CommandResult:
    """Ask the worker to diarize this video when it next has the GPU."""
    from rytp.jobs.queue import enqueue, get_job

    video_id = resolve_video_id(db, video)
    job_id = enqueue(db, "diarize", video_id, priority=priority)
    job = get_job(db, job_id)
    return CommandResult(
        message=f"diarize job {job_id} for video {video_id} is {job.state}"
    )


# -- suggestions -----------------------------------------------------------


def speakers_suggest(
    db: Database, *, video: str, label: str = "", limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> CommandResult:
    """Who each unnamed voice might be. A suggestion, never an action.

    Nothing here writes. Design §6 is explicit that a decade of changing
    microphones makes automatic linking untrustworthy, so the verdict
    column is advice and the `link` command is the only way to act on it.
    """
    video_id = resolve_video_id(db, video)
    if label:
        row = store.find_label(db, video_id, label)
        grouped = {
            row.video_speaker_id: suggest_for_label(
                db, row.video_speaker_id, limit=limit
            )
        }
    else:
        grouped = suggest_for_video(db, video_id, limit=limit)

    by_id = {row.video_speaker_id: row for row in store.label_rows(db, video_id)}
    rows = tuple(
        (
            by_id[video_speaker_id].local_label,
            item.speaker_label,
            f"{item.similarity:.2f}",
            item.verdict,
            item.era,
            "same era" if item.same_era else "other era",
            "similar sound" if item.acoustically_close else "unlike or unmeasured",
        )
        for video_speaker_id, items in grouped.items()
        for item in items
    )
    return CommandResult(
        columns=("label", "person", "score", "verdict", "era", "when", "sound"),
        rows=rows,
        message=None if rows else _nothing_to_suggest(db),
    )


def _nothing_to_suggest(db: Database) -> str:
    """Say which of the two reasons an empty suggestion list has.

    An empty list because nobody embedded anything and an empty list because
    every candidate came from a different diarizer look identical, and the
    second one would otherwise be a mystery (design §6: nothing is applied
    silently, which also means nothing is *withheld* silently).
    """
    engines = engines_in_play(db)
    if len(engines) > 1:
        return (
            "nothing to suggest: the embedded voices come from more than one "
            f"diarizer ({', '.join(sorted(engines))}) and vectors from different "
            "diarizers are not comparable — re-diarize with one engine"
        )
    return (
        "nothing to suggest: embed this video's voices and at least one "
        "named video (`rytp speakers embed <video>`)"
    )


# -- engines and the mapper launcher --------------------------------------


def speakers_engines(db: Database) -> CommandResult:
    """Which diarizers and embedders this machine could actually run."""
    rows = [
        (name, "diarizer", token, out_of_process, state)
        for name, token, out_of_process, state in diarizer_rows(db)
    ]
    rows += [
        (name, "embedder", "no", out_of_process, state)
        for name, out_of_process, state in embedder_rows(db)
    ]
    return CommandResult(
        columns=("name", "kind", "token", "subprocess", "state"), rows=tuple(rows)
    )


def _run_mapper(db: Database, video_id: int) -> None:
    """Open the interactive mapper. A seam, so the test never starts Textual."""
    from rytp.tui.screens.speakers import run_mapper

    run_mapper(db, video_id)


def speakers_map(db: Database, *, video: str) -> CommandResult:
    """Open the two-pane mapper for one video (design §10)."""
    video_id = resolve_video_id(db, video)
    if not store.label_rows(db, video_id):
        raise NotFoundError(
            f"video {video_id} has no labels to map; diarize it first "
            f"(`rytp speakers diarize {video_id}`)"
        )
    _run_mapper(db, video_id)
    remaining = sum(1 for row in store.label_rows(db, video_id) if row.speaker_id is None)
    return CommandResult(message=f"{remaining} labels left unnamed")


# -- registration ----------------------------------------------------------

_VIDEO_ID = Param("video", str, "Video row id or external id.", positional=True)
_LABEL = Param("label", str, "Diarizer label, for example SPEAKER_00.", positional=True)

register(
    Command(
        name="speakers.add",
        group="speakers",
        summary="Add one person to the global roster.",
        params=(
            Param("label", str, "The person's name.", positional=True),
            Param("aliases", str, "Comma-separated alternative names.", default=""),
            Param("notes", str, "Free text.", default=None),
        ),
        handler=speakers_add,
    )
)
register(
    Command(
        name="speakers.list",
        group="speakers",
        summary="Show the global roster.",
        params=(),
        handler=speakers_list,
    )
)
register(
    Command(
        name="speakers.alias",
        group="speakers",
        summary="Add aliases to somebody already on the roster.",
        params=(
            Param("speaker", str, "Roster id, name or alias.", positional=True),
            Param("aliases", str, "Comma-separated names to add.", positional=True),
        ),
        handler=speakers_alias,
    )
)
register(
    Command(
        name="speakers.remove",
        group="speakers",
        summary="Delete a person from the roster. Their labels stay, unnamed.",
        params=(
            Param("speaker", str, "Roster id, name or alias.", positional=True),
            Param("yes", bool, "Confirm when many labels name them.", default=False),
        ),
        handler=speakers_remove,
    )
)
register(
    Command(
        name="speakers.labels",
        group="speakers",
        summary="Show one video's diarizer labels and who they are.",
        params=(_VIDEO_ID,),
        handler=speakers_labels,
    )
)
register(
    Command(
        name="speakers.link",
        group="speakers",
        summary="Say which person one of a video's labels is.",
        params=(
            _VIDEO_ID,
            _LABEL,
            Param("speaker", str, "Roster id, name or alias.", positional=True),
        ),
        handler=speakers_link,
    )
)
register(
    Command(
        name="speakers.unlink",
        group="speakers",
        summary="Detach one label of one video from its person. The roster is kept.",
        params=(_VIDEO_ID, _LABEL),
        handler=speakers_unlink,
    )
)
register(
    Command(
        name="speakers.diarize",
        group="speakers",
        summary="Work out who spoke when in one video. Opt-in; uses the GPU.",
        params=(
            _VIDEO_ID,
            Param("diarizer", str, "Engine name; default from settings.", default=""),
            Param("embedder", str, "Also embed the voices, with this engine.", default=""),
        ),
        handler=speakers_diarize,
        long_running=True,
    )
)
register(
    Command(
        name="speakers.embed",
        group="speakers",
        summary="Fingerprint one video's voices for cross-video suggestions.",
        params=(
            _VIDEO_ID,
            Param("embedder", str, "Engine name; default from settings.", default=""),
        ),
        handler=speakers_embed,
        long_running=True,
    )
)
register(
    Command(
        name="speakers.enqueue",
        group="speakers",
        summary="Queue one video for diarization by the worker.",
        params=(
            _VIDEO_ID,
            Param("priority", int, "Higher runs sooner.", default=0),
        ),
        handler=speakers_enqueue,
    )
)
register(
    Command(
        name="speakers.suggest",
        group="speakers",
        summary="Who each unnamed voice might be. Advice only; changes nothing.",
        params=(
            _VIDEO_ID,
            Param("label", str, "One label only; default every unnamed one.", default=""),
            Param("limit", int, "Candidates per label.", default=C.SPEAKER_SUGGEST_LIMIT),
        ),
        handler=speakers_suggest,
    )
)
register(
    Command(
        name="speakers.engines",
        group="speakers",
        summary="Which diarizers and embedders could run on this machine.",
        params=(),
        handler=speakers_engines,
    )
)
register(
    Command(
        name="speakers.map",
        group="speakers",
        summary="Open the two-pane mapper for one video.",
        params=(_VIDEO_ID,),
        handler=speakers_map,
        cli_only=True,
    )
)

# Importing this registers Part 7's `doctor` checks (contracts §5). It sits
# here rather than in `rytp/diarize/__init__.py` because the engine package
# is imported inside a foreign interpreter, which has no `rytp.commands`.
from rytp.diarize import health as _health  # noqa: E402,F401
