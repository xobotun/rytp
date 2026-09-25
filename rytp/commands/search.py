"""Index, search, playback and transcript commands (contracts §5).

Design §10: "Each operation is defined once — name, arguments, result
shape — and both the CLI and the TUI are generated from that
definition." Handlers here are thin: they translate scalars into calls on
`rytp.index` and shape the answer into a `CommandResult`. They never
print and never exit; an expected failure is a `RytpError` and the
surface decides what it looks like.
"""

from __future__ import annotations

from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.commands import (
    Command,
    CommandResult,
    HealthCheck,
    HealthResult,
    Param,
    register,
    register_check,
    resolve_speaker_filter,
    resolve_video_id,
)
from rytp.db import Database
from rytp.index import export
from rytp.index.export import export_clip, play_clip, timestamp, transcript_blocks, write_transcript
from rytp.index.search import MatchTier, anchor_filename, search, span_for_anchor
from rytp.index.utterances import drop_utterances, index_video, videos_needing_index
from rytp.jobs import queue
from rytp.models import NotFoundError, RytpError

#: A `tier` column rather than a yes/no `cut`: contracts §3 has three
#: transcript tiers and `aligned` already means cuttable, while `timed`
#: and `caption` are two different reasons a hit cannot be cut yet.
_SEARCH_COLUMNS = ("anchor", "video", "time", "speaker", "tier", "text")


def _shorten(text: str) -> str:
    """One line of a result table, not a paragraph."""
    if len(text) <= C.SEARCH_TEXT_TRUNCATE_CHARS:
        return text
    return text[: C.SEARCH_TEXT_TRUNCATE_CHARS - 1].rstrip() + "…"


def index_build(db: Database, *, video: str | None = None, enqueue: bool = False) -> CommandResult:
    """Rebuild utterances. One video, or every video that needs it.

    With no video, the targets come from `videos_needing_index`, which is
    derived from database state (design §5) — so this is also the repair
    command after a transcript is replaced or a video is diarized.
    """
    if video:
        targets = [resolve_video_id(db, video)]
    else:
        targets = videos_needing_index(db)
        if not targets:
            # BUGS.md entry 4: say *why* there is nothing, and what to do
            # next, rather than stopping at "nothing to index". The two
            # cases read very differently to the person running it.
            transcribed = int(
                db.conn.execute(
                    "SELECT COUNT(DISTINCT video_id) FROM words"
                ).fetchone()[0]
            )
            if not transcribed:
                return CommandResult(
                    message="nothing to index: no video has been transcribed yet; "
                    "run `rytp transcribe run <video>` first"
                )
            return CommandResult(
                message=f"nothing to index: all {transcribed} transcribed video(s) "
                "are already indexed"
            )
    if enqueue:
        for target in targets:
            queue.enqueue(db, "index", target)
        return CommandResult(message=f"queued {len(targets)} index job(s)")
    rows = tuple((str(target), str(index_video(db, target))) for target in targets)
    return CommandResult(
        columns=("video", "utterances"),
        rows=rows,
        message=f"indexed {len(rows)} video(s)",
    )


def search_words(
    db: Database,
    *,
    query: str,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    cuttable: bool = False,
    video: str | None = None,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
) -> CommandResult:
    """Where a word or phrase was said."""
    video_id = resolve_video_id(db, video) if video else None
    # Contracts §5: one shared resolver, two identifier spaces, one pinned
    # signature. It raises when a person cannot be found (naming near
    # matches) and when --video-local-speaker arrives without a video, so
    # this handler does no speaker logic at all. `None` back means no
    # filter was asked for; an empty id set means one was asked for and
    # matched nothing, which `search` turns into no rows rather than all.
    speaker_filter = resolve_speaker_filter(
        db, speaker=speaker, video_local_speaker=video_local_speaker,
        video_id=video_id,
    )
    speaker_ids = None if speaker_filter is None else speaker_filter.video_speaker_ids
    result = search(
        db,
        query,
        speaker_ids=speaker_ids,
        cuttable_only=cuttable,
        video_id=video_id or 0,
        limit=limit,
    )
    if not result.hits:
        return CommandResult(message=f"no hits for {query!r}")
    rows = tuple(
        (
            hit.anchor,
            hit.video_title,
            timestamp(hit.start_ms),
            hit.speaker or C.NULL_CELL,
            hit.source,
            _shorten(hit.text),
        )
        for hit in result.hits
    )
    # Which tier answered is part of the answer: a stem hit is an
    # inflection of what was typed, not what was typed (design §7).
    tier = (
        "exact match"
        if result.tier is MatchTier.EXACT
        else "stem match, so these are inflected forms"
    )
    crossed = sum(1 for hit in result.hits if hit.crosses_utterances)
    note = f"; {crossed} spanning an utterance boundary" if crossed else ""
    uncuttable = sum(1 for hit in result.hits if not hit.cuttable)
    if uncuttable:
        note += f"; {uncuttable} not cuttable yet (tier is not `aligned`)"
    if speaker_filter is not None:
        note += f"; {speaker_filter.description}"
    return CommandResult(
        columns=_SEARCH_COLUMNS,
        rows=rows,
        message=f"{len(rows)} hit(s), {tier}{note}",
    )


def search_play(db: Database, *, anchor: str, pad_ms: int = C.CLIP_PAD_MS) -> CommandResult:
    """Play one anchor, as printed by `search words`."""
    span = span_for_anchor(db, anchor)
    play_clip(db, span.video_id, span.start_ms, span.end_ms, pad_ms=pad_ms)
    return CommandResult(message=f"played {anchor}: {span.text}")


def search_export(
    db: Database,
    *,
    query: str,
    out_dir: Path,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    cuttable: bool = False,
    video: str | None = None,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
    pad_ms: int = C.CLIP_PAD_MS,
) -> CommandResult:
    """Write every hit of a query out as a WAV, one file per hit."""
    video_id = resolve_video_id(db, video) if video else None
    speaker_filter = resolve_speaker_filter(
        db, speaker=speaker, video_local_speaker=video_local_speaker,
        video_id=video_id,
    )
    result = search(
        db,
        query,
        speaker_ids=None if speaker_filter is None else speaker_filter.video_speaker_ids,
        cuttable_only=cuttable,
        video_id=video_id or 0,
        limit=limit,
    )
    if not result.hits:
        raise NotFoundError(f"no hits for {query!r}; nothing to export")
    config.ensure_dir(out_dir)
    rows = []
    for hit in result.hits:
        out = out_dir / anchor_filename(hit.anchor)
        export_clip(db, hit.video_id, hit.start_ms, hit.end_ms, out, pad_ms=pad_ms)
        rows.append((hit.anchor, str(out)))
    return CommandResult(
        columns=("anchor", "file"),
        rows=tuple(rows),
        message=f"wrote {len(rows)} clip(s) to {out_dir}",
    )


def index_drop(db: Database, *, video: str) -> CommandResult:
    """Delete one video's utterances (contracts §5, Deletion)."""
    video_id = resolve_video_id(db, video)
    dropped = drop_utterances(db, video_id)
    return CommandResult(
        message=(
            f"dropped {dropped} utterance(s) from video {video_id}; "
            f"rebuild with `rytp index build --video {video_id}`"
        )
    )


def transcript_build(db: Database, *, video: str) -> CommandResult:
    """Write `data/transcripts/{video_id}.md`.

    No `--line-length`: unlike `transcript show`, this writes a durable
    file, and a hard wrap would bake one terminal's width into a stored
    artefact (BUGS.md entry 15). `transcript show` has the knob because
    what it renders is thrown away the moment the command exits.
    """
    video_id = resolve_video_id(db, video)
    return CommandResult(message=f"wrote {write_transcript(db, video_id)}")


def transcript_show(
    db: Database, *, video: str, line_length: int = C.TRANSCRIPT_DEFAULT_LINE_LENGTH
) -> CommandResult:
    """The transcript as rows, without writing a file.

    `transcript.build` produces the durable markdown at
    `data/transcripts/{video_id}.md`; this is the same content as a table,
    so the TUI can show a transcript and the CLI can read one without
    leaving a file behind. Both render from these rows, which is what
    keeps the two surfaces saying the same thing (design §10).

    `line_length` is the maximum characters a row's text may hold, not a
    wrap width (BUGS.md entry 37 — it replaces an earlier `textwrap.fill`
    that folded a long block into a multi-line cell `DataTable.add_row`
    then clipped to its first line). An utterance longer than that is
    split at a word boundary into *more* rows instead, each with its own
    start, end and anchor — `transcript_blocks` does the splitting, since
    only there are the individual word rows, with their own `start_ms`,
    still available. It applies only here: `transcript.build` writes a
    regenerable markdown file, not a terminal-sized table, so it has no
    equivalent knob.
    """
    video_id = resolve_video_id(db, video)
    if line_length < 1:
        raise RytpError(f"line_length must be at least 1, got {line_length}")
    blocks = transcript_blocks(db, video_id, max_chars=line_length)
    if not blocks:
        raise RytpError(
            f"video {video_id} has no utterances; run `rytp index build "
            f"--video {video_id}` first"
        )
    return CommandResult(
        columns=("anchor", "start", "end", "speaker", "text"),
        rows=tuple(
            (
                block.anchor,
                timestamp(block.start_ms),
                timestamp(block.end_ms),
                block.speaker,
                block.text,
            )
            for block in blocks
        ),
        message=f"{len(blocks)} block(s) in video {video_id}",
    )


def _check_ffplay(db: Database) -> HealthResult:
    """Playback needs it; export does not.

    Returns `ok=False` when it is missing, because it is; `required=False`
    on the registration is what keeps that non-fatal (contracts §5).
    """
    del db
    found = export._ffplay_binary()
    if found:
        return HealthResult(ok=True, detail=f"ffplay at {found}")
    return HealthResult(
        ok=False,
        detail="ffplay not found on PATH; `rytp search play` will not work "
        "(`rytp search export` still will)",
        remedy="install ffmpeg — ffplay ships with it: "
        "https://ffmpeg.org/download.html",
    )


def _check_fts5(db: Database) -> HealthResult:
    """The one Part 4 cannot survive without.

    Every search goes through an FTS5 virtual table. A Python whose
    bundled SQLite was compiled without FTS5 does not error — it just
    never matches, so the corpus looks empty and the user concludes the
    words are not in it.
    """
    enabled = db.conn.execute(
        "SELECT 1 FROM pragma_compile_options WHERE compile_options = 'ENABLE_FTS5'"
    ).fetchone()
    if enabled:
        return HealthResult(ok=True, detail="sqlite3 has FTS5 compiled in")
    return HealthResult(
        ok=False,
        detail="this Python's sqlite3 was built without FTS5, so `rytp search` "
        "cannot match anything and `rytp index build` cannot create its index",
        remedy="use a Python built against a full SQLite — the python.org "
        "installers and pyenv both are — or `pip install pysqlite3-binary`. "
        "Confirm with: python -c \"import sqlite3; "
        "sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)')\"",
    )


register_check(
    HealthCheck(
        name="ffplay",
        summary="ffplay, for playing a search hit.",
        run=_check_ffplay,
        # Advisory: `search export` works without it, only `search play`
        # does not. The result still reports the truth (contracts §5).
        required=False,
    )
)

register_check(
    HealthCheck(
        name="fts5",
        summary="FTS5 in the running Python's SQLite — every search needs it.",
        run=_check_fts5,
        # required=True by default, and correctly so: without FTS5 this
        # whole part is inert, and `doctor` should say so with its exit
        # code rather than only in its output.
        required=True,
    )
)


register(
    Command(
        name="index.build",
        group="index",
        summary="Rebuild utterances and the search index.",
        params=(
            Param(
                "video",
                str,
                "Video to index, by id or external id. Omit to index every video "
                "whose utterances are missing or stale.",
                default=None,
            ),
            Param(
                "enqueue",
                bool,
                "Queue index jobs for the worker instead of doing the work now.",
                default=False,
            ),
        ),
        handler=index_build,
        long_running=True,
    )
)

register(
    Command(
        name="search.words",
        group="search",
        summary="Find every place a word or phrase was said.",
        params=(
            Param("query", str, "Word or phrase to look for.", positional=True),
            Param("speaker", str, "Only hits by this person: a roster label or alias.",
                  default=None, short="-s"),
            Param("video_local_speaker", str,
                  "Only hits by this raw diarizer label, e.g. SPEAKER_00."
                  " Requires --video.", default=None),
            Param("cuttable", bool, "Only hits that can be cut (aligned words).",
                  default=False),
            Param("video", str, "Only hits in this video, by id or external id. Omit "
                  "to search every video.", default=None),
            Param("limit", int, "Maximum hits.", default=C.SEARCH_DEFAULT_LIMIT, short="-n"),
        ),
        handler=search_words,
    )
)

register(
    Command(
        name="search.play",
        group="search",
        summary="Play one search hit, named by the anchor `search words` printed.",
        params=(
            Param("anchor", str, "Anchor of the span, e.g. v12:340-341.", positional=True),
            Param("pad_ms", int, "Milliseconds of padding on each side.",
                  default=C.CLIP_PAD_MS),
        ),
        handler=search_play,
        long_running=True,
    )
)

register(
    Command(
        name="search.export",
        group="search",
        summary="Write every hit of a query out as a WAV file.",
        params=(
            Param("query", str, "Word or phrase to look for.", positional=True),
            Param("out_dir", Path, "Directory to write the clips into.", positional=True),
            Param("speaker", str, "Only hits by this person: a roster label or alias.",
                  default=None, short="-s"),
            Param("video_local_speaker", str,
                  "Only hits by this raw diarizer label, e.g. SPEAKER_00."
                  " Requires --video.", default=None),
            Param("cuttable", bool, "Only hits that can be cut (aligned words).",
                  default=False),
            Param("video", str, "Only hits in this video, by id or external id. Omit "
                  "to search every video.", default=None),
            Param("limit", int, "Maximum hits.", default=C.SEARCH_DEFAULT_LIMIT, short="-n"),
            Param("pad_ms", int, "Milliseconds of padding on each side.",
                  default=C.CLIP_PAD_MS),
        ),
        handler=search_export,
        long_running=True,
    )
)

register(
    Command(
        name="index.drop",
        group="index",
        summary="Delete a video's utterances. Rebuilt by `index build`.",
        params=(
            Param("video", str, "Video whose utterances to drop.", positional=True),
        ),
        handler=index_drop,
    )
)

register(
    Command(
        name="transcript.build",
        group="transcript",
        summary="Write the regenerable markdown transcript for one video. No "
        "--line-length: splitting a long block into extra rows is a table "
        "concern — see `transcript show` for that.",
        params=(
            Param("video", str, "Video to write a transcript for.", positional=True),
        ),
        handler=transcript_build,
    )
)

register(
    Command(
        name="transcript.show",
        group="transcript",
        summary="Read a video's transcript as a table, without writing a file.",
        params=(
            Param("video", str, "Video to read.", positional=True),
            Param(
                "line_length",
                int,
                "Maximum characters a row's text may hold. A longer block is "
                "split into more rows at a word boundary, each with its own "
                "timestamps and anchor, rather than wrapped. Only here, not "
                "on `transcript build`: that command writes a durable "
                "markdown file, not a terminal-sized table.",
                default=C.TRANSCRIPT_DEFAULT_LINE_LENGTH,
            ),
        ),
        handler=transcript_show,
    )
)
