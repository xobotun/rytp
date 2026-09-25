"""Assembly commands: plan a cut list, read one back, suggest stand-ins.

Design §8 is reached from here and from the TUI, through the one registry
entry each (contracts §5). None of these is ``long_running``: there is no
``assemble`` job kind in contracts §5 and design §5 never queues
assembly, so a long-running mark would leave the TUI with a command it
refuses to run and no worker to run it.
"""

from __future__ import annotations

from rytp import constants as C
from rytp import timefmt
from rytp.assemble import (
    SLOT_FRAGMENT,
    AssembleControls,
    CutList,
    MatchFilters,
    assemble_target,
    cutlist_path,
    newest_cutlist_paths,
    read_cutlist,
    retarget_cutlist,
    suggest_substitutions,
    write_cutlist,
)
from rytp.commands import Command, CommandResult, Param, register, resolve_speaker_filter
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, RytpError

__all__ = [
    "assemble_list",
    "assemble_plan",
    "assemble_remove",
    "assemble_retarget",
    "assemble_show",
    "assemble_suggest",
    "format_ms",
    "orphaned_renders",
    "parse_ids",
]

_SLOT_COLUMNS = ("#", "kind", "target", "source", "in", "out", "text")


def parse_ids(text: str) -> tuple[int, ...]:
    """A comma-separated list of video ids (contracts §5 multi-value flags)."""
    ids: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            raise InvalidInputError(
                f"{piece!r} is not a video id; give a comma-separated list like '3,7'"
            ) from None
        if value not in ids:
            ids.append(value)
    return tuple(ids)


def format_ms(value: int) -> str:
    """Milliseconds as a timestamp a person can find in a player.

    A seek target — the same purpose ``render run``'s report prints for a
    fragment's source and output offsets — so this delegates to
    :func:`rytp.timefmt.format_seek` and always carries the hour field.
    Before BUGS.md entry 31 this printed `M:SS.mmm` (no hour below one
    hour) while the render report printed `H:MM:SS.mmm`; the two are now
    the same shape for the same measurement.
    """
    return timefmt.format_seek(value)


def _slot_rows(cutlist: CutList) -> tuple[tuple[str, ...], ...]:
    """The cut list as a table, gaps included and in timeline order."""
    rows: list[tuple[str, ...]] = []
    for index, slot in enumerate(cutlist.slots, start=1):
        span = (
            f"{slot.target_first}"
            if slot.target_first == slot.target_last
            else f"{slot.target_first}-{slot.target_last}"
        )
        if slot.kind == SLOT_FRAGMENT and slot.start_ms is not None and slot.end_ms is not None:
            source = str(slot.video_id)
            start, end = format_ms(slot.start_ms), format_ms(slot.end_ms)
        else:
            source, start, end = C.NULL_CELL, C.NULL_CELL, C.NULL_CELL
        rows.append((str(index), slot.kind, span, source, start, end, slot.text))
    return tuple(rows)


def _summary(cutlist: CutList) -> str:
    """One line: how many pieces, from how many sources, how long, what is missing.

    BUGS.md entry 26's acceptance criterion: an empty or partial result
    must never read as silence. ``cutlist.gap_notes`` (plan-time only,
    never persisted) names why each missing word matched nothing, so
    "not found" is never the whole story.
    """
    sources = {slot.video_id for slot in cutlist.fragments if slot.video_id is not None}
    timed = sum(1 for slot in cutlist.fragments if slot.tier == "timed")
    parts = [
        f"{len(cutlist.fragments)} fragment{'' if len(cutlist.fragments) == 1 else 's'}",
        f"{len(sources)} source{'' if len(sources) == 1 else 's'}",
        format_ms(cutlist.duration_ms),
    ]
    if timed:
        parts.append(f"{timed} from the timed tier (--allow-timed)")
    if cutlist.gaps:
        missing = ", ".join(slot.text for slot in cutlist.gaps)
        parts.append(
            f"{len(cutlist.gaps)} word{'' if len(cutlist.gaps) == 1 else 's'} "
            f"not found: {missing}"
        )
    if cutlist.gap_notes:
        parts.append("; ".join(cutlist.gap_notes))
    return "; ".join(parts)


def assemble_plan(
    db: Database,
    *,
    target: str,
    name: str = "",
    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY,
    seed: int = 0,
    pad: int = C.ASSEMBLE_DEFAULT_PAD_MS,
    speaker: str | None = None,
    exclude: str = "",
    min_align: float = C.ASSEMBLE_MIN_ALIGN_SCORE,
    allow_timed: bool = False,
    force: bool = False,
) -> CommandResult:
    """Work out which real fragments say ``target``, and write the cut list."""
    controls = AssembleControls(
        consistency=consistency,
        seed=seed,
        pad_ms=pad,
        speaker=speaker,
        exclude=parse_ids(exclude),
        min_align_score=min_align,
        allow_timed=allow_timed,
    )
    cutlist = assemble_target(db, target, name=name or None, controls=controls)
    path = cutlist_path(cutlist.name)
    if path.exists() and not force:
        raise RytpError(f"{path} already exists; pass --force to replace it")
    write_cutlist(cutlist, path)
    return CommandResult(
        columns=_SLOT_COLUMNS,
        rows=_slot_rows(cutlist),
        message=f"wrote {path} — {_summary(cutlist)}",
    )


def assemble_list(db: Database) -> CommandResult:
    """Every cut list's bare name — copy-pasteable into `assemble show`.

    BUGS.md entry 27: the only way to enumerate cut lists before this was
    to look in ``data/cutlists/`` by hand. Output is bare names, never
    paths and never ``<name>.toml``, exactly what `assemble show` and
    `render run` already accept. Most recently modified first, so a fresh
    plan is the first thing the owner sees.
    """
    names = [path.stem for path in newest_cutlist_paths()]
    return CommandResult(
        columns=("name",),
        rows=tuple((name,) for name in names),
        message=f"{len(names)} cut list{'' if len(names) == 1 else 's'}",
    )


def assemble_show(db: Database, *, name: str = "") -> CommandResult:
    """Read a cut list back, validate it, and say what is in it."""
    if not name:
        # Typer would otherwise refuse this before the handler ever runs
        # (`Missing argument 'name'`), which cannot point anywhere useful
        # (BUGS.md entry 27). Giving the param a default and raising here
        # keeps the pointer inside this command's own files.
        raise InvalidInputError("no cut list named; try `assemble list`")
    cutlist = read_cutlist(name)
    wanted = sorted({slot.video_id for slot in cutlist.fragments if slot.video_id is not None})
    known: set[int] = set()
    if wanted:
        inline = ", ".join(str(video_id) for video_id in wanted)
        known = {
            int(row["id"])
            for row in db.conn.execute(f"SELECT id FROM videos WHERE id IN ({inline})")
        }
    message = _summary(cutlist)
    missing = [video_id for video_id in wanted if video_id not in known]
    if missing:
        # Showing a broken cut list is the point of this command — it is
        # the file you are about to repair. Rendering one is not: part 6
        # aborts, because a silently shortened video is worse than a
        # refusal. Say so here rather than letting render be the surprise.
        listed = ", ".join(str(video_id) for video_id in missing)
        message += f"; NOT RENDERABLE — not in the catalog: {listed}"
    return CommandResult(columns=_SLOT_COLUMNS, rows=_slot_rows(cutlist), message=message)


def assemble_retarget(db: Database, *, name: str, target: str) -> CommandResult:
    """Edit a cut list's target sentence, replanning only what changed.

    The owner's ask: adding a word, or swapping one that has no cuttable
    form for one that does, without re-running the whole plan and losing
    every nudge, snap, swap and substitution already done by hand. See
    `rytp.assemble.retarget_cutlist` for the algorithm and the cohesion
    trade-off it accepts in exchange for keeping the rest of the file
    untouched.
    """
    cutlist = read_cutlist(name)
    result = retarget_cutlist(db, cutlist, target)
    path = cutlist_path(name)
    write_cutlist(result.cutlist, path)
    message = (
        f"wrote {path} — kept {result.kept} slot{'' if result.kept == 1 else 's'}, "
        f"replanned {result.replanned}"
    )
    if result.gap_notes:
        message += "; " + "; ".join(result.gap_notes)
    return CommandResult(
        columns=_SLOT_COLUMNS,
        rows=_slot_rows(result.cutlist),
        message=message,
    )


def assemble_suggest(
    db: Database,
    *,
    word: str,
    limit: int = C.ASSEMBLE_SUBSTITUTION_LIMIT,
    speaker: str | None = None,
    exclude: str = "",
) -> CommandResult:
    """Ranked stand-ins for a word: same stem first, then closest spelling."""
    speaker_filter = resolve_speaker_filter(db, speaker=speaker) if speaker else None
    filters = MatchFilters(
        exclude_video_ids=frozenset(parse_ids(exclude)),
        video_speaker_ids=(
            None if speaker_filter is None else speaker_filter.video_speaker_ids
        ),
    )
    hits = suggest_substitutions(db, word.strip().lower(), filters, limit=limit)
    return CommandResult(
        columns=("word", "why", "distance", "occurrences", "source", "in", "out"),
        rows=tuple(
            (
                hit.text,
                hit.reason,
                str(hit.distance),
                str(hit.occurrences),
                str(hit.run.video_id),
                format_ms(hit.run.start_ms),
                format_ms(hit.run.end_ms),
            )
            for hit in hits
        ),
        message=(
            f"{len(hits)} suggestion{'' if len(hits) == 1 else 's'} for {word!r}"
            if hits
            else f"nothing close to {word!r} is cuttable in the corpus"
        ),
    )


register(
    Command(
        name="assemble.plan",
        group="assemble",
        summary="Find real fragments that say a sentence and write a cut list.",
        params=(
            Param("target", str, "The sentence to assemble.", positional=True),
            Param("name", str, "Cut list name. Defaults to a slug of the target.", default=""),
            Param(
                "consistency",
                float,
                "0.0 fewest seams .. 1.0 most consistent sound.",
                default=C.ASSEMBLE_DEFAULT_CONSISTENCY,
            ),
            Param("seed", int, "Shake up choices among near-equal candidates.", default=0),
            Param(
                "pad",
                int,
                "Milliseconds of tail added after each fragment.",
                default=C.ASSEMBLE_DEFAULT_PAD_MS,
            ),
            Param(
                "speaker",
                str,
                "Only cut words said by this roster person (label or alias).",
                default=None,
            ),
            Param("exclude", str, "Video ids never to cut from, comma-separated.", default=""),
            Param(
                "min_align",
                float,
                "Skip energy-scale aligned words below this score (0.0-1.0).",
                default=C.ASSEMBLE_MIN_ALIGN_SCORE,
            ),
            Param(
                "allow_timed",
                bool,
                "Also cut timed-tier words: no threshold, an override not a gate.",
                default=False,
            ),
            Param("force", bool, "Replace an existing cut list of this name.", default=False),
        ),
        handler=assemble_plan,
    )
)

register(
    Command(
        name="assemble.show",
        group="assemble",
        summary="Read a cut list back and show its fragments and gaps.",
        params=(Param("name", str, "Cut list name.", positional=True, default=""),),
        handler=assemble_show,
    )
)

register(
    Command(
        name="assemble.retarget",
        group="assemble",
        summary=(
            "Edit a cut list's target sentence, keeping hand-tuned slots for "
            "the words that didn't change."
        ),
        params=(
            Param("name", str, "Cut list name.", positional=True),
            Param("target", str, "The new target sentence.", positional=True),
        ),
        handler=assemble_retarget,
    )
)

register(
    Command(
        name="assemble.list",
        group="assemble",
        summary="List every cut list by name.",
        params=(),
        handler=assemble_list,
    )
)

register(
    Command(
        name="assemble.suggest",
        group="assemble",
        summary="Ranked stand-ins for a word the corpus does not say.",
        params=(
            Param("word", str, "The missing word.", positional=True),
            Param(
                "limit",
                int,
                "How many suggestions.",
                default=C.ASSEMBLE_SUBSTITUTION_LIMIT,
                short="-n",
            ),
            Param(
                "speaker",
                str,
                "Only consider words said by this roster person (label or alias).",
                default=None,
            ),
            Param("exclude", str, "Video ids to ignore, comma-separated.", default=""),
        ),
        handler=assemble_suggest,
    )
)


def orphaned_renders(db: Database, name: str) -> tuple[tuple[str, ...], ...]:
    """Renders that named this cut list, as table rows.

    Part 6 owns the ``renders`` table and ``render.remove``; this only
    looks, so that removing a cut list does not orphan them in silence
    (contracts §5, "Deletion").
    """
    return tuple(
        (str(row["id"]), str(row["state"]), str(row["output_path"] or C.NULL_CELL))
        for row in db.conn.execute(
            "SELECT id, state, output_path FROM renders WHERE cutlist_name = ? "
            "ORDER BY id DESC",
            (name,),
        )
    )


def assemble_remove(
    db: Database, *, name: str, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    """Delete a cut list file. Nothing else — the renders are part 6's.

    Contracts §5: anything that deletes a file supports ``--dry-run`` and
    requires ``--yes``, and removal is synchronous rather than a job. The
    caution is earned here: design §8 made hand-edited timings the
    intended workflow, so a cut list can hold work that exists nowhere
    else and cannot be regenerated.
    """
    path = cutlist_path(name)  # validates the name; rejects path traversal
    if not path.exists():
        raise NotFoundError(f"no cut list named {name!r} at {path}")
    size = path.stat().st_size
    renders = orphaned_renders(db, path.stem)
    note = ""
    if renders:
        listed = ", ".join(row[0] for row in renders)
        # Warn, never refuse: the rendered file stands on its own, and
        # only its provenance is lost. Part 6 aborts on a dangling
        # video_id because there the *input* is corrupt; this is not that.
        note = (
            f"; {len(renders)} render(s) still reference it by name and keep their "
            f"output ({listed}); a queued render of it parks as blocked — use "
            f"`rytp render remove <id>` to delete an output too"
        )

    if dry_run:
        return CommandResult(
            columns=("render", "state", "output"),
            rows=renders,
            message=f"would remove {path} ({size} bytes){note}",
        )
    if not yes:
        raise RytpError(
            f"{path} ({size} bytes) would be deleted and a cut list can hold "
            f"hand-edited timings that exist nowhere else; pass --yes to confirm, "
            f"or --dry-run to see what would go{note}"
        )
    path.unlink()
    return CommandResult(
        columns=("render", "state", "output"),
        rows=renders,
        message=f"removed {path} ({size} bytes){note}",
    )


register(
    Command(
        name="assemble.remove",
        group="assemble",
        summary="Delete a cut list file. Renders that used it are reported, not touched.",
        params=(
            Param("name", str, "Cut list name.", positional=True),
            Param("dry_run", bool, "Show what would be deleted and stop.", default=False),
            Param("yes", bool, "Confirm the deletion. Required.", default=False),
        ),
        handler=assemble_remove,
    )
)
