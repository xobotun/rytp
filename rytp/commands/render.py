"""The render commands. design §9, contracts §5.

Thin over :mod:`rytp.render.run`. The two seams that tests replace —
loading a cut list and resolving the binaries — are module-level
functions rather than inline calls, so the suite never needs Part 5 on
disk or ffmpeg on PATH.
"""

from __future__ import annotations

from rytp import config
from rytp import constants as C
from rytp.commands import (
    REQUIRED,
    Command,
    CommandResult,
    Param,
    register,
    resolve_speaker_filter,
    resolve_video_id,
)
from rytp.db import Database
from rytp.db.queries import clamp_limit
from rytp.models import InvalidInputError, RytpError
from rytp.render.ffmpeg import Tools
from rytp.render.pauses import (
    PauseStats,
    applied_gap_ms,
    gaps_for_speaker_ids,
    gaps_for_video,
    gaps_for_video_speaker,
    summarize_gaps,
)
from rytp.render.report import format_timecode
from rytp.render.run import (
    RenderOptions,
    RenderRequest,
    create_render,
    payload_for,
    remove_render,
    render_cutlist,
    request_from_cutlist,
)


def _tools() -> Tools:
    """Resolve ffmpeg and ffprobe. A seam so tests can fake both."""
    return Tools.resolve()


def _load_request(name: str) -> RenderRequest:
    """Read a cut list by name. The last of two Part 5 import sites."""
    path = config.paths().cutlist(name)
    if not path.exists():
        raise RytpError(
            f"no cut list named {name!r} at {path}; "
            f"make one with `rytp assemble` first"
        )
    from rytp.assemble.cutlist import load_cutlist

    return request_from_cutlist(load_cutlist(path))


def _options(
    *,
    canvas: str,
    height: int,
    fps: int,
    gap_ms: int,
    loudnorm: bool,
    preset: str,
    crf: int,
    keep_intermediates: bool,
    render_name: str,
) -> RenderOptions:
    return RenderOptions(
        canvas_mode=canvas,
        height=height,
        fps=fps,
        gap_ms=gap_ms,
        loudnorm=loudnorm,
        preset=preset,
        crf=crf,
        keep_intermediates=keep_intermediates,
        render_id=render_name,
    )


def _run_handler(
    db: Database,
    *,
    name: str,
    canvas: str = C.RENDER_DEFAULT_CANVAS_MODE,
    height: int = 0,
    fps: int = 0,
    gap_ms: int = -1,
    loudnorm: bool = True,
    preset: str = C.RENDER_VIDEO_PRESET,
    crf: int = C.RENDER_VIDEO_CRF,
    keep_intermediates: bool = False,
    render_name: str = "",
    dry_run: bool = False,
    enqueue: bool = False,
) -> CommandResult:
    options = _options(
        canvas=canvas, height=height, fps=fps, gap_ms=gap_ms, loudnorm=loudnorm,
        preset=preset, crf=crf, keep_intermediates=keep_intermediates,
        render_name=render_name,
    )
    request = _load_request(name)
    if enqueue:
        from rytp.jobs.queue import enqueue as enqueue_job

        queued_render_id = create_render(db, cutlist_name=name, options=options)
        job_id = enqueue_job(db, "render", queued_render_id, payload=payload_for(options))
        return CommandResult(
            columns=("job", "render", "cut list"),
            rows=((str(job_id), str(queued_render_id), name),),
            message=f"queued as job {job_id}; run `rytp worker` to drain it",
        )
    # A dry run decides nothing durable, so it opens no row.
    render_row_id = None if dry_run else create_render(
        db, cutlist_name=name, options=options
    )
    result = render_cutlist(
        db, request, options, tools=_tools(), dry_run=dry_run,
        render_row_id=render_row_id,
    )
    return CommandResult(
        columns=("render", "output", "report", "fragments", "sources", "duration"),
        rows=(
            (
                result.render_id,
                str(result.output_path),
                str(result.report_path),
                str(result.fragment_count),
                str(result.source_count),
                format_timecode(result.duration_ms),
            ),
        ),
        message="dry run: nothing was written" if dry_run else None,
    )


def _pause_row(stats: PauseStats) -> tuple[str, ...]:
    return (
        stats.scope,
        stats.key,
        str(stats.n_samples),
        str(stats.median_ms),
        f"{stats.zero_fraction:.0%}",
        str(applied_gap_ms(stats)),
        stats.reason or "usable",
    )


def _pauses_handler(
    db: Database, *, video: str | None = None, speaker: str | None = None
) -> CommandResult:
    if not video and not speaker:
        raise InvalidInputError("pass --video, --speaker, or both")
    video_row_id = resolve_video_id(db, video) if video else None
    rows: list[tuple[str, ...]] = []
    if speaker:
        # Contracts §5: one shared resolver, never a local one. It matches
        # a roster label or alias, never a raw diarizer label, expands to
        # video_speakers ids, and raises rather than quietly returning
        # nothing when the name matches nobody.
        speaker_filter = resolve_speaker_filter(db, speaker=speaker, video_id=video_row_id)
        assert speaker_filter is not None  # a non-empty --speaker always resolves
        rows.append(
            _pause_row(
                summarize_gaps(
                    gaps_for_speaker_ids(db, speaker_filter.video_speaker_ids),
                    scope="speaker",
                    key=speaker_filter.description,
                )
            )
        )
    video_id = video_row_id
    if video_id:
        labels = db.conn.execute(
            "SELECT vs.id, vs.local_label, s.label AS roster "
            "FROM video_speakers vs LEFT JOIN speakers s ON s.id = vs.speaker_id "
            "WHERE vs.video_id = ? ORDER BY vs.id",
            (video_id,),
        ).fetchall()
        for label in labels:
            key = label["roster"] or label["local_label"]
            rows.append(
                _pause_row(
                    summarize_gaps(
                        gaps_for_video_speaker(db, int(label["id"])),
                        scope="video_speaker",
                        key=str(key),
                    )
                )
            )
        rows.append(
            _pause_row(
                summarize_gaps(
                    gaps_for_video(db, video_id), scope="video", key=str(video_id)
                )
            )
        )
    return CommandResult(
        columns=(
            "scope", "key", "samples", "median ms", "zeros", "would insert", "verdict"
        ),
        rows=tuple(rows),
    )


register(
    Command(
        name="render.run",
        group="render",
        summary="Cut, normalise and join a cut list into one uploadable file.",
        params=(
            Param(
                name="name",
                type=str,
                help="Cut list name, as in cutlists/{name}.toml.",
                default=REQUIRED,
                positional=True,
            ),
            Param(
                name="canvas",
                type=str,
                help="Output shape: '16:9' keeps a fixed horizontal frame and "
                "boxes anything vertical or 4:3 into it; 'bbox' is the biggest "
                "box every source fits at its native size, so vertical or "
                "wide-aspect sources keep their own shape instead.",
                default=C.RENDER_DEFAULT_CANVAS_MODE,
                choices=C.RENDER_CANVAS_MODES,
            ),
            Param(
                name="height",
                type=int,
                help="Canvas height in pixels; 0 takes the tallest source.",
                default=0,
            ),
            Param(
                name="fps",
                type=int,
                help="Output frame rate; 0 takes the commonest source rate.",
                default=0,
            ),
            Param(
                name="gap_ms",
                type=int,
                help="Freeze-frame gap between fragments: -1 measures it from "
                "the aligned transcripts, 0 turns gaps off, >0 sets it.",
                default=-1,
            ),
            Param(
                name="loudnorm",
                type=bool,
                help="Match each source's loudness and normalise the result.",
                default=True,
            ),
            Param(
                name="preset",
                type=str,
                help="x264 preset.",
                default=C.RENDER_VIDEO_PRESET,
            ),
            Param(
                name="crf", type=int, help="x264 CRF.", default=C.RENDER_VIDEO_CRF
            ),
            Param(
                name="keep_intermediates",
                type=bool,
                help="Keep the per-fragment files next to the output.",
                default=False,
            ),
            Param(
                name="render_name",
                type=str,
                help="Output directory name; derived from the cut list by default.",
                default="",
            ),
            Param(
                name="dry_run",
                type=bool,
                help="Plan the render and print it without encoding anything.",
                default=False,
            ),
            Param(
                name="enqueue",
                type=bool,
                help="Queue the render for the worker instead of running it now.",
                default=False,
            ),
        ),
        handler=_run_handler,
        long_running=True,
    )
)


def _list_handler(db: Database, *, limit: int = C.DEFAULT_LIST_LIMIT) -> CommandResult:
    rows = db.conn.execute(
        "SELECT id, cutlist_name, state, canvas_mode, output_path, created_at "
        "FROM renders ORDER BY id DESC LIMIT ?",
        (clamp_limit(limit),),
    ).fetchall()
    return CommandResult(
        columns=("render", "cut list", "state", "canvas", "output", "created"),
        rows=tuple(
            (
                str(row["id"]),
                str(row["cutlist_name"]),
                str(row["state"]),
                str(row["canvas_mode"]),
                str(row["output_path"] or C.NULL_CELL),
                str(row["created_at"]),
            )
            for row in rows
        ),
    )


register(
    Command(
        name="render.pauses",
        group="render",
        summary="Show the measured between-word pauses a render would insert.",
        params=(
            Param(
                name="video",
                type=str,
                help="Show every diarized label of this video, plus the video itself.",
                default=None,
            ),
            Param(
                name="speaker",
                type=str,
                help="Show one roster speaker or alias, pooled across every video.",
                default=None,
            ),
        ),
        handler=_pauses_handler,
    )
)


def _remove_handler(
    db: Database, *, render_id: int, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    if not dry_run and not yes:
        raise InvalidInputError(
            f"removing render {render_id} deletes its output directory; "
            "pass --yes to confirm, or --dry-run to see what would go"
        )
    removal = remove_render(db, render_id, dry_run=dry_run)
    detail = [
        f"cut list {removal.cutlist_name!r}",
        f"{len(removal.files)} file(s), {removal.total_bytes:,} bytes",
    ]
    if not removal.cutlist_present:
        detail.append("whose cut list is already gone")
    if removal.jobs_cancelled:
        detail.append(
            f"{removal.jobs_cancelled} queued job(s) "
            f"{'cancelled' if removal.removed else 'would be cancelled'}"
        )
    return CommandResult(
        columns=("file", "bytes"),
        rows=tuple((path, f"{size:,}") for path, size in removal.files),
        message=(
            f"{'removed' if removal.removed else 'would remove'} "
            f"render {render_id} — " + "; ".join(detail)
        ),
    )


register(
    Command(
        name="render.list",
        group="render",
        summary="List past renders: what was made, from which cut list, and where.",
        params=(
            Param(
                name="limit",
                type=int,
                help="How many to show, newest first.",
                default=C.DEFAULT_LIST_LIMIT,
            ),
        ),
        handler=_list_handler,
    )
)

register(
    Command(
        name="render.remove",
        group="render",
        summary="Delete a render's output directory and its row.",
        params=(
            Param(
                name="render_id",
                type=int,
                help="Render id, as `render list` shows it.",
                default=REQUIRED,
                positional=True,
            ),
            Param(
                name="dry_run",
                type=bool,
                help="List what would be deleted, with byte counts, and stop.",
                default=False,
            ),
            Param(
                name="yes",
                type=bool,
                help="Confirm. Required, because this deletes files.",
                default=False,
            ),
        ),
        handler=_remove_handler,
    )
)
