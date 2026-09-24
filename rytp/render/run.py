"""Planning and running a render. design §9.

Two phases, deliberately separated. :func:`plan_render` reads the cut
list, the catalog and the sources and decides everything — which files,
what shape, how loud, how long each seam is, where each fragment lands
on the output timeline — without encoding a frame. :func:`render_cutlist`
then executes that plan. The split is what makes ``--dry-run`` free and
what lets the whole of this module be tested with a fake runner.

The cut list arrives as Part 5's ``CutList``, but nothing here imports
``rytp.assemble``: :func:`request_from_cutlist` reads attributes off
whatever it is handed. One adapter, one place to change.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import assets_for
from rytp.models import NotFoundError, utc_now_iso
from rytp.render.canvas import (
    Canvas,
    SourceGeometry,
    plan_canvas,
    probe_geometry,
    video_filter_chain,
)
from rytp.render.ffmpeg import (
    FfmpegFailedError,
    RenderError,
    Tools,
    concat_command,
    fragment_command,
    gain_for,
    parse_loudnorm_json,
    programme_loudness_command,
    run_command,
    source_loudness_command,
    write_concat_list,
)
from rytp.render.pauses import PauseStats, applied_gap_ms, measure_pause_stats
from rytp.render.report import (
    FragmentReport,
    MissingWord,
    RenderReport,
    SourceReport,
    SubstitutionRef,
    render_markdown,
)

#: Everything that is not a letter, digit or underscore becomes a dash.
#: ``\w`` is Unicode-aware, so a Cyrillic cut-list name survives intact,
#: while "..", "/" and a backslash cannot reach a path.
_UNSAFE = re.compile(r"[^\w-]+", re.UNICODE)


class MissingMediaError(RenderError):
    """A fragment's source file is not on disk. Names the fetch command."""


@dataclass(frozen=True)
class RenderOptions:
    """Every knob, with the defaults design §9 asks for."""

    canvas_mode: str = C.RENDER_DEFAULT_CANVAS_MODE
    height: int = 0  # 0 = the tallest source, capped
    fps: int = 0  # 0 = the commonest source rate
    gap_ms: int = -1  # -1 = measure, 0 = no gaps, >0 = this many
    loudnorm: bool = True
    preset: str = C.RENDER_VIDEO_PRESET
    crf: int = C.RENDER_VIDEO_CRF
    keep_intermediates: bool = False
    render_id: str = ""

    @property
    def gap_policy(self) -> str:
        """One phrase for the report."""
        if self.gap_ms == 0:
            return "off"
        if self.gap_ms > 0:
            return f"fixed at {self.gap_ms} ms"
        return "measured from aligned transcripts"


@dataclass(frozen=True)
class RenderFragment:
    """One cut, as the render needs it. Part 5's fragment slot, flattened."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    video_speaker_id: int | None = None
    speaker_label: str | None = None
    gap_before_ms: int | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class RenderRequest:
    """What to render: the fragments, plus what the report needs to say."""

    name: str
    fragments: tuple[RenderFragment, ...]
    target_text: str = ""
    missing: tuple[MissingWord, ...] = ()

    @property
    def assembled_text(self) -> str:
        return " ".join(f.text.strip() for f in self.fragments if f.text.strip())


def request_from_cutlist(cutlist: object) -> RenderRequest:
    """Adapt Part 5's ``CutList`` — by attribute, never by import.

    Part 5's format is one ordered ``slots`` sequence; ``kind`` is
    ``"fragment"`` or ``"gap"``, and a gap *is* a missing word carrying
    its ranked substitutions. Document order is the output timeline, so
    the order here is preserved exactly.
    """
    fragments: list[RenderFragment] = []
    missing: list[MissingWord] = []
    for slot in getattr(cutlist, "slots", ()):
        if getattr(slot, "kind", "") == "fragment":
            fragments.append(
                RenderFragment(
                    video_id=int(slot.video_id),
                    first_word_ord=int(slot.first_word_ord),
                    last_word_ord=int(slot.last_word_ord),
                    start_ms=int(slot.start_ms),
                    end_ms=int(slot.end_ms),
                    text=str(slot.text),
                    video_speaker_id=getattr(slot, "video_speaker_id", None),
                    speaker_label=getattr(slot, "speaker_label", None),
                    gap_before_ms=getattr(slot, "gap_before_ms", None),
                )
            )
            continue
        missing.append(
            MissingWord(
                text=str(slot.text),
                position=int(getattr(slot, "target_first", 0)),
                substitutions=tuple(
                    SubstitutionRef(
                        text=str(sub.text),
                        reason=str(getattr(sub, "reason", "")),
                        video_id=int(getattr(sub, "video_id", 0)),
                        start_ms=int(getattr(sub, "start_ms", 0)),
                        end_ms=int(getattr(sub, "end_ms", 0)),
                        occurrences=int(getattr(sub, "occurrences", 0)),
                    )
                    for sub in getattr(slot, "substitutions", ())
                ),
            )
        )
    return RenderRequest(
        name=str(getattr(cutlist, "name", "")),
        fragments=tuple(fragments),
        target_text=str(getattr(cutlist, "target", "")),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class SourceMedia:
    """One source video, resolved to files and measured."""

    video_id: int
    title: str
    url: str | None
    video_path: Path
    audio_path: Path
    geometry: SourceGeometry
    measured_lufs: float | None = None
    gain_db: float | None = None


def _video_row(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT id, title, url FROM videos WHERE id = ?", (video_id,)
    ).fetchone()


def _newest_existing(db: Database, video_id: int, role: str) -> Path | None:
    """The newest asset of a role whose file is actually there.

    Newest wins because design §4 says upgrading a rendition is just an
    insert; existence is checked because an asset row whose file was
    deleted outside the tool must not look usable.
    """
    for row in reversed(assets_for(db, video_id, role)):
        path = Path(row["path"])
        if path.exists():
            return path
    return None


def resolve_sources(
    db: Database, fragments: Sequence[RenderFragment], *, tools: Tools
) -> tuple[SourceMedia, ...]:
    """Resolve every distinct source, or refuse with all the reasons.

    Video comes from a rendition asset and audio from the audio asset
    (design §4). A local file registered as a single ``container`` asset
    stands in for either. Every problem is collected before raising, so
    one render tells the owner about all of them instead of one per run.
    """
    ordered: list[int] = []
    for fragment in fragments:
        if fragment.video_id not in ordered:
            ordered.append(fragment.video_id)
    problems: list[str] = []
    sources: list[SourceMedia] = []
    for video_id in ordered:
        row = _video_row(db, video_id)
        if row is None:
            problems.append(f"video {video_id} is not in the catalog")
            continue
        container = _newest_existing(db, video_id, "container")
        video_path = _newest_existing(db, video_id, "video") or container
        audio_path = _newest_existing(db, video_id, "audio") or container
        if video_path is None:
            problems.append(
                f"video {video_id} has no video rendition on disk — "
                f"run: rytp fetch-video {video_id}"
            )
            continue
        if audio_path is None:
            problems.append(
                f"video {video_id} has no audio asset on disk — "
                f"run: rytp fetch-video {video_id}"
            )
            continue
        sources.append(
            SourceMedia(
                video_id=video_id,
                title=str(row["title"]),
                url=row["url"],
                video_path=video_path,
                audio_path=audio_path,
                geometry=probe_geometry(
                    video_path, runner=tools.runner, binary=tools.ffprobe
                ),
            )
        )
    if problems:
        raise MissingMediaError("cannot render: " + "; ".join(problems))
    return tuple(sources)


def _stored_loudness(db: Database, video_id: int) -> float | None:
    row = db.conn.execute(
        "SELECT loudness_lufs FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None or row["loudness_lufs"] is None:
        return None
    return float(row["loudness_lufs"])


def measure_source_loudness(
    db: Database, source: SourceMedia, *, around_ms: int, tools: Tools
) -> float | None:
    """This source's integrated loudness, in LUFS, or ``None``.

    Part 3's fingerprint already measures the whole video, so that value
    is used when it is there. Otherwise one bounded window is scanned —
    two minutes around the first fragment this render takes from the
    source. Scanning a whole hour to place a single gain is not worth
    the minutes, and a two-minute window of continuous speech is plenty
    for an R128 integrated measurement.
    """
    stored = _stored_loudness(db, source.video_id)
    if stored is not None:
        return stored
    window_start = max(0, around_ms - C.RENDER_LOUDNESS_WINDOW_MS // 2)
    result = run_command(
        source_loudness_command(
            path=source.audio_path,
            start_ms=window_start,
            duration_ms=C.RENDER_LOUDNESS_WINDOW_MS,
            binary=tools.ffmpeg,
        ),
        runner=tools.runner,
        what=f"loudness scan of video {source.video_id}",
    )
    measured = parse_loudnorm_json(result.stderr)
    return None if measured is None else measured.input_i


@dataclass(frozen=True)
class SeamGap:
    """How long the freeze-frame between two fragments lasts, and why."""

    gap_ms: int
    origin: str  # "off" | "override" | "fixed" | "measured" | "fallback" | "end"
    note: str = ""


def plan_gap(
    db: Database,
    *,
    previous: RenderFragment,
    following: RenderFragment,
    options: RenderOptions,
    cache: dict[tuple[str, str], PauseStats],
) -> SeamGap:
    """Decide one seam.

    Precedence, most specific last except for the off switch:

    1. ``gap_ms == 0`` — off, and it wins over everything. A switch that
       something else can override is not a switch.
    2. ``gap_before_ms`` on the *following* fragment — a human wrote it
       into the cut list, which design §8 makes the durable artifact.
    3. ``gap_ms > 0`` — one fixed length everywhere.
    4. Measured (the default). The pause belongs to the speaker who just
       stopped talking, so it is measured for ``previous``.
    """
    if options.gap_ms == 0:
        return SeamGap(0, "off")
    if following.gap_before_ms is not None:
        return SeamGap(max(0, int(following.gap_before_ms)), "override")
    if options.gap_ms > 0:
        return SeamGap(options.gap_ms, "fixed")
    key = (
        previous.speaker_label or "",
        ""
        if previous.video_speaker_id is None
        else str(previous.video_speaker_id),
    )
    cache_key = (f"{previous.video_id}", f"{key[0]}|{key[1]}")
    stats = cache.get(cache_key)
    if stats is None:
        stats = measure_pause_stats(
            db,
            video_id=previous.video_id,
            video_speaker_id=previous.video_speaker_id,
            speaker_label=previous.speaker_label,
        )
        cache[cache_key] = stats
    return SeamGap(
        applied_gap_ms(stats),
        "fallback" if stats.degenerate else "measured",
        stats.description if stats.degenerate else "",
    )


@dataclass(frozen=True)
class PlannedFragment:
    """One fragment with everything the encoder needs decided."""

    ord: int
    fragment: RenderFragment
    source: SourceMedia
    gap_after_ms: int
    gap_origin: str
    output_start_ms: int
    output_end_ms: int
    intermediate: Path


@dataclass(frozen=True)
class RenderPlan:
    """A whole render, decided but not yet executed."""

    render_id: str
    request: RenderRequest
    options: RenderOptions
    canvas: Canvas
    fragments: tuple[PlannedFragment, ...]
    sources: tuple[SourceMedia, ...]
    output_dir: Path
    output_path: Path
    report_path: Path
    list_file: Path
    duration_ms: int
    notes: tuple[str, ...]

    @property
    def fragments_dir(self) -> Path:
        """Where the per-fragment intermediates go, under the output dir."""
        return self.output_dir / "fragments"


def sanitize_id(text: str) -> str:
    """A path-safe, still-readable directory name."""
    return _UNSAFE.sub("-", text.strip()).strip("-")[: C.RENDER_ID_MAX_CHARS]


def plan_fingerprint(request: RenderRequest, options: RenderOptions) -> str:
    """A hash over exactly what reaches ffmpeg, and nothing else.

    Word ordinals are deliberately **not** in it. They are provenance —
    they never reach a filter graph, and Part 5 allows a hand-edit to
    drop them. Hashing them would move byte-identical output to a new
    ``render_id`` because somebody tidied a TOML file, which is the
    opposite of what the hash is for.
    """
    payload = {
        "name": request.name,
        "target": request.target_text,
        "fragments": [
            [f.video_id, f.start_ms, f.end_ms, f.gap_before_ms]
            for f in request.fragments
        ],
        "canvas_mode": options.canvas_mode,
        "height": options.height,
        "fps": options.fps,
        "gap_ms": options.gap_ms,
        "loudnorm": options.loudnorm,
        "preset": options.preset,
        "crf": options.crf,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_render_id(request: RenderRequest, options: RenderOptions) -> str:
    """``{cut list name}-{8 hex}``, stable for the same inputs.

    Deterministic on purpose: re-rendering the same cut list with the
    same options overwrites the same directory instead of littering
    ``output/`` with near-identical copies, and changing any option that
    affects the output makes a new one.
    """
    if options.render_id:
        return sanitize_id(options.render_id) or "render"
    room = C.RENDER_ID_MAX_CHARS - C.RENDER_ID_HASH_CHARS - 1
    stem = sanitize_id(request.name)[:room] or "render"
    digest = plan_fingerprint(request, options)[: C.RENDER_ID_HASH_CHARS]
    return f"{stem}-{digest}"


def plan_render(
    db: Database,
    request: RenderRequest,
    options: RenderOptions | None = None,
    *,
    tools: Tools | None = None,
) -> RenderPlan:
    """Decide the whole render. Reads; writes and encodes nothing."""
    options = options or RenderOptions()
    if not request.fragments:
        raise RenderError(f"cut list {request.name!r} has no fragments to render")
    for fragment in request.fragments:
        if fragment.duration_ms <= 0:
            raise RenderError(
                f"fragment from video {fragment.video_id} has a non-positive "
                f"duration: {fragment.start_ms}..{fragment.end_ms} ms"
            )
    tools = tools or Tools.resolve()
    notes: list[str] = []

    sources = list(resolve_sources(db, request.fragments, tools=tools))
    canvas = plan_canvas(
        [source.geometry for source in sources],
        mode=options.canvas_mode,
        height=options.height,
        fps=options.fps,
    )

    if options.loudnorm:
        first_use: dict[int, int] = {}
        for fragment in request.fragments:
            first_use.setdefault(fragment.video_id, fragment.start_ms)
        measured_sources = []
        for source in sources:
            lufs = measure_source_loudness(
                db, source, around_ms=first_use[source.video_id], tools=tools
            )
            if lufs is None:
                notes.append(
                    f"video {source.video_id}: loudness could not be measured; "
                    "its level was left alone"
                )
            measured_sources.append(
                replace(source, measured_lufs=lufs, gain_db=gain_for(lufs))
            )
        sources = measured_sources

    by_id = {source.video_id: source for source in sources}
    render_id = derive_render_id(request, options)
    output_dir = config.paths().output_dir(render_id)
    fragments_dir = output_dir / "fragments"

    planned: list[PlannedFragment] = []
    cache: dict[tuple[str, str], PauseStats] = {}
    clock = 0
    for index, fragment in enumerate(request.fragments):
        following = (
            request.fragments[index + 1] if index + 1 < len(request.fragments) else None
        )
        seam = (
            SeamGap(0, "end")
            if following is None
            else plan_gap(
                db,
                previous=fragment,
                following=following,
                options=options,
                cache=cache,
            )
        )
        if seam.note and seam.note not in notes:
            notes.append(seam.note)
        planned.append(
            PlannedFragment(
                ord=index,
                fragment=fragment,
                source=by_id[fragment.video_id],
                gap_after_ms=seam.gap_ms,
                gap_origin=seam.origin,
                output_start_ms=clock,
                output_end_ms=clock + fragment.duration_ms,
                intermediate=fragments_dir
                / f"{index:04d}{C.RENDER_INTERMEDIATE_SUFFIX}",
            )
        )
        clock += fragment.duration_ms + seam.gap_ms

    return RenderPlan(
        render_id=render_id,
        request=request,
        options=options,
        canvas=canvas,
        fragments=tuple(planned),
        sources=tuple(sources),
        output_dir=output_dir,
        output_path=output_dir / "output.mp4",
        report_path=output_dir / "report.md",
        list_file=output_dir / "concat.txt",
        duration_ms=clock,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Execution. Everything above decided; this part only does.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderResult:
    """What a finished render hands back to a command or a job."""

    render_id: str
    output_path: Path
    report_path: Path
    duration_ms: int
    fragment_count: int
    source_count: int
    plan: RenderPlan


def _speaker_of(fragment: RenderFragment) -> str | None:
    """A human-readable speaker, or the diarized label, or nothing."""
    if fragment.speaker_label:
        return fragment.speaker_label
    if fragment.video_speaker_id is not None:
        return f"label {fragment.video_speaker_id}"
    return None


def build_report(
    plan: RenderPlan, *, notes: Sequence[str], created_at: str
) -> RenderReport:
    """Turn a plan plus what execution learned into the report tree."""
    fragments = tuple(
        FragmentReport(
            ord=planned.ord,
            video_id=planned.source.video_id,
            video_title=planned.source.title,
            video_url=planned.source.url,
            speaker=_speaker_of(planned.fragment),
            source_start_ms=planned.fragment.start_ms,
            source_end_ms=planned.fragment.end_ms,
            output_start_ms=planned.output_start_ms,
            output_end_ms=planned.output_end_ms,
            gap_after_ms=planned.gap_after_ms,
            gap_origin=planned.gap_origin,
            text=planned.fragment.text,
        )
        for planned in plan.fragments
    )
    sources = []
    for source in plan.sources:
        mine = [p for p in plan.fragments if p.source.video_id == source.video_id]
        geometry = source.geometry
        sources.append(
            SourceReport(
                video_id=source.video_id,
                title=source.title,
                url=source.url,
                fragment_count=len(mine),
                used_ms=sum(p.fragment.duration_ms for p in mine),
                measured_lufs=source.measured_lufs,
                gain_db=source.gain_db,
                first_output_ms=min(p.output_start_ms for p in mine),
                geometry=f"{geometry.width}x{geometry.height} @ {round(geometry.fps)} fps",
            )
        )
    return RenderReport(
        render_id=plan.render_id,
        cutlist=plan.request.name,
        created_at=created_at,
        output_path=str(plan.output_path),
        duration_ms=plan.duration_ms,
        canvas=plan.canvas.label,
        loudnorm=plan.options.loudnorm,
        gap_policy=plan.options.gap_policy,
        target_text=plan.request.target_text,
        assembled_text=plan.request.assembled_text,
        fragments=fragments,
        sources=tuple(sources),
        missing=plan.request.missing,
        notes=tuple(notes),
    )


def render_cutlist(
    db: Database,
    request: RenderRequest,
    options: RenderOptions | None = None,
    *,
    tools: Tools | None = None,
    dry_run: bool = False,
    render_row_id: int | None = None,
) -> RenderResult:
    """Plan, then execute: ``output/{render_id}/output.mp4`` and its report.

    One encode per fragment into a concat-safe intermediate, then one
    joining pass. ``dry_run`` stops after planning — same answer, no
    files, no encoder — which is how you check the canvas, the gaps and
    the sources before committing an hour of CPU to them.

    ``render_row_id`` is a ``renders`` row opened by the caller; when
    given it is closed ``rendered`` or ``failed`` here, so a crash
    mid-encode leaves the row saying so rather than saying nothing.
    """
    options = options or RenderOptions()
    tools = tools or Tools.resolve()
    try:
        plan = plan_render(db, request, options, tools=tools)
    except Exception:
        if render_row_id is not None:
            finish_render(db, render_row_id, state="failed")
        raise
    notes = list(plan.notes)

    if dry_run:
        return RenderResult(
            render_id=plan.render_id,
            output_path=plan.output_path,
            report_path=plan.report_path,
            duration_ms=plan.duration_ms,
            fragment_count=len(plan.fragments),
            source_count=len(plan.sources),
            plan=plan,
        )

    try:
        _encode(plan, options, tools, notes)
    except Exception:
        if render_row_id is not None:
            finish_render(db, render_row_id, state="failed")
        raise
    if render_row_id is not None:
        finish_render(
            db, render_row_id, state="rendered", output_path=plan.output_path
        )

    return RenderResult(
        render_id=plan.render_id,
        output_path=plan.output_path,
        report_path=plan.report_path,
        duration_ms=plan.duration_ms,
        fragment_count=len(plan.fragments),
        source_count=len(plan.sources),
        plan=plan,
    )


def _encode(
    plan: RenderPlan, options: RenderOptions, tools: Tools, notes: list[str]
) -> None:
    """Everything that touches the disk: fragments, join, report, cleanup."""
    config.ensure_dir(plan.fragments_dir)
    total = len(plan.fragments)
    for planned in plan.fragments:
        run_command(
            fragment_command(
                video_path=planned.source.video_path,
                audio_path=planned.source.audio_path,
                start_ms=planned.fragment.start_ms,
                end_ms=planned.fragment.end_ms,
                video_filter=video_filter_chain(
                    plan.canvas, gap_ms=planned.gap_after_ms
                ),
                out_path=planned.intermediate,
                gain_db=planned.source.gain_db,
                gap_ms=planned.gap_after_ms,
                preset=options.preset,
                crf=options.crf,
                binary=tools.ffmpeg,
            ),
            runner=tools.runner,
            what=f"fragment {planned.ord + 1}/{total}",
        )

    write_concat_list(plan.list_file, [p.intermediate for p in plan.fragments])

    measured = None
    if options.loudnorm:
        scan = run_command(
            programme_loudness_command(
                list_file=plan.list_file, binary=tools.ffmpeg
            ),
            runner=tools.runner,
            what="loudness scan of the assembled programme",
        )
        measured = parse_loudnorm_json(scan.stderr)
        if measured is None:
            notes.append(
                "the assembled programme could not be measured; its level was "
                "left alone"
            )

    run_command(
        concat_command(
            list_file=plan.list_file,
            out_path=plan.output_path,
            measured=measured,
            binary=tools.ffmpeg,
        ),
        runner=tools.runner,
        what="joining the fragments",
    )
    if not plan.output_path.exists():
        raise FfmpegFailedError(
            f"ffmpeg reported success but wrote no {plan.output_path.name}"
        )

    plan.report_path.write_text(
        render_markdown(build_report(plan, notes=notes, created_at=utc_now_iso())),
        encoding="utf-8",
    )

    if not options.keep_intermediates:
        # Kept on failure, because that is when they are worth looking at.
        shutil.rmtree(plan.fragments_dir, ignore_errors=True)
        plan.list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The job entry point (contracts §5 "Job handlers"; Part 2 owns the registry)
# ---------------------------------------------------------------------------


def create_render(
    db: Database,
    *,
    cutlist_name: str,
    options: RenderOptions,
    now: str | None = None,
) -> int:
    """Open a ``renders`` row and return its id — the job's ``target_id``.

    Contracts §3 gives a render its own identity, so a job addresses it
    the way every other kind addresses its target: an integer primary
    key, no hashing of names into integers and nothing to verify later.
    The row is opened before any work, so a render that dies mid-encode
    is still on record as ``planned``.
    """
    with db.transaction():
        cursor = db.conn.execute(
            "INSERT INTO renders (cutlist_name, output_path, canvas_mode, state, "
            "created_at) VALUES (?, NULL, ?, 'planned', ?)",
            (cutlist_name, options.canvas_mode, now or utc_now_iso()),
        )
        assert cursor.lastrowid is not None  # an INSERT always sets it
        return int(cursor.lastrowid)


def get_render(db: Database, render_id: int) -> sqlite3.Row:
    """One render row, or :class:`NotFoundError`."""
    row = db.conn.execute(
        "SELECT * FROM renders WHERE id = ?", (render_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"no render {render_id}")
    return row


def finish_render(
    db: Database,
    render_id: int,
    *,
    state: str,
    output_path: Path | None = None,
    now: str | None = None,
) -> None:
    """Close a render row as ``rendered`` or ``failed``."""
    with db.transaction():
        db.conn.execute(
            "UPDATE renders SET state = ?, "
            "output_path = COALESCE(?, output_path), finished_at = ? WHERE id = ?",
            (
                state,
                None if output_path is None else str(output_path),
                now or utc_now_iso(),
                render_id,
            ),
        )


def payload_for(options: RenderOptions) -> dict[str, object]:
    """The ``jobs.payload_json`` body for one render.

    Options only: which cut list it is lives in the ``renders`` row the
    job targets, so there is one place for it rather than two that can
    disagree.
    """
    return {
        "canvas_mode": options.canvas_mode,
        "height": options.height,
        "fps": options.fps,
        "gap_ms": options.gap_ms,
        "loudnorm": options.loudnorm,
        "preset": options.preset,
        "crf": options.crf,
        "keep_intermediates": options.keep_intermediates,
        "render_id": options.render_id,
    }


def options_from_payload(payload: dict) -> RenderOptions:
    """Rebuild the options a queued render was enqueued with."""
    defaults = RenderOptions()
    return RenderOptions(
        canvas_mode=str(payload.get("canvas_mode", defaults.canvas_mode)),
        height=int(payload.get("height", defaults.height)),
        fps=int(payload.get("fps", defaults.fps)),
        gap_ms=int(payload.get("gap_ms", defaults.gap_ms)),
        loudnorm=bool(payload.get("loudnorm", defaults.loudnorm)),
        preset=str(payload.get("preset", defaults.preset)),
        crf=int(payload.get("crf", defaults.crf)),
        keep_intermediates=bool(
            payload.get("keep_intermediates", defaults.keep_intermediates)
        ),
        render_id=str(payload.get("render_id", defaults.render_id)),
    )


def run_render_job(db: Database, target_id: int, payload: dict) -> None:
    """The ``render`` job kind's handler. contracts §5.

    ``target_id`` is a ``renders.id``: the row says which cut list this
    is, so nothing is re-derived from the payload and nothing needs
    checking against it. Idempotent — the output path is derived from
    the cut list and the options, so a repeat overwrites the same file.
    """
    row = get_render(db, target_id)
    name = str(row["cutlist_name"])
    path = config.paths().cutlist(name)
    if not path.exists():
        finish_render(db, target_id, state="failed")
        raise RenderError(f"cut list {name!r} not found at {path}")
    # Lazy, and one of only two import sites for Part 5 (contracts §1).
    from rytp.assemble.cutlist import load_cutlist

    render_cutlist(
        db,
        request_from_cutlist(load_cutlist(path)),
        options_from_payload(payload),
        render_row_id=target_id,
    )


# ---------------------------------------------------------------------------
# Removal (contracts §5 "Deletion")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderRemoval:
    """What removing one render would take, or did take."""

    render_id: int
    cutlist_name: str
    cutlist_present: bool
    output_dir: Path | None
    files: tuple[tuple[str, int], ...]  # (path, bytes)
    total_bytes: int
    jobs_cancelled: int
    removed: bool


def _removable_output_dir(row: sqlite3.Row) -> Path | None:
    """The directory this render owns, if it is safely inside the tree.

    ``renders.output_path`` is text a human can edit, so the path is
    checked against the data tree's ``output/`` root before anything
    recursive happens to it. Outside the tree — or empty — means there
    is nothing this command is willing to delete, and only the row goes.
    """
    raw = row["output_path"]
    if not raw:
        return None
    candidate = Path(str(raw)).parent
    # Paths() has no output root accessor; every render directory is a
    # child of this one, which is the same thing.
    root = config.paths().output_dir("_").parent
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def plan_removal(db: Database, render_id: int) -> RenderRemoval:
    """Exactly what :func:`remove_render` would do. Touches nothing."""
    row = get_render(db, render_id)
    name = str(row["cutlist_name"])
    directory = _removable_output_dir(row)
    files: list[tuple[str, int]] = []
    if directory is not None and directory.is_dir():
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            files.append((str(path), path.stat().st_size))
    queued = db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'render' AND target_id = ? "
        "AND state IN ('pending', 'blocked', 'running')",
        (render_id,),
    ).fetchone()[0]
    return RenderRemoval(
        render_id=render_id,
        cutlist_name=name,
        cutlist_present=config.paths().cutlist(name).exists(),
        output_dir=directory,
        files=tuple(files),
        total_bytes=sum(size for _, size in files),
        jobs_cancelled=int(queued),
        removed=False,
    )


def remove_render(
    db: Database, render_id: int, *, dry_run: bool = False
) -> RenderRemoval:
    """Delete a render's output directory and its row. contracts §5.

    Synchronous, never a job: a half-deleted render recovered from a
    crashed queue is worse than a slow command. The job cancellation and
    the row deletion share one transaction because ``jobs`` has no
    foreign key to lean on — a surviving job would target a render that
    no longer exists.
    """
    plan = plan_removal(db, render_id)
    if dry_run:
        return plan
    if plan.output_dir is not None:
        shutil.rmtree(plan.output_dir, ignore_errors=True)
    with db.transaction():
        db.conn.execute(
            "UPDATE jobs SET state = 'cancelled', finished_at = ? "
            "WHERE kind = 'render' AND target_id = ? "
            "AND state IN ('pending', 'blocked', 'running')",
            (utc_now_iso(), render_id),
        )
        db.conn.execute("DELETE FROM renders WHERE id = ?", (render_id,))
    return replace(plan, removed=True)
