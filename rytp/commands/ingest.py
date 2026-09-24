"""Acquisition and queue commands. design §5, contracts §5.

Handlers never print and never exit — they return a
:class:`~rytp.commands.CommandResult` or raise, and the CLI or the TUI
decides what that looks like.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from rytp import config
from rytp import constants as C
from rytp.acquire import acquire_media
from rytp.acquire.captions import acquire_captions
from rytp.acquire.local import register_local_container
from rytp.audio.extract import ensure_wav, prune_wav_cache
from rytp.commands import (
    REQUIRED,
    Command,
    CommandResult,
    HealthCheck,
    HealthResult,
    Param,
    register,
    register_check,
    resolve_video_id,
)
from rytp.db import Database
from rytp.db.queries import get_setting
from rytp.jobs import JOB_KINDS
from rytp.jobs import queue as Q
from rytp.jobs import worker as W
from rytp.models import InvalidInputError, NotFoundError, RytpError


def _video_source(db: Database, video_id: int) -> str:
    row = db.conn.execute(
        "SELECT source FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"video {video_id} is not in the catalog")
    return str(row["source"])


def _select_videos(
    db: Database, *, video_id: int, channel_id: int, pending: bool, limit: int
) -> list[tuple[int, str]]:
    """Which videos this invocation ingests: one named, or a filtered set."""
    if video_id:
        row = db.conn.execute(
            "SELECT id, source FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"video {video_id} is not in the catalog")
        return [(int(row["id"]), str(row["source"]))]

    where: list[str] = []
    params: list[Any] = []
    if channel_id:
        where.append("v.channel_id = ?")
        params.append(channel_id)
    if pending:
        # "Not yet ingested" = no extract_wav job, which every chain contains.
        where.append(
            "NOT EXISTS (SELECT 1 FROM jobs j "
            "WHERE j.target_id = v.id AND j.kind = 'extract_wav')"
        )
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT v.id, v.source FROM videos v {clause} ORDER BY v.id LIMIT ?",
        params,
    ).fetchall()
    return [(int(r["id"]), str(r["source"])) for r in rows]


def _reject_unknown_aligner(name: str) -> None:
    """Refuse a misspelled aligner now, not after five failed retries.

    contracts §3: "a job enqueued with an unregistered aligner name is
    rejected at enqueue time". If Part 3 is not installed there is no
    registry to check against and no `align` kind registered either, so
    there is nothing to reject.

    **The adapter package is imported here on purpose**, mirroring
    :func:`_reject_unknown_transcriber`: an aligner registers itself at
    import time and every Part 3 import of the adapter package is deferred
    into a function body (contracts §1), so reading `ALIGNERS` without
    importing it first would see an empty dict and reject every name,
    including a perfectly valid `default_aligner` the moment it is set.
    """
    try:
        import rytp.transcribe.align  # noqa: F401 - importing is registering
        from rytp.transcribe.registry import ALIGNERS
    except ImportError:
        return
    if name not in ALIGNERS:
        raise InvalidInputError(
            f"setting {C.SETTING_DEFAULT_ALIGNER}={name!r} is not a registered "
            f"aligner; available: {', '.join(sorted(ALIGNERS)) or '(none)'}"
        )


def _reject_unknown_transcriber(name: str) -> None:
    """Refuse a misspelled transcriber now, not after five failed retries.

    contracts §3 "Choosing a transcriber" applies the same enqueue-time rule
    the aligner gets. If Part 3 is not installed there is no registry to
    check against and no `transcribe` kind registered either, so there is
    nothing to reject.

    **The adapter package is imported here on purpose.** A Part 3 engine
    registers itself at import time and every Part 3 import of the adapter
    packages is deferred into a function body, so reading `TRANSCRIBERS`
    without importing them first sees an empty dict — which would reject
    every name, including the configured default, on every ingest.
    """
    try:
        import rytp.transcribe.engines  # noqa: F401 - importing is registering
        from rytp.transcribe.registry import TRANSCRIBERS
    except ImportError:
        return
    if name not in TRANSCRIBERS:
        raise InvalidInputError(
            f"setting {C.SETTING_DEFAULT_TRANSCRIBER}={name!r} is not a registered "
            f"transcriber; available: {', '.join(sorted(TRANSCRIBERS)) or '(none)'}"
        )


def _transcriber_payload(db: Database, explicit: str) -> tuple[dict[str, str], bool]:
    """The engine every `transcribe` job is stamped with, and where it came from.

    contracts §3: a `transcribe` job that names no engine resolves one at run
    time, where nothing announces the choice — and the corpus is Russian,
    the Russian-specific engine roughly halves Whisper's word error rate
    there, and the whole archive is 35-90 GPU-hours. Getting that wrong
    silently is expensive to discover late.

    Unlike :func:`_aligner_payload` this never returns "do not enqueue":
    an empty aligner reads coherently as "no alignment, the words stay
    `timed`", but an empty transcriber would make `--transcribe` do nothing.
    So the setting always resolves to a name.

    The second element is True when the name came from the setting rather
    than from an explicit `--transcriber`, which is what the caller reports.
    """
    name = explicit.strip()
    from_setting = not name
    if from_setting:
        name = (
            get_setting(db, C.SETTING_DEFAULT_TRANSCRIBER) or ""
        ).strip() or C.DEFAULT_TRANSCRIBER_FALLBACK
    _reject_unknown_transcriber(name)
    return {"transcriber": name}, from_setting


def _aligner_payload(db: Database) -> dict[str, str] | None:
    """The payload every `align` job needs, or None meaning "do not enqueue".

    Part 3's align handler raises without `payload["aligner"]`, so an align
    job created without one is a job that fails five times and dies. The
    answer is a configurable default that is empty out of the box:
    alignment needs MFA installed, and nobody should discover that through
    five silent retries.
    """
    name = (get_setting(db, C.SETTING_DEFAULT_ALIGNER) or "").strip()
    if not name:
        return None
    _reject_unknown_aligner(name)
    return {"aligner": name}


def ingest(
    db: Database,
    *,
    video: str = "",
    channel_id: int = 0,
    pending: bool = False,
    transcribe: bool = False,
    transcriber: str = "",
    limit: int = C.INGEST_DEFAULT_LIMIT,
    priority: int = 0,
    dry_run: bool = False,
) -> CommandResult:
    """Enqueue the chain for one video, or for a filtered set of them.

    design §11 M2 is "catalog the channel, pull captions for everything" at
    roughly 1,600 videos, so the bulk form is the point of this command, not
    a convenience: `rytp ingest --pending --limit 2000` is M2.
    """
    video_id = resolve_video_id(db, video) if video else 0
    targets = _select_videos(
        db, video_id=video_id, channel_id=channel_id, pending=pending, limit=limit
    )
    if dry_run:
        return CommandResult(
            columns=("video", "source"),
            rows=tuple((str(v), src) for v, src in targets),
            message=f"would ingest {len(targets)} video(s)",
        )

    # Both resolved once, before anything is enqueued, so a misspelled name
    # fails the whole command instead of poisoning 1,600 jobs.
    engine: dict[str, str] | None = None
    engine_from_setting = False
    if transcribe:
        engine, engine_from_setting = _transcriber_payload(db, transcriber)
    aligner = _aligner_payload(db) if transcribe else None
    unaligned = transcribe and aligner is None
    payloads = {"transcribe": engine, "align": aligner}

    states = ("pending", "blocked", "done")
    tally: dict[str, dict[str, int]] = {}
    order: list[str] = []
    missing: set[str] = set()
    skipped: list[str] = []
    ingested = 0

    for vid, source in targets:
        if source == C.LOCAL_SOURCE:
            try:
                register_local_container(db, vid)
            except RytpError as exc:
                # One unreadable file must not abort a 1,600-video run.
                skipped.append(f"{vid}: {exc}")
                continue
            chain = C.INGEST_CHAIN_LOCAL
        else:
            chain = C.INGEST_CHAIN_REMOTE
        if transcribe:
            chain = (*chain, *C.INGEST_CHAIN_TRANSCRIBE)

        ingested += 1
        for kind in chain:
            # Kinds owned by Parts 3, 4 and 6 appear here only once those
            # parts have registered them. Part 2 stays shippable on its own
            # and the chain completes itself as they land.
            if kind not in JOB_KINDS:
                missing.add(kind)
                continue
            if kind == "align" and aligner is None:
                continue
            if kind not in tally:
                tally[kind] = dict.fromkeys(("queued", *states), 0)
                order.append(kind)
            job_id = Q.enqueue(
                db, kind, vid, priority=priority, payload=payloads.get(kind),
            )
            state = Q.get_job(db, job_id).state
            tally[kind]["queued"] += 1
            if state in tally[kind]:
                tally[kind][state] += 1

    parts = [f"ingested {ingested} video(s)"]
    if engine is not None and engine_from_setting:
        # contracts §3: when the default was used rather than an explicit
        # choice, say which engine it picked. This one line is the whole
        # mechanism that keeps 35-90 GPU-hours deliberate instead of
        # accidental; `rytp transcribe compare` is how you revisit it.
        parts.append(
            f"transcriber {engine['transcriber']!r}, from setting "
            f"{C.SETTING_DEFAULT_TRANSCRIBER} (pass --transcriber to override)"
        )
    if unaligned:
        parts.append(
            f"no {C.SETTING_DEFAULT_ALIGNER} set, so no align jobs — words will "
            f"stay in the `timed` tier (searchable, not cuttable) until you "
            f"align them deliberately"
        )
    if missing:
        parts.append(
            f"not yet available, skipped: {', '.join(sorted(missing))}"
        )
    if skipped:
        parts.append(f"{len(skipped)} skipped ({'; '.join(skipped[:3])})")
    return CommandResult(
        columns=("kind", "queued", *states),
        rows=tuple(
            (kind, *(str(tally[kind][c]) for c in ("queued", *states)))
            for kind in order
        ),
        message="; ".join(parts),
    )


def fetch_video(db: Database, *, video: str, captions: bool = True) -> CommandResult:
    """Acquire one video right now, without going through the queue."""
    video_id = resolve_video_id(db, video)
    source = _video_source(db, video_id)
    notes: list[str] = []
    if source == C.LOCAL_SOURCE:
        notes.append(register_local_container(db, video_id))
    else:
        notes.append(acquire_media(db, video_id))
        if captions:
            result = acquire_captions(db, video_id)
            notes.append(result.message)
            if result.fallback_note:
                notes.append(result.fallback_note)
    notes.append(f"wav cached at {ensure_wav(db, video_id)}")
    Q.reconcile(db)
    return CommandResult(message="; ".join(notes))


def worker(
    db: Database, *, pool: str = "all", once: bool = False, max_jobs: int = 0
) -> CommandResult:
    """Run the worker until Ctrl-C, or until the queue empties with --once."""
    pools = C.POOLS if pool == "all" else (pool,)
    report = W.run_worker(
        db, W.WorkerOptions(pools=pools, once=once, max_jobs=max_jobs)
    )
    return CommandResult(
        message=(
            f"{report.claimed} claimed, {report.done} done, "
            f"{report.failed} failed, {report.deferred} deferred, "
            f"{report.blocked} blocked"
        )
    )


register(
    Command(
        name="ingest",
        group="",
        summary="Enqueue the acquisition chain for one video or a whole channel.",
        params=(
            Param("video", str, "Catalog id, external id, or catalogued URL of one "
                  "video — a URL only resolves once it is registered with `videos "
                  "add`. Empty uses the filters.", default="", positional=True),
            Param("channel_id", int, "Only videos of this channel. 0 means any.",
                  default=0, short="-c"),
            Param("pending", bool, "Only videos that have never been ingested.",
                  default=False),
            Param("transcribe", bool,
                  "Also enqueue transcription and alignment (tier 2, GPU).",
                  default=False),
            Param("transcriber", str,
                  "Registered transcriber to stamp on the transcribe jobs. "
                  "Empty uses the default_transcriber setting, and the result "
                  "says which engine that chose.",
                  default=""),
            Param("limit", int, "Maximum videos to touch in the bulk form.",
                  default=C.INGEST_DEFAULT_LIMIT, short="-n"),
            Param("priority", int, "Higher runs first.", default=0, short="-p"),
            Param("dry_run", bool, "List what would be ingested, enqueue nothing.",
                  default=False),
        ),
        handler=ingest,
    )
)

register(
    Command(
        name="fetch-video",
        group="",
        summary="Download one video's assets and cache its WAV, right now.",
        params=(
            Param("video", str, "Catalog id, external id, or catalogued URL of the "
                  "video — a URL only resolves once it is registered with `videos "
                  "add`.", default=REQUIRED, positional=True),
            Param("captions", bool, "Also fetch the caption track.", default=True),
        ),
        handler=fetch_video,
        long_running=True,
    )
)

register(
    Command(
        name="worker",
        group="",
        summary="Run the job queue's worker pools.",
        params=(
            Param("pool", str, "Which pool to run.", default="all",
                  choices=("all", *C.POOLS)),
            Param("once", bool, "Drain the queue and exit instead of waiting.",
                  default=False),
            Param("max_jobs", int, "Stop after this many jobs. 0 means no limit.",
                  default=0),
        ),
        handler=worker,
        long_running=True,
    )
)


_JOB_STATES: tuple[str, ...] = (
    "pending", "running", "done", "failed", "blocked", "cancelled"
)


def jobs_list(
    db: Database, *, state: str = "", pool: str = "", kind: str = "",
    limit: int = C.JOB_LIST_LIMIT,
) -> CommandResult:
    """Show jobs, highest priority first."""
    jobs = Q.list_jobs(
        db, state=state or None, pool=pool or None, kind=kind or None, limit=limit
    )

    def _cell(text: str | None) -> str:
        return next(iter((text or "").splitlines()), "")[: C.JOB_ERROR_PREVIEW_CHARS]

    rows = tuple(
        (
            str(j.id), j.kind, str(j.target_id), j.state, j.pool, str(j.attempts),
            j.not_before or "", _cell(j.last_error), _cell(j.note),
        )
        for j in jobs
    )
    return CommandResult(
        columns=("id", "kind", "target", "state", "pool", "attempts",
                 "not_before", "error", "note"),
        rows=rows,
        message=None if rows else "no jobs match",
    )


def jobs_stats(db: Database) -> CommandResult:
    """Counts by state and pool, plus how much of the queue is backing off."""
    s = Q.stats(db)
    rows: list[tuple[str, str]] = [(name, str(n)) for name, n in s.by_state.items()]
    rows += [(f"pending on {pool}", str(n)) for pool, n in s.pending_by_pool.items()]
    # design §5: throttling must be visible, not a mystery stall.
    rows.append(("throttled", str(s.throttled)))
    rows.append(("next attempt", s.next_not_before or ""))
    # Forty videos that quietly lost their speaker labels must be visible
    # here, not only by reading every row of `jobs list`.
    rows.append(("with notes", str(s.noted)))
    rows.append(("paused", "yes" if s.paused else "no"))
    return CommandResult(columns=("metric", "value"), rows=tuple(rows))


def jobs_retry(
    db: Database, *, job_id: int = 0, kind: str = "", state: str = "failed"
) -> CommandResult:
    """Send settled jobs back to pending, then re-ask what is runnable."""
    changed = Q.retry(
        db, job_id=job_id or None, kind=kind or None, state=state
    )
    reopened = Q.reconcile(db)
    return CommandResult(
        message=f"{changed} job(s) retried, {reopened} reopened by reconcile"
    )


def queue_pause(db: Database) -> CommandResult:
    """Stop every pool from claiming. Running jobs finish."""
    Q.pause(db)
    return CommandResult(message="queue paused; running jobs will finish")


def queue_resume(db: Database) -> CommandResult:
    Q.resume(db)
    return CommandResult(message="queue resumed")


def cache_prune(db: Database, *, video: str | None = None, dry_run: bool = False) -> CommandResult:
    """Delete cached WAVs. They are regenerable, and the jobs reopen.

    design §4 calls the WAV a cache with an explicit prune command; design
    §5 makes pruning it enough to bring the extract jobs back, which is why
    this reconciles afterwards instead of leaving a stale ``done``.
    """
    video_id = resolve_video_id(db, video) if video else None
    freed = prune_wav_cache(db, video_id=video_id, dry_run=dry_run)
    total = sum(size for _, size in freed)
    reopened = 0 if dry_run else Q.reconcile(db, kinds=("extract_wav",))
    verb = "would free" if dry_run else "freed"
    return CommandResult(
        columns=("file", "bytes"),
        rows=tuple((p.name, str(size)) for p, size in freed),
        message=(
            f"{verb} {total} bytes across {len(freed)} file(s); "
            f"{reopened} extract job(s) reopened"
        ),
    )


register(
    Command(
        name="jobs.list",
        group="jobs",
        summary="List queued, running and finished jobs.",
        params=(
            Param("state", str, "Only this state.", default="",
                  choices=("", *_JOB_STATES)),
            Param("pool", str, "Only this pool.", default="", choices=("", *C.POOLS)),
            Param("kind", str, "Only this job kind.", default=""),
            Param("limit", int, "Maximum rows.", default=C.JOB_LIST_LIMIT, short="-n"),
        ),
        handler=jobs_list,
    )
)

register(
    Command(
        name="jobs.stats",
        group="jobs",
        summary="Summarise the queue, including how much of it is backing off.",
        params=(),
        handler=jobs_stats,
    )
)

register(
    Command(
        name="jobs.retry",
        group="jobs",
        summary="Send settled jobs back to pending.",
        params=(
            Param("job_id", int, "One job id. 0 means every match.", default=0),
            Param("kind", str, "Only this job kind.", default=""),
            Param("state", str, "Which state to revive.", default="failed",
                  choices=("failed", "blocked", "cancelled")),
        ),
        handler=jobs_retry,
    )
)

register(
    Command(
        name="queue.pause",
        group="queue",
        summary="Stop the worker pools from claiming new jobs.",
        params=(),
        handler=queue_pause,
    )
)

register(
    Command(
        name="queue.resume",
        group="queue",
        summary="Let the worker pools claim jobs again.",
        params=(),
        handler=queue_resume,
    )
)

register(
    Command(
        name="cache.prune",
        group="cache",
        summary="Delete cached WAVs. They are regenerable.",
        params=(
            Param("video", str, "One video's WAV, by id or external id. Omit to prune "
                  "all of them.", default=None),
            Param("dry_run", bool, "Report what would go, delete nothing.",
                  default=False),
        ),
        handler=cache_prune,
    )
)


def jobs_cancel(
    db: Database,
    *,
    job_id: int = 0,
    kind: str = "",
    state: str = "",
    target_id: int = 0,
    dry_run: bool = False,
) -> CommandResult:
    """Drop queued or failed jobs, by id or by filter."""
    if not (job_id or kind or state or target_id):
        raise InvalidInputError(
            "jobs cancel needs at least one filter (--job-id, --kind, --state "
            "or --target-id); cancelling the whole queue by accident is worse "
            "than typing one more flag"
        )
    sel_job_id = job_id or None
    sel_kind = kind or None
    sel_state = state or None
    sel_target_id = target_id or None
    running = Q.running_matches(
        db, job_id=sel_job_id, kind=sel_kind, state=sel_state, target_id=sel_target_id
    )
    doomed = Q.cancellable(
        db, job_id=sel_job_id, kind=sel_kind, state=sel_state, target_id=sel_target_id
    )

    # A worker is mid-download inside yt-dlp or ffmpeg; there is no checkpoint
    # for it to abandon the job at, so marking it cancelled would be a lie the
    # worker overwrites with 'done' a minute later.
    if job_id and running:
        raise InvalidInputError(
            f"job {job_id} is running; stop the worker first "
            f"(`rytp queue pause`, then Ctrl-C it). A worker that has already "
            f"died leaves its jobs to be reclaimed on the next start."
        )

    rows = tuple((str(j.id), j.kind, str(j.target_id), j.state) for j in doomed)
    if dry_run:
        return CommandResult(
            columns=("id", "kind", "target", "state"),
            rows=rows,
            message=f"would cancel {len(doomed)} job(s)",
        )

    Q.cancel(db, doomed)
    parts = [f"cancelled {len(doomed)} job(s)"]
    if running:
        parts.append(
            f"{len(running)} still running and left alone — pause the queue "
            f"and stop the worker to cancel those"
        )
    parts.append("`rytp jobs retry --state cancelled` brings them back")
    return CommandResult(columns=("id", "kind", "target", "state"), rows=rows,
                         message="; ".join(parts))


def _has_better_than_captions(db: Database, video_id: int) -> bool:
    """Is there a transcript that supersedes the downloaded captions?

    Contracts §3's tier table: `timed` is already far better *text* than
    captions even though it is not cuttable, so either tier supersedes them.
    """
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source IN ('timed', 'aligned') "
        "LIMIT 1",
        (video_id,),
    ).fetchone()
    return row is not None


def assets_remove(
    db: Database, *, asset_id: int, yes: bool = False, dry_run: bool = False
) -> CommandResult:
    """Delete one asset and its file. Typically a superseded rendition."""
    row = db.conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"asset {asset_id} does not exist")

    path = Path(row["path"])
    size = path.stat().st_size if path.is_file() else 0
    label = f"{row['role']}"
    if row["format_id"]:
        label += f" {row['format_id']}"

    warnings: list[str] = []
    if row["role"] in ("audio", "container"):
        warnings.append(
            f"WARNING: this is video {row['video_id']}'s {row['role']}; removing "
            f"it breaks re-align, render and WAV extraction until it is fetched "
            f"again"
        )
    if row["role"] == "captions" and not _has_better_than_captions(db, int(row["video_id"])):
        warnings.append(
            "WARNING: this video's caption words have not been superseded by an "
            "aligned transcript, so removing the track loses the only text there is"
        )

    if dry_run:
        return CommandResult(
            columns=("asset", "role", "path", "bytes"),
            rows=((str(asset_id), label, str(path), str(size)),),
            message="; ".join([f"would free {size} bytes", *warnings]),
        )
    if not yes:
        raise InvalidInputError(
            f"assets remove deletes {path} ({size} bytes) from disk; pass --yes "
            f"to confirm, or --dry-run to see what would go"
        )

    if path.is_file():
        path.unlink()
    db.conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
    # The queue must reflect what is on disk: dropping a rendition makes its
    # download job runnable again, which is design §5's self-healing seen
    # from the deletion side.
    reopened = Q.reconcile(db, target_id=int(row["video_id"]), target_kind="video")
    return CommandResult(
        message="; ".join([
            f"removed {label} asset {asset_id}, freed {size} bytes",
            f"{reopened} job(s) reopened",
            *warnings,
        ])
    )


register(
    Command(
        name="jobs.cancel",
        group="jobs",
        summary="Drop queued or failed jobs from the queue, by id or filter.",
        params=(
            Param("job_id", int, "One job id. 0 means use the filters.", default=0),
            Param("kind", str, "Only this job kind.", default=""),
            Param("state", str, "Only this state.", default="",
                  choices=("", *_JOB_STATES)),
            Param("target_id", int, "Only jobs for this target. 0 means any.",
                  default=0),
            Param("dry_run", bool, "List what would be cancelled, change nothing.",
                  default=False),
        ),
        handler=jobs_cancel,
    )
)

register(
    Command(
        name="assets.remove",
        group="assets",
        summary="Delete one asset and its file, to reclaim disk after an upgrade.",
        params=(
            Param("asset_id", int, "Id of the asset, from `rytp jobs list` or the "
                  "assets table.", default=REQUIRED, positional=True),
            Param("yes", bool, "Confirm; required because a file is deleted.",
                  default=False),
            Param("dry_run", bool, "Report what would go, delete nothing.",
                  default=False),
        ),
        handler=assets_remove,
    )
)


# ---------------------------------------------------------------------------
# doctor checks (contracts §5, Health checks). Part 1 owns the command and the
# registry; these four are Part 2's. None of them may raise, and a missing
# optional dependency is reported rather than failed.
# ---------------------------------------------------------------------------

_FFMPEG_REMEDY = (
    "install ffmpeg from https://ffmpeg.org/ and put its bin/ directory on PATH"
)


def _run_version(exe: str) -> subprocess.CompletedProcess[str]:
    """Isolated so tests can replace it without a real binary."""
    return subprocess.run(
        [exe, "-version"], check=False, capture_output=True, text=True,
        timeout=C.HEALTH_VERSION_TIMEOUT_S,
    )


def _binary_check(name: str) -> HealthResult:
    """Is this binary on PATH, and what does it say its version is?"""
    try:
        exe = shutil.which(name)
        if exe is None:
            return HealthResult(
                ok=False, detail=f"{name} is not on PATH", remedy=_FFMPEG_REMEDY
            )
        proc = _run_version(exe)
        first = next(iter(proc.stdout.splitlines()), "").strip()
        if proc.returncode != 0:
            return HealthResult(
                ok=False,
                detail=f"{exe} exited {proc.returncode} for -version",
                remedy=_FFMPEG_REMEDY,
            )
        return HealthResult(ok=True, detail=f"{exe}: {first or 'version unknown'}")
    except Exception as exc:
        return HealthResult(ok=False, detail=f"{name}: {exc}", remedy=_FFMPEG_REMEDY)


def _ytdlp_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("yt-dlp")
    except PackageNotFoundError:
        return None


def check_ytdlp(db: Database) -> HealthResult:
    """Is yt-dlp installed?

    ``ok`` reports honestly that it is not. The check is registered with
    ``required=False``, which is what keeps that from failing ``doctor``:
    local-file workflows never touch the network, and a report that fails
    every day for an extra the owner does not use is a report nobody reads.
    """
    del db
    try:
        found = _ytdlp_version()
    except Exception as exc:
        return HealthResult(ok=False, detail=f"yt-dlp: {exc}")
    if found is None:
        return HealthResult(
            ok=False,
            detail="yt-dlp is not installed; downloading and channel sync will "
                   "not work, everything else will",
            remedy="pip install -e '.[yt-dlp]'",
        )
    return HealthResult(ok=True, detail=f"yt-dlp {found}")


def _free_bytes() -> int:
    return shutil.disk_usage(config.data_root()).free


def _headroom(db: Database, free: int) -> tuple[int, int]:
    """How many more videos fit, and what one is measured to cost.

    design §1 puts the corpus at ~1,600 videos of about an hour, so the
    number of bytes left is much less useful than the number of videos left.
    The per-video figure comes from what the catalogue has actually cost,
    falling back to an estimate only while there is nothing to measure.
    """
    row = db.conn.execute(
        "SELECT COUNT(DISTINCT video_id) AS videos, SUM(bytes) AS total FROM assets"
    ).fetchone()
    videos = int(row["videos"] or 0)
    total = int(row["total"] or 0)
    per_video = total // videos if videos and total else C.ESTIMATED_BYTES_PER_VIDEO
    per_video = max(per_video, 1)
    return free // per_video, per_video


def check_disk(db: Database) -> HealthResult:
    """Can the data volume be read at all? Required: nothing works if not."""
    del db
    try:
        free = _free_bytes()
    except Exception as exc:
        return HealthResult(
            ok=False,
            detail=f"cannot read free space on {config.data_root()}: {exc}",
            remedy=f"check that {config.data_root()} exists and is readable, or "
                   f"point RYTP_DATA somewhere that is",
        )
    # Deliberately ok even at one byte free: a full drive is a different
    # finding, and disk-headroom is the check that reports it.
    return HealthResult(
        ok=True,
        detail=f"{config.data_root()} readable, {free // C.BYTES_PER_GIB} GiB free",
    )


def check_disk_headroom(db: Database) -> HealthResult:
    """Is there room for more of the corpus? Advisory, not required."""
    try:
        free = _free_bytes()
    except Exception as exc:
        return HealthResult(ok=False, detail=f"cannot measure free space: {exc}")

    fits, per_video = _headroom(db, free)
    detail = (
        f"{free // C.BYTES_PER_GIB} GiB free; room for about {fits} more "
        f"video(s) at {per_video // (C.BYTES_PER_GIB // 1024)} MiB each"
    )
    if free < C.HEALTH_DISK_FREE_MIN_BYTES:
        return HealthResult(
            ok=False,
            detail=detail,
            remedy="free space, or reclaim some with `rytp cache prune --yes` "
                   "and `rytp assets remove <id> --yes` on superseded renditions",
        )
    return HealthResult(ok=True, detail=detail)


register_check(HealthCheck(
    name="ffmpeg",
    summary="ffmpeg is on PATH; every audio and video operation needs it.",
    run=lambda db: _binary_check("ffmpeg"),
))
register_check(HealthCheck(
    name="ffprobe",
    summary="ffprobe is on PATH; media is inspected with it.",
    run=lambda db: _binary_check("ffprobe"),
))
register_check(HealthCheck(
    name="yt-dlp",
    summary="yt-dlp is importable; needed only to download and sync channels.",
    run=check_ytdlp,
    required=False,
))
register_check(HealthCheck(
    name="disk",
    summary="The data volume is readable.",
    run=check_disk,
))
register_check(HealthCheck(
    name="disk-headroom",
    summary="Free space on the data volume, counted in videos that still fit.",
    run=check_disk_headroom,
    required=False,
))
