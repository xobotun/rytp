"""The worker process. design §5.

Three pools run at once — ``network`` (1 slot, and it stays 1), ``gpu`` (1)
and ``cpu`` (2-3) — so a download and a transcription overlap. Each slot is
a thread with its own :class:`~rytp.db.Database`; SQLite connections are not
shareable across threads and WAL makes several of them cheap.

Two rules that are easy to get wrong and expensive to get wrong:

* A throttle response freezes the **pool**, not the job. Deferring only the
  job that was rate-limited would let the loop claim the next download
  immediately and hit the same server again.
* ``attempts`` is spent at claim time (see :func:`rytp.jobs.queue.claim`),
  so a job that kills its worker still converges on ``failed`` instead of
  being retried forever after every restart.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire import policy as P
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.jobs import Readiness, resolve_job_kind
from rytp.jobs import queue as Q
from rytp.models import RytpError


class WorkerAlreadyRunning(RytpError):
    """Another worker holds a live lease on this database."""


@dataclass(frozen=True)
class WorkerOptions:
    """How this worker run behaves."""

    pools: tuple[str, ...] = C.POOLS
    once: bool = False
    max_jobs: int = 0                        # 0 means "no limit"
    poll_interval_s: float = C.WORKER_POLL_INTERVAL_S


@dataclass
class WorkerReport:
    """What one worker run did. Returned to the CLI for its summary line."""

    claimed: int = 0
    done: int = 0
    failed: int = 0
    deferred: int = 0
    blocked: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# The lease. A crashed worker must not strand jobs in ``running`` forever,
# and the jobs table has no column to record who owns a row, so ownership
# lives in settings instead.
# --------------------------------------------------------------------------


def _lease(db: Database) -> dict[str, object] | None:
    raw = get_setting(db, C.SETTING_WORKER_LEASE)
    if not raw:
        return None
    try:
        return dict(json.loads(raw))
    except (ValueError, TypeError):
        return None


def acquire_lease(db: Database, *, now: datetime | None = None) -> None:
    """Claim the right to run, taking over from a worker that died.

    Raises :class:`WorkerAlreadyRunning` when the current lease's heartbeat
    is younger than :data:`rytp.constants.WORKER_LEASE_STALE_S`.
    """
    stamp = now or utcnow()
    held = _lease(db)
    if held is not None:
        beat = str(held.get("heartbeat") or "")
        cutoff = (stamp - timedelta(seconds=C.WORKER_LEASE_STALE_S)).isoformat()
        if beat > cutoff:
            raise WorkerAlreadyRunning(
                f"a worker is already running (pid {held.get('pid')}, last seen "
                f"{beat}); stop it, or wait "
                f"{int(C.WORKER_LEASE_STALE_S)}s for its lease to go stale"
            )
    set_setting(
        db,
        C.SETTING_WORKER_LEASE,
        json.dumps({"pid": os.getpid(), "started_at": stamp.isoformat(),
                    "heartbeat": stamp.isoformat()}),
    )
    Q.reclaim_running(db, now=stamp)


def heartbeat(db: Database, *, now: datetime | None = None) -> None:
    """Keep the lease fresh so nobody else takes it while we are working."""
    held = _lease(db) or {"pid": os.getpid()}
    held["heartbeat"] = (now or utcnow()).isoformat()
    set_setting(db, C.SETTING_WORKER_LEASE, json.dumps(held))


def release_lease(db: Database) -> None:
    """Give the lease up on a clean exit."""
    set_setting(db, C.SETTING_WORKER_LEASE, "")


# --------------------------------------------------------------------------
# One iteration of one pool. Everything above this is bookkeeping; this is
# the part with the policy in it.
# --------------------------------------------------------------------------


def _may_claim(db: Database, pool: str, policy: P.DownloadPolicy, now: datetime) -> bool:
    """Gate the network pool on the cooldown and the daily cap."""
    if Q.is_paused(db):
        return False
    if pool != "network":
        return True
    if P.is_cooling_down(db, now=now):
        return False
    try:
        P.check_daily_cap(db, policy, now=now)
    except P.DailyCapReached:
        return False
    return True


def _handle_failure(
    db: Database,
    job: Q.Job,
    exc: Exception,
    *,
    policy: P.DownloadPolicy,
    report: WorkerReport,
    now: datetime,
) -> None:
    """Decide between freezing the pool, failing, and trying again later."""
    message = str(exc)
    throttled = isinstance(exc, P.RateLimited) or (
        P.classify_error(message) is P.ErrorKind.THROTTLED
    )
    permanent = isinstance(exc, P.PermanentAcquireError) or (
        P.classify_error(message) is P.ErrorKind.UNAVAILABLE
    )

    if throttled:
        until = P.record_throttle(db, policy, now=now)
        # The same timestamp goes on the job so `rytp jobs stats` reports a
        # throttled count instead of an unexplained pause.
        Q.defer(db, job.id, not_before=until, error=message, refund_attempt=True)
        with report.lock:
            report.deferred += 1
        return

    if permanent or job.attempts >= C.JOB_MAX_ATTEMPTS:
        Q.fail(db, job.id, error=message, now=now)
        with report.lock:
            report.failed += 1
        return

    retry_at = (now + timedelta(seconds=C.TRANSIENT_BACKOFF_S)).isoformat()
    Q.defer(db, job.id, not_before=retry_at, error=message)
    with report.lock:
        report.deferred += 1


def run_pool_once(
    db: Database,
    pool: str,
    *,
    policy: P.DownloadPolicy,
    rng: random.Random,
    report: WorkerReport,
    sleep,
    now_fn=utcnow,
) -> bool:
    """Claim and run at most one job. Returns False when there was nothing."""
    now = now_fn()
    if not _may_claim(db, pool, policy, now):
        return False

    job = Q.claim(db, pool, now=now)
    if job is None:
        return False
    with report.lock:
        report.claimed += 1

    spec = resolve_job_kind(job.kind)
    readiness = spec.readiness(db, job.target_id)
    if readiness is Readiness.SATISFIED:
        Q.finish(db, job.id, now=now)
        with report.lock:
            report.done += 1
        return True
    if readiness is Readiness.BLOCKED:
        Q.block(db, job.id, reason="prerequisites are not in place", now=now)
        with report.lock:
            report.blocked += 1
        return True

    try:
        note = spec.handler(db, job.target_id, job.payload)
    except Exception as exc:
        _handle_failure(db, job, exc, policy=policy, report=report, now=now_fn())
    else:
        if pool == "network":
            P.clear_throttle(db)
        # A note never changes the outcome: this job is done, not failed.
        # It is how a warning from a bulk run survives at all — on the worker
        # path there is nobody watching stdout.
        Q.finish(db, job.id, note=note, now=now_fn())
        # This is what closes the pipeline. `ingest` parks extract_wav as
        # blocked because there is no audio yet; finishing the download has
        # to be what turns it into work, or --once exits with no WAV and the
        # cpu thread polls an empty pending set forever. Scoped to the
        # target's namespace so a render's id can never collide with a video.
        Q.unblock(db, target_id=job.target_id, target_kind=spec.target_kind,
                  now=now_fn())
        with report.lock:
            report.done += 1

    # design §5: a randomised pause between videos, every time the network
    # pool actually reached out — successes and failures alike, because both
    # made a request.
    if pool == "network":
        delay = (
            P.next_delay_s(policy, rng)
            if job.kind == "download"
            else policy.file_delay_s
        )
        sleep(delay)
    return True


# --------------------------------------------------------------------------
# The process.
# --------------------------------------------------------------------------


def _budget_spent(options: WorkerOptions, report: WorkerReport) -> bool:
    if options.max_jobs <= 0:
        return False
    with report.lock:
        return report.claimed >= options.max_jobs


def _drain_inline(db: Database, options: WorkerOptions, report: WorkerReport,
                  *, sleep, rng, now_fn) -> None:
    """``--once``: walk every pool to exhaustion, repeatedly.

    Single-threaded on purpose — it is what the tests and a quick manual run
    want, and determinism is worth more here than overlap. The outer loop
    matters: finishing a download unblocks a cpu job, so one pass in pool
    order would only work by accident.
    """
    policy = P.DownloadPolicy.from_settings(db)
    progressed = True
    while progressed and not _budget_spent(options, report):
        progressed = False
        for pool in options.pools:
            while not _budget_spent(options, report):
                if not run_pool_once(db, pool, policy=policy, rng=rng,
                                     report=report, sleep=sleep, now_fn=now_fn):
                    break
                progressed = True


def _pool_loop(db_path: Path, pool: str, options: WorkerOptions,
               report: WorkerReport, stop: threading.Event, *, open_db) -> None:
    """One slot. Owns its own connection for its whole life."""
    db = open_db(db_path)
    rng = random.Random()
    try:
        while not stop.is_set() and not _budget_spent(options, report):
            policy = P.DownloadPolicy.from_settings(db)
            busy = run_pool_once(
                db, pool, policy=policy, rng=rng, report=report,
                sleep=lambda s: stop.wait(s), now_fn=utcnow,
            )
            if not busy:
                stop.wait(options.poll_interval_s)
    finally:
        db.close()


def run_worker(
    db: Database,
    options: WorkerOptions | None = None,
    *,
    sleep=time.sleep,
    rng: random.Random | None = None,
    now_fn=utcnow,
    open_db=Database,
) -> WorkerReport:
    """Run the worker. ``--once`` drains inline; otherwise, threads."""
    options = options or WorkerOptions()
    report = WorkerReport()
    rng = rng or random.Random()

    acquire_lease(db, now=now_fn())
    Q.reconcile(db, now=now_fn())
    try:
        if options.once:
            _drain_inline(db, options, report, sleep=sleep, rng=rng, now_fn=now_fn)
            return report

        db_path = config.paths().db
        stop = threading.Event()
        threads: list[threading.Thread] = []
        for pool in options.pools:
            for slot in range(P.pool_size(db, pool)):
                thread = threading.Thread(
                    target=_pool_loop,
                    args=(db_path, pool, options, report, stop),
                    kwargs={"open_db": open_db},
                    name=f"rytp-{pool}-{slot}",
                    daemon=True,
                )
                thread.start()
                threads.append(thread)
        try:
            # Windows has no SIGTERM, so Ctrl-C is the only stop signal. The
            # supervisor polls often and heartbeats on its own slower clock,
            # so shutdown is prompt even with a long heartbeat interval.
            last_beat = float("-inf")
            while any(t.is_alive() for t in threads):
                if time.monotonic() - last_beat >= C.WORKER_HEARTBEAT_INTERVAL_S:
                    heartbeat(db, now=now_fn())
                    last_beat = time.monotonic()
                stop.wait(options.poll_interval_s)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=C.WORKER_JOIN_TIMEOUT_S)
        return report
    finally:
        release_lease(db)
