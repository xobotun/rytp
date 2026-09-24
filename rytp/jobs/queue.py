"""The jobs table. design §5.

State lives entirely in SQLite, so killing the process loses nothing: a
worker that dies mid-job leaves a ``running`` row, and the next worker
reclaims it (:func:`reclaim_running`). Pause is a settings flag rather than
a job state, so pausing never rewrites thousands of rows and never has to be
undone correctly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.jobs import JOB_KINDS, Readiness, resolve_job_kind
from rytp.models import NotFoundError

#: States a full reconcile may reopen from. ``running`` is excluded because
#: a live worker owns it; ``cancelled`` because a human said no.
_RECONCILABLE: tuple[str, ...] = ("done", "blocked")


@dataclass(frozen=True)
class Job:
    """One row of the jobs table, with payload_json already decoded."""

    id: int
    kind: str
    target_id: int
    state: str
    pool: str
    priority: int
    attempts: int
    not_before: str | None
    last_error: str | None
    note: str | None = None
    #: Advisory, worker-written while the job runs (contracts §5). Distinct
    #: from ``note``: it is lossy, may be stale, and is cleared the moment
    #: the job leaves ``running``, whichever state it lands in.
    progress: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=int(row["id"]),
            kind=row["kind"],
            target_id=int(row["target_id"]),
            state=row["state"],
            pool=row["pool"],
            priority=int(row["priority"]),
            attempts=int(row["attempts"]),
            not_before=row["not_before"],
            last_error=row["last_error"],
            note=row["note"],
            progress=row["progress"],
            payload=json.loads(row["payload_json"] or "{}"),
        )


@dataclass(frozen=True)
class JobStats:
    """What ``rytp jobs stats`` shows. design §5: throttling must be visible."""

    by_state: dict[str, int]
    pending_by_pool: dict[str, int]
    throttled: int
    next_not_before: str | None
    paused: bool
    #: Jobs that succeeded but left a warning. A batch that quietly discarded
    #: speaker labels on forty videos has to be visible without reading every
    #: row, which is the whole reason notes exist.
    noted: int = 0


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat()


def _state_for(readiness: Readiness) -> str:
    return {
        Readiness.READY: "pending",
        Readiness.BLOCKED: "blocked",
        Readiness.SATISFIED: "done",
    }[readiness]


def enqueue(
    db: Database,
    kind: str,
    target_id: int,
    *,
    priority: int = 0,
    payload: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> int:
    """Create or refresh one job, and return its id.

    Idempotent by ``UNIQUE (kind, target_id)``. Re-enqueuing recomputes the
    state from the readiness predicate, so running ``rytp ingest`` twice on a
    video whose media was deleted does the right thing instead of nothing. A
    job somebody is currently running is left strictly alone.
    """
    spec = resolve_job_kind(kind)
    state = _state_for(spec.readiness(db, target_id))
    stamp = _now(now)
    with db.transaction():
        db.conn.execute(
            """
            INSERT INTO jobs (kind, target_id, state, pool, priority, attempts,
                              payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT (kind, target_id) DO UPDATE SET
                priority   = MAX(jobs.priority, excluded.priority),
                state      = CASE WHEN jobs.state = 'running'
                                  THEN jobs.state ELSE excluded.state END,
                not_before = CASE WHEN jobs.state = 'running'
                                  THEN jobs.not_before ELSE NULL END
            """,
            (kind, target_id, state, spec.pool, priority,
             json.dumps(payload or {}), stamp),
        )
        row = db.conn.execute(
            "SELECT id FROM jobs WHERE kind = ? AND target_id = ?", (kind, target_id)
        ).fetchone()
    return int(row["id"])


def claim(db: Database, pool: str, *, now: datetime | None = None) -> Job | None:
    """Atomically take the next runnable job of one pool.

    The guarded ``UPDATE`` is what makes this safe across threads and
    processes: a worker that lost the race updates zero rows and tries the
    next candidate. ``attempts`` is spent here, at claim time, so a job that
    kills its worker outright still counts against
    :data:`rytp.constants.JOB_MAX_ATTEMPTS`.
    """
    stamp = _now(now)
    for _ in range(C.CLAIM_RETRY_LIMIT):
        candidate = db.conn.execute(
            """
            SELECT id FROM jobs
            WHERE state = 'pending' AND pool = ?
              AND (not_before IS NULL OR not_before <= ?)
            ORDER BY priority DESC, id ASC
            LIMIT 1
            """,
            (pool, stamp),
        ).fetchone()
        if candidate is None:
            return None
        cur = db.conn.execute(
            """
            UPDATE jobs
            SET state = 'running', started_at = ?, finished_at = NULL,
                attempts = attempts + 1, last_error = NULL, progress = NULL
            WHERE id = ? AND state = 'pending'
            """,
            (stamp, candidate["id"]),
        )
        if cur.rowcount == 1:
            row = db.conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (candidate["id"],)
            ).fetchone()
            return Job.from_row(row)
    return None


def finish(
    db: Database,
    job_id: int,
    *,
    note: str | None = None,
    now: datetime | None = None,
) -> None:
    """Mark a job done, recording the handler's note if it left one.

    ``note`` is written unconditionally, so a job that succeeds cleanly after
    a run that left a warning clears it: the column reflects the last run,
    not a history.
    """
    db.conn.execute(
        "UPDATE jobs SET state = 'done', finished_at = ?, last_error = NULL, "
        "note = ?, progress = NULL WHERE id = ?",
        (_now(now), (note or None) and note[: C.JOB_NOTE_MAX_CHARS], job_id),
    )


def block(db: Database, job_id: int, *, reason: str, now: datetime | None = None) -> None:
    """Park a job whose prerequisites are not there. Reconcile reopens it."""
    db.conn.execute(
        "UPDATE jobs SET state = 'blocked', finished_at = ?, last_error = ?, "
        "progress = NULL WHERE id = ?",
        (_now(now), reason, job_id),
    )


def defer(
    db: Database,
    job_id: int,
    *,
    not_before: str,
    error: str | None = None,
    refund_attempt: bool = False,
) -> None:
    """Put a job back in the queue, invisible until ``not_before``.

    ``refund_attempt`` is for throttling: being rate-limited says nothing
    about this job, so it must not push it towards ``failed``.
    """
    # ``started_at`` is deliberately left alone: the attempt reached the
    # network, so it must keep counting against the daily cap.
    db.conn.execute(
        "UPDATE jobs SET state = 'pending', not_before = ?, last_error = ?, "
        "attempts = MAX(attempts - ?, 0), progress = NULL WHERE id = ?",
        (not_before, error, 1 if refund_attempt else 0, job_id),
    )


def fail(db: Database, job_id: int, *, error: str, now: datetime | None = None) -> None:
    """Terminal failure. Only ``rytp jobs retry`` brings it back."""
    db.conn.execute(
        "UPDATE jobs SET state = 'failed', finished_at = ?, last_error = ?, "
        "progress = NULL WHERE id = ?",
        (_now(now), error[: C.JOB_ERROR_MAX_CHARS], job_id),
    )


def set_progress(db: Database, job_id: int, text: str | None) -> None:
    """Write the worker's advisory ``jobs.progress`` line (contracts §5).

    Deliberately a bare autocommitting statement, never wrapped in
    :meth:`Database.transaction`: callers must invoke this outside a
    handler's own transaction, or the write joins it and is invisible until
    that transaction commits — which for a multi-minute handler defeats the
    entire point (plan Task 8a). Not gated on ``state = 'running'``: the
    worker is the only caller and it already knows the job is running.
    """
    db.conn.execute(
        "UPDATE jobs SET progress = ? WHERE id = ?",
        ((text or None) and text[: C.JOB_NOTE_MAX_CHARS], job_id),
    )


def reclaim_running(db: Database, *, now: datetime | None = None) -> int:
    """Return every ``running`` job to ``pending``. Called at worker startup.

    A killed worker leaves rows claiming to be running forever; this is what
    stops them stranding. The spent attempt is deliberately not refunded.
    """
    cur = db.conn.execute(
        "UPDATE jobs SET state = 'pending', started_at = NULL, progress = NULL, "
        "last_error = 'reclaimed from a worker that did not finish' "
        "WHERE state = 'running'"
    )
    del now
    return int(cur.rowcount)


def _reevaluate(
    db: Database,
    states: Sequence[str],
    *,
    kinds: Sequence[str] | None,
    target_id: int | None,
    target_kind: str | None,
    now: datetime | None,
) -> int:
    """Re-ask the readiness predicates about settled jobs. design §5.

    ``UNIQUE (kind, target_id)`` means a ``done`` job cannot simply be
    enqueued again, so this is the path by which the world changing — a
    pruned WAV, a deleted rendition, a prerequisite that finally arrived —
    turns back into work. Returns how many jobs changed state.
    """
    wanted = tuple(kinds) if kinds else tuple(JOB_KINDS)
    if target_kind is not None:
        # ``target_id`` means different things to different kinds — a video
        # for everything Part 2 owns, a ``renders`` row for ``render`` — so
        # re-evaluating "everything with this target_id" must never cross
        # the namespace boundary.
        wanted = tuple(k for k in wanted if JOB_KINDS[k].target_kind == target_kind)
    if "done" in states:
        wanted = tuple(k for k in wanted if JOB_KINDS[k].reopenable)
    if not wanted:
        return 0
    kind_marks = ", ".join("?" for _ in wanted)
    state_marks = ", ".join("?" for _ in states)
    sql = (
        f"SELECT id, kind, target_id, state FROM jobs "
        f"WHERE state IN ({state_marks}) AND kind IN ({kind_marks})"
    )
    params: list[Any] = [*states, *wanted]
    if target_id is not None:
        sql += " AND target_id = ?"
        params.append(target_id)

    changed = 0
    for row in db.conn.execute(sql, params).fetchall():
        spec = JOB_KINDS.get(row["kind"])
        if spec is None:
            continue
        new_state = _state_for(spec.readiness(db, int(row["target_id"])))
        if new_state != row["state"]:
            db.conn.execute(
                "UPDATE jobs SET state = ?, not_before = NULL, finished_at = ? "
                "WHERE id = ? AND state = ?",
                (new_state, None if new_state == "pending" else _now(now),
                 row["id"], row["state"]),
            )
            changed += 1
    return changed


def reconcile(
    db: Database,
    *,
    kinds: Sequence[str] | None = None,
    target_id: int | None = None,
    target_kind: str | None = None,
    now: datetime | None = None,
) -> int:
    """Full sweep of settled jobs: ``done`` and ``blocked`` alike.

    Called where the world may have changed behind the queue's back — worker
    startup, ``cache.prune``, ``jobs.retry``. Not called after every finished
    job: a predicate that still answers READY right after a successful run
    would be flipped straight back to ``pending`` and loop. Kinds marked
    ``reopenable=False`` are never reopened from ``done``, so re-timing
    (``align``) or a diarization run (``diarize``) the user asked for once
    does not silently re-run on the next worker start.
    """
    return _reevaluate(db, _RECONCILABLE, kinds=kinds, target_id=target_id,
                       target_kind=target_kind, now=now)


def unblock(
    db: Database,
    *,
    target_id: int | None = None,
    target_kind: str | None = None,
    now: datetime | None = None,
) -> int:
    """Re-check only the ``blocked`` jobs. Safe to call constantly.

    This is what closes the pipeline: ``ingest`` parks ``extract_wav`` as
    blocked because there is no audio yet, and the moment the download job
    finishes this turns it into work. Scoped to one target by default,
    because a full sweep stats every asset of every settled job and the
    corpus is ~1,600 videos.
    """
    return _reevaluate(db, ("blocked",), kinds=None, target_id=target_id,
                       target_kind=target_kind, now=now)


def retry(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str = "failed",
    now: datetime | None = None,
) -> int:
    """Send settled jobs back to ``pending`` with a clean slate."""
    del now
    where = ["state = ?"]
    params: list[Any] = [state]
    if job_id is not None:
        where.append("id = ?")
        params.append(job_id)
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    cur = db.conn.execute(
        "UPDATE jobs SET state = 'pending', attempts = 0, not_before = NULL, "
        "last_error = NULL, note = NULL, progress = NULL, started_at = NULL, "
        f"finished_at = NULL WHERE {' AND '.join(where)}",
        params,
    )
    return int(cur.rowcount)


def get_job(db: Database, job_id: int) -> Job:
    """One job by id, for the command layer's result rows."""
    row = db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"job {job_id} does not exist")
    return Job.from_row(row)


def list_jobs(
    db: Database,
    *,
    state: str | None = None,
    pool: str | None = None,
    kind: str | None = None,
    limit: int = C.JOB_LIST_LIMIT,
) -> list[Job]:
    """Jobs, newest priority first, for the CLI and the TUI."""
    where: list[str] = []
    params: list[Any] = []
    for column, value in (("state", state), ("pool", pool), ("kind", kind)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = db.conn.execute(
        f"SELECT * FROM jobs {clause} ORDER BY priority DESC, id ASC LIMIT ?",
        params,
    ).fetchall()
    return [Job.from_row(r) for r in rows]


def stats(db: Database, *, now: datetime | None = None) -> JobStats:
    """Counts by state and pool, plus how much of the queue is throttled."""
    stamp = _now(now)
    by_state = {
        row["state"]: int(row["n"])
        for row in db.conn.execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
        ).fetchall()
    }
    pending_by_pool = {
        row["pool"]: int(row["n"])
        for row in db.conn.execute(
            "SELECT pool, COUNT(*) AS n FROM jobs WHERE state = 'pending' GROUP BY pool"
        ).fetchall()
    }
    throttled_row = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(not_before) AS next FROM jobs "
        "WHERE state = 'pending' AND not_before IS NOT NULL AND not_before > ?",
        (stamp,),
    ).fetchone()
    noted_row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE note IS NOT NULL AND note != ''"
    ).fetchone()
    return JobStats(
        by_state={s: by_state.get(s, 0) for s in
                  ("pending", "running", "done", "failed", "blocked", "cancelled")},
        pending_by_pool={p: pending_by_pool.get(p, 0) for p in C.POOLS},
        throttled=int(throttled_row["n"]),
        next_not_before=throttled_row["next"],
        paused=is_paused(db),
        noted=int(noted_row["n"]),
    )


def pause(db: Database) -> None:
    """Stop every pool from claiming. Instant, lock-free, loses nothing."""
    set_setting(db, C.SETTING_QUEUE_PAUSED, "1")


def resume(db: Database) -> None:
    set_setting(db, C.SETTING_QUEUE_PAUSED, "0")


def is_paused(db: Database) -> bool:
    return get_setting(db, C.SETTING_QUEUE_PAUSED) == "1"


def _filter_clause(
    *, job_id: int | None, kind: str | None, state: str | None, target_id: int | None
) -> tuple[str, list[Any]]:
    """Shared WHERE for the filter-based job commands."""
    where: list[str] = []
    params: list[Any] = []
    for column, value in (("id", job_id), ("kind", kind),
                          ("state", state), ("target_id", target_id)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    return (" AND ".join(where) if where else ""), params


def running_matches(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str | None = None,
    target_id: int | None = None,
) -> list[Job]:
    """The jobs a cancel would hit that are currently running."""
    clause, params = _filter_clause(job_id=job_id, kind=kind, state=state,
                                    target_id=target_id)
    sql = "SELECT * FROM jobs WHERE state = 'running'"
    if clause:
        sql += f" AND {clause}"
    return [Job.from_row(r) for r in db.conn.execute(sql, params).fetchall()]


def cancellable(
    db: Database,
    *,
    job_id: int | None = None,
    kind: str | None = None,
    state: str | None = None,
    target_id: int | None = None,
) -> list[Job]:
    """The jobs a cancel would actually change: everything but running."""
    clause, params = _filter_clause(job_id=job_id, kind=kind, state=state,
                                    target_id=target_id)
    sql = "SELECT * FROM jobs WHERE state NOT IN ('running', 'cancelled')"
    if clause:
        sql += f" AND {clause}"
    sql += " ORDER BY id"
    return [Job.from_row(r) for r in db.conn.execute(sql, params).fetchall()]


def cancel(db: Database, jobs: Sequence[Job], *, now: datetime | None = None) -> int:
    """Drop jobs out of the queue. Never touches a running job.

    ``cancelled`` is terminal as far as the queue is concerned — ``reconcile``
    only reopens ``done`` and ``blocked`` — but it is not a permanent veto:
    ``rytp jobs retry --state cancelled`` and re-running ``rytp ingest`` both
    bring a job back, deliberately.
    """
    if not jobs:
        return 0
    stamp = _now(now)
    with db.transaction():
        db.conn.executemany(
            "UPDATE jobs SET state = 'cancelled', finished_at = ?, "
            "last_error = NULL, note = NULL WHERE id = ? AND state != 'running'",
            [(stamp, j.id) for j in jobs],
        )
    return len(jobs)
