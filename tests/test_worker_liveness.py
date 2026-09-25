"""BUGS.md entry 41: a queue nobody is draining must say so.

Enqueueing writes a row; a separate `rytp worker` process does the work.
Nothing said so, so the owner twice queued work, saw nothing happen, and
had to be told. The lease already recorded whether a worker was alive.
"""

from __future__ import annotations

from datetime import timedelta

from rytp import constants as C
from rytp.commands import resolve
from rytp.db import Database
from rytp.jobs import queue as Q
from rytp.jobs import worker as W
from tests.fakes import make_video


def test_a_fresh_lease_reads_as_live(db: Database) -> None:
    assert W.live_lease(db) is None
    W.acquire_lease(db)
    held = W.live_lease(db)
    assert held is not None and "pid" in held


def test_a_stale_lease_reads_as_dead(db: Database) -> None:
    """A worker that died leaves its lease behind; it must not count."""
    W.acquire_lease(db)
    later = W.utcnow() + timedelta(seconds=C.WORKER_LEASE_STALE_S + 1)
    assert W.live_lease(db, now=later) is None


def _pending_job(db: Database) -> int:
    """A row in `pending`, written directly.

    `Q.enqueue` derives the state from readiness, and every real kind is
    BLOCKED on a bare video — correctly, since a blocked job is waiting on
    prerequisites rather than on a worker.
    """
    video_id = make_video(db)
    cur = db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
        "VALUES ('extract_wav', ?, 'pending', 'cpu', '{}', ?)",
        (video_id, W.utcnow().isoformat()),
    )
    db.conn.commit()
    return int(cur.lastrowid or 0)


def test_jobs_list_says_how_to_start_a_worker_when_none_runs(db: Database) -> None:
    _pending_job(db)
    result = resolve("jobs.list").handler(db)
    assert result.message is not None
    assert "no worker is running" in result.message
    assert "rytp worker" in result.message


def test_jobs_list_stays_quiet_while_a_worker_holds_the_lease(db: Database) -> None:
    _pending_job(db)
    W.acquire_lease(db)
    assert resolve("jobs.list").handler(db).message is None


def test_no_warning_when_nothing_is_pending(db: Database) -> None:
    """A settled queue needs no worker, so saying one is absent is noise."""
    job_id = _pending_job(db)
    Q.finish(db, job_id)
    assert resolve("jobs.list").handler(db).message is None


def test_a_blocked_job_is_not_waiting_on_a_worker(db: Database) -> None:
    """Blocked means prerequisites are missing, not that nothing is draining."""
    Q.enqueue(db, "extract_wav", make_video(db))
    rows = Q.list_jobs(db)
    assert rows and rows[0].state == "blocked"
    assert resolve("jobs.list").handler(db).message is None
