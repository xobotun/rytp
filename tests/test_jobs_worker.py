"""Tests for the worker (design §5). No network, no threads unless asked."""

from __future__ import annotations

import io
import os
import random
import threading
from datetime import UTC, datetime, timedelta

import pytest

from rytp import constants as C
from rytp import progress as PR
from rytp.acquire.policy import (
    DownloadPolicy,
    RateLimited,
    VideoUnavailable,
    cooldown_until,
    is_cooling_down,
)
from rytp.db import Database
from rytp.db.queries import insert_asset, set_setting
from rytp.jobs import Readiness
from rytp.jobs import queue as Q
from rytp.jobs import worker as W
from tests.fakes import make_video, temp_job_kind, touch


class _FakeTty(io.StringIO):
    """Claims to be a terminal, like `tests/test_progress.py`'s fixture."""

    def isatty(self) -> bool:  # type: ignore[override]
        return True


class _FakeNonTty(io.StringIO):
    """A redirected-to-a-file stream: never a terminal."""

    def isatty(self) -> bool:  # type: ignore[override]
        return False

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def _fixed_now():
    return NOW


def _tick(db: Database, pool: str, *, report=None, sleeps=None, now=NOW):
    """Run exactly one pool iteration with everything deterministic."""
    report = report or W.WorkerReport()
    sleeps = sleeps if sleeps is not None else []
    did = W.run_pool_once(
        db,
        pool,
        policy=DownloadPolicy.from_settings(db),
        rng=random.Random(7),
        report=report,
        sleep=sleeps.append,
        now_fn=lambda: now,
    )
    return did, report, sleeps


def test_a_handlers_note_lands_on_the_job_row(db: Database) -> None:
    # Part 3's "re-transcribing discarded this video's speaker mapping" only
    # reaches a bulk operator this way; on the worker path nobody reads stdout.
    with temp_job_kind("t_note", "cpu", lambda db_, t, p: "labels discarded"):
        vid = make_video(db)
        Q.enqueue(db, "t_note", vid, now=NOW)
        did, report, _ = _tick(db, "cpu")
    assert did is True and report.done == 1
    job = Q.list_jobs(db)[0]
    assert job.state == "done"          # a note is not a failure
    assert job.note == "labels discarded"


def test_a_routine_success_leaves_no_note(db: Database) -> None:
    with temp_job_kind("t_quiet", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_quiet", make_video(db), now=NOW)
        _tick(db, "cpu")
    assert Q.list_jobs(db)[0].note is None


def test_a_successful_job_is_marked_done(db: Database) -> None:
    calls: list[int] = []

    def handler(db_, t, p):
        calls.append(t)

    with temp_job_kind("t_ok", "cpu", handler):
        vid = make_video(db)
        Q.enqueue(db, "t_ok", vid, now=NOW)
        did, report, _ = _tick(db, "cpu")
    assert did is True
    assert calls == [vid]
    assert report.done == 1
    assert Q.list_jobs(db)[0].state == "done"


def test_an_idle_pool_reports_nothing_to_do(db: Database) -> None:
    did, report, _ = _tick(db, "cpu")
    assert did is False and report.claimed == 0


def test_a_paused_queue_claims_nothing(db: Database) -> None:
    with temp_job_kind("t_ok", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_ok", make_video(db), now=NOW)
        Q.pause(db)
        did, report, _ = _tick(db, "cpu")
    assert did is False and report.claimed == 0
    assert Q.list_jobs(db)[0].state == "pending"


def test_a_job_satisfied_since_enqueue_skips_its_handler(db: Database) -> None:
    calls: list[int] = []

    def handler(db_, t, p):
        calls.append(t)

    with temp_job_kind(
        "t_sat", "cpu", handler, readiness=lambda db_, t: Readiness.SATISFIED
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_sat', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        did, report, _ = _tick(db, "cpu")
    assert did is True and calls == []
    assert report.done == 1 and Q.list_jobs(db)[0].state == "done"
    # BUGS.md entry 40: skipping must not look like working. Without the
    # note this row is a blank-noted `done`, exactly what a successful run
    # leaves — which is how "I queued a transcribe and nothing happened"
    # presents when the video already had words.
    assert Q.list_jobs(db)[0].note == C.JOB_ALREADY_SATISFIED_NOTE


def test_a_job_blocked_since_enqueue_is_parked(db: Database) -> None:
    with temp_job_kind(
        "t_blk", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED,
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_blk', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        did, report, _ = _tick(db, "cpu")
    assert did is True and report.blocked == 1
    assert Q.list_jobs(db)[0].state == "blocked"


def test_a_throttle_freezes_the_whole_network_pool(db: Database) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429: Too Many Requests")

    with temp_job_kind("t_429", "network", boom):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        Q.enqueue(db, "t_429", a, now=NOW)
        Q.enqueue(db, "t_429", b, now=NOW)

        did, report, _ = _tick(db, "network")
        assert did is True and report.deferred == 1
        assert is_cooling_down(db, now=NOW) is True

        # The second job must NOT be attempted — that is the whole point.
        did_again, report2, _ = _tick(db, "network")
    assert did_again is False and report2.claimed == 0


def test_the_throttled_job_keeps_its_attempt_and_shows_the_cooldown(
    db: Database,
) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429")

    with temp_job_kind("t_429", "network", boom):
        vid = make_video(db)
        Q.enqueue(db, "t_429", vid, now=NOW)
        _tick(db, "network")
    job = Q.list_jobs(db)[0]
    assert job.state == "pending"
    assert job.attempts == 0                      # claim spent it, throttle refunded it
    assert job.not_before == cooldown_until(db)   # visible in `jobs stats`
    assert Q.stats(db, now=NOW).throttled == 1


def test_the_cooldown_ladder_climbs_across_ticks(db: Database) -> None:
    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429")

    with temp_job_kind("t_429", "network", boom):
        vid = make_video(db)
        Q.enqueue(db, "t_429", vid, now=NOW)
        _tick(db, "network", now=NOW)
        first = cooldown_until(db)
        later = NOW + timedelta(seconds=C.THROTTLE_BACKOFF_LADDER_S[0] + 1)
        _tick(db, "network", now=later)
        second = cooldown_until(db)
    assert first == (NOW + timedelta(seconds=300)).isoformat()
    assert second == (later + timedelta(seconds=900)).isoformat()


def test_a_clean_download_resets_the_ladder(db: Database) -> None:
    with temp_job_kind("t_dl", "network", lambda db_, t, p: None):
        set_setting(db, C.SETTING_THROTTLE_STREAK, "3")
        Q.enqueue(db, "t_dl", make_video(db), now=NOW)
        _tick(db, "network")
    assert is_cooling_down(db, now=NOW) is False
    assert cooldown_until(db) is None


def test_a_dead_video_fails_permanently_without_freezing_the_pool(
    db: Database,
) -> None:
    def gone(db_, t, p):
        raise VideoUnavailable("Private video")

    with temp_job_kind("t_gone", "network", gone):
        vid = make_video(db)
        Q.enqueue(db, "t_gone", vid, now=NOW)
        did, report, _ = _tick(db, "network")
    assert did is True and report.failed == 1
    assert Q.list_jobs(db)[0].state == "failed"
    assert is_cooling_down(db, now=NOW) is False


def test_a_transient_error_is_retried_until_the_attempt_budget_runs_out(
    db: Database,
) -> None:
    def flaky(db_, t, p):
        raise RuntimeError("unable to download video data: timed out")

    with temp_job_kind("t_flaky", "cpu", flaky):
        vid = make_video(db)
        Q.enqueue(db, "t_flaky", vid, now=NOW)
        for i in range(C.JOB_MAX_ATTEMPTS):
            at = NOW + timedelta(seconds=C.TRANSIENT_BACKOFF_S * (i + 1))
            _tick(db, "cpu", now=at)
    job = Q.list_jobs(db)[0]
    assert job.state == "failed"
    assert job.attempts == C.JOB_MAX_ATTEMPTS
    assert "timed out" in (job.last_error or "")


def test_a_download_is_followed_by_the_randomised_delay(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The delay rule keys off the kind name "download", so this test swaps
    # that kind's handler rather than registering a differently named one.
    from rytp.jobs import JOB_KINDS, JobKind

    real = JOB_KINDS["download"]
    monkeypatch.setitem(
        JOB_KINDS,
        "download",
        JobKind(name="download", pool="network", readiness=real.readiness,
                handler=lambda db_, t, p: None, summary=real.summary),
    )
    Q.enqueue(db, "download", make_video(db), now=NOW)
    _, _, sleeps = _tick(db, "network")
    assert len(sleeps) == 1
    assert C.DOWNLOAD_DELAY_MIN_S <= sleeps[0] <= C.DOWNLOAD_DELAY_MAX_S


def test_a_captions_fetch_uses_the_shorter_delay(db: Database) -> None:
    with temp_job_kind("t_caps", "network", lambda db_, t, p: None):
        Q.enqueue(db, "t_caps", make_video(db), now=NOW)
        _, _, sleeps = _tick(db, "network")
    assert sleeps == [C.DOWNLOAD_FILE_DELAY_S]


def test_a_cpu_job_is_not_followed_by_a_delay(db: Database, tmp_path) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_cpu", make_video(db), now=NOW)
        _, _, sleeps = _tick(db, "cpu")
    assert sleeps == []


def test_the_daily_cap_stops_the_network_pool_claiming(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "1")
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at, "
        "started_at) VALUES ('download', 999, 'done', 'network', '{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    with temp_job_kind("t_dl", "network", lambda db_, t, p: None):
        Q.enqueue(db, "t_dl", make_video(db), now=NOW)
        did, report, _ = _tick(db, "network")
    assert did is False and report.claimed == 0
    assert Q.list_jobs(db, kind="t_dl")[0].state == "pending"


def test_the_lease_refuses_a_second_live_worker(db: Database) -> None:
    W.acquire_lease(db, now=NOW)
    with pytest.raises(W.WorkerAlreadyRunning, match="already running"):
        W.acquire_lease(db, now=NOW + timedelta(seconds=1))


def test_a_stale_lease_is_taken_over_and_orphans_reclaimed(db: Database) -> None:
    with temp_job_kind("t_ok", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_ok", make_video(db), now=NOW)
        Q.claim(db, "cpu", now=NOW)          # a worker claims it, then dies
        W.acquire_lease(db, now=NOW)
        stale = NOW + timedelta(seconds=C.WORKER_LEASE_STALE_S + 1)
        W.acquire_lease(db, now=stale)       # the next worker takes over
        assert Q.reclaim_running(db, now=stale) == 0  # startup already did it
    job = Q.list_jobs(db)[0]
    assert job.state == "pending" and job.attempts == 1


def test_releasing_the_lease_lets_the_next_worker_in(db: Database) -> None:
    W.acquire_lease(db, now=NOW)
    W.release_lease(db)
    W.acquire_lease(db, now=NOW)  # does not raise


def test_run_worker_once_drains_every_pool_and_reconciles(
    db: Database, tmp_path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    with temp_job_kind("t_net", "network", lambda db_, t, p: None), temp_job_kind(
        "t_cpu", "cpu", lambda db_, t, p: None
    ):
        Q.enqueue(db, "t_net", vid, now=NOW)
        Q.enqueue(db, "t_cpu", vid, now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(once=True),
            sleep=lambda s: None,
            rng=random.Random(7),
            now_fn=_fixed_now,
        )
    assert report.done == 2
    assert {j.state for j in Q.list_jobs(db)} == {"done"}


def test_run_worker_once_honours_max_jobs(db: Database) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        Q.enqueue(db, "t_cpu", a, now=NOW)
        Q.enqueue(db, "t_cpu", b, now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(pools=("cpu",), once=True, max_jobs=1),
            sleep=lambda s: None,
            rng=random.Random(7),
            now_fn=_fixed_now,
        )
    assert report.done == 1
    assert sorted(j.state for j in Q.list_jobs(db)) == ["done", "pending"]


def test_the_threaded_worker_starts_works_and_stops(db: Database) -> None:
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_cpu", make_video(db), now=NOW)
        report = W.run_worker(
            db,
            W.WorkerOptions(pools=("cpu",), max_jobs=1, poll_interval_s=0.01),
        )
    assert report.done == 1
    assert Q.list_jobs(db)[0].state == "done"
    assert [t for t in threading.enumerate() if t.name.startswith("rytp-")] == []


# ---------------------------------------------------------------------------
# The progress seam (plan §1c, Task 8a; BUGS.md entries 3, 10, 22).
# ---------------------------------------------------------------------------


def test_a_handlers_progress_report_lands_on_the_running_job(db: Database) -> None:
    seen: list[str | None] = []

    def handler(db_, t, p):
        from rytp import progress

        progress.report("stage", done=1, total=2, detail="x")
        row = db_.conn.execute(
            "SELECT progress FROM jobs WHERE target_id = ?", (t,)
        ).fetchone()
        seen.append(row["progress"])

    with temp_job_kind("t_prog", "cpu", handler):
        Q.enqueue(db, "t_prog", make_video(db), now=NOW)
        _tick(db, "cpu")
    assert seen == ["stage 1/2 x"]


def test_progress_is_cleared_once_the_job_leaves_running(db: Database) -> None:
    def handler(db_, t, p):
        from rytp import progress

        progress.report("stage", done=1, total=2)

    with temp_job_kind("t_prog_done", "cpu", handler):
        Q.enqueue(db, "t_prog_done", make_video(db), now=NOW)
        _tick(db, "cpu")
    job = Q.list_jobs(db)[0]
    assert job.state == "done"
    assert job.progress is None


def test_a_failed_jobs_progress_is_also_cleared(db: Database) -> None:
    def gone(db_, t, p):
        from rytp import progress

        progress.report("stage", done=1, total=2)
        raise VideoUnavailable("Private video")

    with temp_job_kind("t_prog_fail", "network", gone):
        Q.enqueue(db, "t_prog_fail", make_video(db), now=NOW)
        _tick(db, "network")
    job = Q.list_jobs(db)[0]
    assert job.state == "failed"
    assert job.progress is None


def test_the_worker_throttles_progress_writes(db: Database) -> None:
    seen: list[str | None] = []

    def handler(db_, t, p):
        from rytp import progress

        progress.report("stage", done=1, total=10)
        row = db_.conn.execute(
            "SELECT progress FROM jobs WHERE target_id = ?", (t,)
        ).fetchone()
        seen.append(row["progress"])
        # Fired immediately after: the throttle (2000 ms) has not elapsed and
        # this is not the final call, so it must not overwrite the first.
        progress.report("stage", done=2, total=10)
        row = db_.conn.execute(
            "SELECT progress FROM jobs WHERE target_id = ?", (t,)
        ).fetchone()
        seen.append(row["progress"])

    with temp_job_kind("t_prog_throttle", "cpu", handler):
        Q.enqueue(db, "t_prog_throttle", make_video(db), now=NOW)
        _tick(db, "cpu")
    assert seen == ["stage 1/10", "stage 1/10"]


def test_a_final_progress_call_bypasses_the_throttle(db: Database) -> None:
    seen: list[str | None] = []

    def handler(db_, t, p):
        from rytp import progress

        progress.report("stage", done=1, total=2)
        progress.report("stage", done=2, total=2)
        row = db_.conn.execute(
            "SELECT progress FROM jobs WHERE target_id = ?", (t,)
        ).fetchone()
        seen.append(row["progress"])

    with temp_job_kind("t_prog_final", "cpu", handler):
        Q.enqueue(db, "t_prog_final", make_video(db), now=NOW)
        _tick(db, "cpu")
    assert seen == ["stage 2/2"]


def test_a_handler_that_reports_nothing_leaves_progress_untouched(db: Database) -> None:
    with temp_job_kind("t_prog_quiet", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_prog_quiet", make_video(db), now=NOW)
        _tick(db, "cpu")
    assert Q.list_jobs(db)[0].progress is None


# ---------------------------------------------------------------------------
# The worker's own output (BUGS.md entries 3, 10, 22, 41): `rytp worker`
# used to claim jobs, run them and exit in total silence, because
# installing the database progress sink replaced the terminal's sink for
# the whole handler call instead of joining it.
# ---------------------------------------------------------------------------


def test_a_claimed_job_is_announced(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)
    with temp_job_kind("t_ok", "cpu", lambda db_, t, p: None):
        vid = make_video(db)
        Q.enqueue(db, "t_ok", vid, now=NOW)
        _tick(db, "cpu")
    lines = fake.getvalue()
    assert f"job 1 claimed: kind=t_ok target={vid} pool=cpu" in lines


def test_a_successful_job_settles_with_its_note_and_no_control_characters(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)
    with temp_job_kind("t_note", "cpu", lambda db_, t, p: "labels discarded"):
        Q.enqueue(db, "t_note", make_video(db), now=NOW)
        _tick(db, "cpu")
    lines = fake.getvalue()
    assert "job 1 done in" in lines
    assert "note=labels discarded" in lines
    # Redirected to a file (the owner's `cmd /c "... > run.log 2>&1"`):
    # plain lines only, never a repaint escape.
    assert "\x1b" not in lines
    assert "\r" not in lines


def test_a_job_already_satisfied_is_announced_settled_not_silently_dropped(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)
    with temp_job_kind(
        "t_sat", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.SATISFIED,
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_sat', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        _tick(db, "cpu")
    lines = fake.getvalue()
    assert "job 1 claimed" in lines
    assert "job 1 done in" in lines
    assert f"note={C.JOB_ALREADY_SATISFIED_NOTE}" in lines


def test_a_blocked_job_is_announced_settled(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)
    with temp_job_kind(
        "t_blk", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED,
    ):
        vid = make_video(db)
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, created_at) "
            "VALUES ('t_blk', ?, 'pending', 'cpu', '{}', ?)",
            (vid, NOW.isoformat()),
        )
        _tick(db, "cpu")
    lines = fake.getvalue()
    assert "job 1 blocked in" in lines


def test_a_failed_job_settles_with_its_error(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)

    def gone(db_, t, p):
        raise VideoUnavailable("Private video")

    with temp_job_kind("t_gone", "network", gone):
        Q.enqueue(db, "t_gone", make_video(db), now=NOW)
        _tick(db, "network")
    lines = fake.getvalue()
    assert "job 1 failed in" in lines
    assert "error=Private video" in lines


def test_a_throttled_job_settles_as_deferred(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)

    def boom(db_, t, p):
        raise RateLimited("HTTP Error 429")

    with temp_job_kind("t_429", "network", boom):
        Q.enqueue(db, "t_429", make_video(db), now=NOW)
        _tick(db, "network")
    lines = fake.getvalue()
    assert "job 1 deferred in" in lines


def test_the_worker_still_writes_the_db_progress_column_off_tty(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(PR.sys, "stderr", fake)

    def handler(db_, t, p):
        PR.report("stage", done=1, total=2)

    with temp_job_kind("t_prog", "cpu", handler):
        Q.enqueue(db, "t_prog", make_video(db), now=NOW)
        _tick(db, "cpu")
    # The animation itself must not reach a non-tty stream (BUGS.md entry
    # 22's precedent, still honoured), but the lifecycle lines do, and the
    # db write (already covered above) is unaffected either way.
    assert "stage 1/2" not in fake.getvalue()
    assert "job 1 claimed" in fake.getvalue()
    assert "job 1 done" in fake.getvalue()


def test_a_handlers_progress_reaches_the_terminal_and_the_db_together(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bug: installing the db sink used to *replace* the terminal's
    # sink for the whole handler call, so a foreground `rytp worker` never
    # printed a chunk's progress even when stderr was a live terminal.
    fake = _FakeTty()
    monkeypatch.setattr(PR.sys, "stderr", fake)

    def handler(db_, t, p):
        PR.report("chunk", done=1, total=2)

    with temp_job_kind("t_prog_tty", "cpu", handler):
        vid = make_video(db)
        Q.enqueue(db, "t_prog_tty", vid, now=NOW)
        _tick(db, "cpu")
        row = db.conn.execute(
            "SELECT progress FROM jobs WHERE target_id = ?", (vid,)
        ).fetchone()
    assert "chunk 1/2" in fake.getvalue()
    assert row["progress"] is None  # cleared once the job left `running`


def test_current_and_combine_fan_a_report_out_to_both_sinks() -> None:
    calls: list[str] = []
    with PR.install(lambda s, d, t, x: calls.append(f"outer:{s}")):
        outer = PR.current()
        with PR.install(PR.combine(outer, lambda s, d, t, x: calls.append(f"inner:{s}"))):
            PR.report("stage")
    assert calls == ["outer:stage", "inner:stage"]


def test_combine_runs_every_sink_even_if_one_raises() -> None:
    calls: list[str] = []

    def boom(s, d, t, x):
        raise RuntimeError("a broken sink must not silence the others")

    sink = PR.combine(boom, lambda s, d, t, x: calls.append(s))
    sink("stage", None, None, "")
    assert calls == ["stage"]


def test_stats_log_is_silent_on_an_idle_queue(db: Database) -> None:
    with temp_job_kind("t_quiet", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_quiet", make_video(db), now=NOW)
        _tick(db, "cpu")  # runs to completion: nothing pending or running
    lines: list[str] = []
    monkeypatch_log = W._log
    try:
        W._log = lines.append  # type: ignore[assignment]
        W._log_stats(db)
    finally:
        W._log = monkeypatch_log  # type: ignore[assignment]
    assert lines == []


def test_stats_log_reports_counts_while_work_remains(db: Database) -> None:
    with temp_job_kind("t_pending", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_pending", make_video(db), now=NOW)
    lines: list[str] = []
    monkeypatch_log = W._log
    try:
        W._log = lines.append  # type: ignore[assignment]
        W._log_stats(db)
    finally:
        W._log = monkeypatch_log  # type: ignore[assignment]
    assert len(lines) == 1
    assert "pending=1" in lines[0]


def test_the_threaded_worker_logs_lifecycle_lines_too(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(W.sys, "stderr", fake)
    with temp_job_kind("t_cpu", "cpu", lambda db_, t, p: None):
        Q.enqueue(db, "t_cpu", make_video(db), now=NOW)
        W.run_worker(
            db,
            W.WorkerOptions(pools=("cpu",), max_jobs=1, poll_interval_s=0.01),
        )
    lines = fake.getvalue()
    assert "job 1 claimed: kind=t_cpu" in lines
    assert "job 1 done in" in lines


def test_the_worker_says_it_started(db: Database, capsys) -> None:
    """A worker started against an empty queue must not be silent.

    Otherwise it is indistinguishable from one that failed to start —
    the doubt BUGS.md entries 41 and 42 are both about. The pid is named
    so an operator can match this line to the lease `jobs list` reads.
    """
    W.run_worker(db, W.WorkerOptions(pools=("cpu",), once=True))
    err = capsys.readouterr().err
    assert "watching pools" in err
    assert "cpu" in err
    assert str(os.getpid()) in err

