"""Tests for the job queue (design §5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.jobs import Readiness
from rytp.jobs import queue as Q
from rytp.models import NotFoundError
from tests.fakes import make_video, temp_job_kind, touch

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def test_enqueue_starts_a_ready_job_pending(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    job = Q.list_jobs(db)[0]
    assert job.id == job_id
    assert (job.kind, job.state, job.pool, job.attempts) == (
        "download", "pending", "network", 0
    )


def test_enqueue_starts_a_blocked_job_blocked(db: Database) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id="C:/clip.mkv", url=None)
    Q.enqueue(db, "download", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "blocked"


def test_enqueue_marks_an_already_satisfied_job_done(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="captions",
                 path=str(touch(tmp_path / "captions.json3")))
    Q.enqueue(db, "captions", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "done"


def test_enqueue_is_idempotent_and_keeps_one_row(db: Database) -> None:
    vid = make_video(db)
    first = Q.enqueue(db, "download", vid, now=NOW)
    second = Q.enqueue(db, "download", vid, now=NOW)
    assert first == second
    assert len(Q.list_jobs(db)) == 1


def test_enqueue_raises_priority_but_never_lowers_it(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, priority=5, now=NOW)
    Q.enqueue(db, "download", vid, priority=1, now=NOW)
    assert Q.list_jobs(db)[0].priority == 5


def test_claim_moves_pending_to_running_and_burns_an_attempt(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    job = Q.claim(db, "network", now=NOW)
    assert job is not None
    assert job.state == "running" and job.attempts == 1
    assert Q.claim(db, "network", now=NOW) is None  # nothing left to claim


def test_claim_ignores_other_pools(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.claim(db, "network", now=NOW) is None
    assert Q.claim(db, "cpu", now=NOW) is not None


def test_claim_respects_priority_then_insertion_order(db: Database) -> None:
    low = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    high = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", low, priority=0, now=NOW)
    Q.enqueue(db, "download", high, priority=9, now=NOW)
    assert Q.claim(db, "network", now=NOW).target_id == high


def test_claim_skips_a_job_whose_not_before_is_in_the_future(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    later = (NOW + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=later, error="429")
    assert Q.claim(db, "network", now=NOW) is None
    assert Q.claim(db, "network", now=NOW + timedelta(minutes=6)) is not None


def test_defer_can_refund_the_attempt(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.defer(db, job_id, not_before=NOW.isoformat(), error="429", refund_attempt=True)
    job = Q.list_jobs(db)[0]
    assert job.state == "pending" and job.attempts == 0


def test_finish_and_fail_are_terminal(db: Database) -> None:
    vid = make_video(db)
    a = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.finish(db, a, now=NOW)
    done = Q.list_jobs(db)[0]
    assert done.state == "done" and done.last_error is None
    Q.fail(db, a, error="boom", now=NOW)
    assert Q.list_jobs(db)[0].last_error == "boom"


def test_reclaim_running_returns_orphans_to_pending(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)      # the worker is now killed
    assert Q.reclaim_running(db, now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert job.state == "pending"
    assert job.attempts == 1             # the attempt is still spent


def test_reconcile_reopens_a_done_job_whose_output_vanished(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    wav = touch(config.paths().cache_wav(vid))
    job_id = Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "done"

    wav.unlink()                          # `rytp cache prune`
    assert Q.reconcile(db, now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert job.id == job_id and job.state == "pending"


def test_reconcile_unblocks_a_job_whose_prerequisite_arrived(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert Q.list_jobs(db)[0].state == "blocked"
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert Q.reconcile(db, now=NOW) == 1
    assert Q.list_jobs(db)[0].state == "pending"


def test_unblock_touches_only_blocked_jobs(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)          # blocked: no audio yet
    Q.enqueue(db, "download", vid, now=NOW)             # pending
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert Q.unblock(db, target_id=vid, now=NOW) == 1
    states = {j.kind: j.state for j in Q.list_jobs(db)}
    assert states == {"extract_wav": "pending", "download": "pending"}


def test_unblock_never_reopens_a_done_job(db: Database, tmp_path: Path) -> None:
    # A predicate that still answers READY after a successful run must not
    # send the job round again — that is why the worker calls unblock, not
    # reconcile, after every finish.
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    job_id = Q.enqueue(db, "extract_wav", vid, now=NOW)
    Q.finish(db, job_id, now=NOW)
    assert Q.unblock(db, target_id=vid, now=NOW) == 0
    assert Q.get_job(db, job_id).state == "done"


def test_reconcile_can_be_scoped_to_one_video(db: Database, tmp_path: Path) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "extract_wav", a, now=NOW)
    Q.enqueue(db, "extract_wav", b, now=NOW)
    for vid, name in ((a, "a.m4a"), (b, "b.m4a")):
        insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / name)))
    assert Q.reconcile(db, target_id=a, now=NOW) == 1
    states = {j.target_id: j.state for j in Q.list_jobs(db)}
    assert states == {a: "pending", b: "blocked"}


def test_unblock_never_crosses_a_target_namespace(db: Database, tmp_path: Path) -> None:
    # A render's target_id is a renders.id, so it shares no numbering with
    # videos.id. Finishing one must not re-evaluate the other.
    vid = make_video(db)
    Q.enqueue(db, "extract_wav", vid, now=NOW)          # blocked: no audio yet
    with temp_job_kind(
        "t_render", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED, target_kind="render",
    ):
        Q.enqueue(db, "t_render", vid, now=NOW)         # same integer, other namespace
        insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
        assert Q.unblock(db, target_id=vid, target_kind="render", now=NOW) == 0
        assert Q.unblock(db, target_id=vid, target_kind="video", now=NOW) == 1
    states = {j.kind: j.state for j in Q.list_jobs(db)}
    assert states["extract_wav"] == "pending"
    assert states["t_render"] == "blocked"


def test_reconcile_never_reopens_a_kind_marked_not_reopenable(db: Database) -> None:
    # A render is something the user asked for once. Re-deriving it on every
    # worker start because its output file is gone would be a nasty surprise.
    vid = make_video(db)
    with temp_job_kind(
        "t_render", "cpu", lambda db_, t, p: None, target_kind="render",
        reopenable=False,
    ):
        job_id = Q.enqueue(db, "t_render", vid, now=NOW)
        Q.finish(db, job_id, now=NOW)
        assert Q.reconcile(db, now=NOW) == 0
        assert Q.get_job(db, job_id).state == "done"
        # Asking again explicitly still re-runs it.
        Q.enqueue(db, "t_render", vid, now=NOW)
        assert Q.get_job(db, job_id).state == "pending"


def test_reconcile_leaves_running_and_failed_jobs_alone(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    assert Q.reconcile(db, now=NOW) == 0
    assert Q.list_jobs(db)[0].state == "running"


def test_retry_resets_failed_jobs_and_clears_the_backoff(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.fail(db, job_id, error="boom", now=NOW)
    assert Q.retry(db, state="failed", now=NOW) == 1
    job = Q.list_jobs(db)[0]
    assert (job.state, job.attempts, job.not_before) == ("pending", 0, None)


def test_retry_can_target_a_single_job(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    ja = Q.enqueue(db, "download", a, now=NOW)
    jb = Q.enqueue(db, "download", b, now=NOW)
    Q.fail(db, ja, error="x", now=NOW)
    Q.fail(db, jb, error="y", now=NOW)
    assert Q.retry(db, job_id=ja, now=NOW) == 1
    states = {j.id: j.state for j in Q.list_jobs(db)}
    assert states[ja] == "pending" and states[jb] == "failed"


def test_list_jobs_filters(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "download", vid, now=NOW)
    Q.enqueue(db, "extract_wav", vid, now=NOW)
    assert [j.kind for j in Q.list_jobs(db, pool="cpu")] == ["extract_wav"]
    assert [j.kind for j in Q.list_jobs(db, kind="download")] == ["download"]
    assert Q.list_jobs(db, state="failed") == []


def test_a_handlers_note_is_stored_and_replaced_not_accumulated(
    db: Database,
) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.finish(db, job_id, note="discarded the speaker mapping", now=NOW)
    job = Q.get_job(db, job_id)
    # A note never affects state: this job succeeded.
    assert job.state == "done"
    assert job.note == "discarded the speaker mapping"

    Q.finish(db, job_id, note=None, now=NOW)
    assert Q.get_job(db, job_id).note is None


def test_stats_counts_jobs_carrying_a_note(db: Database) -> None:
    for i in range(3):
        vid = make_video(db, external_id=f"VIDEO_{i}",
                         url=f"https://example.invalid/{i}")
        job_id = Q.enqueue(db, "download", vid, now=NOW)
        Q.finish(db, job_id, note="lost speaker labels" if i < 2 else None, now=NOW)
    assert Q.stats(db, now=NOW).noted == 2


def test_get_job_round_trips_and_complains_about_a_bad_id(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, payload={"note": "hi"}, now=NOW)
    job = Q.get_job(db, job_id)
    assert job.kind == "download" and job.payload == {"note": "hi"}
    with pytest.raises(NotFoundError, match="4242"):
        Q.get_job(db, 4242)


def test_stats_reports_throttling_rather_than_a_mystery_stall(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    later = (NOW + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=later, error="HTTP Error 429")
    s = Q.stats(db, now=NOW)
    assert s.by_state["pending"] == 1
    assert s.throttled == 1
    assert s.next_not_before == later
    assert s.paused is False


def test_pause_is_a_setting_not_a_state(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    assert Q.is_paused(db) is False
    Q.pause(db)
    assert Q.is_paused(db) is True
    assert Q.list_jobs(db)[0].state == "pending"   # jobs are untouched
    Q.resume(db)
    assert Q.is_paused(db) is False


def test_two_connections_cannot_claim_the_same_job(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid, now=NOW)
    other = Database(config.paths().db)
    try:
        first = Q.claim(db, "network", now=NOW)
        second = Q.claim(other, "network", now=NOW)
    finally:
        other.close()
    assert first is not None
    assert second is None


# ---------------------------------------------------------------------------
# ``jobs.progress`` — advisory, worker-written, distinct from ``note``
# (contracts §5, plan §1c, Task 8a).
# ---------------------------------------------------------------------------


def test_set_progress_writes_the_column(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38 clip.mp4")
    assert Q.list_jobs(db)[0].progress == "download 12/38 clip.mp4"


def test_claim_starts_a_job_with_no_stale_progress(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.defer(db, job_id, not_before=NOW.isoformat())  # back to pending
    Q.claim(db, "network", now=NOW)                  # claimed again
    assert Q.list_jobs(db)[0].progress is None


def test_finish_clears_progress(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.finish(db, job_id, now=NOW)
    assert Q.list_jobs(db)[0].progress is None


def test_fail_clears_progress(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.fail(db, job_id, error="boom", now=NOW)
    assert Q.list_jobs(db)[0].progress is None


def test_block_clears_progress(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.block(db, job_id, reason="prerequisites are not in place", now=NOW)
    assert Q.list_jobs(db)[0].progress is None


def test_defer_clears_progress(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.defer(db, job_id, not_before=NOW.isoformat(), error="429")
    assert Q.list_jobs(db)[0].progress is None


def test_reclaim_running_clears_progress(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.reclaim_running(db, now=NOW)
    assert Q.list_jobs(db)[0].progress is None


def test_progress_is_distinct_from_note(db: Database) -> None:
    """A note survives ``finish``; progress never does — the two must not
    be conflated (contracts §5)."""
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid, now=NOW)
    Q.claim(db, "network", now=NOW)
    Q.set_progress(db, job_id, "download 12/38")
    Q.finish(db, job_id, note="discarded 4 speaker labels", now=NOW)
    job = Q.list_jobs(db)[0]
    assert job.note == "discarded 4 speaker labels"
    assert job.progress is None
