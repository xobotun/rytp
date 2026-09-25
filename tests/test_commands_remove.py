"""Tests for Part 2's deletion commands (contracts §5, Deletion)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.commands import resolve
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for, insert_asset
from rytp.jobs import queue as Q
from rytp.models import InvalidInputError, NotFoundError
from tests.fakes import make_video, touch


def _cancel(db: Database, **kwargs: object):
    args: dict[str, object] = {"job_id": 0, "kind": "", "state": "", "target_id": 0,
                               "dry_run": False}
    args.update(kwargs)
    return resolve("jobs.cancel").handler(db, **args)


def _remove(db: Database, **kwargs: object):
    args: dict[str, object] = {"asset_id": 0, "yes": False, "dry_run": False}
    args.update(kwargs)
    return resolve("assets.remove").handler(db, **args)


def test_cancel_one_job_by_id(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    result = _cancel(db, job_id=job_id)
    assert "1" in (result.message or "")
    assert Q.get_job(db, job_id).state == "cancelled"


def test_cancel_a_whole_bad_batch_by_filter(db: Database) -> None:
    ids = []
    for i in range(3):
        vid = make_video(db, external_id=f"VIDEO_{i}",
                         url=f"https://example.invalid/{i}")
        ids.append(Q.enqueue(db, "download", vid))
        Q.enqueue(db, "captions", vid)
    _cancel(db, kind="download")
    assert {Q.get_job(db, i).state for i in ids} == {"cancelled"}
    assert {j.state for j in Q.list_jobs(db, kind="captions")} == {"pending"}


def test_cancel_can_target_one_video(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", a)
    Q.enqueue(db, "download", b)
    _cancel(db, target_id=a)
    states = {j.target_id: j.state for j in Q.list_jobs(db)}
    assert states == {a: "cancelled", b: "pending"}


def test_cancel_refuses_a_running_job_and_says_what_to_do(db: Database) -> None:
    vid = make_video(db)
    Q.enqueue(db, "download", vid)
    job = Q.claim(db, "network")
    assert job is not None
    with pytest.raises(InvalidInputError, match="queue pause"):
        _cancel(db, job_id=job.id)
    assert Q.get_job(db, job.id).state == "running"


def test_cancel_by_filter_skips_running_jobs_rather_than_failing(
    db: Database,
) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    Q.enqueue(db, "download", a)
    Q.enqueue(db, "download", b)
    running = Q.claim(db, "network")
    assert running is not None
    result = _cancel(db, kind="download")
    assert "1 still running" in (result.message or "")
    assert Q.get_job(db, running.id).state == "running"
    others = [j for j in Q.list_jobs(db) if j.id != running.id]
    assert {j.state for j in others} == {"cancelled"}


def test_cancel_dry_run_changes_nothing(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    result = _cancel(db, kind="download", dry_run=True)
    assert len(result.rows) == 1
    assert Q.get_job(db, job_id).state == "pending"


def test_cancel_dry_run_lists_newest_job_first(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    first = Q.enqueue(db, "download", a)
    second = Q.enqueue(db, "download", b)
    result = _cancel(db, kind="download", dry_run=True)
    assert [row[0] for row in result.rows] == [str(second), str(first)]


def test_cancel_with_no_filter_at_all_is_refused(db: Database) -> None:
    # "rytp jobs cancel" with nothing set would wipe the queue by accident.
    with pytest.raises(InvalidInputError, match="filter"):
        _cancel(db)


def test_a_cancelled_job_is_not_resurrected_by_reconcile(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    _cancel(db, job_id=job_id)
    assert Q.reconcile(db) == 0
    assert Q.get_job(db, job_id).state == "cancelled"


def test_remove_a_superseded_rendition(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    old = touch(tmp_path / "v360.mp4", b"\x00" * 360)
    insert_asset(db, video_id=vid, role="video", format_id="360", path=str(old))
    new = touch(tmp_path / "v1080.mp4")
    insert_asset(db, video_id=vid, role="video", format_id="1080", path=str(new))
    asset_id = next(
        r["id"] for r in assets_for(db, vid, "video") if r["format_id"] == "360"
    )

    result = _remove(db, asset_id=asset_id, yes=True)
    assert not old.exists()
    assert new.exists()
    assert "360" in (result.message or "")
    assert [r["format_id"] for r in assets_for(db, vid, "video")] == ["1080"]


def test_remove_without_yes_refuses_because_it_deletes_a_file(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4")
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    with pytest.raises(InvalidInputError, match="--yes"):
        _remove(db, asset_id=asset_id)
    assert p.exists()


def test_remove_dry_run_reports_bytes_and_deletes_nothing(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4", b"\x00" * 1234)
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    result = _remove(db, asset_id=asset_id, dry_run=True)
    assert "1234" in (result.message or "")
    assert p.exists()
    assert asset_for(db, vid, "video") is not None


def test_removing_the_audio_warns_loudly_but_still_works(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    asset_id = insert_asset(db, video_id=vid, role="audio", path=str(p))
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "re-align" in (result.message or "")
    assert not p.exists()
    assert asset_for(db, vid, "audio") is None


def test_removing_captions_warns_while_the_words_are_still_caption_tier(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "captions.json3")
    asset_id = insert_asset(db, video_id=vid, role="captions", path=str(p))
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, "
        "stem, source, engine) VALUES (?, 0, 0, 'a', 'a', 'a', 'caption', 'x')",
        (vid,),
    )
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "not been superseded" in (result.message or "")


def test_removing_captions_is_quiet_once_a_better_transcript_exists(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "captions.json3")
    asset_id = insert_asset(db, video_id=vid, role="captions", path=str(p))
    db.conn.execute(
        # contracts §3: `timed` is not cuttable but is still better text
        # than captions, so it supersedes them.
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, "
        "normalized_text, stem, source, engine) "
        "VALUES (?, 0, 0, 10, 'a', 'a', 'a', 'timed', 'x')",
        (vid,),
    )
    result = _remove(db, asset_id=asset_id, yes=True)
    assert "not been superseded" not in (result.message or "")


def test_removing_a_rendition_makes_its_download_job_runnable_again(
    db: Database, tmp_path: Path
) -> None:
    # design §5's self-healing, from the deletion side.
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio",
                 path=str(touch(tmp_path / "audio.m4a")))
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="720",
                            path=str(touch(tmp_path / "v720.mp4")))
    job_id = Q.enqueue(db, "download", vid)
    assert Q.get_job(db, job_id).state == "done"

    _remove(db, asset_id=asset_id, yes=True)
    assert Q.get_job(db, job_id).state == "pending"


def test_remove_of_an_unknown_asset_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        _remove(db, asset_id=4242, yes=True)


def test_remove_tolerates_a_file_already_gone(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "v360.mp4")
    asset_id = insert_asset(db, video_id=vid, role="video", format_id="360",
                            path=str(p))
    p.unlink()
    _remove(db, asset_id=asset_id, yes=True)
    assert assets_for(db, vid, "video") == []
