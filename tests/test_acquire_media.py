"""Tests for the download stage (design §4, §5). No network, ever."""

from __future__ import annotations

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire import acquire_media
from rytp.acquire.policy import AcquireError, RateLimited, VideoUnavailable
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for, insert_asset
from rytp.jobs import Readiness
from rytp.jobs.readiness import download_readiness
from tests.fakes import AUDIO_FILE_SPEC, FakeYtDlpRunner, make_video


def test_audio_and_video_land_as_two_separate_assets(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    message = acquire_media(db, vid, runner=runner)

    audio = asset_for(db, vid, "audio")
    video = asset_for(db, vid, "video")
    assert audio is not None and video is not None
    assert audio["path"].endswith("audio.m4a")
    assert video["path"].endswith("video-136.mp4")
    assert audio["abr"] == 129.5
    assert (video["width"], video["height"]) == (1280, 720)
    assert audio["bytes"] > 0
    assert "audio.m4a" in message and "video-136.mp4" in message


def test_the_two_files_live_under_the_contracted_layout(db: Database) -> None:
    vid = make_video(db)
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    expected_dir = config.paths().media_dir(vid)
    for row in assets_for(db, vid):
        assert row["path"].startswith(str(expected_dir))


def test_nothing_is_merged(db: Database) -> None:
    # design §4: a merged file would make a rendition upgrade impossible
    # without re-fetching the audio a transcript is aligned to.
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_media(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download")
    assert "," in call["format_selector"]
    assert "+" not in call["format_selector"]
    assert call["format_selector"] == f"{C.DOWNLOAD_FORMAT_AUDIO},{C.DOWNLOAD_FORMAT_VIDEO}"


def test_the_rate_limit_from_settings_reaches_the_runner(db: Database) -> None:
    from rytp.db.queries import set_setting

    set_setting(db, C.SETTING_DOWNLOAD_RATE_LIMIT, "42000")
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_media(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download")
    assert call["rate_limit_bps"] == 42000


def test_the_job_becomes_satisfied_afterwards(db: Database) -> None:
    vid = make_video(db)
    assert download_readiness(db, vid) is Readiness.READY
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    assert download_readiness(db, vid) is Readiness.SATISFIED


def test_a_rendition_upgrade_leaves_the_audio_alone(db: Database) -> None:
    vid = make_video(db)
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    audio_before = asset_for(db, vid, "audio")

    upgraded = dict(AUDIO_FILE_SPEC)
    hd = {"name": "137.mp4", "format_id": "137", "ext": "mp4",
          "vcodec": "avc1.640028", "acodec": "none", "width": 1920, "height": 1080}
    acquire_media(db, vid, runner=FakeYtDlpRunner(files=[upgraded, hd]))

    assert {r["format_id"] for r in assets_for(db, vid, "video")} == {"136", "137"}
    audio_after = asset_for(db, vid, "audio")
    assert audio_after["path"] == audio_before["path"]


def test_stale_asset_rows_are_pruned_before_a_refetch(db: Database, tmp_path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="999",
                 path=str(tmp_path / "gone.mp4"))
    acquire_media(db, vid, runner=FakeYtDlpRunner())
    assert {r["format_id"] for r in assets_for(db, vid, "video")} == {"136"}


def test_a_local_video_is_refused(db: Database, tmp_path) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"), url=None)
    with pytest.raises(AcquireError, match="local"):
        acquire_media(db, vid, runner=FakeYtDlpRunner())


def test_a_missing_video_is_refused(db: Database) -> None:
    with pytest.raises(AcquireError, match="4242"):
        acquire_media(db, 4242, runner=FakeYtDlpRunner())


def test_no_audio_format_is_a_plain_failure(db: Database) -> None:
    vid = make_video(db)
    video_only = {"name": "136.mp4", "format_id": "136", "ext": "mp4",
                  "vcodec": "avc1", "acodec": "none"}
    with pytest.raises(AcquireError, match="no audio"):
        acquire_media(db, vid, runner=FakeYtDlpRunner(files=[video_only]))


def test_no_video_rendition_keeps_the_audio_and_fails_permanently(db: Database) -> None:
    vid = make_video(db)
    audio_only = dict(AUDIO_FILE_SPEC)
    with pytest.raises(VideoUnavailable, match="no video"):
        acquire_media(db, vid, runner=FakeYtDlpRunner(files=[audio_only]))
    # The audio we did get is kept — throwing it away would mean fetching it twice.
    assert asset_for(db, vid, "audio") is not None


def test_runner_failures_are_translated(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429: Too Many Requests"))
    with pytest.raises(RateLimited):
        acquire_media(db, vid, runner=runner)
