"""Tests for job-kind registration and readiness (design §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.jobs import (
    JOB_KINDS,
    JobKind,
    Readiness,
    kinds_for_pool,
    register_job_kind,
    resolve_job_kind,
)
from rytp.jobs.readiness import (
    captions_readiness,
    download_readiness,
    extract_wav_readiness,
)
from tests.fakes import make_video, touch


def test_the_three_part_two_kinds_are_registered() -> None:
    assert resolve_job_kind("download").pool == "network"
    assert resolve_job_kind("captions").pool == "network"
    assert resolve_job_kind("extract_wav").pool == "cpu"
    assert set(kinds_for_pool("network")) >= {"download", "captions"}


def test_job_handlers_is_the_contracted_flat_view() -> None:
    # contracts §5 requires rytp/jobs/__init__.py to expose
    # JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]].
    from rytp.jobs import JOB_HANDLERS

    assert set(JOB_HANDLERS) == set(JOB_KINDS)
    for name, kind in JOB_KINDS.items():
        assert JOB_HANDLERS[name] is kind.handler


def test_kinds_target_videos_by_default_and_are_reopenable() -> None:
    for name in ("download", "captions", "extract_wav"):
        kind = resolve_job_kind(name)
        assert kind.target_kind == "video"
        assert kind.reopenable is True


def test_unknown_kind_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="download"):
        resolve_job_kind("nope")


def test_registering_a_duplicate_is_refused() -> None:
    kind = JobKind(
        name="download", pool="cpu", readiness=lambda db, t: Readiness.READY,
        handler=lambda db, t, p: None, summary="x",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_job_kind(kind)


def test_registering_an_unknown_pool_is_refused() -> None:
    kind = JobKind(
        name="invented", pool="quantum", readiness=lambda db, t: Readiness.READY,
        handler=lambda db, t, p: None, summary="x",
    )
    with pytest.raises(ValueError, match="quantum"):
        register_job_kind(kind)
    assert "invented" not in JOB_KINDS


def test_download_is_ready_for_a_catalogued_remote_video(db: Database) -> None:
    vid = make_video(db)
    assert download_readiness(db, vid) is Readiness.READY


def test_download_is_satisfied_only_with_both_audio_and_video(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert download_readiness(db, vid) is Readiness.READY
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v.mp4")))
    assert download_readiness(db, vid) is Readiness.SATISFIED


def test_deleting_the_media_makes_download_ready_again(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    audio = touch(tmp_path / "a.m4a")
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v.mp4")))
    assert download_readiness(db, vid) is Readiness.SATISFIED
    audio.unlink()
    assert download_readiness(db, vid) is Readiness.READY


def test_local_videos_never_get_a_download_job(db: Database) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id="C:/archive/clip.mkv", url=None)
    assert download_readiness(db, vid) is Readiness.BLOCKED
    assert captions_readiness(db, vid) is Readiness.BLOCKED


def test_a_remote_video_without_a_url_is_blocked(db: Database) -> None:
    vid = make_video(db, url=None)
    assert download_readiness(db, vid) is Readiness.BLOCKED


def test_a_missing_video_is_blocked_not_an_error(db: Database) -> None:
    assert download_readiness(db, 4242) is Readiness.BLOCKED
    assert extract_wav_readiness(db, 4242) is Readiness.BLOCKED


def test_captions_is_satisfied_once_the_track_is_on_disk(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    assert captions_readiness(db, vid) is Readiness.READY
    insert_asset(db, video_id=vid, role="captions",
                 path=str(touch(tmp_path / "captions.json3")))
    assert captions_readiness(db, vid) is Readiness.SATISFIED


def test_extract_wav_needs_audio_or_a_container(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    assert extract_wav_readiness(db, vid) is Readiness.BLOCKED
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    assert extract_wav_readiness(db, vid) is Readiness.READY


def test_a_container_alone_is_enough_for_extract_wav(db: Database, tmp_path: Path) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"), url=None)
    insert_asset(db, video_id=vid, role="container", path=str(touch(tmp_path / "clip.mkv")))
    assert extract_wav_readiness(db, vid) is Readiness.READY


def test_pruning_the_wav_makes_extract_ready_again(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    wav = config.paths().cache_wav(vid)
    touch(wav)
    assert extract_wav_readiness(db, vid) is Readiness.SATISFIED
    wav.unlink()  # `rytp cache prune`
    assert extract_wav_readiness(db, vid) is Readiness.READY
