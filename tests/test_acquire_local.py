"""Tests for local file registration (design §4, §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.acquire.local import register_local_container
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import asset_for, assets_for
from rytp.jobs import Readiness
from rytp.jobs.readiness import download_readiness, extract_wav_readiness
from tests.fakes import make_video, touch


def _local(db: Database, path: Path) -> int:
    return make_video(db, source=C.LOCAL_SOURCE, external_id=str(path), url=None,
                      title=path.name)


def test_a_local_file_becomes_a_container_asset(db: Database, tmp_path: Path) -> None:
    clip = touch(tmp_path / "clip.mkv", b"\x00" * 64)
    vid = _local(db, clip)
    message = register_local_container(db, vid)
    row = asset_for(db, vid, "container")
    assert row is not None
    assert row["path"] == str(clip)
    assert row["bytes"] == 64
    assert "clip.mkv" in message


def test_the_file_is_referenced_in_place_and_not_copied(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "elsewhere" / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    assert clip.exists()
    assert asset_for(db, vid, "container")["path"] == str(clip)


def test_registration_unblocks_extract_but_never_download(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    assert extract_wav_readiness(db, vid) is Readiness.READY
    assert download_readiness(db, vid) is Readiness.BLOCKED


def test_registering_twice_keeps_one_row(db: Database, tmp_path: Path) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = _local(db, clip)
    register_local_container(db, vid)
    register_local_container(db, vid)
    assert len(assets_for(db, vid, "container")) == 1


def test_a_missing_file_is_reported_in_one_line(db: Database, tmp_path: Path) -> None:
    vid = _local(db, tmp_path / "gone.mkv")
    with pytest.raises(MissingAssetError, match=r"gone\.mkv"):
        register_local_container(db, vid)


def test_a_remote_video_is_refused(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(AcquireError, match="not a local file"):
        register_local_container(db, vid)
