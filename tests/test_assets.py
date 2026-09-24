"""Tests for the asset helpers (design §4, contracts §3)."""

from __future__ import annotations

from pathlib import Path

from rytp.db import Database
from rytp.db.queries import (
    asset_for,
    assets_for,
    has_usable_asset,
    insert_asset,
    prune_missing_assets,
)
from tests.fakes import make_video, touch


def test_insert_and_read_back(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    asset_id = insert_asset(
        db, video_id=vid, role="audio", path=str(p), format_id="140",
        size_bytes=16, abr=128.0,
    )
    row = asset_for(db, vid, "audio")
    assert row is not None
    assert row["id"] == asset_id
    assert row["path"] == str(p)
    assert row["format_id"] == "140"
    assert row["abr"] == 128.0
    assert row["acquired_at"]  # ISO-8601, stamped for us


def test_audio_is_a_singleton_role(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a1.m4a")))
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a2.m4a")))
    rows = assets_for(db, vid, "audio")
    assert len(rows) == 1
    assert rows[0]["path"].endswith("a2.m4a")


def test_video_renditions_accumulate_but_one_per_format(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360.mp4")), height=360)
    insert_asset(db, video_id=vid, role="video", format_id="720",
                 path=str(touch(tmp_path / "v720.mp4")), height=720)
    # Re-fetching the same format replaces that rendition, not the other one.
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360b.mp4")), height=360)
    rows = assets_for(db, vid, "video")
    assert sorted(r["format_id"] for r in rows) == ["360", "720"]
    assert {Path(r["path"]).name for r in rows} == {"v360b.mp4", "v720.mp4"}


def test_asset_for_returns_the_newest_rendition(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="video", format_id="360",
                 path=str(touch(tmp_path / "v360.mp4")))
    insert_asset(db, video_id=vid, role="video", format_id="1080",
                 path=str(touch(tmp_path / "v1080.mp4")))
    row = asset_for(db, vid, "video")
    assert row is not None and row["format_id"] == "1080"


def test_has_usable_asset_is_false_when_the_file_is_gone(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    p = touch(tmp_path / "audio.m4a")
    insert_asset(db, video_id=vid, role="audio", path=str(p))
    assert has_usable_asset(db, vid, "audio") is True
    p.unlink()
    assert has_usable_asset(db, vid, "audio") is False


def test_prune_missing_assets_drops_only_the_dead_rows(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    alive = touch(tmp_path / "audio.m4a")
    dead = touch(tmp_path / "v720.mp4")
    insert_asset(db, video_id=vid, role="audio", path=str(alive))
    insert_asset(db, video_id=vid, role="video", format_id="720", path=str(dead))
    dead.unlink()
    assert prune_missing_assets(db, vid) == 1
    assert [r["role"] for r in assets_for(db, vid)] == ["audio"]


def test_assets_are_deleted_with_their_video(db: Database, tmp_path: Path) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    db.conn.execute("DELETE FROM videos WHERE id = ?", (vid,))
    assert assets_for(db, vid) == []
