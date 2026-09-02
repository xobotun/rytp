"""Tests for ``rytp.channels``."""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp import channels
from rytp.channels import add_channel, list_videos, register_video, sync_channel
from rytp.download.ytdlp import ChannelVideo, VideoMetadata


class FakeRunner:
    """Minimal YtDlpRunner fake."""

    def __init__(self) -> None:
        self.list_channel_return: list[ChannelVideo] = []
        self.probe_return: VideoMetadata | None = None

    def list_channel(self, url: str) -> list[ChannelVideo]:
        return list(self.list_channel_return)

    def probe(self, url: str) -> VideoMetadata:
        assert self.probe_return is not None, "probe_return not configured"
        return self.probe_return


def test_add_channel_inserts_and_returns_id(db) -> None:
    runner = FakeRunner()
    cid = add_channel(db, runner, "https://www.youtube.com/@test", title="Test Channel")
    assert cid > 0
    row = db.conn.execute(
        "SELECT * FROM channels WHERE id = ?", (cid,)
    ).fetchone()
    assert row["url"] == "https://www.youtube.com/@test"
    assert row["title"] == "Test Channel"


def test_add_channel_probes_title_when_missing(db) -> None:
    runner = FakeRunner()
    runner.probe_return = VideoMetadata(
        youtube_id="x", title="Probed Title", duration=10, kind="video",
        url="https://example.com/x", published_at=None,
    )
    cid = add_channel(db, runner, "https://example.com/")
    row = db.conn.execute("SELECT title FROM channels WHERE id = ?", (cid,)).fetchone()
    assert row["title"] == "Probed Title"


def test_sync_channel_inserts_videos(db) -> None:
    runner = FakeRunner()
    cid = add_channel(db, runner, "https://example.com/c", title="C")
    runner.list_channel_return = [
        ChannelVideo(
            youtube_id=f"vid{i}", title=f"V{i}", url=f"https://example.com/v{i}",
            duration=100, kind="video", published_at=None,
        )
        for i in range(3)
    ]
    n = sync_channel(db, runner, cid)
    assert n == 3
    rows = db.conn.execute(
        "SELECT youtube_id FROM videos WHERE channel_id = ? ORDER BY youtube_id",
        (cid,),
    ).fetchall()
    assert [r["youtube_id"] for r in rows] == ["vid0", "vid1", "vid2"]


def test_sync_channel_is_idempotent(db) -> None:
    runner = FakeRunner()
    cid = add_channel(db, runner, "https://example.com/c", title="C")
    runner.list_channel_return = [
        ChannelVideo(
            youtube_id="vid0", title="V0", url="https://example.com/v0",
            duration=100, kind="video", published_at=None,
        )
    ]
    n1 = sync_channel(db, runner, cid)
    n2 = sync_channel(db, runner, cid)
    assert n1 == 1
    assert n2 == 0  # no new rows


def test_list_videos_filters(db, fake_video_row) -> None:
    db.upsert_video(fake_video_row)
    db.upsert_video({**fake_video_row, "source": "local", "youtube_id": None, "local_path": "/tmp/x.mp4"})
    videos = list_videos(db)
    assert len(videos) == 2
    by_source = list_videos(db, source="local")
    assert len(by_source) == 1
    assert by_source[0].source == "local"


def test_register_video_local_path_marks_downloaded(db, tmp_path: Path) -> None:
    runner = FakeRunner()
    f = tmp_path / "movie.mp4"
    f.write_bytes(b"x")
    vid = register_video(db, runner, str(f))
    row = db.conn.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["source"] == "local"
    assert row["downloaded"] == 1
    assert row["local_path"] == str(f.resolve())
    # Title defaults to file stem
    assert row["title"] == "movie"


def test_register_video_youtube_url(db) -> None:
    runner = FakeRunner()
    runner.probe_return = VideoMetadata(
        youtube_id="vid123", title="YT", duration=600, kind="video",
        url="https://www.youtube.com/watch?v=vid123", published_at="2024-01-01",
    )
    vid = register_video(
        db, runner, "https://www.youtube.com/watch?v=vid123"
    )
    row = db.conn.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["source"] == "youtube"
    assert row["youtube_id"] == "vid123"
    assert row["downloaded"] == 0


def test_register_video_extracts_youtube_id_when_metadata_missing(db) -> None:
    runner = FakeRunner()
    runner.probe_return = VideoMetadata(
        youtube_id=None, title="YT", duration=600, kind="video",
        url="https://www.youtube.com/watch?v=abcdef12345", published_at=None,
    )
    vid = register_video(db, runner, "https://www.youtube.com/watch?v=abcdef12345")
    row = db.conn.execute("SELECT youtube_id FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["youtube_id"] == "abcdef12345"


def test_register_video_non_youtube_url_uses_ytdlp_source(db) -> None:
    runner = FakeRunner()
    runner.probe_return = VideoMetadata(
        youtube_id=None, title="Other", duration=100, kind="video",
        url="https://vimeo.com/12345", published_at=None,
    )
    vid = register_video(db, runner, "https://vimeo.com/12345")
    row = db.conn.execute("SELECT source FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["source"] == "ytdlp"