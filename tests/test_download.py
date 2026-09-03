"""Tests for the top-level ``rytp.download`` module (download_one, run_queued_item)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.download import download_one, run_queued_item
from rytp.download.queue import Queue
from rytp.download.ytdlp import DownloadResult, YtDlpRunner


class _FakeRunner:
    """Minimal YtDlpRunner for tests; records calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls: list[dict[str, Any]] = []

    def download(  # type: ignore[override]
        self,
        url: str,
        out_dir: Path,
        *,
        format_selector: str,
        resume: bool,
    ) -> DownloadResult:
        self.calls.append(
            dict(
                url=url,
                out_dir=out_dir,
                format_selector=format_selector,
                resume=resume,
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "fake.mp4"
        target.write_bytes(b"\x00" * 8)
        return DownloadResult(
            path=target,
            youtube_id="abc",
            title="Test Title",
            duration=42,
            was_resumed=False,
        )


class _MissingFileRunner(_FakeRunner):
    """Returns a DownloadResult pointing at a file that doesn't exist."""

    def download(self, url, out_dir, *, format_selector, resume):  # type: ignore[override]
        return DownloadResult(
            path=out_dir / "does_not_exist.mp4",
            youtube_id="x",
            title="t",
            duration=1,
            was_resumed=False,
        )


def test_download_one_updates_row(db: Database, data_dir: Path) -> None:
    """End-to-end: register a video with a URL, download with a fake runner,
    and verify the videos row reflects the result."""
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, youtube_id, url, title, duration)
        VALUES ('youtube', 'video', 'abc', 'https://example.com/abc', 'orig', 0)
        """
    )
    vid = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    db.conn.commit()

    runner = _FakeRunner(Path("ignored"))
    returned = download_one(
        db, vid, runner=runner, format_selector="best", resume=False
    )
    assert returned == vid
    row = db.conn.execute(
        "SELECT downloaded, downloaded_path, title, duration FROM videos WHERE id = ?",
        (vid,),
    ).fetchone()
    assert row["downloaded"] == 1
    assert row["downloaded_path"].endswith("fake.mp4")
    # COALESCE(title, title) preserves the new title from the runner.
    assert row["title"] == "Test Title"
    assert row["duration"] == 42
    assert len(runner.calls) == 1
    assert runner.calls[0]["url"] == "https://example.com/abc"
    assert runner.calls[0]["resume"] is False
    # data_dir is the test sandbox root; download should have created
    # the media dir.
    assert Path(row["downloaded_path"]).exists()


def test_download_one_unknown_id_raises(db: Database) -> None:
    runner = _FakeRunner(Path("ignored"))
    with pytest.raises(ValueError, match="not found"):
        download_one(db, 9999, runner=runner)


def test_download_one_no_url_raises(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO videos (source, kind, local_path, title) "
        "VALUES ('local', 'video', 'C:/x.mp4', 't')"
    )
    db.conn.commit()
    runner = _FakeRunner(Path("ignored"))
    with pytest.raises(ValueError, match="no URL"):
        download_one(db, 1, runner=runner)


def test_download_one_missing_output_file_raises(db: Database) -> None:
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, youtube_id, url, title)
        VALUES ('youtube', 'video', 'x', 'https://example.com/x', 't')
        """
    )
    db.conn.commit()
    runner = _MissingFileRunner(Path("ignored"))
    with pytest.raises(RuntimeError, match="missing"):
        download_one(db, 1, runner=runner)


def test_run_queued_item_marks_done_on_success(db: Database) -> None:
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, youtube_id, url, title)
        VALUES ('youtube', 'video', 'q', 'https://example.com/q', 't')
        """
    )
    vid = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    qids = Queue(db).enqueue([vid])
    db.conn.commit()

    runner = _FakeRunner(Path("ignored"))
    ok, msg = run_queued_item(db, qids[0], runner=runner)
    assert ok is True
    assert "downloaded" in msg
    status = db.conn.execute(
        "SELECT status FROM queue_items WHERE id = ?", (qids[0],)
    ).fetchone()["status"]
    assert status == "done"


def test_run_queued_item_marks_failed_on_error(db: Database) -> None:
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, youtube_id, url, title)
        VALUES ('youtube', 'video', 'q', 'https://example.com/q', 't')
        """
    )
    vid = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    qids = Queue(db).enqueue([vid])
    db.conn.commit()

    runner = _MissingFileRunner(Path("ignored"))
    ok, msg = run_queued_item(db, qids[0], runner=runner)
    assert ok is False
    assert "missing" in msg.lower()
    row = db.conn.execute(
        "SELECT status, last_error FROM queue_items WHERE id = ?", (qids[0],)
    ).fetchone()
    assert row["status"] == "failed"
    assert "missing" in row["last_error"].lower()


def test_download_one_stores_separate_audio_path(db: Database) -> None:
    """When the runner returns a separate audio_path, it should be stored."""
    db.conn.execute(
        """
        INSERT INTO videos (source, kind, youtube_id, url, title, duration)
        VALUES ('youtube', 'video', 'sep', 'https://example.com/sep', 'orig', 0)
        """
    )
    vid = db.conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    db.conn.commit()

    class _SeparateAudioRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def download(  # type: ignore[override]
            self,
            url: str,
            out_dir: Path,
            *,
            format_selector: str,
            resume: bool,
        ) -> DownloadResult:
            out_dir.mkdir(parents=True, exist_ok=True)
            video_file = out_dir / "merged.mp4"
            audio_file = out_dir / "audio.webm"
            video_file.write_bytes(b"\x00" * 8)
            audio_file.write_bytes(b"\x00" * 8)
            return DownloadResult(
                path=video_file,
                youtube_id="sep",
                title="Test Title",
                duration=42,
                was_resumed=False,
                audio_path=audio_file,
            )

    runner = _SeparateAudioRunner()
    returned = download_one(db, vid, runner=runner, format_selector="best", resume=False)
    assert returned == vid
    row = db.conn.execute(
        "SELECT downloaded, downloaded_path, downloaded_audio_path FROM videos WHERE id = ?",
        (vid,),
    ).fetchone()
    assert row["downloaded"] == 1
    assert row["downloaded_path"].endswith("merged.mp4")
    assert row["downloaded_audio_path"].endswith("audio.webm")
