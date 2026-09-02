"""Tests for the persistent download queue."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.download.queue import Queue


@pytest.fixture()
def queue(db: Database) -> Queue:
    return Queue(db)


def test_enqueue_idempotent(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """enqueue is idempotent: same video_id twice creates only one row."""
    video_id = db.upsert_video(fake_video_row)

    # Enqueue twice
    ids1 = queue.enqueue([video_id])
    ids2 = queue.enqueue([video_id])

    assert len(ids1) == 1
    assert len(ids2) == 0  # Second enqueue returns no new IDs

    # Verify only one row in queue_items
    count = db.conn.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0]
    assert count == 1


def test_enqueue_multiple_videos(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """Enqueue multiple distinct videos."""
    video_id1 = db.upsert_video({**fake_video_row, "youtube_id": "aaaaaaaaaaa", "title": "Video 1"})
    video_id2 = db.upsert_video({**fake_video_row, "youtube_id": "bbbbbbbbbbb", "title": "Video 2"})

    ids = queue.enqueue([video_id1, video_id2])

    assert len(ids) == 2
    count = db.conn.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0]
    assert count == 2


def test_claim_next_returns_oldest_pending(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """claim_next returns the oldest pending item."""
    video_id1 = db.upsert_video({**fake_video_row, "youtube_id": "aaaaaaaaaaa", "title": "Video 1"})
    video_id2 = db.upsert_video({**fake_video_row, "youtube_id": "bbbbbbbbbbb", "title": "Video 2"})

    queue.enqueue([video_id1, video_id2])

    first = queue.claim_next()
    assert first is not None
    assert first["video_id"] == video_id1
    assert first["status"] == "running"

    second = queue.claim_next()
    assert second is not None
    assert second["video_id"] == video_id2


def test_claim_next_empty_queue_returns_none(queue: Queue) -> None:
    """claim_next on empty queue returns None."""
    result = queue.claim_next()
    assert result is None


def test_mark_done(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """After mark_done, item is no longer pending."""
    video_id = db.upsert_video(fake_video_row)
    queue_id = queue.enqueue([video_id])[0]

    queue.mark_done(queue_id)

    row = db.conn.execute("SELECT status FROM queue_items WHERE id = ?", (queue_id,)).fetchone()
    assert row["status"] == "done"

    # claim_next should return None now
    assert queue.claim_next() is None


def test_mark_failed_records_error(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """mark_failed records the error in last_error and increments attempts."""
    video_id = db.upsert_video(fake_video_row)
    queue_id = queue.enqueue([video_id])[0]

    queue.mark_failed(queue_id, "Network error")

    row = db.conn.execute(
        "SELECT status, last_error, attempts FROM queue_items WHERE id = ?", (queue_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert row["last_error"] == "Network error"
    assert row["attempts"] == 1


def test_pause_resume_flips_flag(queue: Queue, db: Database) -> None:
    """pause() and resume() flip settings.queue_paused."""
    assert queue.is_paused() is False

    queue.pause()
    assert queue.is_paused() is True

    # Verify persisted in settings table
    val = db.get_setting("queue_paused")
    assert val == "1"

    queue.resume()
    assert queue.is_paused() is False

    val = db.get_setting("queue_paused")
    assert val == "0"


def test_pause_resume_persists_across_reopen(db: Database, fake_video_row: dict[str, object]) -> None:
    """Queue pause state survives closing and reopening the database."""
    queue = Queue(db)
    queue.pause()
    db.close()

    # Reopen
    db2 = Database(db._path)
    queue2 = Queue(db2)
    assert queue2.is_paused() is True
    db2.close()


def test_stats_returns_correct_counts(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """stats() returns correct counts after operations."""
    video_id1 = db.upsert_video({**fake_video_row, "youtube_id": "aaaaaaaaaaa", "title": "Video 1"})
    video_id2 = db.upsert_video({**fake_video_row, "youtube_id": "bbbbbbbbbbb", "title": "Video 2"})
    video_id3 = db.upsert_video({**fake_video_row, "youtube_id": "ccccccccccc", "title": "Video 3"})

    queue.enqueue([video_id1, video_id2, video_id3])

    stats = queue.stats()
    assert stats["pending"] == 3

    q1 = queue.claim_next()
    assert q1 is not None
    stats = queue.stats()
    assert stats["pending"] == 2
    assert stats["running"] == 1

    queue.mark_done(q1["id"])
    stats = queue.stats()
    assert stats["pending"] == 2
    assert stats["running"] == 0
    assert stats["done"] == 1

    q2 = queue.claim_next()
    assert q2 is not None
    queue.mark_failed(q2["id"], "Error")
    stats = queue.stats()
    assert stats["pending"] == 1
    assert stats["failed"] == 1


def test_list_pending(queue: Queue, db: Database, fake_video_row: dict[str, object]) -> None:
    """list_pending returns pending items in order."""
    video_id1 = db.upsert_video({**fake_video_row, "youtube_id": "aaaaaaaaaaa", "title": "Video 1"})
    video_id2 = db.upsert_video({**fake_video_row, "youtube_id": "bbbbbbbbbbb", "title": "Video 2"})

    queue.enqueue([video_id1, video_id2])

    pending = queue.list_pending(limit=100)
    assert len(pending) == 2
    assert pending[0]["video_id"] == video_id1
    assert pending[1]["video_id"] == video_id2

    queue.claim_next()
    pending = queue.list_pending(limit=100)
    assert len(pending) == 1
    assert pending[0]["video_id"] == video_id2