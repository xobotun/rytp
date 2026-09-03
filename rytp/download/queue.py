"""Persistent download queue backed by the queue_items table.

State lives entirely in the ``queue_items`` table — there's no
in-memory queue, so a worker crash doesn't lose pending downloads
or duplicate work.

The pause/resume flag lives in the ``settings`` table under the key
``queue_paused``, which the worker reads on every claim. This makes
pause/resume an instant, lock-free operation.

Public surface:

* :class:`Queue` — wrapper around the ``queue_items`` table.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from rytp import constants as C
from rytp.db import Database


class Queue:
    """Wrapper around the queue_items table.

    One Queue per process — call Queue(db) to bind it to a Database.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def enqueue(self, video_ids: Iterable[int]) -> list[int]:
        """Add video IDs to the queue. Idempotent: skips already pending/running."""
        enqueued: list[int] = []
        now = datetime.now(UTC).isoformat()

        for video_id in video_ids:
            # Check if already pending or running
            existing = self._db.conn.execute(
                "SELECT id FROM queue_items WHERE video_id = ? AND status IN ('pending', 'running')",
                (video_id,),
            ).fetchone()
            if existing:
                continue

            cursor = self._db.conn.execute(
                """
                INSERT INTO queue_items (video_id, status, attempts, enqueued_at)
                VALUES (?, 'pending', 0, ?)
                """,
                (video_id, now),
            )
            enqueued.append(cursor.lastrowid)
        self._db.conn.commit()
        return enqueued

    def list_pending(self, limit: int = C.QUEUE_LIST_LIMIT) -> list[sqlite3.Row]:
        """List pending queue items, oldest first."""
        return self._db.conn.execute(
            "SELECT * FROM queue_items WHERE status = 'pending' ORDER BY enqueued_at LIMIT ?",
            (limit,),
        ).fetchall()

    def claim_next(self) -> sqlite3.Row | None:
        """Atomically claim the next pending item (status ``pending`` → ``running``).

        Uses SQLite's ``RETURNING`` clause when available (3.35+) and
        falls back to a two-step ``SELECT`` + ``UPDATE`` with a
        rowcount guard for older builds. The two-step approach is
        safe under contention: the ``UPDATE ... WHERE status='pending'``
        only affects rows still in the pending state, so a worker that
        lost the race gets ``rowcount == 0`` and returns ``None``.

        Returns:
            The claimed ``queue_items`` row (with ``started_at``
            populated), or ``None`` if the queue is empty.
        """
        now = datetime.now(UTC).isoformat()

        # Try RETURNING clause (SQLite 3.35+)
        try:
            row = self._db.conn.execute(
                """
                UPDATE queue_items
                SET status = 'running', started_at = ?
                WHERE id = (
                    SELECT id FROM queue_items
                    WHERE status = 'pending'
                    ORDER BY enqueued_at
                    LIMIT 1
                )
                RETURNING *
                """,
                (now,),
            ).fetchone()
            if row:
                self._db.conn.commit()
                return row
        except sqlite3.OperationalError:
            # RETURNING not supported; fall back to two-step
            pass

        # Fallback: select then update with rowcount check
        row = self._db.conn.execute(
            "SELECT id FROM queue_items WHERE status = 'pending' ORDER BY enqueued_at LIMIT 1"
        ).fetchone()
        if not row:
            return None

        item_id = row["id"]
        cursor = self._db.conn.execute(
            "UPDATE queue_items SET status = 'running', started_at = ? WHERE id = ? AND status = 'pending'",
            (now, item_id),
        )
        if cursor.rowcount == 0:
            self._db.conn.commit()
            return None  # Lost the race

        self._db.conn.commit()
        return self._db.conn.execute(
            "SELECT * FROM queue_items WHERE id = ?", (item_id,)
        ).fetchone()

    def mark_done(self, queue_item_id: int) -> None:
        """Mark a queue item as done."""
        now = datetime.now(UTC).isoformat()
        self._db.conn.execute(
            "UPDATE queue_items SET status = 'done', finished_at = ? WHERE id = ?",
            (now, queue_item_id),
        )
        self._db.conn.commit()

    def mark_failed(self, queue_item_id: int, error: str) -> None:
        """Mark a queue item as failed and record the error."""
        now = datetime.now(UTC).isoformat()
        self._db.conn.execute(
            "UPDATE queue_items SET status = 'failed', finished_at = ?, last_error = ?, attempts = attempts + 1 WHERE id = ?",
            (now, error, queue_item_id),
        )
        self._db.conn.commit()

    def pause(self) -> None:
        """Pause the queue by setting the global flag."""
        self._db.set_setting("queue_paused", "1")

    def resume(self) -> None:
        """Resume the queue by clearing the global flag."""
        self._db.set_setting("queue_paused", "0")

    def is_paused(self) -> bool:
        """Check if the queue is paused."""
        val = self._db.get_setting("queue_paused", "0")
        return val == "1"

    def stats(self) -> dict[str, int]:
        """Return counts by status."""
        rows = self._db.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM queue_items GROUP BY status"
        ).fetchall()
        result = {"pending": 0, "running": 0, "done": 0, "failed": 0, "paused": 0}
        for row in rows:
            result[row["status"]] = row["cnt"]
        return result