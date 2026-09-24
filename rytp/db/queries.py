"""Parameterised SQL for the catalog tables.

Command handlers call these; they never build SQL themselves. Later plan
parts append their own sections (assets, jobs, words, utterances) — this
file is additive, so do not reorganise what is already here.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError, utc_now_iso

__all__ = [
    "asset_for",
    "assets_for",
    "clamp_limit",
    "find_channel",
    "get_channel",
    "get_setting",
    "get_video",
    "has_usable_asset",
    "insert_asset",
    "insert_channel",
    "list_channels",
    "list_videos",
    "mark_channel_synced",
    "prune_missing_assets",
    "set_setting",
    "upsert_video",
]


def clamp_limit(limit: int) -> int:
    """Keep a caller from paging the whole corpus into memory (design §10)."""
    return max(1, min(int(limit), C.MAX_LIST_LIMIT))


# -- channels ---------------------------------------------------------


def insert_channel(db: Database, *, url: str, title: str) -> tuple[int, bool]:
    """Register a channel by URL. Returns ``(channel_id, created)``."""
    with db.transaction():
        existing = db.conn.execute(
            "SELECT id FROM channels WHERE url = ?", (url,)
        ).fetchone()
        if existing is not None:
            db.conn.execute(
                "UPDATE channels SET title = ? WHERE id = ?", (title, existing["id"])
            )
            return int(existing["id"]), False
        cur = db.conn.execute(
            "INSERT INTO channels (url, title) VALUES (?, ?)", (url, title)
        )
        new_id = cur.lastrowid
        assert new_id is not None  # an INSERT always sets it
        return new_id, True


def get_channel(db: Database, channel_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT * FROM channels WHERE id = ?", (channel_id,)
    ).fetchone()


def find_channel(db: Database, ref: str) -> sqlite3.Row | None:
    """Look a channel up by URL, then by exact title."""
    row = db.conn.execute("SELECT * FROM channels WHERE url = ?", (ref,)).fetchone()
    if row is not None:
        return row
    return db.conn.execute("SELECT * FROM channels WHERE title = ?", (ref,)).fetchone()


def list_channels(db: Database, *, limit: int) -> list[sqlite3.Row]:
    """Channels newest first, each with the number of videos catalogued."""
    return db.conn.execute(
        """
        SELECT c.id, c.url, c.title, c.last_synced_at,
               (SELECT COUNT(*) FROM videos v WHERE v.channel_id = c.id) AS n_videos
        FROM channels c
        ORDER BY c.id DESC
        LIMIT ?
        """,
        (clamp_limit(limit),),
    ).fetchall()


def mark_channel_synced(db: Database, channel_id: int, when: str) -> None:
    db.conn.execute(
        "UPDATE channels SET last_synced_at = ? WHERE id = ?", (when, channel_id)
    )


# -- videos -----------------------------------------------------------


def upsert_video(
    db: Database,
    *,
    source: str,
    kind: str,
    channel_id: int | None,
    external_id: str,
    url: str | None,
    title: str,
    duration_ms: int | None,
    published_at: str | None,
    metadata_json: str = "{}",
) -> tuple[int, bool]:
    """Insert or refresh a catalog row. Returns ``(video_id, created)``.

    The natural key is ``(source, external_id)``. For a remote video that
    is the site's video id; for ``source='local'`` it is the resolved
    absolute path, which is what makes re-registering the same file
    idempotent — ``videos`` has no path column (design §4: "All path and
    download columns move out").

    ``created_at`` is written once and never touched again.
    """
    if not external_id:
        raise InvalidInputError(
            "a video needs an external_id: the site's video id, or the absolute "
            "path for a local file. Without one, UNIQUE (source, external_id) "
            "cannot dedupe and every add would create a new row."
        )
    with db.transaction():
        existing = db.conn.execute(
            "SELECT id FROM videos WHERE source = ? AND external_id = ?",
            (source, external_id),
        ).fetchone()
        if existing is not None:
            db.conn.execute(
                """
                UPDATE videos
                   SET kind = ?, channel_id = ?, url = ?, title = ?,
                       duration_ms = ?, published_at = ?, metadata_json = ?
                 WHERE id = ?
                """,
                (
                    kind,
                    channel_id,
                    url,
                    title,
                    duration_ms,
                    published_at,
                    metadata_json,
                    existing["id"],
                ),
            )
            return int(existing["id"]), False
        cur = db.conn.execute(
            """
            INSERT INTO videos (source, kind, channel_id, external_id, url, title,
                                duration_ms, published_at, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                kind,
                channel_id,
                external_id,
                url,
                title,
                duration_ms,
                published_at,
                metadata_json,
                utc_now_iso(),
            ),
        )
        new_id = cur.lastrowid
        assert new_id is not None  # an INSERT always sets it
        return new_id, True


def get_video(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def list_videos(
    db: Database,
    *,
    channel_id: int | None = None,
    kind: str | None = None,
    source: str | None = None,
    search: str | None = None,
    limit: int,
) -> list[sqlite3.Row]:
    """Catalog rows newest first, narrowed by whichever filters are given.

    ``search`` is a substring match on the title — deliberately dumb.
    Real search is the utterance FTS index (design §7); this only helps
    you find a video you half remember.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if channel_id is not None:
        clauses.append("v.channel_id = ?")
        params.append(channel_id)
    if kind is not None:
        clauses.append("v.kind = ?")
        params.append(kind)
    if source is not None:
        clauses.append("v.source = ?")
        params.append(source)
    if search:
        clauses.append("v.title LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(clamp_limit(limit))
    return db.conn.execute(
        f"""
        SELECT v.*, c.title AS channel_title
        FROM videos v
        LEFT JOIN channels c ON c.id = v.channel_id
        {where}
        ORDER BY v.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


# -- settings ---------------------------------------------------------


def get_setting(db: Database, key: str, default: str | None = None) -> str | None:
    """Read one operational value (design §5: everything tunable lives here)."""
    row = db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_setting(db: Database, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# ---------------------------------------------------------------------------
# Assets (design §4, contracts §3). Owned by Part 2.
# ---------------------------------------------------------------------------


def insert_asset(
    db: Database,
    *,
    video_id: int,
    role: str,
    path: str,
    format_id: str | None = None,
    size_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
    abr: float | None = None,
    acquired_at: str | None = None,
) -> int:
    """Record one asset, superseding whatever it replaces.

    The schema has no uniqueness constraint for it, so the design §4 rule —
    at most one audio, one captions and one container per video, any number
    of video renditions but only one per ``format_id`` — is enforced here,
    in the only function that writes the table.
    """
    superseded_sql: str
    superseded_params: tuple[Any, ...]
    if role in C.SINGLETON_ASSET_ROLES:
        superseded_sql = "DELETE FROM assets WHERE video_id = ? AND role = ?"
        superseded_params = (video_id, role)
    else:
        superseded_sql = (
            "DELETE FROM assets WHERE video_id = ? AND role = ? AND format_id IS ?"
        )
        superseded_params = (video_id, role, format_id)
    stamp = acquired_at or datetime.now(UTC).isoformat()
    with db.transaction():
        db.conn.execute(superseded_sql, superseded_params)
        cur = db.conn.execute(
            "INSERT INTO assets (video_id, role, format_id, path, bytes, width, "
            "height, abr, acquired_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, role, format_id, path, size_bytes, width, height, abr, stamp),
        )
        assert cur.lastrowid is not None  # an INSERT always sets it
        asset_id = int(cur.lastrowid)
    return asset_id


def assets_for(db: Database, video_id: int, role: str | None = None) -> list[sqlite3.Row]:
    """Every asset of a video, oldest first, optionally filtered by role."""
    if role is None:
        return list(
            db.conn.execute(
                "SELECT * FROM assets WHERE video_id = ? ORDER BY id", (video_id,)
            ).fetchall()
        )
    return list(
        db.conn.execute(
            "SELECT * FROM assets WHERE video_id = ? AND role = ? ORDER BY id",
            (video_id, role),
        ).fetchall()
    )


def asset_for(db: Database, video_id: int, role: str) -> sqlite3.Row | None:
    """The current asset of a role — the newest rendition, for ``video``."""
    return db.conn.execute(
        "SELECT * FROM assets WHERE video_id = ? AND role = ? ORDER BY id DESC LIMIT 1",
        (video_id, role),
    ).fetchone()


def has_usable_asset(db: Database, video_id: int, role: str) -> bool:
    """True when a row of this role exists *and* its file is still on disk.

    Readiness depends on this rather than on the row alone: deleting media
    outside the tool must make the download job runnable again (design §5).
    """
    return any(Path(row["path"]).exists() for row in assets_for(db, video_id, role))


def prune_missing_assets(db: Database, video_id: int) -> int:
    """Delete asset rows whose file has vanished. Returns the count."""
    doomed = [
        row["id"] for row in assets_for(db, video_id) if not Path(row["path"]).exists()
    ]
    if doomed:
        with db.transaction():
            db.conn.executemany(
                "DELETE FROM assets WHERE id = ?", [(i,) for i in doomed]
            )
    return len(doomed)
