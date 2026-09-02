"""Higher-level channel and video orchestration.

Wraps :mod:`rytp.download.ytdlp` with the database calls so the CLI
layer can stay thin. The split is intentional: ``ytdlp.py`` is a
pluggable transport (tested with a fake runner), and this module is
the policy that decides how the transport's output maps onto the
``channels`` / ``videos`` tables.

Public surface:

* :func:`add_channel` — register a YouTube (or yt-dlp-compatible)
  channel.
* :func:`sync_channel` — re-list the channel and upsert all videos.
* :func:`list_videos` — browse the videos table with optional filters.
* :func:`register_video` — register a single URL or local file path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.download.ytdlp import (
    ChannelVideo,
    DownloadResult,
    VideoMetadata,
    YtDlpRunner,
)
from rytp.models import Video


def add_channel(db: Database, runner: YtDlpRunner, url: str, title: str | None = None) -> int:
    """Register a channel in the ``channels`` table.

    If ``title`` is not given, the runner is asked to probe the
    channel and its title is used; if the probe fails (rate-limited,
    private, etc.), the URL itself becomes the title as a fallback.

    Returns:
        Channel row id.
    """
    if title is None:
        # Try to get title from the channel
        try:
            info = runner.probe(url)
            title = info.title
        except Exception:
            title = url
    cursor = db.conn.execute(
        "INSERT OR REPLACE INTO channels (url, title) VALUES (?, ?)",
        (url, title),
    )
    db.conn.commit()
    return cursor.lastrowid


def sync_channel(db: Database, runner: YtDlpRunner, channel_id_or_name: int | str) -> int:
    """Re-list the channel and upsert every video into the ``videos`` table.

    The channel can be identified by its row id (int) or by its
    title/URL (str); the lookup tries title first, then the URL as a
    fallback.

    Idempotent: re-running with an unchanged channel returns ``0``
    (no new videos). Only videos new to the DB count toward the
    returned value; updates to existing videos (e.g. title spelling
    fixes) don't count.

    Returns:
        Number of *newly inserted* video rows.

    Raises:
        ValueError: if the channel id/name doesn't resolve.
    """
    # Look up the channel
    if isinstance(channel_id_or_name, int):
        row = db.conn.execute(
            "SELECT id, url FROM channels WHERE id = ?", (channel_id_or_name,)
        ).fetchone()
    else:
        row = db.conn.execute(
            "SELECT id, url FROM channels WHERE title = ? OR url = ?",
            (channel_id_or_name, channel_id_or_name),
        ).fetchone()

    if not row:
        raise ValueError(f"Channel not found: {channel_id_or_name}")

    channel_id = row["id"]
    channel_url = row["url"]

    # List videos from the channel
    videos = runner.list_channel(channel_url)

    # Get existing video IDs before sync
    before_ids = set(
        r["id"]
        for r in db.conn.execute(
            "SELECT id FROM videos WHERE channel_id = ?", (channel_id,)
        ).fetchall()
    )

    # Upsert each video
    for v in videos:
        row = _channel_video_to_row(v, channel_id)
        db.upsert_video(row)

    # Update last_synced_at
    from datetime import datetime
    db.conn.execute(
        "UPDATE channels SET last_synced_at = ? WHERE id = ?",
        (datetime.utcnow().isoformat(), channel_id),
    )
    db.conn.commit()

    # Get new video IDs
    after_ids = set(
        r["id"]
        for r in db.conn.execute(
            "SELECT id FROM videos WHERE channel_id = ?", (channel_id,)
        ).fetchall()
    )

    return len(after_ids - before_ids)


def list_videos(
    db: Database,
    *,
    channel_id: int | None = None,
    kind: str | None = None,
    source: str | None = None,
) -> list[Video]:
    """Browse the ``videos`` table with optional filters.

    All filters are AND-combined and None-passes-through:

    * ``channel_id`` — restrict to a single channel.
    * ``kind`` — ``"video"``, ``"short"``, ``"livestream"``, or
      ``"other"``.
    * ``source`` — ``"youtube"``, ``"ytdlp"``, or ``"local"``.

    Returns :class:`rytp.models.Video` dataclasses (parsed from
    ``sqlite3.Row`` via :meth:`Video.from_row`).
    """
    where_clauses = []
    params: list[Any] = []

    if channel_id is not None:
        where_clauses.append("channel_id = ?")
        params.append(channel_id)
    if kind is not None:
        where_clauses.append("kind = ?")
        params.append(kind)
    if source is not None:
        where_clauses.append("source = ?")
        params.append(source)

    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    sql = f"SELECT * FROM videos{where_sql} ORDER BY id DESC"
    rows = db.conn.execute(sql, params).fetchall()
    return [Video.from_row(row) for row in rows]


def register_video(
    db: Database,
    runner: YtDlpRunner,
    url_or_path: str,
) -> int:
    """Register a single video: URL, any yt-dlp URL, or local file path.

    Local paths (anything that looks like a path or exists on disk)
    skip the probe and are inserted with ``source='local'``,
    ``downloaded=True``, and ``local_path`` pointing at the resolved
    absolute path. URLs go through ``runner.probe`` to extract
    metadata; the ``source`` is ``"youtube"`` if the URL is a
    YouTube URL (and ``youtube_id`` is set from the probe or the
    URL), otherwise ``"ytdlp"``.
    """
    # Check if it looks like a local file path
    if _is_local_path(url_or_path):
        # Local file
        path = Path(url_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {url_or_path}")

        row: dict[str, Any] = {
            "source": "local",
            "kind": "video",
            "channel_id": None,
            "youtube_id": None,
            "url": None,
            "local_path": str(path.resolve()),
            "title": path.stem,
            "duration": None,
            "published_at": None,
            "downloaded": True,
            "downloaded_path": str(path.resolve()),
            "metadata_json": "{}",
        }
        return db.upsert_video(row)

    # Remote URL - probe for metadata
    metadata = runner.probe(url_or_path)

    # Extract youtube_id from URL if not in metadata
    youtube_id = metadata.youtube_id
    if not youtube_id:
        youtube_id = _extract_youtube_id(url_or_path)

    source = "youtube" if youtube_id and _is_youtube_url(url_or_path) else "ytdlp"

    row = {
        "source": source,
        "kind": metadata.kind,
        "channel_id": None,
        "youtube_id": youtube_id,
        "url": url_or_path,
        "local_path": None,
        "title": metadata.title,
        "duration": metadata.duration,
        "published_at": metadata.published_at,
        "downloaded": False,
        "downloaded_path": None,
        "metadata_json": "{}",
    }
    return db.upsert_video(row)


def _is_local_path(s: str) -> bool:
    """Check if a string looks like a local file path."""
    # Windows absolute path
    if re.match(r"^[A-Za-z]:[\\/]", s):
        return True
    # Unix absolute path
    if s.startswith("/") or s.startswith("./") or s.startswith("../"):
        return True
    # Path exists check
    if os.path.exists(s):
        return True
    return False


def _is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube URL."""
    return "youtube.com" in url or "youtu.be" in url


def _extract_youtube_id(url: str) -> str | None:
    """Extract a YouTube video id from various URL formats.

    YouTube ids are always exactly :data:`C.YOUTUBE_ID_LENGTH`
    characters long, but URLs sometimes append extra junk
    (``?t=42``, ``&list=...``), so we trim the captured group.
    """
    # youtu.be/xxx
    match = re.search(r"youtu\.be/([^/?&]+)", url)
    if match:
        return match.group(1)[: C.YOUTUBE_ID_LENGTH]
    # youtube.com/watch?v=xxx
    match = re.search(r"[?&]v=([^&]+)", url)
    if match:
        return match.group(1)[: C.YOUTUBE_ID_LENGTH]
    # youtube.com/embed/xxx
    match = re.search(r"embed/([^/?&]+)", url)
    if match:
        return match.group(1)[: C.YOUTUBE_ID_LENGTH]
    # youtube.com/shorts/xxx
    match = re.search(r"shorts/([^/?&]+)", url)
    if match:
        return match.group(1)[: C.YOUTUBE_ID_LENGTH]
    return None


def _channel_video_to_row(v: ChannelVideo, channel_id: int) -> dict[str, Any]:
    """Convert a ChannelVideo to a videos table row dict."""
    return {
        "source": "youtube",
        "kind": v.kind,
        "channel_id": channel_id,
        "youtube_id": v.youtube_id,
        "url": v.url,
        "local_path": None,
        "title": v.title,
        "duration": v.duration,
        "published_at": v.published_at,
        "downloaded": False,
        "downloaded_path": None,
        "metadata_json": "{}",
    }