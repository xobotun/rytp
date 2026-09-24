"""Readiness predicates. design §5.

Every predicate answers from the database and the filesystem as they are
*now*. None of them look at the job row, and in particular none of them look
at ``payload_json`` — a predicate that depended on how a job was enqueued
would stop being a statement about the world and the self-healing property
would go with it.
"""

from __future__ import annotations

import sqlite3

from rytp import constants as C
from rytp.config import paths
from rytp.db import Database
from rytp.db.queries import has_usable_asset
from rytp.jobs import Readiness


def _video(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT id, source, url FROM videos WHERE id = ?", (video_id,)
    ).fetchone()


def _downloadable(row: sqlite3.Row | None) -> bool:
    """A remote, catalogued video we are allowed to fetch from."""
    return row is not None and row["source"] != C.LOCAL_SOURCE and bool(row["url"])


def download_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable when the video is remote and its media is not on disk.

    Satisfied needs *both* the audio and a video rendition: design §4 keeps
    them as separate assets so a rendition can be upgraded later, and a
    half-finished download must not look complete.
    """
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if has_usable_asset(db, video_id, "audio") and has_usable_asset(db, video_id, "video"):
        return Readiness.SATISFIED
    if not _downloadable(row):
        return Readiness.BLOCKED
    return Readiness.READY


def captions_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable for every remote video. design §6: captions always, early."""
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if has_usable_asset(db, video_id, "captions"):
        return Readiness.SATISFIED
    if not _downloadable(row):
        return Readiness.BLOCKED
    return Readiness.READY


def extract_wav_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once there is something to decode from.

    The cached WAV is not recorded in any table (contracts §7), so its
    presence on disk is the whole answer — delete it and this goes straight
    back to READY.
    """
    row = _video(db, video_id)
    if row is None:
        return Readiness.BLOCKED
    if paths().cache_wav(video_id).exists():
        return Readiness.SATISFIED
    if has_usable_asset(db, video_id, "audio") or has_usable_asset(db, video_id, "container"):
        return Readiness.READY
    return Readiness.BLOCKED
