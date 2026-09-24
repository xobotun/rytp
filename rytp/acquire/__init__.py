"""Acquisition: turning a catalogued video into assets on disk. design §5.

The stage entry points live here; the yt-dlp glue is in
:mod:`rytp.acquire.ytdlp` and the politeness rules in
:mod:`rytp.acquire.policy`.

design §4 is the rule this module exists to keep: **audio and video are
downloaded together and kept as separate files.** yt-dlp fetches them
separately anyway, and not merging them is what makes a later 360p -> 1080p
rendition upgrade free — the audio a transcript is aligned to is never
touched.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import (
    AcquireError,
    DownloadPolicy,
    VideoUnavailable,
)
from rytp.acquire.ytdlp import (
    RealYtDlpRunner,
    YtDlpRunner,
    translate_error,
)
from rytp.db import Database
from rytp.db.queries import insert_asset, prune_missing_assets

__all__ = ["acquire_media", "fetchable_video"]


def fetchable_video(db: Database, video_id: int) -> sqlite3.Row:
    """The catalog row, or a one-line explanation of why we cannot fetch it."""
    row = db.conn.execute(
        "SELECT id, source, url, title FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise AcquireError(f"video {video_id} is not in the catalog")
    if row["source"] == C.LOCAL_SOURCE:
        raise AcquireError(
            f"video {video_id} is a local file; local files register as a "
            f"container asset and never get download jobs"
        )
    if not row["url"]:
        raise AcquireError(f"video {video_id} has no url to fetch from")
    return row


def _place(src: Path, dst: Path) -> Path:
    """Move a downloaded file to its contracted name. Same directory, so
    ``os.replace`` is atomic on Windows as well as POSIX."""
    if src != dst:
        os.replace(src, dst)
    return dst


def acquire_media(
    db: Database,
    video_id: int,
    *,
    runner: YtDlpRunner | None = None,
    policy: DownloadPolicy | None = None,
) -> str:
    """Fetch the audio and one video rendition as two separate assets."""
    row = fetchable_video(db, video_id)
    policy = policy or DownloadPolicy.from_settings(db)
    runner = runner or RealYtDlpRunner()

    prune_missing_assets(db, video_id)
    out_dir = config.ensure_dir(config.paths().media_dir(video_id))
    selector = f"{policy.format_audio},{policy.format_video}"

    try:
        files = runner.download(
            row["url"],
            out_template=str(out_dir / "%(format_id)s.%(ext)s"),
            format_selector=selector,
            rate_limit_bps=policy.rate_limit_bps,
            sleep_requests_s=policy.sleep_requests_s,
        )
    except Exception as exc:
        raise translate_error(exc) from exc

    audio = next((f for f in files if f.has_audio and not f.has_video), None)
    video = next((f for f in files if f.has_video), None)

    if audio is None:
        raise AcquireError(
            f"video {video_id}: no audio-only format came back from "
            f"selector {policy.format_audio!r}"
        )
    audio_path = _place(audio.path, out_dir / f"audio.{audio.ext}")
    insert_asset(
        db,
        video_id=video_id,
        role="audio",
        path=str(audio_path),
        format_id=audio.format_id,
        size_bytes=audio.size_bytes or audio_path.stat().st_size,
        abr=audio.abr,
    )

    if video is None:
        # The audio is already recorded — re-fetching it later would be a
        # second pointless request. The job fails permanently and shows up
        # in `rytp jobs list --state failed`.
        raise VideoUnavailable(
            f"video {video_id}: audio saved, but no video rendition came back "
            f"from selector {policy.format_video!r}"
        )
    video_path = _place(video.path, out_dir / f"video-{video.format_id}.{video.ext}")
    insert_asset(
        db,
        video_id=video_id,
        role="video",
        path=str(video_path),
        format_id=video.format_id,
        size_bytes=video.size_bytes or video_path.stat().st_size,
        width=video.width,
        height=video.height,
    )
    return f"video {video_id}: {audio_path.name} + {video_path.name}"
