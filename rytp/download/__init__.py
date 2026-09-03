"""Top-level helpers for the download stage.

The actual yt-dlp glue lives in :mod:`rytp.download.ytdlp`. The
helpers here are the things the CLI cares about: a single
``download_one`` function that takes a video row (or a fresh URL),
runs the yt-dlp download, and records the result in the ``videos``
table.

Used by:

* ``rytp download <video-id-or-url>`` — direct CLI invocation.
* ``rytp queue worker`` — the long-running queue consumer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.config import paths as config_paths
from rytp.db import Database
from rytp.download.ytdlp import (
    RealYtDlpRunner,
    VideoMetadata,
    YtDlpRunner,
)


# Default yt-dlp format selector, from the user's sample run.
DEFAULT_FORMAT_SELECTOR = "worstvideo[height=720]+bestaudio[language=ru]"


def _resolve_url_and_id(
    db: Database, video_id_or_url: str | int
) -> tuple[int, str]:
    """Look up a video by id (int / numeric str) or by ``url`` (str).

    Returns ``(video_id, url)`` for an existing video, or raises
    :class:`ValueError` if no row matches. URLs that don't match an
    existing video raise the same error — the canonical entry point
    for *new* videos is :func:`rytp.channels.register_video` followed
    by ``rytp download <video_id>``.
    """
    if isinstance(video_id_or_url, int) or (
        isinstance(video_id_or_url, str) and video_id_or_url.isdigit()
    ):
        vid = int(video_id_or_url)
        row = db.conn.execute(
            "SELECT id, url, local_path FROM videos WHERE id = ?", (vid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"video_id {vid} not found in videos table")
        if not row["url"]:
            raise ValueError(
                f"video_id {vid} has no URL — register the URL with "
                "`rytp videos add <url>` first"
            )
        return row["id"], row["url"]

    raise ValueError(
        f"expected a video id (int) or an existing video's URL, got "
        f"{video_id_or_url!r}. To download a brand-new URL, run "
        f"`rytp videos add <url>` first."
    )


def download_one(
    db: Database,
    video_id_or_url: str | int,
    *,
    out_dir: Path | None = None,
    runner: YtDlpRunner | None = None,
    format_selector: str = DEFAULT_FORMAT_SELECTOR,
    resume: bool = True,
) -> int:
    """Download one video and update the ``videos`` row in place.

    The download lands under ``out_dir`` (default
    ``config.paths.media``). After the download, the ``videos`` row
    is updated to set ``downloaded = 1`` and ``downloaded_path`` to
    the file yt-dlp produced. If audio was downloaded separately,
    ``downloaded_audio_path`` is also set.

    Args:
        db: Open Database.
        video_id_or_url: Either an int (or numeric str) ``videos.id``
            or the URL of an existing video row.
        out_dir: Where the media file is written. Defaults to
            ``config.paths.media``.
        runner: Optional :class:`YtDlpRunner` (test seam). Defaults
            to :class:`RealYtDlpRunner` which requires
            ``pip install rytp[yt-dlp]``.
        format_selector: yt-dlp format string. Default
            :data:`DEFAULT_FORMAT_SELECTOR` matches the user's
            sample run.
        resume: Whether to attempt to resume a partial download
            (default True). yt-dlp's ``continuedl`` is on by default
            anyway; this flag only controls the pre-existence check
            that sets ``DownloadResult.was_resumed``.

    Returns:
        The ``videos.id`` of the updated row.

    Raises:
        ImportError: when yt-dlp isn't installed and no runner was
            supplied.
        ValueError: when the video id / URL doesn't resolve.
        RuntimeError: when yt-dlp runs but produces no output file.
    """
    video_id, url = _resolve_url_and_id(db, video_id_or_url)
    out_dir = out_dir or config_paths.media
    out_dir.mkdir(parents=True, exist_ok=True)
    if runner is None:
        runner = RealYtDlpRunner()

    result = runner.download(
        url, out_dir, format_selector=format_selector, resume=resume
    )
    if not result.path.exists():
        raise RuntimeError(
            f"yt-dlp reported success but {result.path} is missing"
        )

    # Update the videos row in place.
    db.conn.execute(
        """
        UPDATE videos
        SET downloaded = 1,
            downloaded_path = ?,
            downloaded_audio_path = ?,
            title = COALESCE(?, title),
            duration = COALESCE(?, duration)
        WHERE id = ?
        """,
        (str(result.path), str(result.audio_path) if result.audio_path else None, result.title, result.duration, video_id),
    )
    db.conn.commit()
    return video_id


def run_queued_item(
    db: Database,
    queue_item_id: int,
    *,
    runner: YtDlpRunner | None = None,
) -> tuple[bool, str]:
    """Run one queue item: look up its video and call :func:`download_one`.

    Returns ``(ok, message)`` for the worker's outer loop. On
    success the queue item is marked done; on failure it's marked
    failed with the error string.
    """
    from rytp.download.queue import Queue

    row = db.conn.execute(
        "SELECT video_id FROM queue_items WHERE id = ?", (queue_item_id,)
    ).fetchone()
    if row is None:
        return False, f"queue_items.id={queue_item_id} not found"

    try:
        download_one(db, row["video_id"], runner=runner)
    except (ImportError, ValueError, RuntimeError) as e:
        Queue(db).mark_failed(queue_item_id, str(e))
        return False, str(e)
    Queue(db).mark_done(queue_item_id)
    return True, f"video {row['video_id']} downloaded"
