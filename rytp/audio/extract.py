"""The regenerable WAV cache. design §4, contracts §7.

``cache/wav/{video_id}.wav`` is 16 kHz mono int16 and is **not recorded in
any table**. Its presence on disk is the whole truth about it, which is what
lets ``rytp cache prune`` free gigabytes and have the extract jobs quietly
become runnable again (design §5).

ffmpeg writes to a ``.part`` sibling that is moved into place only on
success, so a worker killed mid-decode never leaves a truncated file that
looks like a valid cache entry.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.acquire.policy import AcquireError, MissingAssetError
from rytp.db import Database
from rytp.db.queries import asset_for


class FfmpegNotFoundError(AcquireError):
    """ffmpeg is not on PATH."""


class ExtractionError(AcquireError):
    """ffmpeg returned non-zero, or produced nothing."""


def wav_path(video_id: int) -> Path:
    """Where this video's cached WAV lives. Contracts §7."""
    return config.paths().cache_wav(video_id)


def source_asset(db: Database, video_id: int) -> sqlite3.Row:
    """The best thing to decode from: the audio asset, else the container."""
    for role in ("audio", "container"):
        row = asset_for(db, video_id, role)
        if row is not None and Path(row["path"]).exists():
            return row
    raise MissingAssetError(
        f"video {video_id}: no audio or container asset on disk to extract from"
    )


def _ffmpeg_binary() -> str | None:
    """Where ffmpeg is. A module attribute, so a test can replace just this
    rather than reaching into the global ``shutil``."""
    return shutil.which("ffmpeg")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """The single subprocess call, isolated so tests can replace it."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def ensure_wav(db: Database, video_id: int, *, overwrite: bool = False) -> Path:
    """Decode the cached 16 kHz mono WAV, unless it is already there."""
    out = wav_path(video_id)
    if out.exists() and not overwrite:
        return out

    row = source_asset(db, video_id)
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/ "
            "(on Windows, unzip the release and add its bin/ directory to PATH)"
        )

    config.ensure_dir(out.parent)
    partial = out.with_suffix(".wav.part")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i", row["path"],
        "-vn",                      # audio only, even from a container
        "-map", "0:a:0",            # the first audio stream, explicitly
        "-ac", str(C.AUDIO_CHANNELS),
        "-ar", str(C.AUDIO_SAMPLE_RATE_HZ),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(partial),
    ]
    result = _run_ffmpeg(cmd)
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise ExtractionError(
            f"video {video_id}: ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )
    os.replace(partial, out)
    return out


def prune_wav_cache(
    db: Database, *, video_id: int | None = None, dry_run: bool = False
) -> list[tuple[Path, int]]:
    """Delete cached WAVs and report what was freed.

    Safe by construction: the cache is regenerable and nothing references it
    by path, so the worst case is that some extract jobs run again.
    """
    del db  # the cache is filesystem-only on purpose
    if video_id is not None:
        candidates = [wav_path(video_id)]
    else:
        cache_dir = config.paths().cache_wav(0).parent
        candidates = sorted(cache_dir.glob("*.wav")) if cache_dir.is_dir() else []

    freed: list[tuple[Path, int]] = []
    for path in candidates:
        if not path.is_file():
            continue
        freed.append((path, path.stat().st_size))
        if not dry_run:
            path.unlink()
    return freed
