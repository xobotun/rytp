"""Audio extraction: ffmpeg → 16 kHz mono WAV.

DESIGN §3 / §7: ``data/audio/{video_id}.wav`` is the single source of
audio truth. STT, loudnorm, and ``mode=concat`` splice all read from
this file. The audio is extracted exactly once per video.

The canonical format (16 kHz mono int16 WAV) is chosen because:

* Whisper-class STT engines are trained on 16 kHz mono audio.
* Mixing to mono at extract time keeps storage simple — no stereo
  metadata to thread through downstream.
* int16 PCM is the lowest-bit-depth ffmpeg reliably writes; the
  pipeline never needs the dynamic range of float32.

Public surface:

* :func:`extract_audio` — extract a 16 kHz mono WAV from a video file.
* :class:`FfmpegNotFoundError` — raised when ffmpeg isn't on PATH.
* :class:`ExtractionError` — raised when ffmpeg returned non-zero.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from rytp import constants as C
from rytp import config as _config


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg isn't on PATH."""


class ExtractionError(RuntimeError):
    """ffmpeg returned non-zero or the output file is missing."""


def extract_audio(
    video_path: Path,
    out_dir: Path,
    *,
    video_id: str,
    overwrite: bool = False,
) -> Path:
    """Extract a 16 kHz mono WAV from ``video_path``.

    Writes to ``out_dir / f"{video_id}.wav"``. Returns the output path.

    Args:
        video_path: Source media (any format ffmpeg understands).
        out_dir: Target directory — created if missing.
        video_id: Used as the output filename stem.
        overwrite: Re-run ffmpeg even if the output file exists.

    Returns:
        The path to the written WAV.

    Raises:
        FfmpegNotFoundError: ffmpeg isn't on PATH.
        FileNotFoundError: ``video_path`` doesn't exist.
        ExtractionError: ffmpeg returned non-zero, or no output file
            was produced.
    """
    # Look up ffmpeg lazily through the ``_config`` module so tests can
    # monkey-patch ``_config.ffmpeg_binary`` and have the change take
    # effect here.
    ffmpeg = _config.ffmpeg_binary()
    if ffmpeg is None:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/ "
            "(Windows users: download the release zip and add the bin/ "
            "directory to PATH)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.wav"

    if not video_path.exists():
        raise FileNotFoundError(f"video file not found: {video_path}")

    if out_path.exists() and not overwrite:
        return out_path

    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",  # overwrite at the ffmpeg level; we already gated above
        "-i",
        str(video_path),
        "-vn",  # drop video stream — audio only
        "-ac",
        str(C.AUDIO_CHANNELS),
        "-ar",
        str(C.AUDIO_SAMPLE_RATE_HZ),
        "-f",
        "wav",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as e:
        # The ffmpeg binary was on PATH at the lookup above but went
        # missing between then and now (TOCTOU).
        raise FfmpegNotFoundError(str(e)) from e

    if result.returncode != 0:
        raise ExtractionError(
            f"ffmpeg extract failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )

    if not out_path.exists():
        raise ExtractionError(
            f"ffmpeg exited 0 but {out_path} was not created"
        )

    return out_path