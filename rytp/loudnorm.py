"""EBU R128 loudness normalization for splice intermediates.

DESIGN §7: per-clip loudnorm pass with target integrated loudness
``LUFS_TARGET`` (LUFS), true-peak cap ``TRUE_PEAK_DBTP`` (dBTP), range
``LOUDNESS_RANGE_LU`` (LU). Output is a per-clip normalized
intermediate under ``data/output/normalized``.

The two-pass pattern here is the standard ffmpeg ``loudnorm`` recipe:

1. **Measure** — run the filter with ``print_format=json`` to dump
   the input clip's measured ``input_i``, ``input_tp``,
   ``input_lra``, ``input_thresh``, and ``target_offset`` to
   stderr.
2. **Apply** — re-run with those values fed back into the filter as
   ``measured_*`` so the gain is applied in one pass without a
   pre-scan.

We deliberately don't use the measurement-only optimization (DESIGN
§7 "future v2"): ffmpeg's two-pass loudnorm in a filtergraph is
brittle across clip boundaries, and the current per-clip approach
is simple and correct. The cost is one extra encode pass per clip,
which is acceptable for the v1 workload.

Public surface:

* :func:`loudnorm_clip` — normalize one clip in place; returns the
  output path.
* :func:`loudnorm_clips` — normalize many clips (one splice set).
* :class:`LoudnormResult` — return value from :func:`loudnorm_clips`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.config import paths as config_paths


@dataclass(frozen=True)
class LoudnormResult:
    """Result of normalizing one clip.

    Attributes:
        clip_id: Database id of the clip in the ``clips`` table.
        source_path: The input audio (typically the per-video 16 kHz
            mono WAV at ``data/audio/{video_id}.wav``).
        output_path: The normalized intermediate
            (``data/output/normalized/{clip_id}.m4a``).
    """

    clip_id: int
    source_path: Path
    output_path: Path


def _ffmpeg() -> str:
    """Locate ffmpeg on PATH, or raise with a one-line install hint."""
    bin_ = shutil.which("ffmpeg")
    if bin_ is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install from https://ffmpeg.org/."
        )
    return bin_


def loudnorm_clip(
    source_path: Path,
    out_dir: Path,
    *,
    clip_id: int,
    start_ms: int,
    end_ms: int,
    overwrite: bool = False,
) -> Path:
    """Extract ``[start_ms, end_ms]`` from ``source_path`` and loudnorm it.

    The audio is read from ``source_path`` (which is the per-video
    16 kHz mono WAV in ``data/audio/``), sliced to the clip's time
    range, and run through ffmpeg's ``loudnorm`` two-pass filter
    targeting :data:`rytp.constants.LUFS_TARGET`,
    :data:`rytp.constants.TRUE_PEAK_DBTP`, and
    :data:`rytp.constants.LOUDNESS_RANGE_LU`.

    Writes ``out_dir / f"{clip_id}.m4a"`` and returns that path.

    Args:
        source_path: 16 kHz mono WAV to read from.
        out_dir: Where the normalized intermediate is written.
        clip_id: Used as the output filename stem.
        start_ms: Clip start in milliseconds (inclusive).
        end_ms: Clip end in milliseconds (exclusive).
        overwrite: Re-run ffmpeg even if the output file exists.

    Returns:
        Path to the written ``.m4a`` file.

    Raises:
        RuntimeError: if ffmpeg isn't on PATH or the loudnorm JSON
            cannot be parsed from stderr.
        subprocess.CalledProcessError: if either ffmpeg pass fails.
    """
    ffmpeg = _ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}.m4a"
    if out_path.exists() and not overwrite:
        return out_path

    # Convert ms → s for ffmpeg's ``-ss`` / ``-t`` flags.
    start_s = start_ms / C.SECONDS_TO_MS
    duration_s = max(
        C.MIN_CLIP_DURATION_S, (end_ms - start_ms) / C.SECONDS_TO_MS
    )

    # Pass 1: measure. The loudnorm filter with ``print_format=json``
    # writes a trailing JSON block to stderr containing the input
    # clip's loudness stats.
    measure_cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_s:.3f}",
        "-af",
        (
            f"loudnorm=I={C.LUFS_TARGET}"
            f":TP={C.TRUE_PEAK_DBTP}"
            f":LRA={C.LOUDNESS_RANGE_LU}"
            f":print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    measure = subprocess.run(
        measure_cmd, check=True, capture_output=True, text=True
    )
    measured = _parse_loudnorm_json(measure.stderr)

    # Pass 2: apply. The measured values are fed back so the gain is
    # applied in a single encode pass.
    apply_cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_s:.3f}",
        "-af",
        (
            f"loudnorm=I={C.LUFS_TARGET}"
            f":TP={C.TRUE_PEAK_DBTP}"
            f":LRA={C.LOUDNESS_RANGE_LU}"
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            f":linear=true:print_format=summary"
        ),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-vn",
        str(out_path),
    ]
    subprocess.run(apply_cmd, check=True, capture_output=True, text=True)
    return out_path


def _parse_loudnorm_json(stderr: str) -> dict[str, float]:
    """Extract the trailing JSON block from ffmpeg's loudnorm stderr.

    ffmpeg prints the JSON block surrounded by ``[Parsed_loudnorm_0 @
    0x...]`` lines and other filter chatter; we just grab the last
    ``{...}`` pair and parse it.

    Raises:
        RuntimeError: if no ``{...}`` pair is found in the tail.
    """
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(
            f"could not parse loudnorm output: "
            f"{stderr[-C.LOUDNORM_PARSE_TAIL_CHARS:]!r}"
        )
    data = json.loads(stderr[start : end + 1])
    return {
        "input_i": float(data["input_i"]),
        "input_tp": float(data["input_tp"]),
        "input_lra": float(data["input_lra"]),
        "input_thresh": float(data["input_thresh"]),
        "target_offset": float(data["target_offset"]),
    }


def loudnorm_clips(
    clips: Iterable[tuple[int, int, int, int, Path]],
    *,
    out_dir: Path | None = None,
) -> list[LoudnormResult]:
    """Normalize many clips.

    Each tuple is ``(clip_id, video_id, start_ms, end_ms, source_path)``.
    The ``source_path`` is the per-video 16 kHz mono WAV at
    ``data/audio/{video_id}.wav`` (DESIGN §7 "single source of audio
    truth"). Use :func:`loudnorm_clip` directly when you have the
    timing in hand and don't want to round-trip the tuple format.

    Args:
        clips: Iterable of ``(clip_id, video_id, start_ms, end_ms, source_path)``.
        out_dir: Where the intermediates are written. Defaults to
            ``config.paths.normalized`` (``data/output/normalized``).

    Returns:
        One :class:`LoudnormResult` per input clip.
    """
    out_dir = out_dir or config_paths.normalized
    results: list[LoudnormResult] = []
    for clip_id, _vid, start_ms, end_ms, source_path in clips:
        out = loudnorm_clip(
            source_path,
            out_dir,
            clip_id=clip_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        results.append(
            LoudnormResult(clip_id=clip_id, source_path=source_path, output_path=out)
        )
    return results