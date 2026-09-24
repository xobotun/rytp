"""ffmpeg and ffprobe: argument lists as data, execution behind a seam.

design §9. Every command in this module is built by a pure function that
returns a ``list[str]``, and every invocation goes through :data:`Runner`.
That split is not ceremony: the bugs in a render pipeline are in the
argument lists, and a test suite that cannot run without the binary
cannot check them. Contracts §1 also rules out shell pipelines, so a
filter graph is one argument, never a string the shell sees.

This module knows nothing about the database, the canvas or the cut list.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.models import RytpError


class RenderError(RytpError):
    """Anything the render stage refuses to do. One line, no traceback."""


class FfmpegNotFoundError(RenderError):
    """ffmpeg or ffprobe is not on PATH."""


class FfmpegFailedError(RenderError):
    """An invocation exited non-zero, timed out, or wrote nothing."""


@dataclass(frozen=True)
class CompletedRun:
    """What a :data:`Runner` gives back."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


#: Injected at every call site so tests never need the binary.
Runner = Callable[[list[str]], CompletedRun]


@dataclass(frozen=True)
class LoudnormMeasurement:
    """The five numbers ffmpeg's measuring pass hands the applying pass."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


def _which(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise FfmpegNotFoundError(
            f"{name} not found on PATH; install it from https://ffmpeg.org/"
        )
    return found


def ffmpeg_binary() -> str:
    """Absolute path to ffmpeg, or a one-line error naming the download."""
    return _which("ffmpeg")


def ffprobe_binary() -> str:
    """Absolute path to ffprobe, or a one-line error naming the download."""
    return _which("ffprobe")


def seconds(ms: int) -> str:
    """Milliseconds as the fixed-point seconds ffmpeg's -ss/-t want."""
    return f"{ms / C.MS_PER_SECOND:.3f}"


def subprocess_runner(timeout_s: int) -> Runner:
    """The real runner. A factory because ffprobe and ffmpeg wait differently."""

    def run(args: list[str]) -> CompletedRun:
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - guarded by _which
            raise FfmpegNotFoundError(
                f"{args[0]} disappeared between lookup and launch"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise FfmpegFailedError(f"{args[0]} timed out after {timeout_s}s") from exc
        return CompletedRun(
            args=tuple(args),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    return run


def run_command(
    args: list[str],
    *,
    runner: Runner | None = None,
    what: str = "",
    timeout_s: int = C.RENDER_FFMPEG_TIMEOUT_S,
) -> CompletedRun:
    """Run one command, raising :class:`FfmpegFailedError` on a bad exit.

    Only the tail of stderr is quoted: ffmpeg opens with several screens
    of build configuration and puts the actual complaint last.
    """
    result = (runner or subprocess_runner(timeout_s))(args)
    if result.returncode != 0:
        label = what or Path(args[0]).name
        tail = result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS :].strip()
        raise FfmpegFailedError(f"{label} failed (exit {result.returncode}): {tail}")
    return result


def probe_command(path: Path, *, binary: str | None = None) -> list[str]:
    """Ask ffprobe for the first video stream's geometry, as JSON."""
    return [
        binary or ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,sample_aspect_ratio",
        "-of",
        "json",
        str(path),
    ]


def parse_loudnorm_json(stderr: str) -> LoudnormMeasurement | None:
    """Pull the measuring pass's JSON block out of ffmpeg's stderr.

    ffmpeg surrounds it with filter chatter, so the last ``{...}`` pair is
    the block. Silence measures as ``-inf`` and is not a measurement;
    returning ``None`` lets the caller skip normalization rather than
    feed an infinity back into the filter.
    """
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stderr[start : end + 1])
    except ValueError:
        return None
    try:
        values = [
            float(payload[key])
            for key in (
                "input_i",
                "input_tp",
                "input_lra",
                "input_thresh",
                "target_offset",
            )
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if any(math.isinf(v) or math.isnan(v) for v in values):
        return None
    return LoudnormMeasurement(*values)


# ---------------------------------------------------------------------------
# Command construction. Everything below returns a list[str] and runs nothing.
# ---------------------------------------------------------------------------

#: Quiet, non-interactive, overwrite. Used by everything that encodes.
_BASE: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "error", "-y")

#: The measuring passes must NOT be quiet. ffmpeg prints loudnorm's
#: ``print_format=json`` block at INFO level, so ``-loglevel error``
#: would silence the very thing those commands exist to read.
_MEASURE_BASE: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "info", "-y")


@dataclass(frozen=True)
class Tools:
    """The two binaries plus the runner, resolved once per render.

    Looking them up once matters for the test suite as much as for
    speed: :meth:`faked` produces a ``Tools`` that never calls
    ``shutil.which``, which is what lets the whole orchestration suite
    run on a machine with no ffmpeg installed.
    """

    ffmpeg: str
    ffprobe: str
    runner: Runner | None = None

    @classmethod
    def resolve(cls, *, runner: Runner | None = None) -> Tools:
        """Find both binaries now, so a missing one fails before any work."""
        return cls(ffmpeg=ffmpeg_binary(), ffprobe=ffprobe_binary(), runner=runner)

    @classmethod
    def faked(cls, runner: Runner) -> Tools:
        """Tools that name the binaries but never look for them."""
        return cls(ffmpeg="ffmpeg", ffprobe="ffprobe", runner=runner)


def gain_for(measured_lufs: float | None) -> float | None:
    """How many dB to move one source to the programme target.

    ``None`` in, ``None`` out: an unmeasurable source is left alone
    rather than guessed at. The clamp is design-level, not cosmetic — a
    source needing more than :data:`C.RENDER_MAX_GAIN_DB` is damaged, and
    lifting it that far lifts its noise floor into the mix with it.
    """
    if measured_lufs is None:
        return None
    delta = C.RENDER_TARGET_LUFS - measured_lufs
    return max(-C.RENDER_MAX_GAIN_DB, min(C.RENDER_MAX_GAIN_DB, delta))


def audio_filter_chain(*, gain_db: float | None = None, gap_ms: int = 0) -> str:
    """The per-fragment audio chain.

    Always resamples to the programme's rate and layout, so the concat
    demuxer never meets a parameter change. ``alimiter`` carries
    ``level=disabled`` deliberately: its default auto-level *raises*
    quiet material to the ceiling, which would throw away the
    source-to-source matching the gain just established.

    ``apad`` matches the video chain's ``tpad``, so a freeze-frame gap
    and its silence are the same length by construction.
    """
    parts = [
        f"aresample={C.RENDER_AUDIO_RATE_HZ}",
        "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    if gain_db:
        parts.append(f"volume={gain_db:g}dB")
        parts.append(f"alimiter=limit={C.RENDER_LIMITER_CEILING}:level=disabled")
    if gap_ms > 0:
        parts.append(f"apad=pad_dur={seconds(gap_ms)}")
    return ",".join(parts)


def fragment_command(
    *,
    video_path: Path,
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    video_filter: str,
    out_path: Path,
    gain_db: float | None = None,
    gap_ms: int = 0,
    preset: str = C.RENDER_VIDEO_PRESET,
    crf: int = C.RENDER_VIDEO_CRF,
    binary: str | None = None,
) -> list[str]:
    """Cut one fragment into a concat-safe intermediate.

    Video comes from a rendition asset and audio from the audio asset —
    design §4 keeps them apart so a rendition can be upgraded later — so
    there are normally two inputs, seeked identically. A local file
    registered as a single ``container`` asset passes the same path
    twice and is opened once.

    ``-ss`` precedes each ``-i``, which is both the fast seek and an
    accurate one here: the output is re-encoded, so ffmpeg decodes from
    the preceding keyframe and discards the excess.
    """
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        raise RenderError(f"fragment duration is not positive: {start_ms}..{end_ms} ms")
    single_input = video_path == audio_path
    args = [binary or ffmpeg_binary(), *_BASE]
    args += ["-ss", seconds(start_ms), "-t", seconds(duration_ms), "-i", str(video_path)]
    if not single_input:
        args += [
            "-ss", seconds(start_ms), "-t", seconds(duration_ms), "-i", str(audio_path)
        ]
    audio_input = 0 if single_input else 1
    graph = (
        f"[0:v]{video_filter}[v];"
        f"[{audio_input}:a]{audio_filter_chain(gain_db=gain_db, gap_ms=gap_ms)}[a]"
    )
    args += [
        "-filter_complex", graph,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", C.RENDER_VIDEO_CODEC,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", C.RENDER_PIXEL_FORMAT,
        "-c:a", C.RENDER_INTERMEDIATE_AUDIO_CODEC,
        "-ar", str(C.RENDER_AUDIO_RATE_HZ),
        "-ac", str(C.RENDER_AUDIO_CHANNELS),
        "-f", C.RENDER_INTERMEDIATE_FORMAT,
        str(out_path),
    ]
    return args


def concat_list_text(paths: Sequence[Path]) -> str:
    """The concat demuxer's list file.

    Forward slashes even on Windows — the demuxer treats a backslash as
    an escape character — and a literal single quote in a filename has
    to leave and re-enter the quoting, which is the ``'\\''`` dance.
    """
    lines = []
    for path in paths:
        quoted = path.as_posix().replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    return "\n".join(lines) + "\n"


def write_concat_list(list_file: Path, paths: Sequence[Path]) -> Path:
    """Write the list file, creating its directory. Returns the path."""
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text(concat_list_text(paths), encoding="utf-8")
    return list_file


def _measure_filter() -> str:
    return (
        f"loudnorm=I={C.RENDER_TARGET_LUFS}"
        f":TP={C.RENDER_TRUE_PEAK_DBTP}"
        f":LRA={C.RENDER_LOUDNESS_RANGE_LU}"
        ":print_format=json"
    )


def _apply_filter(measured: LoudnormMeasurement) -> str:
    """The second pass, with the first pass's numbers fed back in.

    ``linear=true`` asks for one static gain over the whole programme,
    which is what keeps the cut sounding like a recording rather than a
    compressor. ffmpeg silently falls back to dynamic mode when the
    measured range exceeds the target LRA — which is why
    :data:`C.RENDER_LOUDNESS_RANGE_LU` is 13 rather than the broadcast 11.
    """
    return (
        f"loudnorm=I={C.RENDER_TARGET_LUFS}"
        f":TP={C.RENDER_TRUE_PEAK_DBTP}"
        f":LRA={C.RENDER_LOUDNESS_RANGE_LU}"
        f":measured_I={measured.input_i:g}"
        f":measured_TP={measured.input_tp:g}"
        f":measured_LRA={measured.input_lra:g}"
        f":measured_thresh={measured.input_thresh:g}"
        f":offset={measured.target_offset:g}"
        ":linear=true"
    )


def source_loudness_command(
    *, path: Path, start_ms: int, duration_ms: int, binary: str | None = None
) -> list[str]:
    """Measure one window of one source, to place its gain."""
    return [
        binary or ffmpeg_binary(),
        *_MEASURE_BASE,
        "-ss", seconds(max(0, start_ms)),
        "-t", seconds(duration_ms),
        "-i", str(path),
        "-vn",
        "-af", _measure_filter(),
        "-f", "null",
        "-",
    ]


def programme_loudness_command(
    *, list_file: Path, binary: str | None = None
) -> list[str]:
    """Measure the assembled programme — the first of the two passes."""
    return [
        binary or ffmpeg_binary(),
        *_MEASURE_BASE,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-vn",
        "-af", _measure_filter(),
        "-f", "null",
        "-",
    ]


def concat_command(
    *,
    list_file: Path,
    out_path: Path,
    measured: LoudnormMeasurement | None = None,
    binary: str | None = None,
) -> list[str]:
    """Join the intermediates into the deliverable.

    Video is copied — every intermediate was already encoded to the same
    canvas, rate and pixel format, so there is nothing left to do to it.
    Audio is always re-encoded, with or without ``loudnorm``: PCM is what
    made the seams safe, and an MP4 wants AAC.
    """
    args = [
        binary or ffmpeg_binary(),
        *_BASE,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "copy",
    ]
    if measured is not None:
        args += ["-af", _apply_filter(measured)]
    args += [
        "-c:a", C.RENDER_AUDIO_CODEC,
        "-b:a", C.RENDER_AUDIO_BITRATE,
        "-ar", str(C.RENDER_AUDIO_RATE_HZ),
        "-ac", str(C.RENDER_AUDIO_CHANNELS),
        "-movflags", "+faststart",
        str(out_path),
    ]
    return args
