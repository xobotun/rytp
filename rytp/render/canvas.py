"""Output geometry for a mixed-shape archive. design §9.

A decade of uploads means 4:3 and 16:9 at several heights inside one
render, so every fragment is scaled into one canvas and padded. The two
modes differ only in how the canvas shape is chosen; both keep every
pixel of every source (``force_original_aspect_ratio=decrease`` fits a
source into the canvas, scaling it up or down as needed, and never
crops) and both pad with a flat colour.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.render.ffmpeg import RenderError, Runner, probe_command, run_command, seconds


@dataclass(frozen=True)
class SourceGeometry:
    """One source video's shape, as ffprobe reports it."""

    width: int
    height: int
    fps: float
    sar: float = 1.0

    @property
    def display_width(self) -> int:
        """Width after the pixel aspect ratio, which is what the eye sees."""
        return max(2, round(self.width * self.sar))

    @property
    def aspect(self) -> float:
        return self.display_width / self.height


@dataclass(frozen=True)
class Canvas:
    """The one shape every fragment is scaled and padded into."""

    width: int
    height: int
    fps: int
    mode: str

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps} fps ({self.mode})"


def _even(value: float) -> int:
    """Nearest even integer >= 2, rounding half up. yuv420p needs both

    dimensions even; round-half-up (rather than Python's banker's
    rounding) is what a human expects from "made even" at a .5 boundary.
    """
    return max(2, 2 * math.floor(value / 2 + 0.5))


def _ratio(text: str | None, *, default: float) -> float:
    """Parse ffprobe's ``"30000/1001"`` / ``"16:11"`` fractions."""
    if not text:
        return default
    for separator in ("/", ":"):
        if separator in text:
            head, _, tail = text.partition(separator)
            try:
                numerator, denominator = float(head), float(tail)
            except ValueError:
                return default
            if denominator <= 0 or numerator <= 0:
                return default
            return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return default


def parse_probe_json(text: str) -> SourceGeometry:
    """Turn one ffprobe JSON reply into a :class:`SourceGeometry`."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise RenderError(f"ffprobe returned no usable JSON: {text[:120]!r}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise RenderError("ffprobe found no video stream in the source")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError("ffprobe reported no video stream size") from exc
    if width <= 0 or height <= 0:
        raise RenderError(f"ffprobe reported a {width}x{height} video stream")
    return SourceGeometry(
        width=width,
        height=height,
        fps=_ratio(stream.get("r_frame_rate"), default=float(C.RENDER_FALLBACK_FPS)),
        # "0:1" means "unknown"; square pixels are the only safe reading.
        sar=_ratio(stream.get("sample_aspect_ratio"), default=1.0),
    )


def probe_geometry(
    path: Path, *, runner: Runner | None = None, binary: str | None = None
) -> SourceGeometry:
    """Probe one source file. Call once per source video, not per fragment."""
    result = run_command(
        probe_command(path, binary=binary),
        runner=runner,
        what=f"ffprobe {path.name}",
        timeout_s=C.RENDER_FFPROBE_TIMEOUT_S,
    )
    return parse_probe_json(result.stdout)


def choose_fps(geometries: Sequence[SourceGeometry], *, override: int = 0) -> int:
    """The output frame rate: the commonest source rate, ties to the higher.

    Measured rather than assumed. This archive is European, so 25 is the
    likely majority, and forcing it to 30 duplicates every fifth frame —
    visible on a talking head, and not worth the file size.
    """
    if override > 0:
        return min(override, C.RENDER_MAX_FPS)
    if not geometries:
        return C.RENDER_FALLBACK_FPS
    counts = Counter(max(1, round(g.fps)) for g in geometries)
    best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return min(best, C.RENDER_MAX_FPS)


def plan_canvas(
    geometries: Sequence[SourceGeometry],
    *,
    mode: str = C.RENDER_DEFAULT_CANVAS_MODE,
    height: int = 0,
    fps: int = 0,
) -> Canvas:
    """Decide the output shape for a set of sources.

    One rule, picked by ``mode``: a canvas is an aspect ratio and a
    height that satisfies it, the two made even. ``16:9`` — the default —
    fixes the aspect at 16:9 regardless of the sources, so a fixed
    horizontal frame boxes any 4:3 or vertical material into it (bars on
    the sides for a vertical source, top and bottom for 4:3) rather than
    letting the shape of one source drive the canvas. ``bbox`` is the
    owner's "smallest bounding box across all sources", which the owner
    clarified means the *biggest* box that contains every source at its
    native size — never a downscale, never a squashed aspect: the aspect
    is ``max(display_width) / max(height)`` over the sources, so a 16:10
    source paired with a 16:9 one yields a 16:10 canvas, and a corpus of
    vertical shorts yields a vertical canvas. When every source shares
    one aspect (an all-16:9 corpus, say), the two modes agree.

    Either way the canvas's *longer* edge is capped at
    :data:`C.RENDER_MAX_CANVAS_LONG_EDGE`, whichever dimension that is —
    a height-only cap would crush a vertical short's width instead of
    its height, and would clip a 16:10 canvas down to 16:9 — and the
    height is floored at :data:`C.RENDER_MIN_CANVAS_HEIGHT`. Both bounds
    are applied by capping the height at the value that keeps the aspect
    exact (``long-edge bound / aspect`` when the canvas is at least as
    wide as it is tall, ``long-edge bound`` outright otherwise) and then
    deriving the width from that height, so the aspect never drifts when
    a bound bites. Sources smaller than the resulting canvas are scaled
    up rather than left at their own size:
    ``force_original_aspect_ratio=decrease`` in :func:`video_filter_chain`
    fits a source to the canvas box in either direction, preserving its
    aspect, and never crops.
    """
    if mode not in C.RENDER_CANVAS_MODES:
        raise RenderError(
            f"unknown canvas mode {mode!r}; available: {', '.join(C.RENDER_CANVAS_MODES)}"
        )
    if not geometries:
        raise RenderError("cannot plan a canvas with no sources")
    tallest = max(g.height for g in geometries)
    if mode == "bbox":
        widest = max(g.display_width for g in geometries)
        aspect = widest / tallest
    else:
        wide, tall = C.RENDER_DEFAULT_ASPECT
        aspect = wide / tall
    if height > 0:
        canvas_height = _even(height)
    else:
        # The long edge is the width when aspect >= 1 (landscape or
        # square) and the height itself otherwise (portrait) — dividing
        # the bound by max(aspect, 1.0) turns either case into a single
        # height cap that keeps the aspect exact.
        height_cap = C.RENDER_MAX_CANVAS_LONG_EDGE / max(aspect, 1.0)
        canvas_height = _even(min(tallest, height_cap))
    canvas_height = max(canvas_height, C.RENDER_MIN_CANVAS_HEIGHT)
    return Canvas(
        width=_even(canvas_height * aspect),
        height=canvas_height,
        fps=choose_fps(geometries, override=fps),
        mode=mode,
    )


def video_filter_chain(canvas: Canvas, *, gap_ms: int = 0) -> str:
    """The per-fragment video chain: fit, pad, normalise, optionally freeze.

    ``force_divisible_by=2`` matters: an odd intermediate width with
    yuv420p is rejected by the encoder, and a 4:3 source scaled into a
    16:9 canvas hits odd widths constantly.

    A gap is a freeze-frame (design §9), produced by ``tpad`` cloning the
    last frame of *this* fragment rather than by a separate filler
    segment — one fewer encode per seam, and the frozen video cannot
    drift away from the silence padded onto the same fragment's audio.
    ``tpad`` comes after ``fps`` so the held frames are at the output rate.
    """
    parts = [
        f"scale={canvas.width}:{canvas.height}"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2",
        f"pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2"
        f":color={C.RENDER_PAD_COLOR}",
        "setsar=1",
        f"fps={canvas.fps}",
        f"format={C.RENDER_PIXEL_FORMAT}",
    ]
    if gap_ms > 0:
        parts.append(f"tpad=stop_mode=clone:stop_duration={seconds(gap_ms)}")
    return ",".join(parts)
