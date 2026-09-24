"""Canvas geometry: one output shape for mixed sources (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import canvas as K
from rytp.render.ffmpeg import RenderError
from tests.render_fakes import RecordingRunner

PROBE_JSON = """\
{"streams": [{"width": 1280, "height": 720, "r_frame_rate": "30000/1001",
              "sample_aspect_ratio": "1:1"}]}
"""

WIDE = K.SourceGeometry(width=1280, height=720, fps=25.0)
FOUR_THREE = K.SourceGeometry(width=640, height=480, fps=25.0)
TALL_WIDE = K.SourceGeometry(width=1920, height=1080, fps=50.0)
CINEMA = K.SourceGeometry(width=1280, height=544, fps=25.0)

# The owner's pinned cases (see canvas.py's plan_canvas docstring). None
# of these need the cap raised: RENDER_MAX_CANVAS_LONG_EDGE (1920) bounds
# whichever edge is longer, and every one of these sits at or under that
# on its long edge already.
SIXTEEN_TEN = K.SourceGeometry(width=1920, height=1200, fps=25.0)
SMALL_SIXTEEN_NINE = K.SourceGeometry(width=640, height=360, fps=25.0)
VERTICAL_SHORT_A = K.SourceGeometry(width=1080, height=1920, fps=30.0)
VERTICAL_SHORT_B = K.SourceGeometry(width=1080, height=1920, fps=30.0)
PORTRAIT = K.SourceGeometry(width=1080, height=1920, fps=30.0)
LANDSCAPE = K.SourceGeometry(width=1920, height=1080, fps=30.0)


def test_parse_probe_json_reads_size_rate_and_pixel_aspect() -> None:
    geometry = K.parse_probe_json(PROBE_JSON)
    assert (geometry.width, geometry.height) == (1280, 720)
    assert geometry.fps == pytest.approx(29.97, abs=0.01)
    assert geometry.sar == pytest.approx(1.0)


def test_parse_probe_json_treats_an_unknown_pixel_aspect_as_square() -> None:
    geometry = K.parse_probe_json(
        '{"streams": [{"width": 720, "height": 576, "r_frame_rate": "25/1",'
        ' "sample_aspect_ratio": "0:1"}]}'
    )
    assert geometry.sar == pytest.approx(1.0)


def test_parse_probe_json_honours_an_anamorphic_pixel_aspect() -> None:
    geometry = K.parse_probe_json(
        '{"streams": [{"width": 720, "height": 576, "r_frame_rate": "25/1",'
        ' "sample_aspect_ratio": "16:11"}]}'
    )
    assert geometry.display_width == 1047
    assert geometry.aspect == pytest.approx(1047 / 576)


def test_parse_probe_json_without_a_video_stream_is_an_error() -> None:
    with pytest.raises(RenderError, match="no video stream"):
        K.parse_probe_json('{"streams": []}')


def test_probe_geometry_runs_ffprobe_once_and_parses_its_stdout() -> None:
    runner = RecordingRunner(stdout=PROBE_JSON, make_outputs=False)
    geometry = K.probe_geometry(Path("media/3/video-720.mp4"), runner=runner, binary="ffprobe")
    assert geometry.height == 720
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "ffprobe"


def test_the_default_canvas_is_sixteen_by_nine() -> None:
    canvas = K.plan_canvas([FOUR_THREE, WIDE], mode="16:9")
    assert (canvas.width, canvas.height) == (1280, 720)
    assert canvas.mode == "16:9"


def test_the_default_canvas_height_is_the_tallest_source() -> None:
    canvas = K.plan_canvas([WIDE, TALL_WIDE], mode="16:9")
    assert canvas.height == 1080


def test_the_canvas_height_is_capped() -> None:
    """3840x2160 has a 3840 long edge; bounded at 1920, this reduces to

    the familiar 1080p height cap for a 16:9 canvas.
    """
    huge = K.SourceGeometry(width=3840, height=2160, fps=25.0)
    canvas = K.plan_canvas([huge], mode="16:9")
    assert (canvas.width, canvas.height) == (1920, 1080)


def test_an_explicit_height_wins_and_is_made_even() -> None:
    canvas = K.plan_canvas([WIDE], mode="16:9", height=721)
    assert canvas.height == 722
    assert canvas.width % 2 == 0


def test_sixteen_by_nine_keeps_a_fixed_frame_and_boxes_a_vertical_short_into_it() -> None:
    """The flag the owner asked for ('enforce 16:9 ... letterbox the shorts')

    already exists as ``--canvas 16:9``, the default: a landscape corpus
    plus one vertical short still plans a fixed 16:9 canvas at the
    landscape source's own resolution, never the short's own aspect.
    """
    landscape = K.SourceGeometry(width=1920, height=1080, fps=25.0)
    vertical_short = K.SourceGeometry(width=1080, height=1920, fps=25.0)
    canvas = K.plan_canvas([landscape, vertical_short], mode="16:9")
    assert (canvas.width, canvas.height) == (1920, 1080)
    assert canvas.mode == "16:9"


def test_bbox_is_the_biggest_box_never_a_downscale_never_a_squash() -> None:
    """Owner's case 1: a 16:10 source plus a smaller 16:9 one.

    The rejected reading derived the aspect from the *widest* source and
    the height from the others, which would give a 16:9 canvas here. The
    owner was explicit: he meant the *biggest* box, at the 16:10 source's
    own native resolution — the 360p 16:9 source scales up into it. Its
    long edge (1920) sits exactly at the cap, so nothing is scaled down.
    """
    canvas = K.plan_canvas([SIXTEEN_TEN, SMALL_SIXTEEN_NINE], mode="bbox")
    assert (canvas.width, canvas.height) == (1920, 1200)


def test_bbox_keeps_a_bunch_of_vertical_shorts_squashed_together_vertical() -> None:
    """Owner's case 2, verbatim: vertical content must stay vertical."""
    canvas = K.plan_canvas([VERTICAL_SHORT_A, VERTICAL_SHORT_B], mode="bbox")
    assert (canvas.width, canvas.height) == (1080, 1920)
    assert canvas.height > canvas.width  # still portrait, not forced landscape


def test_bbox_of_mixed_portrait_and_landscape_is_the_honest_bounding_box() -> None:
    """Owner's case 3: a portrait and a landscape source of the same size.

    Neither is cropped nor squashed, so the only honest bounding box is
    the square that contains both at their native resolution.
    """
    canvas = K.plan_canvas([PORTRAIT, LANDSCAPE], mode="bbox")
    assert (canvas.width, canvas.height) == (1920, 1920)


def test_bbox_preserves_aspect_when_the_long_edge_cap_bites() -> None:
    """4000x3000's long edge (4000, the width) exceeds the 1920 bound.

    Scaling both dimensions down together keeps the 4:3 aspect exact —
    the width lands on the cap, not the height.
    """
    huge_four_three = K.SourceGeometry(width=4000, height=3000, fps=25.0)
    canvas = K.plan_canvas([huge_four_three], mode="bbox")
    assert canvas.width == C.RENDER_MAX_CANVAS_LONG_EDGE
    assert canvas.height == pytest.approx(canvas.width * (3000 / 4000), abs=1)


def test_bbox_and_the_default_agree_when_every_source_is_sixteen_by_nine() -> None:
    """`bbox` yields the tallest source's own resolution.

    That only matches the default 16:9 canvas because both fixtures here
    already are exactly 16:9 — it is not a general property of `bbox`.
    """
    bbox = K.plan_canvas([WIDE, TALL_WIDE], mode="bbox")
    default = K.plan_canvas([WIDE, TALL_WIDE], mode="16:9")
    assert (bbox.width, bbox.height, bbox.fps) == (default.width, default.height, default.fps)


def test_an_unknown_canvas_mode_names_the_known_ones() -> None:
    with pytest.raises(RenderError, match="bbox"):
        K.plan_canvas([WIDE], mode="square")


def test_no_sources_is_an_error_rather_than_a_default_canvas() -> None:
    with pytest.raises(RenderError, match="no sources"):
        K.plan_canvas([], mode="16:9")


def test_the_frame_rate_is_the_commonest_source_rate() -> None:
    assert K.choose_fps([WIDE, FOUR_THREE, TALL_WIDE]) == 25


def test_a_tie_on_frame_rate_takes_the_higher() -> None:
    assert K.choose_fps([WIDE, TALL_WIDE]) == 50


def test_ntsc_rates_round_to_whole_frames() -> None:
    ntsc = K.SourceGeometry(width=640, height=480, fps=29.97)
    film = K.SourceGeometry(width=640, height=480, fps=23.976)
    assert K.choose_fps([ntsc]) == 30
    assert K.choose_fps([film]) == 24


def test_the_frame_rate_is_capped_and_has_a_fallback() -> None:
    fast = K.SourceGeometry(width=640, height=480, fps=240.0)
    assert K.choose_fps([fast]) == C.RENDER_MAX_FPS
    assert K.choose_fps([]) == C.RENDER_FALLBACK_FPS
    assert K.choose_fps([WIDE], override=60) == 60


def test_the_video_filter_scales_inside_the_canvas_and_pads_the_rest() -> None:
    chain = K.video_filter_chain(K.Canvas(width=1280, height=720, fps=25, mode="16:9"))
    assert "scale=1280:720:force_original_aspect_ratio=decrease" in chain
    assert "force_divisible_by=2" in chain
    assert f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color={C.RENDER_PAD_COLOR}" in chain
    assert "setsar=1" in chain
    assert "fps=25" in chain
    assert f"format={C.RENDER_PIXEL_FORMAT}" in chain
    assert "tpad" not in chain


def test_a_gap_freezes_the_last_frame_after_the_rate_conversion() -> None:
    chain = K.video_filter_chain(
        K.Canvas(width=1280, height=720, fps=25, mode="16:9"), gap_ms=240
    )
    assert chain.endswith("tpad=stop_mode=clone:stop_duration=0.240")
    assert chain.index("fps=25") < chain.index("tpad")
