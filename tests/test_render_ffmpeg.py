"""The ffmpeg seam: argument lists are data, execution is injected (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import ffmpeg as F
from tests.render_fakes import RecordingRunner

LOUDNORM_STDERR = """\
[Parsed_loudnorm_0 @ 0x5599]
{
        "input_i" : "-27.24",
        "input_tp" : "-8.51",
        "input_lra" : "6.90",
        "input_thresh" : "-37.51",
        "output_i" : "-16.02",
        "output_tp" : "-1.50",
        "output_lra" : "6.70",
        "output_thresh" : "-26.29",
        "normalization_type" : "dynamic",
        "target_offset" : "0.02"
}
"""


def test_seconds_renders_milliseconds_with_three_decimals() -> None:
    assert F.seconds(0) == "0.000"
    assert F.seconds(1) == "0.001"
    assert F.seconds(612_340) == "612.340"


def test_probe_command_asks_for_the_first_video_stream_as_json() -> None:
    args = F.probe_command(Path("media/3/audio.m4a"), binary="ffprobe")
    assert args[0] == "ffprobe"
    assert "-select_streams" in args and "v:0" in args
    assert args[-1] == str(Path("media/3/audio.m4a"))
    assert "json" in args


def test_parse_loudnorm_json_reads_the_five_measured_values() -> None:
    measured = F.parse_loudnorm_json(LOUDNORM_STDERR)
    assert measured is not None
    assert measured.input_i == pytest.approx(-27.24)
    assert measured.input_tp == pytest.approx(-8.51)
    assert measured.input_lra == pytest.approx(6.90)
    assert measured.input_thresh == pytest.approx(-37.51)
    assert measured.target_offset == pytest.approx(0.02)


def test_parse_loudnorm_json_of_silence_is_unknown() -> None:
    assert F.parse_loudnorm_json('{"input_i" : "-inf", "input_tp" : "-inf"}') is None
    assert F.parse_loudnorm_json("no json at all") is None


def test_run_command_returns_the_recorded_result() -> None:
    runner = RecordingRunner()
    result = F.run_command(["ffmpeg", "-version"], runner=runner)
    assert result.returncode == 0
    assert runner.calls == [["ffmpeg", "-version"]]


def test_run_command_raises_with_the_tail_of_stderr() -> None:
    runner = RecordingRunner(returncode=1, stderr="x" * 5000 + "Invalid argument")
    with pytest.raises(F.FfmpegFailedError) as excinfo:
        F.run_command(["ffmpeg", "-nope"], runner=runner)
    message = str(excinfo.value)
    assert "Invalid argument" in message
    assert len(message) < C.FFMPEG_ERROR_TAIL_CHARS + 200


def test_a_missing_binary_says_where_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    with pytest.raises(F.FfmpegNotFoundError, match=r"ffmpeg\.org"):
        F.ffmpeg_binary()
    with pytest.raises(F.FfmpegNotFoundError, match=r"ffmpeg\.org"):
        F.ffprobe_binary()


def test_render_errors_are_rytp_errors() -> None:
    from rytp.models import RytpError

    assert issubclass(F.RenderError, RytpError)
    assert issubclass(F.FfmpegFailedError, F.RenderError)
    assert issubclass(F.FfmpegNotFoundError, F.RenderError)
