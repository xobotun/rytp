"""Every ffmpeg argument list Part 6 builds (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import ffmpeg as F

VIDEO = Path("/media/3/video-720.mp4")
AUDIO = Path("/media/3/audio.m4a")
OUT = Path("/out/r1/fragments/0000.mkv")
VFILTER = "scale=1280:720:force_original_aspect_ratio=decrease,setsar=1,fps=25"


def _pairs(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_a_fragment_seeks_both_inputs_to_the_same_range() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=612_340, end_ms=613_100,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-i") == [str(VIDEO), str(AUDIO)]
    assert _pairs(args, "-ss") == ["612.340", "612.340"]
    assert _pairs(args, "-t") == ["0.760", "0.760"]


def test_a_fragment_takes_video_from_the_rendition_and_audio_from_the_audio_asset() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]")
    assert "[1:a]" in graph
    assert _pairs(args, "-map") == ["[v]", "[a]"]


def test_a_container_source_is_opened_once_and_mapped_twice() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=VIDEO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-i") == [str(VIDEO)]
    graph = args[args.index("-filter_complex") + 1]
    assert "[0:a]" in graph and "[1:a]" not in graph


def test_the_intermediate_is_matroska_with_pcm_audio() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-c:a") == [C.RENDER_INTERMEDIATE_AUDIO_CODEC]
    assert _pairs(args, "-f") == [C.RENDER_INTERMEDIATE_FORMAT]
    assert _pairs(args, "-ar") == [str(C.RENDER_AUDIO_RATE_HZ)]
    assert _pairs(args, "-ac") == [str(C.RENDER_AUDIO_CHANNELS)]
    assert args[-1] == str(OUT)


def test_encoder_settings_are_overridable_per_render() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, preset="ultrafast", crf=30,
        binary="ffmpeg",
    )
    assert _pairs(args, "-preset") == ["ultrafast"]
    assert _pairs(args, "-crf") == ["30"]


def test_an_empty_fragment_is_refused() -> None:
    with pytest.raises(F.RenderError, match="not positive"):
        F.fragment_command(
            video_path=VIDEO, audio_path=AUDIO, start_ms=1_000, end_ms=1_000,
            video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
        )


def test_tools_can_be_faked_without_either_binary_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    tools = F.Tools.faked(lambda args: F.CompletedRun(tuple(args), 0, "", ""))
    assert (tools.ffmpeg, tools.ffprobe) == ("ffmpeg", "ffprobe")
    assert tools.runner is not None


def test_resolving_tools_without_ffmpeg_is_a_one_line_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    with pytest.raises(F.FfmpegNotFoundError, match=r"ffmpeg\.org"):
        F.Tools.resolve()


def test_the_audio_chain_is_a_plain_resample_when_nothing_is_asked_of_it() -> None:
    chain = F.audio_filter_chain()
    assert chain == (
        f"aresample={C.RENDER_AUDIO_RATE_HZ},"
        "aformat=sample_fmts=s16:channel_layouts=stereo"
    )


def test_a_gain_is_applied_with_a_limiter_that_does_not_auto_level() -> None:
    chain = F.audio_filter_chain(gain_db=3.25)
    assert "volume=3.25dB" in chain
    assert f"alimiter=limit={C.RENDER_LIMITER_CEILING}:level=disabled" in chain


def test_a_zero_gain_adds_no_filter() -> None:
    assert "volume" not in F.audio_filter_chain(gain_db=0.0)


def test_a_gap_pads_the_audio_by_exactly_the_freeze_duration() -> None:
    assert F.audio_filter_chain(gap_ms=240).endswith("apad=pad_dur=0.240")


def test_gain_for_is_the_distance_to_target_clamped() -> None:
    assert F.gain_for(None) is None
    assert F.gain_for(-16.0) == pytest.approx(0.0)
    assert F.gain_for(-23.0) == pytest.approx(7.0)
    assert F.gain_for(-60.0) == pytest.approx(C.RENDER_MAX_GAIN_DB)
    assert F.gain_for(0.0) == pytest.approx(-C.RENDER_MAX_GAIN_DB)


def test_the_concat_list_uses_forward_slashes_and_escapes_quotes() -> None:
    text = F.concat_list_text([Path("/out/r1/a b.mkv"), Path("/out/r1/it's.mkv")])
    assert text.splitlines()[0] == "file '/out/r1/a b.mkv'"
    assert text.splitlines()[1] == "file '/out/r1/it'\\''s.mkv'"


def test_write_concat_list_creates_parents_and_ends_with_a_newline(
    tmp_path: Path,
) -> None:
    listing = F.write_concat_list(tmp_path / "deep" / "concat.txt", [tmp_path / "a.mkv"])
    assert listing.read_text(encoding="utf-8").endswith("\n")


def test_the_source_loudness_scan_reads_one_window_with_no_video() -> None:
    args = F.source_loudness_command(
        path=AUDIO, start_ms=600_000, duration_ms=120_000, binary="ffmpeg"
    )
    assert _pairs(args, "-ss") == ["600.000"]
    assert _pairs(args, "-t") == ["120.000"]
    assert "-vn" in args
    assert "print_format=json" in args[args.index("-af") + 1]
    assert args[-3:] == ["-f", "null", "-"]


def test_the_programme_scan_reads_the_concat_list() -> None:
    args = F.programme_loudness_command(list_file=Path("/out/r1/concat.txt"),
                                        binary="ffmpeg")
    assert _pairs(args, "-f")[0] == "concat"
    assert "-safe" in args and args[args.index("-safe") + 1] == "0"
    assert "print_format=json" in args[args.index("-af") + 1]


def test_the_measuring_passes_are_not_silenced() -> None:
    """loudnorm prints its JSON at INFO; -loglevel error would eat it."""
    for args in (
        F.source_loudness_command(
            path=AUDIO, start_ms=0, duration_ms=1_000, binary="ffmpeg"
        ),
        F.programme_loudness_command(list_file=Path("/c.txt"), binary="ffmpeg"),
    ):
        assert _pairs(args, "-loglevel") == ["info"]
    encode = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=500,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(encode, "-loglevel") == ["error"]


def test_the_final_pass_copies_video_and_always_re_encodes_audio() -> None:
    args = F.concat_command(
        list_file=Path("/out/r1/concat.txt"),
        out_path=Path("/out/r1/output.mp4"),
        binary="ffmpeg",
    )
    assert _pairs(args, "-c:v") == ["copy"]
    assert _pairs(args, "-c:a") == [C.RENDER_AUDIO_CODEC]
    assert "-af" not in args
    assert _pairs(args, "-movflags") == ["+faststart"]
    assert args[-1] == str(Path("/out/r1/output.mp4"))


def test_the_final_pass_feeds_the_measurement_back_in_for_the_second_pass() -> None:
    measured = F.LoudnormMeasurement(
        input_i=-27.24, input_tp=-8.51, input_lra=6.90,
        input_thresh=-37.51, target_offset=0.02,
    )
    args = F.concat_command(
        list_file=Path("/out/r1/concat.txt"),
        out_path=Path("/out/r1/output.mp4"),
        measured=measured,
        binary="ffmpeg",
    )
    graph = args[args.index("-af") + 1]
    assert f"I={C.RENDER_TARGET_LUFS}" in graph
    assert f"TP={C.RENDER_TRUE_PEAK_DBTP}" in graph
    assert f"LRA={C.RENDER_LOUDNESS_RANGE_LU}" in graph
    assert "measured_I=-27.24" in graph
    assert "measured_TP=-8.51" in graph
    assert "measured_LRA=6.9" in graph
    assert "measured_thresh=-37.51" in graph
    assert "offset=0.02" in graph
    assert "linear=true" in graph
