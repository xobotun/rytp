"""Tests for ``rytp.loudnorm``."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rytp import loudnorm


def test_parse_loudnorm_json_extracts_values() -> None:
    stderr = """[Parsed_loudnorm_0 @ 0x123] 
[Parsed_loudnorm_0 @ 0x123] 
{
\t"input_i" : "-23.81",
\t"input_tp" : "-7.04",
\t"input_lra" : "0.50",
\t"input_thresh" : "-33.81",
\t"output_i" : "-23.40",
\t"output_tp" : "-7.00",
\t"output_lra" : "0.10",
\t"output_thresh" : "-33.40",
\t"normalization_type" : "dynamic",
\t"target_offset" : "-0.41"
}
"""
    parsed = loudnorm._parse_loudnorm_json(stderr)
    assert parsed["input_i"] == pytest.approx(-23.81)
    assert parsed["input_tp"] == pytest.approx(-7.04)
    assert parsed["target_offset"] == pytest.approx(-0.41)


def test_parse_loudnorm_json_raises_on_missing() -> None:
    with pytest.raises(RuntimeError, match="could not parse"):
        loudnorm._parse_loudnorm_json("nothing useful")


def test_loudnorm_clip_calls_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With ffmpeg mocked, loudnorm_clip runs both passes and writes the output."""

    # Pretend ffmpeg is available.
    monkeypatch.setattr(loudnorm.shutil, "which", lambda _: "ffmpeg")

    measure_stderr = (
        '{ "input_i" : "-23.0", "input_tp" : "-7.0", '
        '"input_lra" : "0.5", "input_thresh" : "-33.0", '
        '"target_offset" : "-0.4" }'
    )

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # The measure pass goes to /dev/null; the apply pass writes the
        # output file.
        out_arg = cmd[-1]
        if out_arg != "-":
            Path(out_arg).write_bytes(b"x")
        result = MagicMock()
        result.returncode = 0
        result.stderr = measure_stderr if "null" in cmd else ""
        result.stdout = ""
        return result

    monkeypatch.setattr(loudnorm.subprocess, "run", fake_run)

    src = tmp_path / "src.wav"
    src.write_bytes(b"x")
    out = loudnorm.loudnorm_clip(
        src, tmp_path / "out", clip_id=7, start_ms=0, end_ms=2000
    )
    assert out == tmp_path / "out" / "7.m4a"
    assert len(calls) == 2  # measure + apply
    # First call: measure to /dev/null
    assert calls[0][-1] == "-"
    # Second call: apply to the output file
    assert calls[1][-1].endswith("7.m4a")
    # Apply uses the measured values
    apply_filter = next(a for a in calls[1] if a.startswith("loudnorm="))
    assert "measured_I=-23.0" in apply_filter
    assert "offset=-0.4" in apply_filter


def test_loudnorm_clip_skips_when_output_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(loudnorm.shutil, "which", lambda _: "ffmpeg")
    src = tmp_path / "src.wav"
    src.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "5.m4a"
    existing.write_bytes(b"already there")

    def fail(*a, **kw):
        pytest.fail("subprocess.run should not be called when output exists")

    monkeypatch.setattr(loudnorm.subprocess, "run", fail)
    out = loudnorm.loudnorm_clip(
        src, out_dir, clip_id=5, start_ms=0, end_ms=1000
    )
    assert out == existing


def test_loudnorm_clip_raises_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(loudnorm.shutil, "which", lambda _: None)
    src = tmp_path / "src.wav"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        loudnorm.loudnorm_clip(
            src, tmp_path / "out", clip_id=1, start_ms=0, end_ms=1000
        )