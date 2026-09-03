"""Tests for ``rytp.transcribe.extract``."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rytp import config
from rytp.transcribe.extract import (
    ExtractionError,
    FfmpegNotFoundError,
    extract_audio,
)


def _fake_completed_process(returncode: int = 0, stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stderr = stderr
    return p


def test_extract_audio_raises_when_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: None)
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    with pytest.raises(FfmpegNotFoundError):
        extract_audio(video, tmp_path / "out", video_id="abc")


def test_extract_audio_raises_when_video_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "out", video_id="abc")


def test_extract_audio_calls_ffmpeg_with_correct_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    out_dir = tmp_path / "out"
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        # Touch the output file as if ffmpeg had written it
        Path(cmd[-1]).write_bytes(b"RIFF....")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_path = extract_audio(video, out_dir, video_id="vid-1")
    assert out_path == out_dir / "vid-1.wav"
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    assert str(video) in cmd
    assert "16000" in cmd  # -ar 16000
    assert "1" in cmd  # -ac 1
    assert "wav" in cmd


def test_extract_audio_skips_when_output_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "vid-1.wav"
    existing.write_bytes(b"already there")

    def fail_run(*args, **kwargs):
        pytest.fail("subprocess.run should not be called when output exists")

    monkeypatch.setattr(subprocess, "run", fail_run)
    out_path = extract_audio(video, out_dir, video_id="vid-1")
    assert out_path == existing


def test_extract_audio_overwrite_reruns_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "vid-1.wav").write_bytes(b"old")

    called = {"count": 0}

    def fake_run(cmd, **kwargs):
        called["count"] += 1
        Path(cmd[-1]).write_bytes(b"new")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    extract_audio(video, out_dir, video_id="vid-1", overwrite=True)
    assert called["count"] == 1


def test_extract_audio_raises_on_ffmpeg_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _fake_completed_process(returncode=1, stderr="bad input"),
    )
    with pytest.raises(ExtractionError, match="ffmpeg extract failed"):
        extract_audio(video, tmp_path / "out", video_id="vid-1")


def test_extract_audio_raises_when_ffmpeg_missing_at_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FfmpegNotFoundError):
        extract_audio(video, tmp_path / "out", video_id="vid-1")


def test_extract_audio_uses_separate_audio_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When audio_path is provided and exists, it should be used instead of video_path."""
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "audio.webm"
    audio.write_bytes(b"audio")
    out_dir = tmp_path / "out"
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFF....")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_path = extract_audio(video, out_dir, video_id="vid-1", audio_path=audio)
    assert out_path == out_dir / "vid-1.wav"
    cmd = captured["cmd"]
    # The audio file should be the input, not the video file
    assert str(audio) in cmd
    assert str(video) not in cmd


def test_extract_audio_falls_back_to_video_when_audio_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When audio_path is provided but doesn't exist, fall back to video_path."""
    monkeypatch.setattr(config, "ffmpeg_binary", lambda: "ffmpeg")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "audio.webm"  # doesn't exist
    out_dir = tmp_path / "out"
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFF....")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_path = extract_audio(video, out_dir, video_id="vid-1", audio_path=audio)
    assert out_path == out_dir / "vid-1.wav"
    cmd = captured["cmd"]
    # The video file should be the input since audio doesn't exist
    assert str(video) in cmd