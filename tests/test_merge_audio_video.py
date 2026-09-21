"""Tests for the audio/video merge in :mod:`rytp.download.ytdlp`.

When ``yt-dlp`` downloads audio and video as separate files (because
the format selector contains ``+``), rytp merges them into a single
container using ffmpeg. These tests cover the merge function directly.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rytp.download.ytdlp import RealYtDlpRunner


def _fake_completed_process(returncode: int = 0, stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stderr = stderr
    return p


def test_merge_audio_video_constructs_expected_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge command takes both inputs, maps the video from input 0
    and the audio from input 1, copies the video stream and re-encodes
    the audio to AAC.
    """
    monkeypatch.setattr(shutil, "which", lambda _: "ffmpeg")

    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.webm"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate ffmpeg producing the merged output file.
        Path(cmd[-1]).write_bytes(b"merged")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = RealYtDlpRunner.__new__(RealYtDlpRunner)  # bypass __init__
    out = runner._merge_audio_video(tmp_path, video, audio, youtube_id="vid-1")

    assert out == tmp_path / "v_merged.mp4"
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    # Two inputs in the expected order.
    assert str(video) in cmd
    assert str(audio) in cmd
    # Stream mapping: video from input 0, audio from input 1.
    assert "0:v:0" in cmd
    assert "1:a:0" in cmd
    # Video stream copied; audio re-encoded to AAC for compatibility.
    assert "copy" in cmd
    assert "aac" in cmd
    # -shortest so the merged file ends with whichever stream is shorter.
    assert "-shortest" in cmd


def test_merge_audio_video_raises_when_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    runner = RealYtDlpRunner.__new__(RealYtDlpRunner)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        runner._merge_audio_video(
            tmp_path,
            tmp_path / "v.mp4",
            tmp_path / "a.webm",
            youtube_id="vid-1",
        )


def test_merge_audio_video_raises_on_ffmpeg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _fake_completed_process(returncode=1, stderr="bad"),
    )
    runner = RealYtDlpRunner.__new__(RealYtDlpRunner)
    with pytest.raises(RuntimeError, match="ffmpeg merge failed"):
        runner._merge_audio_video(
            tmp_path,
            tmp_path / "v.mp4",
            tmp_path / "a.webm",
            youtube_id="vid-1",
        )


def test_merge_audio_video_preserves_cyrillic_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filenames with non-ASCII characters must round-trip cleanly.

    The user's sample run produced files like
    ``Смерть чиновника [sample_youtube_id].f311.mp4`` — Russian title plus
    the yt-dlp suffix — and the merge output should keep that
    title in the merged name.
    """
    monkeypatch.setattr(shutil, "which", lambda _: "ffmpeg")

    video = tmp_path / "Смерть чиновника [sample_youtube_id].f311.mp4"
    audio = tmp_path / "Смерть чиновника [sample_youtube_id].f251-1.webm"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"merged")
        return _fake_completed_process(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = RealYtDlpRunner.__new__(RealYtDlpRunner)
    out = runner._merge_audio_video(tmp_path, video, audio, youtube_id="sample_youtube_id")

    assert out.name == "Смерть чиновника [sample_youtube_id].f311_merged.mp4"
    assert out.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
def test_merge_audio_video_end_to_end(tmp_path: Path) -> None:
    """Real ffmpeg round-trip: generate a tiny video + audio via lavfi,
    merge them, verify the merged container has both streams."""
    video = tmp_path / "tiny.mp4"
    audio = tmp_path / "silent.webm"
    merged = tmp_path / "tiny_merged.mp4"

    # 1-second color test pattern + 1-second silent audio.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=320x240:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t",
            "1",
            "-c:a",
            "libopus",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )

    runner = RealYtDlpRunner.__new__(RealYtDlpRunner)
    out = runner._merge_audio_video(tmp_path, video, audio, youtube_id="vid-1")

    assert out.exists()
    assert out.stat().st_size > 0

    # Probe the merged file: should have both a video and an audio stream.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = probe.stdout.strip().split("\n")
    assert "video" in streams
    assert "audio" in streams