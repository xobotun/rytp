"""Tests for registering pre-downloaded audio/video pairs.

When the user already has separate audio and video files on disk
(typical of a manual yt-dlp run with a ``+`` format selector, or
files re-pulled from a previous download), they need a way to bring
those files into the DB without going through ``rytp download``.
This module covers that path: :func:`rytp.channels.register_local_video_with_separate_audio`
+ the ``--audio`` option on the ``rytp videos add`` CLI subcommand.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp.channels import register_local_video_with_separate_audio


def _touch(p: Path, data: bytes = b"x") -> Path:
    p.write_bytes(data)
    return p


def test_register_local_pair_stores_both_paths(
    db, tmp_path: Path
) -> None:
    """Register a video + audio pair; both paths land in the row."""
    video = _touch(tmp_path / "movie.mp4")
    audio = _touch(tmp_path / "audio.webm", b"\x00" * 16)

    vid = register_local_video_with_separate_audio(db, video, audio)

    row = db.conn.execute(
        "SELECT * FROM videos WHERE id = ?", (vid,)
    ).fetchone()
    assert row["source"] == "local"
    assert row["downloaded"] == 1
    assert row["downloaded_path"] == str(video.resolve())
    assert row["downloaded_audio_path"] == str(audio.resolve())
    # Title defaults to the video file's stem.
    assert row["title"] == "movie"


def test_register_local_pair_custom_title(db, tmp_path: Path) -> None:
    """Title override should land in the row."""
    video = _touch(tmp_path / "video.mp4")
    audio = _touch(tmp_path / "audio.webm")
    vid = register_local_video_with_separate_audio(
        db, video, audio, title="My Video"
    )
    row = db.conn.execute(
        "SELECT title FROM videos WHERE id = ?", (vid,)
    ).fetchone()
    assert row["title"] == "My Video"


def test_register_local_pair_with_youtube_id_is_idempotent(
    db, tmp_path: Path
) -> None:
    """Re-registering the same youtube_id updates the same row, not a new one."""
    video1 = _touch(tmp_path / "v1.mp4")
    audio1 = _touch(tmp_path / "a1.webm")

    vid = register_local_video_with_separate_audio(
        db, video1, audio1, youtube_id="sample_youtube_id"
    )
    assert vid > 0

    # Different files, same youtube_id → same row, updated paths.
    video2 = _touch(tmp_path / "v2.mp4")
    audio2 = _touch(tmp_path / "a2.webm")
    vid2 = register_local_video_with_separate_audio(
        db, video2, audio2, youtube_id="sample_youtube_id"
    )
    assert vid2 == vid

    row = db.conn.execute(
        "SELECT downloaded_path, downloaded_audio_path FROM videos WHERE id = ?",
        (vid,),
    ).fetchone()
    assert row["downloaded_path"] == str(video2.resolve())
    assert row["downloaded_audio_path"] == str(audio2.resolve())


def test_register_local_pair_raises_on_missing_video(db, tmp_path: Path) -> None:
    video = tmp_path / "missing.mp4"  # not touched
    audio = _touch(tmp_path / "audio.webm")
    with pytest.raises(FileNotFoundError, match="video file"):
        register_local_video_with_separate_audio(db, video, audio)


def test_register_local_pair_raises_on_missing_audio(db, tmp_path: Path) -> None:
    video = _touch(tmp_path / "video.mp4")
    audio = tmp_path / "missing.webm"  # not touched
    with pytest.raises(FileNotFoundError, match="audio file"):
        register_local_video_with_separate_audio(db, video, audio)


# ---------------------------------------------------------------------------
# CLI: rytp videos add <video> --audio <audio>
# ---------------------------------------------------------------------------


def test_videos_add_cli_with_audio_path(
    db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    """End-to-end via the CLI: ``rytp videos add VIDEO --audio AUDIO``."""
    from typer.testing import CliRunner
    from rytp.cli import app

    video = _touch(tmp_path / "movie.mp4")
    audio = _touch(tmp_path / "audio.webm")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "videos",
            "add",
            str(video),
            "--audio",
            str(audio),
            "--youtube-id",
            "sample_youtube_id",
            "--title",
            "Смерть чиновника",
        ],
    )
    assert result.exit_code == 0, result.output

    row = db.conn.execute(
        "SELECT source, downloaded, downloaded_path, downloaded_audio_path, title, youtube_id "
        "FROM videos WHERE downloaded = 1"
    ).fetchone()
    assert row is not None
    assert row["source"] == "local"
    assert row["downloaded_path"] == str(video.resolve())
    assert row["downloaded_audio_path"] == str(audio.resolve())
    assert row["title"] == "Смерть чиновника"
    assert row["youtube_id"] == "sample_youtube_id"


def test_videos_add_cli_audio_missing_raises(
    db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    """If the audio file doesn't exist on disk, ``--audio`` should fail loudly."""
    from typer.testing import CliRunner
    from rytp.cli import app

    video = _touch(tmp_path / "movie.mp4")
    audio = tmp_path / "missing.webm"  # not touched

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["videos", "add", str(video), "--audio", str(audio)],
    )
    assert result.exit_code != 0
    assert "audio file not found" in (result.output + str(result.exception or ""))


# ---------------------------------------------------------------------------
# Audio extraction: separate audio is preferred when present
# ---------------------------------------------------------------------------


def test_extract_audio_prefers_downloaded_audio_path(
    db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    """End-to-end-ish: a row whose ``downloaded_audio_path`` is set
    should cause ``extract_audio`` to read from that file, not the
    video file.

    We mock :func:`subprocess.run` to capture the ffmpeg command and
    verify the audio file is the input.
    """
    import subprocess

    from rytp.transcribe.extract import extract_audio

    video = _touch(tmp_path / "video.mp4", b"video-bytes")
    audio = _touch(tmp_path / "audio.webm", b"audio-bytes")

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Touch the output file so extract_audio thinks it succeeded.
        Path(cmd[-1]).write_bytes(b"RIFF....")
        m = type("R", (), {"returncode": 0, "stderr": ""})()
        return m

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Use a tmp output dir so we don't pollute the test sandbox.
    out_dir = tmp_path / "extracted"
    out = extract_audio(video, out_dir, video_id="vid-1", audio_path=audio)
    assert out == out_dir / "vid-1.wav"
    assert str(audio) in captured["cmd"]
    assert str(video) not in captured["cmd"]