"""Tests for ``rytp.transcribe.chunking``."""
from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from rytp.transcribe.chunking import (
    Chunk,
    ProbeError,
    chunk_audio,
    probe_duration_ms,
)


def _write_wav(path: Path, *, duration_seconds: float, rate: int = 16000) -> None:
    """Write a silent mono int16 WAV of the requested length."""
    n_frames = int(duration_seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


def test_probe_duration_ms_against_synthetic_wav(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    _write_wav(p, duration_seconds=2.5)
    # Without ffprobe on PATH, this falls back to the wave-stdlib path.
    # Result should be ~2500 ms (off-by-one is fine for integer math).
    dur = probe_duration_ms(p)
    assert 2490 < dur <= 2500


def test_probe_duration_ms_raises_for_unparseable(tmp_path: Path) -> None:
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"not a wav")
    with pytest.raises(ProbeError):
        probe_duration_ms(p)


def test_chunk_audio_below_threshold_returns_single_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure no ffmpeg/ffprobe is found so we hit the stdlib path.
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "short.wav"
    _write_wav(p, duration_seconds=60)  # 1 minute → below 30-min threshold
    chunks = chunk_audio(p)
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].start_ms == 0
    assert chunks[0].end_ms == 60_000
    assert chunks[0].audio_path == p


def test_chunk_audio_above_threshold_returns_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "long.wav"
    _write_wav(p, duration_seconds=60 * 70)  # 70 minutes
    chunks = chunk_audio(p, chunk_length_min=25, overlap_min=5)
    # 25-min chunks, 5-min overlap, 20-min step: chunks at 0, 20, 40, 60 → 4 chunks
    assert len(chunks) == 4
    assert chunks[0].start_ms == 0
    assert chunks[0].end_ms == 25 * 60 * 1000
    assert chunks[1].start_ms == 20 * 60 * 1000
    assert chunks[1].end_ms == 45 * 60 * 1000
    assert chunks[3].end_ms == 70 * 60 * 1000  # clamped to duration


def test_chunk_audio_with_out_dir_slices_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "long.wav"
    _write_wav(p, duration_seconds=60 * 35)  # just over threshold
    out = tmp_path / "chunks"
    chunks = chunk_audio(p, chunk_length_min=20, overlap_min=5, out_dir=out)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.audio_path.exists()
        assert c.audio_path.parent == out
        # Chunk file must be shorter than the source
        assert c.audio_path.stat().st_size < p.stat().st_size


def test_chunk_audio_zero_duration_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "empty.wav"
    _write_wav(p, duration_seconds=0)
    chunks = chunk_audio(p)
    assert chunks == []


def test_chunk_audio_at_threshold_is_single_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "edge.wav"
    _write_wav(p, duration_seconds=60 * 30)  # exactly at threshold
    chunks = chunk_audio(p)
    assert len(chunks) == 1
    assert chunks[0].end_ms == 30 * 60 * 1000


def test_chunk_audio_invalid_overlap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    p = tmp_path / "long.wav"
    _write_wav(p, duration_seconds=60 * 60)
    with pytest.raises(ValueError, match="exceed overlap"):
        chunk_audio(p, chunk_length_min=5, overlap_min=5)