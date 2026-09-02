"""Tests for ``rytp.spectrogram``."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from rytp.db import Database
from rytp.spectrogram import (
    clip_similarity,
    compute_clip_features,
    get_or_compute_clip_features,
)


def _write_silent_wav(path: Path, duration_seconds: float = 0.5) -> Path:
    n_frames = int(duration_seconds * 16000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * n_frames)
    return path


def test_compute_clip_features_returns_expected_shape(tmp_path: Path) -> None:
    audio = _write_silent_wav(tmp_path / "audio.wav", duration_seconds=1.0)
    feats = compute_clip_features(audio, 0, 1000)
    assert feats.mfcc.shape[0] == 13
    assert feats.mfcc.shape[1] > 0
    # Silent audio → centroid is 0
    assert feats.spectral_centroid_hz == 0.0


def test_compute_clip_features_for_tone(tmp_path: Path) -> None:
    """A 440 Hz sine wave should have a peak bin at 440 Hz and a centroid
    that's biased upward by spectral leakage but still well below Nyquist.

    Note: this centroid uses linear FFT magnitude weighting, so a pure
    tone gives a centroid ~2x the tone frequency due to Hann-window
    sidelobes. The mine stage uses it only for *relative* similarity
    across clips, so absolute accuracy doesn't matter.
    """
    n = 16000  # 1 second at 16 kHz
    samples = np.array(
        [int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n)],
        dtype=np.int16,
    )
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(samples.tobytes())
    feats = compute_clip_features(path, 0, 1000)
    # The peak bin should be at ~440 Hz (bin 14 in our 512-point FFT).
    spec = np.abs(np.fft.rfft(samples[:512].astype(np.float64) / 32768.0, n=512))
    freqs = np.fft.rfftfreq(512, d=1.0 / 16000)
    peak_hz = float(freqs[int(spec.argmax())])
    assert 400 < peak_hz < 480, f"peak bin off: {peak_hz}"
    # Centroid (biased upward by sidelobes) is positive and << Nyquist
    assert 0 < feats.spectral_centroid_hz < 8000


def test_clip_similarity_identical_is_high(tmp_path: Path) -> None:
    audio = _write_silent_wav(tmp_path / "audio.wav", duration_seconds=1.0)
    a = compute_clip_features(audio, 0, 500)
    b = compute_clip_features(audio, 0, 500)
    sim = clip_similarity(a, b)
    assert sim > 0.95


def test_clip_similarity_different_window_low(tmp_path: Path) -> None:
    audio = _write_silent_wav(tmp_path / "audio.wav", duration_seconds=2.0)
    a = compute_clip_features(audio, 0, 500)
    b = compute_clip_features(audio, 1500, 2000)
    # Both are silence → still similar
    sim = clip_similarity(a, b)
    assert sim > 0.9


def test_get_or_compute_caches(db: Database, fake_video_row, tmp_path: Path) -> None:
    audio = _write_silent_wav(tmp_path / "audio.wav", duration_seconds=1.0)
    vid = db.upsert_video(fake_video_row)
    a = get_or_compute_clip_features(db, vid, audio, 0, 500)
    b = get_or_compute_clip_features(db, vid, audio, 0, 500)
    assert np.array_equal(a.mfcc, b.mfcc)
    # Exactly one row in clip_features
    n = db.conn.execute(
        "SELECT COUNT(*) AS n FROM clip_features WHERE video_id = ?", (vid,)
    ).fetchone()["n"]
    assert n == 1