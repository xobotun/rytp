"""Synthesised audio for tests: a tone, a gap, a tone.

A real recording never has a known correct boundary. A generated one does,
which is why every audio test in Part 3 builds its input here.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from rytp import constants as C

SR = C.WAV_SAMPLE_RATE_HZ


def tone(ms: int, *, freq_hz: float = 220.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    """A sine of ``ms`` milliseconds. 220 Hz sits inside the speech pitch range."""
    n = int(ms * sr / 1000)
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def silence(ms: int, *, amp: float = 0.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Digital silence, or a faint noise floor when ``amp`` is positive.

    A real recording is never digitally silent; a faint floor keeps the tests
    honest about percentile noise-floor estimation.
    """
    n = int(ms * sr / 1000)
    if amp <= 0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, *, sr: int = SR) -> Path:
    """Write 16 kHz mono 16-bit PCM — the one format this pipeline reads."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sr)
        writer.writeframes(pcm.tobytes())
    return path


def tone_gap_tone(
    *,
    lead_ms: int = 200,
    gap_ms: int = 120,
    tail_ms: int = 200,
    floor_amp: float = 0.001,
    sr: int = SR,
) -> tuple[np.ndarray, int, int]:
    """Two words with a measured silence between them.

    Returns ``(samples, gap_start_ms, gap_end_ms)`` — the interval a refined
    boundary is required to land in.
    """
    samples = concat(
        tone(lead_ms, sr=sr),
        silence(gap_ms, amp=floor_amp, sr=sr, seed=1),
        tone(tail_ms, sr=sr),
    )
    return samples, lead_ms, lead_ms + gap_ms
