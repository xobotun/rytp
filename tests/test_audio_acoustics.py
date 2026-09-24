"""The per-video acoustic fingerprint: five numbers, one pass."""
from __future__ import annotations

import numpy as np

from rytp.audio.acoustics import (
    compute_acoustics,
    estimate_f0,
    noise_floor_db,
    reverb_proxy,
    spectral_tilt,
)
from rytp.audio.vad import SpeechSegment
from tests.synth_audio import SR, concat, silence, tone


def test_pitch_of_a_known_sine_is_recovered() -> None:
    mean, spread = estimate_f0(tone(1000, freq_hz=200.0), SR)
    assert mean is not None and spread is not None
    assert abs(mean - 200.0) < 5.0
    assert spread < 5.0


def test_pitch_of_digital_silence_is_unknown() -> None:
    assert estimate_f0(silence(1000), SR) == (None, None)


def test_noise_floor_finds_the_quiet_part_not_the_loud_one() -> None:
    samples = concat(tone(500, amp=0.3), silence(500, amp=0.001, seed=9))
    floor = noise_floor_db(samples, SR)
    assert -75.0 < floor < -45.0


def test_a_dull_recording_has_a_steeper_spectral_tilt_than_a_bright_one() -> None:
    rng = np.random.default_rng(5)
    white = (0.2 * rng.standard_normal(SR)).astype(np.float32)
    dull = np.convolve(white, np.ones(16, dtype=np.float32) / 16.0, mode="same").astype(
        np.float32
    )
    bright_tilt = spectral_tilt(white, SR)
    dull_tilt = spectral_tilt(dull, SR)
    assert bright_tilt is not None and dull_tilt is not None
    assert dull_tilt < bright_tilt - 5.0


def test_a_decaying_tail_reads_as_more_reverberant_than_an_abrupt_stop() -> None:
    body = tone(400)
    tail = tone(150)
    decayed = (tail * np.exp(-np.linspace(0.0, 5.0, tail.size))).astype(np.float32)
    dry = concat(body, silence(400, amp=0.0005, seed=21))
    wet = concat(body, decayed, silence(250, amp=0.0005, seed=22))
    segment = [SpeechSegment(start_ms=0, end_ms=400)]
    dry_value = reverb_proxy(dry, SR, segment)
    wet_value = reverb_proxy(wet, SR, segment)
    assert dry_value is not None and wet_value is not None
    assert wet_value > dry_value + 10.0


def test_reverb_of_nothing_is_unknown() -> None:
    assert reverb_proxy(silence(500), SR, []) is None


def test_compute_acoustics_fills_the_row_and_is_deterministic() -> None:
    samples = concat(tone(600, freq_hz=180.0), silence(400, amp=0.001, seed=31))
    first = compute_acoustics(samples, SR, loudness_lufs=-23.4)
    second = compute_acoustics(samples, SR, loudness_lufs=-23.4)
    assert first == second
    assert first.f0_mean is not None
    assert first.noise_floor_db < 0.0
    assert first.loudness_lufs == -23.4
