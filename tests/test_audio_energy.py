"""Sample access and short-time energy."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from rytp import constants as C
from rytp.audio.energy import (
    AudioFormatError,
    frame_rms,
    index_to_ms,
    ms_to_index,
    read_wav_mono,
    to_db,
)
from tests.synth_audio import SR, silence, tone, write_wav


def test_read_wav_mono_round_trips_a_written_file(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", tone(100, amp=0.5))
    samples, sr = read_wav_mono(path)
    assert sr == SR
    assert samples.shape == (1600,)
    assert samples.dtype == np.float32
    assert 0.49 < float(np.max(samples)) <= 0.5


def test_read_wav_mono_rejects_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SR)
        writer.writeframes(b"\x00\x00\x00\x00" * 100)
    with pytest.raises(AudioFormatError) as excinfo:
        read_wav_mono(path)
    assert "mono" in str(excinfo.value)


def test_read_wav_mono_rejects_8_bit(tmp_path: Path) -> None:
    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(SR)
        writer.writeframes(b"\x80" * 100)
    with pytest.raises(AudioFormatError):
        read_wav_mono(path)


def test_ms_and_index_round_trip() -> None:
    assert ms_to_index(1000, SR) == SR
    assert index_to_ms(SR, SR) == 1000
    assert ms_to_index(1, SR) == 16


def test_frame_rms_of_a_sine_is_the_amplitude_over_root_two() -> None:
    samples = tone(200, amp=0.4)
    rms = frame_rms(samples, ms_to_index(20, SR), ms_to_index(10, SR))
    assert rms.size > 15
    assert np.allclose(rms, 0.4 / np.sqrt(2), atol=0.02)


def test_frame_rms_of_digital_silence_is_zero() -> None:
    rms = frame_rms(silence(50), ms_to_index(20, SR), ms_to_index(10, SR))
    assert rms.size > 0
    assert float(np.max(rms)) == 0.0


def test_frame_rms_of_too_short_input_is_empty() -> None:
    assert frame_rms(tone(1), ms_to_index(20, SR), ms_to_index(10, SR)).size == 0


def test_to_db_floors_digital_silence_instead_of_returning_minus_infinity() -> None:
    db = to_db(np.array([0.0, 1.0], dtype=np.float32))
    assert float(db[0]) == C.DB_FLOOR
    assert float(db[1]) == pytest.approx(0.0, abs=1e-6)
