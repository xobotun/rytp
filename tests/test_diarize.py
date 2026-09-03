"""Tests for the diarize subsystem."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rytp import engines
from rytp.diarize import DiarSegment, Diarizer, NullDiarizer, PyannoteDiarizer
from rytp.diarize.base import NullDiarizer as NullFromBase


def test_null_diarizer_is_registered() -> None:
    assert "none" in engines.DIARIZERS
    assert engines.DIARIZERS["none"] is NullDiarizer


def test_null_diarizer_in_base_module_is_same_class() -> None:
    # Re-export consistency: the diarize package and the base module
    # expose the same class.
    assert NullDiarizer is NullFromBase


def test_null_diarizer_diarize_yields_speaker_00() -> None:
    segs = list(NullDiarizer().diarize(Path("/tmp/audio.wav")))
    assert len(segs) == 1
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].start_ms == 0


def test_pyannote_engine_is_registered() -> None:
    assert "pyannote" in engines.DIARIZERS
    assert engines.DIARIZERS["pyannote"] is PyannoteDiarizer


def test_pyannote_requires_hf_token() -> None:
    assert PyannoteDiarizer.requires_hf_token is True


def test_pyannote_init_raises_when_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(ImportError, match="pip install rytp\\[pyannote\\]"):
        PyannoteDiarizer()


def test_pyannote_init_raises_when_no_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, use_auth_token=None):
            return cls()

    fake_module = type(sys)("pyannote.audio")
    fake_module.Pipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    fake_pkg = type(sys)("pyannote")
    fake_pkg.audio = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        PyannoteDiarizer()


def test_pyannote_init_succeeds_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    class FakePipeline:
        last_init: dict = {}

        @classmethod
        def from_pretrained(cls, model_id, use_auth_token=None):
            cls.last_init = {
                "model_id": model_id,
                "use_auth_token": use_auth_token,
            }
            return cls()

    fake_module = type(sys)("pyannote.audio")
    fake_module.Pipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    fake_pkg = type(sys)("pyannote")
    fake_pkg.audio = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)

    diarizer = PyannoteDiarizer()
    assert FakePipeline.last_init["use_auth_token"] == "hf_test_token"


def test_pyannote_diarize_yields_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After init succeeds, diarize() must yield DiarSegment objects."""

    class FakeAnnotation:
        def itertracks(self, yield_label=True):
            return iter(
                [
                    (_FakeSegment(0.0, 1.5), None, "SPEAKER_00"),
                    (_FakeSegment(1.5, 3.0), None, "SPEAKER_01"),
                ]
            )

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, use_auth_token=None):
            return cls()

        def __call__(self, audio_path):
            return FakeAnnotation()

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    fake_module = type(sys)("pyannote.audio")
    fake_module.Pipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_module)
    fake_pkg = type(sys)("pyannote")
    fake_pkg.audio = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)

    diarizer = PyannoteDiarizer()
    segs = list(diarizer.diarize(Path("/tmp/audio.wav")))
    assert len(segs) == 2
    assert segs[0] == DiarSegment(start_ms=0, end_ms=1500, speaker="SPEAKER_00")
    assert segs[1] == DiarSegment(start_ms=1500, end_ms=3000, speaker="SPEAKER_01")


class _FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


# --- EnergyDiarizer tests --------------------------------------------------


def test_energy_diarizer_is_registered() -> None:
    """The energy diarizer should be registered with name 'energy'."""
    assert "energy" in engines.DIARIZERS


def test_energy_diarizer_validates_sensitivity() -> None:
    """Sensitivity must be in [0.0, 1.0]."""
    from rytp.diarize.energy import EnergyDiarizer

    with pytest.raises(ValueError, match="sensitivity must be in"):
        EnergyDiarizer(sensitivity=-0.1)

    with pytest.raises(ValueError, match="sensitivity must be in"):
        EnergyDiarizer(sensitivity=1.5)


def test_energy_diarizer_does_not_require_hf_token() -> None:
    """Energy diarizer should not require HF_TOKEN."""
    from rytp.diarize.energy import EnergyDiarizer

    assert EnergyDiarizer.requires_hf_token is False
    assert EnergyDiarizer.name == "energy"


def test_energy_diarizer_handles_synthetic_audio(tmp_path: Path) -> None:
    """Energy diarizer should produce segments for a synthetic WAV file."""
    import wave
    import numpy as np

    from rytp.diarize.energy import EnergyDiarizer

    # Create a synthetic WAV with two distinct "speakers" (different energy levels)
    audio_path = tmp_path / "test.wav"
    sample_rate = 16000
    duration_s = 3.0
    n_samples = int(sample_rate * duration_s)

    # Create audio with alternating loud and quiet sections
    samples = np.zeros(n_samples, dtype=np.int16)
    # First speaker: loud (0-1.5s)
    samples[: sample_rate] = np.random.randint(-10000, 10000, sample_rate).astype(np.int16)
    # Silence (1.5-1.7s)
    # Second speaker: quiet (1.7-3.0s)
    samples[int(1.7 * sample_rate) :] = np.random.randint(-3000, 3000, int(1.3 * sample_rate)).astype(np.int16)

    with wave.open(str(audio_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())

    diarizer = EnergyDiarizer(sensitivity=0.5)
    segments = list(diarizer.diarize(audio_path))

    # Should find at least one segment
    assert len(segments) > 0
    # All segments should have valid time ranges
    for seg in segments:
        assert seg.start_ms >= 0
        assert seg.end_ms > seg.start_ms
        assert seg.speaker.startswith("SPEAKER_")


def test_energy_diarizer_sensitivity_affects_speaker_count(tmp_path: Path) -> None:
    """Lower sensitivity should generally produce fewer speakers (more merging)."""
    import wave
    import numpy as np

    from rytp.diarize.energy import EnergyDiarizer

    # Create a synthetic WAV with multiple distinct segments
    audio_path = tmp_path / "test.wav"
    sample_rate = 16000
    duration_s = 10.0
    n_samples = int(sample_rate * duration_s)

    np.random.seed(42)  # Reproducible
    samples = np.zeros(n_samples, dtype=np.int16)

    # Create 4 distinct segments with different characteristics
    for i in range(4):
        start = int(i * 2.5 * sample_rate)
        end = int((i + 2) * sample_rate)
        if end > n_samples:
            end = n_samples
        seg_len = end - start
        if seg_len <= 0:
            continue
        if i % 2 == 0:
            # High energy
            samples[start:end] = np.random.randint(-15000, 15000, seg_len).astype(np.int16)
        else:
            # Lower energy
            samples[start:end] = np.random.randint(-5000, 5000, seg_len).astype(np.int16)

    with wave.open(str(audio_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())

    # Test with different sensitivity levels
    high_sens_speakers = len(set(
        s.speaker for s in EnergyDiarizer(sensitivity=1.0).diarize(audio_path)
    ))
    low_sens_speakers = len(set(
        s.speaker for s in EnergyDiarizer(sensitivity=0.0).diarize(audio_path)
    ))

    # Lower sensitivity should produce fewer or equal speakers
    assert low_sens_speakers <= high_sens_speakers, (
        f"Low sensitivity ({low_sens_speakers}) should produce <= "
        f"high sensitivity ({high_sens_speakers}) speakers"
    )