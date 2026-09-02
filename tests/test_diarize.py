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