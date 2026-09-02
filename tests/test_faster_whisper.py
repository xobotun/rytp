"""Tests for ``rytp.transcribe.faster_whisper``."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rytp.transcribe.faster_whisper import FasterWhisperEngine


def test_engine_class_is_registered() -> None:
    from rytp import engines

    assert "faster-whisper" in engines.STTS
    assert engines.STTS["faster-whisper"] is FasterWhisperEngine


def test_engine_metadata() -> None:
    assert FasterWhisperEngine.name == "faster-whisper"
    assert FasterWhisperEngine.requires_hf_token is False


def test_engine_init_raises_import_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate faster_whisper not being importable.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(ImportError, match="pip install rytp\\[stt\\]"):
        FasterWhisperEngine()


def test_engine_transcribe_yields_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a mocked WhisperModel, transcribe() must yield Word objects with ms timings."""

    class FakeWord:
        def __init__(self, start: float, end: float, word: str, probability: float):
            self.start = start
            self.end = end
            self.word = word
            self.probability = probability

    class FakeSegment:
        def __init__(self, words: list[FakeWord]) -> None:
            self.words = words

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model_size = model_size

        def transcribe(self, audio_path: str, **kwargs: Any):
            seg = FakeSegment(
                [
                    FakeWord(0.0, 0.5, " hello", 0.9),
                    FakeWord(0.5, 1.234, "world", 0.85),
                    FakeWord(1.5, 2.0, " ! ", 0.99),
                ]
            )
            return iter([seg]), {"language": "en"}

    fake_module = type(sys)("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    engine = FasterWhisperEngine(model_size="small", device="cpu", compute_type="int8")
    words = list(engine.transcribe(tmp_path / "audio.wav"))

    assert len(words) == 3
    assert words[0].start_ms == 0
    assert words[0].end_ms == 500
    assert words[0].text == "hello"
    assert words[0].confidence == pytest.approx(0.9)
    assert words[1].start_ms == 500
    assert words[1].end_ms == 1234
    assert words[2].text == "!"  # whitespace stripped


def test_engine_transcribe_passes_language_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen_kwargs: dict[str, Any] = {}

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: Any) -> None:
            pass

        def transcribe(self, audio_path: str, **kwargs: Any):
            seen_kwargs.update(kwargs)
            return iter([]), {"language": "en"}

    fake_module = type(sys)("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    engine = FasterWhisperEngine()
    list(engine.transcribe(tmp_path / "x.wav", language="ru"))
    assert seen_kwargs.get("language") == "ru"
    assert seen_kwargs.get("word_timestamps") is True