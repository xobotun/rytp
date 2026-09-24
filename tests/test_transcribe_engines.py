"""Transcriber adapters: importable without their dependency, and registered."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.models import RytpError
from rytp.transcribe import registry
from rytp.transcribe.base import EngineUnavailable
from rytp.transcribe.engines.gigaam import GigaAMTranscriber, words_from_gigaam
from rytp.transcribe.engines.whisper import FasterWhisperTranscriber, words_from_segments

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


@dataclass
class _FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class _FakeSegment:
    words: list[_FakeWord]


def test_importing_the_engines_package_registers_both_names() -> None:
    import rytp.transcribe.engines  # noqa: F401

    assert "whisper" in registry.TRANSCRIBERS
    assert "gigaam" in registry.TRANSCRIBERS


def test_words_from_segments_converts_seconds_to_milliseconds() -> None:
    segments = [_FakeSegment(words=[_FakeWord(" привет", 0.0, 0.42), _FakeWord("мир", 0.5, 0.9)])]
    words = words_from_segments(segments)
    assert [(w.start_ms, w.end_ms, w.text) for w in words] == [
        (0, 420, "привет"),
        (500, 900, "мир"),
    ]
    assert words[0].confidence == pytest.approx(0.9)


def test_words_from_segments_drops_empty_tokens() -> None:
    segments = [_FakeSegment(words=[_FakeWord("   ", 0.0, 0.1), _FakeWord("да", 0.2, 0.3)])]
    assert [w.text for w in words_from_segments(segments)] == ["да"]


def test_words_from_gigaam_offsets_onto_the_whole_file_timeline() -> None:
    result = {
        "words": [
            {"word": "да", "start": 0.1, "end": 0.4},
            {"word": " ", "start": 0.4, "end": 0.5},
        ]
    }
    words = words_from_gigaam(result, 10_000)
    assert words == [{"start_ms": 10_100, "end_ms": 10_400, "text": "да", "confidence": None}]


def test_gigaam_is_out_of_process_and_ungated() -> None:
    assert GigaAMTranscriber.out_of_process is True
    assert GigaAMTranscriber.requires_hf_token is False
    assert GigaAMTranscriber.max_window_ms == C.VAD_CHUNK_MAX_MS


def test_gigaam_refuses_a_window_longer_than_its_per_call_cap(tmp_path: Path) -> None:
    engine = GigaAMTranscriber(interpreter="/no/such/python")
    with pytest.raises(RytpError) as excinfo:
        list(engine.transcribe(tmp_path / "a.wav", start_ms=0, end_ms=C.VAD_CHUNK_MAX_MS + 1))
    assert "plan_chunks" in str(excinfo.value)


def test_gigaam_defaults_to_auto_device_and_sends_it_to_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.transcribe import engines as engines_module

    seen: dict[str, object] = {}

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"words": [], "device": "cuda"}

    monkeypatch.setattr(engines_module.gigaam, "run_child", fake_run_child)
    engine = GigaAMTranscriber(interpreter="/no/such/python")
    assert engine.device == C.ENGINE_DEFAULT_DEVICE
    list(engine.transcribe(tmp_path / "a.wav"))
    assert seen["request"]["device"] == C.ENGINE_DEFAULT_DEVICE
    # BUGS.md entry 34: the engine reports the concrete device the child
    # actually used, not the request it sent.
    assert engine.device == "cuda"


def test_whisper_is_in_process_and_names_its_extra() -> None:
    assert FasterWhisperTranscriber.out_of_process is False
    assert FasterWhisperTranscriber.extra == "whisper"
    assert FasterWhisperTranscriber.required_module == "faster_whisper"


@pytest.mark.skipif(HAS_FASTER_WHISPER, reason="faster-whisper is installed here")
def test_whisper_without_its_dependency_names_the_extra_instead_of_crashing() -> None:
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(FasterWhisperTranscriber)
    assert "rytp[whisper]" in str(excinfo.value)
