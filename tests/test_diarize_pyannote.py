"""The pyannote adapter: importable without pyannote, gated, out of process."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import base
from rytp.diarize.pyannote import PyannoteDiarizer, segments_from_tracks
from rytp.transcribe.base import EngineUnavailable


def test_the_adapter_imports_and_registers_without_pyannote() -> None:
    assert importlib.util.find_spec("pyannote") is None, "the dev venv must be clean"
    assert base.DIARIZERS["pyannote"] is PyannoteDiarizer
    assert PyannoteDiarizer.requires_hf_token is True
    assert PyannoteDiarizer.out_of_process is True


def test_loading_it_without_a_token_is_refused_before_construction(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable, match="HF_TOKEN"):
        base.load_diarizer(db, "pyannote")


def test_segments_from_tracks_converts_seconds_to_milliseconds() -> None:
    tracks = [(0.0, 1.25, "SPEAKER_00"), (1.25, 3.5, "SPEAKER_01")]
    assert segments_from_tracks(tracks) == [
        {"start_ms": 0, "end_ms": 1_250, "local_label": "SPEAKER_00"},
        {"start_ms": 1_250, "end_ms": 3_500, "local_label": "SPEAKER_01"},
    ]


def test_segments_from_tracks_drops_a_zero_length_turn() -> None:
    assert segments_from_tracks([(1.0, 1.0, "SPEAKER_00")]) == []


def test_the_adapter_sends_the_request_the_child_expects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import pyannote as adapter

    seen: dict[str, object] = {}

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"segments": [{"start_ms": 0, "end_ms": 1_000, "local_label": "SPEAKER_00"}]}

    monkeypatch.setattr(adapter, "run_child", fake_run_child)
    engine = PyannoteDiarizer(interpreter="/opt/pyannote/python", hf_token="t")
    segments = list(engine.diarize(tmp_path / "a.wav"))

    assert [s.local_label for s in segments] == ["SPEAKER_00"]
    assert segments[0].end_ms == 1_000
    assert seen["interpreter"] == "/opt/pyannote/python"
    assert seen["module"] == "rytp.diarize.pyannote"
    request = seen["request"]
    assert request["model"] == C.PYANNOTE_DIARIZATION_MODEL
    assert request["hf_token"] == "t"


def test_the_adapter_reads_the_token_from_the_environment_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import pyannote as adapter

    seen: dict[str, object] = {}
    monkeypatch.setenv("HF_TOKEN", "from-env")

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"segments": []}

    monkeypatch.setattr(adapter, "run_child", fake_run_child)
    list(PyannoteDiarizer(interpreter="/opt/p/python").diarize(tmp_path / "a.wav"))
    assert seen["request"]["hf_token"] == "from-env"


def test_device_defaults_to_auto_and_the_reported_choice_sticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BUGS.md entry 34: pyannote never moved anything to a device at all.
    from rytp.diarize import pyannote as adapter

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        request = kwargs["request"]
        assert request["device"] == C.ENGINE_DEFAULT_DEVICE  # type: ignore[index]
        return {"segments": [], "device": "cuda"}

    monkeypatch.setattr(adapter, "run_child", fake_run_child)
    engine = PyannoteDiarizer(interpreter="/opt/p/python", hf_token="t")
    assert engine.device == C.ENGINE_DEFAULT_DEVICE
    list(engine.diarize(tmp_path / "a.wav"))
    assert engine.device == "cuda"
