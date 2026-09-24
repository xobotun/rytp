"""The diarizer registry: classes in, classes out, gated before construction."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.diarize import base
from rytp.models import RytpError
from rytp.transcribe.base import EngineUnavailable
from tests.fake_speaker_engines import (
    FakeDiarizer,
    GatedDiarizer,
    OutOfProcessDiarizer,
    registered,
)


def test_register_and_resolve_returns_the_class_not_an_instance() -> None:
    with registered(FakeDiarizer):
        assert base.resolve_diarizer("fake-diarizer") is FakeDiarizer


def test_resolve_unknown_diarizer_names_the_available_ones() -> None:
    with registered(FakeDiarizer), pytest.raises(ValueError) as excinfo:
        base.resolve_diarizer("nope")
    assert "nope" in str(excinfo.value)
    assert "fake-diarizer" in str(excinfo.value)


def test_resolve_unknown_diarizer_is_also_a_rytp_error() -> None:
    # BUGS.md entry 16: `speakers diarize 1 --diarizer pyannot` printed a
    # ~60-line traceback because a bare ValueError escaped the CLI's
    # `except RytpError` funnel. The raised type has to satisfy both.
    with registered(FakeDiarizer), pytest.raises(RytpError) as excinfo:
        base.resolve_diarizer("nope")
    assert isinstance(excinfo.value, ValueError)


def test_gate_rejects_a_token_engine_before_construction(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with registered(GatedDiarizer), pytest.raises(EngineUnavailable) as excinfo:
        base.load_diarizer(db, "fake-gated-diarizer")
    assert "HF_TOKEN" in str(excinfo.value)


def test_gate_accepts_a_token_engine_when_the_token_is_set(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    with registered(GatedDiarizer):
        assert isinstance(base.load_diarizer(db, "fake-gated-diarizer"), GatedDiarizer)


def test_load_diarizer_injects_the_interpreter_out_of_process(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote-diarizer", "/opt/remote/python"),
    )
    with registered(OutOfProcessDiarizer):
        engine = base.load_diarizer(db, "fake-remote-diarizer")
    assert engine.interpreter == "/opt/remote/python"


def test_load_diarizer_does_not_inject_an_interpreter_in_process(db: Database) -> None:
    with registered(FakeDiarizer):
        engine = base.load_diarizer(db, "fake-diarizer")
    assert not hasattr(engine, "interpreter")


def test_interpreter_falls_back_to_this_interpreter(db: Database) -> None:
    from rytp.transcribe.registry import interpreter_for

    assert interpreter_for(db, "fake-remote-diarizer") == sys.executable


def test_diarizer_rows_report_token_process_and_availability(db: Database) -> None:
    with registered(FakeDiarizer, GatedDiarizer):
        rows = {row[0]: row for row in base.diarizer_rows(db)}
    assert rows["fake-diarizer"][1] == "no"
    assert rows["fake-diarizer"][2] == "no"
    assert rows["fake-diarizer"][3] == "ready"
    assert rows["fake-gated-diarizer"][1] == "yes"


def test_wav_duration_ms_reads_the_header(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 8000)      # 0.5 s
    assert base.wav_duration_ms(path) == 500
