"""The engine registry: classes in, classes out, gated before construction."""
from __future__ import annotations

import sys

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.models import RytpError, UnknownEngineError
from rytp.transcribe import registry
from rytp.transcribe.base import EngineUnavailable
from tests.fake_engines import (
    FakeAligner,
    FakeTranscriber,
    GatedTranscriber,
    MissingModuleTranscriber,
    OutOfProcessTranscriber,
    registered,
)


def test_register_and_resolve_returns_the_class_not_an_instance() -> None:
    with registered(FakeTranscriber, FakeAligner):
        assert registry.resolve_transcriber("fake") is FakeTranscriber
        assert registry.resolve_aligner("fake-aligner") is FakeAligner


def test_resolve_unknown_transcriber_names_the_available_ones() -> None:
    with registered(FakeTranscriber), pytest.raises(ValueError) as excinfo:
        registry.resolve_transcriber("nope")
    assert "nope" in str(excinfo.value)
    assert "fake" in str(excinfo.value)


def test_resolve_unknown_transcriber_is_also_a_rytp_error() -> None:
    # BUGS.md entry 16: the CLI funnel catches `RytpError`, not `ValueError`.
    # A mistyped engine name has to satisfy both, at once.
    with registered(FakeTranscriber), pytest.raises(RytpError):
        registry.resolve_transcriber("nope")
    with registered(FakeTranscriber):
        try:
            registry.resolve_transcriber("nope")
        except UnknownEngineError as exc:
            assert isinstance(exc, ValueError)
            assert isinstance(exc, RytpError)


def test_resolve_unknown_aligner_is_also_a_rytp_error() -> None:
    with registered(FakeAligner), pytest.raises(RytpError) as excinfo:
        registry.resolve_aligner("nope")
    assert isinstance(excinfo.value, ValueError)


def test_gate_rejects_a_token_engine_before_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(GatedTranscriber)
    assert "HF_TOKEN" in str(excinfo.value)


def test_gate_accepts_a_token_engine_when_the_token_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    registry.check_available(GatedTranscriber)


def test_gate_names_the_extra_for_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(MissingModuleTranscriber)
    assert "rytp[fakeextra]" in str(excinfo.value)


def test_interpreter_falls_back_to_this_interpreter(db: Database) -> None:
    assert registry.interpreter_for(db, "fake-remote") == sys.executable


def test_interpreter_prefers_the_per_engine_setting_over_the_default(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.default", "/opt/default/python"),
    )
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote", "/opt/remote/python"),
    )
    db.conn.commit()
    assert registry.interpreter_for(db, "fake-remote") == "/opt/remote/python"
    assert registry.interpreter_for(db, "other") == "/opt/default/python"


def test_load_transcriber_injects_the_interpreter_for_out_of_process_engines(
    db: Database,
) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote", "/opt/remote/python"),
    )
    db.conn.commit()
    with registered(OutOfProcessTranscriber):
        engine = registry.load_transcriber(db, "fake-remote")
    assert engine.interpreter == "/opt/remote/python"


def test_load_transcriber_does_not_inject_an_interpreter_in_process(db: Database) -> None:
    with registered(FakeTranscriber):
        engine = registry.load_transcriber(db, "fake")
    assert not hasattr(engine, "interpreter")


def test_engine_rows_reports_kind_and_availability(db: Database) -> None:
    with registered(FakeTranscriber, FakeAligner, MissingModuleTranscriber):
        rows = {row[0]: row for row in registry.engine_rows(db)}
    assert rows["fake"][1] == "transcriber"
    assert rows["fake-aligner"][1] == "aligner"
    assert rows["fake"][4] == "ready"
    assert "definitely_not_installed_xyz" in rows["fake-missing"][4]


def test_the_default_transcriber_is_the_russian_engine_out_of_the_box(
    db: Database,
) -> None:
    # contracts §3: the corpus is Russian and gigaam roughly halves Whisper's
    # word error rate there. Part 1 seeds no row for this key, so the value
    # has to come from the fallback or "defaults to gigaam" is not true.
    assert registry.default_transcriber(db) == C.DEFAULT_TRANSCRIBER_FALLBACK
    assert registry.default_transcriber(db) == "gigaam"


def test_check_available_skips_the_probe_with_no_interpreter_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `rytp/commands/transcribe.py`'s summary path is the one caller with no
    # interpreter to hand; it must keep today's weaker, module-blind answer
    # rather than crash. (BUGS.md entry 7.)
    with registered(MissingModuleTranscriber):
        MissingModuleTranscriber.out_of_process = True
        try:
            registry.check_available(MissingModuleTranscriber)  # does not raise
        finally:
            MissingModuleTranscriber.out_of_process = False


def test_check_available_probes_the_real_interpreter_for_an_out_of_process_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setattr(registry, "_PROBE_CACHE", {})
    with registered(MissingModuleTranscriber):
        MissingModuleTranscriber.out_of_process = True
        try:
            with pytest.raises(EngineUnavailable) as excinfo:
                registry.check_available(MissingModuleTranscriber, interpreter=sys.executable)
        finally:
            MissingModuleTranscriber.out_of_process = False
    assert 'install the fakeextra extra: pip install -e ".[fakeextra]"' in str(excinfo.value)


def test_availability_probes_and_reports_module_missing_for_an_out_of_process_engine(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "_PROBE_CACHE", {})
    with registered(MissingModuleTranscriber):
        MissingModuleTranscriber.out_of_process = True
        try:
            state = registry.availability(db, MissingModuleTranscriber)
        finally:
            MissingModuleTranscriber.out_of_process = False
    assert state.startswith("needs definitely_not_installed_xyz")
    assert "interpreter ok" not in state  # BUGS.md entry 7's exact complaint


def test_availability_falls_back_to_interpreter_ok_with_no_required_module(
    db: Database,
) -> None:
    with registered(OutOfProcessTranscriber):
        state = registry.availability(db, OutOfProcessTranscriber)
    assert state == "interpreter ok"


def test_probe_engine_caches_per_interpreter_and_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_PROBE_CACHE", {})
    calls: list[tuple[str, str, str]] = []

    def fake_probe(interpreter: str, module: str, required_module: str) -> dict[str, object]:
        calls.append((interpreter, module, required_module))
        return {"module_ok": True, "module_error": "", "torch": None,
                "cuda_available": None, "device": "cpu"}

    import rytp.transcribe.subproc as subproc_module

    monkeypatch.setattr(subproc_module, "probe", fake_probe)
    registry.probe_engine("/opt/py", "", "gigaam")
    registry.probe_engine("/opt/py", "", "gigaam")
    assert len(calls) == 1


def test_the_setting_wins_over_the_fallback(db: Database) -> None:
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "whisper")
    assert registry.default_transcriber(db) == "whisper"


def test_an_empty_default_transcriber_is_not_a_way_to_turn_it_off(
    db: Database,
) -> None:
    # Empty is meaningful for default_aligner and meaningless here: a
    # transcribe run with no engine does nothing at all, so there is no
    # unset path and this always answers with a name.
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "  ")
    assert registry.default_transcriber(db) == C.DEFAULT_TRANSCRIBER_FALLBACK
