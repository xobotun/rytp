"""The out-of-process seam: a request in, a JSON file out, errors that name themselves."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rytp.models import RytpError
from rytp.transcribe.base import EngineSubprocessError, EngineUnavailable
from rytp.transcribe.registry import TRANSCRIBERS
from rytp.transcribe.subproc import probe, resolve_device, run_child, shutdown_workers

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = "tests.subproc_engine_stub"


@pytest.fixture(autouse=True)
def _no_leftover_workers() -> None:
    """Close every resident worker after each test.

    A real call in this file now spawns a resident worker, not a one-shot
    child, so a test that deliberately crashes or hangs one must not leave
    it (or an un-crashed sibling worker) running for the next test.
    """
    yield
    shutdown_workers()


def test_round_trip_returns_the_child_result() -> None:
    result = run_child(
        interpreter=sys.executable, module=STUB, request={}, repo_root=REPO_ROOT
    )
    assert result["words"][0]["text"] == "ok"


def test_stdout_noise_from_the_child_is_ignored() -> None:
    result = run_child(
        interpreter=sys.executable,
        module=STUB,
        request={"noise": True},
        repo_root=REPO_ROOT,
    )
    assert result["words"][0]["text"] == "ok"


def test_child_exception_becomes_a_named_error_with_the_message() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"boom": True},
            repo_root=REPO_ROOT,
        )
    message = str(excinfo.value)
    assert "RuntimeError" in message
    assert "engine exploded" in message


def test_unimportable_module_is_reported_not_swallowed() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="tests.no_such_engine_module",
            request={},
            repo_root=REPO_ROOT,
        )
    assert "no_such_engine_module" in str(excinfo.value)


def test_missing_interpreter_names_the_path() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter="/no/such/python",
            module=STUB,
            request={},
            repo_root=REPO_ROOT,
        )
    assert "/no/such/python" in str(excinfo.value)


def test_timeout_is_reported(tmp_path: Path) -> None:
    slow = tmp_path / "slow_engine.py"
    slow.write_text(
        "import time\n\n\ndef child_main(request):\n    time.sleep(30)\n    return {}\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="slow_engine",
            request={},
            repo_root=tmp_path,
            timeout_s=1,
        )
    assert "timed out" in str(excinfo.value)


# -- BUGS.md entry 5: no traceback in the message -----------------------


def test_a_generic_child_failure_names_the_type_and_message_but_no_traceback() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"boom": True},
            repo_root=REPO_ROOT,
        )
    message = str(excinfo.value)
    assert "RuntimeError" in message
    assert "engine exploded" in message
    assert "Traceback" not in message
    assert "child_main" not in message  # nothing from the stack frames either


def test_a_missing_dependency_names_the_calling_engines_own_extra() -> None:
    # `tests.subproc_engine_stub` is not a registered engine, so there is no
    # extra to recommend and the seam falls back to the honest generic
    # message — still with no traceback.
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"missing_dependency": True},
            repo_root=REPO_ROOT,
        )
    message = str(excinfo.value)
    assert "ModuleNotFoundError" in message
    assert "Traceback" not in message


class _FakeStubEngine:
    """A fake transcriber whose ``__module__`` matches the stub, so the
    seam can find *an* extra to recommend without a real adapter file."""

    name = "fake-stub-engine"
    requires_hf_token = False
    out_of_process = True
    required_module = "definitely_not_a_real_engine_dep"
    extra = "stubextra"


_FakeStubEngine.__module__ = STUB


def test_a_missing_dependency_gives_a_one_line_install_hint_when_the_engine_is_known() -> None:
    TRANSCRIBERS[_FakeStubEngine.name] = _FakeStubEngine
    try:
        with pytest.raises(EngineUnavailable) as excinfo:
            run_child(
                interpreter=sys.executable,
                module=STUB,
                request={"missing_dependency": True},
                repo_root=REPO_ROOT,
            )
    finally:
        del TRANSCRIBERS[_FakeStubEngine.name]
    message = str(excinfo.value)
    assert message == 'install the stubextra extra: pip install -e ".[stubextra]"'
    assert "Traceback" not in message
    assert isinstance(excinfo.value, RytpError)


def test_a_missing_module_inside_rytp_itself_never_gets_the_extra_hint() -> None:
    # A broken adapter file (bad repo_root, a syntax error) must not be
    # mistaken for a missing optional extra.
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"missing_rytp_module": True},
            repo_root=REPO_ROOT,
        )
    assert "rytp.transcribe.engines.nonexistent" in str(excinfo.value)


# -- the probe: entries 7, 33, 34 ----------------------------------------


def test_probe_reports_module_ok_for_a_module_that_imports() -> None:
    result = probe(sys.executable, "", "json")
    assert result["module_ok"] is True
    assert result["module_error"] == ""


def test_probe_reports_module_missing_with_the_import_error() -> None:
    result = probe(sys.executable, "", "definitely_not_a_real_engine_dep_xyz")
    assert result["module_ok"] is False
    assert "definitely_not_a_real_engine_dep_xyz" in result["module_error"]


def test_probe_reports_no_torch_in_this_dev_venv() -> None:
    # The dev venv this batch runs under has no torch (several other tests
    # rely on the same fact). The probe must say so rather than crash.
    result = probe(sys.executable, "", "json")
    assert result["torch"] is None
    assert result["cuda_available"] is None
    assert result["device"] == "cpu"


def test_probe_never_raises_for_a_missing_interpreter() -> None:
    result = probe("/no/such/python", "", "json")
    assert result["module_ok"] is False
    assert result["device"] == "cpu"


def test_probe_never_raises_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rytp.transcribe.subproc as subproc_module

    def explode(**kwargs: object) -> dict[str, object]:
        raise EngineSubprocessError("engine rytp.transcribe.subproc timed out after 1s")

    # `probe` calls the one-shot child (`_run_once`), never the resident
    # worker pool (`run_child`) — see `subproc.py`'s module docstring.
    monkeypatch.setattr(subproc_module, "_run_once", explode)
    result = probe(sys.executable, "", "json")
    assert result["module_ok"] is False
    assert "timed out" in result["module_error"]


# -- resolve_device --------------------------------------------------------


def test_resolve_device_passes_through_an_explicit_choice() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_falls_back_to_cpu_without_torch() -> None:
    # No torch in this dev venv (confirmed elsewhere in this suite).
    assert resolve_device("auto") == "cpu"


def test_resolve_device_refuses_an_unknown_value() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")
