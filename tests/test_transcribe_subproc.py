"""The out-of-process seam: a request in, a JSON file out, errors that name themselves."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rytp.transcribe.base import EngineSubprocessError
from rytp.transcribe.subproc import run_child

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = "tests.subproc_engine_stub"


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
