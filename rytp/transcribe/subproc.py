"""The out-of-process engine seam (contracts §6, design §6).

The recommended transcriber and the recommended diarizer pin incompatible
versions of their shared dependencies, so an engine may have to live in its own
virtual environment. Such an engine declares ``out_of_process = True`` and never
imports its dependency in this process at all: it serialises a request, runs
*this file* as a script under another interpreter, and reads a JSON response
back out of a file.

Two deliberate choices, both about surviving real engines:

**The response goes to a file, not to stdout.** Model loaders print progress
bars, deprecation warnings and CUDA chatter to both streams. Stdout is treated
as noise; only the file is read.

**The child is launched by path, not as ``-m rytp.transcribe.subproc``.** The
child never imports the ``rytp`` package for its own sake — it puts the repo
root on ``sys.path`` and imports exactly one engine module, which in turn
imports only the standard library, :mod:`rytp.models` and its own dependency.

The child contract: the named module exposes ``child_main(request: dict) ->
dict`` returning JSON-serialisable plain dicts. Everything it raises comes back
as :class:`~rytp.transcribe.base.EngineSubprocessError` with the type, the
message and the tail of stderr.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tail(text: str, lines: int) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return "\n".join(kept)


def run_child(
    *,
    interpreter: str,
    module: str,
    request: dict[str, Any],
    timeout_s: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run ``module.child_main(request)`` under ``interpreter`` and return its result.

    Raises :class:`~rytp.transcribe.base.EngineSubprocessError` for a missing
    interpreter, a timeout, a crashed child, a child that wrote nothing, and a
    child that reported failure.
    """
    from rytp import constants as C
    from rytp.transcribe.base import EngineSubprocessError

    timeout = C.ENGINE_SUBPROCESS_TIMEOUT_S if timeout_s is None else timeout_s
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # ignore_cleanup_errors: after a timeout the killed child may still hold
    # request.json open for a moment, which makes the cleanup raise
    # PermissionError on Windows.
    with tempfile.TemporaryDirectory(prefix="rytp-engine-", ignore_cleanup_errors=True) as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(
            json.dumps({"module": module, "root": str(root), "request": request}),
            encoding="utf-8",
        )
        command = [
            interpreter,
            str(Path(__file__).resolve()),
            str(request_path),
            str(response_path),
        ]
        try:
            proc = subprocess.run(  # fixed argv, no shell
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise EngineSubprocessError(
                f"engine interpreter not found: {interpreter}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineSubprocessError(
                f"engine {module} timed out after {timeout}s"
            ) from exc

        payload = _read_response(response_path)
        stderr_tail = _tail(proc.stderr, C.ENGINE_SUBPROCESS_STDERR_TAIL)
        if payload is None:
            raise EngineSubprocessError(
                f"engine {module} wrote no response (exit {proc.returncode})\n{stderr_tail}"
            )
        if not payload.get("ok"):
            error = payload.get("error") or {}
            raise EngineSubprocessError(
                f"engine {module} failed: {error.get('type', 'Error')}: "
                f"{error.get('message', '')}\n{error.get('traceback', '')}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise EngineSubprocessError(
                f"engine {module} returned {type(result).__name__}, expected an object"
            )
        return result


def _read_response(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _child_main(argv: list[str]) -> int:
    """Child entry point. Standard library only until the engine module loads."""
    request_path = Path(argv[1])
    response_path = Path(argv[2])
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    payload: dict[str, Any]
    try:
        root = envelope["root"]
        if root not in sys.path:
            sys.path.insert(0, root)
        import importlib

        module = importlib.import_module(envelope["module"])
        payload = {"ok": True, "result": module.child_main(envelope["request"])}
    except Exception as exc:  # the boundary exists to report anything
        payload = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv))
