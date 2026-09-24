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
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

_log = logging.getLogger(__name__)


def _tail(text: str, lines: int) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return "\n".join(kept)


def resolve_device(requested: str) -> str:
    """``"auto" | "cuda" | "cpu"`` to a concrete device, run inside a child (plan §1b).

    Standard-library-safe to import from any child's ``child_main``: it only
    touches torch to resolve ``"auto"``, and tolerates torch's absence even
    then (an engine that needs torch to run at all fails later, on its own
    import, with its own message — this helper never manufactures a torch
    error of its own).
    """
    from rytp import constants as C

    if requested not in C.ENGINE_DEVICES:
        raise ValueError(
            f"unknown device {requested!r}; expected one of {', '.join(C.ENGINE_DEVICES)}"
        )
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _extra_for_missing_module(module: str) -> str | None:
    """Which extra installs the dependency ``module`` (the engine's own file) needs.

    Looked up by ``__module__`` rather than by the *missing* module's name:
    two engines can fail on the same missing dependency (``torch``, for
    wav2vec2 and redimnet alike) and only the calling engine's own ``extra``
    is the right answer, never a guess from the failing import's name.
    """
    from rytp.diarize.base import DIARIZERS
    from rytp.diarize.embed import EMBEDDERS
    from rytp.transcribe.registry import ALIGNERS, TRANSCRIBERS

    for table in (TRANSCRIBERS, ALIGNERS, DIARIZERS, EMBEDDERS):
        for name, cls in table.items():
            if getattr(cls, "__module__", None) == module:
                return str(getattr(cls, "extra", None) or name)
    return None


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
    from rytp.transcribe.base import EngineSubprocessError, EngineUnavailable

    timeout = C.ENGINE_SUBPROCESS_TIMEOUT_S if timeout_s is None else timeout_s
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    # BUGS.md entry 11: silence huggingface_hub's per-process symlink warning
    # in every child that might load an HF-backed model; `doctor`'s hf-cache
    # check says the same thing once, as an advisory line with the disk cost,
    # rather than the library repeating a paragraph of Windows advice.
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        C.HF_SYMLINK_WARNING_ENV: os.environ.get(C.HF_SYMLINK_WARNING_ENV, "1"),
    }

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
            error_type = error.get("type", "Error")
            message = error.get("message", "")
            missing_name = error.get("name")
            # BUGS.md entry 5: `run_child` used to embed `error['traceback']`
            # verbatim in the message, so the CLI funnel's "one line, never a
            # traceback" contract printed a traceback anyway. The traceback is
            # still recorded — at debug level — just never in the raised
            # message. A `ModuleNotFoundError` for a dependency (never for
            # `rytp` itself, which would mean a broken adapter file, not a
            # missing extra) gets the friendlier install hint instead.
            _log.debug(
                "engine %s failed: %s: %s\n%s",
                module, error_type, message, error.get("traceback", ""),
            )
            if (
                error_type == "ModuleNotFoundError"
                and missing_name
                and not str(missing_name).startswith("rytp")
            ):
                extra = _extra_for_missing_module(module)
                if extra:
                    raise EngineUnavailable(
                        f'install the {extra} extra: pip install -e ".[{extra}]"'
                    )
            raise EngineSubprocessError(f"engine {module} failed: {error_type}: {message}")
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
                # Only ModuleNotFoundError/ImportError carry this; None
                # otherwise. The parent uses it to tell "a dependency is
                # missing" from "our own adapter file failed to import"
                # (BUGS.md entry 5).
                "name": getattr(exc, "name", None),
            },
        }
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """This module's own probe target (plan §Task 7, "one probe serves four entries").

    :func:`probe` calls :func:`run_child` with ``module="rytp.transcribe.subproc"``,
    so ``_child_main`` imports *this* module and calls this function — the
    "tiny built-in request" the plan asks for, with no real engine invoked.
    Never raises: any exception here is still caught by ``_child_main``'s own
    boundary and comes back as an ``ok: False`` payload, which :func:`probe`
    also tolerates.

    Answers, in one child process, the three things BUGS.md entries 7, 33 and
    34 all separately asked for: does the dependency import, does this
    interpreter's torch see a GPU, and which device would ``"auto"`` pick.
    """
    import importlib

    required_module = str(request.get("required_module") or "")
    engine_module = str(request.get("module") or "")
    module_ok = True
    module_error = ""
    for name in (engine_module, required_module):
        if not name:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # any import failure is the finding, not a crash
            module_ok = False
            module_error = f"{type(exc).__name__}: {exc}"
            break

    torch_version: str | None = None
    cuda_available: bool | None = None
    device = "cpu"
    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "unknown"))
        cuda_available = bool(torch.cuda.is_available())
        device = "cuda" if cuda_available else "cpu"
    except Exception:  # torch absent, or broken — that is itself the answer
        pass

    return {
        "module_ok": module_ok,
        "module_error": module_error,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "device": device,
    }


def probe(interpreter: str, module: str, required_module: str) -> dict[str, Any]:
    """Does ``required_module`` import under ``interpreter``, and which device would run.

    BUGS.md entries 7, 33, 34: one child process answers all three — module
    presence, CUDA availability, and the device ``"auto"`` would resolve to.
    Never raises: a missing interpreter, a timeout, or a crashed child all
    degrade to a ``module_ok: False`` answer rather than propagating, because
    a probe exists specifically to be safe to call from a listing command.
    """
    from rytp import constants as C

    try:
        result = run_child(
            interpreter=interpreter,
            module="rytp.transcribe.subproc",
            request={"module": module, "required_module": required_module},
            timeout_s=C.ENGINE_PROBE_TIMEOUT_S,
        )
    except Exception as exc:
        return {
            "module_ok": False,
            "module_error": f"{type(exc).__name__}: {exc}",
            "torch": None,
            "cuda_available": None,
            "device": "cpu",
        }
    return {
        "module_ok": bool(result.get("module_ok")),
        "module_error": str(result.get("module_error") or ""),
        "torch": result.get("torch"),
        "cuda_available": result.get("cuda_available"),
        "device": str(result.get("device") or "cpu"),
    }


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv))
