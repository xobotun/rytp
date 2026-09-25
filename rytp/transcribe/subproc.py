"""The out-of-process engine seam (contracts §6, design §6).

The recommended transcriber and the recommended diarizer pin incompatible
versions of their shared dependencies, so an engine may have to live in its own
virtual environment. Such an engine declares ``out_of_process = True`` and never
imports its dependency in this process at all: it serialises a request and
hands it to a **persistent worker child** running under another interpreter,
reading a JSON response back out of a file.

Two deliberate choices, both about surviving real engines:

**The response goes to a file, not to stdout.** Model loaders print progress
bars, deprecation warnings and CUDA chatter to both streams. Stdout is treated
as noise; only the file is read.

**The child is launched by path, not as ``-m rytp.transcribe.subproc``.** The
child never imports the ``rytp`` package for its own sake — it puts the repo
root on ``sys.path`` and imports exactly one engine module, which in turn
imports only the standard library, :mod:`rytp.models` and its own dependency.

Persistent worker lifecycle (contracts §6)
-------------------------------------------
:func:`run_child` used to mean "spawn one process, run one request, exit" —
which meant a model load per chunk. It now means "reuse a resident child that
imported the engine module once, and keeps whatever model it loaded cached
across calls":

* One child per ``(interpreter, module, repo root)``, started lazily on first
  use and cached for the life of this process (:data:`_WORKERS`).
* A call is a control line on the child's stdin naming a request file and a
  response file, never a raw stdin/stdout payload — so library noise on
  either stream can never corrupt the protocol. The parent polls for the
  response file (written atomically, temp-then-``os.replace``) rather than
  waiting on a second pipe.
* Calls to one worker are serialised by a per-worker lock. Two pipelines
  wanting the same engine at once queue rather than race for the same
  resident model; the lock is held for up to the call's timeout, so the
  second caller waits silently — there is no queue-position report.
* A worker that dies (crash, kill, `os._exit`) or times out is torn down and
  evicted; the *next* call to that key pays one reload to start a fresh
  child. A worker that merely returns ``ok: False`` (an ordinary exception
  inside ``child_main``) is untouched and stays resident.
* A child whose engine module fails to import at startup answers every
  request with that same import error and never retries the import — fixing
  the adapter file takes effect only after the resident child is torn down
  (a crash, or an explicit :func:`shutdown_workers`).
* Parent death is a free cleanup: the child's stdin pipe hits EOF, its
  read loop ends, and it exits on its own. :func:`shutdown_workers` is also
  registered with :mod:`atexit` for the ordinary shutdown path; neither runs
  after ``os._exit`` or an external kill, which is why the OS still owns the
  process tree as the backstop.
* **Cost, not just benefit.** A resident worker holds its model in memory
  (GPU or otherwise) for as long as it lives — running several out-of-process
  engines back to back leaves all of them loaded at once unless something
  calls :func:`shutdown_workers` (or evicts one key) between stages. Nothing
  in this module does that automatically; a caller that needs the memory
  back must ask for it.

**What is unaffected by all this.** :func:`probe` deliberately stays a
one-shot child (:func:`_run_once`) — it answers "can this engine run at all"
for engines nobody may ever construct (a listing command probes every
registered engine), and its own per-process result cache
(``rytp.transcribe.registry._PROBE_CACHE``) already avoids repeating the
question. Making the probe resident would mean holding a child open — and
possibly a model loaded — for an engine the caller never uses.

**What this module does not fix.** MFA's per-call cost is mostly the
external ``mfa`` binary's own process starting up and loading its acoustic
model — :mod:`rytp.transcribe.align.mfa`'s ``child_main`` still shells out to
that binary fresh on every call, because MFA has no long-lived server mode
to talk to. A resident *Python* worker for MFA still saves the Python
interpreter startup and import cost, but the dominant reload — the `mfa`
process itself — is outside this module's reach.

The child contract: the named module exposes ``child_main(request: dict) ->
dict`` returning JSON-serialisable plain dicts. Everything it raises comes back
as :class:`~rytp.transcribe.base.EngineSubprocessError` with the type, the
message and the tail of stderr.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

# `rytp.progress` is imported lazily, inside functions that need it — never at
# module scope. This file doubles as the child's own entry point (the
# ``if __name__ == "__main__"`` block at the bottom), and the module docstring
# is explicit: the child puts the repo root on `sys.path` and imports exactly
# one engine module, standard-library-only until then. A module-level `rytp`
# import here would run before that `sys.path` insertion and break under a
# foreign interpreter that has no `rytp` package installed at all.

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: How often the parent polls a response file for existence while a worker
#: call is in flight. Not a contracts constant — purely an implementation
#: knob for this file's own poll loop, well below anything a caller could
#: perceive against real model-inference latencies.
_POLL_INTERVAL_S = 0.02

_log = logging.getLogger(__name__)


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


#: Process-level cache for whatever an engine's `child_main` decides is
#: expensive to rebuild — a loaded model, a processor, a pipeline object.
#: Standard library only, so it is safe to import from any child. Keyed by
#: whatever tuple the caller chooses (model name + device is the usual
#: shape); the *loader* runs at most once per key per process, which is
#: exactly the thing a persistent worker makes worth doing — see the module
#: docstring's "the real fix is the model cache, not the process".
_MODEL_CACHE: dict[Any, Any] = {}


def load_cached(key: Any, loader: Any) -> Any:
    """Return ``loader()``'s result, computed once per ``key`` per process.

    Called from inside a ``child_main`` — this process is the resident
    worker, so "once per process" means "once for as long as this engine's
    model stays loaded", which is the whole point of the persistent seam.
    ``loader`` takes no arguments; a caller needing parameters closes over
    them. Not thread-safe by itself, but every ``child_main`` here runs on a
    single-threaded read loop (:func:`_worker_main`), so no lock is needed.
    """
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = loader()
    return _MODEL_CACHE[key]


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


def _handle_failure(module: str, payload: dict[str, Any]) -> Any:
    """Turn an ``ok: False`` (or malformed) payload into the right exception.

    Shared between :func:`_run_once` (the probe's one-shot child) and
    :class:`_Worker` (the persistent seam), so the error-mapping rules —
    including BUGS.md entry 5's "no traceback in the message" and entry 16's
    "the missing dependency gets the calling engine's own install hint" —
    exist exactly once.
    """
    from rytp.transcribe.base import EngineSubprocessError, EngineUnavailable

    error = payload.get("error") or {}
    error_type = error.get("type", "Error")
    message = error.get("message", "")
    missing_name = error.get("name")
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
            raise EngineUnavailable(f'install the {extra} extra: pip install -e ".[{extra}]"')
    raise EngineSubprocessError(f"engine {module} failed: {error_type}: {message}")


def _result_from_payload(module: str, payload: dict[str, Any]) -> dict[str, Any]:
    """``payload["result"]`` if ``ok``, else raise via :func:`_handle_failure`."""
    from rytp.transcribe.base import EngineSubprocessError

    if not payload.get("ok"):
        _handle_failure(module, payload)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise EngineSubprocessError(
            f"engine {module} returned {type(result).__name__}, expected an object"
        )
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` so a concurrent reader never observes a partial file.

    The persistent worker's parent polls for this file while the child is
    still writing it; a plain ``write_text`` would let the poller read a
    truncated JSON document. Write-then-``os.replace`` is atomic on both
    POSIX and Windows as long as both paths are on the same volume, which
    they are here (same temp directory).
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _spawn_env() -> dict[str, str]:
    from rytp import constants as C

    # BUGS.md entry 11: silence huggingface_hub's per-process symlink warning
    # in every child that might load an HF-backed model; `doctor`'s hf-cache
    # check says the same thing once, as an advisory line with the disk cost,
    # rather than the library repeating a paragraph of Windows advice.
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        C.HF_SYMLINK_WARNING_ENV: os.environ.get(C.HF_SYMLINK_WARNING_ENV, "1"),
    }


def _run_once(
    *,
    interpreter: str,
    module: str,
    request: dict[str, Any],
    timeout_s: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run ``module.child_main(request)`` under ``interpreter`` in a fresh, one-shot child.

    This is the pre-persistent-seam behaviour, kept for :func:`probe` only —
    see the module docstring for why the probe deliberately never joins the
    resident worker pool. Every real engine call goes through :func:`run_child`
    instead.
    """
    from rytp import constants as C
    from rytp.progress import report
    from rytp.transcribe.base import EngineSubprocessError

    timeout = C.ENGINE_SUBPROCESS_TIMEOUT_S if timeout_s is None else timeout_s
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    env = _spawn_env()

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
        stderr_tail: deque[str] = deque(maxlen=C.ENGINE_SUBPROCESS_STDERR_TAIL)
        stage = f"engine:{module.rsplit('.', 1)[-1]}"

        def _pump_stderr(pipe: Any) -> None:
            with pipe:
                for raw_line in pipe:
                    line = raw_line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    stderr_tail.append(line)
                    report(stage, detail=line)

        try:
            proc = subprocess.Popen(  # fixed argv, no shell
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except FileNotFoundError as exc:
            raise EngineSubprocessError(
                f"engine interpreter not found: {interpreter}"
            ) from exc

        assert proc.stderr is not None
        # `report()` resolves its sink from a `ContextVar`, and a fresh
        # thread starts with a fresh top-level context rather than
        # inheriting the caller's — so a worker's per-job sink or a test's
        # `progress.install(...)` would silently stop applying inside this
        # thread without an explicit copy (contextvars' documented behaviour,
        # unlike `threading.local()`).
        ctx = contextvars.copy_context()
        reader = threading.Thread(target=ctx.run, args=(_pump_stderr, proc.stderr), daemon=True)
        reader.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            reader.join(timeout=5)
            raise EngineSubprocessError(
                f"engine {module} timed out after {timeout}s"
            ) from exc
        reader.join(timeout=5)

        payload = _read_response(response_path)
        tail_text = "\n".join(stderr_tail)
        if payload is None:
            raise EngineSubprocessError(
                f"engine {module} wrote no response (exit {returncode})\n{tail_text}"
            )
        return _result_from_payload(module, payload)


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
    """One-shot child entry point (only :func:`_run_once`/`probe` launch this).

    Standard library only until the engine module loads.
    """
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


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "name": getattr(exc, "name", None),
    }


def _worker_main(argv: list[str]) -> int:
    """Persistent-worker child entry point: import once, serve requests forever.

    ``argv`` is ``[script, "--worker", module, root]``. Reads one control
    line per call from stdin — ``{"request": path, "response": path}`` — and
    never anything else from stdin or stdout, so library noise on either
    stream cannot corrupt the protocol (the module docstring's "response
    goes to a file" reasoning, now doubled for the control channel too).

    If the engine module fails to import, every subsequent request gets that
    same import error rather than a retried import — an adapter fix only
    takes effect once this resident child is torn down and a fresh one is
    spawned (module docstring).

    Exits when stdin hits EOF, which happens the moment the parent closes it
    (an explicit :func:`shutdown_workers`, or the parent process dying and
    taking the pipe down with it).
    """
    module_name = argv[2]
    root = argv[3] if len(argv) > 3 else None
    if root and root not in sys.path:
        sys.path.insert(0, root)

    import importlib

    module: Any = None
    import_error: dict[str, Any] | None = None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # the whole point: report it, don't crash
        import_error = _error_payload(exc)

    for raw_line in iter(sys.stdin.readline, ""):
        line = raw_line.strip()
        if not line:
            continue
        try:
            control = json.loads(line)
            request_path = Path(control["request"])
            response_path = Path(control["response"])
        except Exception:
            # A malformed control line names no response file to answer on;
            # nothing to do but wait for the next, well-formed one.
            continue
        if import_error is not None:
            payload: dict[str, Any] = {"ok": False, "error": import_error}
        else:
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                payload = {"ok": True, "result": module.child_main(request)}
            except Exception as exc:  # the boundary exists to report anything
                payload = {"ok": False, "error": _error_payload(exc)}
        _atomic_write_json(response_path, payload)
    return 0


class _Worker:
    """One resident child process bound to one ``(interpreter, module, root)``.

    Not constructed directly by callers — see :func:`get_worker`. Owns the
    child's lifecycle: spawn, one call at a time (serialised by
    ``_call_lock``), respawn on the next call if it has died, and
    :meth:`close`.
    """

    def __init__(self, interpreter: str, module: str, root: Path) -> None:
        self.interpreter = interpreter
        self.module = module
        self.root = root
        self._call_lock = threading.Lock()
        self._active_ctx: contextvars.Context | None = None
        self._stage = f"engine:{module.rsplit('.', 1)[-1]}"
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="rytp-worker-", ignore_cleanup_errors=True
        )
        self._counter = 0
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque()
        self._spawn()

    # -- lifecycle ----------------------------------------------------

    def _spawn(self) -> None:
        from rytp import constants as C
        from rytp.transcribe.base import EngineSubprocessError

        self._stderr_tail = deque(maxlen=C.ENGINE_SUBPROCESS_STDERR_TAIL)
        command = [
            self.interpreter,
            str(Path(__file__).resolve()),
            "--worker",
            self.module,
            str(self.root),
        ]
        try:
            self._proc = subprocess.Popen(  # fixed argv, no shell
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_spawn_env(),
            )
        except FileNotFoundError as exc:
            raise EngineSubprocessError(
                f"engine interpreter not found: {self.interpreter}"
            ) from exc
        assert self._proc.stderr is not None
        # Same contextvars trap as `_run_once`'s reader thread — but this one
        # lives for the worker's whole life, not one call, so `call()` swaps
        # `_active_ctx` in and out per call rather than capturing it once here.
        ctx = contextvars.copy_context()
        self._reader = threading.Thread(target=ctx.run, args=(self._pump_stderr,), daemon=True)
        self._reader.start()

    def _pump_stderr(self) -> None:
        from rytp.progress import report

        assert self._proc is not None and self._proc.stderr is not None
        with self._proc.stderr as pipe:
            for raw_line in pipe:
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                self._stderr_tail.append(line)
                active = self._active_ctx
                if active is not None:
                    active.run(report, self._stage, detail=line)
                else:
                    report(self._stage, detail=line)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _join_reader(self) -> None:
        if self._reader is not None:
            self._reader.join(timeout=5)
        self._reader = None

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()
                    proc.wait(timeout=5)
        self._join_reader()

    def close(self) -> None:
        """Ask the child to exit (stdin EOF), then make sure it is gone."""
        with self._call_lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            self._join_reader()
            self._tmpdir.cleanup()

    # -- calls ----------------------------------------------------------

    def call(self, request: dict[str, Any], *, timeout_s: int | None = None) -> dict[str, Any]:
        from rytp import constants as C
        from rytp.transcribe.base import EngineSubprocessError

        timeout = C.ENGINE_SUBPROCESS_TIMEOUT_S if timeout_s is None else timeout_s
        with self._call_lock:
            if not self.alive:
                self._join_reader()
                self._spawn()
            assert self._proc is not None and self._proc.stdin is not None

            self._stderr_tail.clear()
            self._counter += 1
            call_id = self._counter
            request_path = Path(self._tmpdir.name) / f"request-{call_id}.json"
            response_path = Path(self._tmpdir.name) / f"response-{call_id}.json"
            response_path.unlink(missing_ok=True)
            request_path.write_text(json.dumps(request), encoding="utf-8")
            control = json.dumps({"request": str(request_path), "response": str(response_path)})

            self._active_ctx = contextvars.copy_context()
            try:
                self._proc.stdin.write(control + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._active_ctx = None
                self._kill()
                tail = "\n".join(self._stderr_tail)
                raise EngineSubprocessError(
                    f"engine {self.module} worker died before accepting the call: {exc}\n{tail}"
                ) from exc

            deadline = time.monotonic() + timeout
            payload = _read_response(response_path)
            while payload is None and time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    # The child may have written the response and then exited
                    # (a clean worker asked to shut down mid-call would not,
                    # but a crashing one might race the write) — check once
                    # more before concluding it wrote nothing.
                    payload = _read_response(response_path)
                    break
                time.sleep(_POLL_INTERVAL_S)
                payload = _read_response(response_path)
            self._active_ctx = None

            if payload is None:
                died = self._proc.poll() is not None
                returncode = self._proc.poll()
                tail = "\n".join(self._stderr_tail)
                self._kill()
                if died:
                    raise EngineSubprocessError(
                        f"engine {self.module} wrote no response (exit {returncode})\n{tail}"
                    )
                raise EngineSubprocessError(f"engine {self.module} timed out after {timeout}s")

            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
            return _result_from_payload(self.module, payload)


#: Resident workers, one per ``(interpreter, module, repo_root)``, for the
#: life of this process. Guarded by `_WORKERS_LOCK`; see the module
#: docstring's lifecycle section.
_WORKERS: dict[tuple[str, str, str], _Worker] = {}
_WORKERS_LOCK = threading.Lock()


def get_worker(interpreter: str, module: str, repo_root: Path | None = None) -> _Worker:
    """The resident worker for this key, spawning one if there is none alive.

    A dead cached worker (crashed, killed, or timed out on a previous call)
    is closed and replaced rather than reused — self-healing at the cost of
    one reload, matching this project's job-queue philosophy of repairing
    state rather than refusing to proceed.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    key = (interpreter, module, str(root))
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is not None and not worker.alive:
            worker.close()
            worker = None
        if worker is None:
            worker = _Worker(interpreter, module, root)
            _WORKERS[key] = worker
        return worker


def shutdown_workers() -> None:
    """Close every resident worker. Registered with :mod:`atexit`; also a test hook.

    Safe to call with nothing running (an empty pool closes nothing) and
    safe to call more than once.
    """
    with _WORKERS_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.close()


atexit.register(shutdown_workers)


def run_child(
    *,
    interpreter: str,
    module: str,
    request: dict[str, Any],
    timeout_s: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run ``module.child_main(request)`` on the resident worker for this engine.

    Same signature as before the persistent seam existed — no adapter call
    site changes. The first call for a given ``(interpreter, module,
    repo_root)`` pays a spawn and an import; every later call to the same key
    reuses the same child, and — because each ``child_main`` now caches its
    own expensive state via :func:`load_cached` — the same loaded model too.

    Raises :class:`~rytp.transcribe.base.EngineSubprocessError` for a missing
    interpreter, a timeout, a crashed child, a child that wrote nothing, and a
    child that reported failure — identically to the one-shot child this
    replaced.
    """
    worker = get_worker(interpreter, module, repo_root=repo_root)
    return worker.call(request, timeout_s=timeout_s)


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """This module's own probe target (plan §Task 7, "one probe serves four entries").

    :func:`probe` calls :func:`_run_once` with ``module="rytp.transcribe.subproc"``,
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

    Deliberately a one-shot child (:func:`_run_once`), never a resident
    worker — see the module docstring. A probe may run against engines
    nobody ends up using; holding one of those open, possibly with a model
    loaded, would be exactly the waste this file exists to stop.
    """
    from rytp import constants as C

    try:
        result = _run_once(
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
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main(sys.argv))
    raise SystemExit(_child_main(sys.argv))
