"""What `doctor` reports about Part 3 (contracts §5, "Health checks").

The most-asked question on this machine, because these are the dependencies
that actually fail to install and several of them pin conflicting versions of
each other: *which transcribers and aligners can I run right now, and what
would it take to get the others?*

Every probe is cheap and offline. An in-process engine is checked with
``importlib.util.find_spec``, which consults the filesystem and neither
imports nor downloads anything. An out-of-process engine is checked by asking
whether its configured interpreter exists — its dependency lives in another
environment where ``find_spec`` here would be meaningless. The GPU is probed
with ``nvidia-smi`` rather than by importing torch, which costs seconds and
may not be installed at all.

Nothing here raises. Every check Part 3 registers is ``required=False``:
none of these is needed by the base install, so a missing engine reports
``ok=False`` — which is the truth about what was found — without failing
``doctor``. Contracts §5 keeps those two ideas apart on purpose, because the
alternative is claiming ``ok=True`` for a tool that is not there, which makes
the output lie about the one thing the command exists to tell you.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp.transcribe.registry import (
    ALIGNERS,
    TRANSCRIBERS,
    EngineClass,
    availability,
    interpreter_for,
    probe_engine,
)

if TYPE_CHECKING:
    from rytp.commands import HealthResult
    from rytp.db import Database

_READY = ("ready", "interpreter ok")

#: `pip install torch ...` without an index resolves the CPU-only PyPI wheel
#: on Windows (BUGS.md entries 33, 35). Named once so `engine_check` and
#: `docs/setup.md` never disagree about the remedy.
CUDA_TORCH_REMEDY = (
    "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 "
    "(match cuXXX to the installed driver)"
)


def engine_remedy(cls: EngineClass) -> str:
    """The exact command that would make this engine runnable."""
    extra = getattr(cls, "extra", None) or cls.name
    if not getattr(cls, "out_of_process", False):
        return f"pip install rytp[{extra}]"
    if cls.name == "mfa":
        return (
            "install the Montreal Forced Aligner in its own conda environment, then: "
            f"rytp settings set engine.interpreter.{cls.name} <path to that python>"
        )
    return (
        f"pip install rytp[{extra}] into its own virtualenv, then: "
        f"rytp settings set engine.interpreter.{cls.name} <path to that python>"
    )


def _cuda_fact(db: Database, cls: EngineClass) -> dict[str, object] | None:
    """The cached probe's CUDA facts for one out-of-process engine, if any.

    ``None`` for an in-process engine or one with no ``required_module``
    (MFA): there is nothing to probe for either. Reuses the same cached
    result :func:`availability` already produced — no second child process
    (BUGS.md entry 7's "must not spawn six processes twice").
    """
    if not getattr(cls, "out_of_process", False):
        return None
    required = getattr(cls, "required_module", None)
    if not required:
        return None
    interpreter = interpreter_for(db, cls.name)
    if not Path(interpreter).exists():
        return None
    return probe_engine(interpreter, "", required)


def engine_check(db: Database, cls: EngineClass) -> HealthResult:
    """Report one engine, from the class alone — nothing is constructed.

    BUGS.md entry 33: pairs the module-availability fact with the CUDA fact
    from the same probe, so "this engine's interpreter has no CUDA torch"
    is visible next to "this engine can run" instead of silently running
    40x slower on the CPU while the GPU idles.
    """
    from rytp.commands import HealthResult

    state = availability(db, cls)
    ok = state in _READY
    detail = state
    remedy = None if ok else engine_remedy(cls)

    probe = _cuda_fact(db, cls)
    if probe is not None:
        torch_version = probe.get("torch")
        cuda_available = probe.get("cuda_available")
        if torch_version is None:
            pass  # no torch in that interpreter at all; the module check above already says so
        elif cuda_available is False:
            detail += f"; torch {torch_version} has no CUDA (device={probe.get('device')})"
            remedy = remedy or CUDA_TORCH_REMEDY
        elif cuda_available is True:
            detail += f"; CUDA available (device={probe.get('device')})"

    return HealthResult(ok=ok, detail=detail, remedy=remedy)


def _nvidia_smi() -> str | None:
    """Where the driver's own query tool is. A module attribute, so a test can
    replace just this rather than reaching into the global ``shutil``."""
    return shutil.which("nvidia-smi")


def _query_gpu(binary: str) -> tuple[int, str]:
    """Run one fixed query. Isolated so tests can replace it."""
    proc = subprocess.run(
        [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=C.GPU_PROBE_TIMEOUT_S,
        check=False,
    )
    return proc.returncode, proc.stdout


def gpu_check(db: Database) -> HealthResult:
    """Is a CUDA GPU visible? Never fatal — the pipeline runs on the CPU too,
    just slowly enough that the owner wants to know."""
    from rytp.commands import HealthResult

    binary = _nvidia_smi()
    if binary is None:
        return HealthResult(
            ok=False,
            detail="nvidia-smi not found; transcription would run on the CPU",
            remedy="install the NVIDIA driver, or accept CPU-only transcription",
        )
    try:
        code, output = _query_gpu(binary)
    except (OSError, subprocess.SubprocessError) as exc:
        return HealthResult(
            ok=False,
            detail=f"nvidia-smi could not be run: {type(exc).__name__}",
            remedy="check the NVIDIA driver installation",
        )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if code != 0 or not lines:
        return HealthResult(
            ok=False,
            detail="nvidia-smi reported no GPU",
            remedy="check the NVIDIA driver installation",
        )
    return HealthResult(ok=True, detail="; ".join(lines))


def _symlinks_supported() -> bool:
    """Whether this Python can create a symlink, probed rather than guessed.

    ``os.name == "nt"`` is not the answer: Windows Developer Mode or an
    administrator prompt both make symlinks work, and assuming otherwise
    would tell a machine that already supports them to go looking for a
    setting it does not need. A throwaway pair in a fresh temp directory is
    the only way to ask honestly. ``NotImplementedError`` is what old
    Windows Pythons raised for this before ``os.symlink`` gained
    privilege-aware fallbacks; ``OSError`` (``WinError 1314``, "a required
    privilege is not held") is what current ones raise without Developer
    Mode or admin.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="rytp-symlink-probe-") as tmp:
            # A throwaway pair inside a fresh, always-local temp directory.
            root = Path(tmp)
            target = root / "target"
            target.write_text("x", encoding="utf-8")
            link = root / "link"
            os.symlink(target, link)
            return link.is_symlink()
    except (OSError, NotImplementedError):
        return False


def hf_cache_check(db: Database) -> HealthResult:
    """BUGS.md entry 11: say the symlink fact once, instead of the library twice.

    ``huggingface_hub`` prints this same finding itself, per process, with a
    paragraph of Windows Developer Mode advice — noise from a dependency, not
    something rytp is asking the user to act on. ``C.HF_SYMLINK_WARNING_ENV``
    (set by :func:`rytp.transcribe.subproc.run_child` for every child, and by
    :mod:`rytp.transcribe.engines.whisper` before it loads in-process)
    silences that. The underlying condition is real — without symlinks the
    HF cache stores a full duplicate per model revision, which matters on a
    machine meant to hold a video archive — so `doctor` names it instead.
    """
    from rytp.commands import HealthResult

    if _symlinks_supported():
        return HealthResult(ok=True, detail="the Hugging Face cache can use symlinks")
    return HealthResult(
        ok=False,
        detail=(
            "no symlink support: the Hugging Face cache stores a full duplicate "
            "per model revision instead of sharing blobs, which costs extra disk"
        ),
        remedy="enable Windows Developer Mode, or run as an administrator, to allow symlinks",
    )


def register_transcribe_checks() -> None:
    """Register one check per registered engine, plus the GPU probe.

    Idempotent, because importing the command group more than once must not
    be an error. Imports the adapter packages first: an engine that was never
    imported never registered, and would be invisible here exactly as it is
    invisible to ``resolve_transcriber``.
    """
    import rytp.transcribe.align
    import rytp.transcribe.engines  # noqa: F401
    from rytp.commands import HEALTH_CHECKS, HealthCheck, register_check

    engines: list[tuple[str, str, EngineClass]] = [
        ("transcriber", name, cls) for name, cls in sorted(TRANSCRIBERS.items())
    ] + [("aligner", name, cls) for name, cls in sorted(ALIGNERS.items())]
    for kind, name, cls in engines:
        check_name = f"{kind}:{name}"
        if check_name in HEALTH_CHECKS:
            continue
        register_check(
            HealthCheck(
                name=check_name,
                summary=f"the {name} {kind} can run on this machine",
                run=(lambda engine: lambda db: engine_check(db, engine))(cls),
                # Advisory: nothing in Part 3 is required by the base
                # install, so a missing engine reports ok=False honestly
                # without failing `doctor` (contracts §5).
                required=False,
            )
        )
    if "gpu" not in HEALTH_CHECKS:
        register_check(
            HealthCheck(
                name="gpu",
                summary="a CUDA GPU is visible to the driver",
                run=gpu_check,
                required=False,
            )
        )
    if "hf-cache" not in HEALTH_CHECKS:
        register_check(
            HealthCheck(
                name="hf-cache",
                summary="whether the Hugging Face cache can use symlinks",
                run=hf_cache_check,
                required=False,
            )
        )


register_transcribe_checks()
