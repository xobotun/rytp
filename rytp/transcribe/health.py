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

import shutil
import subprocess
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp.transcribe.registry import ALIGNERS, TRANSCRIBERS, EngineClass, availability

if TYPE_CHECKING:
    from rytp.commands import HealthResult
    from rytp.db import Database

_READY = ("ready", "interpreter ok")


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


def engine_check(db: Database, cls: EngineClass) -> HealthResult:
    """Report one engine, from the class alone — nothing is constructed."""
    from rytp.commands import HealthResult

    state = availability(db, cls)
    ok = state in _READY
    return HealthResult(ok=ok, detail=state, remedy=None if ok else engine_remedy(cls))


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


register_transcribe_checks()
