"""What `doctor` reports about speakers (contracts §5).

Two questions, answered without importing an engine, constructing a
pipeline or touching the network: which diarizers could run on this
machine, and whether the Hugging Face token the gated one needs is set.

The second is the one that earns its place. pyannote's community-1 is a
gated model, and without a token the failure arrives deep inside pipeline
construction — after a download has begun — as an HTTP error that names
nothing useful. Two people will hit that and conclude the tool is broken.
Saying it up front, with the page to accept the terms on and the variable
to set, is what this check is for.

Neither check raises and neither is fatal by default. The null diarizer has
no dependencies and is the default, so a missing optional engine is a note.
`ok` goes false only when the diarizer this machine is *configured* to use
cannot run here — a real breakage on this machine, and the only thing in
this file worth a non-zero exit from `doctor`.
"""

from __future__ import annotations

import os

from rytp import constants as C
from rytp.commands import HealthCheck, HealthResult, register_check
from rytp.db import Database

#: The two words Part 3's `availability` uses for "this could run now".
#: Everything else it returns is a reason it could not.
_USABLE_STATES = frozenset({"ready", "interpreter ok"})


def usable(state: str) -> bool:
    """Whether one of `availability`'s answers means the engine can run."""
    return state in _USABLE_STATES


def hf_token() -> str | None:
    """The Hugging Face token, under either accepted name. Never logged."""
    for name in C.HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def configured_diarizer(db: Database) -> str:
    """Which diarizer this machine would use if asked right now.

    The same precedence `rytp.diarize.pipeline.diarizer_name` applies, read
    straight from settings so that importing this module does not drag the
    pipeline — and with it the store — into every `doctor` run.
    """
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_DIARIZER) or C.DEFAULT_DIARIZER


def _states(db: Database) -> dict[str, str]:
    """Each registered diarizer's availability, tolerating a broken engine.

    An engine class is third-party-shaped code; a check that a badly
    written one can crash is a check that makes `doctor` useless exactly
    when it is needed.
    """
    from typing import cast

    from rytp.diarize.base import DIARIZERS
    from rytp.transcribe.registry import EngineClass, availability

    out: dict[str, str] = {}
    for name, cls in sorted(DIARIZERS.items()):
        try:
            out[name] = availability(db, cast("EngineClass", cls))
        except Exception as exc:  # a check must never raise
            out[name] = f"unusable: {type(exc).__name__}"
    return out


def _cuda_note(db: Database, wanted: str) -> str:
    """BUGS.md entry 33 for the configured diarizer: pair "it can run" with
    "does its interpreter's torch see a GPU", the same probe entry 7 already
    pays for. Empty for `none` (no dependency at all) and for any diarizer
    whose class was replaced by a test double with no ``required_module``.
    """
    from rytp.diarize.base import DIARIZERS
    from rytp.transcribe.registry import interpreter_for, probe_engine

    cls = DIARIZERS.get(wanted)
    if cls is None or not getattr(cls, "out_of_process", False):
        return ""
    required = getattr(cls, "required_module", None)
    if not required:
        return ""
    interpreter = interpreter_for(db, wanted)
    from pathlib import Path

    if not Path(interpreter).exists():
        return ""
    probe = probe_engine(interpreter, "", required)
    torch_version = probe.get("torch")
    cuda_available = probe.get("cuda_available")
    if torch_version is None:
        return ""
    if cuda_available is False:
        return f"; torch {torch_version} has no CUDA (device={probe.get('device')})"
    if cuda_available is True:
        return f"; CUDA available (device={probe.get('device')})"
    return ""


def check_diarizers(db: Database) -> HealthResult:
    """Which diarizers could run here, and whether the configured one can.

    `ok` is the plain truth — false when any registered diarizer cannot run,
    because one of them cannot. The check is registered `required=False`, so
    that is a note rather than a failure: `none` needs nothing and is the
    default, and nobody is obliged to install pyannote.
    """
    states = _states(db)
    ready = [name for name, state in states.items() if usable(state)]
    missing = {name: state for name, state in states.items() if not usable(state)}
    wanted = configured_diarizer(db)

    detail = "usable: " + (", ".join(ready) or "none")
    if missing:
        detail += "; unusable: " + ", ".join(
            f"{name} ({state})" for name, state in missing.items()
        )
    detail += f"; configured: {wanted}"

    if wanted not in states:
        return HealthResult(
            ok=False,
            detail=f"{detail}; {wanted!r} is not a registered diarizer",
            remedy=(
                f"set the {C.SETTINGS_DIARIZER} setting to one of: "
                f"{', '.join(sorted(states)) or '(none)'}"
            ),
        )
    if not missing:
        return HealthResult(ok=True, detail=f"{detail}{_cuda_note(db, wanted)}")

    # Lead with the engine this machine is actually set up to use: that is
    # the one whose absence will bite today.
    extras = ", ".join(f"pip install rytp[{name}]" for name in sorted(missing))
    if wanted in missing:
        remedy = (
            f"the configured diarizer {wanted!r} cannot run here: "
            f"pip install rytp[{wanted}] into its own environment, point "
            f"engine.interpreter.{wanted} at that python, or set "
            f"{C.SETTINGS_DIARIZER} to 'none'"
        )
    else:
        remedy = f"{wanted!r} works, so nothing is broken; for the others: {extras}"
    return HealthResult(ok=False, detail=detail, remedy=remedy)


def check_hf_token(db: Database) -> HealthResult:
    """Whether the token the gated diarizer needs is set.

    `ok` is simply whether a token exists — not whether anybody currently
    needs one. The check is `required=False`, so an absent token is a note
    with a remedy rather than a failed `doctor`, and somebody who never
    intends to touch pyannote can ignore it forever.

    Reports that a token *exists*, never what it is: `doctor` output is the
    first thing anybody pastes into a bug report.
    """
    from rytp.diarize.base import DIARIZERS

    gated = sorted(
        name
        for name, cls in DIARIZERS.items()
        if getattr(cls, "requires_hf_token", False)
    )
    variables = " or ".join(C.HF_TOKEN_ENV_VARS)

    if hf_token() is not None:
        return HealthResult(ok=True, detail=f"{variables} is set")

    detail = f"{variables} is not set"
    if gated:
        detail += f"; gated diarizers: {', '.join(gated)}"
    if configured_diarizer(db) not in gated:
        detail += "; nothing configured needs it yet"
    return HealthResult(
        ok=False,
        detail=detail,
        remedy=(
            f"open https://huggingface.co/{C.PYANNOTE_DIARIZATION_MODEL}, accept the "
            f"model terms, then set HF_TOKEN to a token from "
            f"https://huggingface.co/settings/tokens"
        ),
    )


# Both advisory (contracts §5). The null diarizer needs nothing and is the
# default, and nobody is obliged to use a gated model — so a false `ok` here
# is information, never a reason for `doctor` to exit non-zero.
register_check(
    HealthCheck(
        name="diarizers",
        summary="Which diarizers can run on this machine.",
        run=check_diarizers,
        required=False,
    )
)
register_check(
    HealthCheck(
        name="hf-token",
        summary="Whether the Hugging Face token the gated diarizer needs is set.",
        run=check_hf_token,
        required=False,
    )
)
