"""Name to class registries for transcribers and aligners (contracts §6).

Classes, never instances. The Hugging Face token gate and the "is this engine
even installed" listing have to be answerable *before* anything is
constructed, because constructing a gated engine downloads gigabytes only to
discover the token is missing. An engine class is therefore inspected, not
built, until the moment it is used.

Import-light, like :mod:`rytp.transcribe.base`: engine adapters import this
module at the bottom of their own file to register themselves, and those
adapters are imported inside a foreign interpreter.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rytp import constants as C
from rytp.models import UnknownEngineError
from rytp.transcribe.base import Aligner, EngineUnavailable, Transcriber

if TYPE_CHECKING:
    from rytp.db import Database

TRANSCRIBERS: dict[str, type[Transcriber]] = {}
"""Registered transcriber classes, by ``cls.name``."""

ALIGNERS: dict[str, type[Aligner]] = {}
"""Registered aligner classes, by ``cls.name``."""

#: Either kind of registered engine class. Both protocols share
#: ``name``/``requires_hf_token``/``out_of_process``, so the gate, the
#: availability check and `doctor`'s checks all take either without caring
#: which — narrower than bare ``type``, which has none of those attributes.
EngineClass = type[Transcriber] | type[Aligner]


def register_transcriber(cls: type[Transcriber]) -> type[Transcriber]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    TRANSCRIBERS[cls.name] = cls
    return cls


def register_aligner(cls: type[Aligner]) -> type[Aligner]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    ALIGNERS[cls.name] = cls
    return cls


def resolve_transcriber(name: str) -> type[Transcriber]:
    """Look up a transcriber class.

    Raises :class:`~rytp.models.UnknownEngineError`, which is both a
    ``ValueError`` (contracts §6's wording) and a ``RytpError`` (so the CLI
    funnel prints one line and exits 1 instead of a traceback — BUGS.md
    entry 16).
    """
    try:
        return TRANSCRIBERS[name]
    except KeyError:
        available = ", ".join(sorted(TRANSCRIBERS)) or "(none registered)"
        raise UnknownEngineError(
            f"unknown transcriber {name!r}; available: {available}"
        ) from None


def resolve_aligner(name: str) -> type[Aligner]:
    """Look up an aligner class. Raises :class:`~rytp.models.UnknownEngineError`."""
    try:
        return ALIGNERS[name]
    except KeyError:
        available = ", ".join(sorted(ALIGNERS)) or "(none registered)"
        raise UnknownEngineError(
            f"unknown aligner {name!r}; available: {available}"
        ) from None


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _binary_present(interpreter: str, name: str) -> bool:
    """Whether the ``name`` executable can actually be found (BUGS.md entry 7).

    Checked beside the configured interpreter first — the same place a real
    run resolves it (``align/mfa.py``'s ``mfa_binary``) — then on ``PATH``.
    Portable: ``shutil.which`` walks ``PATHEXT`` on Windows by itself, so only
    the beside-interpreter guess needs the explicit ``.exe`` suffix.
    """
    exe_name = f"{name}.exe" if os.name == "nt" else name
    beside = Path(interpreter).parent / exe_name
    if beside.exists():
        return True
    return shutil.which(name) is not None


#: Cached probe results, keyed by ``(interpreter, required_module)``, for the
#: life of the process (BUGS.md entry 7: ``transcribe engines`` lists six
#: engines and must not spawn six processes twice — nor, since `doctor` asks
#: the same question again for the CUDA fact, twelve). Tests that fake
#: :func:`rytp.transcribe.subproc.probe` must also clear this dict, or a
#: faked result leaks into a later, unrelated test.
_PROBE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def probe_engine(interpreter: str, module: str, required_module: str) -> dict[str, Any]:
    """Cached wrapper around :func:`rytp.transcribe.subproc.probe`.

    ``module`` is the engine adapter's own dotted path (may be ``""`` when
    unknown to the caller); ``required_module`` is the third-party dependency
    actually being asked about, and is what keys the cache — two engines that
    both need ``torch`` under the same interpreter share one child process.
    """
    key = (interpreter, required_module)
    cached = _PROBE_CACHE.get(key)
    if cached is None:
        from rytp.transcribe.subproc import probe

        cached = probe(interpreter, module, required_module)
        _PROBE_CACHE[key] = cached
    return cached


def check_available(cls: EngineClass, *, interpreter: str | None = None) -> None:
    """Raise :class:`EngineUnavailable` if this engine cannot run on this machine.

    Checked against the class, never an instance. An in-process engine's
    dependency is checked with ``find_spec`` in this interpreter. An
    out-of-process engine's dependency lives in another interpreter, so
    ``find_spec`` here would be meaningless (BUGS.md entry 7) — pass its
    resolved ``interpreter`` (:func:`interpreter_for`) and the *actual* child
    is probed instead. Every call site in this codebase passes one; a future
    caller with no interpreter to hand gets today's weaker, module-blind
    check rather than a crash — this keyword defaults to ``None`` rather
    than being required, precisely so that stays true.
    """
    if getattr(cls, "requires_hf_token", False) and not _hf_token():
        raise EngineUnavailable(
            f"engine {cls.name!r} needs a Hugging Face token; set HF_TOKEN"
        )
    module = getattr(cls, "required_module", None)
    if not module:
        binary = getattr(cls, "required_binary", None)
        if not binary or not getattr(cls, "out_of_process", False) or interpreter is None:
            return
        if not _binary_present(interpreter, binary):
            extra = getattr(cls, "extra", None) or cls.name
            raise EngineUnavailable(
                f"engine {cls.name!r} needs the {binary!r} binary; it installs "
                f"via conda, not pip — e.g. `conda install -c conda-forge "
                f"montreal-forced-aligner` in the environment named by the "
                f"{extra!r} interpreter setting"
            )
        return
    if getattr(cls, "out_of_process", False):
        if interpreter is None:
            return
        result = probe_engine(interpreter, "", module)
        if not result.get("module_ok"):
            extra = getattr(cls, "extra", None) or cls.name
            reason = result.get("module_error") or f"{module!r} did not import"
            raise EngineUnavailable(
                f'install the {extra} extra: pip install -e ".[{extra}]" ({reason})'
            )
        return
    if not _module_present(module):
        extra = getattr(cls, "extra", None) or cls.name
        raise EngineUnavailable(
            f"engine {cls.name!r} needs {module!r}: pip install rytp[{extra}]"
        )


def setting(db: Database, key: str, default: str | None = None) -> str | None:
    """Read one row of the ``settings`` table (contracts §3)."""
    row = db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else str(row[0])


def default_transcriber(db: Database) -> str:
    """The engine to use when no ``--transcriber`` was named (contracts §3).

    Reads Part 2's ``default_transcriber`` setting, falling back to the
    shipped value — Part 1's migration seeds ``default_aligner`` and not this
    key, so the fallback is what makes "defaults to gigaam" true. **Empty is
    not meaningful here**, unlike for the aligner: an empty aligner reads as
    "no alignment, the words stay `timed`", an empty transcriber would mean
    nothing runs. So this always returns a name.

    `gigaam` is the Russian-specific engine and published benchmarks put it
    at roughly half Whisper's Russian word error rate, which matters at
    35-90 GPU-hours for the corpus. **A starting point, not a verdict:** the
    one published test on *noisy YouTube* audio — which is exactly this
    corpus — favoured a Russian-finetuned Whisper instead. That is what
    ``transcribe compare`` is for; re-run it on real material and rewrite the
    setting rather than treating this value as settled.

    Callers that used this rather than an explicit name must say so, and must
    let :func:`check_available` raise when the chosen engine is not
    installed — never substitute another one. ``words.engine`` records what
    actually ran, which only helps if nothing lies about what it used.
    """
    name = (setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "") or "").strip()
    return name or C.DEFAULT_TRANSCRIBER_FALLBACK


def interpreter_for(db: Database, name: str) -> str:
    """Interpreter an out-of-process engine runs under (design §6).

    Per-engine setting first, then the shared default, then this interpreter —
    which is correct for an engine whose dependencies happen not to conflict.
    """
    value = setting(db, f"{C.SETTINGS_INTERPRETER_PREFIX}{name}")
    if value is None:
        value = setting(db, C.SETTINGS_INTERPRETER_DEFAULT)
    return value or sys.executable


def load_transcriber(db: Database, name: str, **kwargs: Any) -> Transcriber:
    """Resolve, gate, and construct a transcriber."""
    cls = resolve_transcriber(name)
    interpreter = interpreter_for(db, name) if getattr(cls, "out_of_process", False) else None
    check_available(cls, interpreter=interpreter)
    if interpreter is not None:
        kwargs.setdefault("interpreter", interpreter)
    return cls(**kwargs)


def load_aligner(db: Database, name: str, **kwargs: Any) -> Aligner:
    """Resolve, gate, and construct an aligner."""
    cls = resolve_aligner(name)
    interpreter = interpreter_for(db, name) if getattr(cls, "out_of_process", False) else None
    check_available(cls, interpreter=interpreter)
    if interpreter is not None:
        kwargs.setdefault("interpreter", interpreter)
    return cls(**kwargs)


def availability(db: Database, cls: EngineClass) -> str:
    """One human-readable phrase on whether this engine could run right now.

    BUGS.md entry 7: an out-of-process engine used to stop at "the
    interpreter exists", which is a promise the column does not keep — five
    out of five out-of-process engines said ``interpreter ok`` and none of
    them could run. When the engine declares ``required_module``, the
    configured interpreter is actually probed (:func:`probe_engine`) and the
    answer means "this will actually run", not just "a python is there".
    An engine with no ``required_module`` but a ``required_binary`` (MFA — a
    conda binary, not a ``pip``-importable module) is checked with
    :func:`_binary_present` instead, so it does not fall back to the weaker
    ``"interpreter ok"`` either; only an engine with neither attribute does.
    """
    if getattr(cls, "requires_hf_token", False) and not _hf_token():
        return "no HF_TOKEN"
    if getattr(cls, "out_of_process", False):
        interpreter = interpreter_for(db, cls.name)
        path = Path(interpreter)
        if not path.exists():
            return f"interpreter missing: {path}"
        required = getattr(cls, "required_module", None)
        if not required:
            binary = getattr(cls, "required_binary", None)
            if binary:
                return (
                    "ready"
                    if _binary_present(interpreter, binary)
                    else f"needs the {binary} binary (conda, not pip)"
                )
            return "interpreter ok"
        result = probe_engine(interpreter, "", required)
        if not result.get("module_ok"):
            reason = result.get("module_error") or f"{required!r} did not import"
            return f"needs {required} ({reason})"
        return "ready"
    module = getattr(cls, "required_module", None)
    if not module:
        return "ready"
    return "ready" if _module_present(module) else f"needs {module}"


def _rows_for(
    db: Database, kind: str, table: Mapping[str, EngineClass]
) -> list[tuple[str, str, str, str, str]]:
    return [
        (
            name,
            kind,
            "yes" if getattr(cls, "requires_hf_token", False) else "no",
            "yes" if getattr(cls, "out_of_process", False) else "no",
            availability(db, cls),
        )
        for name, cls in sorted(table.items())
    ]


def engine_rows(db: Database) -> list[tuple[str, str, str, str, str]]:
    """Rows for ``rytp transcribe engines``: name, kind, token, process, state."""
    return _rows_for(db, "transcriber", TRANSCRIBERS) + _rows_for(db, "aligner", ALIGNERS)
