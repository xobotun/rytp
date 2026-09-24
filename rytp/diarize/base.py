"""The diarizer protocol and registry (contracts §6, design §6).

Classes, never instances — the Hugging Face token gate has to be answerable
*before* anything is constructed, because building a pyannote pipeline
without a token downloads two gigabytes only to fail on the token.

The gate itself, the per-engine interpreter lookup and the availability
summary are Part 3's (`rytp.transcribe.registry`) and are reused rather
than duplicated: an engine is an engine, and two implementations of "can
this run here" is how the two answers drift apart. Those helpers are
stdlib-only, so importing them keeps this module import-light — which
matters, because `rytp/diarize/pyannote.py` is imported inside a *foreign*
interpreter where only the standard library and its own dependency load.
"""

from __future__ import annotations

import wave
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from rytp.models import DiarSegment, RytpError
from rytp.transcribe.registry import check_available, interpreter_for

if TYPE_CHECKING:
    from rytp.db import Database
    from rytp.transcribe.registry import EngineClass

__all__ = [
    "DIARIZERS",
    "DiarizeError",
    "Diarizer",
    "diarizer_rows",
    "load_diarizer",
    "register_diarizer",
    "resolve_diarizer",
    "wav_duration_ms",
]


class DiarizeError(RytpError):
    """Diarization could not produce usable labels for this video."""


class Diarizer(Protocol):
    """Audio in, labelled time ranges out. Absolute milliseconds."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def diarize(self, audio: Path) -> Iterable[DiarSegment]: ...


DIARIZERS: dict[str, type[Diarizer]] = {}
"""Registered diarizer classes, by ``cls.name``."""


def register_diarizer(cls: type[Diarizer]) -> type[Diarizer]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    DIARIZERS[cls.name] = cls
    return cls


def resolve_diarizer(name: str) -> type[Diarizer]:
    """Look up a diarizer class. Raises ``ValueError`` naming the alternatives."""
    try:
        return DIARIZERS[name]
    except KeyError:
        available = ", ".join(sorted(DIARIZERS)) or "(none registered)"
        raise ValueError(f"unknown diarizer {name!r}; available: {available}") from None


def load_diarizer(db: Database, name: str, **kwargs: Any) -> Diarizer:
    """Resolve, gate, and construct a diarizer."""
    cls = resolve_diarizer(name)
    check_available(cast("EngineClass", cls))
    if getattr(cls, "out_of_process", False):
        kwargs.setdefault("interpreter", interpreter_for(db, name))
    return cls(**kwargs)


def diarizer_rows(db: Database) -> list[tuple[str, str, str, str]]:
    """Rows for ``rytp speakers engines``: name, token, out-of-process, state."""
    from rytp.transcribe.registry import availability

    return [
        (
            name,
            "yes" if getattr(cls, "requires_hf_token", False) else "no",
            "yes" if getattr(cls, "out_of_process", False) else "no",
            availability(db, cast("EngineClass", cls)),
        )
        for name, cls in sorted(DIARIZERS.items())
    ]


def wav_duration_ms(path: Path) -> int:
    """Length of a PCM WAV in milliseconds, from its header alone.

    Standard library only: the null diarizer needs a duration and must not
    pull numpy or :mod:`rytp.audio` into an import path that a foreign
    interpreter may follow.
    """
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
    if rate <= 0:
        raise DiarizeError(f"{path} reports a sample rate of {rate}")
    return frames * 1000 // rate
