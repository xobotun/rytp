"""Engine protocols and registries for the HF_TOKEN pre-launch gate.

The HF_TOKEN pre-launch gate (DESIGN §12) needs to inspect any STT,
Diarizer, or combined engine the user selects *before* running
anything, so we can print a clear paper-cut when a gated engine is
selected without a token set.

To make that work without instantiating the engine (which would
require the actual optional dependency, e.g. ``pyannote.audio``, to
be installed), every engine declares two class attributes:

* ``name`` — unique identifier for CLI selection (``"faster-whisper"``,
  ``"pyannote"``, ``"whisperx"``, etc.).
* ``requires_hf_token: bool`` — whether the gate must check for
  ``HF_TOKEN``.

Engine classes are registered via the ``@register_stt`` /
``@register_diarizer`` / ``@register_combined`` decorators and
resolved by name via ``resolve_stt`` / ``resolve_diarizer`` /
``resolve_combined``.

Two reasons the gate inspects **classes, not instances**:

1. We don't want to download 2 GB of model checkpoints just to tell
   the user their token is missing.
2. Some engines (notably pyannote) refuse to instantiate without a
   token; doing the check first means the error message points at
   the *cause* (missing token) rather than the *symptom*
   (RuntimeError during construction).

Public surface:

* :class:`Word`, :class:`DiarSegment`, :class:`DiarizedWord` —
  typed tuples shared between the engine and the rest of the
  codebase. Re-exported from :mod:`rytp.models` for canonical
  imports.
* :class:`STTEngine`, :class:`Diarizer`,
  :class:`TranscribeDiarizeEngine` — Protocols.
* :class:`NullDiarizer` — the default, no-op diarizer.
* :func:`resolve_stt`, :func:`resolve_diarizer`, :func:`resolve_combined`.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Protocol

from rytp import constants as C


class Word(NamedTuple):
    """A single transcribed word with timing and confidence.

    Attributes:
        start_ms: Word onset in milliseconds from the audio start.
        end_ms: Word offset in milliseconds.
        text: The transcribed text (whitespace-stripped).
        confidence: Engine-reported confidence, default 1.0.
    """

    start_ms: int
    end_ms: int
    text: str
    confidence: float


class DiarSegment(NamedTuple):
    """A diarization segment with a raw speaker label.

    Attributes:
        start_ms: Segment onset in ms.
        end_ms: Segment offset in ms.
        speaker: Raw diarizer label, e.g. ``"SPEAKER_02"``. Mapped to
            a canonical speaker id later by the speaker-mapping stage
            (DESIGN §6 ``speakers map``).
    """

    start_ms: int
    end_ms: int
    speaker: str  # raw label, e.g. "SPEAKER_02"


class DiarizedWord(NamedTuple):
    """A word that already carries a speaker label (combined engine output).

    Used by :class:`TranscribeDiarizeEngine` engines (whisperx, etc.)
    that produce words + speakers in a single pass — DESIGN §5.3.
    """

    start_ms: int
    end_ms: int
    text: str
    confidence: float
    speaker: str  # raw diarizer label, same semantics as DiarSegment.speaker


class STTEngine(Protocol):
    """Protocol for a speech-to-text engine.

    Attributes:
        name: Unique identifier for CLI selection (e.g. ``"faster-whisper"``).
        requires_hf_token: Whether the HF_TOKEN gate must check for a
            token before this engine can run.
    """

    name: str
    requires_hf_token: bool

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> Iterable[Word]:
        """Yield :class:`Word` objects with absolute timestamps (ms)."""
        ...


class Diarizer(Protocol):
    """Protocol for a speaker-diarization engine.

    Attributes:
        name: Unique identifier for CLI selection (e.g. ``"pyannote"``).
        requires_hf_token: Whether the HF_TOKEN gate must check.
    """

    name: str
    requires_hf_token: bool

    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]:
        """Yield :class:`DiarSegment` objects covering the full audio."""
        ...


class TranscribeDiarizeEngine(Protocol):
    """Protocol for engines that produce words + speakers in one pass (DESIGN §5.3)."""

    name: str
    requires_hf_token: bool

    def transcribe_diarize(
        self, audio_path: Path, *, language: str | None = None
    ) -> Iterable[DiarizedWord]:
        """Yield :class:`DiarizedWord` objects with absolute timestamps."""
        ...


class NullDiarizer:
    """No-op diarizer that assigns ``SPEAKER_00`` to the entire audio.

    Used when the user wants to skip diarization entirely. Does not
    require ``HF_TOKEN``. The single yielded segment uses
    :data:`rytp.constants.NULL_DIARIZER_END_MS` as its upper bound —
    large enough to cover any real video without depending on the
    actual audio duration (which the diarizer doesn't know). The
    merger uses interval overlap, so the exact value doesn't matter
    as long as it covers the audio.
    """

    name = "none"
    requires_hf_token = False

    def diarize(self, audio_path: Path) -> Iterable[DiarSegment]:
        # Duration is unknown here; the transcribe stage knows the audio
        # length. We yield a single segment that "covers everything";
        # the merger uses interval overlap, so the upper bound only
        # needs to be larger than any plausible word timestamp.
        yield DiarSegment(
            start_ms=0,
            end_ms=C.NULL_DIARIZER_END_MS,
            speaker=C.NULL_DIARIZER_SPEAKER_LABEL,
        )


# Registry: name -> class
STTS: dict[str, type[STTEngine]] = {}
"""Registered STT engine classes."""
DIARIZERS: dict[str, type[Diarizer]] = {}
"""Registered Diarizer classes."""
COMBINED: dict[str, type[TranscribeDiarizeEngine]] = {}
"""Registered combined engine classes."""


def register_stt(cls: type[STTEngine]) -> type[STTEngine]:
    """Class decorator: register an STT engine under ``cls.name``."""
    STTS[cls.name] = cls
    return cls


def register_diarizer(cls: type[Diarizer]) -> type[Diarizer]:
    """Class decorator: register a Diarizer under ``cls.name``."""
    DIARIZERS[cls.name] = cls
    return cls


def register_combined(cls: type[TranscribeDiarizeEngine]) -> type[TranscribeDiarizeEngine]:
    """Class decorator: register a combined engine under ``cls.name``."""
    COMBINED[cls.name] = cls
    return cls


def resolve_stt(name: str) -> type[STTEngine]:
    """Look up an STT engine class by name. Raises ``ValueError`` if not registered."""
    try:
        return STTS[name]
    except KeyError as e:
        raise ValueError(f"Unknown STT engine: {name!r}. Available: {sorted(STTS)}") from e


def resolve_diarizer(name: str) -> type[Diarizer]:
    """Look up a Diarizer class by name. Raises ``ValueError`` if not registered."""
    try:
        return DIARIZERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown diarizer: {name!r}. Available: {sorted(DIARIZERS)}") from e


def resolve_combined(name: str) -> type[TranscribeDiarizeEngine]:
    """Look up a combined engine class by name. Raises ``ValueError`` if not registered."""
    try:
        return COMBINED[name]
    except KeyError as e:
        raise ValueError(f"Unknown combined engine: {name!r}. Available: {sorted(COMBINED)}") from e


# Register the built-in NullDiarizer so it's available by default.
register_diarizer(NullDiarizer)