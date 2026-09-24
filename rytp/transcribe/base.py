"""Engine protocols for transcription and alignment (contracts §6, design §6).

Import-light on purpose. Every engine adapter imports this module, and an
out-of-process adapter is imported again inside a *foreign* interpreter where
only the standard library and :mod:`rytp.models` are guaranteed to load.
Never import numpy, torch, or anything from :mod:`rytp.audio` here.

Timestamp convention
--------------------
:meth:`Transcriber.transcribe` and :meth:`Aligner.align` receive a window
(``start_ms`` / ``end_ms``) into a whole-file 16 kHz mono WAV and MUST emit
**absolute** timestamps, measured from the start of that file rather than from
the start of the window. An adapter that decodes a slice writes the slice with
:func:`slice_wav_window` and shifts the result with :func:`shift_words`.
"""
from __future__ import annotations

import re
import wave
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from rytp.models import RawWord, RytpError, Span, normalize_text


class EngineUnavailable(RytpError):  # noqa: N818 - exact name pinned by contracts §6
    """A named engine is registered but cannot run here (missing extra or token)."""


class EngineSubprocessError(RytpError):
    """An out-of-process engine failed; the message carries the child's report."""


class Transcriber(Protocol):
    """Audio in, words out. Timestamps are absolute — see the module docstring."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]: ...


class Aligner(Protocol):
    """Audio plus known text in, one :class:`Span` per word out, absolute ms."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]: ...


#: Everything a word row may not contain. ``normalize_text`` turns each of
#: these into a space, so a token spanning one of them would normalize to two.
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def split_token(text: str) -> list[tuple[str, str]]:
    """Split one transcriber or caption token into the tokens that become rows.

    Returns ``(surface, normalized)`` pairs — the surface form for
    ``words.text``, the normalized form for ``words.normalized_text``.

    **Why a hyphenated word becomes two rows.** `normalize_text` replaces
    punctuation with a space, so "кто-то" normalizes to "кто то". A single row
    holding two tokens can never be found: Part 4's index and FTS lookups are
    single-token and Part 5's assembly pointer walk assumes one token per row.
    Stripping the hyphen instead — storing "ктото" — would be worse, because
    the *same* normalizer runs over the user's query, which also splits into
    two tokens, so the stripped row would be unreachable from either
    direction. One row per token is the only form that both sides agree on.

    Punctuation-only input yields nothing, which is how "—" and a stray comma
    are kept out of the ordinal sequence.
    """
    pairs: list[tuple[str, str]] = []
    for piece in _TOKEN_SPLIT_RE.split(text):
        normalized = normalize_text(piece)
        if normalized:
            pairs.append((piece, normalized))
    return pairs


def shift_words(words: Iterable[RawWord], offset_ms: int) -> list[RawWord]:
    """Move window-relative words onto the whole-file timeline."""
    return [
        RawWord(
            start_ms=None if w.start_ms is None else w.start_ms + offset_ms,
            end_ms=None if w.end_ms is None else w.end_ms + offset_ms,
            text=w.text,
            confidence=w.confidence,
        )
        for w in words
    ]


def shift_spans(spans: Iterable[Span], offset_ms: int) -> list[Span]:
    """Move window-relative spans onto the whole-file timeline."""
    return [
        Span(start_ms=s.start_ms + offset_ms, end_ms=s.end_ms + offset_ms, score=s.score)
        for s in spans
    ]


def slice_wav_window(src: Path, dst: Path, start_ms: int, end_ms: int | None) -> Path:
    """Copy ``[start_ms, end_ms)`` of a PCM WAV into ``dst``, standard library only.

    Engines take a file, not an array, and the child process cannot import
    numpy. ``wave`` handles the frame arithmetic and writes a valid header.
    """
    with wave.open(str(src), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        total = reader.getnframes()
        first = min(max(0, start_ms * rate // 1000), total)
        last = total if end_ms is None else min(max(first, end_ms * rate // 1000), total)
        reader.setpos(first)
        frames = reader.readframes(last - first)
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return dst
