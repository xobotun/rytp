"""The types every stage shares, and the error every surface catches.

Contracts §4 fixes the dataclasses below; do not add or rename fields
without changing the contracts document first. ``ChannelEntry`` is the
one addition: it is what ``rytp/acquire/ytdlp.py`` (plan part 2) returns
from ``enumerate_channel`` and ``probe_video``, and it lives here so the
catalog commands can be written and tested before that module exists.

Both text functions are implemented here, not stubbed. Contracts §4 keeps
them in this module on purpose: ``words.stem`` is written by part 3 as
words are created, so putting the stemmer under ``rytp/index/`` would
invert the dependency. There is no ``rytp/index/stem.py``.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "ChannelEntry",
    "DiarSegment",
    "Fragment",
    "InvalidInputError",
    "NotFoundError",
    "RawWord",
    "RytpError",
    "Span",
    "UnknownEngineError",
    "normalize_text",
    "stem_text",
    "utc_now_iso",
]


class RytpError(Exception):
    """An expected failure with a message a user can act on.

    Contracts §8: the surface catches this, prints ``str(exc)`` as one
    line on stderr and exits 1. Never a traceback. Anything that is not
    a ``RytpError`` is a bug and is allowed to propagate.
    """


class NotFoundError(RytpError):
    """A referenced row, command or file does not exist."""


class InvalidInputError(RytpError):
    """Arguments were well-formed but wrong (bad enum value, empty id)."""


class UnknownEngineError(RytpError, ValueError):
    """``resolve_transcriber`` / ``resolve_aligner`` / ``resolve_diarizer`` found no such name.

    Contracts §6, amendment 5 (`docs/superpowers/2026-09-25-contracts-amendments.md`):
    the dual inheritance is deliberate, not decorative. ``ValueError`` keeps
    contracts §6's "raises ``ValueError``" wording literally true and lets a
    caller that already catches ``ValueError`` keep working unchanged.
    ``RytpError`` is what the CLI funnel (§8) catches to print one line and
    exit 1 instead of the ~60-line traceback a bare ``ValueError`` produced
    (BUGS.md entry 16). A caller may catch either base and get the same
    exception.
    """


@dataclass(frozen=True)
class RawWord:
    """What a transcriber emits, before alignment.

    Both timings are optional: a transcriber may emit text only, leaving
    every boundary to the aligner. Timings, when present, are absolute
    against the source audio, never relative to a chunk.
    """

    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Span:
    """A refined time range for one word."""

    start_ms: int
    end_ms: int
    score: float | None


@dataclass(frozen=True)
class DiarSegment:
    """One diarizer segment, labelled per video, not globally."""

    start_ms: int
    end_ms: int
    local_label: str


@dataclass(frozen=True)
class Fragment:
    """One contiguous run taken from one video. The unit of a cut list."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class ChannelEntry:
    """One video as a channel listing or a metadata probe describes it.

    ``kind`` is one of ``constants.VIDEO_KINDS``; a channel listing
    derives it from the tab it came from (design §13). There is no
    ``source`` field: everything reached through yt-dlp is recorded as
    ``constants.REMOTE_SOURCE``.
    """

    external_id: str
    title: str
    url: str
    duration_ms: int | None
    kind: str
    published_at: str | None


# ``\w`` under re.UNICODE keeps Cyrillic letters and digits; everything
# else becomes a space so "кто-то" splits into two searchable tokens.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

# ё/Ё → е/Е. Russian sources spell these interchangeably and a search for
# "еще" must find "ещё". The FTS5 tokenizer cannot do this for us: it is
# configured `remove_diacritics 0` (contracts §3) because the alternative
# also folds й into и, which is a different letter.
_YO_FOLD = str.maketrans({"ё": "е", "Ё": "Е"})


def normalize_text(text: str) -> str:
    """Canonical form used by ``normalized_text`` columns and the FTS index.

    NFC-normalize, fold ё to е, lowercase, replace punctuation with a
    space, collapse whitespace, strip. Idempotent.
    """
    text = unicodedata.normalize("NFC", text).translate(_YO_FOLD).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


_stemmer_local = threading.local()


def _russian_stemmer() -> Any:
    """The snowball stemmer, built once per thread.

    snowballstemmer's stemmer instance keeps mutable working state on
    ``self`` across the calls inside ``stemWords``, so one instance
    shared between threads (a module-level singleton, as this used to
    be) corrupts under concurrent use — Part 2's worker runs the cpu and
    gpu pools as threads in one process, and the index job stems on one
    while transcription stems on the other. A thread-local instance costs
    one construction per worker thread rather than per call, which is
    the same "not free but rare" trade the old singleton made, just
    scoped per thread instead of per process.
    """
    stemmer = getattr(_stemmer_local, "value", None)
    if stemmer is None:
        import snowballstemmer

        stemmer = snowballstemmer.stemmer("russian")
        _stemmer_local.value = stemmer
    return stemmer


def stem_text(normalized: str) -> str:
    """Reduce every token of an already-normalized string to its Russian stem.

    Token for token: ``stem_text(s).split()`` is as long as ``s.split()``,
    which is what lets an utterance's ``stem_text`` be assembled by
    joining per-word ``words.stem`` values.

    **Not idempotent.** ``сказали`` stems to ``сказа``, which stems again
    to ``сказ``. Never re-stem an already-stemmed string (contracts §4).

    Pass the output of :func:`normalize_text`, not raw text: the ё fold
    and the punctuation split have to happen first or the two columns
    stop agreeing.
    """
    tokens = normalized.split()
    if not tokens:
        return ""
    return " ".join(_russian_stemmer().stemWords(tokens))


def utc_now_iso() -> str:
    """Now, as the ISO-8601 UTC string every timestamp column stores (contracts §8)."""
    return datetime.now(UTC).isoformat()
