"""Plain-data dataclasses used across the codebase.

Two layers of types live here:

* **Engine I/O shapes** (``Word``, ``DiarSegment``, ``DiarizedWord``) —
  re-exported from :mod:`rytp.engines` so the rest of the codebase has
  one canonical place to import them. They are protocol shapes: the
  STT, Diarizer, and combined engine interfaces return them.

* **Domain dataclasses** (``Video``, ``Speaker``, ``Clip``) — what we
  build up in the DB layer. Each has a ``from_row`` classmethod for
  cheap construction from ``sqlite3.Row``.

A small helper, :func:`normalize_text`, is the canonical normalizer for
``words.normalized_text`` and the FTS5 shadow.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Re-export engine I/O shapes for a single canonical import path.
from rytp.engines import DiarSegment, DiarizedWord, Word  # noqa: F401

__all__ = [
    "Word",
    "DiarSegment",
    "DiarizedWord",
    "Video",
    "Speaker",
    "Clip",
    "normalize_text",
]


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Canonical text normalizer used by ``words.normalized_text`` and FTS5.

    Steps (in order):

    1. ``str.lower()`` — case-fold.
    2. ``str.strip()`` — drop leading/trailing whitespace.
    3. Drop punctuation (anything outside ``\\w`` or ``\\s`` under
       ``re.UNICODE``).
    4. Collapse runs of whitespace into a single ASCII space.

    Unicode is preserved through every step (no ``encode`` / ``decode``
    round-trip). Empty input yields empty output.
    """
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Video:
    """One row of the ``videos`` table."""

    id: int
    source: str  # "youtube" | "ytdlp" | "local"
    kind: str  # "video" | "short" | "livestream" | "other"
    channel_id: int | None
    youtube_id: str | None
    url: str | None
    local_path: str | None
    title: str
    duration: int | None
    published_at: str | None
    downloaded: bool
    downloaded_path: str | None
    downloaded_audio_path: str | None
    metadata_json: str = "{}"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Video:
        """Construct a Video from a sqlite3.Row from the ``videos`` table."""
        # Handle missing downloaded_audio_path column for backward compatibility
        downloaded_audio_path = row["downloaded_audio_path"] if "downloaded_audio_path" in row.keys() else None
        return cls(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            channel_id=row["channel_id"],
            youtube_id=row["youtube_id"],
            url=row["url"],
            local_path=row["local_path"],
            title=row["title"],
            duration=row["duration"],
            published_at=row["published_at"],
            downloaded=bool(row["downloaded"]),
            downloaded_path=row["downloaded_path"],
            downloaded_audio_path=downloaded_audio_path,
            metadata_json=row["metadata_json"],
        )


@dataclass(frozen=True)
class Speaker:
    """One row of the global ``speakers`` roster."""

    id: int
    label: str
    aliases: list[str] = field(default_factory=list)
    notes: str | None = None
    created_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Speaker:
        """Construct a Speaker from a sqlite3.Row from the ``speakers`` table.

        ``aliases_json`` is decoded from JSON into ``list[str]``.
        """
        raw = row["aliases_json"]
        try:
            aliases = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            aliases = []
        if not isinstance(aliases, list):
            aliases = []
        return cls(
            id=row["id"],
            label=row["label"],
            aliases=[str(a) for a in aliases],
            notes=row["notes"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Clip:
    """One row of the ``clips`` table — output of the mine stage."""

    id: int
    video_id: int
    start_ms: int
    end_ms: int
    source_query: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Clip:
        return cls(
            id=row["id"],
            video_id=row["video_id"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            source_query=row["source_query"],
            created_at=row["created_at"],
        )