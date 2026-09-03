"""Tests for ``rytp.models``."""
from __future__ import annotations

import sqlite3

import pytest

from rytp.models import Clip, Speaker, Video, normalize_text


def _row_conn():
    """A fresh in-memory connection with row_factory set to sqlite3.Row."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


# --- normalize_text ----------------------------------------------------


def test_normalize_lowercase_strip_punct() -> None:
    assert normalize_text("  Hello,  WORLD!  ") == "hello world"


def test_normalize_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text("!!!") == ""


def test_normalize_unicode_preserved() -> None:
    # Cyrillic must round-trip; whitespace and punctuation dropped.
    assert normalize_text("  Привет, Мир!  ") == "привет мир"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("foo\t\n  bar") == "foo bar"


def test_normalize_idempotent() -> None:
    once = normalize_text("Hello, World!")
    twice = normalize_text(once)
    assert once == twice


# --- dataclass frozen-ness ---------------------------------------------


def test_word_is_frozen() -> None:
    # Word is a NamedTuple from rytp.engines
    from rytp.models import Word
    w = Word(start_ms=0, end_ms=100, text="hi", confidence=0.9)
    with pytest.raises(AttributeError):
        w.start_ms = 1  # type: ignore[misc]


def test_diarsegment_is_frozen() -> None:
    from rytp.models import DiarSegment
    s = DiarSegment(start_ms=0, end_ms=100, speaker="SPEAKER_00")
    with pytest.raises(AttributeError):
        s.speaker = "X"  # type: ignore[misc]


def test_video_is_frozen() -> None:
    v = Video(
        id=1,
        source="youtube",
        kind="video",
        channel_id=None,
        youtube_id="abc",
        url=None,
        local_path=None,
        title="t",
        duration=None,
        published_at=None,
        downloaded=False,
        downloaded_path=None,
        downloaded_audio_path=None,
    )
    with pytest.raises(Exception):
        v.title = "other"  # type: ignore[misc]


def test_speaker_is_frozen() -> None:
    s = Speaker(id=1, label="X")
    with pytest.raises(Exception):
        s.label = "Y"  # type: ignore[misc]


def test_clip_is_frozen() -> None:
    c = Clip(
        id=1, video_id=1, start_ms=0, end_ms=100, source_query="q", created_at="2024"
    )
    with pytest.raises(Exception):
        c.source_query = "qq"  # type: ignore[misc]


# --- from_row ----------------------------------------------------------


# --- Video.from_row ---------------------------------------------------

def test_video_from_row() -> None:
    conn = _row_conn()
    conn.execute(
        """
        CREATE TABLE videos (
            id INTEGER, source TEXT, kind TEXT, channel_id INTEGER,
            youtube_id TEXT, url TEXT, local_path TEXT, title TEXT,
            duration INTEGER, published_at TEXT, downloaded INTEGER,
            downloaded_path TEXT, downloaded_audio_path TEXT, metadata_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO videos VALUES (1,'youtube','video',NULL,'yid','u',NULL,'t',100,NULL,1,'/p',NULL,'{}')"
    )
    row = conn.execute("SELECT * FROM videos").fetchone()
    v = Video.from_row(row)
    assert v.id == 1
    assert v.source == "youtube"
    assert v.downloaded is True
    assert v.metadata_json == "{}"


def test_speaker_from_row_parses_aliases_json() -> None:
    conn = _row_conn()
    conn.execute(
        "CREATE TABLE speakers (id INTEGER, label TEXT, aliases_json TEXT, notes TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO speakers VALUES (1, 'Paul', '[\"Paul\",\"Павел\"]', 'comedian', '2024-01-01')"
    )
    row = conn.execute("SELECT * FROM speakers").fetchone()
    s = Speaker.from_row(row)
    assert s.id == 1
    assert s.label == "Paul"
    assert s.aliases == ["Paul", "Павел"]
    assert s.notes == "comedian"
    assert s.created_at == "2024-01-01"


def test_speaker_from_row_handles_bad_aliases_json() -> None:
    conn = _row_conn()
    conn.execute(
        "CREATE TABLE speakers (id INTEGER, label TEXT, aliases_json TEXT, notes TEXT, created_at TEXT)"
    )
    conn.execute("INSERT INTO speakers VALUES (1, 'X', 'not json', NULL, '2024')")
    row = conn.execute("SELECT * FROM speakers").fetchone()
    s = Speaker.from_row(row)
    assert s.aliases == []


def test_clip_from_row() -> None:
    conn = _row_conn()
    conn.execute(
        "CREATE TABLE clips (id INTEGER, video_id INTEGER, start_ms INTEGER, end_ms INTEGER, source_query TEXT, created_at TEXT)"
    )
    conn.execute("INSERT INTO clips VALUES (7, 1, 100, 200, 'hello', '2024')")
    row = conn.execute("SELECT * FROM clips").fetchone()
    c = Clip.from_row(row)
    assert c.id == 7
    assert c.video_id == 1
    assert c.start_ms == 100
    assert c.end_ms == 200
    assert c.source_query == "hello"