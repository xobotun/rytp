"""Tests for the mine stage."""
from __future__ import annotations

import sqlite3

import pytest

from rytp.db import Database
from rytp.mine import (
    Cohesion,
    Window,
    expand_into_windows,
    mine,
    score_windows,
    search_words,
)
from rytp.models import Word


def _has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        opts = conn.execute("PRAGMA compile_options").fetchall()
    except sqlite3.DatabaseError:
        return False
    for row in opts:
        val = row[0] if isinstance(row[0], str) else row[1]
        if "FTS5" in val.upper():
            return True
    return False


requires_fts5 = pytest.mark.skipif(
    not _has_fts5(sqlite3.connect(":memory:")),
    reason="SQLite build lacks FTS5",
)


def _populate(db: Database, fake_video_row, words: list[tuple[int, int, str]]) -> int:
    vid = db.upsert_video(fake_video_row)
    db.insert_words(
        vid,
        [
            Word(start_ms=s, end_ms=e, text=t, confidence=1.0)
            for s, e, t in words
        ],
    )
    return vid


# --- search_words --------------------------------------------------------


@requires_fts5
def test_search_words_finds_substring(db: Database, fake_video_row) -> None:
    _populate(
        db,
        fake_video_row,
        [
            (0, 200, "hello"),
            (300, 500, "world"),
            (600, 800, "goodbye world"),
            (900, 1100, "goodbye"),
        ],
    )
    hits = search_words(db, "world")
    assert len(hits) == 2
    texts = sorted(h[2] for h in hits)
    assert texts == ["goodbye world", "world"]


@requires_fts5
def test_search_words_normalizes_query(db: Database, fake_video_row) -> None:
    _populate(db, fake_video_row, [(0, 200, "Hello")])
    hits = search_words(db, "HELLO!  ")
    assert len(hits) == 1


@requires_fts5
def test_search_words_empty_query_returns_empty(db: Database) -> None:
    assert search_words(db, "") == []
    assert search_words(db, "!!!") == []


# --- expand_into_windows -------------------------------------------------


@requires_fts5
def test_expand_windows_low(db: Database, fake_video_row) -> None:
    vid = _populate(
        db,
        fake_video_row,
        [(0, 200, "hello"), (300, 500, "world"), (1000, 1200, "again")],
    )
    hits = search_words(db, "world")
    windows = expand_into_windows(db, hits, cohesion=Cohesion.LOW)
    assert len(windows) == 1
    assert windows[0].text == "world"


@requires_fts5
def test_expand_windows_med_extends_within_gap_budget(
    db: Database, fake_video_row
) -> None:
    # Two hits 300ms apart → should be joined into one window.
    vid = _populate(
        db,
        fake_video_row,
        [
            (0, 200, "hello"),
            (500, 700, "world"),
            (5000, 5200, "again"),  # > 1s gap → must not join
        ],
    )
    hits = search_words(db, "hello")
    windows = expand_into_windows(db, hits, cohesion=Cohesion.MED)
    assert len(windows) == 1
    assert "hello" in windows[0].text
    assert "world" in windows[0].text
    assert "again" not in windows[0].text


@requires_fts5
def test_expand_windows_high_strict_gap(db: Database, fake_video_row) -> None:
    # At HIGH cohesion the gap budget is 500ms — 300ms apart → join,
    # 600ms apart → split.
    vid = _populate(
        db,
        fake_video_row,
        [
            (0, 200, "hello"),
            (500, 700, "world"),  # gap = 300ms → join
            (1300, 1500, "again"),  # gap = 600ms → split
        ],
    )
    hits = search_words(db, "hello")
    windows = expand_into_windows(db, hits, cohesion=Cohesion.HIGH)
    assert len(windows) == 1
    assert "hello" in windows[0].text and "world" in windows[0].text


@requires_fts5
def test_expand_windows_handles_multi_video(db: Database, fake_video_row) -> None:
    v1 = _populate(db, fake_video_row, [(0, 200, "hello")])
    v2 = _populate(
        db,
        {**fake_video_row, "youtube_id": "v2"},
        [(0, 200, "hello")],
    )
    hits = search_words(db, "hello")
    windows = expand_into_windows(db, hits, cohesion=Cohesion.LOW)
    assert {w.video_id for w in windows} == {v1, v2}


# --- score_windows -------------------------------------------------------


@requires_fts5
def test_score_windows_returns_sorted_by_score(db: Database, fake_video_row) -> None:
    _populate(
        db,
        fake_video_row,
        [
            (0, 100, "a"),
            (200, 250, "b"),
            (400, 450, "c"),
        ],
    )
    windows = [
        Window(video_id=1, start_ms=0, end_ms=100, text="a"),
        Window(video_id=1, start_ms=200, end_ms=250, text="b"),
    ]
    scored = score_windows(db, windows)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in scored)
    # Scores are non-increasing
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


@requires_fts5
def test_score_windows_empty_returns_empty(db: Database) -> None:
    assert score_windows(db, []) == []


# --- mine (end-to-end) ---------------------------------------------------


@requires_fts5
def test_mine_writes_clips_and_returns_ids(
    db: Database, fake_video_row
) -> None:
    _populate(
        db,
        fake_video_row,
        [
            (0, 200, "hello"),
            (300, 500, "world"),
        ],
    )
    ids = mine(db, "world", cohesion="low")
    assert len(ids) == 1
    row = db.conn.execute("SELECT * FROM clips WHERE id = ?", (ids[0],)).fetchone()
    assert row["source_query"] == "world"


@requires_fts5
def test_mine_respects_max_clips(db: Database, fake_video_row) -> None:
    vid = db.upsert_video(fake_video_row)
    # Insert 10 distinct words that each contain "word"; limit to 3.
    for i in range(10):
        word = f"wordish{i}"
        db.conn.execute(
            "INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (vid, i * 100, i * 100 + 50, word, word, 1.0),
        )
    db.conn.commit()
    ids = mine(db, "wordish", cohesion="low", max_clips=3)
    assert len(ids) == 3


@requires_fts5
def test_mine_no_hits_returns_empty(db: Database) -> None:
    assert mine(db, "nothing") == []