"""Tests for ``rytp.db`` — schema, migrations, FTS5, helpers."""
from __future__ import annotations

import sqlite3

import pytest

from rytp.db import Database
from rytp.models import Word


def _all_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


def _all_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r["name"] for r in rows}


def _has_fts5(conn: sqlite3.Connection) -> bool:
    """Check whether the compiled sqlite supports FTS5."""
    try:
        opts = conn.execute("PRAGMA compile_options").fetchall()
    except sqlite3.DatabaseError:
        return False
    for row in opts:
        # compile_options rows are (option_name,) in most builds; some
        # builds surface (compile_option_id, option_name).
        val = row[0] if isinstance(row[0], str) else row[1]
        if "FTS5" in val.upper():
            return True
    return False


requires_fts5 = pytest.mark.skipif(
    not _has_fts5(sqlite3.connect(":memory:")),
    reason="SQLite build lacks FTS5 — skip FTS-specific tests",
)


# --- migrations ---------------------------------------------------------


def test_migrate_idempotent(db: Database) -> None:
    tables_a = _all_tables(db.conn)
    indexes_a = _all_indexes(db.conn)
    db.migrate()  # second call must be a no-op
    tables_b = _all_tables(db.conn)
    indexes_b = _all_indexes(db.conn)
    assert tables_a == tables_b
    assert indexes_a == indexes_b


def test_migrate_creates_all_tables(db: Database) -> None:
    tables = _all_tables(db.conn)
    expected = {
        "schema_version",
        "channels",
        "videos",
        "queue_items",
        "words",
        "videos_speaker_map",
        "transcribe_runs",
        "chunks",
        "speakers",
        "speaker_pause_stats",
        "clips",
        "splice_runs",
        "splice_clips",
        "settings",
        "clip_features",
    }
    missing = expected - tables
    assert not missing, f"missing tables: {missing}"


def test_migrate_creates_required_indexes(db: Database) -> None:
    indexes = _all_indexes(db.conn)
    required = {
        "words_video_start",
        "words_speaker",
        "queue_status",
        "videos_source",
        "videos_channel",
    }
    missing = required - indexes
    assert not missing, f"missing indexes: {missing}"


@requires_fts5
def test_migrate_creates_fts5_shadow_and_triggers(db: Database) -> None:
    tables = _all_tables(db.conn)
    assert "words_fts" in tables
    triggers = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    trigger_names = {r["name"] for r in triggers}
    assert {"words_ai", "words_ad", "words_au"} <= trigger_names


# --- Database lifecycle -------------------------------------------------


def test_database_context_manager(data_dir) -> None:
    from rytp.config import paths as data_paths
    with Database(data_paths.db) as database:
        # Connection is usable; on a fresh DB the only row is sqlite_sequence
        rows = database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert rows == []  # nothing yet
        database.migrate()
        rows = database.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert len(rows) > 0  # schema applied
    # Database is closed at this point — opening a new one against the
    # same file must succeed (the test is implicit: the context manager
    # didn't raise and yielded a working object).


def test_database_close_then_reopen_runs_no_migrations(tmp_path) -> None:
    p = tmp_path / "x.db"
    db = Database(p)
    db.migrate()
    version_before = db.conn.execute(
        "SELECT value FROM schema_version WHERE key='version'"
    ).fetchone()["value"]
    db.close()

    # Reopen: migrate() must not re-run anything (the migration list is
    # already exhausted).
    db2 = Database(p)
    db2.migrate()
    version_after = db2.conn.execute(
        "SELECT value FROM schema_version WHERE key='version'"
    ).fetchone()["value"]
    assert version_before == version_after
    db2.close()


# --- transaction -------------------------------------------------------


def test_transaction_commits_on_success(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO speakers (label, created_at) VALUES (?, ?)",
        ("Alice", "2024-01-01T00:00:00"),
    )
    with db.transaction():
        db.conn.execute(
            "INSERT INTO speakers (label, created_at) VALUES (?, ?)",
            ("Bob", "2024-01-01T00:00:00"),
        )
    rows = db.conn.execute("SELECT label FROM speakers ORDER BY label").fetchall()
    assert [r["label"] for r in rows] == ["Alice", "Bob"]


def test_transaction_rolls_back_on_exception(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO speakers (label, created_at) VALUES (?, ?)",
        ("Alice", "2024-01-01T00:00:00"),
    )
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.conn.execute(
                "INSERT INTO speakers (label, created_at) VALUES (?, ?)",
                ("Bob", "2024-01-01T00:00:00"),
            )
            raise RuntimeError("boom")
    rows = db.conn.execute("SELECT label FROM speakers").fetchall()
    assert [r["label"] for r in rows] == ["Alice"]


# --- helpers -----------------------------------------------------------


def test_upsert_video_round_trip(db: Database) -> None:
    row = {
        "source": "youtube",
        "kind": "video",
        "channel_id": None,
        "youtube_id": "vid1",
        "url": "https://example.com/v1",
        "local_path": None,
        "title": "First",
        "duration": 100,
        "published_at": None,
        "downloaded": False,
        "downloaded_path": None,
        "metadata_json": "{}",
    }
    vid = db.upsert_video(row)
    assert vid > 0

    # Mutable fields update on second call
    row["title"] = "First (renamed)"
    row["downloaded"] = True
    row["downloaded_path"] = "/tmp/v1.mp4"
    vid2 = db.upsert_video(row)
    assert vid2 == vid

    stored = db.conn.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
    assert stored["title"] == "First (renamed)"
    assert stored["downloaded"] == 1
    assert stored["downloaded_path"] == "/tmp/v1.mp4"
    assert stored["youtube_id"] == "vid1"  # immutable


def test_insert_words_writes_normalized_text(db: Database, fake_video_row) -> None:
    vid = db.upsert_video(fake_video_row)
    words = [
        Word(start_ms=0, end_ms=300, text="Hello, world!", confidence=0.95),
        Word(start_ms=400, end_ms=700, text="goodbye world", confidence=0.9),
    ]
    n = db.insert_words(vid, words)
    assert n == 2
    rows = db.conn.execute(
        "SELECT text, normalized_text FROM words WHERE video_id = ? ORDER BY start_ms",
        (vid,),
    ).fetchall()
    assert rows[0]["text"] == "Hello, world!"
    assert rows[0]["normalized_text"] == "hello world"
    assert rows[1]["normalized_text"] == "goodbye world"


def test_settings_round_trip(db: Database) -> None:
    assert db.get_setting("missing_key", "fallback") == "fallback"
    assert db.get_setting("missing_key") is None
    db.set_setting("k", "v")
    assert db.get_setting("k") == "v"
    db.set_setting("k", "v2")  # upsert
    assert db.get_setting("k") == "v2"


@requires_fts5
def test_search_fts_finds_substrings(db: Database, fake_video_row) -> None:
    vid = db.upsert_video(fake_video_row)
    words = [
        Word(start_ms=0, end_ms=300, text="hello world", confidence=1.0),
        Word(start_ms=400, end_ms=700, text="goodbye world", confidence=1.0),
        Word(start_ms=800, end_ms=1000, text="the quick fox", confidence=1.0),
    ]
    db.insert_words(vid, words)
    rows = db.search_fts("world")
    assert len(rows) == 2
    # rows have a rowid column corresponding to words.id
    rowids = {r["rowid"] for r in rows}
    assert len(rowids) == 2


def test_foreign_keys_enforced(db: Database) -> None:
    # No matching videos row → INSERT into words must fail.
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """
            INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (999_999, 0, 100, "x", "x", 1.0),
        )