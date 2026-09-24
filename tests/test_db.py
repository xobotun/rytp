"""The connection setup and the migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rytp.config import ensure_dir, paths
from rytp.db import Database, schema


def test_migration_versions_are_contiguous_and_ascending() -> None:
    """Guards the append-only rule: a duplicate or edited tuple shows up here."""
    versions = [version for version, _sql in schema.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert versions[-1] == schema.LATEST_VERSION


def test_migrate_reports_the_version_it_reached(data_dir: Path) -> None:
    database = Database(paths().db)
    try:
        assert database.schema_version() == 0
        assert database.migrate() == schema.LATEST_VERSION
        assert database.schema_version() == schema.LATEST_VERSION
    finally:
        database.close()


def test_migrate_is_idempotent(db: Database) -> None:
    assert db.migrate() == schema.LATEST_VERSION
    assert db.migrate() == schema.LATEST_VERSION


def test_channels_and_videos_exist_after_migrating(db: Database) -> None:
    names = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"schema_version", "channels", "videos"} <= names


def test_connection_pragmas(db: Database) -> None:
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_rows_are_indexable_by_column_name(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO channels (url, title) VALUES (?, ?)",
        ("https://example.invalid/c/CHANNEL_ONE", "Channel One"),
    )
    row = db.conn.execute("SELECT id, title FROM channels").fetchone()
    assert row["title"] == "Channel One"


def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO videos (source, kind, channel_id, external_id, title, created_at)"
            " VALUES ('ytdlp', 'video', 999, 'VIDEO_A', 'A', '2026-01-01T00:00:00+00:00')"
        )


def test_transaction_commits_on_success(db: Database) -> None:
    with db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/two', 'two')"
        )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 2


def test_transaction_rolls_back_on_error(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'dup')"
        )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_nested_transactions_join_the_outer_one(db: Database) -> None:
    """SQLite has no nested transactions; the inner block must not BEGIN again."""
    with db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        with db.transaction():
            db.conn.execute(
                "INSERT INTO channels (url, title)"
                " VALUES ('https://example.invalid/c/two', 'two')"
            )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 2


def test_an_inner_failure_rolls_back_the_whole_outer_transaction(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        with db.transaction():
            db.conn.execute(
                "INSERT INTO channels (url, title)"
                " VALUES ('https://example.invalid/c/one', 'dup')"
            )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_a_partly_migrated_database_is_brought_forward(data_dir: Path) -> None:
    """Simulates an older install: stop at version 1, reopen, finish."""
    ensure_dir(paths().root)
    first = Database(paths().db)
    try:
        first.migrate_to(1)
        assert first.schema_version() == 1
    finally:
        first.close()

    second = Database(paths().db)
    try:
        assert second.migrate() == schema.LATEST_VERSION
    finally:
        second.close()


def test_database_is_a_context_manager(data_dir: Path) -> None:
    ensure_dir(paths().root)
    with Database(paths().db) as database:
        database.migrate()
    with pytest.raises(sqlite3.ProgrammingError):
        database.conn.execute("SELECT 1")
