"""SQLite database wrapper, migrations, and FTS5 shadow table for rytp.

The schema mirrors DESIGN §4 exactly. Migrations are stored as a list
of ``(version, sql)`` tuples and applied in order. The bookkeeping
table ``schema_version`` is bootstrapped separately from the
migrations themselves, so the runner doesn't have a chicken-and-egg
between its own state and the tables it's creating.

Why 16 migrations and not one big schema? Two reasons:

1. **Audit trail.** Each migration is a small, reviewable diff.
   Reviewing a 500-line ``CREATE TABLE`` block is harder than
   reviewing 16 small ones.
2. **Future flexibility.** A future v2 can add a migration version
   17 that adds a column to ``speakers`` without disturbing the v1
   schema. With one big migration, that's a single 600-line block.

The schema_version bootstrap is deliberately outside the migration
list because the runner needs that table to exist before it can
record its own progress. Putting the bootstrap inside a migration
would mean the first migration has to ``CREATE TABLE`` something
that doesn't exist yet — a chicken-and-egg.

Public surface:

* :class:`Database` — connection wrapper with migrations + helpers.
* :func:`normalize_text` — re-exported from :mod:`rytp.models` for
  ergonomic imports.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from rytp.models import Word, normalize_text
from rytp import constants as C


# Migration list: (version, sql).
# Version 1 creates the channels table; subsequent versions add the
# remaining tables (DESIGN §4) and finally the FTS5 shadow +
# triggers. Splitting into 16 small migrations makes each diff
# reviewable and lets v2+ add migrations without rewriting v1.
# The bootstrap ``schema_version`` table is created separately in
# :meth:`Database.migrate` so we don't have a chicken-and-egg between
# the migration runner and its own bookkeeping.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            last_synced_at TEXT
        );
        """,
    ),
    (
        2,
        """
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL CHECK (source IN ('youtube','ytdlp','local')),
            kind TEXT NOT NULL CHECK (kind IN ('video','short','livestream','other')),
            channel_id INTEGER NULL REFERENCES channels(id),
            youtube_id TEXT NULL,
            url TEXT NULL,
            local_path TEXT NULL,
            title TEXT NOT NULL,
            duration INTEGER NULL,
            published_at TEXT NULL,
            downloaded INTEGER NOT NULL DEFAULT 0,
            downloaded_path TEXT NULL,
            downloaded_audio_path TEXT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE (source, youtube_id),
            UNIQUE (source, local_path)
        );
        """,
    ),
    (
        3,
        """
        CREATE TABLE queue_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            status TEXT NOT NULL CHECK (status IN ('pending','running','done','failed','paused')),
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            enqueued_at TEXT NOT NULL,
            started_at TEXT NULL,
            finished_at TEXT NULL
        );
        """,
    ),
    (
        4,
        """
        CREATE TABLE words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            confidence REAL NOT NULL,
            diarizer_speaker TEXT NULL,
            speaker_id INTEGER NULL REFERENCES speakers(id)
        );
        """,
    ),
    (
        5,
        """
        CREATE TABLE videos_speaker_map (
            video_id INTEGER NOT NULL REFERENCES videos(id),
            diarizer_speaker TEXT NOT NULL,
            speaker_id INTEGER NOT NULL REFERENCES speakers(id),
            PRIMARY KEY (video_id, diarizer_speaker)
        );
        """,
    ),
    (
        6,
        """
        CREATE TABLE transcribe_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            stt_engine TEXT NOT NULL,
            diarizer TEXT NOT NULL,
            combined TEXT NULL,
            language TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NULL
        );
        """,
    ),
    (
        7,
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcribe_run_id INTEGER NOT NULL REFERENCES transcribe_runs(id),
            ord INTEGER NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('pending','done','failed')),
            words_written INTEGER NOT NULL DEFAULT 0
        );
        """,
    ),
    (
        8,
        """
        CREATE TABLE speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT NULL,
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        9,
        """
        CREATE TABLE speaker_pause_stats (
            speaker_id INTEGER PRIMARY KEY REFERENCES speakers(id),
            n_samples INTEGER NOT NULL,
            mean_ms REAL NOT NULL,
            std_ms REAL NOT NULL,
            p10_ms REAL NOT NULL,
            p25_ms REAL NOT NULL,
            p50_ms REAL NOT NULL,
            p75_ms REAL NOT NULL,
            p90_ms REAL NOT NULL,
            min_ms REAL NOT NULL,
            max_ms REAL NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        10,
        """
        CREATE TABLE clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            source_query TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        11,
        """
        CREATE TABLE splice_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            output_path TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        12,
        """
        CREATE TABLE splice_clips (
            splice_run_id INTEGER NOT NULL REFERENCES splice_runs(id),
            ord INTEGER NOT NULL,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            in_ms INTEGER NOT NULL,
            out_ms INTEGER NOT NULL,
            text TEXT NOT NULL,
            adjusted_in_ms INTEGER NULL,
            adjusted_out_ms INTEGER NULL,
            inserted_pause_ms INTEGER NULL,
            video_strategy TEXT NOT NULL,
            PRIMARY KEY (splice_run_id, ord)
        );
        """,
    ),
    (
        13,
        """
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    ),
    (
        14,
        """
        CREATE TABLE clip_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            mfcc_blob BLOB,
            spectral_centroid_hz REAL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        15,
        """
        CREATE INDEX words_video_start ON words(video_id, start_ms);
        CREATE INDEX words_speaker ON words(speaker_id);
        CREATE INDEX queue_status ON queue_items(status);
        CREATE INDEX videos_source ON videos(source);
        CREATE INDEX videos_channel ON videos(channel_id);
        """,
    ),
    (
        16,
        """
        CREATE VIRTUAL TABLE words_fts USING fts5(
            normalized_text,
            content='words',
            content_rowid='id',
            tokenize='porter unicode61'
        );
        CREATE TRIGGER words_ai AFTER INSERT ON words BEGIN
            INSERT INTO words_fts(rowid, normalized_text) VALUES (new.id, new.normalized_text);
        END;
        CREATE TRIGGER words_ad AFTER DELETE ON words BEGIN
            INSERT INTO words_fts(words_fts, rowid, normalized_text) VALUES('delete', old.id, old.normalized_text);
        END;
        CREATE TRIGGER words_au AFTER UPDATE ON words BEGIN
            INSERT INTO words_fts(words_fts, rowid, normalized_text) VALUES('delete', old.id, old.normalized_text);
            INSERT INTO words_fts(rowid, normalized_text) VALUES (new.id, new.normalized_text);
        END;
        """,
    ),
    (
        17,
        """
        -- Add downloaded_audio_path column to videos table for separate audio/video files
        -- Note: for fresh databases, migration 2 already includes this column, so this is a no-op
        -- For existing databases, this adds the column
        -- SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN, so we handle this
        -- in the migrate() method by checking if the column exists first
        SELECT 1;  -- placeholder
        """,
    ),
]


class Database:
    """SQLite connection holder with migration support and helper methods.

    Wraps a single ``sqlite3.Connection`` for the lifetime of the
    :class:`Database` instance. Foreign keys are enforced
    (``PRAGMA foreign_keys = ON``), the row factory is set to
    :class:`sqlite3.Row` so callers can use ``row["column"]`` syntax.

    ``check_same_thread=False`` allows the connection to cross thread
    boundaries — useful for the queue worker (DESIGN §3 "long-running")
    but the caller is responsible for serialization when multiple
    threads share one :class:`Database`.

    Attributes:
        conn: The underlying ``sqlite3.Connection``.
    """

    def __init__(self, path: Path) -> None:
        """Open a SQLite connection at ``path``.

        The connection is opened eagerly so any "permission denied"
        or "unable to open" errors surface immediately rather than
        on the first query.
        """
        self._path = path
        self.conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def migrate(self) -> None:
        """Run all pending migrations to bring the schema up to date.

        Idempotent — calling twice is a no-op. The bookkeeping table
        ``schema_version`` is created here (not as a migration) so
        there's no chicken-and-egg between the runner and its own
        version row.

        Migration runner algorithm:

        1. ``CREATE TABLE IF NOT EXISTS schema_version`` — the
           bookkeeping table.
        2. Read the current version (default ``0`` if the row is
           absent).
        3. For each migration with ``version > current_version``,
           run its SQL and bump the recorded version.

        Each migration is wrapped in its own implicit transaction
        (sqlite's default for ``executescript``), so a partial
        migration doesn't corrupt the schema.
        """
        # Bootstrap the bookkeeping table.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = self.conn.execute(
            "SELECT value FROM schema_version WHERE key = 'version'"
        ).fetchone()
        current_version = int(row["value"]) if row else 0
        if row is None:
            self.conn.execute(
                "INSERT INTO schema_version (key, value) VALUES ('version', '0')"
            )

        # Apply any migrations with version > current_version.
        for version, sql in MIGRATIONS:
            if version > current_version:
                # Special handling for migration 17 (add downloaded_audio_path column)
                # SQLite doesn't support IF NOT EXISTS for ALTER TABLE ADD COLUMN
                if version == 17:
                    # Check if column already exists (it will for fresh databases from migration 2)
                    col_check = self.conn.execute(
                        "PRAGMA table_info(videos)"
                    ).fetchall()
                    has_column = any(row[1] == "downloaded_audio_path" for row in col_check)
                    if not has_column:
                        self.conn.execute(
                            "ALTER TABLE videos ADD COLUMN downloaded_audio_path TEXT NULL"
                        )
                else:
                    self.conn.executescript(sql)
                self.conn.execute(
                    "UPDATE schema_version SET value = ? WHERE key = 'version'",
                    (str(version),),
                )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying :class:`sqlite3.Connection`."""
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterable[None]:
        """Context manager for a ``BEGIN`` / ``COMMIT`` transaction.

        Safe to call after bare DML statements: any implicit
        transaction opened by the prior statement is committed first
        so ``BEGIN`` doesn't fail with ``cannot start a transaction
        within a transaction`` (the implicit-transaction dance that
        Python's :mod:`sqlite3` does on every DML statement).

        Usage::

            with db.transaction():
                db.conn.execute("INSERT INTO ...", (...))
                db.conn.execute("UPDATE ...", (...))

        On exception the transaction is rolled back and the exception
        is re-raised.
        """
        # Close any implicit transaction left by prior DML so we can
        # BEGIN cleanly.
        self.conn.commit()
        self.conn.execute("BEGIN")
        try:
            yield
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # Helper methods
    def upsert_video(self, row: dict[str, Any]) -> int:
        """Insert or update a video row by natural key.

        Two natural keys are supported, tried in order:

        1. ``(source, youtube_id)`` — for yt-dlp-sourced videos.
        2. ``(source, local_path)`` — for local file registrations.

        If either matches, the mutable fields are updated (kind,
        channel_id, url, title, duration, published_at, downloaded,
        downloaded_path, metadata_json) and the existing ``id`` is
        returned. If neither matches, a fresh row is inserted.

        Args:
            row: A dict with the same keys as the ``videos`` table
                columns. Required: ``source``, ``kind``, ``title``.
                Recommended defaults: ``channel_id=None``,
                ``youtube_id=None``, ``local_path=None``,
                ``url=None``, ``duration=None``, ``published_at=None``,
                ``downloaded=False``, ``downloaded_path=None``,
                ``metadata_json='{}'``.

        Returns:
            The row id of the inserted or updated row.
        """
        source = row["source"]
        youtube_id = row.get("youtube_id")
        local_path = row.get("local_path")

        if youtube_id is not None:
            # Try to find existing by (source, youtube_id)
            existing = self.conn.execute(
                "SELECT id FROM videos WHERE source = ? AND youtube_id = ?",
                (source, youtube_id),
            ).fetchone()
            if existing:
                video_id = existing["id"]
                # Update mutable fields
                self.conn.execute(
                    """
                    UPDATE videos SET
                        kind = ?, channel_id = ?, url = ?, local_path = ?,
                        title = ?, duration = ?, published_at = ?,
                        downloaded = ?, downloaded_path = ?, downloaded_audio_path = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        row["kind"],
                        row.get("channel_id"),
                        row.get("url"),
                        row.get("local_path"),
                        row["title"],
                        row.get("duration"),
                        row.get("published_at"),
                        1 if row.get("downloaded") else 0,
                        row.get("downloaded_path"),
                        row.get("downloaded_audio_path"),
                        row.get("metadata_json", "{}"),
                        video_id,
                    ),
                )
                self.conn.commit()
                return video_id

        if local_path is not None:
            existing = self.conn.execute(
                "SELECT id FROM videos WHERE source = ? AND local_path = ?",
                (source, local_path),
            ).fetchone()
            if existing:
                video_id = existing["id"]
                self.conn.execute(
                    """
                    UPDATE videos SET
                        kind = ?, channel_id = ?, youtube_id = ?, url = ?,
                        title = ?, duration = ?, published_at = ?,
                        downloaded = ?, downloaded_path = ?, downloaded_audio_path = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        row["kind"],
                        row.get("channel_id"),
                        row.get("youtube_id"),
                        row.get("url"),
                        row["title"],
                        row.get("duration"),
                        row.get("published_at"),
                        1 if row.get("downloaded") else 0,
                        row.get("downloaded_path"),
                        row.get("downloaded_audio_path"),
                        row.get("metadata_json", "{}"),
                        video_id,
                    ),
                )
                self.conn.commit()
                return video_id

        # Insert new
        cursor = self.conn.execute(
            """
            INSERT INTO videos (
                source, kind, channel_id, youtube_id, url, local_path,
                title, duration, published_at, downloaded, downloaded_path, downloaded_audio_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["source"],
                row["kind"],
                row.get("channel_id"),
                row.get("youtube_id"),
                row.get("url"),
                row.get("local_path"),
                row["title"],
                row.get("duration"),
                row.get("published_at"),
                1 if row.get("downloaded") else 0,
                row.get("downloaded_path"),
                row.get("downloaded_audio_path"),
                row.get("metadata_json", "{}"),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_words(
        self, video_id: int, rows: Iterable[Word], *, diarizer_speaker: str | None = None
    ) -> int:
        """Bulk-insert ``words`` rows for a single video.

        ``rows`` are normalized with :func:`rytp.models.normalize_text`
        and the FTS5 shadow is kept in sync automatically by the
        ``words_ai``, ``words_ad``, ``words_au`` triggers (DESIGN §4).
        Returns the number of rows written.

        Args:
            video_id: FK to the ``videos`` table.
            rows: Iterable of :class:`rytp.models.Word` records.
            diarizer_speaker: Optional raw diarizer label applied to
                every row in this batch (the combined-engine path
                passes it per-row; the STT+Diarize merger path also
                uses this when the merger has resolved the label).

        Returns:
            Number of rows written (also committed).
        """
        data = [
            (
                video_id,
                w.start_ms,
                w.end_ms,
                w.text,
                normalize_text(w.text),
                w.confidence,
                diarizer_speaker,
                None,  # speaker_id — populated later by the speakers stage
            )
            for w in rows
        ]
        self.conn.executemany(
            """
            INSERT INTO words (
                video_id, start_ms, end_ms, text, normalized_text,
                confidence, diarizer_speaker, speaker_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        self.conn.commit()
        return len(data)

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Get a setting value by key.

        Args:
            key: Setting name (e.g. ``"queue_paused"``).
            default: Returned when the key is absent.

        Returns:
            The stored string value, or ``default`` if not present.
        """
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value by key."""
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def search_fts(self, query: str, limit: int = C.DEFAULT_SEARCH_LIMIT) -> list[sqlite3.Row]:
        """Search the FTS5 index for matching words.

        Returns ``sqlite3.Row`` objects whose ``rowid`` column
        corresponds to the ``words.id`` primary key. Use
        :func:`fetch_word_by_id` or join back to ``words`` to
        retrieve the full row.

        Args:
            query: FTS5 MATCH expression (caller is responsible for
                any tokenization/prefix rewriting).
            limit: Maximum rows returned. Defaults to
                :data:`rytp.constants.DEFAULT_SEARCH_LIMIT`.
        """
        rows = self.conn.execute(
            "SELECT rowid FROM words_fts WHERE words_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()
        return rows