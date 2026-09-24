"""Contracts §3, as executable assertions."""

from __future__ import annotations

import sqlite3

import pytest

from rytp.db import Database

NOW = "2026-01-01T00:00:00+00:00"

CONTRACT_TABLES = {
    "channels",
    "videos",
    "assets",
    "speakers",
    "video_speakers",
    "words",
    "utterances",
    "utterances_fts",
    "video_acoustics",
    "jobs",
    "renders",
    "settings",
}

CONTRACT_INDEXES = {
    "videos_channel",
    "videos_source",
    "assets_video_role",
    "assets_one_audio",
    "assets_one_captions",
    "video_speakers_speaker",
    "words_video_ord",
    "words_normalized",
    "words_stem",
    "words_speaker",
    "words_alignable",
    "utterances_video",
    "jobs_claim",
    "renders_cutlist",
}


def names(db: Database, kind: str) -> set[str]:
    return {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))
        # FTS5 keeps its own shadow tables (utterances_fts_data, _idx, …).
        if not row["name"].startswith("sqlite_")
    }


def seed_video(db: Database) -> int:
    db.conn.execute(
        "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'One')"
    )
    cur = db.conn.execute(
        "INSERT INTO videos (source, kind, channel_id, external_id, url, title,"
        " duration_ms, created_at)"
        " VALUES ('ytdlp', 'video', 1, 'VIDEO_A', 'https://example.invalid/w/VIDEO_A',"
        " 'A', 1000, ?)",
        (NOW,),
    )
    return int(cur.lastrowid)


def test_every_contract_table_exists(db: Database) -> None:
    assert names(db, "table") >= CONTRACT_TABLES


def test_every_contract_index_exists(db: Database) -> None:
    assert names(db, "index") >= CONTRACT_INDEXES


def test_no_dropped_table_survives(db: Database) -> None:
    """Design §4 "Dropped": these must not come back."""
    dropped = {
        "clips",
        "clip_features",
        "splice_runs",
        "splice_clips",
        "chunks",
        "queue_items",
        "videos_speaker_map",
        "transcribe_runs",
        "words_fts",
    }
    assert dropped & names(db, "table") == set()


def test_caption_words_may_have_no_end_time(db: Database) -> None:
    """Captions carry per-word starts and no ends (design §6)."""
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, NULL, 'да', 'да', 'да', 'caption', 'captions')",
        (video_id,),
    )
    assert db.conn.execute("SELECT COUNT(*) FROM words").fetchone()[0] == 1


def test_aligned_words_must_have_an_end_time(db: Database) -> None:
    """Cuttable is defined as source='aligned'; a cut needs both boundaries."""
    video_id = seed_video(db)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
            " source, engine) VALUES (?, 0, 0, NULL, 'да', 'да', 'да', 'aligned', 'mfa')",
            (video_id,),
        )


def test_the_three_transcript_tiers_are_accepted_and_nothing_else(db: Database) -> None:
    """Contracts §3: caption, timed, aligned. `timed` is searchable, not cuttable."""
    video_id = seed_video(db)
    for ord_, source in enumerate(("caption", "timed", "aligned")):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine) VALUES (?, ?, 0, 100, 'да', 'да', 'да', ?, 'e')",
            (video_id, ord_, source),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine) VALUES (?, 9, 0, 100, 'да', 'да', 'да', 'guessed', 'e')",
            (video_id,),
        )


def test_the_alignable_index_covers_only_cuttable_words(db: Database) -> None:
    """The assembler reads cuttable words only; a full index would scale with captions."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words_alignable'"
    ).fetchone()[0]
    assert "WHERE source = 'aligned'" in sql


def test_the_default_aligner_setting_ships_empty(db: Database) -> None:
    """Contracts §3: empty means ingest enqueues no align job at all."""
    row = db.conn.execute(
        "SELECT value FROM settings WHERE key = 'default_aligner'"
    ).fetchone()
    assert row is not None
    assert row["value"] == ""


def test_word_ordinals_are_unique_per_video(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')",
        (video_id,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
            " source, engine) VALUES (?, 0, 100, 200, 'нет', 'нет', 'нет', 'aligned', 'mfa')",
            (video_id,),
        )


def test_the_fts_tokenizer_is_unicode61_not_porter(db: Database) -> None:
    """Porter is English-only; Russian stemming is done in Python (contracts §3)."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "unicode61" in sql
    assert "remove_diacritics 0" in sql
    assert "porter" not in sql


def test_fts_mirrors_inserts_updates_and_deletes(db: Database) -> None:
    video_id = seed_video(db)

    def matches(expr: str) -> list[int]:
        return [
            row["rowid"]
            for row in db.conn.execute(
                "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", (expr,)
            )
        ]

    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text)"
        " VALUES (?, 0, 500, 0, 1, 'Привет, мир!', 'привет мир', 'привет мир')",
        (video_id,),
    )
    assert matches('"привет мир"') == [1]

    db.conn.execute(
        "UPDATE utterances SET normalized_text = 'пока мир', stem_text = 'пок мир' WHERE id = 1"
    )
    assert matches('"привет мир"') == []
    assert matches('"пока мир"') == [1]

    db.conn.execute("DELETE FROM utterances WHERE id = 1")
    assert matches('"пока мир"') == []
    # An out-of-sync external-content index fails this; a clean one returns nothing.
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_fts_searches_the_two_columns_independently(db: Database) -> None:
    """Design §7: exact column first, stem column as the fallback."""
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text)"
        " VALUES (?, 0, 500, 0, 1, 'Сказали слово', 'сказали слово', 'сказа слов')",
        (video_id,),
    )
    exact = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сказали слово"',),
    ).fetchall()
    stemmed = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('stem_text : "сказа слов"',),
    ).fetchall()
    missing = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сказа слов"',),
    ).fetchall()
    assert len(exact) == 1
    assert len(stemmed) == 1
    assert missing == []


def test_deleting_a_video_cascades_to_its_rows(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text) VALUES (?, 0, 100, 0, 0, 'да', 'да', 'да')",
        (video_id,),
    )
    db.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    assert db.conn.execute("SELECT COUNT(*) FROM words").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0


def test_a_video_has_at_most_one_audio_and_one_captions_asset(db: Database) -> None:
    """Design §4: one canonical audio, one captions, any number of renditions."""
    video_id = seed_video(db)

    def add_asset(role: str, path: str) -> None:
        db.conn.execute(
            "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
            (video_id, role, path, NOW),
        )

    add_asset("audio", "media/1/audio.m4a")
    add_asset("captions", "media/1/captions.json3")
    add_asset("video", "media/1/video-360.mp4")
    add_asset("video", "media/1/video-1080.mp4")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        add_asset("audio", "media/1/audio-better.opus")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        add_asset("captions", "media/1/captions-2.json3")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM assets WHERE role = 'video'"
    ).fetchone()[0] == 2


def test_a_render_records_its_cut_list_and_state(db: Database) -> None:
    """Part 1 creates this table; part 6 is the only thing that writes to it."""
    db.conn.execute(
        "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at)"
        " VALUES ('monologue', 'pillarbox', 'planned', ?)",
        (NOW,),
    )
    row = db.conn.execute("SELECT * FROM renders").fetchone()
    assert row["cutlist_name"] == "monologue"
    assert row["output_path"] is None
    assert row["finished_at"] is None
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at)"
            " VALUES ('monologue', 'pillarbox', 'halfway', ?)",
            (NOW,),
        )


def test_a_job_is_unique_per_kind_and_target(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('download', 1, 'pending', 'network', ?)",
        (NOW,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 1, 'pending', 'network', ?)",
            (NOW,),
        )


def test_a_job_can_carry_a_non_fatal_note(db: Database) -> None:
    """Contracts §3: a handler's finding, recorded by the worker. `done`, not `failed`.

    Part 1 only creates the column; part 2's worker writes it and shows it
    in `jobs.list`. Without it a warning like "re-transcribing discarded
    this video's speaker mapping" reaches someone running the command by
    hand and nobody running the worker, which is the bulk path.
    """
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, note, created_at)"
        " VALUES ('transcribe', 1, 'done', 'gpu', 'discarded 2 speaker labels', ?)",
        (NOW,),
    )
    row = db.conn.execute("SELECT state, note FROM jobs").fetchone()
    assert row["state"] == "done"
    assert row["note"] == "discarded 2 speaker labels"


def test_a_job_note_is_optional(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('download', 1, 'pending', 'network', ?)",
        (NOW,),
    )
    assert db.conn.execute("SELECT note FROM jobs").fetchone()["note"] is None


def test_job_state_and_pool_are_constrained(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 2, 'sleeping', 'network', ?)",
            (NOW,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 3, 'pending', 'quantum', ?)",
            (NOW,),
        )


def test_a_speaker_label_is_unique_and_a_video_label_is_unique_per_video(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
        " VALUES (?, 'S0', 1, 'pyannote')",
        (video_id,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine)"
            " VALUES (?, 'S0', 'pyannote')",
            (video_id,),
        )


def test_unmapping_a_speaker_keeps_the_video_label(db: Database) -> None:
    """Design §4: linking a label to a person updates one row, not thousands."""
    video_id = seed_video(db)
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
        " VALUES (?, 'S0', 1, 'pyannote')",
        (video_id,),
    )
    db.conn.execute("DELETE FROM speakers WHERE id = 1")
    row = db.conn.execute("SELECT local_label, speaker_id FROM video_speakers").fetchone()
    assert row["local_label"] == "S0"
    assert row["speaker_id"] is None
