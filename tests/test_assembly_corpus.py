"""The corpus fixture builder. Word rows, no audio — design §8 is pure logic."""

from __future__ import annotations

import pytest

from rytp.db import Database
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_acoustics,
    add_speaker,
    add_video,
    add_video_speaker,
    add_words,
    word_rows,
)


def test_add_words_lays_out_ords_and_timings(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все понимаем")
    rows = word_rows(db, video_id)
    assert [row["ord"] for row in rows] == [0, 1, 2]
    assert [row["normalized_text"] for row in rows] == ["мы", "все", "понимаем"]
    assert rows[0]["start_ms"] == 0
    assert rows[0]["end_ms"] == WORD_MS
    assert rows[1]["start_ms"] == WORD_MS + GAP_MS
    assert rows[2]["end_ms"] == 3 * WORD_MS + 2 * GAP_MS


def test_add_words_normalizes_and_stems(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "Ещё Раз")
    rows = word_rows(db, video_id)
    assert [row["text"] for row in rows] == ["Ещё", "Раз"]
    assert [row["normalized_text"] for row in rows] == ["еще", "раз"]
    assert all(row["stem"] for row in rows)


def test_a_second_call_continues_the_ordinals(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    next_ord = add_words(db, video_id, "мы все")
    assert next_ord == 2
    add_words(db, video_id, "понимаем", start_ms=10_000)
    rows = word_rows(db, video_id)
    assert [row["ord"] for row in rows] == [0, 1, 2]
    assert rows[2]["start_ms"] == 10_000


def test_caption_words_have_no_end_and_the_check_constraint_holds(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", source="caption")
    rows = word_rows(db, video_id)
    assert [row["source"] for row in rows] == ["caption", "caption"]
    assert rows[0]["end_ms"] is None
    assert rows[0]["align_score"] is None


def test_timed_words_have_ends_but_no_alignment_score(db: Database) -> None:
    """contracts §3: timed rows are good text and unusable boundaries."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", source="timed")
    rows = word_rows(db, video_id)
    assert [row["source"] for row in rows] == ["timed", "timed"]
    assert rows[0]["end_ms"] == WORD_MS
    assert rows[0]["align_score"] is None


def test_video_speakers_rows_carry_the_engine_that_made_them(db: Database) -> None:
    """contracts §3: engine is NOT NULL and this fixture must supply it."""
    video_id = add_video(db, external_id="VIDEO_A")
    local_id = add_video_speaker(db, video_id, "SPEAKER_00", engine="fake-diarizer")
    row = db.conn.execute(
        "SELECT engine FROM video_speakers WHERE id = ?", (local_id,)
    ).fetchone()
    assert row["engine"] == "fake-diarizer"


def test_words_can_carry_a_speaker(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    speaker_id = add_speaker(db, "host")
    local_id = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=speaker_id)
    add_words(db, video_id, "мы все", video_speaker_id=local_id)
    assert {row["video_speaker_id"] for row in word_rows(db, video_id)} == {local_id}


def test_align_score_may_be_null(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", align_score=None)
    assert word_rows(db, video_id)[0]["align_score"] is None


def test_add_acoustics_defaults_every_field_and_takes_overrides(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_acoustics(db, video_id, f0_mean=180.0)
    row = db.conn.execute(
        "SELECT * FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert row["f0_mean"] == 180.0
    assert row["loudness_lufs"] is not None


def test_add_acoustics_rejects_a_field_the_table_does_not_have(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    with pytest.raises(KeyError, match="brightness"):
        add_acoustics(db, video_id, brightness=1.0)


def test_two_videos_get_distinct_ids(db: Database) -> None:
    first = add_video(db, external_id="VIDEO_A")
    second = add_video(db, external_id="VIDEO_B")
    assert first != second
