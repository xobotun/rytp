"""Caption-tier ingest: starts only, never cuttable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe.captions import (
    CaptionDowngrade,
    ingest_captions,
    parse_json3,
)

SAMPLE = {
    "events": [
        {
            "tStartMs": 0,
            "dDurationMs": 1000,
            "segs": [
                {"utf8": "привет", "tOffsetMs": 0},
                {"utf8": " мир", "tOffsetMs": 320},
            ],
        },
        {"tStartMs": 1000, "dDurationMs": 40, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {
            "tStartMs": 1040,
            "segs": [
                {"utf8": "как", "tOffsetMs": 0},
                {"utf8": " дела", "tOffsetMs": 200},
                {"utf8": "\n"},
            ],
        },
    ]
}


def _make_video(db: Database) -> int:
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, title, external_id, url, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "youtube",
            "video",
            "Sample",
            "VIDEO_A",
            "https://example.invalid/v/VIDEO_A",
            "2026-09-21T00:00:00+00:00",
        ),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def _write_captions(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_parse_json3_uses_event_start_plus_segment_offset() -> None:
    words = parse_json3(SAMPLE)
    assert [(w.start_ms, w.text) for w in words] == [
        (0, "привет"),
        (320, "мир"),
        (1040, "как"),
        (1240, "дела"),
    ]


def test_parse_json3_skips_rollup_repeats_and_whitespace_segments() -> None:
    assert all(w.text.strip() for w in parse_json3(SAMPLE))
    assert len(parse_json3(SAMPLE)) == 4


def test_parse_json3_splits_a_multi_word_segment_on_the_same_start() -> None:
    payload = {"events": [{"tStartMs": 500, "segs": [{"utf8": "два слова", "tOffsetMs": 0}]}]}
    words = parse_json3(payload)
    assert [(w.start_ms, w.text) for w in words] == [(500, "два"), (500, "слова")]


def test_a_hyphenated_caption_word_becomes_two_findable_rows(
    db: Database, tmp_path: Path
) -> None:
    # One row per token, because nothing can match a row whose normalized_text
    # holds a space: Part 4's lookups and Part 5's pointer walk are both
    # single-token, and the user's query normalizes the same way.
    payload = {"events": [{"tStartMs": 0, "segs": [{"utf8": "кто-то", "tOffsetMs": 0}]}]}
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", payload)
    assert ingest_captions(db, video_id, path) == 2
    rows = db.conn.execute(
        "SELECT ord, normalized_text FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[1] for row in rows] == ["кто", "то"]
    assert all(" " not in row[1] for row in rows)


def test_parse_json3_keeps_starts_non_decreasing() -> None:
    payload = {
        "events": [
            {"tStartMs": 1000, "segs": [{"utf8": "поздно", "tOffsetMs": 0}]},
            {"tStartMs": 400, "segs": [{"utf8": "рано", "tOffsetMs": 0}]},
        ]
    }
    starts = [w.start_ms for w in parse_json3(payload)]
    assert starts == sorted(starts)


def test_ingest_writes_caption_rows_with_no_end_times(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    assert ingest_captions(db, video_id, path) == 4
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in rows] == [0, 1, 2, 3]
    assert all(row[2] is None for row in rows)
    assert {row[4] for row in rows} == {"caption"}
    assert {row[5] for row in rows} == {C.CAPTION_ENGINE}


def test_ingest_drops_punctuation_only_tokens(db: Database, tmp_path: Path) -> None:
    payload = {"events": [{"tStartMs": 0, "segs": [{"utf8": "—", "tOffsetMs": 0},
                                                   {"utf8": " слово", "tOffsetMs": 100}]}]}
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", payload)
    assert ingest_captions(db, video_id, path) == 1
    text = db.conn.execute(
        "SELECT text FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert text == "слово"


def test_ingest_twice_replaces_rather_than_appends(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    ingest_captions(db, video_id, path)
    ingest_captions(db, video_id, path)
    count = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert count == 4


def test_ingest_refuses_to_downgrade_an_aligned_video(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
        "stem, source, engine) VALUES (?, 0, 0, 100, 'х', 'х', 'х', 'aligned', 'fake')",
        (video_id,),
    )
    db.conn.commit()
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    with pytest.raises(CaptionDowngrade):
        ingest_captions(db, video_id, path)


def test_ingest_without_a_captions_asset_says_so(db: Database) -> None:
    video_id = _make_video(db)
    with pytest.raises(RytpError) as excinfo:
        ingest_captions(db, video_id)
    assert str(video_id) in str(excinfo.value)


def test_ingest_uses_the_captions_asset_when_no_path_is_given(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(path), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    assert ingest_captions(db, video_id) == 4


def test_ingest_clears_both_derived_tables(db: Database, tmp_path: Path) -> None:
    # Contracts §4: replacing a video's words deletes its utterances *and* its
    # video_speakers, in the same transaction.
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 100, 0, 1, 'x', 'x', 'x')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    db.conn.commit()
    ingest_captions(db, video_id, _write_captions(tmp_path / "captions.json3", SAMPLE))
    for table in ("utterances", "video_speakers"):
        left = db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        assert left == 0


def test_a_file_that_is_not_json3_is_reported(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    bad = tmp_path / "captions.json3"
    bad.write_text("<xml/>", encoding="utf-8")
    with pytest.raises(RytpError) as excinfo:
        ingest_captions(db, video_id, bad)
    assert "json3" in str(excinfo.value)
