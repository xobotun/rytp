"""Tests for ``rytp.speakers`` (roster + per-video map + pause stats)."""
from __future__ import annotations

import pytest

from rytp.models import Speaker
from rytp.speakers import (
    PauseStats,
    SpeakerMapper,
    add_speaker,
    compute_pause_stats,
    distinct_diarizer_speakers,
    list_speakers,
    map_diarizer_to_speaker,
    recompute_pause_stats,
    update_speaker,
    video_speaker_map,
)


# --- global roster ---------------------------------------------------------


def test_add_speaker_returns_id(db) -> None:
    sid, inserted = add_speaker(db, "Alice", aliases=("Al", "Алиса"))
    assert sid > 0
    assert inserted is True
    s = list_speakers(db)[0]
    assert s.label == "Alice"
    assert s.aliases == ["Al", "Алиса"]


def test_add_speaker_is_idempotent(db) -> None:
    sid1, _ = add_speaker(db, "Alice")
    sid2, _ = add_speaker(db, "Alice")
    assert sid1 == sid2


def test_add_speaker_existing_returns_inserted_false(db) -> None:
    sid1, inserted1 = add_speaker(db, "Alice")
    sid2, inserted2 = add_speaker(db, "Alice", aliases=("new_alias",))
    assert sid1 == sid2
    assert inserted1 is True
    assert inserted2 is False
    # Existing aliases were *not* updated by the second call.
    s = list_speakers(db)[0]
    assert s.aliases == []


def test_update_speaker(db) -> None:
    sid, _ = add_speaker(db, "Alice")
    update_speaker(db, sid, aliases=("Al",), notes="comedian")
    s = list_speakers(db)[0]
    assert s.aliases == ["Al"]
    assert s.notes == "comedian"


def test_list_speakers_ordered_by_label(db) -> None:
    _, _ = add_speaker(db, "Charlie")
    _, _ = add_speaker(db, "Alice")
    _, _ = add_speaker(db, "Bob")
    labels = [s.label for s in list_speakers(db)]
    assert labels == ["Alice", "Bob", "Charlie"]


# --- per-video mapping ------------------------------------------------------


def _make_video_with_words(
    db, fake_video_row, words_with_speakers: list[tuple[int, str]]
) -> int:
    """Insert a video + words carrying diarizer_speaker labels."""
    vid = db.upsert_video(fake_video_row)
    db.conn.executemany(
        """
        INSERT INTO words (video_id, start_ms, end_ms, text, normalized_text,
                           confidence, diarizer_speaker, speaker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        [
            (vid, s, e, t, t.lower(), 1.0, spk)
            for (s, e, t, spk) in (
                (s, s + 100, f"w{i}", spk) for i, (s, spk) in enumerate(words_with_speakers)
            )
        ],
    )
    db.conn.commit()
    return vid


def test_distinct_diarizer_speakers(db, fake_video_row) -> None:
    vid = _make_video_with_words(
        db,
        fake_video_row,
        [(0, "A"), (200, "B"), (400, "A"), (600, "C")],
    )
    labels = distinct_diarizer_speakers(db, vid)
    assert labels == ["A", "B", "C"]


def test_map_diarizer_to_speaker_writes_through_to_words(db, fake_video_row) -> None:
    vid = _make_video_with_words(
        db, fake_video_row, [(0, "A"), (200, "B"), (400, "A")]
    )
    sid, _ = add_speaker(db, "Alice")
    map_diarizer_to_speaker(db, vid, "A", sid)
    # The mapping row is written
    mapping = video_speaker_map(db, vid)
    assert mapping == {"A": sid}
    # The two "A" words got their speaker_id back-filled
    rows = db.conn.execute(
        "SELECT diarizer_speaker, speaker_id FROM words WHERE video_id = ? ORDER BY start_ms",
        (vid,),
    ).fetchall()
    assert rows[0]["speaker_id"] == sid
    assert rows[1]["speaker_id"] is None  # B isn't mapped
    assert rows[2]["speaker_id"] == sid


def test_map_diarizer_to_speaker_clear(db, fake_video_row) -> None:
    vid = _make_video_with_words(db, fake_video_row, [(0, "A")])
    sid, _ = add_speaker(db, "Alice")
    map_diarizer_to_speaker(db, vid, "A", sid)
    map_diarizer_to_speaker(db, vid, "A", None)
    rows = db.conn.execute(
        "SELECT speaker_id FROM words WHERE video_id = ?", (vid,)
    ).fetchall()
    assert rows[0]["speaker_id"] is None


# --- pause stats ------------------------------------------------------------


def test_compute_pause_stats_simple(db, fake_video_row) -> None:
    vid = _make_video_with_words(
        db, fake_video_row, [(0, "A"), (200, "A"), (400, "A"), (600, "A")]
    )
    sid, _ = add_speaker(db, "Alice")
    # Mark every word's speaker_id to make compute_pause_stats count them.
    db.conn.execute(
        "UPDATE words SET speaker_id = ? WHERE video_id = ?", (sid, vid)
    )
    db.conn.commit()
    stats = compute_pause_stats(db, sid)
    assert stats is not None
    # Pauses: 200-100=100, 400-300=100, 600-500=100 → all 100ms
    assert stats.n_samples == 3
    assert stats.min_ms == 100
    assert stats.max_ms == 100
    assert stats.mean_ms == pytest.approx(100.0)


def test_compute_pause_stats_skips_across_videos(db, fake_video_row) -> None:
    v1 = _make_video_with_words(db, fake_video_row, [(0, "A"), (200, "A")])
    v2 = _make_video_with_words(
        db, {**fake_video_row, "youtube_id": "v2"}, [(0, "A"), (200, "A")]
    )
    sid, _ = add_speaker(db, "Alice")
    db.conn.execute("UPDATE words SET speaker_id = ?", (sid,))
    db.conn.commit()
    stats = compute_pause_stats(db, sid)
    assert stats is not None
    # Two videos each contribute one pause sample of 100ms → n_samples=2
    assert stats.n_samples == 2
    assert stats.mean_ms == 100.0


def test_recompute_pause_stats_persists(db, fake_video_row) -> None:
    _make_video_with_words(
        db, fake_video_row, [(0, "A"), (200, "A"), (400, "A")]
    )
    sid, _ = add_speaker(db, "Alice")
    db.conn.execute("UPDATE words SET speaker_id = ?", (sid,))
    db.conn.commit()
    n = recompute_pause_stats(db)
    assert n == 1
    row = db.conn.execute(
        "SELECT * FROM speaker_pause_stats WHERE speaker_id = ?", (sid,)
    ).fetchone()
    assert row is not None
    assert row["n_samples"] == 2  # two inter-word gaps


def test_recompute_pause_stats_handles_no_samples(db) -> None:
    _, _ = add_speaker(db, "Lonely")
    n = recompute_pause_stats(db)
    assert n == 0  # no words → no stats row written


# --- SpeakerMapper (TUI backing) -------------------------------------------


def test_speaker_mapper_left_and_right_pane(db, fake_video_row) -> None:
    vid = _make_video_with_words(
        db, fake_video_row, [(0, "SPEAKER_00"), (200, "SPEAKER_01")]
    )
    _, _ = add_speaker(db, "Alice")
    _, _ = add_speaker(db, "Bob")
    mapper = SpeakerMapper(db, vid)
    assert mapper.left_pane() == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.label for s in mapper.right_pane()] == ["Alice", "Bob"]


def test_speaker_mapper_fuzzy_match(db, fake_video_row) -> None:
    vid = _make_video_with_words(db, fake_video_row, [(0, "SPEAKER_07")])
    _, _ = add_speaker(db, "Alice", aliases=("Al",))
    mapper = SpeakerMapper(db, vid)
    # No match
    assert mapper.fuzzy_match("SPEAKER_07") is None
    # Match by label
    _, _ = add_speaker(db, "SPEAKER_07 Person")
    assert mapper.fuzzy_match("SPEAKER_07").label == "SPEAKER_07 Person"