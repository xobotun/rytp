"""Grouping words into the rows the FTS index shadows (design §4, §7)."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.index.utterances import (
    IndexedWord,
    build_utterances,
    drop_utterances,
    implied_end_ms,
    index_video,
    words_for,
)
from rytp.models import NotFoundError, normalize_text, stem_text

NOW = "2026-09-21T00:00:00+00:00"


def make_video(db: Database, title: str = "Sample") -> int:
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, created_at)"
        " VALUES ('ytdlp', 'video', ?, 'https://example.invalid/w/VIDEO_A', ?, ?)",
        (f"VIDEO_{title}", title, NOW),
    )
    return int(cursor.lastrowid)


def make_speaker(db: Database, video_id: int, local_label: str) -> int:
    """A diarizer label for one video.

    `engine` is NOT NULL (contracts §3) and supplying it is the inserting
    part's job — a fixture that omits it fails at insert time, in
    whichever test happens to run first.
    """
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) VALUES (?, ?, 'fake')",
        (video_id, local_label),
    )
    return int(cursor.lastrowid)


def seed_words(
    db: Database,
    video_id: int,
    spec: list[tuple[str, int, int | None]],
    *,
    source: str = "aligned",
    speaker_ids: list[int | None] | None = None,
) -> None:
    """Insert word rows from (text, start_ms, end_ms) triples."""
    speakers = speaker_ids or [None] * len(spec)
    for ordinal, ((text, start, end), speaker) in enumerate(zip(spec, speakers, strict=True)):
        normalized = normalize_text(text)
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine, video_speaker_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'fake', ?)",
            (video_id, ordinal, start, end, text, normalized, stem_text(normalized),
             source, speaker),
        )


def word(
    ordinal: int,
    text: str,
    start: int,
    end: int | None,
    *,
    speaker: int | None = None,
    source: str = "aligned",
) -> IndexedWord:
    normalized = normalize_text(text)
    return IndexedWord(
        ord=ordinal,
        start_ms=start,
        end_ms=end,
        text=text,
        normalized_text=normalized,
        stem=stem_text(normalized),
        source=source,
        video_speaker_id=speaker,
    )


def test_a_tight_run_with_one_speaker_is_one_utterance() -> None:
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300, 700),
        word(2, "друзья", 750, 1200),
    ]
    drafts = build_utterances(words)
    assert len(drafts) == 1
    assert drafts[0].first_word_ord == 0
    assert drafts[0].last_word_ord == 2
    assert drafts[0].start_ms == 0
    assert drafts[0].end_ms == 1200
    assert drafts[0].text == "Добрый вечер друзья"
    assert drafts[0].normalized_text == "добрый вечер друзья"


def test_a_speaker_change_starts_a_new_utterance() -> None:
    words = [
        word(0, "Добрый", 0, 300, speaker=1),
        word(1, "вечер", 300, 700, speaker=1),
        word(2, "Здравствуйте", 720, 1200, speaker=2),
    ]
    drafts = build_utterances(words)
    assert [(d.first_word_ord, d.last_word_ord) for d in drafts] == [(0, 1), (2, 2)]
    assert [d.video_speaker_id for d in drafts] == [1, 2]


def test_unlabelled_words_are_one_group_because_null_equals_null() -> None:
    """An undiarized video must not split on every word."""
    words = [word(i, "слово", i * 300, i * 300 + 250) for i in range(5)]
    assert len(build_utterances(words)) == 1


def test_a_silence_at_or_over_the_threshold_splits() -> None:
    gap = C.UTTERANCE_SILENCE_GAP_MS
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300, 700),
        word(2, "Сегодня", 700 + gap, 1000 + gap),
    ]
    assert [(d.first_word_ord, d.last_word_ord) for d in build_utterances(words)] == [
        (0, 1),
        (2, 2),
    ]


def test_a_silence_under_the_threshold_does_not_split() -> None:
    gap = C.UTTERANCE_SILENCE_GAP_MS - 1
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300 + gap, 700 + gap),
    ]
    assert len(build_utterances(words)) == 1


def test_the_word_cap_splits_an_unbroken_monologue() -> None:
    count = C.UTTERANCE_MAX_WORDS * 2 + 3
    words = [word(i, "слово", i * 100, i * 100 + 100) for i in range(count)]
    drafts = build_utterances(words)
    assert len(drafts) == 3
    assert [d.last_word_ord - d.first_word_ord + 1 for d in drafts] == [
        C.UTTERANCE_MAX_WORDS,
        C.UTTERANCE_MAX_WORDS,
        3,
    ]


def test_the_duration_cap_splits_a_slow_run_under_the_word_cap() -> None:
    """Short words, long gaps: every gap stays under the silence threshold
    and the run stays under the word cap, so only the duration cap can
    split this."""
    step = C.UTTERANCE_SILENCE_GAP_MS - 50
    words = [word(i, "слово", i * step, i * step + 40) for i in range(35)]
    drafts = build_utterances(words)
    assert len(drafts) > 1
    first_length = drafts[0].last_word_ord - drafts[0].first_word_ord + 1
    assert first_length < C.UTTERANCE_MAX_WORDS
    for draft in drafts:
        assert draft.end_ms - draft.start_ms <= C.UTTERANCE_MAX_DURATION_MS


def test_a_caption_word_with_no_end_time_borrows_the_next_start() -> None:
    words = [
        word(0, "Добрый", 0, None, source="caption"),
        word(1, "вечер", 200, None, source="caption"),
    ]
    assert implied_end_ms(words, 0) == 200


def test_a_caption_words_implied_end_is_capped_so_silence_still_splits() -> None:
    """Without the cap the implied end would be the next start and no
    caption-tier gap could ever reach the silence threshold."""
    far = C.CAPTION_WORD_FALLBACK_MS + C.UTTERANCE_SILENCE_GAP_MS + 1000
    words = [
        word(0, "Добрый", 0, None, source="caption"),
        word(1, "вечер", far, None, source="caption"),
    ]
    assert implied_end_ms(words, 0) == C.CAPTION_WORD_FALLBACK_MS
    assert len(build_utterances(words)) == 2


def test_the_last_caption_word_gets_the_fallback_duration() -> None:
    words = [word(0, "Добрый", 1000, None, source="caption")]
    drafts = build_utterances(words)
    assert drafts[0].end_ms == 1000 + C.CAPTION_WORD_FALLBACK_MS


def test_caption_words_sharing_one_start_do_not_split_and_do_not_go_backwards() -> None:
    words = [
        word(0, "Добрый", 500, None, source="caption"),
        word(1, "вечер", 500, None, source="caption"),
        word(2, "друзья", 500, None, source="caption"),
    ]
    drafts = build_utterances(words)
    assert len(drafts) == 1
    assert drafts[0].end_ms >= drafts[0].start_ms


def test_the_utterance_columns_are_a_join_of_the_word_columns() -> None:
    """Join the per-word columns; never re-run the text functions.

    Stemming is not idempotent — `сказали` stems to `сказа`, which stems
    again — so re-deriving `stem_text` from the joined string would drift
    away from `words.stem`. Joining also keeps the two columns
    token-for-token parallel, which is what lets a phrase position in the
    stem column mean the same word as in the exact column.
    """
    words = [word(0, "Кто", 0, 300), word(1, "то", 300, 600), word(2, "сказали", 600, 1100)]
    draft = build_utterances(words)[0]
    assert draft.text == "Кто то сказали"
    assert draft.normalized_text == "кто то сказали"
    assert draft.stem_text == "кто то сказа"
    assert len(draft.normalized_text.split()) == len(draft.stem_text.split())
    assert draft.stem_text != stem_text(draft.stem_text)


def test_no_words_means_no_utterances() -> None:
    assert build_utterances([]) == []


def test_index_video_writes_rows_and_returns_the_count(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    assert index_video(db, video_id) == 1
    row = db.conn.execute(
        "SELECT * FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert row["normalized_text"] == "добрый вечер"
    assert row["first_word_ord"] == 0
    assert row["last_word_ord"] == 1


def test_index_video_is_idempotent(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    first = index_video(db, video_id)
    second = index_video(db, video_id)
    assert first == second == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_index_video_only_touches_its_own_video(db: Database) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    seed_words(db, first, [("Добрый", 0, 300)])
    seed_words(db, second, [("Здравствуйте", 0, 400)])
    index_video(db, first)
    index_video(db, second)
    index_video(db, first)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 2


def test_reindexing_after_the_words_change_replaces_the_rows(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    seed_words(db, video_id, [("Сегодня", 0, 500)])
    index_video(db, video_id)
    rows = db.conn.execute(
        "SELECT normalized_text FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert [r["normalized_text"] for r in rows] == ["сегодня"]


def test_index_video_populates_the_fts_shadow(db: Database) -> None:
    """Part 1's triggers do this; the test proves the write path reaches them."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    hits = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchall()
    assert len(hits) == 1


def test_reindexing_removes_the_old_text_from_the_fts_shadow(db: Database) -> None:
    """A stale FTS row would be a hit pointing at text that no longer exists.

    Part 1's `utterances_ad` trigger deletes it; this asserts the write
    path actually reaches the trigger rather than trusting that it does.
    """
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    seed_words(db, video_id, [("Сегодня", 0, 400), ("поговорим", 400, 900)])
    index_video(db, video_id)
    stale = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchall()
    fresh = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сегодня поговорим"',),
    ).fetchall()
    assert stale == []
    assert len(fresh) == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_reindexing_leaves_the_fts_shadow_consistent(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    index_video(db, video_id)
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_drop_utterances_removes_them_and_reports_how_many(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    assert drop_utterances(db, video_id) == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_dropping_clears_the_fts_shadow_too(db: Database) -> None:
    """The failure mode most worth pinning: a surviving `utterances_fts`
    row is a search hit for text that is no longer anywhere."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    drop_utterances(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_dropping_leaves_the_words_alone(db: Database) -> None:
    """Derived data only. Removing words is `transcribe.remove` (Part 3)."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    drop_utterances(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2


def test_dropping_twice_is_not_an_error(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300)])
    index_video(db, video_id)
    assert drop_utterances(db, video_id) == 1
    assert drop_utterances(db, video_id) == 0


def test_dropping_only_touches_its_own_video(db: Database) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    seed_words(db, first, [("Добрый", 0, 300)])
    seed_words(db, second, [("Здравствуйте", 0, 400)])
    index_video(db, first)
    index_video(db, second)
    drop_utterances(db, first)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 1


def test_dropping_refuses_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        drop_utterances(db, 404)


def test_index_video_refuses_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError, match="404"):
        index_video(db, 404)


def test_words_for_returns_them_in_ordinal_order(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700), ("друзья", 700, 1100)])
    got = words_for(db, video_id)
    assert [w.ord for w in got] == [0, 1, 2]
    assert [w.text for w in got] == ["Добрый", "вечер", "друзья"]


def test_a_diarized_video_splits_per_turn(db: Database) -> None:
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    seed_words(
        db,
        video_id,
        [("Добрый", 0, 300), ("вечер", 300, 700), ("Здравствуйте", 720, 1300)],
        speaker_ids=[host, host, guest],
    )
    assert index_video(db, video_id) == 2
    rows = db.conn.execute(
        "SELECT video_speaker_id FROM utterances WHERE video_id = ? ORDER BY start_ms",
        (video_id,),
    ).fetchall()
    assert [r["video_speaker_id"] for r in rows] == [host, guest]
