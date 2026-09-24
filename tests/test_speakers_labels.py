"""Per-video labels: stamped onto words once, linked to a person one row at a time."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.models import NotFoundError
from tests.fakes import make_video

ENGINE = "fake:test"

#: The diarizer name a test label claims to come from. `video_speakers.engine`
#: is NOT NULL (contracts §3), so every label needs one; only the tests that
#: are *about* provenance care which.
DIARIZER = "fake-diarizer"


def add_label(
    db: Database, video_id: int, local_label: str, *, engine: str = DIARIZER
) -> int:
    """`store.upsert_video_speaker` with the test engine filled in."""
    return store.upsert_video_speaker(db, video_id, local_label, engine=engine)


def add_words(db: Database, video_id: int, spans: list[tuple[int, int]]) -> list[int]:
    """Insert aligned words at the given (start_ms, end_ms) and return their ids."""
    ids: list[int] = []
    for ordinal, (start, end) in enumerate(spans):
        cursor = db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
            "stem, source, engine) VALUES (?, ?, ?, ?, ?, ?, ?, 'aligned', ?)",
            (video_id, ordinal, start, end, f"w{ordinal}", f"w{ordinal}", f"w{ordinal}", ENGINE),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def test_upsert_is_idempotent_on_video_and_label(db: Database) -> None:
    video_id = make_video(db)
    first = add_label(db, video_id, "SPEAKER_00")
    second = add_label(db, video_id, "SPEAKER_00")
    assert first == second
    assert len(store.label_rows(db, video_id)) == 1


def test_a_label_starts_unlinked(db: Database) -> None:
    video_id = make_video(db)
    add_label(db, video_id, "SPEAKER_04")
    row = store.label_rows(db, video_id)[0]
    assert row.local_label == "SPEAKER_04"
    assert row.speaker_id is None
    assert row.speaker_label is None


def test_a_label_records_which_diarizer_produced_it(db: Database) -> None:
    # contracts §3, design §11: a corpus built with more than one engine has
    # to stay interpretable, which is why words.engine exists and why this
    # column does too.
    video_id = make_video(db)
    store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="pyannote")
    assert store.label_rows(db, video_id)[0].engine == "pyannote"


def test_re_running_with_another_diarizer_overwrites_the_provenance(
    db: Database,
) -> None:
    video_id = make_video(db)
    first = store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="none")
    second = store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="pyannote")
    assert first == second
    assert store.get_label(db, first).engine == "pyannote"


def test_stamp_words_writes_the_label_onto_each_word(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800), (800, 1_200)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    assert store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id)) == 3
    row = store.label_rows(db, video_id)[0]
    assert (row.n_words, row.speech_ms) == (3, 1_200)


def test_stamp_words_clears_words_left_out_of_the_assignment(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    store.stamp_words(db, video_id, {word_ids[0]: label_id})
    remaining = db.conn.execute(
        "SELECT video_speaker_id FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    assert [row[0] for row in remaining] == [label_id, None]


def test_linking_a_label_to_a_person_changes_exactly_one_row(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(i * 400, i * 400 + 300) for i in range(50)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    speaker_id, _ = store.add_speaker(db, "Host One")

    before = db.conn.total_changes
    store.link_video_speaker(db, label_id, speaker_id)
    assert db.conn.total_changes - before == 1

    assert store.get_label(db, label_id).speaker_label == "Host One"


def test_unlinking_also_changes_exactly_one_row(db: Database) -> None:
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)

    before = db.conn.total_changes
    store.link_video_speaker(db, label_id, None)
    assert db.conn.total_changes - before == 1
    assert store.get_label(db, label_id).speaker_id is None


def test_linking_never_touches_the_word_rows(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    speaker_id, _ = store.add_speaker(db, "Host One")

    def fingerprint() -> str:
        return db.conn.execute(
            "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
        ).fetchone()[0]

    before = fingerprint()
    store.link_video_speaker(db, label_id, speaker_id)
    assert fingerprint() == before


def test_deleting_a_speaker_leaves_the_label_but_unlinks_it(db: Database) -> None:
    # ON DELETE SET NULL (contracts §3): the local voice survives the roster.
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)
    db.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
    assert store.get_label(db, label_id).speaker_id is None


def test_find_label_names_the_missing_one(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(NotFoundError, match="SPEAKER_09"):
        store.find_label(db, video_id, "SPEAKER_09")


def test_clear_video_speakers_removes_the_rows_and_nulls_the_words(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    assert store.clear_video_speakers(db, video_id) == 1
    assert store.label_rows(db, video_id) == []
    assert db.conn.execute("SELECT video_speaker_id FROM words").fetchone()[0] is None


def test_word_spans_returns_ids_and_timings_in_ordinal_order(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    assert store.word_spans(db, video_id) == [
        (word_ids[0], 0, 400),
        (word_ids[1], 400, 800),
    ]


def test_diarized_videos_lists_only_videos_with_labels(db: Database) -> None:
    mapped = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    bare = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    add_label(db, mapped, "SPEAKER_00")
    rows = store.diarized_videos(db)
    assert [row.video_id for row in rows] == [mapped]
    assert bare not in [row.video_id for row in rows]
    assert rows[0].n_unmapped == 1


def test_diarized_videos_can_show_only_the_ones_still_unmapped(db: Database) -> None:
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)
    assert store.diarized_videos(db, only_unmapped=True) == []
    assert len(store.diarized_videos(db)) == 1


def test_linked_labels_finds_every_label_of_one_person(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    for external_id in ("VIDEO_A", "VIDEO_B"):
        video_id = make_video(
            db, external_id=external_id, url=f"https://example.invalid/{external_id}"
        )
        label_id = add_label(db, video_id, "SPEAKER_00")
        store.link_video_speaker(db, label_id, speaker_id)
    assert len(store.linked_labels(db, speaker_id)) == 2
