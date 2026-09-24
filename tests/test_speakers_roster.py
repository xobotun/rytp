"""The global roster: one row per real person, addressable by alias."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.models import InvalidInputError, NotFoundError
from tests.fakes import make_video


def test_add_speaker_returns_the_id_and_says_it_was_created(db: Database) -> None:
    speaker_id, created = store.add_speaker(db, "Host One", aliases=("host", "h1"))
    assert created is True
    speaker = store.get_speaker(db, speaker_id)
    assert speaker.label == "Host One"
    assert speaker.aliases == ("host", "h1")


def test_adding_the_same_label_twice_is_idempotent(db: Database) -> None:
    first, _ = store.add_speaker(db, "Host One")
    second, created_second = store.add_speaker(db, "Host One", aliases=("ignored",))
    assert second == first
    assert created_second is False
    # The second call must not silently rewrite the aliases.
    assert store.get_speaker(db, first).aliases == ()


def test_an_empty_label_is_refused(db: Database) -> None:
    with pytest.raises(InvalidInputError):
        store.add_speaker(db, "   ")


def test_get_speaker_names_the_missing_id(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        store.get_speaker(db, 4242)


def test_aliases_round_trip_through_json(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Каспар Хаузер", aliases=("каспар", "кх"))
    assert store.get_speaker(db, speaker_id).aliases == ("каспар", "кх")


def test_add_aliases_appends_without_duplicating(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One", aliases=("host",))
    assert store.add_aliases(db, speaker_id, ("HOST", "boss")) == ("host", "boss")


def test_update_speaker_can_rename_and_renotate(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.update_speaker(db, speaker_id, label="Host Uno", notes="the one in the hat")
    speaker = store.get_speaker(db, speaker_id)
    assert (speaker.label, speaker.notes) == ("Host Uno", "the one in the hat")


def test_find_speaker_accepts_an_id_a_label_or_an_alias(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Guest Two", aliases=("g2",))
    assert store.find_speaker(db, str(speaker_id)).id == speaker_id
    assert store.find_speaker(db, "Guest Two").id == speaker_id
    assert store.find_speaker(db, "guest two").id == speaker_id
    assert store.find_speaker(db, "G2").id == speaker_id
    assert store.find_speaker(db, "nobody") is None


def test_resolve_speaker_raises_rather_than_returning_none(db: Database) -> None:
    with pytest.raises(NotFoundError, match="nobody"):
        store.resolve_speaker(db, "nobody")


def test_alias_collisions_report_another_speakers_label_and_aliases(db: Database) -> None:
    one, _ = store.add_speaker(db, "Host One", aliases=("boss",))
    two, _ = store.add_speaker(db, "Guest Two")
    collisions = store.find_alias_collisions(db, ("boss", "guest two", "fresh"))
    assert sorted(alias for alias, _ in collisions) == ["boss", "guest two"]
    assert {sid for _, sid in collisions} == {one, two}


def test_removing_a_speaker_leaves_their_labels_as_unnamed_voices(
    db: Database,
) -> None:
    # contracts §5: ON DELETE SET NULL, not cascade. Losing the roster entry
    # must not lose the knowledge that this video had two distinct speakers.
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_id = make_video(db)
    for local in ("SPEAKER_00", "SPEAKER_01"):
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine) "
            "VALUES (?, ?, 'fake-diarizer')",
            (video_id, local),
        )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE local_label = 'SPEAKER_00'",
        (speaker_id,),
    )

    assert store.remove_speaker(db, speaker_id) == 1
    assert store.find_speaker(db, "Host One") is None
    rows = db.conn.execute(
        "SELECT local_label, speaker_id FROM video_speakers WHERE video_id = ? "
        "ORDER BY local_label",
        (video_id,),
    ).fetchall()
    assert [(r["local_label"], r["speaker_id"]) for r in rows] == [
        ("SPEAKER_00", None),
        ("SPEAKER_01", None),
    ]


def test_removing_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        store.remove_speaker(db, 4242)


def test_a_speaker_does_not_collide_with_itself(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One", aliases=("boss",))
    assert store.find_alias_collisions(db, ("boss",), exclude_speaker_id=speaker_id) == []


def test_parse_aliases_splits_and_trims_a_comma_separated_flag() -> None:
    assert store.parse_aliases(" host , , h1 ,host ") == ("host", "h1")
    assert store.parse_aliases("") == ()


def test_roster_rows_count_labels_and_videos(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    video_b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for video_id in (video_a, video_b):
        for label in ("SPEAKER_00", "SPEAKER_01"):
            db.conn.execute(
                "INSERT INTO video_speakers (video_id, local_label, engine) "
                "VALUES (?, ?, 'fake-diarizer')",
                (video_id, label),
            )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE local_label = 'SPEAKER_00'",
        (speaker_id,),
    )
    row = store.roster_rows(db)[0]
    assert (row.label, row.n_labels, row.n_videos) == ("Host One", 2, 2)


def test_roster_rows_are_ordered_by_label(db: Database) -> None:
    store.add_speaker(db, "Guest Two")
    store.add_speaker(db, "Host One")
    assert [row.label for row in store.roster_rows(db)] == ["Guest Two", "Host One"]
