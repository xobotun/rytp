"""Two speaker identifier spaces, one resolver (contracts §5)."""

from __future__ import annotations

import pytest

from rytp.commands import SPEAKER_PARAMS, SpeakerFilter, resolve_speaker_filter
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture()
def roster(db: Database) -> Database:
    """Two videos, four diarized labels, three named people, one with aliases.

    `Ведущий` is mapped in both videos, `Гость` in one, `Призрак` in none
    — the roster entry that resolves but matches nothing.
    """
    for external_id in ("VIDEO_A", "VIDEO_B"):
        db.conn.execute(
            "INSERT INTO videos (source, kind, external_id, title, created_at)"
            " VALUES ('ytdlp', 'video', ?, ?, ?)",
            (external_id, external_id, NOW),
        )
    db.conn.execute(
        "INSERT INTO speakers (label, aliases_json, created_at)"
        " VALUES ('Ведущий', '[\"Host\", \"ведущий канала\"]', ?)",
        (NOW,),
    )
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Гость', ?)", (NOW,))
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Призрак', ?)", (NOW,))
    rows = [
        (1, "SPEAKER_00", 1),  # video_speakers.id 1
        (1, "SPEAKER_01", 2),  # 2
        (1, "SPEAKER_02", None),  # 3 — an unnamed voice
        (2, "SPEAKER_00", 1),  # 4 — the host again, in the other video
    ]
    for video_id, label, speaker_id in rows:
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
            " VALUES (?, ?, ?, 'pyannote')",
            (video_id, label, speaker_id),
        )
    return db


# -- rule 1: nothing asked for means no filter ------------------------


def test_no_speaker_named_returns_none(roster: Database) -> None:
    """Contracts §5 rule 1: the "show everything" case, distinct from a miss."""
    assert resolve_speaker_filter(roster) is None


def test_a_video_alone_is_not_a_speaker_filter(roster: Database) -> None:
    assert resolve_speaker_filter(roster, video_id=1) is None


# -- rule 3: the expansion happens here -------------------------------


def test_a_roster_name_expands_to_every_video_speaker_row(roster: Database) -> None:
    """Consumers filter words.video_speaker_id; the join belongs here, not there."""
    result = resolve_speaker_filter(roster, speaker="Ведущий")
    assert result is not None
    assert result.video_speaker_ids == frozenset({1, 4})
    assert result.description == 'speaker "Ведущий"'


def test_an_alias_resolves_to_the_same_rows_and_the_canonical_label(
    roster: Database,
) -> None:
    by_alias = resolve_speaker_filter(roster, speaker="Host")
    by_label = resolve_speaker_filter(roster, speaker="Ведущий")
    assert by_alias is not None and by_label is not None
    assert by_alias.video_speaker_ids == by_label.video_speaker_ids
    assert by_alias.description == 'speaker "Ведущий"'


def test_a_roster_name_can_be_narrowed_to_one_video(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Ведущий", video_id=2)
    assert result is not None
    assert result.video_speaker_ids == frozenset({4})
    assert "in video 2" in result.description


# -- rule 2: empty means nothing, not everything ----------------------


def test_an_unmapped_person_resolves_to_an_empty_set(roster: Database) -> None:
    """Contracts §5 rule 2: a real person nobody has mapped to a video yet."""
    result = resolve_speaker_filter(roster, speaker="Призрак")
    assert result is not None
    assert result.video_speaker_ids == frozenset()
    assert result.description == 'speaker "Призрак"'


def test_an_empty_result_is_distinguishable_from_no_filter(roster: Database) -> None:
    """The bug this guards: `IN ()` silently matching everything."""
    unmapped = resolve_speaker_filter(roster, speaker="Призрак")
    assert unmapped is not None  # not None: a filter WAS requested
    assert not unmapped.video_speaker_ids  # and it matches nothing
    assert resolve_speaker_filter(roster) is None  # this is "everything"


def test_narrowing_to_the_wrong_video_is_also_empty(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Гость", video_id=2)
    assert result is not None
    assert result.video_speaker_ids == frozenset()


# -- the local-label space --------------------------------------------


def test_a_local_label_resolves_within_its_video(roster: Database) -> None:
    result = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_01", video_id=1)
    assert result is not None
    assert result.video_speaker_ids == frozenset({2})
    assert result.description == "local label SPEAKER_01 in video 1"


def test_the_same_local_label_in_another_video_is_another_row(
    roster: Database,
) -> None:
    """SPEAKER_00 exists in both videos and they are different voices."""
    first = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00", video_id=1)
    second = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00", video_id=2)
    assert first is not None and second is not None
    assert first.video_speaker_ids == frozenset({1})
    assert second.video_speaker_ids == frozenset({4})


def test_an_unnamed_voice_still_resolves_by_its_local_label(roster: Database) -> None:
    result = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_02", video_id=1)
    assert result is not None
    assert result.video_speaker_ids == frozenset({3})


def test_a_local_label_without_a_video_is_rejected(roster: Database) -> None:
    """Contracts §5: alone it would match the first voice of every diarized video."""
    with pytest.raises(InvalidInputError, match="requires --video"):
        resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00")


def test_the_two_identifier_spaces_cannot_be_combined(roster: Database) -> None:
    with pytest.raises(InvalidInputError, match="one or the other"):
        resolve_speaker_filter(
            roster, speaker="Ведущий", video_local_speaker="SPEAKER_00", video_id=1
        )


# -- rule 4: unresolvable input raises --------------------------------


def test_speaker_never_accepts_a_raw_diarizer_label(roster: Database) -> None:
    with pytest.raises(NotFoundError, match="no speaker matches"):
        resolve_speaker_filter(roster, speaker="SPEAKER_00")


def test_an_unknown_speaker_names_the_nearest_roster_entries(roster: Database) -> None:
    """Rule 4: "no such person" and "that person said nothing" are different answers."""
    with pytest.raises(NotFoundError) as excinfo:
        resolve_speaker_filter(roster, speaker="Ведущй")
    assert "Ведущий" in str(excinfo.value)


def test_an_unknown_speaker_against_an_empty_roster_says_so(db: Database) -> None:
    with pytest.raises(NotFoundError, match="roster is empty"):
        resolve_speaker_filter(db, speaker="Ведущий")


def test_an_unknown_local_label_lists_the_labels_that_video_has(
    roster: Database,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        resolve_speaker_filter(roster, video_local_speaker="SPEAKER_09", video_id=1)
    message = str(excinfo.value)
    assert "SPEAKER_00" in message
    assert "SPEAKER_02" in message


def test_the_result_is_frozen_and_hashable(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Ведущий")
    assert result is not None
    assert isinstance(result.video_speaker_ids, frozenset)
    assert hash(result)


def test_the_shared_params_are_what_later_parts_splice_in() -> None:
    assert [param.name for param in SPEAKER_PARAMS] == [
        "speaker",
        "video_local_speaker",
        "video",
    ]
    assert all(param.default is None for param in SPEAKER_PARAMS)


def test_the_pinned_signature_has_not_drifted() -> None:
    """Parts 4, 5 and 7 call this; contracts §5 fixes the keywords."""
    import inspect

    parameters = inspect.signature(resolve_speaker_filter).parameters
    assert list(parameters) == ["db", "speaker", "video_local_speaker", "video_id"]
    assert [f.name for f in SpeakerFilter.__dataclass_fields__.values()] == [
        "video_speaker_ids",
        "description",
    ]
