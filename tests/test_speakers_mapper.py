"""The mapper, with no terminal in sight."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.mapper import MappingSession, format_ms, match_roster
from rytp.models import NotFoundError
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def roster(**kwargs: object) -> store.RosterRow:
    base: dict[str, object] = {
        "speaker_id": 1,
        "label": "Host One",
        "aliases": (),
        "notes": None,
        "n_labels": 0,
        "n_videos": 0,
    }
    base.update(kwargs)
    return store.RosterRow(**base)  # type: ignore[arg-type]


# -- pure matching ---------------------------------------------------------


def test_an_empty_query_matches_everything_in_order() -> None:
    rows = [roster(speaker_id=1, label="Guest Two"), roster(speaker_id=2, label="Host One")]
    assert match_roster(rows, "") == rows


def test_a_label_prefix_beats_an_alias_prefix() -> None:
    rows = [
        roster(speaker_id=1, label="Guest Two", aliases=("ho",)),
        roster(speaker_id=2, label="Host One"),
    ]
    assert [r.speaker_id for r in match_roster(rows, "ho")] == [2, 1]


def test_an_alias_prefix_beats_a_label_substring() -> None:
    rows = [
        roster(speaker_id=1, label="The Host One"),
        roster(speaker_id=2, label="Guest Two", aliases=("hostile",)),
    ]
    assert [r.speaker_id for r in match_roster(rows, "host")] == [2, 1]


def test_matching_ignores_case_on_both_sides() -> None:
    rows = [roster(speaker_id=1, label="Каспар Хаузер", aliases=("КХ",))]
    assert match_roster(rows, "кх") == rows
    assert match_roster(rows, "КАСПАР") == rows


def test_a_query_matching_nothing_returns_nothing() -> None:
    assert match_roster([roster()], "zzz") == []


def test_ties_are_broken_by_label_so_the_order_is_stable() -> None:
    rows = [roster(speaker_id=1, label="Host Two"), roster(speaker_id=2, label="Host One")]
    assert [r.label for r in match_roster(rows, "host")] == ["Host One", "Host Two"]


def test_format_ms_is_minutes_and_seconds() -> None:
    assert format_ms(0) == "0:00"
    assert format_ms(63_000) == "1:03"
    assert format_ms(3_723_000) == "62:03"


# -- the session -----------------------------------------------------------


@pytest.fixture
def session(db: Database) -> MappingSession:
    video_id = make_video(db, title="An Interview")
    word_ids = add_words(db, video_id, [(0, 400), (500, 900), (1_000, 1_400)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db, video_id, {word_ids[0]: host, word_ids[1]: host, word_ids[2]: guest}
    )
    store.add_speaker(db, "Host One", aliases=("h1",))
    store.add_speaker(db, "Guest Two", aliases=("g2",))
    return MappingSession(db, video_id)


def test_the_left_pane_is_the_videos_labels_with_their_weight(
    session: MappingSession,
) -> None:
    columns, rows = session.label_table()
    assert columns[0] == "label"
    assert [row[0] for row in rows] == ["SPEAKER_00", "SPEAKER_01"]
    assert rows[0][1] == "2"          # two words
    assert rows[0][3] == "—"          # nobody yet


def test_the_right_pane_is_the_roster(session: MappingSession) -> None:
    _columns, rows = session.roster_table()
    assert [row[0] for row in rows] == ["Guest Two", "Host One"]


def test_filtering_narrows_the_right_pane_only(session: MappingSession) -> None:
    session.set_query("h1")
    assert [row.label for row in session.matches] == ["Host One"]
    assert len(session.labels) == 2


def test_filtering_resets_the_roster_cursor(session: MappingSession) -> None:
    session.move_roster(1)
    session.set_query("h1")
    assert session.roster_index == 0
    assert session.selected_roster.label == "Host One"


def test_assign_links_the_selected_label_to_the_selected_person(
    session: MappingSession,
) -> None:
    session.set_query("Host One")
    message = session.assign()
    assert "SPEAKER_00" in message and "Host One" in message
    assert session.labels[0].speaker_label == "Host One"


def test_assign_advances_to_the_next_unmapped_label(session: MappingSession) -> None:
    session.set_query("Host One")
    session.assign()
    assert session.selected_label.local_label == "SPEAKER_01"


def test_assign_with_nothing_selected_says_so_and_writes_nothing(db: Database) -> None:
    empty = MappingSession(db, make_video(db))
    before = db.conn.total_changes
    assert "no label" in empty.assign().lower()
    assert db.conn.total_changes == before


def test_assign_to_accepts_an_alias(session: MappingSession) -> None:
    session.assign_to("g2")
    assert session.labels[0].speaker_label == "Guest Two"


def test_assign_to_an_unknown_name_raises(session: MappingSession) -> None:
    with pytest.raises(NotFoundError):
        session.assign_to("nobody at all")


def test_unassign_puts_a_label_back_to_nobody(session: MappingSession) -> None:
    session.assign_to("Host One")
    session.select_label(0)
    session.unassign()
    assert session.labels[0].speaker_id is None


def test_create_and_assign_adds_a_roster_entry_and_links_it(
    session: MappingSession,
) -> None:
    message = session.create_and_assign("Каспар Хаузер")
    assert "Каспар Хаузер" in message
    assert session.labels[0].speaker_label == "Каспар Хаузер"
    assert [row.label for row in session.roster] == [
        "Guest Two",
        "Host One",
        "Каспар Хаузер",
    ]


def test_create_and_assign_reuses_an_existing_label(session: MappingSession) -> None:
    session.create_and_assign("Host One")
    assert len(session.roster) == 2


def test_n_unmapped_counts_down_as_work_is_done(session: MappingSession) -> None:
    assert session.n_unmapped() == 2
    session.assign_to("Host One")
    assert session.n_unmapped() == 1


def test_the_cursors_never_leave_the_list(session: MappingSession) -> None:
    session.move_label(-5)
    assert session.label_index == 0
    session.move_label(99)
    assert session.label_index == 1
    session.move_roster(99)
    assert session.roster_index == 1


def test_a_session_over_a_video_with_no_labels_is_usable(db: Database) -> None:
    empty = MappingSession(db, make_video(db))
    assert empty.label_table()[1] == []
    assert empty.selected_label is None
    assert empty.n_unmapped() == 0
