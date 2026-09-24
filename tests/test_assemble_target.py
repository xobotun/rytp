"""Target text in, cut list out — the call both surfaces go through."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.assemble import AssembleControls, assemble_target, speaker_labels
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, RytpError
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_speaker,
    add_video,
    add_video_speaker,
    add_words,
)

CREATED = "2026-09-21T09:00:00+00:00"


@pytest.fixture()
def two_videos(db: Database) -> tuple[int, int]:
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "это неизбежно")
    return first, second


def test_it_assembles_across_two_videos(db: Database, two_videos: tuple[int, int]) -> None:
    first, second = two_videos
    cutlist = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    assert [slot.video_id for slot in cutlist.fragments] == [first, second]
    assert cutlist.gaps == ()
    assert cutlist.target == "мы все понимаем это неизбежно"
    assert cutlist.created_at == CREATED
    assert cutlist.schema_version == C.CUTLIST_SCHEMA_VERSION


def test_the_name_defaults_to_a_slug_of_the_target(
    db: Database, two_videos: tuple[int, int]
) -> None:
    cutlist = assemble_target(db, "Мы всё понимаем!", created_at=CREATED)
    assert cutlist.name == "мы-все-понимаем"
    named = assemble_target(db, "Мы всё понимаем!", name="кино", created_at=CREATED)
    assert named.name == "кино"


def test_the_controls_are_recorded_in_the_file(
    db: Database, two_videos: tuple[int, int]
) -> None:
    controls = AssembleControls(
        consistency=0.8, seed=5, pad_ms=40, speaker=None, exclude=(99,), min_align_score=0.1
    )
    params = assemble_target(db, "мы все", controls=controls, created_at=CREATED).params
    assert params.consistency == 0.8
    assert params.seed == 5
    assert params.pad_ms == 40
    assert params.exclude == (99,)
    assert params.min_align_score == 0.1


def test_padding_reaches_the_written_timings(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plain = assemble_target(db, "мы все", created_at=CREATED)
    padded = assemble_target(
        db, "мы все", controls=AssembleControls(pad_ms=60), created_at=CREATED
    )
    assert plain.fragments[0].end_ms == 2 * WORD_MS + GAP_MS
    assert padded.fragments[0].end_ms == 2 * WORD_MS + GAP_MS + 60


def test_a_missing_word_becomes_a_gap_with_substitutions(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы дело")
    cutlist = assemble_target(db, "мы дела", created_at=CREATED)
    assert len(cutlist.fragments) == 1
    gap = cutlist.gaps[0]
    assert gap.text == "дела"
    assert gap.substitutions[0].text == "дело"
    assert gap.substitutions[0].video_id == video_id
    assert gap.substitutions[0].end_ms > gap.substitutions[0].start_ms


def test_excluded_videos_are_not_used(db: Database, two_videos: tuple[int, int]) -> None:
    first, _second = two_videos
    cutlist = assemble_target(
        db, "мы все понимаем", controls=AssembleControls(exclude=(first,)), created_at=CREATED
    )
    assert cutlist.fragments == ()
    assert len(cutlist.gaps) == 3


def test_a_speaker_filter_restricts_the_sources_and_labels_them(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host_row = add_speaker(db, "host")
    host = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=host_row)
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы все", video_speaker_id=guest)
    add_words(db, video_id, "мы все", start_ms=100_000, video_speaker_id=host)
    cutlist = assemble_target(
        db, "мы все", controls=AssembleControls(speaker="host"), created_at=CREATED
    )
    assert cutlist.fragments[0].start_ms == 100_000
    assert cutlist.fragments[0].video_speaker_id == host
    assert cutlist.fragments[0].speaker_label == "host"


def test_an_unknown_speaker_is_refused_by_the_shared_resolver(db: Database) -> None:
    """contracts §5: resolution failure is an error, never a silent empty result."""
    add_speaker(db, "host")
    with pytest.raises(RytpError):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="guest"))


def test_a_raw_diarizer_label_is_not_a_speaker(db: Database) -> None:
    """contracts §5: --speaker never accepts a raw diarizer label."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_video_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "мы все")
    with pytest.raises(RytpError):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="SPEAKER_00"))


def test_a_speaker_nobody_has_been_mapped_to_says_so(db: Database) -> None:
    add_speaker(db, "host")
    with pytest.raises(NotFoundError, match="not mapped"):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="host"))


def test_no_speaker_requested_means_none_not_empty_string(
    db: Database, two_videos: tuple[int, int]
) -> None:
    """The two defaults must not collapse: None is "not asked", "" would be
    "asked for a speaker literally named the empty string"."""
    cutlist = assemble_target(db, "мы все понимаем", created_at=CREATED)
    assert cutlist.params.speaker == ""
    assert AssembleControls().speaker is None


def test_an_empty_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        assemble_target(db, "   !!!   ")


def test_an_absurdly_long_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="words"):
        assemble_target(db, "слово " * (C.ASSEMBLE_MAX_TARGET_WORDS + 1))


def test_the_controls_validate_themselves() -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        AssembleControls(consistency=2.0).validated()
    with pytest.raises(InvalidInputError, match="pad"):
        AssembleControls(pad_ms=-1).validated()
    with pytest.raises(InvalidInputError, match="pad"):
        AssembleControls(pad_ms=C.ASSEMBLE_MAX_PAD_MS + 1).validated()
    with pytest.raises(InvalidInputError, match="align"):
        AssembleControls(min_align_score=1.5).validated()
    assert AssembleControls().validated() == AssembleControls()


def test_speaker_labels_maps_only_the_mapped_ones(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    roster = add_speaker(db, "host")
    mapped = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=roster)
    unmapped = add_video_speaker(db, video_id, "SPEAKER_01")
    assert speaker_labels(db, [mapped, unmapped]) == {mapped: "host"}
    assert speaker_labels(db, []) == {}


def test_the_same_call_twice_gives_an_equal_cut_list(
    db: Database, two_videos: tuple[int, int]
) -> None:
    first = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    second = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    assert first == second
