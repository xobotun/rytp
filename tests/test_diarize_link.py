"""Cross-video suggestions: scoped, banded, and never applied."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import pack_embedding
from rytp.diarize.link import (
    MATCH,
    NO_MATCH,
    REVIEW,
    acoustic_distance,
    engines_in_play,
    enrolment,
    era_of,
    suggest_for_label,
    suggest_for_video,
    verdict_for,
)
from tests.fakes import make_video
from tests.test_speakers_labels import DIARIZER, add_label, add_words

# Unit vectors whose cosine with HOST_VOICE is exactly the first component,
# so every expected verdict below can be read off the thresholds by eye.
HOST_VOICE = [1.0, 0.0, 0.0, 0.0]
NEAR_HOST = [0.95, 0.3122499, 0.0, 0.0]     # 0.95: over HI and over HI_CROSS_ERA
MID_HOST = [0.75, 0.6614378, 0.0, 0.0]      # 0.75: over HI, under HI_CROSS_ERA
HALF_WAY = [0.6, 0.8, 0.0, 0.0]             # 0.60: in the band either way
OTHER_VOICE = [0.0, 1.0, 0.0, 0.0]          # 0.00: under LO


def a_labelled_video(
    db: Database,
    *,
    external_id: str,
    published_at: str,
    vector: list[float] | None = None,
    speaker_id: int | None = None,
    acoustics: dict[str, float] | None = None,
    engine: str = DIARIZER,
) -> tuple[int, int]:
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    label_id = add_label(db, video_id, "SPEAKER_00", engine=engine)
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    if vector is not None:
        store.set_embedding(db, label_id, pack_embedding(vector))
    if speaker_id is not None:
        store.link_video_speaker(db, label_id, speaker_id)
    if acoustics is not None:
        columns = ", ".join(acoustics)
        marks = ", ".join("?" for _ in acoustics)
        db.conn.execute(
            f"INSERT INTO video_acoustics (video_id, {columns}, computed_at) "
            f"VALUES (?, {marks}, '2026-01-01T00:00:00+00:00')",
            (video_id, *acoustics.values()),
        )
    return video_id, label_id


# -- eras ------------------------------------------------------------------


def test_eras_are_two_year_buckets() -> None:
    assert era_of("2014-03-02T00:00:00+00:00") == era_of("2015-11-30")
    assert era_of("2014-03-02") != era_of("2016-01-01")


def test_an_era_label_names_its_range() -> None:
    assert era_of("2015-06-01") == "2014-2015"


def test_a_video_with_no_date_lands_in_the_unknown_bucket() -> None:
    assert era_of(None) == C.SPEAKER_ERA_UNKNOWN
    assert era_of("") == C.SPEAKER_ERA_UNKNOWN
    assert era_of("not a date") == C.SPEAKER_ERA_UNKNOWN


# -- acoustic scoping ------------------------------------------------------


def test_two_similar_recordings_are_close(db: Database) -> None:
    a, _ = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    b, _ = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-06-01",
        acoustics={"f0_mean": 124.0, "noise_floor_db": -53.0},
    )
    assert acoustic_distance(db, a, b) < C.ACOUSTIC_MAX_DISTANCE


def test_two_different_recordings_are_far(db: Database) -> None:
    a, _ = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        acoustics={"f0_mean": 110.0, "noise_floor_db": -70.0},
    )
    b, _ = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-06-01",
        acoustics={"f0_mean": 200.0, "noise_floor_db": -30.0},
    )
    assert acoustic_distance(db, a, b) > C.ACOUSTIC_MAX_DISTANCE


def test_distance_is_unknown_when_a_video_was_never_fingerprinted(db: Database) -> None:
    a, _ = a_labelled_video(db, external_id="VIDEO_A", published_at="2020-01-01")
    b, _ = a_labelled_video(db, external_id="VIDEO_B", published_at="2020-06-01")
    assert acoustic_distance(db, a, b) is None


# -- the band --------------------------------------------------------------


def test_a_high_score_within_an_era_is_a_match() -> None:
    assert verdict_for(0.9, strict=False) == MATCH


def test_the_same_score_across_eras_only_earns_a_review() -> None:
    # design §6: a decade of changing microphones roughly triples the error.
    assert verdict_for(0.75, strict=False) == MATCH
    assert verdict_for(0.75, strict=True) == REVIEW


def test_a_low_score_is_a_confident_non_match() -> None:
    assert verdict_for(0.1, strict=False) == NO_MATCH
    assert verdict_for(0.1, strict=True) == NO_MATCH


def test_the_middle_of_the_band_goes_to_the_human() -> None:
    middle = (C.SPEAKER_MATCH_LO + C.SPEAKER_MATCH_HI) / 2
    assert verdict_for(middle, strict=False) == REVIEW


# -- enrolment -------------------------------------------------------------


def test_enrolment_groups_a_persons_labels_by_era(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2014-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2022-01-01",
        vector=NEAR_HOST, speaker_id=speaker_id,
    )
    grouped = enrolment(db, speaker_id)
    assert sorted(grouped) == ["2014-2015", "2022-2023"]
    assert len(grouped["2014-2015"]) == 1


def test_enrolment_carries_the_engine_each_sample_came_from(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2014-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    [sample] = enrolment(db, speaker_id)["2014-2015"]
    assert sample.engine == "pyannote"


def test_engines_in_play_reports_only_engines_with_embeddings(db: Database) -> None:
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, engine="pyannote",
    )
    # No vector, so this engine is not in play for comparison purposes.
    a_labelled_video(db, external_id="VIDEO_B", published_at="2020-02-01", engine="none")
    assert engines_in_play(db) == {"pyannote"}


# -- suggestions -----------------------------------------------------------


def test_a_near_identical_voice_in_the_same_era_is_suggested_as_a_match(
    db: Database,
) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01",
        vector=NEAR_HOST, acoustics={"f0_mean": 122.0, "noise_floor_db": -54.0},
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.speaker_label == "Host One"
    assert suggestion.verdict == MATCH
    assert suggestion.same_era is True
    assert suggestion.acoustically_close is True
    assert suggestion.similarity == pytest.approx(0.95, abs=0.01)


def test_the_same_voice_a_decade_apart_only_earns_a_review(db: Database) -> None:
    # 0.75 would be a confident match inside one era. Across fourteen years
    # it is only a suggestion, which is the whole point of the second bar.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2010-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2024-01-01",
        vector=MID_HOST, acoustics={"f0_mean": 122.0, "noise_floor_db": -54.0},
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.same_era is False
    assert suggestion.verdict == REVIEW


def test_an_unfingerprinted_pair_is_treated_as_strictly_as_a_cross_era_one(
    db: Database,
) -> None:
    # Same era, same 0.75 as the test above, but nobody ran `fingerprint`:
    # not knowing whether the recordings resemble each other is the same
    # kind of ignorance as knowing they were made a decade apart.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01", vector=MID_HOST
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.same_era is True
    assert suggestion.acoustically_close is False
    assert suggestion.verdict == REVIEW


def test_a_different_voice_is_reported_as_a_non_match_not_hidden(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01", vector=OTHER_VOICE
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.verdict == NO_MATCH


def test_suggestions_are_ranked_best_first(db: Database) -> None:
    near, _ = store.add_speaker(db, "Host One")
    far, _ = store.add_speaker(db, "Guest Two")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=near,
    )
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HALF_WAY, speaker_id=far,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_C", published_at="2020-09-01", vector=NEAR_HOST
    )
    assert [s.speaker_label for s in suggest_for_label(db, query_label)] == [
        "Host One",
        "Guest Two",
    ]


def test_a_person_already_on_another_label_of_this_video_is_not_suggested(
    db: Database,
) -> None:
    # One person cannot be two voices in the same recording.
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_id = make_video(db, published_at="2020-01-01")
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    taken = add_label(db, video_id, "SPEAKER_00")
    query = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: taken, word_ids[1]: query})
    store.set_embedding(db, query, pack_embedding(NEAR_HOST))
    store.link_video_speaker(db, taken, speaker_id)
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    assert suggest_for_label(db, query) == []


def test_a_label_with_no_embedding_gets_no_suggestions(db: Database) -> None:
    store.add_speaker(db, "Host One")
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01"
    )
    assert suggest_for_label(db, query_label) == []


def test_a_roster_entry_with_no_embedded_labels_is_skipped(db: Database) -> None:
    store.add_speaker(db, "Host One")
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01", vector=HOST_VOICE
    )
    assert suggest_for_label(db, query_label) == []


def test_labels_from_different_diarizers_are_never_compared(db: Database) -> None:
    # Different engines draw different turn boundaries, so a vector measured
    # over one engine's segments is not a measurement of the same thing.
    # Identical vectors, and still no suggestion.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, engine="none",
    )
    assert suggest_for_label(db, query_label) == []
    # …and the caller can find out why rather than guessing.
    assert engines_in_play(db) == {"none", "pyannote"}


def test_the_same_diarizer_on_both_sides_does_compare(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, engine="pyannote",
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.engine == "pyannote"
    assert suggestion.verdict == MATCH


def test_a_vector_of_another_dimension_is_skipped_not_fatal(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=[1.0, 0.0, 0.0], speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=HOST_VOICE
    )
    assert suggest_for_label(db, query_label) == []


def test_generating_suggestions_writes_nothing(db: Database) -> None:
    # design §6: cross-video linking is never applied silently.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    before = db.conn.total_changes
    assert suggest_for_label(db, query_label)
    assert db.conn.total_changes == before


def test_suggest_for_video_covers_every_unlinked_label(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    video_id, label_id = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    assert list(suggest_for_video(db, video_id)) == [label_id]


def test_the_mapper_exposes_suggestions_for_the_highlighted_label(db: Database) -> None:
    from rytp.diarize.mapper import MappingSession

    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    video_id, _label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    session = MappingSession(db, video_id)
    _columns, rows = session.suggestion_table()
    assert rows[0][0] == "Host One"
