"""Pure arithmetic over diarizer segments: no database, no audio, no engine."""

from __future__ import annotations

from rytp import constants as C
from rytp.diarize.segments import (
    label_words,
    local_labels,
    merge_segments,
    speech_ms_by_label,
    windows_for_label,
)
from rytp.models import DiarSegment


def seg(start: int, end: int, label: str = "SPEAKER_00") -> DiarSegment:
    return DiarSegment(start_ms=start, end_ms=end, local_label=label)


# -- merging ---------------------------------------------------------------


def test_touching_segments_of_one_label_merge() -> None:
    assert merge_segments([seg(0, 100), seg(100, 250)]) == [seg(0, 250)]


def test_overlapping_segments_of_one_label_merge() -> None:
    assert merge_segments([seg(0, 200), seg(150, 400)]) == [seg(0, 400)]


def test_segments_of_different_labels_never_merge() -> None:
    merged = merge_segments([seg(0, 200, "A"), seg(100, 400, "B")])
    assert merged == [seg(0, 200, "A"), seg(100, 400, "B")]


def test_merging_is_stable_and_sorted_by_start() -> None:
    merged = merge_segments([seg(500, 600, "B"), seg(0, 100, "A"), seg(200, 300, "A")])
    assert [(s.start_ms, s.local_label) for s in merged] == [
        (0, "A"),
        (200, "A"),
        (500, "B"),
    ]


# -- totals ----------------------------------------------------------------


def test_speech_totals_do_not_double_count_an_overlap() -> None:
    assert speech_ms_by_label([seg(0, 1_000, "A"), seg(500, 1_500, "A")]) == {"A": 1_500}


def test_local_labels_are_unique_and_sorted() -> None:
    assert local_labels([seg(0, 1, "B"), seg(2, 3, "A"), seg(4, 5, "B")]) == ("A", "B")


# -- labelling words -------------------------------------------------------


def test_each_word_takes_the_label_it_overlaps_most() -> None:
    words = [(1, 0, 400), (2, 400, 900), (3, 900, 1_400)]
    segments = [seg(0, 500, "A"), seg(500, 1_500, "B")]
    assert label_words(words, segments) == {1: "A", 2: "B", 3: "B"}


def test_a_word_straddling_a_boundary_goes_to_the_larger_share() -> None:
    # 100 ms in A, 300 ms in B.
    assert label_words([(1, 400, 800)], [seg(0, 500, "A"), seg(500, 900, "B")]) == {1: "B"}


def test_a_word_overlapping_nothing_is_left_unlabelled() -> None:
    # A gap in diarization is not an error: it is a stretch nobody labelled.
    assert label_words([(1, 5_000, 5_400)], [seg(0, 500, "A")]) == {}


def test_a_tie_goes_to_the_earlier_segment() -> None:
    assert label_words([(1, 400, 600)], [seg(0, 500, "A"), seg(500, 1_000, "B")]) == {1: "A"}


def test_a_zero_length_word_is_placed_by_its_start() -> None:
    # Caption-tier rows have no end; an aligned row can still collapse.
    assert label_words([(1, 600, 600), (2, 700, None)], [seg(500, 1_000, "A")]) == {
        1: "A",
        2: "A",
    }


def test_words_may_arrive_in_any_order() -> None:
    words = [(3, 900, 1_400), (1, 0, 400), (2, 400, 900)]
    segments = [seg(0, 500, "A"), seg(500, 1_500, "B")]
    assert label_words(words, segments) == {1: "A", 2: "B", 3: "B"}


def test_no_segments_labels_nothing() -> None:
    assert label_words([(1, 0, 100)], []) == {}


def test_labelling_ten_thousand_words_is_not_quadratic() -> None:
    # A real hour: ~10k words against ~2k segments. If this takes more than
    # a moment the implementation went back to scanning every segment.
    words = [(i, i * 350, i * 350 + 300) for i in range(10_000)]
    segments = [seg(i * 1_750, (i + 1) * 1_750, f"SPEAKER_{i % 3:02d}") for i in range(2_000)]
    assert len(label_words(words, segments)) == 10_000


# -- embedding windows -----------------------------------------------------


def test_windows_prefer_the_longest_segments_and_stay_chronological() -> None:
    segments = [
        seg(0, 2_000, "A"),
        seg(10_000, 15_000, "A"),
        seg(20_000, 21_500, "A"),
        seg(5_000, 6_000, "B"),
    ]
    assert windows_for_label(segments, "A", max_total_ms=7_000) == [
        (0, 2_000),
        (10_000, 15_000),
    ]


def test_windows_truncate_the_last_segment_to_the_budget() -> None:
    assert windows_for_label([seg(0, 60_000, "A")], "A", max_total_ms=30_000) == [
        (0, 30_000)
    ]


def test_windows_skip_segments_below_the_minimum_length() -> None:
    segments = [seg(0, 400, "A"), seg(1_000, 6_000, "A")]
    assert windows_for_label(segments, "A", min_segment_ms=1_000) == [(1_000, 6_000)]


def test_windows_fall_back_to_short_segments_when_there_is_nothing_else() -> None:
    # A label made entirely of interjections still deserves a vector; the
    # "is there enough speech at all" decision belongs to the caller.
    assert windows_for_label([seg(0, 400, "A"), seg(900, 1_200, "A")], "A") == [
        (0, 400),
        (900, 1_200),
    ]


def test_windows_for_an_unknown_label_are_empty() -> None:
    assert windows_for_label([seg(0, 1_000, "A")], "B") == []


def test_window_defaults_come_from_constants() -> None:
    assert windows_for_label([seg(0, 10_000_000, "A")], "A") == [
        (0, C.EMBED_MAX_SPEECH_MS)
    ]
