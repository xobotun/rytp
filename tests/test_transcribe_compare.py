"""Where two engines disagree: on text, and on when each word happened."""
from __future__ import annotations

import pytest

from rytp.models import Span
from rytp.transcribe.compare import EngineRun, align_tokens, compare_pair


def _run(label: str, words: tuple[str, ...], starts: tuple[int, ...]) -> EngineRun:
    return EngineRun(
        label=label,
        words=words,
        spans=tuple(
            Span(start_ms=start, end_ms=start + 100, score=None) for start in starts
        ),
        elapsed_s=0.0,
    )


def test_identical_sequences_align_one_to_one() -> None:
    pairs = align_tokens(["а", "б", "в"], ["а", "б", "в"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_a_substitution_is_a_matched_pair_with_different_text() -> None:
    pairs = align_tokens(["а", "б", "в"], ["а", "х", "в"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_an_insertion_and_a_deletion_are_reported_as_such() -> None:
    assert align_tokens(["а", "в"], ["а", "б", "в"]) == [(0, 0), (None, 1), (1, 2)]
    assert align_tokens(["а", "б", "в"], ["а", "в"]) == [(0, 0), (1, None), (2, 1)]


def test_alignment_of_an_empty_side_is_all_insertions() -> None:
    assert align_tokens([], ["а", "б"]) == [(None, 0), (None, 1)]


def test_alignment_is_deterministic() -> None:
    a = ["один", "два", "три", "четыре"]
    b = ["один", "три", "четыре", "пять"]
    assert align_tokens(a, b) == align_tokens(a, b)


def test_compare_pair_counts_every_kind_of_difference() -> None:
    left = _run("a", ("один", "два", "три"), (0, 200, 400))
    right = _run("b", ("один", "икс", "три", "четыре"), (0, 210, 430, 600))
    stats = compare_pair(left, right)
    assert stats.a == "a"
    assert stats.b == "b"
    assert stats.matches == 2
    assert stats.substitutions == 1
    assert stats.insertions == 1
    assert stats.deletions == 0
    assert stats.disagreement == pytest.approx(2 / 3)


def test_compare_pair_measures_the_timing_gap_on_matched_words_only() -> None:
    left = _run("a", ("один", "два"), (0, 200))
    right = _run("b", ("один", "два"), (10, 240))
    stats = compare_pair(left, right)
    assert stats.median_abs_start_delta_ms == pytest.approx(25.0)
    assert stats.p90_abs_start_delta_ms is not None


def test_compare_pair_with_nothing_in_common_reports_no_timing_gap() -> None:
    stats = compare_pair(_run("a", ("один",), (0,)), _run("b", ("икс",), (0,)))
    assert stats.median_abs_start_delta_ms is None


def test_text_comparison_ignores_case_and_punctuation() -> None:
    stats = compare_pair(_run("a", ("Один,",), (0,)), _run("b", ("один",), (0,)))
    assert stats.substitutions == 0
    assert stats.matches == 1
