"""Boundary refinement: a measured boundary beats a claimed one."""
from __future__ import annotations

import itertools

from rytp import constants as C
from rytp.audio.energy import (
    boundary_quality,
    enforce_monotonic,
    refine_boundaries,
    silence_depth_db,
)
from rytp.models import Span
from tests.synth_audio import SR, concat, silence, tone, tone_gap_tone


def test_a_boundary_claimed_early_moves_into_the_measured_silence() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=0.5),
        Span(start_ms=230, end_ms=520, score=0.5),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert gap_start <= refined[0].end_ms <= gap_end
    assert refined[0].end_ms == refined[1].start_ms


def test_adjacent_words_keep_sharing_one_boundary() -> None:
    samples, _, _ = tone_gap_tone()
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    for left, right in itertools.pairwise(refined):
        assert left.end_ms == right.start_ms


def test_a_boundary_never_moves_further_than_the_rail() -> None:
    samples, _, _ = tone_gap_tone(lead_ms=400, gap_ms=200, tail_ms=400)
    claimed = [
        Span(start_ms=0, end_ms=100, score=None),
        Span(start_ms=100, end_ms=1000, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert abs(refined[0].end_ms - 100) <= C.BOUNDARY_SEARCH_MS


def test_a_second_pass_stays_inside_the_measured_silence() -> None:
    # Not exact idempotence on noisy audio: a second pass re-centres the search
    # window, so a marginally quieter frame just outside the old one can win.
    # The property that matters is that it converges inside the real gap.
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    once = refine_boundaries(samples, SR, claimed)
    twice = refine_boundaries(samples, SR, once)
    assert gap_start <= once[0].end_ms <= gap_end
    assert gap_start <= twice[0].end_ms <= gap_end


def test_the_outer_edges_do_not_move_without_a_speech_edge_to_move_to() -> None:
    samples, _, _ = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert refined[0].start_ms == 0
    assert refined[-1].end_ms == 520


def test_refinement_inside_digital_silence_is_exactly_stable() -> None:
    # With no noise every frame in the gap ties, so the tie-breaks decide the
    # answer. They must hold the boundary still rather than walk it.
    samples = concat(tone(200), silence(200), tone(200))
    claimed = [
        Span(start_ms=0, end_ms=300, score=None),
        Span(start_ms=300, end_ms=600, score=None),
    ]
    once = refine_boundaries(samples, SR, claimed)
    assert refine_boundaries(samples, SR, once) == once


def test_refinement_is_deterministic() -> None:
    samples, _, _ = tone_gap_tone()
    claimed = [Span(start_ms=0, end_ms=230, score=None), Span(start_ms=230, end_ms=520, score=None)]
    assert refine_boundaries(samples, SR, claimed) == refine_boundaries(samples, SR, claimed)


def test_refined_spans_stay_ordered_and_never_cross() -> None:
    samples = concat(*[part for _ in range(4) for part in (tone(150), silence(100, amp=0.001))])
    claimed = [
        Span(start_ms=i * 250, end_ms=(i + 1) * 250, score=None) for i in range(4)
    ]
    refined = refine_boundaries(samples, SR, claimed)
    for span in refined:
        assert span.start_ms < span.end_ms
    for left, right in itertools.pairwise(refined):
        assert left.end_ms <= right.start_ms


def test_a_vad_edge_inside_the_window_wins_over_the_energy_search() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [Span(start_ms=0, end_ms=230, score=None), Span(start_ms=230, end_ms=520, score=None)]
    refined = refine_boundaries(
        samples, SR, claimed, speech=[(0, gap_start), (gap_end, 520)]
    )
    assert gap_start <= refined[0].end_ms <= gap_end


def test_enforce_monotonic_widens_a_zero_duration_word() -> None:
    spans = [
        Span(start_ms=100, end_ms=100, score=None),
        Span(start_ms=100, end_ms=300, score=None),
    ]
    fixed = enforce_monotonic(spans, total_ms=1000)
    assert fixed[0].end_ms - fixed[0].start_ms >= C.MIN_WORD_DURATION_MS
    assert fixed[1].start_ms >= fixed[0].end_ms


def test_enforce_monotonic_clamps_to_the_audio_length() -> None:
    spans = [Span(start_ms=900, end_ms=1200, score=None)]
    fixed = enforce_monotonic(spans, total_ms=1000)
    assert fixed[0].end_ms <= 1000


def test_silence_depth_is_large_in_a_gap_and_small_inside_a_tone() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    in_gap = silence_depth_db(samples, SR, (gap_start + gap_end) // 2)
    in_speech = silence_depth_db(samples, SR, 100)
    assert in_gap > 20.0
    assert in_speech < 6.0


def test_boundary_quality_is_bounded() -> None:
    assert boundary_quality(0.0) == 0.0
    assert boundary_quality(1000.0) == 1.0
    assert 0.0 < boundary_quality(C.SILENCE_DEPTH_FULL_DB / 2) < 1.0
