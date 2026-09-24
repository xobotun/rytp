"""One knob, two endpoint profiles, and the costs the DP minimises."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rytp import constants as C
from rytp.assemble.score import (
    Weights,
    acoustic_distance,
    fragment_cost,
    jitter,
    load_acoustics,
    rounded,
    transition_cost,
    weights_for,
)
from rytp.db import Database
from rytp.models import InvalidInputError
from tests.assembly_corpus import add_acoustics, add_video


@dataclass(frozen=True)
class FakeRun:
    """Structurally a CandidateRun, without importing the matcher."""

    n_words: int = 3
    first_align: float = 1.0
    last_align: float = 1.0
    mean_align: float = 1.0


def test_the_knob_ends_are_the_two_named_profiles() -> None:
    assert weights_for(0.0) == Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    assert weights_for(1.0) == Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)


def test_the_knob_is_a_straight_line_between_them() -> None:
    middle = weights_for(0.5)
    low = Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    high = Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)
    assert middle.seam == pytest.approx((low.seam + high.seam) / 2)
    assert middle.switch == pytest.approx((low.switch + high.switch) / 2)
    assert middle.acoustic == pytest.approx((low.acoustic + high.acoustic) / 2)


def test_turning_the_knob_up_makes_seams_cheaper_and_switches_dearer() -> None:
    assert weights_for(1.0).seam < weights_for(0.0).seam
    assert weights_for(1.0).switch > weights_for(0.0).switch
    assert weights_for(1.0).acoustic > weights_for(0.0).acoustic


def test_the_seam_weight_never_reaches_zero() -> None:
    """Free seams would shred a long run into single words for no gain."""
    assert weights_for(1.0).seam > 0.0


def test_the_knob_rejects_a_value_off_the_dial() -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        weights_for(1.5)
    with pytest.raises(InvalidInputError, match="consistency"):
        weights_for(-0.1)


def test_a_perfectly_aligned_fragment_costs_exactly_one_seam() -> None:
    weights = weights_for(0.0)
    assert fragment_cost(FakeRun(), weights) == pytest.approx(weights.seam)


def test_a_badly_anchored_fragment_costs_more() -> None:
    weights = weights_for(0.0)
    good = fragment_cost(FakeRun(first_align=0.95, last_align=0.95, mean_align=0.95), weights)
    bad = fragment_cost(FakeRun(first_align=0.2, last_align=0.2, mean_align=0.2), weights)
    assert bad > good


def test_bad_edges_cost_more_than_a_bad_interior() -> None:
    """Only the first and last boundary are actually cut."""
    weights = weights_for(0.0)
    bad_edges = fragment_cost(FakeRun(first_align=0.2, last_align=0.2, mean_align=0.8), weights)
    bad_middle = fragment_cost(FakeRun(first_align=0.9, last_align=0.9, mean_align=0.4), weights)
    assert bad_edges > bad_middle


def test_staying_in_one_video_is_free() -> None:
    weights = weights_for(1.0)
    assert transition_cost(None, 1, weights, {}) == 0.0
    assert transition_cost(1, 1, weights, {}) == 0.0


def test_changing_video_costs_more_when_the_sources_sound_different() -> None:
    weights = weights_for(1.0)
    acoustics = {
        1: {"f0_mean": 120.0, "loudness_lufs": -23.0},
        2: {"f0_mean": 122.0, "loudness_lufs": -23.5},
        3: {"f0_mean": 230.0, "loudness_lufs": -14.0},
    }
    near = transition_cost(1, 2, weights, acoustics)
    far = transition_cost(1, 3, weights, acoustics)
    assert 0 < near < far


def test_at_fewest_seams_acoustics_do_not_enter_the_price() -> None:
    weights = weights_for(0.0)
    acoustics = {1: {"f0_mean": 120.0}, 2: {"f0_mean": 400.0}}
    assert transition_cost(1, 2, weights, acoustics) == pytest.approx(weights.switch)


def test_identical_acoustics_are_zero_apart() -> None:
    profile = {"f0_mean": 120.0, "noise_floor_db": -60.0}
    assert acoustic_distance(profile, dict(profile)) == 0.0


def test_an_unmeasured_video_is_middling_rather_than_excluded() -> None:
    assert acoustic_distance(None, {"f0_mean": 120.0}) == C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    assert acoustic_distance({}, {}) == C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE


def test_distance_is_symmetric_and_capped_per_field() -> None:
    near = {"f0_mean": 120.0, "loudness_lufs": -23.0}
    far = {"f0_mean": 9_000.0, "loudness_lufs": -23.0}
    assert acoustic_distance(near, far) == acoustic_distance(far, near)
    # one absurd field is clamped, so the shared field still counts for half
    assert acoustic_distance(near, far) == pytest.approx(C.ASSEMBLE_ACOUSTIC_DISTANCE_CAP / 2)


def test_only_fields_both_videos_have_are_compared() -> None:
    assert acoustic_distance({"f0_mean": 120.0, "reverb_proxy": 0.9}, {"f0_mean": 120.0}) == 0.0


def test_load_acoustics_reads_rows_and_skips_the_missing(db: Database) -> None:
    measured = add_video(db, external_id="VIDEO_A")
    unmeasured = add_video(db, external_id="VIDEO_B")
    add_acoustics(db, measured, f0_mean=180.0)
    loaded = load_acoustics(db, [measured, unmeasured])
    assert loaded[measured]["f0_mean"] == 180.0
    assert unmeasured not in loaded


def test_load_acoustics_skips_a_null_column(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_acoustics(db, video_id)
    db.conn.execute(
        "UPDATE video_acoustics SET reverb_proxy = NULL WHERE video_id = ?", (video_id,)
    )
    db.conn.commit()
    assert "reverb_proxy" not in load_acoustics(db, [video_id])[video_id]


def test_load_acoustics_on_no_videos_is_empty(db: Database) -> None:
    assert load_acoustics(db, []) == {}


def test_seed_zero_means_no_jitter_at_all() -> None:
    assert jitter(0, 1, 0, 0) == 0.0


def test_jitter_is_stable_across_processes_and_bounded() -> None:
    """blake2b, not hash(): Python salts str hashing per process."""
    value = jitter(7, 3, 120, 2)
    assert value == jitter(7, 3, 120, 2)
    assert 0.0 <= value <= C.ASSEMBLE_SEED_JITTER
    assert jitter(7, 3, 120, 2) == pytest.approx(0.03881913275923114)


def test_jitter_separates_candidates_and_seeds() -> None:
    assert jitter(7, 3, 120, 2) != jitter(7, 4, 120, 2)
    assert jitter(7, 3, 120, 2) != jitter(8, 3, 120, 2)


def test_rounded_makes_float_noise_tie() -> None:
    assert rounded(1.0) == rounded(1.0 + 1e-12)
    assert rounded(1.0) != rounded(1.1)
