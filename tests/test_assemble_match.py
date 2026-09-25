"""The pointer walk: occurrences of the first word, then forward by ordinal."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.assemble.match import (
    SLOT_FRAGMENT,
    SLOT_GAP,
    MatchFilters,
    Plan,
    SubstitutionHit,
    build_run_table,
    diagnose_absence,
    edit_distance_at_most,
    find_occurrences,
    pad_fragments,
    plan_coverage,
    suggest_substitutions,
    tokenize,
)
from rytp.assemble.score import load_acoustics, weights_for
from rytp.db import Database
from rytp.models import InvalidInputError
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_acoustics,
    add_video,
    add_video_speaker,
    add_words,
)

TARGET = "мы все понимаем что это неизбежно"


def energy_floor(floor: float) -> MatchFilters:
    """A filter whose only change from the default is the energy-scale floor
    (plan §1a) — everything the fixtures here write is energy-scale by
    default (see ``tests.assembly_corpus.add_words``)."""
    return MatchFilters(
        min_align_by_scale={**C.ASSEMBLE_MIN_ALIGN_BY_SCALE, C.ALIGN_SCALE_ENERGY: floor}
    )


@pytest.fixture()
def corpus(db: Database) -> tuple[int, int]:
    """Two overlapping videos. Neither says the whole target; together they do."""
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем что это")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "все понимаем что это неизбежно")
    return first, second


def test_tokenize_normalizes_and_splits() -> None:
    assert tokenize("  Мы  ВСЁ, понимаем! ") == ("мы", "все", "понимаем")


def test_tokenize_splits_a_hyphenated_word_into_two_tokens() -> None:
    assert tokenize("кто-то") == ("кто", "то")


def test_a_hyphenated_word_is_assemblable(db: Database) -> None:
    """Contracts §4: a stored row holds exactly one token, on both sides.

    Part 3's split_token() writes "кто-то" as two ordinally adjacent rows
    with a measured interior boundary, and the target normalizes to the
    same two tokens — so a run walks straight through the hyphen.
    """
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "кто-то еще")
    assert [row.normalized_text for row in find_occurrences(db, "кто", MatchFilters())] == ["кто"]
    table = build_run_table(db, tokenize("кто-то еще"), MatchFilters())
    run = table[(0, 3)][video_id]
    assert run.n_words == 3
    assert run.first_word_ord == 0
    assert run.last_word_ord == 2


def test_find_occurrences_returns_every_video_that_says_the_word(
    db: Database, corpus: tuple[int, int]
) -> None:
    first, second = corpus
    found = find_occurrences(db, "все", MatchFilters())
    assert {row.video_id for row in found} == {first, second}
    assert {row.ord for row in found} == {0, 1}


def test_find_occurrences_coalesces_a_null_align_score(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=None)
    assert find_occurrences(db, "неизбежно", MatchFilters())[0].align_score == pytest.approx(
        C.ASSEMBLE_DEFAULT_ALIGN_SCORE
    )


def test_only_aligned_words_are_eligible(db: Database) -> None:
    """contracts §3: cuttable is source = 'aligned' and nothing else.

    Captions carry no ends at all; `timed` rows carry a transcriber's own
    timestamps, which put 78.7% of word gaps at exactly zero and cannot be
    cut on. Both are searchable elsewhere and neither is assemblable.
    """
    for tier in ("caption", "timed"):
        video_id = add_video(db, external_id=f"VIDEO_{tier}")
        add_words(db, video_id, "неизбежно", source=tier)
        assert find_occurrences(db, "неизбежно", MatchFilters()) == [], tier
    cuttable = add_video(db, external_id="VIDEO_OK")
    add_words(db, cuttable, "неизбежно", source="aligned")
    assert [row.video_id for row in find_occurrences(db, "неизбежно", MatchFilters())] == [
        cuttable
    ]


def test_an_empty_speaker_filter_matches_nothing_not_everything(db: Database) -> None:
    """contracts §5 rule 2: a resolved-but-unmapped speaker returns nothing."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно")
    assert find_occurrences(db, "неизбежно", MatchFilters()) != []
    empty = MatchFilters(video_speaker_ids=frozenset())
    assert find_occurrences(db, "неизбежно", empty) == []
    assert build_run_table(db, ("неизбежно",), empty) == {}


def test_an_aligned_word_with_no_score_is_usable_but_outranked(db: Database) -> None:
    """A null score on an aligned row means the aligner reported none (MFA),
    not that the row is unaligned — that case is the `timed` tier now."""
    scoreless = add_video(db, external_id="VIDEO_A")
    add_words(db, scoreless, "неизбежно", align_score=None)
    measured = add_video(db, external_id="VIDEO_B")
    add_words(db, measured, "неизбежно", align_score=0.95)
    found = find_occurrences(db, "неизбежно", MatchFilters())
    assert [row.video_id for row in found] == [measured, scoreless]
    assert found[1].align_score == pytest.approx(C.ASSEMBLE_DEFAULT_ALIGN_SCORE)


def test_excluded_videos_are_dropped(db: Database, corpus: tuple[int, int]) -> None:
    first, second = corpus
    found = find_occurrences(db, "все", MatchFilters(exclude_video_ids=frozenset({first})))
    assert {row.video_id for row in found} == {second}


def test_min_align_score_drops_badly_anchored_words(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=0.3)
    assert find_occurrences(db, "неизбежно", energy_floor(0.5)) == []
    assert find_occurrences(db, "неизбежно", energy_floor(0.2)) != []


# -- plan §1a: per-scale eligibility (BUGS.md entries 26, 28) --------------


def test_a_wav2vec2_shaped_logprob_corpus_is_eligible(db: Database) -> None:
    """entry 26: log-probabilities are negative by definition, so a fixed
    0.0 floor rejected every wav2vec2-aligned word. The scale-keyed floor
    fixes it without ever converting the score."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=-0.96, align_scale=C.ALIGN_SCALE_LOGPROB)
    assert find_occurrences(db, "неизбежно", MatchFilters()) != []


def test_a_logprob_word_below_its_own_floor_is_excluded(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=-9.0, align_scale=C.ALIGN_SCALE_LOGPROB)
    assert find_occurrences(db, "неизбежно", MatchFilters()) == []


def test_a_none_scale_word_has_no_floor_at_all(db: Database) -> None:
    """MFA reports no score, and contracts §3 permits that: `align_scale =
    'none'` with a NULL score is a real aligner having nothing to report,
    not something to gate on."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=None, align_scale=C.ALIGN_SCALE_NONE)
    assert find_occurrences(db, "неизбежно", MatchFilters()) != []


def test_an_unknown_scale_word_is_excluded_regardless_of_its_score(db: Database) -> None:
    """entry 36's fabricated boundaries, and entry 28's un-identifiable
    sign flips: `unknown` is excluded outright, never floored — a very
    good-looking score proves nothing about a corrupted row."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=0.99, align_scale=C.ALIGN_SCALE_UNKNOWN)
    assert find_occurrences(db, "неизбежно", MatchFilters()) == []


def test_ordering_within_a_scale_still_prefers_the_better_score(db: Database) -> None:
    """entry 28: ordering must never cross scales, but within one scale
    the best-anchored word still comes first."""
    worse = add_video(db, external_id="VIDEO_A")
    add_words(db, worse, "неизбежно", align_score=-3.0, align_scale=C.ALIGN_SCALE_LOGPROB)
    better = add_video(db, external_id="VIDEO_B")
    add_words(db, better, "неизбежно", align_score=-0.1, align_scale=C.ALIGN_SCALE_LOGPROB)
    found = find_occurrences(db, "неизбежно", MatchFilters())
    assert [row.video_id for row in found] == [better, worse]


def test_timed_words_are_invisible_by_default_even_when_scored_well(db: Database) -> None:
    """D1: `--allow-timed` gates by tier alone. A `timed` row is never
    admitted by score, however good it looks, unless the flag is set."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", source="timed")
    assert find_occurrences(db, "неизбежно", MatchFilters()) == []
    assert find_occurrences(db, "неизбежно", MatchFilters(allow_timed=True)) != []


def test_allow_timed_admits_timed_words_with_no_threshold(db: Database) -> None:
    """D1: an override, not a gate — no threshold, no quality logic."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(
        db, video_id, "неизбежно", source="timed", align_score=None, align_scale=None
    )
    found = find_occurrences(db, "неизбежно", MatchFilters(allow_timed=True))
    assert [row.source for row in found] == ["timed"]


def test_allow_timed_still_excludes_caption_tier(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", source="caption")
    assert find_occurrences(db, "неизбежно", MatchFilters(allow_timed=True)) == []


def test_the_occurrence_lookup_admits_both_tiers_under_allow_timed(db: Database) -> None:
    """The UNION ALL branches independently: an aligned word and a timed
    word for the same token both surface once the flag is set."""
    aligned_video = add_video(db, external_id="VIDEO_A")
    add_words(db, aligned_video, "неизбежно")
    timed_video = add_video(db, external_id="VIDEO_B")
    add_words(db, timed_video, "неизбежно", source="timed")
    found = find_occurrences(db, "неизбежно", MatchFilters(allow_timed=True))
    assert {row.video_id for row in found} == {aligned_video, timed_video}


def test_a_run_reports_the_weakest_tier_it_contains(db: Database) -> None:
    """A run mixing an aligned and a timed word (only possible under
    --allow-timed) must report `timed`, not silently look fully aligned."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы")
    add_words(db, video_id, "все", start_ms=WORD_MS, source="timed")
    filters = MatchFilters(allow_timed=True)
    table = build_run_table(db, ("мы", "все"), filters)
    run = table[(0, 2)][video_id]
    assert run.tier == "timed"


def test_occurrences_are_capped_per_video_to_keep_sources_diverse(db: Database) -> None:
    crowded = add_video(db, external_id="VIDEO_A")
    for _ in range(10):
        add_words(db, crowded, "так так так так")
    quiet = add_video(db, external_id="VIDEO_B")
    add_words(db, quiet, "так")
    found = find_occurrences(db, "так", MatchFilters(max_seeds_per_video=2))
    assert len([row for row in found if row.video_id == crowded]) == 2
    assert quiet in {row.video_id for row in found}


def test_occurrences_are_ordered_deterministically(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "так так так")
    db.conn.execute("UPDATE words SET align_score = 0.99 WHERE ord = 2")
    db.conn.commit()
    found = find_occurrences(db, "так", MatchFilters())
    assert [row.ord for row in found] == [2, 0, 1]
    assert find_occurrences(db, "так", MatchFilters()) == found


def test_run_table_records_every_prefix_length(db: Database, corpus: tuple[int, int]) -> None:
    first, _second = corpus
    table = build_run_table(db, tokenize(TARGET), MatchFilters())
    for length in range(1, 6):
        assert first in table[(0, length)], length
    assert table[(0, 5)][first].text == "мы все понимаем что это"
    assert (0, 6) not in table  # video A never says "неизбежно"


def test_run_table_finds_the_long_run_starting_at_position_one(
    db: Database, corpus: tuple[int, int]
) -> None:
    _first, second = corpus
    table = build_run_table(db, tokenize(TARGET), MatchFilters())
    run = table[(1, 5)][second]
    assert run.text == "все понимаем что это неизбежно"
    assert run.first_word_ord == 0
    assert run.last_word_ord == 4
    assert run.n_words == 5


def test_a_run_carries_the_span_of_its_words(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    run = table[(0, 2)][video_id]
    assert run.start_ms == 0
    assert run.end_ms == 2 * WORD_MS + GAP_MS


def test_a_run_does_not_span_a_long_silence(db: Database) -> None:
    """Design §9 gives pause length to the renderer; a fragment must not
    smuggle in a pause the target sentence does not have."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы")
    add_words(db, video_id, "все", start_ms=WORD_MS + 5_000)
    table = build_run_table(db, ("мы", "все"), MatchFilters(max_internal_gap_ms=800))
    assert (0, 2) not in table
    assert video_id in table[(0, 1)]
    relaxed = build_run_table(db, ("мы", "все"), MatchFilters(max_internal_gap_ms=6_000))
    assert video_id in relaxed[(0, 2)]


def test_a_run_does_not_span_a_speaker_change(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host = add_video_speaker(db, video_id, "SPEAKER_00")
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы", video_speaker_id=host)
    add_words(db, video_id, "все", start_ms=WORD_MS + GAP_MS, video_speaker_id=guest)
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    assert (0, 2) not in table
    assert video_id in table[(0, 1)]


def test_an_unknown_word_produces_no_entry_at_its_position(
    db: Database, corpus: tuple[int, int]
) -> None:
    tokens = tokenize("мы все зеленеем")
    table = build_run_table(db, tokens, MatchFilters())
    assert (2, 1) not in table
    assert (0, 3) not in table


def test_max_run_words_caps_the_longest_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    text = " ".join(["так"] * 10)
    add_words(db, video_id, text)
    table = build_run_table(db, tokenize(text), MatchFilters(max_run_words=3))
    assert (0, 3) in table
    assert (0, 4) not in table


def test_the_best_run_per_video_wins_on_alignment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", align_score=0.4)
    add_words(db, video_id, "мы все", start_ms=100_000, align_score=0.95)
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    assert table[(0, 2)][video_id].first_word_ord == 2


def test_a_speaker_filter_restricts_the_walk(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host = add_video_speaker(db, video_id, "SPEAKER_00")
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы все", video_speaker_id=host)
    add_words(db, video_id, "мы все", start_ms=100_000, video_speaker_id=guest)
    filters = MatchFilters(video_speaker_ids=frozenset({guest}))
    table = build_run_table(db, ("мы", "все"), filters)
    assert table[(0, 2)][video_id].first_word_ord == 2


def make_plan(
    db: Database,
    target: str,
    *,
    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY,
    seed: int = 0,
    filters: MatchFilters | None = None,
) -> Plan:
    """Tokenize, walk, score, cover — what Task 9 will do for real."""
    tokens = tokenize(target)
    table = build_run_table(db, tokens, filters or MatchFilters())
    videos = {run.video_id for bucket in table.values() for run in bucket.values()}
    return plan_coverage(
        target,
        tokens,
        table,
        weights=weights_for(consistency),
        acoustics=load_acoustics(db, videos),
        seed=seed,
    )


def test_one_video_that_says_it_all_gives_one_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все понимаем")
    plan = make_plan(db, "мы все понимаем")
    assert len(plan.slots) == 1
    slot = plan.slots[0]
    assert slot.kind == SLOT_FRAGMENT
    assert slot.run is not None
    assert slot.run.video_id == video_id
    assert (slot.target_first, slot.target_last) == (0, 2)
    assert plan.n_sources == 1


def test_the_greedy_counterexample_from_the_plan(db: Database) -> None:
    """Greedy takes A's longer run and pays a source switch; the DP takes
    B's shorter run and covers the whole target from one video."""
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем что")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "мы все понимаем")
    add_words(db, second, "что это", start_ms=60_000)

    tokens = tokenize("мы все понимаем что это")
    table = build_run_table(db, tokens, MatchFilters())
    assert first in table[(0, 4)]  # greedy's four-word run exists
    assert (0, 5) not in table  # and nobody covers the whole target

    plan = make_plan(db, "мы все понимаем что это")
    assert len(plan.fragments) == 2
    assert plan.n_sources == 1
    assert plan.source_video_ids == (second,)


def test_the_knob_trades_fragments_for_sources(db: Database) -> None:
    """Three cuts in one video beat two cuts across two that sound nothing alike."""
    lone = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, lone, "мы все")
    add_words(db, lone, "понимаем это", start_ms=60_000)
    add_words(db, lone, "неизбежно", start_ms=120_000)
    wide = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, wide, "мы все понимаем это")
    tail = add_video(db, external_id="VIDEO_C", title="C")
    add_words(db, tail, "неизбежно")
    add_acoustics(db, lone, f0_mean=120.0)
    add_acoustics(db, wide, f0_mean=110.0, loudness_lufs=-23.0)
    add_acoustics(db, tail, f0_mean=260.0, loudness_lufs=-9.0)

    target = "мы все понимаем это неизбежно"
    seams = make_plan(db, target, consistency=0.0)
    assert len(seams.fragments) == 2
    assert seams.n_sources == 2

    consistent = make_plan(db, target, consistency=1.0)
    assert len(consistent.fragments) == 3
    assert consistent.n_sources == 1
    assert consistent.source_video_ids == (lone,)


def test_a_badly_anchored_run_loses_to_an_equal_length_clean_one(db: Database) -> None:
    sloppy = add_video(db, external_id="VIDEO_A")
    add_words(db, sloppy, "мы все", align_score=0.25)
    clean = add_video(db, external_id="VIDEO_B")
    add_words(db, clean, "мы все", align_score=0.95)
    plan = make_plan(db, "мы все", consistency=0.0)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.video_id == clean


def test_a_missing_word_becomes_a_gap_and_the_rest_is_still_cut(db: Database) -> None:
    """Design §8: 'report the gap plainly and render everything else'."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все зеленеем")
    assert [slot.kind for slot in plan.slots] == [SLOT_FRAGMENT, SLOT_GAP]
    gap = plan.gaps[0]
    assert gap.text == "зеленеем"
    assert (gap.target_first, gap.target_last) == (2, 2)
    assert gap.run is None
    assert len(plan.fragments) == 1


def test_every_word_missing_gives_a_gap_per_word_and_no_crash(db: Database) -> None:
    add_video(db, external_id="VIDEO_A")
    plan = make_plan(db, "мы все")
    assert [slot.kind for slot in plan.slots] == [SLOT_GAP, SLOT_GAP]
    assert [slot.text for slot in plan.slots] == ["мы", "все"]
    assert plan.n_sources == 0


def test_a_gap_is_never_preferred_to_a_real_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы", align_score=0.01)
    plan = make_plan(db, "мы", consistency=1.0)
    assert plan.slots[0].kind == SLOT_FRAGMENT
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.video_id == video_id


def test_slot_costs_add_up_to_the_total(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    add_words(db, video_id, "это", start_ms=60_000)
    plan = make_plan(db, "мы все это зеленеем")
    assert plan.total_cost == pytest.approx(sum(slot.cost for slot in plan.slots))


def test_alternatives_rank_the_other_sources_and_exclude_the_chosen_one(
    db: Database,
) -> None:
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все", align_score=0.95)
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все", align_score=0.80)
    third = add_video(db, external_id="VIDEO_C")
    add_words(db, third, "мы все", align_score=0.60)
    plan = make_plan(db, "мы все", consistency=0.0)
    chosen = plan.slots[0]
    assert chosen.run is not None
    assert chosen.run.video_id == first
    assert [scored.run.video_id for scored in chosen.alternatives] == [second, third]
    assert chosen.alternatives[0].cost < chosen.alternatives[1].cost


def test_alternatives_are_capped(db: Database) -> None:
    for index in range(6):
        video_id = add_video(db, external_id=f"VIDEO_{index}")
        add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все")
    assert len(plan.slots[0].alternatives) == C.ASSEMBLE_MAX_ALTERNATIVES


def test_the_same_input_gives_the_same_plan(db: Database) -> None:
    """Design §8: 'Determinism. Same input gives the same output.'"""
    for index in range(4):
        video_id = add_video(db, external_id=f"VIDEO_{index}")
        add_words(db, video_id, "мы все понимаем")
    assert make_plan(db, "мы все понимаем") == make_plan(db, "мы все понимаем")


def test_a_seed_shakes_up_a_tie(db: Database) -> None:
    """Design §8: 'A user-supplied seed shakes up choices among near-equal
    candidates.' Two identical sources: seed 0 always picks the same one,
    and some seeds pick the other."""
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все")
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все")
    unseeded = {
        make_plan(db, "мы все").slots[0].run.video_id  # type: ignore[union-attr]
        for _ in range(3)
    }
    assert unseeded == {first}
    seeded = {
        make_plan(db, "мы все", seed=seed).slots[0].run.video_id  # type: ignore[union-attr]
        for seed in range(1, 25)
    }
    assert seeded == {first, second}


def test_a_seed_cannot_overturn_a_real_preference(db: Database) -> None:
    good = add_video(db, external_id="VIDEO_A")
    add_words(db, good, "мы все", align_score=0.95)
    bad = add_video(db, external_id="VIDEO_B")
    add_words(db, bad, "мы все", align_score=0.20)
    chosen = {
        make_plan(db, "мы все", seed=seed).slots[0].run.video_id  # type: ignore[union-attr]
        for seed in range(1, 25)
    }
    assert chosen == {good}
    assert bad not in chosen


def test_an_empty_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        plan_coverage("   ", (), {}, weights=weights_for(0.0), acoustics={})


def test_zero_padding_changes_nothing(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все")
    assert pad_fragments(db, plan, 0) == plan


def test_padding_extends_the_tail(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=600_000)
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 50)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS + 50


def test_padding_stops_at_the_next_word(db: Database) -> None:
    """Otherwise the fragment quietly says one more word than the target."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все это")
    plan = pad_fragments(db, make_plan(db, "мы все"), 5_000)
    run = plan.slots[0].run
    assert run is not None
    next_start = db.conn.execute(
        "SELECT start_ms FROM words WHERE video_id = ? AND ord = 2", (video_id,)
    ).fetchone()["start_ms"]
    assert run.end_ms == next_start


def test_padding_stops_at_the_end_of_the_video(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=1_000)
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 5_000)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 1_000


def test_padding_is_unclamped_when_nothing_bounds_it(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 250)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS + 250


def test_padding_never_shortens_a_fragment(db: Database) -> None:
    """A clamp below the measured end must not drag the cut backwards.

    Possible whenever a stored duration is shorter than the last word's
    end — a rounded probe, or a re-downloaded rendition.
    """
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=500)
    add_words(db, video_id, "мы все")
    unpadded = make_plan(db, "мы все")
    padded = pad_fragments(db, unpadded, 400)
    assert unpadded.slots[0].run is not None
    assert padded.slots[0].run is not None
    assert unpadded.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS
    assert padded.slots[0].run.end_ms == unpadded.slots[0].run.end_ms


def test_alternatives_are_padded_too(db: Database) -> None:
    """A swapped-in alternative must be usable without re-running."""
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все", align_score=0.95)
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все", align_score=0.80)
    plan = pad_fragments(db, make_plan(db, "мы все", consistency=0.0), 50)
    assert plan.slots[0].alternatives[0].run.end_ms == 2 * WORD_MS + GAP_MS + 50


def test_padding_leaves_gaps_alone(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все зеленеем"), 50)
    assert plan.gaps[0].run is None
    assert plan.gaps[0].text == "зеленеем"


def test_edit_distance_counts_substitutions_and_insertions() -> None:
    assert edit_distance_at_most("дела", "дела", 2) == 0
    assert edit_distance_at_most("дела", "тела", 2) == 1
    assert edit_distance_at_most("дела", "делами", 2) == 2


def test_edit_distance_gives_up_past_the_budget() -> None:
    assert edit_distance_at_most("дела", "неизбежно", 2) is None
    assert edit_distance_at_most("дела", "делами", 1) is None


def test_edit_distance_shortcuts_on_length_alone() -> None:
    assert edit_distance_at_most("а", "аааааааа", 2) is None


@pytest.fixture()
def substitution_corpus(db: Database) -> int:
    """A corpus that says everything near "дела" except "дела"."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело тела неизбежно")
    add_words(db, video_id, "дело делами", start_ms=60_000)
    return video_id


def test_same_stem_is_offered_before_merely_similar_spelling(
    db: Database, substitution_corpus: int
) -> None:
    hits = suggest_substitutions(db, "дела", MatchFilters())
    assert [hit.text for hit in hits] == ["дело", "делами", "тела"]
    assert [hit.reason for hit in hits] == ["stem", "stem", "edit"]


def test_a_substitution_carries_a_cuttable_occurrence(
    db: Database, substitution_corpus: int
) -> None:
    hit = suggest_substitutions(db, "дела", MatchFilters())[0]
    assert isinstance(hit, SubstitutionHit)
    assert hit.run.video_id == substitution_corpus
    assert hit.run.n_words == 1
    assert hit.run.end_ms > hit.run.start_ms
    assert hit.occurrences == 2


def test_nothing_remotely_close_is_offered(db: Database, substitution_corpus: int) -> None:
    offered = {hit.text for hit in suggest_substitutions(db, "дела", MatchFilters())}
    assert offered
    assert "неизбежно" not in offered


def test_the_word_itself_is_never_offered(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дела дело")
    assert "дела" not in {hit.text for hit in suggest_substitutions(db, "дела", MatchFilters())}


def test_caption_words_are_never_offered_as_substitutions(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело", source="caption")
    assert suggest_substitutions(db, "дела", MatchFilters()) == ()


def test_an_excluded_video_offers_nothing(db: Database, substitution_corpus: int) -> None:
    filters = MatchFilters(exclude_video_ids=frozenset({substitution_corpus}))
    assert suggest_substitutions(db, "дела", filters) == ()


def test_substitutions_are_limited(db: Database, substitution_corpus: int) -> None:
    assert len(suggest_substitutions(db, "дела", MatchFilters(), limit=1)) == 1


def test_substitutions_are_deterministic(db: Database, substitution_corpus: int) -> None:
    assert suggest_substitutions(db, "дела", MatchFilters()) == suggest_substitutions(
        db, "дела", MatchFilters()
    )


def test_an_absent_word_names_the_forms_the_corpus_does_have(db: Database) -> None:
    """BUGS.md entry 46: search finds an inflected form by stem and shows a
    hit; assembly refuses, because cutting `каннибализмом` to say
    `каннибализм` puts the wrong word in the video. The refusal is right —
    reporting it as "never said in the corpus" while search displays a hit
    is what made it read as a contradiction."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "нас окружает каннибализмом сегодня", source="timed")

    diagnosis = diagnose_absence(db, "каннибализм", MatchFilters())

    assert diagnosis.total == 0, "the exact form is genuinely absent"
    assert diagnosis.stem_forms, "but a form sharing its stem is present"
    assert diagnosis.stem_forms[0][0] == "каннибализмом"


def test_a_word_absent_in_every_form_names_nothing(db: Database) -> None:
    """The original message is still right when it is right."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "совершенно другие слова", source="timed")

    diagnosis = diagnose_absence(db, "каннибализм", MatchFilters())

    assert diagnosis.total == 0
    assert diagnosis.stem_forms == ()

