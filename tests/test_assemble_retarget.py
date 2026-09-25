"""Editing a cut list's target sentence without discarding hand work.

The owner's ask (BUGS.md entry 46's `каннибализм`): today the only remedy
for a target word with no cuttable form, or for wanting to add or drop a
word, is `assemble plan` with a new sentence — which throws away every
hand edit on every slot, not just the one the edit actually touches.
`retarget_cutlist` replans only the words that changed, using
`difflib.SequenceMatcher` to tell "kept" runs from "changed" ones.
"""

from __future__ import annotations

import dataclasses

import pytest

from rytp import constants as C
from rytp.assemble import RetargetResult, assemble_target, retarget_cutlist
from rytp.db import Database
from rytp.models import InvalidInputError
from tests.assembly_corpus import add_video, add_words


@pytest.fixture()
def corpus(db: Database) -> None:
    """Two videos, neither saying the whole target alone — so a plan of
    "мы все понимаем это неизбежно" needs two fragments, one per video,
    which is what makes "keep one, replan the other" checkable."""
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "это неизбежно")


def planned(db: Database) -> object:
    return assemble_target(db, "мы все понимаем это неизбежно", name="demo")


# -- hand edits survive -------------------------------------------------


def test_a_hand_nudged_slot_that_the_edit_never_touches_comes_back_byte_identical(
    db: Database, corpus: None
) -> None:
    """Add a word to the *other* fragment's span; the untouched fragment's
    hand-tuned timings must be carried over exactly, not re-derived — the
    format's header says timings are "authoritative and never recomputed",
    and this is exactly why."""
    cutlist = planned(db)
    assert [s.target_first for s in cutlist.slots] == [0, 3]  # two fragments

    # A nudge, the way the TUI's `[`/`]` would make one: hand-edit the
    # first fragment's boundaries to something the planner would never
    # have produced on its own.
    nudged = dataclasses.replace(
        cutlist,
        slots=(
            dataclasses.replace(cutlist.slots[0], start_ms=12_345, end_ms=67_890),
            cutlist.slots[1],
        ),
    )

    result = retarget_cutlist(db, nudged, "мы все понимаем это точно неизбежно")
    assert isinstance(result, RetargetResult)
    assert result.kept == 1

    kept_slot = result.cutlist.slots[0]
    assert (kept_slot.start_ms, kept_slot.end_ms) == (12_345, 67_890)
    assert kept_slot.video_id == nudged.slots[0].video_id
    assert kept_slot.target_first == 0 and kept_slot.target_last == 2  # unshifted

    # The new word gets its own slot: a gap, because "точно" was never said.
    new_word_slots = [s for s in result.cutlist.slots if s.target_first >= 3]
    assert any(s.kind == "gap" and s.text == "точно" for s in new_word_slots)


def test_adding_a_word_after_every_kept_slot_only_creates_a_new_slot(
    db: Database, corpus: None
) -> None:
    cutlist = planned(db)
    result = retarget_cutlist(db, cutlist, cutlist.target + " точно")
    assert result.kept == 2
    assert result.replanned == 1
    tail = result.cutlist.slots[-1]
    assert tail.kind == "gap"
    assert tail.text == "точно"
    assert tail.target_first == 5 and tail.target_last == 5
    # The two original fragments are otherwise untouched.
    assert result.cutlist.slots[0] == dataclasses.replace(
        cutlist.slots[0], target_first=0, target_last=2
    )
    assert result.cutlist.slots[1] == dataclasses.replace(
        cutlist.slots[1], target_first=3, target_last=4
    )


def test_removing_a_word_shifts_the_later_slot_back(db: Database, corpus: None) -> None:
    cutlist = planned(db)
    result = retarget_cutlist(db, cutlist, "мы все понимаем неизбежно")
    # "это" removed: the first fragment ("мы все понимаем") is unaffected,
    # the second ("это неизбежно") straddles the deletion and is replanned,
    # landing as a single fragment again once "это" drops out of it.
    assert result.cutlist.slots[0].target_first == 0
    assert result.cutlist.slots[0].target_last == 2
    remaining = [s for s in result.cutlist.slots if s.target_first == 3]
    assert len(remaining) == 1
    assert remaining[0].text == "неизбежно"


def test_removing_the_middle_of_a_kept_fragment_replans_the_whole_fragment(
    db: Database, corpus: None
) -> None:
    """A fragment is one clip with one pair of timestamps: dropping "все"
    out of "мы все понимаем" cannot keep half of that fragment, so the
    whole thing goes back through the planner."""
    cutlist = planned(db)
    result = retarget_cutlist(db, cutlist, "мы понимаем это неизбежно")
    assert result.kept == 1  # only "это неизбежно" survives untouched
    first_two = [s for s in result.cutlist.slots if s.target_first < 2]
    assert [s.text for s in first_two] == ["мы", "понимаем"]


# -- edge cases named in the brief --------------------------------------


def test_retargeting_to_an_unchanged_target_keeps_every_slot(db: Database, corpus: None) -> None:
    cutlist = planned(db)
    result = retarget_cutlist(db, cutlist, cutlist.target)
    assert result.kept == len(cutlist.slots)
    assert result.replanned == 0
    assert result.cutlist.slots == cutlist.slots


def test_retargeting_to_an_empty_target_raises(db: Database, corpus: None) -> None:
    cutlist = planned(db)
    with pytest.raises(InvalidInputError, match="empty"):
        retarget_cutlist(db, cutlist, "   ")


def test_retargeting_past_the_word_limit_raises(db: Database, corpus: None) -> None:
    cutlist = planned(db)
    too_long = " ".join(f"слово{i}" for i in range(C.ASSEMBLE_MAX_TARGET_WORDS + 1))
    with pytest.raises(InvalidInputError, match="limit"):
        retarget_cutlist(db, cutlist, too_long)


def test_a_cut_list_whose_slots_no_longer_partition_the_target_refuses(
    db: Database, corpus: None
) -> None:
    """A hand edit could in principle renumber a slot into nonsense; there
    is then no sound way to map an old target position onto a new one, so
    this refuses rather than silently mis-map."""
    cutlist = planned(db)
    broken = dataclasses.replace(
        cutlist,
        slots=(
            dataclasses.replace(cutlist.slots[0], target_first=0, target_last=0),
            cutlist.slots[1],
        ),
    )
    with pytest.raises(InvalidInputError, match="partition"):
        retarget_cutlist(db, broken, cutlist.target + " ещё")


def test_gap_notes_report_a_replanned_word_still_missing(db: Database, corpus: None) -> None:
    cutlist = planned(db)
    result = retarget_cutlist(db, cutlist, cutlist.target + " каннибализм")
    assert result.gap_notes
    assert "каннибализм" in result.gap_notes[0]
