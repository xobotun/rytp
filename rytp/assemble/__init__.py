"""Assembly: a target sentence in, a cut list out (design §8).

    "You type a sentence. The tool finds real fragments where those words
    were actually said, cuts them out of the source videos, and glues
    them into a single file" — design §1.

This module is the seam between the three below it. :mod:`match` finds
what the corpus can say and chooses a covering, :mod:`score` prices the
choices, :mod:`cutlist` makes the result durable. Nothing here writes to
the database and nothing here enqueues a job: assembly is a read, and
rendering is part 6's.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistError,
    CutlistParams,
    Slot,
    Substitution,
    cutlist_name,
    cutlist_path,
    cutlists_dir,
    dumps_cutlist,
    load_cutlist,
    newest_cutlist_paths,
    read_cutlist,
    slots_from_plan,
    validate_name,
    write_cutlist,
)
from rytp.assemble.match import (
    SLOT_FRAGMENT,
    SLOT_GAP,
    AbsenceDiagnosis,
    MatchFilters,
    Plan,
    SubstitutionHit,
    build_run_table,
    diagnose_absence,
    pad_fragments,
    plan_coverage,
    suggest_substitutions,
    tokenize,
)
from rytp.assemble.score import load_acoustics, weights_for
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, utc_now_iso

__all__ = [
    "SLOT_FRAGMENT",
    "SLOT_GAP",
    "AbsenceDiagnosis",
    "Alternative",
    "AssembleControls",
    "CutList",
    "CutlistError",
    "CutlistParams",
    "MatchFilters",
    "Plan",
    "RetargetResult",
    "Slot",
    "Substitution",
    "assemble_target",
    "cutlist_name",
    "cutlist_path",
    "cutlists_dir",
    "diagnose_absence",
    "dumps_cutlist",
    "gap_note",
    "load_cutlist",
    "newest_cutlist_paths",
    "read_cutlist",
    "retarget_cutlist",
    "speaker_labels",
    "suggest_substitutions",
    "tokenize",
    "validate_name",
    "write_cutlist",
]


@dataclass(frozen=True)
class AssembleControls:
    """Design §8's controls, in one value both surfaces can pass around.

    ``speaker`` defaults to ``None``, not ``""``: ``None`` means "no
    filter was asked for" and is passed straight through to
    ``resolve_speaker_filter`` (contracts §5), which reserves ``""`` — a
    real, if odd, roster label — for something a caller actually typed.
    Collapsing the two would make "not asked" indistinguishable from "an
    empty name was given".
    """

    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY
    seed: int = 0
    pad_ms: int = C.ASSEMBLE_DEFAULT_PAD_MS
    speaker: str | None = None
    exclude: tuple[int, ...] = ()
    #: The *energy*-scale floor only (plan §1a; BUGS.md entries 26, 28).
    #: The ``logprob`` floor is ``C.ASSEMBLE_MIN_ALIGN_BY_SCALE`` and has no
    #: flag this batch — a threshold cannot be written against a scale
    #: that changes per engine, which is the whole reason the scales are
    #: never converted into one another.
    min_align_score: float = C.ASSEMBLE_MIN_ALIGN_SCORE
    #: D1: an override, not a gate. Admits `timed`-tier words alongside
    #: `aligned`, with no threshold and no quality logic.
    allow_timed: bool = False

    def validated(self) -> AssembleControls:
        """Reject an out-of-range control before any work is done."""
        weights_for(self.consistency)  # raises InvalidInputError, names the range
        if not 0 <= self.pad_ms <= C.ASSEMBLE_MAX_PAD_MS:
            raise InvalidInputError(
                f"pad must be between 0 and {C.ASSEMBLE_MAX_PAD_MS} ms, got {self.pad_ms}"
            )
        if not 0.0 <= self.min_align_score <= 1.0:
            raise InvalidInputError(
                f"min align score must be between 0.0 and 1.0, got {self.min_align_score}"
            )
        return replace(self, exclude=tuple(sorted({int(item) for item in self.exclude})))


def speaker_labels(db: Database, video_speaker_ids: Iterable[int]) -> dict[int, str]:
    """Per-video label row id -> roster label, for the ones that have one."""
    wanted = sorted({int(item) for item in video_speaker_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(item) for item in wanted)
    return {
        int(row["id"]): str(row["label"])
        for row in db.conn.execute(
            "SELECT video_speakers.id AS id, speakers.label AS label "
            "FROM video_speakers JOIN speakers ON speakers.id = video_speakers.speaker_id "
            f"WHERE video_speakers.id IN ({inline})"
        )
    }


def _filters(db: Database, controls: AssembleControls) -> MatchFilters:
    """Turn the controls into what the matcher understands.

    ``resolve_speaker_filter`` is imported here, not at module level:
    ``rytp.commands`` registers ``rytp.commands.assemble`` at the bottom
    of its own ``__init__.py``, and that module imports from this one, so
    a module-level import here would form an import cycle depending on
    which package a caller happens to import first. Deferring it to call
    time breaks the cycle without changing which resolver is used —
    still the one shared function contracts §5 pins.
    """
    labels: frozenset[int] | None = None
    if controls.speaker is not None:
        from rytp.commands import resolve_speaker_filter

        # contracts §5: one shared resolver, and it matches speakers.label
        # then speakers.aliases_json. A raw SPEAKER_00 is not accepted here
        # and resolution failure raises, naming the closest roster entries.
        speaker_filter = resolve_speaker_filter(db, speaker=controls.speaker)
        labels = None if speaker_filter is None else speaker_filter.video_speaker_ids
        if not labels:
            raise NotFoundError(
                f"speaker {controls.speaker!r} is not mapped to any video yet, so "
                "nothing can be cut for them; map a diarized video to them first"
            )
    # The energy floor is the one flag-controlled entry; every other scale
    # keeps the fixed floor Task 2a supplied (plan §1a).
    floors: dict[str, float | None] = dict(C.ASSEMBLE_MIN_ALIGN_BY_SCALE)
    floors[C.ALIGN_SCALE_ENERGY] = controls.min_align_score
    return MatchFilters(
        exclude_video_ids=frozenset(controls.exclude),
        video_speaker_ids=labels,
        min_align_by_scale=floors,
        allow_timed=controls.allow_timed,
    )


def _params_from_controls(controls: AssembleControls) -> CutlistParams:
    return CutlistParams(
        consistency=controls.consistency,
        seed=controls.seed,
        pad_ms=controls.pad_ms,
        speaker=controls.speaker or "",
        exclude=controls.exclude,
        min_align_score=controls.min_align_score,
        allow_timed=controls.allow_timed,
    )


def _controls_from_params(params: CutlistParams) -> AssembleControls:
    """The knob positions a cut list was produced with, recovered from its
    own recorded ``[params]`` table.

    A retarget (BUGS.md entries 26/44's "the only remedy throws away every
    hand edit") prices the newly planned words on exactly the terms the rest
    of the cut list was — same consistency weight, same pad, same speaker
    and tier filters — rather than the current defaults, which could differ
    from what produced the file on disk.
    """
    return AssembleControls(
        consistency=params.consistency,
        seed=params.seed,
        pad_ms=params.pad_ms,
        speaker=params.speaker or None,
        exclude=params.exclude,
        min_align_score=params.min_align_score,
        allow_timed=params.allow_timed,
    ).validated()


def _plan_only(
    db: Database,
    target_text: str,
    tokens: Sequence[str],
    controls: AssembleControls,
    filters: MatchFilters,
) -> Plan:
    """The coverage search alone, padded — shared by a whole-sentence plan
    and a retargeted run's replan of just the words that changed."""
    table = build_run_table(db, tokens, filters)
    videos = {run.video_id for bucket in table.values() for run in bucket.values()}
    plan = plan_coverage(
        target_text,
        tokens,
        table,
        weights=weights_for(controls.consistency),
        acoustics=load_acoustics(db, videos),
        seed=controls.seed,
    )
    return pad_fragments(db, plan, controls.pad_ms)


def _dress(
    db: Database,
    plan: Plan,
    controls: AssembleControls,
    filters: MatchFilters,
) -> tuple[tuple[Slot, ...], tuple[str, ...]]:
    """Slots plus one gap note per hole — the part of building a cut list
    that is the same whether the plan covers a whole sentence or just the
    run a retarget replanned."""
    substitutions: dict[int, Sequence[SubstitutionHit]] = {
        slot.target_first: suggest_substitutions(db, slot.text, filters)
        for slot in plan.gaps
    }
    used_speakers = {
        slot.run.video_speaker_id
        for slot in plan.fragments
        if slot.run is not None and slot.run.video_speaker_id is not None
    }
    gap_notes = tuple(
        gap_note(diagnose_absence(db, slot.text, filters), controls.allow_timed)
        for slot in plan.gaps
    )
    slots = slots_from_plan(
        plan, substitutions=substitutions, speaker_labels=speaker_labels(db, used_speakers)
    )
    return slots, gap_notes


def assemble_target(
    db: Database,
    target: str,
    *,
    name: str | None = None,
    controls: AssembleControls | None = None,
    created_at: str | None = None,
) -> CutList:
    """Find real fragments that say ``target``, and return the cut list.

    ``created_at`` is a parameter rather than a call to the clock so that
    design §8's "same input gives the same output" is something a test
    can check and a re-run can reproduce.
    """
    controls = (controls or AssembleControls()).validated()
    tokens = tokenize(target)
    if not tokens:
        raise InvalidInputError(f"the target {target!r} is empty once normalized")
    if len(tokens) > C.ASSEMBLE_MAX_TARGET_WORDS:
        raise InvalidInputError(
            f"the target is {len(tokens)} words; the limit is "
            f"{C.ASSEMBLE_MAX_TARGET_WORDS} words — assemble a sentence at a time"
        )

    filters = _filters(db, controls)
    plan = _plan_only(db, target, tokens, controls, filters)
    slots, gap_notes = _dress(db, plan, controls, filters)

    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name=validate_name(name or cutlist_name(target)),
        target=target,
        created_at=created_at or utc_now_iso(),
        params=_params_from_controls(controls),
        slots=slots,
        gap_notes=gap_notes,
    )


@dataclass(frozen=True)
class RetargetResult:
    """What replanning only the changed words produced.

    ``kept``/``replanned`` count *slots*, not target words, so the count a
    command prints reads the same way `assemble plan`'s own summary does.
    """

    cutlist: CutList
    kept: int
    replanned: int
    gap_notes: tuple[str, ...] = ()


def _validate_retargetable(slots: Sequence[Slot], token_count: int, where: str) -> None:
    """A retarget needs ``target_first``/``target_last`` to still be a
    clean, ordered partition of the recorded target's tokens — the
    invariant :func:`rytp.assemble.match.plan_coverage` always produces and
    that a hand-edit could, in principle, break (renumbering a slot,
    duplicating a range). Without it there is no sound way to map an old
    target position onto a new one, so this refuses rather than guess.
    """
    expected = 0
    for slot in slots:
        if slot.target_last < slot.target_first or slot.target_first != expected:
            raise InvalidInputError(
                f"{where}: its slots no longer partition the target's words in "
                f"order (expected target word {expected} next, a slot claims "
                f"{slot.target_first}-{slot.target_last}) — re-plan it from "
                "scratch instead of retargeting"
            )
        expected = slot.target_last + 1
    if expected != token_count:
        raise InvalidInputError(
            f"{where}: its slots cover {expected} target word(s) but its "
            f"recorded target has {token_count} — re-plan it from scratch "
            "instead of retargeting"
        )


def _kept_shift(
    equal_spans: Sequence[tuple[int, int, int, int]], first: int, last: int
) -> int | None:
    """The constant new-minus-old offset for a token span, if the *whole*
    span sits inside one ``difflib`` "equal" opcode — ``None`` otherwise.

    A span straddling two opcodes (part kept, part changed) cannot be kept
    piecemeal: a fragment slot is one clip with one pair of timestamps, so
    half of it being untouched by the edit does not mean half of it can be
    preserved while the other half is replanned. The whole slot goes back
    through the planner in that case.
    """
    for i1, i2, j1, _j2 in equal_spans:
        if i1 <= first and last < i2:
            return j1 - i1
    return None


def retarget_cutlist(db: Database, cutlist: CutList, new_target: str) -> RetargetResult:
    """Edit ``cutlist``'s target sentence, replanning only the words that changed.

    The owner's ask (docs/superpowers/.../"CLAUDE.md" for this task; BUGS.md
    entry 46's ``каннибализм``): today the only way to fix a target word
    with no cuttable form, or to add or drop a word, is `assemble plan`
    with a new sentence — which throws away every hand edit (nudged
    boundaries, snapped seams, swapped alternatives, adopted
    substitutions) on every slot, not just the one that changed.

    **The algorithm.** ``difflib.SequenceMatcher`` diffs the old target's
    tokens against the new one's. A slot is *kept, renumbered* exactly when
    its whole ``target_first..target_last`` span lies inside one "equal"
    opcode — then it is carried over byte-for-byte except for the shifted
    target positions. Every other stretch of the new target — an edited
    word, an insertion, the tail after a deletion — is a contiguous run of
    new-target tokens with no kept slot covering it, and *that* run alone is
    planned for real, through the same matcher/scorer `assemble_target`
    uses, dressed with the cut list's own recorded params
    (:func:`_controls_from_params`) so new material is priced on the same
    terms as the old.

    **The cohesion trade-off, stated rather than buried.** `plan_coverage`'s
    consistency weighting prefers drawing from few sources across the
    *whole* target it is given — that is what "few seams" means. Planning
    an inserted run in isolation cannot see the fragments on either side of
    it, so a word added to an existing sentence may come from a source the
    original whole-sentence plan would have avoided, or add a source where
    the original had none. This function does not widen the window to
    compensate — the changed run is planned alone, cohesion narrowed to
    that run only — because widening it would mean re-planning (and
    silently discarding hand edits on) slots the edit never touched, which
    is exactly what this exists to stop doing. A user chasing maximum
    cohesion after a big edit still has `assemble plan` with `--force`
    available; this trades some of that for keeping the rest of the file
    exactly as hand-tuned.

    Raises :class:`InvalidInputError` if the cut list's slots no longer
    cleanly partition its recorded target (see
    :func:`_validate_retargetable`) or if the new target is empty or too
    long — the same limits `assemble_target` enforces.
    """
    controls = _controls_from_params(cutlist.params)
    new_tokens = tokenize(new_target)
    if not new_tokens:
        raise InvalidInputError(f"the target {new_target!r} is empty once normalized")
    if len(new_tokens) > C.ASSEMBLE_MAX_TARGET_WORDS:
        raise InvalidInputError(
            f"the target is {len(new_tokens)} words; the limit is "
            f"{C.ASSEMBLE_MAX_TARGET_WORDS} words — assemble a sentence at a time"
        )

    old_tokens = tokenize(cutlist.target)
    _validate_retargetable(cutlist.slots, len(old_tokens), f"cut list {cutlist.name!r}")

    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    equal_spans = tuple(
        (i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == "equal"
    )

    kept: list[tuple[int, int, Slot]] = []
    for slot in cutlist.slots:
        shift = _kept_shift(equal_spans, slot.target_first, slot.target_last)
        if shift is not None:
            kept.append((slot.target_first + shift, slot.target_last + shift, slot))
    kept.sort(key=lambda item: item[0])

    filters = _filters(db, controls)
    new_slots: list[Slot] = []
    gap_notes: list[str] = []
    replanned = 0
    pos = 0
    # A trailing `None` slot is a sentinel forcing the loop to also cover
    # the run, if any, between the last kept slot and the end of the
    # target — there is otherwise nothing left in `kept` to trigger it.
    sentinel: tuple[int, int, Slot | None] = (len(new_tokens), len(new_tokens), None)
    for new_first, new_last, kept_slot in (*kept, sentinel):
        if pos < new_first:
            run_tokens = new_tokens[pos:new_first]
            plan = _plan_only(db, " ".join(run_tokens), run_tokens, controls, filters)
            run_slots, run_notes = _dress(db, plan, controls, filters)
            new_slots.extend(
                replace(
                    slot_,
                    target_first=slot_.target_first + pos,
                    target_last=slot_.target_last + pos,
                )
                for slot_ in run_slots
            )
            gap_notes.extend(run_notes)
            replanned += len(run_slots)
        if kept_slot is not None:
            new_slots.append(replace(kept_slot, target_first=new_first, target_last=new_last))
            pos = new_last + 1

    return RetargetResult(
        cutlist=replace(
            cutlist, target=new_target, slots=tuple(new_slots), gap_notes=tuple(gap_notes)
        ),
        kept=len(kept),
        replanned=replanned,
        gap_notes=tuple(gap_notes),
    )


def gap_note(diagnosis: AbsenceDiagnosis, allow_timed: bool) -> str:
    """One line saying why a target word matched nothing (BUGS.md entry 26).

    The acceptance criterion this exists for: "a user must never again
    read 'no fragments' and conclude the corpus does not contain the
    phrase." Names the count for each exclusion reason that actually
    contributed, and the fix for the common ones.
    """
    if diagnosis.total == 0:
        if diagnosis.stem_forms:
            forms = ", ".join(
                f"{form} ({n})" for form, n in diagnosis.stem_forms
            )
            return (
                f"{diagnosis.token!r}: not said in this exact form; the corpus "
                f"has {forms} — `assemble suggest {diagnosis.token}` ranks "
                f"stand-ins"
            )
        return f"{diagnosis.token!r}: never said in the corpus"
    parts = [f"{diagnosis.total} occurrence{'' if diagnosis.total == 1 else 's'}"]
    if diagnosis.excluded_tier:
        hint = "" if allow_timed else " (see --allow-timed)"
        parts.append(f"{diagnosis.excluded_tier} not in a cuttable tier{hint}")
    if diagnosis.excluded_unknown_scale:
        parts.append(
            f"{diagnosis.excluded_unknown_scale} of unknown alignment scale "
            "(re-run `transcribe align`)"
        )
    if diagnosis.excluded_below_floor:
        parts.append(f"{diagnosis.excluded_below_floor} below their scale's floor")
    return f"{diagnosis.token!r}: " + ", ".join(parts)
