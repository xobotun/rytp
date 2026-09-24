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
    dumps_cutlist,
    from_plan,
    load_cutlist,
    read_cutlist,
    validate_name,
    write_cutlist,
)
from rytp.assemble.match import (
    SLOT_FRAGMENT,
    SLOT_GAP,
    MatchFilters,
    Plan,
    SubstitutionHit,
    build_run_table,
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
    "Alternative",
    "AssembleControls",
    "CutList",
    "CutlistError",
    "CutlistParams",
    "MatchFilters",
    "Plan",
    "Slot",
    "Substitution",
    "assemble_target",
    "cutlist_name",
    "cutlist_path",
    "dumps_cutlist",
    "load_cutlist",
    "read_cutlist",
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
    min_align_score: float = C.ASSEMBLE_MIN_ALIGN_SCORE

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
    return MatchFilters(
        exclude_video_ids=frozenset(controls.exclude),
        video_speaker_ids=labels,
        min_align_score=controls.min_align_score,
    )


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
    table = build_run_table(db, tokens, filters)
    videos = {run.video_id for bucket in table.values() for run in bucket.values()}
    plan = plan_coverage(
        target,
        tokens,
        table,
        weights=weights_for(controls.consistency),
        acoustics=load_acoustics(db, videos),
        seed=controls.seed,
    )
    plan = pad_fragments(db, plan, controls.pad_ms)

    substitutions: dict[int, Sequence[SubstitutionHit]] = {
        slot.target_first: suggest_substitutions(db, slot.text, filters)
        for slot in plan.gaps
    }
    used_speakers = {
        slot.run.video_speaker_id
        for slot in plan.fragments
        if slot.run is not None and slot.run.video_speaker_id is not None
    }

    return from_plan(
        plan,
        name=validate_name(name or cutlist_name(target)),
        params=CutlistParams(
            consistency=controls.consistency,
            seed=controls.seed,
            pad_ms=controls.pad_ms,
            speaker=controls.speaker or "",
            exclude=controls.exclude,
            min_align_score=controls.min_align_score,
        ),
        created_at=created_at or utc_now_iso(),
        substitutions=substitutions,
        speaker_labels=speaker_labels(db, used_speakers),
    )
