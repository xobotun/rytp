"""Walk a target sentence against the corpus (design §7 "Assembly matching").

The corpus read is two SQL shapes and nothing else. ``words(normalized_text)``
finds where a token was said; ``words(video_id, ord)`` steps forward from
there. No n-gram table, no FTS, no fuzziness: an assembly match is exact or
it is not a match.

Extension is level-synchronous. Every candidate that survived the previous
token advances by one ordinal in one batched query, so the number of
queries is bounded by the length of the target rather than by how many
times its first word was ever said. A run is recorded at every prefix
length, because the coverage search below needs the short edges as well as
the long ones.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from rytp import constants as C
from rytp.assemble.score import (
    AcousticsMap,
    Weights,
    fragment_cost,
    jitter,
    rounded,
    transition_cost,
)
from rytp.db import Database
from rytp.models import InvalidInputError, normalize_text, stem_text

__all__ = [
    "SLOT_FRAGMENT",
    "SLOT_GAP",
    "AbsenceDiagnosis",
    "CandidateRun",
    "MatchFilters",
    "Plan",
    "PlanSlot",
    "RunTable",
    "ScoredRun",
    "SubstitutionHit",
    "WordRow",
    "build_run_table",
    "diagnose_absence",
    "edit_distance",
    "edit_distance_at_most",
    "find_occurrences",
    "occurrence_query",
    "pad_fragments",
    "plan_coverage",
    "suggest_substitutions",
    "tokenize",
]


@dataclass(frozen=True)
class WordRow:
    """One cuttable word. ``align_score`` is already coalesced, never None."""

    video_id: int
    ord: int
    start_ms: int
    end_ms: int
    text: str
    normalized_text: str
    align_score: float
    video_speaker_id: int | None
    source: str


@dataclass(frozen=True)
class CandidateRun:
    """A contiguous run of cuttable words in one video, matching the target."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    n_words: int
    first_align: float
    last_align: float
    mean_align: float
    video_speaker_id: int | None
    #: The weakest ``words.source`` any word in this run carries (plan
    #: §1a / D1). ``WORD_SOURCE_RANK`` orders tiers weakest first for
    #: exactly this: a run is only as trustworthy as its worst word, so a
    #: fragment that mixes an ``aligned`` word with a ``timed`` one (only
    #: possible under ``--allow-timed``) must still be reported as
    #: ``timed`` rather than silently look fully aligned.
    tier: str


def _default_min_align_by_scale() -> dict[str, float | None]:
    return dict(C.ASSEMBLE_MIN_ALIGN_BY_SCALE)


@dataclass(frozen=True)
class MatchFilters:
    """Everything that narrows what may be cut (design §8 "Controls")."""

    exclude_video_ids: frozenset[int] = frozenset()
    video_speaker_ids: frozenset[int] | None = None
    #: Per-``align_scale`` floor (plan §1a; BUGS.md entries 26, 28). ``None``
    #: for a scale means no threshold applies to it. ``unknown`` is never
    #: consulted here regardless of what it maps to —
    #: ``C.ASSEMBLE_EXCLUDE_UNKNOWN_SCALE`` excludes it outright.
    min_align_by_scale: Mapping[str, float | None] = field(
        default_factory=_default_min_align_by_scale
    )
    #: D1: an override, not a gate. Admits ``timed`` words with no
    #: threshold and no quality logic, alongside the always-eligible
    #: ``aligned`` tier.
    allow_timed: bool = False
    max_internal_gap_ms: int = C.ASSEMBLE_MAX_INTERNAL_GAP_MS
    max_run_words: int = C.ASSEMBLE_MAX_RUN_WORDS
    max_occurrences: int = C.ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN
    max_seeds_per_video: int = C.ASSEMBLE_MAX_SEEDS_PER_VIDEO


#: (target position, run length in words) -> video_id -> the best run there.
RunTable = dict[tuple[int, int], dict[int, CandidateRun]]

_SELECT = (
    "SELECT video_id, ord, start_ms, end_ms, text, normalized_text, "
    "COALESCE(align_score, :default_align) AS align_score, video_speaker_id, source "
)


def tokenize(target: str) -> tuple[str, ...]:
    """The target as the tokens the corpus stores, in order.

    Uses the same normalizer the ``normalized_text`` column was written
    with, so a match is a string equality and never a guess.
    """
    return tuple(normalize_text(target).split())


def _int_list(values: Iterable[int]) -> str:
    """Inline a set of integers into SQL.

    Safe because every element is coerced with ``int()`` first, and
    necessary because the exclusion list has no fixed arity and would
    otherwise need a parameter per element on every query.
    """
    return ", ".join(str(int(value)) for value in values)


def _scale_clause(min_align_by_scale: Mapping[str, float | None]) -> str:
    """Eligibility by score, scoped to the scale that produced it (plan §1a).

    A disjunction over the declared scales: a row is eligible when its
    ``align_scale`` has a floor and its ``align_score`` clears it, or when
    its scale has no floor at all (``none`` — MFA reports no score, and
    contracts §3 permits that) and so nothing gates it. ``unknown`` is
    never a disjunct here, regardless of what the mapping says about it —
    ``C.ASSEMBLE_EXCLUDE_UNKNOWN_SCALE`` rows are entry 36's fabricated
    boundaries and entry 28's un-identifiable sign flips, excluded
    outright rather than floored (BUGS.md entries 26, 28).

    The scale names and floors are Python constants, never user input, so
    embedding them as SQL literals is the same trade the surrounding code
    already makes for ``source = 'aligned'``.
    """
    disjuncts: list[str] = []
    for scale, floor in min_align_by_scale.items():
        if C.ASSEMBLE_EXCLUDE_UNKNOWN_SCALE and scale == C.ALIGN_SCALE_UNKNOWN:
            continue
        if floor is None:
            disjuncts.append(f"align_scale = '{scale}'")
        else:
            disjuncts.append(f"(align_scale = '{scale}' AND align_score >= {float(floor)!r})")
    return "(" + " OR ".join(disjuncts) + ")" if disjuncts else "1 = 0"


def _shared_clauses(filters: MatchFilters) -> list[str]:
    """Scope narrowing that applies regardless of tier: excluded videos,
    a speaker filter. Factored out so both the per-tier ``UNION ALL``
    branches and the pair-keyed extension query say it once."""
    clauses: list[str] = []
    if filters.exclude_video_ids:
        clauses.append(f"video_id NOT IN ({_int_list(filters.exclude_video_ids)})")
    if filters.video_speaker_ids is not None:
        # An empty set means "a real person, mapped to no video yet". It must
        # match nothing. SQLite does accept an empty `IN ()` as false, but that
        # is a non-standard extension and contracts §5 calls this exact case
        # "the classic silent-wrong-answer bug in a SQL IN clause" — so say it
        # outright rather than lean on a dialect quirk.
        if not filters.video_speaker_ids:
            clauses.append("1 = 0")
        else:
            clauses.append(f"video_speaker_id IN ({_int_list(filters.video_speaker_ids)})")
    return clauses


def _tier_clause(source: str, filters: MatchFilters) -> str:
    """One tier's full eligibility, as one clause: cuttable, in scope,
    well enough anchored. ``end_ms IS NOT NULL`` is implied by the table's
    CHECK constraint but stated anyway, because a cut without an end is
    not a cut.

    D1: ``timed`` carries no score floor at all — it is an override, not a
    gate, with "no threshold and no quality logic" by design, so only the
    ``aligned`` tier ever consults :func:`_scale_clause`.
    """
    if source == C.ALIGNED_WORD_SOURCE:
        clauses = [
            "source = 'aligned'",
            "end_ms IS NOT NULL",
            _scale_clause(filters.min_align_by_scale),
        ]
    else:
        clauses = ["source = 'timed'", "end_ms IS NOT NULL"]
    clauses.extend(_shared_clauses(filters))
    return " AND ".join(clauses)


def _eligibility_sql(filters: MatchFilters) -> str:
    """The full WHERE tail for a pair-keyed lookup (extension steps).

    Expressed as an ``OR`` of per-tier clauses rather than the ``UNION
    ALL`` :func:`occurrence_query` uses: this query already selects by an
    explicit ``(video_id, ord) IN (VALUES …)`` list, served entirely by
    the composite ``words(video_id, ord)`` index — there is no partial
    index on ``source`` for this shape to preserve (plan Task 6, "how to
    implement D1").
    """
    tiers = [_tier_clause(C.ALIGNED_WORD_SOURCE, filters)]
    if filters.allow_timed:
        tiers.append(_tier_clause("timed", filters))
    return "(" + " OR ".join(f"({tier})" for tier in tiers) + ")"


def _row(record: sqlite3.Row) -> WordRow:
    """One result row, with every value given its real type once."""
    speaker = record["video_speaker_id"]
    return WordRow(
        video_id=int(record["video_id"]),
        ord=int(record["ord"]),
        start_ms=int(record["start_ms"]),
        end_ms=int(record["end_ms"]),
        text=str(record["text"]),
        normalized_text=str(record["normalized_text"]),
        align_score=float(record["align_score"]),
        video_speaker_id=None if speaker is None else int(speaker),
        source=str(record["source"]),
    )


def occurrence_query(
    filters: MatchFilters, token: str = ""
) -> tuple[str, dict[str, object]]:
    """The hot lookup, as SQL and parameters.

    Separated from :func:`find_occurrences` so a test can run
    ``EXPLAIN QUERY PLAN`` over the real statement. By default the inner
    ``WHERE`` is written to sit directly on contracts §3's partial index
    ``words_alignable ON words(normalized_text) WHERE source = 'aligned'``:
    most of the corpus is caption-tier (design §6), so a plan that matched
    the token first and filtered by tier afterwards would scale with the
    whole archive instead of the cuttable part of it.

    Under ``--allow-timed`` the query becomes a ``UNION ALL`` of two
    per-tier ``SELECT``s, one per source literal, each able to sit on its
    own partial index (``words_alignable`` / ``words_timed``). A single
    query using ``source IN ('aligned', 'timed')`` would imply neither
    index's predicate, and SQLite would fall back to scanning
    ``words_normalized`` — precisely the caption-tier scan the partial
    indexes exist to avoid (plan Task 6, "how to implement D1"). The
    window function ranking seeds per video is applied *outside* that
    union, over the combined rows, so ``max_seeds_per_video`` still caps
    per video rather than per tier.
    """
    tiers = [C.ALIGNED_WORD_SOURCE]
    if filters.allow_timed:
        tiers.append("timed")
    if len(tiers) == 1:
        # The default shape, unchanged from before per-scale eligibility:
        # one SELECT directly off `words`, so it sits on `words_alignable`
        # exactly as it always has.
        source = f"words WHERE normalized_text = :token AND {_tier_clause(tiers[0], filters)}"
    else:
        # `--allow-timed`: a UNION ALL of two per-tier SELECTs, each with
        # its own `source = '...'` literal, so each half can sit on its
        # own partial index. `source IN ('aligned', 'timed')` would imply
        # neither index's predicate and fall back to scanning
        # `words_normalized` (plan Task 6, "how to implement D1").
        branches = " UNION ALL ".join(
            f"SELECT * FROM words WHERE normalized_text = :token AND {_tier_clause(tier, filters)}"
            for tier in tiers
        )
        source = f"({branches})"
    sql = (
        f"{_SELECT}FROM (SELECT *, ROW_NUMBER() OVER ("
        "  PARTITION BY video_id"
        "  ORDER BY COALESCE(align_score, :default_align) DESC, ord"
        f") AS seed_rank FROM {source}"
        ") WHERE seed_rank <= :max_seeds"
        " ORDER BY align_score DESC, video_id, ord"
        " LIMIT :max_occurrences"
    )
    return sql, {
        "token": token,
        "default_align": C.ASSEMBLE_DEFAULT_ALIGN_SCORE,
        "max_seeds": filters.max_seeds_per_video,
        "max_occurrences": filters.max_occurrences,
    }


def find_occurrences(db: Database, token: str, filters: MatchFilters) -> list[WordRow]:
    """Every eligible place ``token`` was said, best-anchored first.

    Two caps apply, and the per-video one is applied *inside* the query.
    Applying it afterwards would be useless: a single talkative video
    would fill the overall limit before any other source was seen, and
    the consistency knob would then have nothing to choose between.
    """
    sql, params = occurrence_query(filters, token)
    return [_row(record) for record in db.conn.execute(sql, params)]


def _fetch_next(
    db: Database, wanted: Sequence[tuple[int, int]], filters: MatchFilters
) -> dict[tuple[int, int], WordRow]:
    """One batched read of the ``(video_id, ord)`` pairs the frontier needs.

    SQLite has supported row values in ``IN (VALUES …)`` since 3.15, and
    3.11 bundles far newer; the composite ``words(video_id, ord)`` index
    serves it directly. Batched because the oldest host-parameter limit
    is 999 and each pair spends two.
    """
    found: dict[tuple[int, int], WordRow] = {}
    tail = _eligibility_sql(filters)
    for start in range(0, len(wanted), C.ASSEMBLE_SQL_BATCH):
        batch = wanted[start : start + C.ASSEMBLE_SQL_BATCH]
        pairs = ", ".join(f"(:v{index}, :o{index})" for index in range(len(batch)))
        params: dict[str, object] = {"default_align": C.ASSEMBLE_DEFAULT_ALIGN_SCORE}
        for index, (video_id, ordinal) in enumerate(batch):
            params[f"v{index}"] = video_id
            params[f"o{index}"] = ordinal
        sql = f"{_SELECT}FROM words WHERE (video_id, ord) IN (VALUES {pairs}) AND {tail}"
        for record in db.conn.execute(sql, params):
            row = _row(record)
            found[(row.video_id, row.ord)] = row
    return found


def _weakest_tier(words: Sequence[WordRow]) -> str:
    """The weakest ``source`` any word in ``words`` carries.

    ``WORD_SOURCE_RANK`` orders tiers weakest first for exactly this
    purpose (its own docstring: "the order is what lets a run report the
    weakest tier it contains"). Every word here is already ``aligned`` or
    ``timed`` — a caption-tier row is never eligible — so a rank lookup
    that saw one is a bug worth failing loudly on rather than guessing.
    """
    return min(words, key=lambda word: C.WORD_SOURCE_RANK.index(word.source)).source


def _run_from(words: Sequence[WordRow]) -> CandidateRun:
    """Freeze a list of consecutive words into the run they form."""
    scores = [word.align_score for word in words]
    return CandidateRun(
        video_id=words[0].video_id,
        first_word_ord=words[0].ord,
        last_word_ord=words[-1].ord,
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        text=" ".join(word.text for word in words),
        n_words=len(words),
        first_align=words[0].align_score,
        last_align=words[-1].align_score,
        mean_align=sum(scores) / len(scores),
        video_speaker_id=words[0].video_speaker_id,
        tier=_weakest_tier(words),
    )


def _better(candidate: CandidateRun, incumbent: CandidateRun) -> bool:
    """Which of two runs of the same shape in the same video to keep.

    Two runs of equal length in the same video differ in cost only
    through their alignment terms (see :mod:`rytp.assemble.score`), so
    ranking by alignment here is ranking by cost — without this module
    needing to know the knob. The ordinal breaks the last tie so the
    answer never depends on row order.
    """
    key_new = (
        -candidate.mean_align,
        -(candidate.first_align + candidate.last_align),
        candidate.first_word_ord,
    )
    key_old = (
        -incumbent.mean_align,
        -(incumbent.first_align + incumbent.last_align),
        incumbent.first_word_ord,
    )
    return key_new < key_old


def _record(table: RunTable, position: int, run: CandidateRun) -> None:
    bucket = table.setdefault((position, run.n_words), {})
    incumbent = bucket.get(run.video_id)
    if incumbent is None or _better(run, incumbent):
        bucket[run.video_id] = run


def _extends(previous: WordRow, nxt: WordRow, filters: MatchFilters) -> bool:
    """May a run step from ``previous`` to ``nxt``?

    Two reasons to stop that ordinal adjacency alone does not catch: a
    silence long enough that the cut would carry a pause the target does
    not have, and a change of voice mid-fragment, which is wrong however
    the run is filtered.
    """
    if nxt.start_ms - previous.end_ms > filters.max_internal_gap_ms:
        return False
    return not (
        previous.video_speaker_id is not None
        and nxt.video_speaker_id is not None
        and previous.video_speaker_id != nxt.video_speaker_id
    )


def build_run_table(db: Database, tokens: Sequence[str], filters: MatchFilters) -> RunTable:
    """Every contiguous corpus run that matches the target, at every prefix.

    Occurrence lists are cached per token, so a target that repeats a word
    reads the corpus for it once.
    """
    table: RunTable = {}
    max_words = max(1, filters.max_run_words)
    seeds: dict[str, list[WordRow]] = {}
    for position, token in enumerate(tokens):
        if token not in seeds:
            seeds[token] = find_occurrences(db, token, filters)
        alive: list[list[WordRow]] = []
        for occurrence in seeds[token]:
            alive.append([occurrence])
            _record(table, position, _run_from([occurrence]))
        step = 1
        while alive and step < max_words and position + step < len(tokens):
            expected = tokens[position + step]
            wanted = [(run[-1].video_id, run[-1].ord + 1) for run in alive]
            found = _fetch_next(db, wanted, filters)
            grown: list[list[WordRow]] = []
            for run in alive:
                nxt = found.get((run[-1].video_id, run[-1].ord + 1))
                if nxt is None or nxt.normalized_text != expected:
                    continue
                if not _extends(run[-1], nxt, filters):
                    continue
                longer = [*run, nxt]
                _record(table, position, _run_from(longer))
                grown.append(longer)
            alive = grown
            step += 1
    return table


#: Slot kinds. A cut list is one ordered list of these, fragments and
#: holes together, because the renderer has to walk the timeline in order.
SLOT_FRAGMENT: Final = "fragment"
SLOT_GAP: Final = "gap"

#: DP state for "no fragment emitted yet". An int, not None, so the state
#: keys stay sortable and iteration order stays deterministic.
_NO_SOURCE: Final = -1


@dataclass(frozen=True)
class ScoredRun:
    """A candidate the search did not take, and what it would have cost."""

    run: CandidateRun
    cost: float


@dataclass(frozen=True)
class PlanSlot:
    """One position in the output: a fragment, or a hole where a word was."""

    kind: str
    target_first: int
    target_last: int  # inclusive
    text: str
    cost: float
    run: CandidateRun | None = None
    alternatives: tuple[ScoredRun, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A whole target, covered."""

    target: str
    tokens: tuple[str, ...]
    slots: tuple[PlanSlot, ...]
    total_cost: float

    @property
    def fragments(self) -> tuple[PlanSlot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_FRAGMENT)

    @property
    def gaps(self) -> tuple[PlanSlot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_GAP)

    @property
    def source_video_ids(self) -> tuple[int, ...]:
        """Distinct source videos, in the order they are first used."""
        seen: list[int] = []
        for slot in self.fragments:
            if slot.run is not None and slot.run.video_id not in seen:
                seen.append(slot.run.video_id)
        return tuple(seen)

    @property
    def n_sources(self) -> int:
        return len(self.source_video_ids)


@dataclass(frozen=True)
class _Step:
    """How the search arrived at one DP node."""

    cost: float
    prev_position: int
    prev_source: int
    run: CandidateRun | None


def _step_key(step: _Step) -> tuple[float, int, int, int, int]:
    """A total order over arrivals at one node, so ties never depend on luck.

    Cost is rounded first: two candidates the cost model considers
    equivalent must actually tie, or float noise decides instead of the
    tie-break — and design §8 requires the same input to give the same
    output.
    """
    run = step.run
    return (
        rounded(step.cost),
        0 if run is not None else 1,
        run.video_id if run is not None else -1,
        run.first_word_ord if run is not None else -1,
        step.prev_position,
    )


def _relax(node: dict[int, _Step], source: int, step: _Step) -> None:
    incumbent = node.get(source)
    if incumbent is None or _step_key(step) < _step_key(incumbent):
        node[source] = step


def _arrival_costs(
    node: Sequence[tuple[int, _Step]],
    candidates: Iterable[int],
    weights: Weights,
    acoustics: AcousticsMap,
) -> dict[int, tuple[float, int]]:
    """For each candidate source, the cheapest way to arrive in it.

    The transition term depends on the pair of videos and not on how long
    the fragment is, so it is computed once per (previous source, next
    source) pair rather than once per edge. That is what turns the inner
    loop from O(N·L·V²) into O(N·(V² + L·V)).
    """
    arrivals: dict[int, tuple[float, int]] = {}
    for video_id in sorted(candidates):
        best_cost = math.inf
        best_source = _NO_SOURCE
        for source, step in node:
            previous = None if source == _NO_SOURCE else source
            cost = step.cost + transition_cost(previous, video_id, weights, acoustics)
            if (rounded(cost), source) < (rounded(best_cost), best_source):
                best_cost, best_source = cost, source
        arrivals[video_id] = (best_cost, best_source)
    return arrivals


def _alternatives(
    table: RunTable,
    position: int,
    length: int,
    chosen: _Step,
    weights: Weights,
    acoustics: AcousticsMap,
    seed: int,
    limit: int,
) -> tuple[ScoredRun, ...]:
    """The runs of the same shape the search did not take, ranked.

    Design §8: "Every fragment keeps its ranked alternatives so a choice
    can be swapped without re-running." They are priced against the
    fragment that actually precedes this one, so swapping one in is a
    like-for-like comparison rather than a context-free score.
    """
    previous = None if chosen.prev_source == _NO_SOURCE else chosen.prev_source
    taken = chosen.run.video_id if chosen.run is not None else None
    scored = [
        ScoredRun(
            run=run,
            cost=(
                fragment_cost(run, weights)
                + transition_cost(previous, video_id, weights, acoustics)
                + jitter(seed, video_id, run.first_word_ord, position)
            ),
        )
        for video_id, run in table.get((position, length), {}).items()
        if video_id != taken
    ]
    scored.sort(key=lambda item: (rounded(item.cost), item.run.video_id, item.run.first_word_ord))
    return tuple(scored[:limit])


def plan_coverage(
    target: str,
    tokens: Sequence[str],
    table: RunTable,
    *,
    weights: Weights,
    acoustics: AcousticsMap,
    seed: int = 0,
    max_alternatives: int = C.ASSEMBLE_MAX_ALTERNATIVES,
) -> Plan:
    """Choose a segmentation and a source for every slot, together.

    A shortest path over ``(target position, last source video)``. The gap
    edge — step over one token for ``ASSEMBLE_GAP_COST`` — always exists,
    so a target the corpus cannot say still produces a plan: design §8
    says "report the gap plainly and render everything else", not "fail".
    """
    if not tokens:
        raise InvalidInputError("the target is empty: there is nothing to assemble")
    count = len(tokens)

    lengths: dict[int, list[int]] = {}
    for position, length in table:
        lengths.setdefault(position, []).append(length)
    for available in lengths.values():
        available.sort()

    best: list[dict[int, _Step]] = [{} for _ in range(count + 1)]
    best[0][_NO_SOURCE] = _Step(cost=0.0, prev_position=-1, prev_source=_NO_SOURCE, run=None)

    for position in range(count):
        node = sorted(best[position].items())
        if not node:
            continue
        for source, step in node:
            _relax(
                best[position + 1],
                source,
                _Step(
                    cost=step.cost + C.ASSEMBLE_GAP_COST,
                    prev_position=position,
                    prev_source=source,
                    run=None,
                ),
            )
        available = lengths.get(position, [])
        candidates = {video_id for length in available for video_id in table[(position, length)]}
        arrivals = _arrival_costs(node, candidates, weights, acoustics)
        for length in available:
            for video_id, run in sorted(table[(position, length)].items()):
                arrival, source = arrivals[video_id]
                _relax(
                    best[position + length],
                    video_id,
                    _Step(
                        cost=(
                            arrival
                            + fragment_cost(run, weights)
                            + jitter(seed, video_id, run.first_word_ord, position)
                        ),
                        prev_position=position,
                        prev_source=source,
                        run=run,
                    ),
                )

    final_source, final_step = min(
        best[count].items(), key=lambda item: (_step_key(item[1]), item[0])
    )

    walked: list[_Step] = []
    position, source = count, final_source
    while position > 0:
        step = best[position][source]
        walked.append(step)
        position, source = step.prev_position, step.prev_source
    walked.reverse()

    slots: list[PlanSlot] = []
    for step in walked:
        start = step.prev_position
        end = start + (1 if step.run is None else step.run.n_words)
        edge_cost = step.cost - best[start][step.prev_source].cost
        if step.run is None:
            slots.append(
                PlanSlot(
                    kind=SLOT_GAP,
                    target_first=start,
                    target_last=end - 1,
                    text=tokens[start],
                    cost=edge_cost,
                )
            )
        else:
            slots.append(
                PlanSlot(
                    kind=SLOT_FRAGMENT,
                    target_first=start,
                    target_last=end - 1,
                    text=step.run.text,
                    cost=edge_cost,
                    run=step.run,
                    alternatives=_alternatives(
                        table,
                        start,
                        step.run.n_words,
                        step,
                        weights,
                        acoustics,
                        seed,
                        max_alternatives,
                    ),
                )
            )

    return Plan(
        target=target,
        tokens=tuple(tokens),
        slots=tuple(slots),
        total_cost=final_step.cost,
    )


def _tail_limits(db: Database, runs: Sequence[CandidateRun]) -> dict[tuple[int, int], int]:
    """How far past its last word each run may reach, keyed by (video, ord).

    Two bounds, whichever is nearer: the start of the next word in that
    video, and the video's own duration. A video with neither — no next
    word and no known duration — is unbounded here; ffmpeg will stop at
    the end of the file regardless, and guessing a limit would be worse
    than letting the renderer clamp it.
    """
    if not runs:
        return {}
    wanted = sorted({(run.video_id, run.last_word_ord + 1) for run in runs})
    limits: dict[tuple[int, int], int] = {}
    for start in range(0, len(wanted), C.ASSEMBLE_SQL_BATCH):
        batch = wanted[start : start + C.ASSEMBLE_SQL_BATCH]
        pairs = ", ".join(f"(:v{index}, :o{index})" for index in range(len(batch)))
        params: dict[str, object] = {}
        for index, (video_id, ordinal) in enumerate(batch):
            params[f"v{index}"] = video_id
            params[f"o{index}"] = ordinal
        for record in db.conn.execute(
            f"SELECT video_id, ord, start_ms FROM words "
            f"WHERE (video_id, ord) IN (VALUES {pairs})",
            params,
        ):
            limits[(int(record["video_id"]), int(record["ord"]))] = int(record["start_ms"])
    return limits


def _durations(db: Database, video_ids: Iterable[int]) -> dict[int, int]:
    """Known durations, so a fragment at the very end cannot overrun the file."""
    wanted = sorted({int(video_id) for video_id in video_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(video_id) for video_id in wanted)
    return {
        int(row["id"]): int(row["duration_ms"])
        for row in db.conn.execute(f"SELECT id, duration_ms FROM videos WHERE id IN ({inline})")
        if row["duration_ms"] is not None
    }


def pad_fragments(db: Database, plan: Plan, pad_ms: int) -> Plan:
    """Extend every fragment's tail by ``pad_ms``, clamped to what is there.

    Design §8 lists padding among the controls. It is applied here rather
    than at render time so that the cut list stays exactly what it claims
    to be — an in point and an out point — and a hand-edited timing is
    never silently re-derived.
    """
    if pad_ms <= 0:
        return plan

    runs = [slot.run for slot in plan.slots if slot.run is not None]
    runs.extend(scored.run for slot in plan.slots for scored in slot.alternatives)
    limits = _tail_limits(db, runs)
    durations = _durations(db, (run.video_id for run in runs))

    def padded(run: CandidateRun) -> CandidateRun:
        ceiling = limits.get((run.video_id, run.last_word_ord + 1))
        duration = durations.get(run.video_id)
        if duration is not None:
            ceiling = duration if ceiling is None else min(ceiling, duration)
        end_ms = run.end_ms + pad_ms
        if ceiling is not None:
            end_ms = min(end_ms, ceiling)
        # A clamp below the measured end must not drag the cut backwards.
        return replace(run, end_ms=max(end_ms, run.end_ms))

    slots = tuple(
        replace(
            slot,
            run=None if slot.run is None else padded(slot.run),
            alternatives=tuple(
                replace(scored, run=padded(scored.run)) for scored in slot.alternatives
            ),
        )
        for slot in plan.slots
    )
    return replace(plan, slots=slots)


@dataclass(frozen=True)
class AbsenceDiagnosis:
    """Why one target word matched nothing, broken down by cause.

    BUGS.md entry 26: "a user sees 'no fragments' and reasonably concludes
    the corpus does not contain the phrase" — this is what makes that
    conclusion checkable instead of assumed. The three counts are
    mutually exclusive and, together with what is left over, account for
    every occurrence of the token in the corpus.
    """

    token: str
    total: int
    excluded_tier: int  # caption tier, or timed tier without --allow-timed
    excluded_unknown_scale: int  # BUGS.md entry 36's fabricated boundaries
    excluded_below_floor: int  # scored, but the score did not clear its floor
    #: Words sharing this token's stem, when the exact form was never said.
    #: Search falls back to stem matching, so it happily shows a hit for an
    #: inflected form — and then assembly reported "never said in the
    #: corpus", which reads as a contradiction. Assembly is right to refuse
    #: (cutting `каннибализмом` to say `каннибализм` puts the wrong word in
    #: the video); it was the wording that was wrong.
    stem_forms: tuple[tuple[str, int], ...] = ()


def diagnose_absence(db: Database, token: str, filters: MatchFilters) -> AbsenceDiagnosis:
    """Classify every occurrence of ``token``, cuttable or not, by why it
    would not have matched under ``filters``.

    Meant to run once a real search already found nothing for this token
    (design §8: "report the gap plainly") — a single aggregate query, not
    the hot path :func:`find_occurrences` is, so it need not sit on an
    index the way that one must.
    """
    aligned_ok = _scale_clause(filters.min_align_by_scale)
    timed_excluded = "0" if filters.allow_timed else "1"
    unknown = C.ALIGN_SCALE_UNKNOWN
    sql = (
        "SELECT COUNT(*) AS total,"
        " SUM(CASE"
        "   WHEN source NOT IN ('aligned', 'timed') THEN 1"
        f"   WHEN source = 'timed' THEN {timed_excluded}"
        "   ELSE 0 END) AS excluded_tier,"
        " SUM(CASE WHEN source = 'aligned'"
        f"   AND (align_scale IS NULL OR align_scale = '{unknown}')"
        "   THEN 1 ELSE 0 END) AS excluded_unknown_scale,"
        " SUM(CASE WHEN source = 'aligned'"
        f"   AND align_scale IS NOT NULL AND align_scale != '{unknown}'"
        f"   AND NOT {aligned_ok} THEN 1 ELSE 0 END) AS excluded_below_floor"
        " FROM words WHERE normalized_text = :token"
    )
    row = db.conn.execute(sql, {"token": token}).fetchone()
    total = int(row["total"] or 0)
    return AbsenceDiagnosis(
        token=token,
        total=total,
        excluded_tier=int(row["excluded_tier"] or 0),
        excluded_unknown_scale=int(row["excluded_unknown_scale"] or 0),
        excluded_below_floor=int(row["excluded_below_floor"] or 0),
        stem_forms=() if total else _stem_forms(db, token),
    )


def _stem_forms(db: Database, token: str) -> tuple[tuple[str, int], ...]:
    """Forms the corpus *does* have that share this token's stem.

    Only consulted when the exact form was never said. `words_stem` already
    indexes the column, and this runs once per missing word on a path that
    is already off the hot loop.
    """
    from rytp.models import stem_text

    stem = stem_text(token)
    if not stem:
        return ()
    rows = db.conn.execute(
        "SELECT normalized_text AS form, COUNT(*) AS n FROM words"
        " WHERE stem = :stem AND normalized_text != :token"
        " GROUP BY normalized_text ORDER BY n DESC, normalized_text"
        " LIMIT :limit",
        {"stem": stem, "token": token, "limit": C.ASSEMBLE_STEM_FORM_LIMIT},
    ).fetchall()
    return tuple((str(r["form"]), int(r["n"])) for r in rows)


@dataclass(frozen=True)
class SubstitutionHit:
    """A word the corpus does say, offered in place of one it does not."""

    text: str
    reason: str  # "stem" | "edit"
    distance: int
    occurrences: int
    run: CandidateRun


def edit_distance_at_most(first: str, second: str, max_distance: int) -> int | None:
    """Levenshtein distance, or ``None`` once it is certainly past the budget.

    Written out rather than pulled in: the standard library has no edit
    distance (``difflib`` measures something else), and a dependency for
    twenty lines is not worth it. The early abort matters — it is what
    makes scanning a whole vocabulary slice cheap, because most candidates
    are eliminated after a row or two.
    """
    if abs(len(first) - len(second)) > max_distance:
        return None
    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, start=1):
        current = [index]
        row_min = index
        for position, right in enumerate(second, start=1):
            value = min(
                previous[position] + 1,
                current[position - 1] + 1,
                previous[position - 1] + (0 if left == right else 1),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= max_distance else None


def edit_distance(first: str, second: str) -> int:
    """Full Levenshtein distance — the budgeted form with an unreachable budget.

    Used only to report how far a same-stem suggestion is: the stem tier
    is chosen by the stem, not by distance, so the number is information
    rather than a filter.
    """
    distance = edit_distance_at_most(first, second, len(first) + len(second))
    return len(first) + len(second) if distance is None else distance


def _vocabulary(
    db: Database, token: str, filters: MatchFilters, *, same_stem: bool
) -> list[tuple[str, int]]:
    """Distinct cuttable spellings, with how often each is said.

    ``same_stem`` picks the tier: the stem index for tier one, a length
    window for tier two. Both exclude the token itself and any row whose
    normalized text holds a space — a hyphenated source word lands in one
    row as two words, and offering it as a substitution for one word
    would be wrong.
    """
    tail = _eligibility_sql(filters)
    params: dict[str, object] = {
        "token": token,
        "limit": C.ASSEMBLE_SUBSTITUTION_VOCAB_LIMIT,
    }
    if same_stem:
        params["stem"] = stem_text(token)
        where = "stem = :stem"
    else:
        window = C.ASSEMBLE_SUBSTITUTION_LEN_WINDOW
        params["low"] = max(1, len(token) - window)
        params["high"] = len(token) + window
        where = "LENGTH(normalized_text) BETWEEN :low AND :high"
    rows = db.conn.execute(
        f"SELECT normalized_text, COUNT(*) AS n FROM words "
        f"WHERE {where} AND normalized_text <> :token "
        f"AND normalized_text NOT LIKE '% %' AND {tail} "
        f"GROUP BY normalized_text ORDER BY n DESC, normalized_text LIMIT :limit",
        params,
    )
    return [(str(row["normalized_text"]), int(row["n"])) for row in rows]


def suggest_substitutions(
    db: Database,
    token: str,
    filters: MatchFilters,
    *,
    limit: int = C.ASSEMBLE_SUBSTITUTION_LIMIT,
) -> tuple[SubstitutionHit, ...]:
    """Ranked stand-ins for a word the corpus never says.

    Design §8: same stem first, then orthographically close by edit
    distance. Nothing here is ever applied — the caller writes these
    beside the gap and the owner decides.
    """
    ranked: list[tuple[tuple[int, int, str], str, str, int, int]] = []
    seen: set[str] = set()

    for form, occurrences in _vocabulary(db, token, filters, same_stem=True):
        seen.add(form)
        ranked.append(
            ((0, -occurrences, form), form, "stem", edit_distance(token, form), occurrences)
        )

    for form, occurrences in _vocabulary(db, token, filters, same_stem=False):
        if form in seen:
            continue
        distance = edit_distance_at_most(token, form, C.ASSEMBLE_SUBSTITUTION_MAX_DISTANCE)
        if distance is None:
            continue
        ranked.append(((1, distance, form), form, "edit", distance, occurrences))

    ranked.sort(key=lambda item: item[0])

    hits: list[SubstitutionHit] = []
    for _key, form, reason, distance, occurrences in ranked:
        if len(hits) == limit:
            break
        found = find_occurrences(db, form, filters)
        if not found:
            continue
        hits.append(
            SubstitutionHit(
                text=form,
                reason=reason,
                distance=distance,
                occurrences=occurrences,
                run=_run_from([found[0]]),
            )
        )
    return tuple(hits)
