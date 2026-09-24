"""Run several engines over the same audio and report where they disagree.

Design §11, milestone M0. Hand-marking a reference set is the only way to get
a true error figure, and it was deliberately postponed. What replaces it is
this: engines are swappable, every word row records the engine that produced
it, and this module turns "which transcriber wins on *this* audio" and "is
alignment needed at all" into a command rather than a project.

Three kinds of disagreement are reported, because they fail differently:

* **Text** — a wrong word is worse than a missing one, since the tool will
  confidently cut audio that does not say what the index claims (design §3).
* **Timing** — how far apart two engines place the same word.
* **Boundaries** — how much silence actually sits where each engine claims a
  word ends. That last one is measured from the audio, so it needs no
  reference transcript at all.
"""
from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rytp import constants as C
from rytp.audio.energy import index_to_ms, read_wav_mono, refine_boundaries, silence_depth_db
from rytp.models import RawWord, Span, normalize_text
from rytp.transcribe.pipeline import AlignmentMismatchError
from rytp.transcribe.registry import load_aligner, load_transcriber

if TYPE_CHECKING:
    from rytp.db import Database


@dataclass(frozen=True)
class EngineRun:
    """One engine's output over one window of one file."""

    label: str
    words: tuple[str, ...]
    spans: tuple[Span, ...]
    elapsed_s: float


@dataclass(frozen=True)
class PairStats:
    """How far apart two engines are, on text and on time."""

    a: str
    b: str
    matches: int
    substitutions: int
    insertions: int
    deletions: int
    disagreement: float
    median_abs_start_delta_ms: float | None
    p90_abs_start_delta_ms: float | None


def align_tokens(
    a: Sequence[str], b: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Wagner-Fischer alignment of two token sequences.

    Returns index pairs: ``(i, j)`` for a match or substitution, ``(i, None)``
    for a deletion, ``(None, j)`` for an insertion. Backtracking prefers
    match/substitute, then deletion, then insertion, so two runs of the
    comparison command produce the same report.

    Quadratic in the token counts, which is why the comparison command works
    on a window (default two minutes) rather than a whole video.
    """
    n, m = len(a), len(b)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = cost[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost[i][j] = min(substitute, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        diagonal = (
            cost[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            if i > 0 and j > 0
            else None
        )
        if diagonal is not None and cost[i][j] == diagonal:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare_pair(left: EngineRun, right: EngineRun) -> PairStats:
    """Text and timing disagreement between two runs over the same audio."""
    a = [normalize_text(word) for word in left.words]
    b = [normalize_text(word) for word in right.words]
    pairs = align_tokens(a, b)

    matches = substitutions = insertions = deletions = 0
    deltas: list[float] = []
    for i, j in pairs:
        if i is None:
            insertions += 1
        elif j is None:
            deletions += 1
        elif a[i] == b[j]:
            matches += 1
            deltas.append(abs(left.spans[i].start_ms - right.spans[j].start_ms))
        else:
            substitutions += 1
    denominator = max(len(a), 1)
    return PairStats(
        a=left.label,
        b=right.label,
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        disagreement=(substitutions + insertions + deletions) / denominator,
        median_abs_start_delta_ms=_percentile(deltas, 50),
        p90_abs_start_delta_ms=_percentile(deltas, 90),
    )


@dataclass(frozen=True)
class BoundaryStats:
    """What the audio says about one engine's claimed boundaries."""

    label: str
    n_words: int
    zero_gap_fraction: float
    median_gap_ms: float
    zero_duration_words: int
    median_silence_depth_db: float | None
    median_shift_ms: float | None
    capped_shifts: int
    elapsed_s: float


def boundary_shifts(claimed: Sequence[Span], refined: Sequence[Span]) -> list[int]:
    """How far energy refinement had to move each boundary."""
    shifts = [
        abs(after.start_ms - before.start_ms)
        for before, after in zip(claimed, refined, strict=True)
    ]
    if claimed:
        shifts.append(abs(refined[-1].end_ms - claimed[-1].end_ms))
    return shifts


def count_capped(shifts: Sequence[int]) -> int:
    """Boundaries that ran into the search rail.

    The direct signal that an aligner is further out than refinement is
    allowed to correct: refinement stops at
    :data:`rytp.constants.BOUNDARY_SEARCH_MS`, so a boundary pinned there was
    still heading somewhere else.
    """
    return sum(1 for shift in shifts if shift >= C.BOUNDARY_SEARCH_MS)


def boundary_stats(run: EngineRun, samples: np.ndarray, sr: int) -> BoundaryStats:
    """Measure one engine's boundaries against the audio itself.

    ``zero_gap_fraction`` is the measurement that motivated this whole design:
    a transcriber that sets ``word[i].end == word[i+1].start`` by construction
    scores 1.0 here and its timings cannot be cut on.
    """
    spans = list(run.spans)
    if not spans:
        return BoundaryStats(
            label=run.label,
            n_words=0,
            zero_gap_fraction=0.0,
            median_gap_ms=0.0,
            zero_duration_words=0,
            median_silence_depth_db=None,
            median_shift_ms=None,
            capped_shifts=0,
            elapsed_s=run.elapsed_s,
        )
    gaps = [
        right.start_ms - left.end_ms for left, right in itertools.pairwise(spans)
    ]
    boundaries = [span.start_ms for span in spans] + [spans[-1].end_ms]
    depths = [silence_depth_db(samples, sr, boundary) for boundary in boundaries]
    shifts = boundary_shifts(spans, refine_boundaries(samples, sr, spans))
    return BoundaryStats(
        label=run.label,
        n_words=len(spans),
        zero_gap_fraction=(sum(1 for gap in gaps if gap == 0) / len(gaps)) if gaps else 0.0,
        median_gap_ms=_percentile([float(gap) for gap in gaps], 50) or 0.0,
        zero_duration_words=sum(1 for span in spans if span.end_ms <= span.start_ms),
        median_silence_depth_db=_percentile(depths, 50),
        median_shift_ms=_percentile([float(shift) for shift in shifts], 50),
        capped_shifts=count_capped(shifts),
        elapsed_s=run.elapsed_s,
    )


def _raw_spans(words: Sequence[RawWord]) -> tuple[Span, ...] | None:
    """A transcriber's own timings, or ``None`` when it did not time everything.

    Contracts §4 lets a transcriber emit text only. Such an engine has no
    standalone run to compare — only its pairings with an aligner do — so the
    comparison quietly leaves it out of the boundary table rather than
    inventing timings for it.
    """
    spans: list[Span] = []
    for word in words:
        start, end = word.start_ms, word.end_ms
        if start is None or end is None:
            return None
        spans.append(Span(start_ms=start, end_ms=end, score=word.confidence))
    return tuple(spans)


def run_comparison(
    db: Database,
    *,
    wav_path: Path,
    transcribers: Sequence[str],
    aligners: Sequence[str] = (),
    start_ms: int = 0,
    end_ms: int | None = None,
    language: str | None = C.DEFAULT_LANGUAGE,
) -> list[EngineRun]:
    """Run every transcriber, and every transcriber-aligner pairing, over one window.

    The transcriber-only run is kept alongside its aligned pairings on
    purpose: the difference between them is the answer to "is alignment
    needed at all" (design §11). A transcriber that emits text without
    timings has no standalone run and contributes only its pairings.
    """
    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    window_end = total_ms if end_ms is None else min(end_ms, total_ms)
    runs: list[EngineRun] = []
    for name in transcribers:
        engine = load_transcriber(db, name)
        started = time.monotonic()
        produced = [
            word
            for word in engine.transcribe(
                wav_path, language=language, start_ms=start_ms, end_ms=window_end
            )
            if normalize_text(word.text)
        ]
        elapsed = time.monotonic() - started
        if all(word.start_ms is not None for word in produced):
            produced.sort(key=lambda word: (word.start_ms or 0, word.text))
        texts = tuple(word.text for word in produced)
        raw = _raw_spans(produced)
        if raw is not None:
            runs.append(
                EngineRun(label=name, words=texts, spans=raw, elapsed_s=elapsed)
            )
        for aligner_name in aligners:
            aligner = load_aligner(db, aligner_name)
            started = time.monotonic()
            spans = list(
                aligner.align(
                    wav_path, list(texts), start_ms=start_ms, end_ms=window_end
                )
            )
            aligner_elapsed = time.monotonic() - started
            if len(spans) != len(texts):
                raise AlignmentMismatchError(
                    f"aligner {aligner_name!r} returned {len(spans)} spans for "
                    f"{len(texts)} words from {name!r}"
                )
            runs.append(
                EngineRun(
                    label=f"{name}+{aligner_name}",
                    words=texts,
                    spans=tuple(spans),
                    elapsed_s=aligner_elapsed,
                )
            )
    return runs


_COMPARISON_COLUMNS = (
    "engine",
    "words",
    "zero-gap",
    "median gap ms",
    "zero-length",
    "silence depth dB",
    "median shift ms",
    "shifts at rail",
    "seconds",
)


def _format(value: float | None, places: int = 1) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def comparison_rows(
    runs: Sequence[EngineRun], samples: np.ndarray, sr: int
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Boundary statistics as a table, for ``CommandResult``."""
    rows: list[tuple[str, ...]] = []
    for run in runs:
        stats = boundary_stats(run, samples, sr)
        rows.append(
            (
                stats.label,
                str(stats.n_words),
                f"{stats.zero_gap_fraction:.1%}",
                _format(stats.median_gap_ms),
                str(stats.zero_duration_words),
                _format(stats.median_silence_depth_db),
                _format(stats.median_shift_ms),
                str(stats.capped_shifts),
                _format(stats.elapsed_s),
            )
        )
    return _COMPARISON_COLUMNS, rows


def render_report(
    runs: Sequence[EngineRun], samples: np.ndarray, sr: int, *, source: str
) -> str:
    """The full markdown comparison: boundaries, disagreement, and a sample."""
    columns, rows = comparison_rows(runs, samples, sr)
    lines = [
        "# Engine comparison",
        "",
        f"Source: `{source}`",
        "",
        "## Boundaries",
        "",
        "Measured from the audio, so no reference transcript is needed. A high",
        "zero-gap percentage means the engine assigns `word[i].end ==",
        "word[i+1].start` by construction and its timings cannot be cut on.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    lines += [
        "",
        "## Disagreement",
        "",
        "| a | b | matched | subs | ins | del | disagreement | median Δstart ms | p90 Δstart ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, left in enumerate(runs):
        for right in runs[index + 1 :]:
            stats = compare_pair(left, right)
            lines.append(
                f"| {stats.a} | {stats.b} | {stats.matches} | {stats.substitutions} "
                f"| {stats.insertions} | {stats.deletions} "
                f"| {stats.disagreement:.1%} "
                f"| {_format(stats.median_abs_start_delta_ms)} "
                f"| {_format(stats.p90_abs_start_delta_ms)} |"
            )

    lines += ["", "## First words", ""]
    for run in runs:
        preview = " ".join(run.words[: C.COMPARE_SAMPLE_WORDS])
        lines += [f"**{run.label}**", "", f"> {preview}", ""]
    return "\n".join(lines) + "\n"
