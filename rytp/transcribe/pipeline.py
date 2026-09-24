"""Aligned-tier transcription: text, rough timings, then measured boundaries.

Design §6. Three sources, because no single one can be trusted:

1. **Text** from the best available transcriber.
2. **Rough timings** from forced alignment against that text.
3. **The final boundary** at the local energy minimum between two words,
   snapped to a zero crossing, with voice-activity edges taken for free.

Promoting a video to this tier deletes its caption words — one tier per video,
no run history to reason about. It also drops the video's utterances, which
copy word timings and ordinals, and its diarized speaker labels, because the
ordinals those labels were attached to no longer mean the same thing. Design
§3: erase and replace, don't reconcile.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from rytp import constants as C
from rytp.audio.energy import (
    boundary_quality,
    enforce_monotonic,
    index_to_ms,
    read_wav_mono,
    refine_boundaries,
    silence_depth_db,
)
from rytp.audio.vad import AudioChunk, detect_speech, plan_chunks
from rytp.db import Database
from rytp.models import RawWord, RytpError, Span, normalize_text, stem_text
from rytp.transcribe.base import Aligner, split_token
from rytp.transcribe.registry import load_aligner, load_transcriber


class AlignmentMismatchError(RytpError):
    """The aligner returned a different number of spans than there were words."""


@dataclass(frozen=True)
class TranscribeOutcome:
    """What one transcription run produced, including which tier it wrote."""

    video_id: int
    n_words: int
    n_chunks: int
    engine: str
    #: The `words.source` tier this run wrote: "timed" or "aligned".
    source: str
    #: Speaker labels destroyed along with the old transcript, if any.
    speakers_lost: int
    median_align_score: float | None
    #: `words.align_scale` this run actually wrote for the words that carry a
    #: score (plan §1a) — the most common non-null scale, so the report can
    #: name what `median_align_score` is a median *of*. ``None`` when no word
    #: got a score at all (no aligner, no refine; or a scoreless aligner with
    #: refine off).
    align_scale: str | None
    #: The aligner's chosen device (plan §1b), or ``None`` when no aligner ran
    #: this call — there is nothing to report a device for.
    align_device: str | None
    #: Advisory findings drained from the aligner's `notes` channel
    #: (contracts §6) across every chunk. Never a failure.
    align_notes: tuple[str, ...] = ()


_WORD_COLUMNS = (
    "video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
    "confidence, align_score, align_scale, source, engine, video_speaker_id"
)
_INSERT_WORD = (
    f"INSERT INTO words ({_WORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _validate_scale(scale: str | None) -> str | None:
    """Python-side enforcement of contracts §3: SQLite cannot `CHECK` a column
    added by `ALTER TABLE`, so every writer of `align_scale` validates here."""
    if scale is not None and scale not in C.ALIGN_SCALES:
        raise RytpError(f"invalid align_scale {scale!r}; expected one of {C.ALIGN_SCALES}")
    return scale


def resolve_word_scale(
    pre_score: float | None, *, refine: bool, aligner_scale: str | None
) -> str | None:
    """`words.align_scale` for one word, following plan §1a's table exactly.

    ``pre_score`` is the word's `Span.score` as it stood right after
    transcription/alignment, before :func:`refine_boundaries` moved boundaries
    and before :func:`_score_boundaries` filled any still-``None`` score with
    the measured energy quality. ``aligner_scale`` is the aligner engine's own
    `score_scale`, or ``None`` when no aligner ran at all.

    An aligner that reported a score keeps its own scale regardless of
    ``refine``: refinement moves boundaries, it does not re-measure alignment
    (plan §1a). A ``None`` score — whether because no aligner ran, or because
    one ran and reported nothing (MFA) — is filled by the energy measure when
    ``refine`` is on, so that value's honest scale is ``energy``, not the
    aligner's own (an aligner that reports nothing didn't produce the number
    on the row; the energy refiner did). With no refine, an aligner that
    reported nothing stays ``none`` — a real aligner ran and had nothing to
    say — and no aligner at all stays unscored, ``None``.
    """
    if pre_score is not None:
        # The aligner (or, before an aligner-only design, nothing else)
        # supplied a real number: its own scale, kept regardless of refine.
        assert aligner_scale is not None  # a score with no aligner never reaches here
        return aligner_scale
    if refine:
        return C.ALIGN_SCALE_ENERGY
    return C.ALIGN_SCALE_NONE if aligner_scale is not None else None


def _summarize_scores(
    spans: Sequence[Span], scales: Sequence[str | None]
) -> tuple[str | None, float | None]:
    """The scale and median to report for a run (plan §1a, for the CLI table).

    Keyed off words that actually carry a score — `align_scale = 'none'`
    words never do (a real aligner ran and had nothing to say), so a run
    where every word is `none` reports ``(None, None)`` exactly like a run
    with no score at all, rather than a "median none score" heading over a
    blank cell. Almost always uniform otherwise; wav2vec2's rare interpolated
    word (no CTC frames of its own) can mix `logprob` with `energy` in one
    run when `--refine` is on, so the majority scale is picked rather than
    purity assumed, and the median is taken only over words at *that* scale —
    never blended across scales, which would be entry 13's defect again in
    miniature.
    """
    scored = [
        (scale, span.score)
        for span, scale in zip(spans, scales, strict=True)
        if scale is not None and span.score is not None
    ]
    if not scored:
        return None, None
    reported = Counter(scale for scale, _score in scored).most_common(1)[0][0]
    scores = [score for scale, score in scored if scale == reported]
    return reported, float(median(scores))


def tier_for(aligner: str | None) -> str:
    """Which `words.source` tier this run produces (contracts §3).

    An aligner ran, so every boundary was actually placed: ``aligned``, which
    is the definition of cuttable. No aligner, so the boundaries are the
    transcriber's own: ``timed`` — searchable, not cuttable.

    This is the whole point of the three-tier scheme. Measured on the owner's
    real data, 78.7% of the transcriber's word gaps are exactly zero, because
    it sets ``word[i].end == word[i+1].start`` and absorbs every pause into a
    neighbour. Energy refinement still runs on a `timed` transcript and still
    improves it, but it relocates a boundary inside a window — it cannot place
    one that was never there. Writing these rows ``aligned`` and hoping is
    exactly what the design exists to prevent, and nothing downstream could
    have caught it: ``align_score`` is null without an aligner and no consumer
    reads ``words.engine``.
    """
    return "aligned" if aligner else "timed"


def _drain_notes(engine: Any) -> list[str]:
    """Read and clear an engine's advisory notes channel (contracts §6).

    ``getattr`` with a default: an engine that never has anything to report
    simply never defines ``notes``, and this must not require that it does.
    """
    notes = list(getattr(engine, "notes", ()))
    if notes and hasattr(engine, "notes"):
        engine.notes.clear()
    return notes


def engine_tag(transcriber: str, aligner: str | None, refined: bool) -> str:
    """The ``words.engine`` value: every stage that touched these timings.

    A corpus built with more than one engine stays interpretable only if each
    row says what made it (design §11, M0).
    """
    parts = [transcriber]
    if aligner:
        parts.append(aligner)
    if refined:
        parts.append("energy")
    return "+".join(parts)


def transcribe_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    transcriber: str,
    aligner: str | None = None,
    language: str | None = C.DEFAULT_LANGUAGE,
    refine: bool = True,
    transcriber_kwargs: dict[str, Any] | None = None,
    aligner_kwargs: dict[str, Any] | None = None,
) -> TranscribeOutcome:
    """Transcribe, align and refine one video, then replace its words."""
    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    speech = detect_speech(samples, sr)
    chunks = plan_chunks(speech, total_ms=total_ms) or [
        AudioChunk(ord=0, start_ms=0, end_ms=total_ms)
    ]
    engine = load_transcriber(db, transcriber, **(transcriber_kwargs or {}))
    aligner_engine = (
        load_aligner(db, aligner, **(aligner_kwargs or {})) if aligner else None
    )

    words: list[RawWord] = []
    spans: list[Span] = []
    align_notes: list[str] = []
    for chunk in chunks:
        produced = [
            word
            for word in engine.transcribe(
                wav_path, language=language, start_ms=chunk.start_ms, end_ms=chunk.end_ms
            )
            if normalize_text(word.text)
        ]
        if all(word.start_ms is not None for word in produced):
            produced.sort(key=lambda word: (word.start_ms or 0, word.text))
        # Otherwise the transcriber emitted text only (contracts §4 allows it)
        # and its words are already in spoken order; there is nothing to sort by.
        if not produced:
            continue
        words.extend(produced)
        spans.extend(
            _chunk_spans(
                chunk,
                produced,
                aligner_engine,
                wav_path,
                transcriber=transcriber,
                aligner=aligner,
                notes=align_notes,
            )
        )
    # One row per token, before refinement so an invented interior boundary
    # gets measured like any other.
    words, spans = _expand_tokens(words, spans)
    if not words:
        raise RytpError(
            f"transcriber {transcriber!r} produced no words for video {video_id}"
        )

    # Snapshot each span's score before refinement touches anything, so the
    # per-word scale (plan §1a) can tell "the aligner supplied this" from
    # "the energy refiner filled this in" after `_score_boundaries` runs.
    pre_refine_scores = [span.score for span in spans]
    aligner_scale = aligner_engine.score_scale if aligner_engine is not None else None

    if refine:
        spans = refine_boundaries(
            samples,
            sr,
            spans,
            speech=[(segment.start_ms, segment.end_ms) for segment in speech],
        )
        spans = _score_boundaries(samples, sr, spans)
    else:
        spans = enforce_monotonic(spans, total_ms=total_ms)

    scales = [
        _validate_scale(resolve_word_scale(pre, refine=refine, aligner_scale=aligner_scale))
        for pre in pre_refine_scores
    ]

    tag = engine_tag(transcriber, aligner, refine)
    tier = tier_for(aligner)
    removed = replace_words(db, video_id, words, spans, tag, source=tier, align_scales=scales)
    reported_scale, reported_median = _summarize_scores(spans, scales)
    return TranscribeOutcome(
        video_id=video_id,
        n_words=len(words),
        n_chunks=len(chunks),
        engine=tag,
        source=tier,
        speakers_lost=removed.video_speakers,
        median_align_score=reported_median,
        align_scale=reported_scale,
        align_device=aligner_engine.device if aligner_engine is not None else None,
        align_notes=tuple(align_notes),
    )


def _chunk_spans(
    chunk: AudioChunk,
    words: Sequence[RawWord],
    aligner_engine: Aligner | None,
    wav_path: Path,
    *,
    transcriber: str,
    aligner: str | None,
    notes: list[str] | None = None,
) -> list[Span]:
    if aligner_engine is not None:
        spans = list(
            aligner_engine.align(
                wav_path,
                [word.text for word in words],
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
        )
        if notes is not None:
            notes.extend(_drain_notes(aligner_engine))
        if len(spans) != len(words):
            raise AlignmentMismatchError(
                f"aligner {aligner!r} returned {len(spans)} spans for {len(words)} "
                f"words in chunk {chunk.ord} [{chunk.start_ms}..{chunk.end_ms}]"
            )
        return spans
    # No aligner: the transcriber's own timings have to be complete. Contracts
    # §4 lets a transcriber emit text with no timings at all, and Whisper-class
    # engines routinely omit an end, so both halves are checked.
    #
    # The span's score starts out `None` here, deliberately not
    # `word.confidence`: `confidence` already has its own `words` column, and
    # a transcriber's per-word confidence is not an alignment score — folding
    # it into `align_score` would be a fifth, unlabelled scale on top of the
    # four plan §1a names. Leaving it `None` lets the boundary-quality energy
    # measure fill it when `--refine` is on (row 3 of the table) and leaves it
    # unscored when refine is off (row 4), which is exactly the distinction
    # BUGS.md entry 13 asked for.
    own_spans: list[Span] = []
    for word in words:
        start, end = word.start_ms, word.end_ms
        if start is None or end is None:
            raise RytpError(
                f"transcriber {transcriber!r} does not time every word "
                f"(first untimed: {word.text!r}); pass an aligner"
            )
        own_spans.append(Span(start_ms=start, end_ms=end, score=None))
    return own_spans


def _expand_tokens(
    words: Sequence[RawWord], spans: Sequence[Span]
) -> tuple[list[RawWord], list[Span]]:
    """One word row per token, splitting a multi-token word's span in proportion.

    A transcriber emits "кто-то" as a single token, and `normalize_text`
    replaces the hyphen with a space — so one row would hold two tokens, which
    nothing can ever find. Part 4's index and FTS lookups are single-token and
    Part 5's assembly pointer walk assumes one token per row. Stripping the
    hyphen into "ктото" would be worse rather than better: the same normalizer
    runs over the user's query, which also splits, so the stripped row would be
    unreachable from either direction.

    The interior boundary starts out interpolated by character count. Because
    this runs *before* :func:`refine_boundaries`, it is then measured like
    every other boundary — for "кто-то" the energy minimum lands in the stop
    closure of the /t/, which is exactly where the split belongs.
    """
    out_words: list[RawWord] = []
    out_spans: list[Span] = []
    for word, span in zip(words, spans, strict=True):
        pieces = split_token(word.text)
        if not pieces:
            continue
        if len(pieces) == 1:
            out_words.append(word)
            out_spans.append(span)
            continue
        total = sum(len(normalized) for _surface, normalized in pieces)
        cursor = span.start_ms
        for index, (surface, normalized) in enumerate(pieces):
            share = round((span.end_ms - span.start_ms) * len(normalized) / max(total, 1))
            end = (
                span.end_ms
                if index == len(pieces) - 1
                else min(cursor + share, span.end_ms)
            )
            out_words.append(
                RawWord(
                    text=surface,
                    start_ms=cursor,
                    end_ms=max(end, cursor),
                    confidence=word.confidence,
                )
            )
            out_spans.append(Span(start_ms=cursor, end_ms=max(end, cursor), score=span.score))
            cursor = max(end, cursor)
    return out_words, out_spans


@dataclass(frozen=True)
class TranscriptRemoval:
    """How much a transcript invalidation actually removed."""

    words: int
    utterances: int
    video_speakers: int


def _count(db: Database, table: str, video_id: int) -> int:
    if table not in ("words", "utterances", "video_speakers"):
        raise ValueError(f"not a per-video derived table: {table!r}")
    row = db.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
    ).fetchone()
    return int(row[0])


def invalidate_transcript(db: Database, video_id: int) -> TranscriptRemoval:
    """Drop a video's words and everything derived from them.

    Contracts §4's cross-part invariant, in one place so it cannot drift:
    `utterances` copy word timings and ordinals, and a `video_speakers` label
    is attached to ordinals that are about to change or disappear, so all
    three go together or none do. Called by every stage that replaces a
    video's transcript and by ``transcribe.remove``.

    ``db.transaction()`` is re-entrant (contracts §8), so this is safe to call
    from inside a larger write — which is exactly how `replace_words` uses it.

    The cached WAV is deliberately untouched: it is regenerable and Part 2's
    ``cache.prune`` owns it.
    """
    with db.transaction():
        removed = TranscriptRemoval(
            words=_count(db, "words", video_id),
            utterances=_count(db, "utterances", video_id),
            video_speakers=_count(db, "video_speakers", video_id),
        )
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    return removed


def speaker_loss_warning(video_id: int, speakers_lost: int) -> str | None:
    """Warn when replacing a transcript threw away human work, permanently.

    Contracts §4 makes replacing a video's words delete its `video_speakers`,
    and Part 7 registers `diarize` with ``reopenable=False`` so `reconcile`
    will never bring it back. Both decisions are right on their own; together
    they mean a re-transcribed video loses its speaker dimension for good and
    says nothing about it. This is that something.
    """
    if speakers_lost <= 0:
        return None
    return (
        f"warning: dropped {speakers_lost} speaker label(s) with the previous "
        f"transcript. Nothing re-derives them — run `rytp speakers diarize "
        f"{video_id}` and re-map the labels to get the speaker dimension back."
    )


def enqueue_index(db: Database, video_id: int) -> None:
    """Ask for the video's utterances to be rebuilt (contracts §5).

    Transcribing or re-aligning deletes the video's utterances, so its search
    index is stale the moment this stage finishes. Nothing else enqueues the
    rebuild — Part 2's chain stops at ``extract_wav`` — so without this a
    freshly transcribed video stays invisible to search until somebody runs a
    command by hand. Part 4's ``index`` kind is reopenable and its readiness
    is derived from the data, so enqueueing is the whole of the coordination.

    Called by the callers rather than from inside :func:`transcribe_video`, so
    the stage itself stays testable without a registered job queue.
    """
    from rytp.jobs import JOB_KINDS
    from rytp.jobs.queue import enqueue

    if "index" not in JOB_KINDS:
        # Part 4 registers that kind. Until it lands there is nothing to ask
        # for, and a transcribed video simply has no index yet.
        return
    enqueue(db, "index", video_id)


def _score_boundaries(samples: Any, sr: int, spans: Sequence[Span]) -> list[Span]:
    """Fill a missing alignment score with the measured quality of its boundaries.

    MFA reports no per-word confidence. How much silence actually sits at each
    end of the word is a better signal anyway, and it is already measured.

    Cost is two :func:`silence_depth_db` calls per word, each a handful of
    small numpy reductions — one to three seconds for an hour of speech. That
    is deliberate and measured; do not "optimise" it by widening the windows,
    which would change what the score means.
    """
    scored: list[Span] = []
    for span in spans:
        if span.score is not None:
            scored.append(span)
            continue
        depth = min(
            silence_depth_db(samples, sr, span.start_ms),
            silence_depth_db(samples, sr, span.end_ms),
        )
        scored.append(
            Span(start_ms=span.start_ms, end_ms=span.end_ms, score=boundary_quality(depth))
        )
    return scored


def replace_words(
    db: Database,
    video_id: int,
    words: Sequence[RawWord],
    spans: Sequence[Span],
    engine: str,
    *,
    source: str,
    align_scales: Sequence[str | None] | None = None,
) -> TranscriptRemoval:
    """Swap in a fresh transcript for one video, in one transaction.

    ``source`` is the tier from :func:`tier_for`. ``align_scales`` is one
    `words.align_scale` value per word (plan §1a); omitted or ``None``
    entries write ``NULL``, which is correct for a word with no score at all.
    Returns what the replacement destroyed, so a caller can warn about
    speaker labels it cannot rebuild.
    """
    scales = list(align_scales) if align_scales is not None else [None] * len(words)
    if len(scales) != len(words):
        raise RytpError(
            f"{len(scales)} align_scales for {len(words)} words: must match 1:1"
        )
    rows = []
    for ordinal, (word, span, scale) in enumerate(zip(words, spans, scales, strict=True)):
        normalized = normalize_text(word.text)
        rows.append(
            (
                video_id,
                ordinal,
                span.start_ms,
                span.end_ms,
                word.text,
                normalized,
                stem_text(normalized),
                word.confidence,
                span.score,
                _validate_scale(scale),
                source,
                engine,
                None,
            )
        )
    with db.transaction():
        removed = invalidate_transcript(db, video_id)
        db.conn.executemany(_INSERT_WORD, rows)
    return removed


def _chunk_for(chunks: Sequence[AudioChunk], ms: int) -> int:
    for chunk in chunks:
        if chunk.start_ms <= ms < chunk.end_ms:
            return chunk.ord
    return min(
        chunks, key=lambda chunk: min(abs(ms - chunk.start_ms), abs(ms - chunk.end_ms))
    ).ord


def realign_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    aligner: str,
    refine: bool = True,
    aligner_kwargs: dict[str, Any] | None = None,
) -> TranscribeOutcome:
    """Align a video's existing words, upgrading the tier in place.

    Two jobs in one. A `timed` transcript — the transcriber's own timestamps,
    searchable but not cuttable — becomes `aligned` and therefore cuttable,
    which is the promotion path contracts §3 describes. An already-`aligned`
    transcript gets re-timed by a different aligner, which is what makes
    aligners swappable without paying for transcription again.

    Either way the text is kept, so ordinals and the speaker labels hanging
    off them stay valid; only the timings, the alignment scores, the engine
    tag and the source tier change.
    """
    rows = db.conn.execute(
        "SELECT ord, start_ms, text, confidence, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    if not rows:
        raise RytpError(f"video {video_id} has no words to re-align")
    if any(str(row[4]) == "caption" for row in rows):
        raise RytpError(
            f"video {video_id} is caption-tier; run transcribe run to promote it"
        )

    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    speech = detect_speech(samples, sr)
    chunks = plan_chunks(speech, total_ms=total_ms) or [
        AudioChunk(ord=0, start_ms=0, end_ms=total_ms)
    ]
    engine = load_aligner(db, aligner, **(aligner_kwargs or {}))

    buckets: dict[int, list[int]] = {chunk.ord: [] for chunk in chunks}
    for index, row in enumerate(rows):
        buckets[_chunk_for(chunks, int(row[1]))].append(index)

    by_index: dict[int, Span] = {}
    align_notes: list[str] = []
    for chunk in chunks:
        indexes = buckets[chunk.ord]
        if not indexes:
            continue
        spans = list(
            engine.align(
                wav_path,
                [str(rows[index][2]) for index in indexes],
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
        )
        align_notes.extend(_drain_notes(engine))
        if len(spans) != len(indexes):
            raise AlignmentMismatchError(
                f"aligner {aligner!r} returned {len(spans)} spans for {len(indexes)} "
                f"words in chunk {chunk.ord} [{chunk.start_ms}..{chunk.end_ms}]"
            )
        for index, span in zip(indexes, spans, strict=True):
            by_index[index] = span

    ordered = [by_index[index] for index in range(len(rows))]
    # Snapshot before refinement, same reasoning as `transcribe_video`: this
    # is a real aligner every time (the command requires `--aligner`), so its
    # `score_scale` applies to every word whose score it actually supplied.
    pre_refine_scores = [span.score for span in ordered]
    aligner_scale = engine.score_scale
    if refine:
        ordered = refine_boundaries(
            samples,
            sr,
            ordered,
            speech=[(segment.start_ms, segment.end_ms) for segment in speech],
        )
        ordered = _score_boundaries(samples, sr, ordered)
    else:
        ordered = enforce_monotonic(ordered, total_ms=total_ms)

    scales = [
        _validate_scale(resolve_word_scale(pre, refine=refine, aligner_scale=aligner_scale))
        for pre in pre_refine_scores
    ]

    base = str(rows[0][5]).split("+")[0]
    tag = engine_tag(base, aligner, refine)
    with db.transaction():
        # Utterances copy word timings, so they are stale; Part 4 rebuilds them.
        # Speaker labels are kept: the text and the ordinals did not change.
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.executemany(
            "UPDATE words SET start_ms = ?, end_ms = ?, align_score = ?, "
            "align_scale = ?, engine = ?, source = 'aligned' "
            "WHERE video_id = ? AND ord = ?",
            [
                (
                    span.start_ms,
                    span.end_ms,
                    span.score,
                    scale,
                    tag,
                    video_id,
                    int(rows[index][0]),
                )
                for index, (span, scale) in enumerate(zip(ordered, scales, strict=True))
            ],
        )
    reported_scale, reported_median = _summarize_scores(ordered, scales)
    return TranscribeOutcome(
        video_id=video_id,
        n_words=len(ordered),
        n_chunks=len(chunks),
        engine=tag,
        source="aligned",
        speakers_lost=0,
        median_align_score=reported_median,
        align_scale=reported_scale,
        align_device=engine.device,
        align_notes=tuple(align_notes),
    )
