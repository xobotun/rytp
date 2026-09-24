"""Energy-based voice activity detection and chunk planning (design §6).

Two jobs. First, a transcriber caps a single call — GigaAM v3 at 25 s — so an
hour of audio has to be split, and split *at silence*, because a cut through a
word costs that word. Second, the edge of a detected speech segment is a true
boundary for free: there is measured silence on one side of it, which is
exactly what boundary refinement is hunting for.

No model is used. A percentile noise floor with a hysteresis band does both
jobs, has no dependency, and runs over a whole corpus without touching the
GPU. Anything smarter can be swapped in behind :func:`detect_speech`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rytp import constants as C
from rytp.audio.energy import frame_rms, index_to_ms, ms_to_index, to_db


@dataclass(frozen=True)
class SpeechSegment:
    """A stretch of audio that contains speech."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class AudioChunk:
    """One window handed to a transcriber, within its per-call cap."""

    ord: int
    start_ms: int
    end_ms: int


def _hysteresis_runs(db: np.ndarray, enter: float, leave: float) -> list[tuple[int, int]]:
    """Frame index runs that go above ``enter`` and stay above ``leave``.

    A plain loop: an hour of audio is 360k frames, which costs a fraction of a
    second, and the state machine reads far more clearly than a vectorised
    equivalent would.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(db.size):
        value = float(db[index])
        if start is None:
            if value >= enter:
                start = index
        elif value < leave:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, db.size))
    return runs


def _merge_close(spans: Sequence[tuple[int, int]], max_gap_ms: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= max_gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detect_speech(
    samples: np.ndarray, sr: int, *, pad_ms: int | None = None
) -> list[SpeechSegment]:
    """Find the stretches of speech in ``samples``.

    ``pad_ms`` defaults to :data:`rytp.constants.VAD_PAD_MS`; acoustics passes
    0, because a padded segment end would put the reverberation tail it wants
    to measure inside the segment.
    """
    frame_len = max(1, ms_to_index(C.VAD_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.VAD_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    total_ms = index_to_ms(samples.size, sr)
    if db.size == 0:
        return []
    floor = float(np.percentile(db, C.VAD_NOISE_PERCENTILE))
    runs = _hysteresis_runs(db, floor + C.VAD_ENTER_DB, floor + C.VAD_EXIT_DB)
    raw = [
        (
            index_to_ms(first * hop, sr),
            min(index_to_ms(last * hop + frame_len, sr), total_ms),
        )
        for first, last in runs
    ]
    kept = [
        span
        for span in _merge_close(raw, C.VAD_MIN_SILENCE_MS)
        if span[1] - span[0] >= C.VAD_MIN_SPEECH_MS
    ]
    if not raw:
        # No hysteresis run crossed the noise floor at all. Two very
        # different files land here: genuine near-total silence, and a file
        # so uniformly loud (or loud enough for long enough) that the
        # percentile floor itself sits inside speech and 12 dB above it is
        # unreachable. The two are told apart against the *global* floor,
        # not the local one: comparing to the local percentile floor would
        # make both cases fail this check for the same reason they landed
        # here in the first place.
        #
        # Gated on `raw`, not on `kept` — a short burst amid real silence
        # produces a genuine `raw` run that the min-speech-length filter
        # below correctly discards; it must not be re-judged as "no silence
        # anywhere" just because nothing survived that filter.
        if float(np.max(db)) > C.DB_FLOOR + C.VAD_ENTER_DB:
            return [SpeechSegment(start_ms=0, end_ms=total_ms)]
        return []
    if not kept:
        return []
    pad = C.VAD_PAD_MS if pad_ms is None else pad_ms
    padded = [(max(0, start - pad), min(total_ms, end + pad)) for start, end in kept]
    return [SpeechSegment(start_ms=start, end_ms=end) for start, end in _merge_close(padded, 0)]


def plan_chunks(
    speech: Sequence[SpeechSegment],
    *,
    total_ms: int,
    max_chunk_ms: int | None = None,
    target_ms: int | None = None,
) -> list[AudioChunk]:
    """Pack speech segments into windows a transcriber will accept.

    Segments are grouped while the group stays under ``target_ms``, so a chunk
    normally begins and ends in silence. ``max_chunk_ms`` is the transcriber's
    hard per-call cap (design §6: 25 s for GigaAM v3) and is never exceeded.
    """
    max_ms = C.VAD_CHUNK_MAX_MS if max_chunk_ms is None else max_chunk_ms
    target = min(C.VAD_CHUNK_TARGET_MS if target_ms is None else target_ms, max_ms)
    if not speech:
        return []

    grouped: list[tuple[int, int]] = []
    start, end = speech[0].start_ms, speech[0].end_ms
    for segment in speech[1:]:
        if segment.end_ms - start <= target:
            end = segment.end_ms
        else:
            grouped.append((start, end))
            start, end = segment.start_ms, segment.end_ms
    grouped.append((start, end))

    pieces: list[tuple[int, int]] = []
    for group_start, group_end in grouped:
        span = group_end - group_start
        if span <= max_ms:
            pieces.append((group_start, group_end))
            continue
        # One unbroken speech run longer than the cap. Split it into equal
        # pieces: this cuts mid-speech and costs at most the word straddling
        # each cut, but a run this long without a 150 ms silence is rare.
        count = -(-span // max_ms)
        step = -(-span // count)
        pieces.extend(
            (group_start + i * step, min(group_start + (i + 1) * step, group_end))
            for i in range(count)
        )
    return [
        AudioChunk(ord=index, start_ms=max(0, piece_start), end_ms=min(piece_end, total_ms))
        for index, (piece_start, piece_end) in enumerate(pieces)
    ]
