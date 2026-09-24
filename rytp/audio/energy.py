"""Sample access, short-time energy, and the measured word boundary (design §6).

A transcriber's own word boundaries are not usable for cutting. On the owner's
existing data 78.7% of words end exactly where the next one begins, so every
pause has been absorbed into a neighbouring word and some words have zero
duration. The boundary this module produces is *measured*: the quietest point
between two words, snapped to a zero crossing. Two adjacent words sharing one
boundary is then correct, because the shared point is real silence rather than
a guess.

Everything here works on an in-memory float32 array rather than a file. A
one-hour video is refined thousands of times; reading the audio once and
slicing is the difference between seconds and minutes.
"""
from __future__ import annotations

import itertools
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rytp import constants as C
from rytp.models import RytpError, Span


class AudioFormatError(RytpError):
    """The WAV is not the 16 kHz mono 16-bit PCM every audio stage expects."""


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV as float32 in [-1, 1], with its sample rate."""
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    if channels != 1:
        raise AudioFormatError(f"{path}: expected mono audio, found {channels} channels")
    if width != 2:
        raise AudioFormatError(f"{path}: expected 16-bit PCM, found {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def ms_to_index(ms: float, sr: int) -> int:
    """Millisecond position to sample index, rounded to nearest."""
    return round(ms * sr / 1000.0)


def index_to_ms(index: int, sr: int) -> int:
    """Sample index to millisecond position, rounded to nearest."""
    return round(index * 1000.0 / sr)


def frame_rms(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Root-mean-square energy of every frame, as float32.

    Computed from a cumulative sum of squares rather than a sliding window,
    because a sliding window over an hour of audio materialises tens of
    gigabytes while the cumulative sum needs one array the size of the input.
    """
    if frame_len <= 0 or hop <= 0 or samples.size < frame_len:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (samples.size - frame_len) // hop
    csum = np.square(samples, dtype=np.float64)
    np.cumsum(csum, out=csum)
    starts: np.ndarray = np.arange(n_frames, dtype=np.int64) * hop
    ends: np.ndarray = starts + frame_len
    before = np.where(starts > 0, csum[np.maximum(starts - 1, 0)], 0.0)
    totals = np.maximum(csum[ends - 1] - before, 0.0)
    return np.sqrt(totals / frame_len).astype(np.float32)


def to_db(rms: np.ndarray) -> np.ndarray:
    """RMS to dBFS, floored so digital silence is a number rather than -inf."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, C.DB_EPSILON))
    return np.maximum(db, C.DB_FLOOR).astype(np.float32)


def find_energy_minimum(
    samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float, *, prefer_ms: int
) -> int:
    """Millisecond of the quietest short-time frame in ``[lo_ms, hi_ms]``.

    Ties — which digital silence produces in bulk — resolve to the frame
    nearest ``prefer_ms``, so the result is deterministic and never drifts to
    the edge of the search window.
    """
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    if hi - lo < frame_len:
        return prefer_ms
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    if rms.size == 0:
        return prefer_ms
    centres: np.ndarray = lo + np.arange(rms.size, dtype=np.int64) * hop + frame_len // 2
    tied: np.ndarray = np.flatnonzero(rms <= float(rms.min()) + C.ENERGY_TIE_RMS)
    prefer_index = ms_to_index(prefer_ms, sr)
    pick = int(tied[int(np.argmin(np.abs(centres[tied] - prefer_index)))])
    return index_to_ms(int(centres[pick]), sr)


def snap_to_zero_crossing(
    samples: np.ndarray, sr: int, at_ms: int, *, search_ms: int | None = None
) -> int:
    """Nudge a boundary onto the nearest zero crossing, then round to a millisecond.

    The database stores milliseconds and one millisecond is sixteen samples at
    16 kHz, so the crossing itself cannot be stored. Of the two millisecond
    marks bracketing it the quieter one is kept, which is the best a
    millisecond grid can do. The snap matters most when the minimum is not
    fully silent; inside real silence the rounding lands in silence anyway.
    """
    span = max(1, ms_to_index(C.ZERO_CROSSING_SEARCH_MS if search_ms is None else search_ms, sr))
    centre = ms_to_index(at_ms, sr)
    lo = max(1, centre - span)
    hi = min(samples.size, centre + span + 1)
    if hi - lo <= 1:
        return at_ms
    window = samples[lo - 1 : hi]
    crossings: np.ndarray = (
        np.flatnonzero(np.signbit(window[:-1]) != np.signbit(window[1:])) + lo
    )
    if crossings.size:
        index = int(crossings[int(np.argmin(np.abs(crossings - centre)))])
    else:
        # No crossing at all (digital silence, or a constant offset). Take the
        # quietest sample — and break the ties toward the centre, because
        # inside digital silence every sample ties and picking the window edge
        # would walk the boundary a few milliseconds left on every pass.
        magnitudes = np.abs(samples[lo:hi])
        tied: np.ndarray = (
            np.flatnonzero(magnitudes <= float(magnitudes.min()) + C.ENERGY_TIE_RMS) + lo
        )
        index = int(tied[int(np.argmin(np.abs(tied - centre)))])
    return _quietest_millisecond(samples, sr, index)


def _quietest_millisecond(samples: np.ndarray, sr: int, index: int) -> int:
    lower = index * 1000 // sr
    candidates = [
        ms for ms in (lower, lower + 1) if 0 <= ms_to_index(ms, sr) < samples.size
    ]
    if not candidates:
        return max(0, index_to_ms(index, sr))
    return min(candidates, key=lambda ms: (abs(float(samples[ms_to_index(ms, sr)])), ms))


def _median_db(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    return C.DB_FLOOR if rms.size == 0 else float(np.median(to_db(rms)))


def _min_db(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    return C.DB_FLOOR if rms.size == 0 else float(np.min(to_db(rms)))


def silence_depth_db(samples: np.ndarray, sr: int, boundary_ms: int) -> float:
    """How much quieter this boundary is than the speech around it, in dB.

    This is the number that answers "is there actually silence where the
    aligner claims a word ends" — the measurement ``transcribe compare``
    reports and the one that stands in for a per-word alignment confidence
    when the aligner does not provide one.
    """
    reference = max(
        _median_db(samples, sr, boundary_ms - C.SILENCE_DEPTH_REF_MS, boundary_ms),
        _median_db(samples, sr, boundary_ms, boundary_ms + C.SILENCE_DEPTH_REF_MS),
    )
    floor = _min_db(
        samples,
        sr,
        boundary_ms - C.SILENCE_DEPTH_WINDOW_MS,
        boundary_ms + C.SILENCE_DEPTH_WINDOW_MS,
    )
    if floor <= C.DB_FLOOR:
        # Digital silence at the boundary: as good as a boundary gets.
        return C.SILENCE_DEPTH_FULL_DB
    return max(0.0, reference - floor)


def boundary_quality(depth_db: float) -> float:
    """Silence depth as a 0..1 score, for ``words.align_score``."""
    return min(1.0, max(0.0, depth_db / C.SILENCE_DEPTH_FULL_DB))


def _speech_edges(speech: Sequence[tuple[int, int]]) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for start_ms, end_ms in speech:
        edges.append((start_ms, "start"))
        edges.append((end_ms, "end"))
    return sorted(edges)


def _refine_one(
    samples: np.ndarray,
    sr: int,
    *,
    claimed: int,
    lo_ms: int,
    hi_ms: int,
    edges: Sequence[tuple[int, str]],
    total_ms: int,
) -> int:
    lo = max(0, min(lo_ms, claimed))
    hi = min(total_ms, max(hi_ms, claimed))
    if hi <= lo:
        return max(0, min(claimed, total_ms))
    near = [edge for edge in edges if lo <= edge[0] <= hi]
    gaps = [
        (end_ms, start_ms)
        for end_ms, end_kind in near
        for start_ms, start_kind in near
        if end_kind == "end" and start_kind == "start" and start_ms >= end_ms
    ]
    if gaps:
        # A speech segment ends and the next begins inside the window: the true
        # silence is the interval between them, so search there rather than
        # snapping to either edge.
        end_ms, start_ms = min(gaps, key=lambda pair: abs((pair[0] + pair[1]) // 2 - claimed))
        lo, hi = end_ms, max(start_ms, end_ms + 1)
    elif len(near) == 1 and abs(near[0][0] - claimed) <= C.VAD_EDGE_SNAP_MS:
        # One speech edge, close by: it is measured silence on one side, which
        # beats any search (design §6).
        return near[0][0]
    minimum = find_energy_minimum(samples, sr, lo, hi, prefer_ms=claimed)
    return snap_to_zero_crossing(samples, sr, minimum)


def _refine_edge(
    *, claimed: int, edges: Sequence[tuple[int, str]], kind: str, total_ms: int
) -> int:
    """The very first start and the very last end: a speech edge, or nothing.

    There is no "between two words" to search at the outer edges, so an energy
    hunt there would only find the quietest ripple inside the first or last
    word and would move again on every pass. A nearby voice-activity edge is a
    real boundary and is taken; otherwise the aligner's claim stands.
    """
    candidates = [
        ms
        for ms, edge_kind in edges
        if edge_kind == kind and abs(ms - claimed) <= C.VAD_EDGE_SNAP_MS
    ]
    chosen = (
        min(candidates, key=lambda ms: (abs(ms - claimed), ms)) if candidates else claimed
    )
    return max(0, min(chosen, total_ms))


def refine_boundaries(
    samples: np.ndarray,
    sr: int,
    spans: Sequence[Span],
    *,
    speech: Sequence[tuple[int, int]] = (),
) -> list[Span]:
    """Replace every claimed boundary with a measured one (design §6).

    ``spans`` are the aligner's rough timings, in order. ``speech`` is the
    output of :func:`rytp.audio.vad.detect_speech` as ``(start_ms, end_ms)``
    pairs; where a speech edge sits near a claimed boundary it is used
    directly, because it is a true boundary for free.

    Boundaries are shared: word *i* ends exactly where word *i+1* begins. That
    is correct under this scheme, because the shared point is measured silence.
    """
    if not spans:
        return []
    total_ms = index_to_ms(samples.size, sr)
    edges = _speech_edges(speech)
    cuts: list[int] = []

    cuts.append(
        _refine_edge(
            claimed=spans[0].start_ms, edges=edges, kind="start", total_ms=total_ms
        )
    )
    for left, right in itertools.pairwise(spans):
        claimed = (left.end_ms + right.start_ms) // 2
        left_mid = (left.start_ms + left.end_ms) // 2
        right_mid = (right.start_ms + right.end_ms) // 2
        cuts.append(
            _refine_one(
                samples,
                sr,
                claimed=claimed,
                lo_ms=max(left_mid, claimed - C.BOUNDARY_SEARCH_MS),
                hi_ms=min(right_mid, claimed + C.BOUNDARY_SEARCH_MS),
                edges=edges,
                total_ms=total_ms,
            )
        )
    cuts.append(
        _refine_edge(claimed=spans[-1].end_ms, edges=edges, kind="end", total_ms=total_ms)
    )
    refined = [
        Span(start_ms=cuts[i], end_ms=cuts[i + 1], score=spans[i].score)
        for i in range(len(spans))
    ]
    return enforce_monotonic(refined, total_ms=total_ms)


def enforce_monotonic(spans: Sequence[Span], *, total_ms: int) -> list[Span]:
    """Guarantee ``start < end <= next start`` and a minimum word duration.

    A shared boundary survives untouched; only a degenerate word is widened,
    which pushes its neighbour's start forward by the same amount.
    """
    out: list[Span] = []
    previous_end = 0
    for span in spans:
        start = max(span.start_ms, previous_end)
        end = max(span.end_ms, start + C.MIN_WORD_DURATION_MS)
        if total_ms > 0:
            end = min(end, total_ms)
            start = min(start, max(0, end - 1))
        out.append(Span(start_ms=start, end_ms=end, score=span.score))
        previous_end = end
    return out
