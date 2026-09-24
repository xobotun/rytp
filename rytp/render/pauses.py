"""Between-word pause statistics. design §9.

The gap the render inserts between two fragments defaults to how long
this speaker actually pauses between words. That is only measurable on
aligned-tier words: caption rows have no end time at all (contracts §3),
and a transcriber's own timings are worse than useless here — design §6
measured 78.7% of Whisper's word gaps at exactly zero, because it sets
``word[i].end == word[i+1].start`` and absorbs every pause into the
neighbouring word.

So this module does not trust its own input. It measures, then asks
whether the distribution it measured could possibly be real, and says
plainly when it could not. A degenerate distribution falls back to a
constant and the reason reaches the report.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database

_COLUMNS = "video_id, ord, start_ms, end_ms, video_speaker_id"


@dataclass(frozen=True)
class PauseStats:
    """What the corpus says about one speaker's between-word pauses."""

    scope: str  # "speaker" | "video_speaker" | "video"
    key: str  # roster label, label row id, or video id — for the report
    n_samples: int
    median_ms: int
    zero_fraction: float
    degenerate: bool
    reason: str  # empty when the distribution is usable

    @property
    def description(self) -> str:
        """One line for the report's notes section."""
        if self.degenerate:
            return (
                f"{self.scope} {self.key}: {self.reason}; "
                f"fell back to {C.PAUSE_FALLBACK_GAP_MS} ms"
            )
        return (
            f"{self.scope} {self.key}: median {self.median_ms} ms "
            f"from {self.n_samples} gaps"
        )


def _pairwise_gaps(rows: Sequence[sqlite3.Row]) -> list[int]:
    """Gaps between consecutive words of one speaker, in milliseconds.

    A pair counts only when the two words are adjacent ordinals in the
    same video and carry the same speaker label — otherwise the "gap" is
    a turn boundary or the seam of an unrelated passage. Gaps longer
    than :data:`C.PAUSE_MAX_GAP_MS` are silences, not pauses, and
    negative ones are overlapping boundaries; both are dropped.
    """
    gaps: list[int] = []
    previous: sqlite3.Row | None = None
    for row in rows:
        if (
            previous is not None
            and row["video_id"] == previous["video_id"]
            and row["ord"] == previous["ord"] + 1
            and row["video_speaker_id"] == previous["video_speaker_id"]
            and previous["end_ms"] is not None
        ):
            gap = int(row["start_ms"]) - int(previous["end_ms"])
            if 0 <= gap <= C.PAUSE_MAX_GAP_MS:
                gaps.append(gap)
        previous = row
    return gaps


def gaps_for_video(db: Database, video_id: int) -> list[int]:
    """Every between-word gap in one video's aligned words."""
    rows = db.conn.execute(
        f"SELECT {_COLUMNS} FROM words "
        "WHERE video_id = ? AND source = ? ORDER BY ord LIMIT ?",
        (video_id, C.ALIGNED_WORD_SOURCE, C.PAUSE_SAMPLE_LIMIT),
    ).fetchall()
    return _pairwise_gaps(rows)


def gaps_for_speaker_ids(db: Database, video_speaker_ids: Collection[int]) -> list[int]:
    """Gaps pooled over a set of diarized labels — the primitive.

    An empty set means "a speaker who is mapped to no video yet", which
    is no gaps rather than every gap. Contracts §5 pins that distinction
    for :func:`rytp.commands.resolve_speaker_filter`, and it holds here
    for the same reason: an ``IN ()`` that quietly matched everything is
    the classic silent-wrong-answer bug.
    """
    ids = tuple(video_speaker_ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = db.conn.execute(
        f"SELECT {_COLUMNS} FROM words "
        f"WHERE video_speaker_id IN ({placeholders}) AND source = ? "
        "ORDER BY video_id, ord LIMIT ?",
        (*ids, C.ALIGNED_WORD_SOURCE, C.PAUSE_SAMPLE_LIMIT),
    ).fetchall()
    return _pairwise_gaps(rows)


def gaps_for_video_speaker(db: Database, video_speaker_id: int) -> list[int]:
    """Gaps for one diarized label of one video."""
    return gaps_for_speaker_ids(db, (video_speaker_id,))


def gaps_for_speaker_label(db: Database, label: str) -> list[int]:
    """Gaps for one person, pooled over every video they are mapped in.

    Pooling is what makes the statistic usable: one video rarely holds
    enough aligned words for a speaker to clear
    :data:`C.PAUSE_MIN_SAMPLES`, and a person's speaking rhythm is more
    theirs than the recording's.

    This is the *render-time* lookup, reached from a cut list's
    ``speaker_label``, and it is deliberately forgiving: an unknown
    label yields no samples and the caller falls back to per-video
    statistics (design §9). A command taking ``--speaker`` from a human
    must instead go through ``rytp.commands.resolve_speaker_filter``,
    which raises on an unresolvable name — a stale cut list should not
    abort a render, but a typed name that matches nobody must not
    silently widen to the whole video.
    """
    ids = [
        int(row["id"])
        for row in db.conn.execute(
            "SELECT vs.id FROM video_speakers vs "
            "JOIN speakers s ON s.id = vs.speaker_id WHERE s.label = ?",
            (label,),
        ).fetchall()
    ]
    return gaps_for_speaker_ids(db, ids)


def summarize_gaps(samples: Sequence[int], *, scope: str, key: str) -> PauseStats:
    """Reduce measured gaps to one number, refusing when they are junk."""
    n = len(samples)
    if n == 0:
        return PauseStats(
            scope=scope, key=key, n_samples=0, median_ms=0, zero_fraction=1.0,
            degenerate=True, reason="no aligned words to measure",
        )
    zeros = sum(1 for gap in samples if gap <= C.PAUSE_ZERO_GAP_MS)
    zero_fraction = zeros / n
    median_ms = round(statistics.median(samples))
    reason = ""
    if n < C.PAUSE_MIN_SAMPLES:
        reason = f"only {n} samples, fewer than {C.PAUSE_MIN_SAMPLES}"
    elif zero_fraction > C.PAUSE_DEGENERATE_ZERO_FRACTION:
        reason = (
            f"{zero_fraction:.0%} of gaps are exactly zero — the transcript's "
            "boundaries absorbed the pauses"
        )
    elif median_ms <= C.PAUSE_ZERO_GAP_MS:
        reason = "the median gap is zero"
    return PauseStats(
        scope=scope, key=key, n_samples=n, median_ms=median_ms,
        zero_fraction=zero_fraction, degenerate=bool(reason), reason=reason,
    )


def measure_pause_stats(
    db: Database,
    *,
    video_id: int,
    video_speaker_id: int | None = None,
    speaker_label: str | None = None,
) -> PauseStats:
    """Pause statistics at the narrowest scope with data behind it.

    design §9: per speaker where the video is diarized, per video
    otherwise. A named speaker pools across the corpus; a diarized label
    with no roster entry is still a distinct voice and is scoped to
    itself; everything else falls back to the video. An empty result at
    a narrow scope falls through to the wider one rather than reporting
    "no data" for a video that plainly has words.
    """
    if speaker_label:
        samples = gaps_for_speaker_label(db, speaker_label)
        if samples:
            return summarize_gaps(samples, scope="speaker", key=speaker_label)
    if video_speaker_id is not None:
        samples = gaps_for_video_speaker(db, video_speaker_id)
        if samples:
            return summarize_gaps(
                samples, scope="video_speaker", key=str(video_speaker_id)
            )
    return summarize_gaps(gaps_for_video(db, video_id), scope="video", key=str(video_id))


def applied_gap_ms(stats: PauseStats) -> int:
    """The gap to actually insert: the median, clamped, or the fallback."""
    if stats.degenerate:
        return C.PAUSE_FALLBACK_GAP_MS
    return max(
        C.PAUSE_MIN_APPLIED_GAP_MS, min(C.PAUSE_MAX_APPLIED_GAP_MS, stats.median_ms)
    )
