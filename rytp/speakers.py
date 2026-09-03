"""Speaker roster management and per-video diarizer→speaker mapping.

DESIGN §4 (speakers table, videos_speaker_map), §6 (speakers group
subcommands), §8 (pause-stats computation).

Public surface:

* :func:`add_speaker` — insert into the global ``speakers`` table.
* :func:`list_speakers` — return all roster entries.
* :func:`map_diarizer_to_speaker` — assign a canonical speaker to one
  raw diarizer label for one video.
* :func:`video_speaker_map` — return the full per-video map.
* :func:`recompute_pause_stats` — recompute ``speaker_pause_stats``
  rows from the ``words`` table.
* :class:`SpeakerMapper` — backing state for the textual TUI mapper.
"""
from __future__ import annotations

from datetime import UTC as _UTC, datetime as _dt
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from rytp.db import Database
from rytp.models import Speaker


# ---------------------------------------------------------------------------
# Global roster
# ---------------------------------------------------------------------------


def add_speaker(
    db: Database,
    label: str,
    *,
    aliases: Iterable[str] = (),
    notes: str | None = None,
) -> int:
    """Insert a speaker into the global roster.

    Idempotent on ``label``: if the speaker already exists, returns the
    existing id without modification (use ``update_speaker`` to change
    aliases/notes).

    Returns the speaker id.
    """
    existing = db.conn.execute(
        "SELECT id FROM speakers WHERE label = ?", (label,)
    ).fetchone()
    if existing:
        return existing["id"]
    cur = db.conn.execute(
        """
        INSERT INTO speakers (label, aliases_json, notes, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            label,
            json.dumps(list(aliases)),
            notes,
            _dt.now(_UTC).isoformat(),
        ),
    )
    db.conn.commit()
    return cur.lastrowid


def update_speaker(
    db: Database,
    speaker_id: int,
    *,
    aliases: Iterable[str] | None = None,
    notes: str | None = None,
) -> None:
    """Update mutable fields of an existing speaker row."""
    sets: list[str] = []
    params: list[object] = []
    if aliases is not None:
        sets.append("aliases_json = ?")
        params.append(json.dumps(list(aliases)))
    if notes is not None:
        sets.append("notes = ?")
        params.append(notes)
    if not sets:
        return
    params.append(speaker_id)
    db.conn.execute(
        f"UPDATE speakers SET {', '.join(sets)} WHERE id = ?",
        params,
    )
    db.conn.commit()


def list_speakers(db: Database) -> list[Speaker]:
    """Return the global roster, ordered by label."""
    rows = db.conn.execute("SELECT * FROM speakers ORDER BY label").fetchall()
    return [Speaker.from_row(r) for r in rows]


def get_speaker_by_label(db: Database, label: str) -> Speaker | None:
    row = db.conn.execute(
        "SELECT * FROM speakers WHERE label = ?", (label,)
    ).fetchone()
    return Speaker.from_row(row) if row else None


# ---------------------------------------------------------------------------
# Per-video diarizer→speaker mapping
# ---------------------------------------------------------------------------


def map_diarizer_to_speaker(
    db: Database,
    video_id: int,
    diarizer_speaker: str,
    speaker_id: int,
) -> None:
    """Assign a canonical speaker to one raw diarizer label for one video.

    Upserts into ``videos_speaker_map`` and (because the canonical
    speaker is now known for that video+label) writes through to
    ``words.speaker_id`` for every row of that video whose
    ``diarizer_speaker`` matches.

    Passing ``speaker_id=None`` clears the mapping for that label.
    """
    with db.transaction():
        if speaker_id is None:
            db.conn.execute(
                """
                DELETE FROM videos_speaker_map
                WHERE video_id = ? AND diarizer_speaker = ?
                """,
                (video_id, diarizer_speaker),
            )
            db.conn.execute(
                """
                UPDATE words SET speaker_id = NULL
                WHERE video_id = ? AND diarizer_speaker = ?
                """,
                (video_id, diarizer_speaker),
            )
        else:
            db.conn.execute(
                """
                INSERT INTO videos_speaker_map (video_id, diarizer_speaker, speaker_id)
                VALUES (?, ?, ?)
                ON CONFLICT(video_id, diarizer_speaker)
                DO UPDATE SET speaker_id = excluded.speaker_id
                """,
                (video_id, diarizer_speaker, speaker_id),
            )
            db.conn.execute(
                """
                UPDATE words SET speaker_id = ?
                WHERE video_id = ? AND diarizer_speaker = ?
                """,
                (speaker_id, video_id, diarizer_speaker),
            )


def video_speaker_map(db: Database, video_id: int) -> dict[str, int | None]:
    """Return ``{diarizer_speaker: canonical_speaker_id or None}`` for the video."""
    rows = db.conn.execute(
        "SELECT diarizer_speaker, speaker_id FROM videos_speaker_map WHERE video_id = ?",
        (video_id,),
    ).fetchall()
    return {r["diarizer_speaker"]: r["speaker_id"] for r in rows}


def distinct_diarizer_speakers(db: Database, video_id: int) -> list[str]:
    """Return the unique raw diarizer labels seen in ``words`` for this video.

    The labels are sorted alphabetically; this is the input to the TUI
    mapper's left pane.
    """
    rows = db.conn.execute(
        """
        SELECT DISTINCT diarizer_speaker FROM words
        WHERE video_id = ? AND diarizer_speaker IS NOT NULL
        ORDER BY diarizer_speaker
        """,
        (video_id,),
    ).fetchall()
    return [r["diarizer_speaker"] for r in rows]


# ---------------------------------------------------------------------------
# Pause-stats computation (DESIGN §8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PauseStats:
    """A computed summary of inter-word pause durations for one speaker."""

    n_samples: int
    mean_ms: float
    std_ms: float
    p10_ms: float
    p25_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


def compute_pause_stats(db: Database, speaker_id: int) -> PauseStats | None:
    """Compute :class:`PauseStats` for ``speaker_id`` from the ``words`` table.

    Walks every word of every video mapped to ``speaker_id`` and
    treats each consecutive pair **within the same video** as one
    inter-word pause sample. Cross-video gaps are dropped — the
    rationale is that two videos were not uttered in succession, so
    the "pause" between them isn't a real speaker timing.

    Returns ``None`` if there are no samples (below
    :data:`C.MIN_SAMPLES_FOR_PAUSE_STATS`).
    """
    rows = db.conn.execute(
        """
        SELECT w.video_id, w.start_ms, w.end_ms
        FROM words w
        WHERE w.speaker_id = ?
        ORDER BY w.video_id, w.start_ms
        """,
        (speaker_id,),
    ).fetchall()
    samples: list[int] = []
    last_video_id: int | None = None
    last_end_ms = 0
    for r in rows:
        if r["video_id"] != last_video_id:
            last_video_id = r["video_id"]
            last_end_ms = r["end_ms"]
            continue
        pause = r["start_ms"] - last_end_ms
        if pause >= 0:
            samples.append(pause)
        last_end_ms = r["end_ms"]
    if not samples:
        return None
    samples.sort()
    return _summarize(samples)


def _summarize(sorted_samples: list[int]) -> PauseStats:
    n = len(sorted_samples)

    def pct(p: float) -> float:
        # Linear interpolation between order statistics.
        if n == 1:
            return float(sorted_samples[0])
        rank = p * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac

    mean = sum(sorted_samples) / n
    var = sum((x - mean) ** 2 for x in sorted_samples) / n
    return PauseStats(
        n_samples=n,
        mean_ms=mean,
        std_ms=var**0.5,
        p10_ms=pct(0.10),
        p25_ms=pct(0.25),
        p50_ms=pct(0.50),
        p75_ms=pct(0.75),
        p90_ms=pct(0.90),
        min_ms=float(sorted_samples[0]),
        max_ms=float(sorted_samples[-1]),
    )


def recompute_pause_stats(db: Database) -> int:
    """Recompute ``speaker_pause_stats`` for every speaker that has words.

    Returns the number of rows written.

    In v1 we recompute every speaker that has any mapped words, with no
    dirty-flag optimization (DESIGN §8 mentions a future optimization).
    """
    speakers = db.conn.execute(
        """
        SELECT DISTINCT speaker_id FROM words WHERE speaker_id IS NOT NULL
        """
    ).fetchall()
    n_written = 0
    now = _dt.now(_UTC).isoformat()
    for row in speakers:
        sid = row["speaker_id"]
        stats = compute_pause_stats(db, sid)
        if stats is None:
            continue
        db.conn.execute(
            """
            INSERT INTO speaker_pause_stats (
                speaker_id, n_samples, mean_ms, std_ms,
                p10_ms, p25_ms, p50_ms, p75_ms, p90_ms,
                min_ms, max_ms, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(speaker_id) DO UPDATE SET
                n_samples = excluded.n_samples,
                mean_ms = excluded.mean_ms,
                std_ms = excluded.std_ms,
                p10_ms = excluded.p10_ms,
                p25_ms = excluded.p25_ms,
                p50_ms = excluded.p50_ms,
                p75_ms = excluded.p75_ms,
                p90_ms = excluded.p90_ms,
                min_ms = excluded.min_ms,
                max_ms = excluded.max_ms,
                updated_at = excluded.updated_at
            """,
            (
                sid,
                stats.n_samples,
                stats.mean_ms,
                stats.std_ms,
                stats.p10_ms,
                stats.p25_ms,
                stats.p50_ms,
                stats.p75_ms,
                stats.p90_ms,
                stats.min_ms,
                stats.max_ms,
                now,
            ),
        )
        n_written += 1
    db.conn.commit()
    return n_written


def get_pause_stats(db: Database, speaker_id: int) -> PauseStats | None:
    """Return the cached pause stats for a speaker, or None if missing."""
    row = db.conn.execute(
        "SELECT * FROM speaker_pause_stats WHERE speaker_id = ?", (speaker_id,)
    ).fetchone()
    if row is None:
        return None
    return PauseStats(
        n_samples=row["n_samples"],
        mean_ms=row["mean_ms"],
        std_ms=row["std_ms"],
        p10_ms=row["p10_ms"],
        p25_ms=row["p25_ms"],
        p50_ms=row["p50_ms"],
        p75_ms=row["p75_ms"],
        p90_ms=row["p90_ms"],
        min_ms=row["min_ms"],
        max_ms=row["max_ms"],
    )


# ---------------------------------------------------------------------------
# TUI mapper state
# ---------------------------------------------------------------------------


@dataclass
class SpeakerMapper:
    """Backing state for the textual two-pane mapper (DESIGN §6).

    The actual textual app lives in :mod:`rytp.tui.app`; this class is
    the testable, headless side of the same logic. Tests use the
    dataclass methods directly; the TUI binds them to keystrokes.
    """

    db: Database
    video_id: int

    def left_pane(self) -> list[str]:
        """Return the raw diarizer labels seen in this video."""
        return distinct_diarizer_speakers(self.db, self.video_id)

    def right_pane(self) -> list[Speaker]:
        """Return the global roster."""
        return list_speakers(self.db)

    def current_mapping(self) -> dict[str, int | None]:
        return video_speaker_map(self.db, self.video_id)

    def assign(self, diarizer_speaker: str, speaker_id: int | None) -> None:
        map_diarizer_to_speaker(self.db, self.video_id, diarizer_speaker, speaker_id)

    def fuzzy_match(self, diarizer_speaker: str) -> Speaker | None:
        """Suggest a roster entry whose label/aliases share a token with
        the raw diarizer label (case-insensitive substring).
        """
        needle = diarizer_speaker.lower()
        for s in self.right_pane():
            if s.label.lower() in needle or needle in s.label.lower():
                return s
            for alias in s.aliases:
                al = alias.lower()
                if al and (al in needle or needle in al):
                    return s
        return None