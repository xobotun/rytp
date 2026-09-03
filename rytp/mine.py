"""The mine stage — datamining the corpus.

DESIGN §7: given a query string and a cohesion level, find candidate
word matches via FTS5, expand into windows (single-word at low
cohesion, multi-word longest-uninterrupted at high cohesion), score
each window by length + lexical density + spectral similarity to
neighbors, and write the result set to the ``clips`` table.

Three cohesion levels map to different window budgets:

* ``LOW`` — each FTS hit is its own window. Useful for first-pass
  data exploration.
* ``MED`` — 5-second windows, gaps < 1 s. Default. Catches short
  phrases.
* ``HIGH`` — 15-second windows, gaps < 500 ms. Forces tight
  uninterruptedness; best for catchphrases and repeated lines.

The scoring formula (DESIGN §7) is a fixed linear combination:

    score = SCORE_WEIGHT_LENGTH * length_norm
          + SCORE_WEIGHT_DENSITY * density_norm
          + SCORE_WEIGHT_SPECTRAL * spectral_norm

where each component is normalized to ``[0, 1]``. Tuning any of the
weights or the references in :mod:`rytp.constants` is a one-line
change; the algorithm itself does not need to be re-derived.

Public surface:

* :func:`mine` — produce a ``clips`` set for a query.
* :func:`search_words` — FTS5 candidate finder (substring prefix).
* :func:`expand_into_windows` — windowing by cohesion level.
* :func:`score_windows` — score candidates.
* :class:`Cohesion` — enum-like triplet ``LOW | MED | HIGH``.
* :class:`Window` — one candidate clip.
"""
from __future__ import annotations

from datetime import UTC as _UTC, datetime as _dt
import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass

from rytp import constants as C
from rytp.config import paths as config_paths
from rytp.db import Database
from rytp.models import normalize_text
from rytp.spectrogram import (
    ClipFeatures,
    clip_similarity,
    get_or_compute_clip_features,
)


class Cohesion(str, enum.Enum):
    """Cohesion level for window expansion.

    The string values match the ``--cohesion`` CLI flag values
    (``low``, ``med``, ``high``) — typer passes the string and we
    convert via :meth:`Cohesion`.
    """

    LOW = "low"
    MED = "med"
    HIGH = "high"


@dataclass(frozen=True)
class Window:
    """One candidate clip produced by the mine stage.

    Attributes:
        video_id: FK to the ``videos`` table.
        start_ms: Clip start, inclusive.
        end_ms: Clip end, exclusive.
        text: Concatenated text of every word in the window.
    """

    video_id: int
    start_ms: int
    end_ms: int
    text: str


def search_words(
    db: Database, query: str, *, limit: int = C.DEFAULT_SEARCH_LIMIT
) -> list[tuple[int, int, str]]:
    """Find ``words`` rows whose normalized text matches ``query``.

    Returns ``(video_id, word_id, text)`` triples — the third element
    is the **original** text (not the normalized form).

    The query is normalized before the lookup so the user can type
    natural text ("Hello, World!") and still hit the index. Each
    token is rewritten as a prefix query (``word*``) — FTS5's default
    term matching only hits whole words, but DESIGN §7 specifies
    substring search, so we approximate that with prefix matching.

    For more aggressive substring matching you'd need to rebuild the
    FTS5 index with ``prefix=2,3,4`` (so ``wor*`` matches ``word``).
    Not done in v1; the prefix form catches every common case.

    Args:
        db: Open Database with the FTS5 shadow table populated.
        query: User search string.
        limit: Maximum rows returned.

    Returns:
        ``(video_id, word_id, text)`` triples.
    """
    norm = normalize_text(query)
    if not norm:
        return []
    tokens = [t for t in norm.split(" ") if t]
    if not tokens:
        return []
    # Convert "hello world" → "hello* world*" for prefix substring
    # matching. Empty tokens (from double spaces) are dropped.
    fts_query = " ".join(f"{t}*" for t in tokens)
    rows = db.conn.execute(
        """
        SELECT w.video_id, w.id, w.text
        FROM words_fts fts
        JOIN words w ON w.id = fts.rowid
        WHERE words_fts MATCH ?
        ORDER BY fts.rowid
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [(r["video_id"], r["id"], r["text"]) for r in rows]


def expand_into_windows(
    db: Database,
    hits: Iterable[tuple[int, int, str]],
    *,
    cohesion: Cohesion = Cohesion.MED,
) -> list[Window]:
    """Expand candidate word hits into windows based on cohesion.

    * ``LOW`` — each hit is its own window (just the word itself).
    * ``MED`` — extend forward to the next words up to
      :data:`C.COHESION_MED_WINDOW_MS` total, stopping at any gap
      > :data:`C.COHESION_MED_MAX_GAP_MS`.
    * ``HIGH`` — same as MED but
      :data:`C.COHESION_HIGH_WINDOW_MS` /
      :data:`C.COHESION_HIGH_MAX_GAP_MS`.

    Implementation note: for MED/HIGH we fetch the **full timeline**
    of words in each video (not just the FTS hits), so the forward
    walk has access to every neighboring word. Without this, a
    single-word FTS hit would have nothing to expand into.

    Returns windows sorted by ``(video_id, start_ms)``.
    """
    if cohesion == Cohesion.HIGH:
        max_window_ms = C.COHESION_HIGH_WINDOW_MS
        max_gap_ms = C.COHESION_HIGH_MAX_GAP_MS
    elif cohesion == Cohesion.MED:
        max_window_ms = C.COHESION_MED_WINDOW_MS
        max_gap_ms = C.COHESION_MED_MAX_GAP_MS
    else:  # LOW
        max_window_ms = C.COHESION_LOW_WINDOW_MS
        max_gap_ms = C.COHESION_LOW_MAX_GAP_MS

    # Group hits by video.
    hits_by_video: dict[int, list[tuple[int, str]]] = {}
    for video_id, word_id, text in hits:
        hits_by_video.setdefault(video_id, []).append((word_id, text))

    windows: list[Window] = []
    for video_id, hit_items in hits_by_video.items():
        if not hit_items:
            continue
        hit_ids = {wid for wid, _ in hit_items}

        # For MED/HIGH we need the full timeline; for LOW we only need
        # the hits themselves.
        if cohesion == Cohesion.LOW:
            placeholders = ",".join("?" * len(hit_ids))
            rows = db.conn.execute(
                f"""
                SELECT id, video_id, start_ms, end_ms, text FROM words
                WHERE video_id = ? AND id IN ({placeholders})
                ORDER BY start_ms
                """,
                (video_id, *hit_ids),
            ).fetchall()
        else:
            rows = db.conn.execute(
                """
                SELECT id, video_id, start_ms, end_ms, text FROM words
                WHERE video_id = ?
                ORDER BY start_ms
                """,
                (video_id,),
            ).fetchall()

        rows = list(rows)
        if not rows:
            continue

        if cohesion == Cohesion.LOW:
            for r in rows:
                windows.append(
                    Window(
                        video_id=r["video_id"],
                        start_ms=r["start_ms"],
                        end_ms=r["end_ms"],
                        text=r["text"],
                    )
                )
        else:
            # For each FTS hit, locate it in the timeline and walk
            # forward while within budget.
            hit_id_to_row = {r["id"]: r for r in rows}
            for hit_id, _text in hit_items:
                if hit_id not in hit_id_to_row:
                    continue
                start_row = hit_id_to_row[hit_id]
                start = start_row["start_ms"]
                end = start_row["end_ms"]
                text = start_row["text"]
                idx = rows.index(start_row)
                j = idx + 1
                while j < len(rows):
                    nxt = rows[j]
                    gap = nxt["start_ms"] - end
                    if gap > max_gap_ms:
                        break
                    new_end = nxt["end_ms"]
                    if new_end - start > max_window_ms:
                        break
                    end = new_end
                    text = text + " " + nxt["text"]
                    j += 1
                windows.append(
                    Window(
                        video_id=video_id,
                        start_ms=start,
                        end_ms=end,
                        text=text,
                    )
                )

    windows.sort(key=lambda w: (w.video_id, w.start_ms))
    return windows


def score_windows(
    db: Database,
    windows: list[Window],
) -> list[tuple[Window, float]]:
    """Score each window by length, lexical density, and spectral
    similarity to neighbors.

    Final score = ``SCORE_WEIGHT_LENGTH * length_norm``
                + ``SCORE_WEIGHT_DENSITY * density_norm``
                + ``SCORE_WEIGHT_SPECTRAL * spectral_norm``.

    * ``length_norm`` = log(1 + duration_ms) / log(1 +
      ``LENGTH_NORM_REFERENCE_MS``). The log curve compresses long
      clips so a 5-minute window doesn't dominate a 10-second one.
    * ``density_norm`` = ``min(1.0, words_per_second / DENSITY_NORM_WPS)``.
    * ``spectral_norm`` = mean cosine similarity to the
      ``SPECTRAL_NEIGHBOR_K`` nearest other windows in the same
      video, scaled from ``[-1, 1]`` to ``[0, 1]``.

    Args:
        db: Database to read/write ``clip_features`` rows.
        windows: Candidates from :func:`expand_into_windows`.

    Returns:
        ``(Window, score)`` pairs sorted by score, descending.
    """
    if not windows:
        return []

    by_video: dict[int, list[tuple[int, Window]]] = {}
    for idx, w in enumerate(windows):
        by_video.setdefault(w.video_id, []).append((idx, w))

    # Pre-fetch features per video for all the windows.
    features: dict[int, ClipFeatures | None] = {}
    for video_id, items in by_video.items():
        audio = config_paths.audio / f"{video_id}.wav"
        if not audio.exists():
            features[video_id] = None
            continue
        feats: list[ClipFeatures] = []
        for _, w in items:
            try:
                f = get_or_compute_clip_features(
                    db, video_id, audio, w.start_ms, w.end_ms
                )
                feats.append(f)
            except Exception:
                feats.append(None)  # type: ignore[arg-type]
        for (idx, _), f in zip(items, feats):
            features[idx] = f

    scores: list[tuple[Window, float]] = []
    for idx, w in enumerate(windows):
        duration_ms = max(1, w.end_ms - w.start_ms)

        length_norm = math.log1p(duration_ms) / math.log1p(
            C.LENGTH_NORM_REFERENCE_MS
        )
        n_words = max(1, len(w.text.split()))
        words_per_second = n_words / (duration_ms / C.SECONDS_TO_MS)
        density_norm = min(1.0, words_per_second / C.DENSITY_NORM_WPS)

        # Spectral similarity to K nearest neighbors in the same video.
        sims: list[float] = []
        f_self = features.get(idx)
        if f_self is not None:
            for jdx in by_video.get(w.video_id, []):
                if jdx[0] == idx:
                    continue
                other = features.get(jdx[0])
                if other is None:
                    continue
                sims.append(clip_similarity(f_self, other))
                if len(sims) >= C.SPECTRAL_NEIGHBOR_K:
                    break
        spectral = (sum(sims) / len(sims)) if sims else 0.0
        spectral_norm = (spectral + 1.0) / 2.0  # [-1, 1] → [0, 1]

        final = (
            C.SCORE_WEIGHT_LENGTH * length_norm
            + C.SCORE_WEIGHT_DENSITY * density_norm
            + C.SCORE_WEIGHT_SPECTRAL * spectral_norm
        )
        scores.append((w, float(final)))

    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def mine(
    db: Database,
    query: str,
    *,
    cohesion: Cohesion | str = Cohesion.MED,
    max_clips: int = C.DEFAULT_MAX_CLIPS,
) -> list[int]:
    """End-to-end mine. Returns the inserted clip ids.

    Writes one row per chosen window into the ``clips`` table.

    Args:
        db: Open Database.
        query: Search string (will be normalized).
        cohesion: One of :class:`Cohesion` — also accepts the string
            value for convenience with CLI flags.
        max_clips: Maximum number of clips written (top-N by score).

    Returns:
        Inserted clip row ids, in score order.
    """
    if isinstance(cohesion, str):
        cohesion = Cohesion(cohesion)

    hits = search_words(db, query)
    windows = expand_into_windows(db, hits, cohesion=cohesion)
    if not windows:
        return []

    scored = score_windows(db, windows)
    chosen = scored[:max_clips]

    now = _dt.now(_UTC).isoformat()
    ids: list[int] = []
    for w, _score in chosen:
        cur = db.conn.execute(
            """
            INSERT INTO clips (video_id, start_ms, end_ms, source_query, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (w.video_id, w.start_ms, w.end_ms, query, now),
        )
        ids.append(cur.lastrowid)
    db.conn.commit()
    return ids