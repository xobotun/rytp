"""Build a corpus out of word rows, for the assembly tests.

Design §8 is pure logic over ``words`` and ``video_acoustics``: nothing in
part 5 opens an audio file, so nothing in its tests needs one. A test says
what was said, in which video, by whom, and these helpers turn that into
rows the matcher can walk.

Timings are synthetic but realistic in shape: every word lasts ``WORD_MS``
and every inter-word silence lasts ``GAP_MS``, so a run's span is
predictable arithmetic a test can assert on. Pass ``gap_ms`` explicitly to
simulate a long pause that must stop a run from extending across it.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from rytp.db import Database
from rytp.models import normalize_text, stem_text, utc_now_iso

#: Splits one whitespace-separated word on internal punctuation, mirroring
#: `normalize_text`'s own `[^\w]` fold so "кто-то" becomes two pieces. Kept
#: separate from `normalize_text` because it must preserve original casing
#: per piece, not fold it.
_PIECE_RE = re.compile(r"[^\w]+", re.UNICODE)

#: Synthetic duration of one spoken word.
WORD_MS = 300

#: Synthetic silence between two consecutive words.
GAP_MS = 100

#: Every column of ``video_acoustics`` except the key and the timestamp,
#: with a plausible default so a test overrides only what it cares about.
ACOUSTIC_DEFAULTS: dict[str, float] = {
    "f0_mean": 120.0,
    "f0_std": 20.0,
    "spectral_tilt": -1.0,
    "noise_floor_db": -60.0,
    "reverb_proxy": 0.2,
    "loudness_lufs": -23.0,
}


def add_video(
    db: Database,
    *,
    external_id: str,
    title: str = "Sample",
    duration_ms: int | None = None,
    source: str = "youtube",
    kind: str = "video",
) -> int:
    """Catalog one video. ``external_id`` is a placeholder, never a real id."""
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, duration_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            source,
            kind,
            external_id,
            f"https://example.invalid/v/{external_id}",
            title,
            duration_ms,
            utc_now_iso(),
        ),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_speaker(db: Database, label: str) -> int:
    """One row in the global roster."""
    cursor = db.conn.execute(
        "INSERT INTO speakers (label, created_at) VALUES (?, ?)", (label, utc_now_iso())
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_video_speaker(
    db: Database,
    video_id: int,
    local_label: str,
    *,
    speaker_id: int | None = None,
    engine: str = "fake",
) -> int:
    """One diarizer label in one video, optionally mapped to the roster.

    ``engine`` is NOT NULL in contracts §3, and §3 puts supplying every
    NOT NULL column on the part whose fixtures insert the row.
    """
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine) "
        "VALUES (?, ?, ?, ?)",
        (video_id, local_label, speaker_id, engine),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_words(
    db: Database,
    video_id: int,
    text: str,
    *,
    start_ms: int = 0,
    word_ms: int = WORD_MS,
    gap_ms: int = GAP_MS,
    source: str = "aligned",
    align_score: float | None = 0.9,
    engine: str = "fake",
    video_speaker_id: int | None = None,
) -> int:
    """Append ``text`` to a video as word rows. Returns the next free ordinal.

    Ordinals continue after whatever the video already has, so a test can
    build one video from several calls with different speakers or gaps.

    The tier decides which columns may be filled (contracts §3, "Three
    transcript tiers"). ``caption`` rows have no ``end_ms``, which the
    table's CHECK constraint enforces. ``timed`` rows have real ends but
    no ``align_score``, because nothing measured one — that is what makes
    them searchable and not cuttable. Only ``aligned`` rows carry a score,
    and even then it may be None when the aligner reports no confidence.
    """
    row = db.conn.execute(
        "SELECT COALESCE(MAX(ord) + 1, 0) AS next FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()
    ordinal = int(row["next"])
    captions = source == "caption"
    scored = source == "aligned"
    cursor_ms = start_ms
    rows: list[tuple[Any, ...]] = []
    # Split each whitespace-separated word on internal punctuation too, so
    # "кто-то" becomes two ordinally adjacent rows rather than one row whose
    # normalized_text holds a space. Contracts §4: "A stored word row holds
    # exactly one token, and the tokens stored for a piece of text are
    # exactly normalize_text(text).split()." Splitting on the original text
    # (rather than re-splitting the whole normalized string) keeps a plain
    # word's original casing in `text`, which test_add_words_normalizes_and_stems
    # depends on.
    for raw_token in text.split():
        pieces = [piece for piece in _PIECE_RE.split(raw_token) if piece]
        for piece in pieces:
            normalized = normalize_text(piece)
            if not normalized:
                continue
            rows.append(
                (
                    video_id,
                    ordinal,
                    cursor_ms,
                    None if captions else cursor_ms + word_ms,
                    piece,
                    normalized,
                    stem_text(normalized),
                    None,
                    align_score if scored else None,
                    source,
                    engine,
                    video_speaker_id,
                )
            )
            ordinal += 1
            cursor_ms += word_ms
        cursor_ms += gap_ms
    db.conn.executemany(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
        "confidence, align_score, source, engine, video_speaker_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.conn.commit()
    return ordinal


def add_acoustics(db: Database, video_id: int, **fields: float) -> None:
    """One ``video_acoustics`` row; every unnamed column takes its default."""
    values = dict(ACOUSTIC_DEFAULTS)
    for key, value in fields.items():
        if key not in ACOUSTIC_DEFAULTS:
            raise KeyError(f"video_acoustics has no column {key!r}")
        values[key] = value
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    db.conn.execute(
        f"INSERT INTO video_acoustics (video_id, {columns}, computed_at) "
        f"VALUES (?, {placeholders}, ?)",
        (video_id, *values.values(), utc_now_iso()),
    )
    db.conn.commit()


def word_rows(db: Database, video_id: int) -> list[sqlite3.Row]:
    """Every word of one video, in reading order."""
    return list(
        db.conn.execute("SELECT * FROM words WHERE video_id = ? ORDER BY ord", (video_id,))
    )
