"""Every SQL statement about speakers and their per-video labels.

Two levels, and the distinction is the design (design §4):

``speakers``
    The roster of real people. Small — tens of rows — which is why alias
    matching happens in Python rather than in an index.

``video_speakers``
    One row per diarizer label per video. ``SPEAKER_04`` in some video is a
    real, citable entity whether or not anybody ever named it, and
    ``speaker_id`` is nullable precisely to say "a distinct voice I never
    bothered to name".

Linking a label to a person updates **one row**. The old implementation
back-filled `words.speaker_id` across thousands of rows on every mapping;
the new schema points `words.video_speaker_id` at the label instead, so the
person can be decided, changed and undecided for the cost of one UPDATE.

This module holds the queries and nothing else, so that `rytp/commands/`
never writes SQL and `rytp/tui/` never imports `rytp/commands/`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database
from rytp.diarize.segments import WordSpan
from rytp.models import DiarSegment, InvalidInputError, NotFoundError, utc_now_iso


@dataclass(frozen=True)
class Speaker:
    """One row of the global roster."""

    id: int
    label: str
    aliases: tuple[str, ...]
    notes: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Speaker:
        return cls(
            id=int(row["id"]),
            label=row["label"],
            aliases=_decode_aliases(row["aliases_json"]),
            notes=row["notes"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class RosterRow:
    """A roster entry with the two counts the mapper's right pane shows."""

    speaker_id: int
    label: str
    aliases: tuple[str, ...]
    notes: str | None
    n_labels: int
    n_videos: int


def _decode_aliases(raw: str | None) -> tuple[str, ...]:
    """Tolerate a malformed aliases_json rather than failing a whole listing."""
    try:
        values = json.loads(raw or "[]")
    except ValueError:
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def parse_aliases(text: str) -> tuple[str, ...]:
    """Split a comma-separated flag into aliases (contracts §5).

    `Param.type` is a scalar, so a multi-valued flag is a string the handler
    parses. Trims, drops empties, and de-duplicates case-insensitively while
    keeping the first spelling the user typed.
    """
    out: list[str] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        alias = chunk.strip()
        if not alias or alias.casefold() in seen:
            continue
        seen.add(alias.casefold())
        out.append(alias)
    return tuple(out)


def add_speaker(
    db: Database,
    label: str,
    *,
    aliases: Iterable[str] = (),
    notes: str | None = None,
) -> tuple[int, bool]:
    """Insert a roster entry. Returns ``(speaker_id, was_created)``.

    Idempotent on ``label``: an existing speaker is returned untouched
    rather than having its aliases silently rewritten, and the ``False``
    lets the caller say so.
    """
    clean = label.strip()
    if not clean:
        raise InvalidInputError("a speaker needs a label")
    existing = db.conn.execute(
        "SELECT id FROM speakers WHERE label = ?", (clean,)
    ).fetchone()
    if existing is not None:
        return int(existing["id"]), False
    cursor = db.conn.execute(
        "INSERT INTO speakers (label, aliases_json, notes, created_at) VALUES (?, ?, ?, ?)",
        (clean, json.dumps(list(aliases), ensure_ascii=False), notes, utc_now_iso()),
    )
    assert cursor.lastrowid is not None  # an INSERT always sets it
    return int(cursor.lastrowid), True


def update_speaker(
    db: Database,
    speaker_id: int,
    *,
    label: str | None = None,
    aliases: Sequence[str] | None = None,
    notes: str | None = None,
) -> None:
    """Change the mutable fields of an existing roster entry."""
    get_speaker(db, speaker_id)
    sets: list[str] = []
    params: list[object] = []
    if label is not None:
        clean = label.strip()
        if not clean:
            raise InvalidInputError("a speaker needs a label")
        sets.append("label = ?")
        params.append(clean)
    if aliases is not None:
        sets.append("aliases_json = ?")
        params.append(json.dumps(list(aliases), ensure_ascii=False))
    if notes is not None:
        sets.append("notes = ?")
        params.append(notes)
    if not sets:
        return
    params.append(speaker_id)
    db.conn.execute(f"UPDATE speakers SET {', '.join(sets)} WHERE id = ?", params)


def add_aliases(db: Database, speaker_id: int, aliases: Iterable[str]) -> tuple[str, ...]:
    """Append aliases, case-insensitively de-duplicated. Returns the new set."""
    current = get_speaker(db, speaker_id).aliases
    merged = parse_aliases(",".join([*current, *aliases]))
    update_speaker(db, speaker_id, aliases=merged)
    return merged


def list_speakers(db: Database) -> list[Speaker]:
    """The whole roster, ordered by label."""
    rows = db.conn.execute("SELECT * FROM speakers ORDER BY label").fetchall()
    return [Speaker.from_row(row) for row in rows]


def roster_rows(db: Database) -> list[RosterRow]:
    """The roster plus how many labels and videos each person is linked to."""
    rows = db.conn.execute(
        """
        SELECT s.id, s.label, s.aliases_json, s.notes,
               COUNT(vs.id)                AS n_labels,
               COUNT(DISTINCT vs.video_id) AS n_videos
        FROM speakers s
        LEFT JOIN video_speakers vs ON vs.speaker_id = s.id
        GROUP BY s.id
        ORDER BY s.label
        """
    ).fetchall()
    return [
        RosterRow(
            speaker_id=int(row["id"]),
            label=row["label"],
            aliases=_decode_aliases(row["aliases_json"]),
            notes=row["notes"],
            n_labels=int(row["n_labels"]),
            n_videos=int(row["n_videos"]),
        )
        for row in rows
    ]


def get_speaker(db: Database, speaker_id: int) -> Speaker:
    """One roster entry by id. Raises :class:`NotFoundError`."""
    row = db.conn.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"speaker {speaker_id} is not in the roster")
    return Speaker.from_row(row)


def find_speaker(db: Database, ref: str) -> Speaker | None:
    """Resolve a user-typed reference: an id, a label, or an alias.

    Case-insensitive on label and alias, because nobody types a roster entry
    the same way twice. Labels are tried before aliases so an exact name
    always wins.
    """
    needle = ref.strip()
    if not needle:
        return None
    if needle.isdigit():
        row = db.conn.execute(
            "SELECT * FROM speakers WHERE id = ?", (int(needle),)
        ).fetchone()
        if row is not None:
            return Speaker.from_row(row)
    folded = needle.casefold()
    roster = list_speakers(db)
    for speaker in roster:
        if speaker.label.casefold() == folded:
            return speaker
    for speaker in roster:
        if any(alias.casefold() == folded for alias in speaker.aliases):
            return speaker
    return None


def resolve_speaker(db: Database, ref: str) -> Speaker:
    """:func:`find_speaker`, raising :class:`NotFoundError` instead of returning None."""
    speaker = find_speaker(db, ref)
    if speaker is None:
        raise NotFoundError(f"no speaker matches {ref!r}")
    return speaker


def remove_speaker(db: Database, speaker_id: int) -> int:
    """Delete a roster entry. Returns how many local labels it left unnamed.

    `video_speakers.speaker_id` is `ON DELETE SET NULL` (contracts §3), and
    that is the whole design rather than a convenience: removing a person
    from the roster must not destroy the knowledge that some video had four
    distinct speakers. The labels survive as voices nobody has named, which
    is exactly what they were before anybody named them.

    No files are involved, so contracts §5 asks for no `--dry-run` and no
    `--yes`; the caller decides whether the number returned is large enough
    to be worth confirming first.
    """
    get_speaker(db, speaker_id)
    linked = db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE speaker_id = ?", (speaker_id,)
    ).fetchone()[0]
    db.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
    return int(linked)


def find_alias_collisions(
    db: Database,
    aliases: Iterable[str],
    *,
    exclude_speaker_id: int | None = None,
) -> list[tuple[str, int]]:
    """``[(alias, other_speaker_id), …]`` for aliases already spoken for.

    An alias that matches another person's *label* collides too: the roster
    is looked up by either, so a duplicate would make lookups ambiguous.
    Not an error by itself — the caller decides whether to warn or refuse.
    """
    wanted = {alias.casefold(): alias for alias in aliases if alias.strip()}
    if not wanted:
        return []
    found: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for speaker in list_speakers(db):
        if speaker.id == exclude_speaker_id:
            continue
        for candidate in (speaker.label, *speaker.aliases):
            typed = wanted.get(candidate.casefold())
            if typed is not None and (typed, speaker.id) not in seen:
                seen.add((typed, speaker.id))
                found.append((typed, speaker.id))
    return found


@dataclass(frozen=True)
class LabelRow:
    """One diarizer label of one video, with everything a pane needs to show it."""

    video_speaker_id: int
    video_id: int
    local_label: str
    speaker_id: int | None
    speaker_label: str | None
    engine: str
    n_words: int
    speech_ms: int
    has_embedding: bool


@dataclass(frozen=True)
class VideoRow:
    """A diarized video, for the mapper's video chooser."""

    video_id: int
    title: str
    published_at: str | None
    n_labels: int
    n_unmapped: int


_LABEL_SELECT = """
    SELECT vs.id, vs.video_id, vs.local_label, vs.speaker_id, vs.engine,
           s.label                  AS speaker_label,
           vs.embedding IS NOT NULL AS has_embedding,
           COUNT(w.id)              AS n_words,
           COALESCE(SUM(COALESCE(w.end_ms, w.start_ms) - w.start_ms), 0) AS speech_ms
    FROM video_speakers vs
    LEFT JOIN speakers s ON s.id = vs.speaker_id
    LEFT JOIN words w    ON w.video_speaker_id = vs.id
"""


def _label_row(row: sqlite3.Row) -> LabelRow:
    return LabelRow(
        video_speaker_id=int(row["id"]),
        video_id=int(row["video_id"]),
        local_label=row["local_label"],
        speaker_id=None if row["speaker_id"] is None else int(row["speaker_id"]),
        speaker_label=row["speaker_label"],
        engine=row["engine"],
        n_words=int(row["n_words"]),
        speech_ms=int(row["speech_ms"]),
        has_embedding=bool(row["has_embedding"]),
    )


def upsert_video_speaker(
    db: Database, video_id: int, local_label: str, *, engine: str
) -> int:
    """Create (or find) the row for one diarizer label of one video.

    ``engine`` is which diarizer produced the label (contracts §3). It is
    recorded for the same reason `words.engine` is: design §11 says no stage
    may assume a particular engine, so a corpus built with more than one has
    to stay interpretable. A re-run with a *different* diarizer overwrites
    it, because the label it is describing is that run's.

    Never clears an existing `speaker_id`: re-running diarization on a video
    whose labels happen to come out the same should not throw the mapping
    away. Re-*transcribing* does throw it away, and that is Part 3's job in
    the same transaction that replaces the words (contracts §4) — not this
    function's.
    """
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) VALUES (?, ?, ?) "
        "ON CONFLICT (video_id, local_label) DO UPDATE SET engine = excluded.engine",
        (video_id, local_label, engine),
    )
    row = db.conn.execute(
        "SELECT id FROM video_speakers WHERE video_id = ? AND local_label = ?",
        (video_id, local_label),
    ).fetchone()
    return int(row["id"])


def label_rows(db: Database, video_id: int) -> list[LabelRow]:
    """Every local label of one video, ordered by label name."""
    rows = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.video_id = ? GROUP BY vs.id ORDER BY vs.local_label",
        (video_id,),
    ).fetchall()
    return [_label_row(row) for row in rows]


def get_label(db: Database, video_speaker_id: int) -> LabelRow:
    """One local label by its id. Raises :class:`NotFoundError`."""
    row = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.id = ? GROUP BY vs.id", (video_speaker_id,)
    ).fetchone()
    if row is None or row["id"] is None:
        raise NotFoundError(f"video speaker {video_speaker_id} does not exist")
    return _label_row(row)


def find_label(db: Database, video_id: int, local_label: str) -> LabelRow:
    """One local label by video and name. Raises :class:`NotFoundError`."""
    row = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.video_id = ? AND vs.local_label = ? GROUP BY vs.id",
        (video_id, local_label),
    ).fetchone()
    if row is None or row["id"] is None:
        raise NotFoundError(f"video {video_id} has no label {local_label!r}")
    return _label_row(row)


def link_video_speaker(
    db: Database, video_speaker_id: int, speaker_id: int | None
) -> None:
    """Say who a local label is — or, with ``None``, say that nobody knows.

    **One row.** This is the whole point of the two-level model: the words
    already point at the label, so deciding the person is a single UPDATE
    rather than a back-fill across thousands of rows.
    """
    get_label(db, video_speaker_id)
    if speaker_id is not None:
        get_speaker(db, speaker_id)
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE id = ?",
        (speaker_id, video_speaker_id),
    )


def set_embedding(db: Database, video_speaker_id: int, blob: bytes | None) -> None:
    """Store (or clear) the voice vector for one local label."""
    db.conn.execute(
        "UPDATE video_speakers SET embedding = ? WHERE id = ?", (blob, video_speaker_id)
    )


def clear_video_speakers(db: Database, video_id: int) -> int:
    """Drop every label of one video. `words.video_speaker_id` follows.

    `ON DELETE SET NULL` on `words.video_speaker_id` (contracts §3) does the
    second half, so there is no second statement to forget.
    """
    cursor = db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))
    return cursor.rowcount


def word_spans(db: Database, video_id: int) -> list[WordSpan]:
    """Every word of one video as ``(id, start_ms, end_ms)``, in ordinal order."""
    rows = db.conn.execute(
        "SELECT id, start_ms, end_ms FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    return [(int(r["id"]), int(r["start_ms"]), r["end_ms"]) for r in rows]


def stamp_words(db: Database, video_id: int, by_word_id: Mapping[int, int]) -> int:
    """Point each word at its local label; clear every word not in the map.

    This is the *only* place `words.video_speaker_id` is written, and it is
    written once per diarization run. Clearing the rest matters: a second
    run that finds fewer speakers must not leave words pointing at labels
    the new run did not produce.
    """
    with db.transaction():
        db.conn.execute(
            "UPDATE words SET video_speaker_id = NULL WHERE video_id = ?", (video_id,)
        )
        db.conn.executemany(
            "UPDATE words SET video_speaker_id = ? WHERE id = ?",
            [(label_id, word_id) for word_id, label_id in by_word_id.items()],
        )
    return len(by_word_id)


def diarized_videos(
    db: Database, *, only_unmapped: bool = False, limit: int = C.SPEAKER_LIST_LIMIT
) -> list[VideoRow]:
    """Videos that have local labels, newest first — the mapper's front door."""
    having = "HAVING SUM(vs.speaker_id IS NULL) > 0" if only_unmapped else ""
    rows = db.conn.execute(
        f"""
        SELECT v.id, v.title, v.published_at,
               COUNT(vs.id)               AS n_labels,
               SUM(vs.speaker_id IS NULL) AS n_unmapped
        FROM videos v
        JOIN video_speakers vs ON vs.video_id = v.id
        GROUP BY v.id
        {having}
        ORDER BY v.published_at DESC, v.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        VideoRow(
            video_id=int(row["id"]),
            title=row["title"],
            published_at=row["published_at"],
            n_labels=int(row["n_labels"]),
            n_unmapped=int(row["n_unmapped"]),
        )
        for row in rows
    ]


def linked_labels(db: Database, speaker_id: int) -> list[LabelRow]:
    """Every local label linked to one person, across every video."""
    rows = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.speaker_id = ? GROUP BY vs.id ORDER BY vs.video_id",
        (speaker_id,),
    ).fetchall()
    return [_label_row(row) for row in rows]

def label_segments(db: Database, video_id: int) -> list[DiarSegment]:
    """Reconstruct who-spoke-when from the labelled words.

    Diarizer segments are not stored — the schema has no table for them and
    does not need one, because every word already carries the label that
    spoke it. Maximal runs of consecutive words with the same label become
    one segment, and an unlabelled word breaks a run, which is correct: the
    stretch around a voice nobody labelled is not this speaker's.

    This is what makes re-embedding cheap. Re-running the diarizer is the
    largest GPU cost in the project (design §6); re-running an embedder
    over thirty seconds of reconstructed speech is nothing.
    """
    rows = db.conn.execute(
        """
        SELECT w.start_ms, w.end_ms, w.video_speaker_id, vs.local_label
        FROM words w
        LEFT JOIN video_speakers vs ON vs.id = w.video_speaker_id
        WHERE w.video_id = ?
        ORDER BY w.ord
        """,
        (video_id,),
    ).fetchall()

    segments: list[DiarSegment] = []
    current: int | None = None
    start = end = 0
    label = ""

    def close() -> None:
        nonlocal current
        if current is not None:
            segments.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))
            current = None

    for row in rows:
        if row["video_speaker_id"] is None:
            close()
            continue
        if int(row["video_speaker_id"]) != current:
            close()
            current = int(row["video_speaker_id"])
            label = row["local_label"]
            start = int(row["start_ms"])
        end = int(row["end_ms"] if row["end_ms"] is not None else row["start_ms"])
    close()
    return segments
