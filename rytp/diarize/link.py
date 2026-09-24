"""Suggesting, never applying, cross-video speaker links (design §6).

The corpus spans more than a decade: microphones changed, rooms changed,
the main speaker aged. That roughly triples voice-embedding error, so
automatic linking across eras is untrustworthy and **must never be applied
silently**. Nothing in this module writes to the database. The only writer
is `store.link_video_speaker`, and the only things that call it are a
command a human ran and a keystroke a human pressed.

Three mechanisms, all from design §6:

**Per-era enrolment centroids.** A person's linked labels are bucketed by
the publication date of their video and each bucket is averaged, so a 2013
voice is compared against how that person sounded in 2013 rather than
against a smear of their whole life.

**Acoustic scoping.** `video_acoustics` (Part 3) is one cheap row per video
saying what the recording sounds like. It exists to tell the assembler
which sources blend; this is its second consumer, deciding which
recordings resemble each other enough for a voice comparison to mean
anything.

**A two-threshold band.** Above HI is a confident match, at or below LO a
confident non-match, and the space between goes to the human. Across eras
— or when the acoustics are unknown, which is the same kind of ignorance —
only a much higher score counts as confident. Every one of those numbers
is a guess until somebody measures this corpus; they are marked
UNVALIDATED in `constants.py` and are meant to be re-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import centroid, comparable, cosine_similarity, unpack_embedding

#: The three answers. Strings rather than an enum because they are rendered
#: straight into a `CommandResult` row and a TUI cell.
MATCH = "match"
REVIEW = "review"
NO_MATCH = "no-match"


@dataclass(frozen=True)
class EnrolmentSample:
    """One embedded label of one person: where it came from and what it sounds like."""

    video_id: int
    engine: str
    vector: list[float]


@dataclass(frozen=True)
class Suggestion:
    """One roster candidate for one local label. Advisory, always."""

    video_speaker_id: int
    speaker_id: int
    speaker_label: str
    similarity: float
    era: str
    same_era: bool
    acoustically_close: bool
    engine: str
    verdict: str


def era_of(published_at: str | None) -> str:
    """Which era bucket a publication date falls in.

    Buckets are `C.SPEAKER_ERA_YEARS` wide and named by their range, so a
    label in a listing explains itself. A video with no usable date — a
    local file, usually — gets its own bucket rather than being silently
    filed under some year.
    """
    if not published_at:
        return C.SPEAKER_ERA_UNKNOWN
    head = published_at[:4]
    if not head.isdigit():
        return C.SPEAKER_ERA_UNKNOWN
    year = int(head)
    start = year - (year % C.SPEAKER_ERA_YEARS)
    return f"{start}-{start + C.SPEAKER_ERA_YEARS - 1}"


def acoustic_distance(db: Database, video_a: int, video_b: int) -> float | None:
    """How unlike two recordings sound, or ``None`` if nobody measured.

    The mean of the per-feature differences, each divided by its scale, so
    1.0 means "one typical spread apart on average". Features missing from
    either row are skipped; if none survive, the answer is ignorance rather
    than zero — and ignorance is treated strictly by the caller.
    """
    columns = ", ".join(name for name, _scale in C.ACOUSTIC_FEATURE_SCALES)
    rows = {
        int(row["video_id"]): row
        for row in db.conn.execute(
            f"SELECT video_id, {columns} FROM video_acoustics WHERE video_id IN (?, ?)",
            (video_a, video_b),
        ).fetchall()
    }
    left, right = rows.get(video_a), rows.get(video_b)
    if left is None or right is None:
        return None
    ratios: list[float] = []
    for name, scale in C.ACOUSTIC_FEATURE_SCALES:
        if left[name] is None or right[name] is None:
            continue
        ratios.append(abs(float(left[name]) - float(right[name])) / scale)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def enrolment(db: Database, speaker_id: int) -> dict[str, list[EnrolmentSample]]:
    """One person's embedded labels, grouped by era."""
    grouped: dict[str, list[EnrolmentSample]] = {}
    for label in store.linked_labels(db, speaker_id):
        row = db.conn.execute(
            "SELECT vs.embedding, v.published_at FROM video_speakers vs "
            "JOIN videos v ON v.id = vs.video_id WHERE vs.id = ?",
            (label.video_speaker_id,),
        ).fetchone()
        if row is None or row["embedding"] is None:
            continue
        grouped.setdefault(era_of(row["published_at"]), []).append(
            EnrolmentSample(
                video_id=label.video_id,
                engine=label.engine,
                vector=unpack_embedding(bytes(row["embedding"])),
            )
        )
    return grouped


def engines_in_play(db: Database) -> set[str]:
    """Every diarizer that produced a label currently carrying an embedding.

    More than one means some pairs are simply not comparable, which is a
    thing to *say* rather than to leave as a mysteriously short list.
    """
    rows = db.conn.execute(
        "SELECT DISTINCT engine FROM video_speakers WHERE embedding IS NOT NULL"
    ).fetchall()
    return {row["engine"] for row in rows}


def verdict_for(similarity: float, *, strict: bool) -> str:
    """Place a score in the two-threshold band (design §6)."""
    high = C.SPEAKER_MATCH_HI_CROSS_ERA if strict else C.SPEAKER_MATCH_HI
    if similarity >= high:
        return MATCH
    if similarity <= C.SPEAKER_MATCH_LO:
        return NO_MATCH
    return REVIEW


def suggest_for_label(
    db: Database, video_speaker_id: int, *, limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> list[Suggestion]:
    """Rank the roster against one local label's voice. Writes nothing."""
    label = store.get_label(db, video_speaker_id)
    row = db.conn.execute(
        "SELECT vs.embedding, v.published_at FROM video_speakers vs "
        "JOIN videos v ON v.id = vs.video_id WHERE vs.id = ?",
        (video_speaker_id,),
    ).fetchone()
    if row is None or row["embedding"] is None:
        return []
    query = unpack_embedding(bytes(row["embedding"]))
    query_era = era_of(row["published_at"])
    query_engine = label.engine

    # One person cannot be two voices in one recording.
    taken = {
        other.speaker_id
        for other in store.label_rows(db, label.video_id)
        if other.speaker_id is not None and other.video_speaker_id != video_speaker_id
    }

    found: list[Suggestion] = []
    for person in store.roster_rows(db):
        if person.speaker_id in taken:
            continue
        best: Suggestion | None = None
        for era, members in enrolment(db, person.speaker_id).items():
            # Two rules for "can these be compared at all", and neither is
            # negotiable. Different diarizers draw different turn
            # boundaries, so a vector measured over one engine's segments
            # says nothing about a vector measured over another's; and two
            # embedders produce two dimensionalities. Both are skipped
            # rather than scored, and `engines_in_play` is how a caller
            # explains a suspiciously short list.
            usable = [
                sample
                for sample in members
                if sample.engine == query_engine and comparable(sample.vector, query)
            ]
            if not usable:
                continue
            similarity = cosine_similarity(
                query, centroid([sample.vector for sample in usable])
            )
            distances = [
                distance
                for sample in usable
                if (distance := acoustic_distance(db, label.video_id, sample.video_id))
                is not None
            ]
            close = bool(distances) and min(distances) <= C.ACOUSTIC_MAX_DISTANCE
            same_era = era == query_era and era != C.SPEAKER_ERA_UNKNOWN
            candidate = Suggestion(
                video_speaker_id=video_speaker_id,
                speaker_id=person.speaker_id,
                speaker_label=person.label,
                similarity=similarity,
                era=era,
                same_era=same_era,
                acoustically_close=close,
                engine=query_engine,
                verdict=verdict_for(similarity, strict=not (same_era and close)),
            )
            if best is None or candidate.similarity > best.similarity:
                best = candidate
        if best is not None:
            found.append(best)

    found.sort(key=lambda s: (-s.similarity, s.speaker_label))
    return found[:limit]


def suggest_for_video(
    db: Database, video_id: int, *, limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> dict[int, list[Suggestion]]:
    """Suggestions for every label of one video that nobody has named yet."""
    out: dict[int, list[Suggestion]] = {}
    for label in store.label_rows(db, video_id):
        if label.speaker_id is not None:
            continue
        found = suggest_for_label(db, label.video_speaker_id, limit=limit)
        if found:
            out[label.video_speaker_id] = found
    return out
