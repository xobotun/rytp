"""Grouping words into utterances — the rows the FTS index shadows.

Design §4: "Contiguous runs of words, split on speaker change where
speakers are known and on a silence threshold otherwise — so caption-tier
and undiarized videos still get sensible units."

Why this exists at all: the implementation being replaced shadowed the
`words` table with FTS, one word per row, so a two-term query became an
implicit AND inside a single row and could never match. Verified against
the owner's corpus: `добрый*` returned one hit and `добрый* вечер*`
returned none, with the two words adjacent in the text. Indexing a
sentence-sized run instead is the structural fix.

Contracts §4 makes `utterances` derived data: whenever a video's words
are replaced they are deleted in the same transaction, and re-deriving
them is this module's job. `index_video` is therefore written to be
idempotent and cheap rather than incremental.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp.db import Database
from rytp.models import NotFoundError

if TYPE_CHECKING:  # pragma: no cover - import cycle, see index_readiness
    from rytp.jobs import Readiness

__all__ = [
    "IndexedWord",
    "UtteranceDraft",
    "build_utterances",
    "drop_utterances",
    "implied_end_ms",
    "index_readiness",
    "index_video",
    "indexed_videos",
    "videos_needing_index",
    "words_for",
]


@dataclass(frozen=True)
class IndexedWord:
    """One `words` row, as the builder needs to see it.

    `end_ms` is None exactly for caption-sourced rows (contracts §3).
    `normalized_text` is exactly one token and `stem` is its stem, also
    one token: contracts §4 makes a row and a token the same thing, and
    Part 3's `split_token` is what guarantees it.
    """

    ord: int
    start_ms: int
    end_ms: int | None
    text: str
    normalized_text: str
    stem: str
    source: str
    video_speaker_id: int | None


@dataclass(frozen=True)
class UtteranceDraft:
    """One `utterances` row before it has an id."""

    video_speaker_id: int | None
    start_ms: int
    end_ms: int
    first_word_ord: int
    last_word_ord: int
    text: str
    normalized_text: str
    stem_text: str


_SELECT_WORDS = (
    "SELECT ord, start_ms, end_ms, text, normalized_text, stem, source, video_speaker_id"
    " FROM words WHERE video_id = ? ORDER BY ord"
)

_INSERT_UTTERANCE = (
    "INSERT INTO utterances (video_id, video_speaker_id, start_ms, end_ms,"
    " first_word_ord, last_word_ord, text, normalized_text, stem_text)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def implied_end_ms(words: Sequence[IndexedWord], position: int) -> int:
    """When the word at `position` stops, inventing an end for caption rows.

    Aligned rows have a real `end_ms`. Caption rows have none (design §6:
    captions carry per-word starts on a 40 ms grid and no ends), so the
    end is the next word's start, capped at `CAPTION_WORD_FALLBACK_MS`
    after this word's start. The cap is the whole reason the silence rule
    works on caption tier: without it the implied end always equals the
    next start and every gap is zero.
    """
    word = words[position]
    if word.end_ms is not None:
        return word.end_ms
    fallback = word.start_ms + C.CAPTION_WORD_FALLBACK_MS
    if position + 1 >= len(words):
        return fallback
    # max() guards the caption case where several words share one start.
    return max(word.start_ms, min(words[position + 1].start_ms, fallback))


def _starts_new_utterance(
    words: Sequence[IndexedWord], run: Sequence[int], position: int
) -> bool:
    """Does the word at `position` begin a new utterance?"""
    previous = words[run[-1]]
    word = words[position]
    if word.video_speaker_id != previous.video_speaker_id:
        return True
    if len(run) >= C.UTTERANCE_MAX_WORDS:
        return True
    if word.start_ms - implied_end_ms(words, run[-1]) >= C.UTTERANCE_SILENCE_GAP_MS:
        return True
    span = implied_end_ms(words, position) - words[run[0]].start_ms
    return span > C.UTTERANCE_MAX_DURATION_MS


def _draft(words: Sequence[IndexedWord], run: Sequence[int]) -> UtteranceDraft:
    members = [words[position] for position in run]
    first = members[0]
    last = members[-1]
    return UtteranceDraft(
        video_speaker_id=first.video_speaker_id,
        start_ms=first.start_ms,
        end_ms=max(implied_end_ms(words, run[-1]), first.start_ms),
        first_word_ord=first.ord,
        last_word_ord=last.ord,
        # Joined from the per-word columns, never re-derived from the
        # joined string. Stemming is not idempotent (сказали → сказа,
        # which stems again), so re-running it over the joined text would
        # desynchronise utterances.stem_text from words.stem and the stem
        # tier would quietly stop matching.
        text=" ".join(member.text for member in members),
        normalized_text=" ".join(member.normalized_text for member in members),
        stem_text=" ".join(member.stem for member in members),
    )


def build_utterances(words: Sequence[IndexedWord]) -> list[UtteranceDraft]:
    """Group a video's words into utterances. Pure: no database, no clock."""
    drafts: list[UtteranceDraft] = []
    run: list[int] = []
    for position in range(len(words)):
        if run and _starts_new_utterance(words, run, position):
            drafts.append(_draft(words, run))
            run = []
        run.append(position)
    if run:
        drafts.append(_draft(words, run))
    return drafts


def words_for(db: Database, video_id: int) -> list[IndexedWord]:
    """Every word of one video, in ordinal order."""
    return [
        IndexedWord(
            ord=int(row["ord"]),
            start_ms=int(row["start_ms"]),
            end_ms=None if row["end_ms"] is None else int(row["end_ms"]),
            text=str(row["text"]),
            normalized_text=str(row["normalized_text"]),
            stem=str(row["stem"]),
            source=str(row["source"]),
            video_speaker_id=(
                None if row["video_speaker_id"] is None else int(row["video_speaker_id"])
            ),
        )
        for row in db.conn.execute(_SELECT_WORDS, (video_id,))
    ]


def index_video(db: Database, video_id: int) -> int:
    """Rebuild one video's utterances from its words. Returns how many.

    Delete-and-rewrite rather than incremental, because contracts §4 makes
    utterances disposable derived data and a whole video is a few thousand
    rows. Part 1's triggers keep `utterances_fts` in step.
    """
    if db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is None:
        raise NotFoundError(f"no video with id {video_id}")
    drafts = build_utterances(words_for(db, video_id))
    rows = [
        (
            video_id,
            draft.video_speaker_id,
            draft.start_ms,
            draft.end_ms,
            draft.first_word_ord,
            draft.last_word_ord,
            draft.text,
            draft.normalized_text,
            draft.stem_text,
        )
        for draft in drafts
    ]
    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.executemany(_INSERT_UTTERANCE, rows)
    return len(rows)


def drop_utterances(db: Database, video_id: int) -> int:
    """Delete one video's utterances. Returns how many went.

    Contracts §5 gives every group a `remove`, and this is Part 4's. It
    needs no `--dry-run` and no `--yes`: utterances are derived data with
    no file behind them, and `index build` puts them straight back.

    The words are untouched — deleting them is `transcribe.remove`, which
    is Part 3's and takes the utterances with it (contracts §4).
    """
    if db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is None:
        raise NotFoundError(f"no video with id {video_id}")
    with db.transaction():
        count = int(
            db.conn.execute(
                "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        )
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    return count


def index_readiness(db: Database, video_id: int) -> Readiness:
    """Design §5: answered from database state alone, never from a payload.

    Three ways to need re-indexing, and the third is the one that is easy
    to miss: Part 7 fills `words.video_speaker_id` **in place** rather than
    replacing words, so the contracts §4 delete-the-utterances invariant
    never fires and a video indexed before diarization keeps utterances
    that were never split on speaker.
    """
    # Imported in the body deliberately. `rytp/jobs/__init__.py` imports
    # this module at load time to register the `index` kind, so a
    # module-level `from rytp.jobs import Readiness` would be a real cycle
    # for anyone who imports `rytp.index.utterances` first.
    from rytp.jobs import Readiness

    have = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(ord) AS lo, MAX(ord) AS hi,"
        " COUNT(video_speaker_id) AS labelled FROM words WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if not have["n"]:
        return Readiness.BLOCKED
    built = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(first_word_ord) AS lo, MAX(last_word_ord) AS hi,"
        " COUNT(video_speaker_id) AS labelled FROM utterances WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if not built["n"]:
        return Readiness.READY
    if (built["lo"], built["hi"]) != (have["lo"], have["hi"]):
        return Readiness.READY
    if have["labelled"] and not built["labelled"]:
        return Readiness.READY
    return Readiness.SATISFIED


def videos_needing_index(db: Database) -> list[int]:
    """Every video whose utterances are missing or stale, in id order."""
    from rytp.jobs import Readiness

    ids = [
        int(row["video_id"])
        for row in db.conn.execute("SELECT DISTINCT video_id FROM words ORDER BY video_id")
    ]
    return [i for i in ids if index_readiness(db, i) is Readiness.READY]


def indexed_videos(db: Database) -> list[tuple[int, str, int]]:
    """(video_id, title, utterance count) for every indexed video.

    Navigation, not an operation on data, which is why this is a plain
    query rather than a command — the same call Part 7's speaker picker
    makes against `diarized_videos`.
    """
    return [
        (int(row["video_id"]), str(row["title"]), int(row["n"]))
        for row in db.conn.execute(
            "SELECT u.video_id AS video_id, v.title AS title, COUNT(*) AS n"
            " FROM utterances u JOIN videos v ON v.id = u.video_id"
            " GROUP BY u.video_id, v.title ORDER BY u.video_id"
        )
    ]
