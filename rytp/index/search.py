"""Backward lookup: a word or a phrase, to every place it was said.

Design §7. Two tiers over the same two-column FTS shadow — the exact
`normalized_text` column first, the Python-stemmed `stem_text` column
second — and the result says which one answered, because an inflection
is not what the user typed.

The replaced implementation got this wrong twice, and both failures are
measured rather than theoretical. It shadowed the `words` table one word
per row, which makes a two-term query an implicit AND inside a single row
that can never match (`добрый*` → 1 hit, `добрый* вечер*` → 0, with the
words adjacent in the corpus). And it used the `porter` tokenizer, which
is English-only and therefore did no Russian stemming at all
(`ощущения*` missed `ощущение`). Utterances fix the first; the `stem`
columns and `unicode61` fix the second.

A query is always one phrase. Tokens are normalized and wrapped in a
single FTS `"..."`, so adjacency is enforced and a token that spells
`OR` or `NEAR` is text rather than an operator. There are no prefix
wildcards: the old code needed them because nothing stemmed.

Tier order is strict: the exact phrase over `normalized_text`, then the
same phrase walked across utterance boundaries, then the stemmed phrase
over `stem_text`, then that walked. Anything else would report an
inflection while an exact occurrence existed.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, normalize_text, stem_text

__all__ = [
    "AnchorSpan",
    "MatchTier",
    "SearchHit",
    "SearchResult",
    "anchor_filename",
    "anchor_for",
    "fts_phrase",
    "parse_anchor",
    "query_tokens",
    "rarest_token_index",
    "search",
    "span_for_anchor",
    "tokens_for_tier",
    "walk_matches",
]


class MatchTier(StrEnum):
    """Which column answered. Reported, so a user knows an inflection."""

    EXACT = "exact"
    STEM = "stem"


#: (tier, utterances_fts column, words column). Ordered: exact first.
#: These column names are interpolated into SQL, so they must stay
#: literals from this tuple and never come from a caller.
_TIERS: Final = (
    (MatchTier.EXACT, "normalized_text", "normalized_text"),
    (MatchTier.STEM, "stem_text", "stem"),
)

_ANCHOR_RE: Final = re.compile(r"^v(\d+):(\d+)-(\d+)$")


@dataclass(frozen=True)
class SearchHit:
    """One place a phrase was said."""

    video_id: int
    video_title: str
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    speaker: str | None
    text: str
    source: str  # contracts §3: caption | timed | aligned
    cuttable: bool
    tier: MatchTier
    crosses_utterances: bool

    @property
    def anchor(self) -> str:
        """Stable handle for this span: video id plus word-ordinal range."""
        return anchor_for(self.video_id, self.first_word_ord, self.last_word_ord)


@dataclass(frozen=True)
class SearchResult:
    """Hits plus which tier produced them. `tier` is None when none did."""

    tier: MatchTier | None
    tokens: tuple[str, ...]
    hits: tuple[SearchHit, ...]


@dataclass(frozen=True)
class AnchorSpan:
    """What an anchor points at, resolved against the words table."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    source: str
    cuttable: bool


# -- anchors ----------------------------------------------------------


def anchor_for(video_id: int, first_word_ord: int, last_word_ord: int) -> str:
    """The stable handle a search prints and a transcript block carries."""
    return f"v{video_id}:{first_word_ord}-{last_word_ord}"


def parse_anchor(anchor: str) -> tuple[int, int, int]:
    """(video_id, first_word_ord, last_word_ord) from `v12:340-341`."""
    match = _ANCHOR_RE.match(anchor.strip())
    if match is None:
        raise InvalidInputError(
            f"{anchor!r} is not an anchor; expected v<video>:<first>-<last>, e.g. v12:340-341"
        )
    video_id, first, last = (int(match[1]), int(match[2]), int(match[3]))
    if last < first:
        raise InvalidInputError(f"{anchor!r} ends before it starts")
    return video_id, first, last


def anchor_filename(anchor: str, suffix: str = ".wav") -> str:
    """An anchor as a filename. Windows has no ':' in a path component."""
    return anchor.replace(":", "_") + suffix


# -- queries ----------------------------------------------------------


def query_tokens(query: str) -> tuple[str, ...]:
    """Normalize a user query into the tokens the index stores."""
    tokens = tuple(normalize_text(query).split())
    if not tokens:
        raise InvalidInputError(f"nothing searchable in {query!r}")
    return tokens


def tokens_for_tier(tokens: Sequence[str], tier: MatchTier) -> tuple[str, ...]:
    """The same query, spelled for one tier's column."""
    if tier is MatchTier.EXACT:
        return tuple(tokens)
    return tuple(stem_text(" ".join(tokens)).split())


def fts_phrase(column: str, tokens: Sequence[str]) -> str:
    """One column-scoped FTS5 phrase query.

    Tokens have been through `normalize_text`, so they hold only word
    characters and cannot carry a quote or an operator; stripping `"` is
    belt and braces for a caller that hands over raw text.
    """
    phrase = " ".join(token.replace('"', "") for token in tokens)
    return f'{column} : "{phrase}"'


# -- locating a phrase inside a range of words ------------------------

_SELECT_RANGE: Final = (
    "SELECT ord, start_ms, end_ms, text, normalized_text, stem, source, video_speaker_id"
    " FROM words WHERE video_id = ? AND ord BETWEEN ? AND ? ORDER BY ord"
)


@dataclass(frozen=True)
class _WordRun:
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    source: str
    cuttable: bool
    video_speaker_id: int | None


def _end_of(row: sqlite3.Row) -> int:
    """A word's end, inventing one for caption rows (contracts §3)."""
    if row["end_ms"] is not None:
        return int(row["end_ms"])
    return int(row["start_ms"]) + C.CAPTION_WORD_FALLBACK_MS


def _tier_of(rows: Sequence[sqlite3.Row]) -> str:
    """The weakest tier in a run — a run is only as cuttable as its worst row.

    Contracts §3 ranks them `caption` < `timed` < `aligned`. A run should
    be uniform in practice, since a tier is promoted for a whole video at
    a time, but reporting the weakest is the answer that cannot mislead.
    """
    return min(
        (str(row["source"]) for row in rows),
        key=lambda source: C.WORD_SOURCE_RANK.index(source)
        if source in C.WORD_SOURCE_RANK
        else -1,
    )


def _run_of(rows: Sequence[sqlite3.Row]) -> _WordRun:
    first, last = rows[0], rows[-1]
    source = _tier_of(rows)
    return _WordRun(
        first_word_ord=int(first["ord"]),
        last_word_ord=int(last["ord"]),
        start_ms=int(first["start_ms"]),
        end_ms=max(_end_of(last), int(first["start_ms"])),
        text=" ".join(str(row["text"]) for row in rows),
        source=source,
        cuttable=source == C.CUTTABLE_SOURCE,
        video_speaker_id=(
            None if first["video_speaker_id"] is None else int(first["video_speaker_id"])
        ),
    )


def _locate(
    rows: Sequence[sqlite3.Row], tokens: Sequence[str], word_column: str
) -> _WordRun | None:
    """The word rows spanning the first occurrence of `tokens`.

    A plain subsequence search, because contracts §4 stores exactly one
    token per row: `кто-то` is two rows with a measured boundary between
    them, not one row holding two tokens.
    """
    wanted = list(tokens)
    width = len(wanted)
    if width == 0 or width > len(rows):
        return None
    for start in range(len(rows) - width + 1):
        window = rows[start : start + width]
        if [str(row[word_column]) for row in window] == wanted:
            return _run_of(window)
    return None


def _words_in(db: Database, video_id: int, lo: int, hi: int) -> list[sqlite3.Row]:
    return list(db.conn.execute(_SELECT_RANGE, (video_id, lo, hi)))


# -- the FTS tier -----------------------------------------------------

_CANDIDATE_COLUMNS: Final = (
    "u.video_id AS video_id, u.first_word_ord AS first_word_ord,"
    " u.last_word_ord AS last_word_ord, u.text AS text, v.title AS video_title,"
    " COALESCE(s.label, vs.local_label) AS speaker"
)

#: An utterance is cuttable only if every word in it is aligned.
_HAS_UNCUTTABLE_WORD: Final = (
    "EXISTS (SELECT 1 FROM words w WHERE w.video_id = u.video_id"
    " AND w.ord BETWEEN u.first_word_ord AND u.last_word_ord"
    f" AND w.source <> '{C.CUTTABLE_SOURCE}')"
)


def _fts_candidates(
    db: Database,
    expr: str,
    *,
    speaker_ids: frozenset[int] | None,
    cuttable_only: bool,
    video_id: int,
    limit: int,
) -> list[sqlite3.Row]:
    """Utterances matching one FTS expression, filtered in SQL.

    The filters are applied before LIMIT on purpose: filtering afterwards
    would silently return fewer rows than asked for.
    """
    sql = [
        f"SELECT {_CANDIDATE_COLUMNS}, bm25(utterances_fts) AS score",
        "FROM utterances_fts",
        "JOIN utterances u ON u.id = utterances_fts.rowid",
        "JOIN videos v ON v.id = u.video_id",
        "LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id",
        "LEFT JOIN speakers s ON s.id = vs.speaker_id",
        "WHERE utterances_fts MATCH ?",
    ]
    params: list[Any] = [expr]
    if video_id:
        sql.append("AND u.video_id = ?")
        params.append(video_id)
    if speaker_ids:
        # Resolved ids, never a label: contracts §5 keeps the two speaker
        # identifier spaces apart and resolves both in rytp.commands.
        # `search` has already returned early for an empty set, so the
        # IN list below is never empty.
        placeholders = ", ".join("?" * len(speaker_ids))
        sql.append(f"AND u.video_speaker_id IN ({placeholders})")
        params.extend(sorted(speaker_ids))
    if cuttable_only:
        sql.append(f"AND NOT {_HAS_UNCUTTABLE_WORD}")
    sql.append("ORDER BY score, u.video_id, u.start_ms LIMIT ?")
    params.append(limit)
    return list(db.conn.execute("\n".join(sql), params))


def _hit_from_candidate(
    db: Database,
    row: sqlite3.Row,
    tokens: Sequence[str],
    word_column: str,
    tier: MatchTier,
) -> SearchHit:
    rows = _words_in(
        db, int(row["video_id"]), int(row["first_word_ord"]), int(row["last_word_ord"])
    )
    # FTS matched, so the utterance holds the phrase. If the row scan
    # disagrees, the cause is a character unicode61 tokenizes on that
    # `normalize_text` keeps — `_` is the only one — so the row is one
    # token and the index saw two. Falling back to the whole utterance is
    # honest; dropping the hit would not be.
    run = _locate(rows, tokens, word_column) or _run_of(rows)
    return SearchHit(
        video_id=int(row["video_id"]),
        video_title=str(row["video_title"]),
        first_word_ord=run.first_word_ord,
        last_word_ord=run.last_word_ord,
        start_ms=run.start_ms,
        end_ms=run.end_ms,
        speaker=None if row["speaker"] is None else str(row["speaker"]),
        text=str(row["text"]),
        source=run.source,
        cuttable=run.cuttable,
        tier=tier,
        crosses_utterances=False,
    )


# -- the cross-boundary walk ------------------------------------------


def rarest_token_index(db: Database, tokens: Sequence[str], word_column: str) -> int:
    """Which token to anchor the walk on: the one with fewest occurrences.

    Counting is capped at ``SEARCH_RARITY_PROBE_LIMIT`` by a subquery with
    its own LIMIT, because the ranking only needs to distinguish "rare"
    from "everywhere" and a full count of a Russian function word would
    scan hundreds of thousands of index entries.

    ``word_column`` comes from ``_TIERS`` and is never caller-supplied,
    which is what makes the interpolation below safe.
    """
    best_index = 0
    best_count: int | None = None
    for index, token in enumerate(tokens):
        count = int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM words WHERE {word_column} = ? LIMIT ?)",
                (token, C.SEARCH_RARITY_PROBE_LIMIT),
            ).fetchone()[0]
        )
        if best_count is None or count < best_count:
            best_index, best_count = index, count
        if count == 0:
            break
    return best_index


def _row_run(
    rows: Sequence[sqlite3.Row], tokens: Sequence[str], word_column: str
) -> _WordRun | None:
    """One words row per query token, in order, from one speaker.

    Contracts §4 stores one token per row, so this is the same row-wise
    comparison :func:`_locate` makes; the difference is that the window
    is fixed by the anchor's ordinal rather than searched for, and that a
    partial window is a miss rather than a shorter match.

    The single-speaker requirement is the rule that makes bridging safe.
    A split on silence or on a length cap is an artifact of how this code
    cut the transcript and gets bridged; a split on a speaker change is a
    fact about the recording and never does.
    """
    if len(rows) != len(tokens):
        return None
    if any(
        str(row[word_column]) != token for row, token in zip(rows, tokens, strict=True)
    ):
        return None
    if len({row["video_speaker_id"] for row in rows}) != 1:
        return None
    return _run_of(rows)


_CONTEXT_SQL: Final = (
    "SELECT u.text AS text, COALESCE(s.label, vs.local_label) AS speaker"
    " FROM utterances u"
    " LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id"
    " LEFT JOIN speakers s ON s.id = vs.speaker_id"
    " WHERE u.video_id = ? AND u.last_word_ord >= ? AND u.first_word_ord <= ?"
    " ORDER BY u.first_word_ord"
)


def _context(db: Database, video_id: int, lo: int, hi: int) -> tuple[str, str | None, bool]:
    """The surrounding sentence(s), the speaker, and whether it spans rows."""
    rows = list(db.conn.execute(_CONTEXT_SQL, (video_id, lo, hi)))
    if not rows:
        return "", None, False
    text = " ".join(str(row["text"]) for row in rows)
    speaker = None if rows[0]["speaker"] is None else str(rows[0]["speaker"])
    return text, speaker, len(rows) > 1


def walk_matches(
    db: Database,
    tokens: Sequence[str],
    word_column: str,
    tier: MatchTier,
    *,
    speaker_ids: frozenset[int] | None,
    cuttable_only: bool,
    video_id: int,
    limit: int,
) -> list[SearchHit]:
    """Occurrences an utterance split hid from the FTS phrase query.

    Anchors on the rarest token through ``words(normalized_text)`` /
    ``words(stem)`` — design §8's assembly access path — then walks by
    ordinal, which nothing about utterance boundaries constrains.
    """
    anchor_index = rarest_token_index(db, tokens, word_column)
    sql = [f"SELECT video_id, ord FROM words WHERE {word_column} = ?"]
    params: list[Any] = [tokens[anchor_index]]
    if video_id:
        sql.append("AND video_id = ?")
        params.append(video_id)
    sql.append("ORDER BY video_id, ord LIMIT ?")
    params.append(C.SEARCH_WALK_ANCHOR_LIMIT)

    titles: dict[int, str] = {}
    hits: list[SearchHit] = []
    for anchor in db.conn.execute("\n".join(sql), params):
        found = int(anchor["video_id"])
        lo = int(anchor["ord"]) - anchor_index
        if lo < 0:
            continue
        run = _row_run(_words_in(db, found, lo, lo + len(tokens) - 1), tokens, word_column)
        if run is None:
            continue
        if cuttable_only and not run.cuttable:
            continue
        # `_row_run` already required one speaker across the whole run, so
        # the run's id is the run's speaker.
        if speaker_ids is not None and run.video_speaker_id not in speaker_ids:
            continue
        text, found_speaker, crosses = _context(
            db, found, run.first_word_ord, run.last_word_ord
        )
        if not text:
            # The words exist but no utterance covers them, so this video
            # is not indexed — `index.drop` ran, or `index build` has not.
            # The walk bridges utterance *splits*; it is not a substitute
            # for the index, and answering here would make `index.drop` a
            # lie and make result quality depend on whether a video
            # happened to be indexed.
            continue
        if found not in titles:
            row = db.conn.execute("SELECT title FROM videos WHERE id = ?", (found,)).fetchone()
            titles[found] = "" if row is None else str(row["title"])
        hits.append(
            SearchHit(
                video_id=found,
                video_title=titles[found],
                first_word_ord=run.first_word_ord,
                last_word_ord=run.last_word_ord,
                start_ms=run.start_ms,
                end_ms=run.end_ms,
                speaker=found_speaker,
                text=text or run.text,
                source=run.source,
                cuttable=run.cuttable,
                tier=tier,
                crosses_utterances=crosses,
            )
        )
        if len(hits) >= limit:
            break
    return hits


# -- the public lookup ------------------------------------------------


def search(
    db: Database,
    query: str,
    *,
    speaker_ids: frozenset[int] | None = None,
    cuttable_only: bool = False,
    video_id: int = 0,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
) -> SearchResult:
    """Every place a phrase was said. Exact column first, stems second.

    `speaker_ids` is a set of `video_speakers.id`, already resolved — this
    layer never sees a label. Contracts §5 gives the two speaker
    identifier spaces one shared resolver in `rytp.commands`, and keeping
    the resolution there is what stops `--speaker` meaning two different
    things in two commands. `None` means no speaker filter; an **empty**
    set means a filter that nothing can satisfy, which is not the same
    thing and must return no hits rather than everything.

    `video_id=0` means every video — the commands layer passes a scalar
    and 0 is not a row id.
    """
    base = query_tokens(query)
    if speaker_ids is not None and not speaker_ids:
        return SearchResult(tier=None, tokens=base, hits=())
    capped = max(1, min(int(limit), C.SEARCH_MAX_LIMIT))
    for tier, fts_column, word_column in _TIERS:
        tokens = tokens_for_tier(base, tier)
        hits = [
            _hit_from_candidate(db, row, tokens, word_column, tier)
            for row in _fts_candidates(
                db,
                fts_phrase(fts_column, tokens),
                speaker_ids=speaker_ids,
                cuttable_only=cuttable_only,
                video_id=video_id,
                limit=capped,
            )
        ]
        if not hits and len(tokens) > 1:
            # An utterance boundary is this code's decision, not a fact
            # about the recording, so a split must not hide a phrase.
            # Fallback-only: for a given tier either the FTS hits or the
            # walked hits are returned, never both, so nothing is deduped.
            hits = walk_matches(
                db,
                tokens,
                word_column,
                tier,
                speaker_ids=speaker_ids,
                cuttable_only=cuttable_only,
                video_id=video_id,
                limit=capped,
            )
        if hits:
            return SearchResult(tier=tier, tokens=tokens, hits=tuple(hits))
    return SearchResult(tier=None, tokens=base, hits=())


def span_for_anchor(db: Database, anchor: str) -> AnchorSpan:
    """Resolve `v12:340-341` against the words table."""
    video_id, first, last = parse_anchor(anchor)
    rows = _words_in(db, video_id, first, last)
    if not rows:
        raise NotFoundError(f"anchor {anchor} matches no words")
    run = _run_of(rows)
    return AnchorSpan(
        video_id=video_id,
        first_word_ord=run.first_word_ord,
        last_word_ord=run.last_word_ord,
        start_ms=run.start_ms,
        end_ms=run.end_ms,
        text=run.text,
        source=run.source,
        cuttable=run.cuttable,
    )
