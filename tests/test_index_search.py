"""The backward index: phrase to places it was said (design §7)."""

from __future__ import annotations

import re

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.index.search import (
    MatchTier,
    anchor_filename,
    anchor_for,
    fts_phrase,
    parse_anchor,
    query_tokens,
    rarest_token_index,
    search,
    span_for_anchor,
)
from rytp.index.utterances import index_video
from rytp.models import InvalidInputError, NotFoundError, normalize_text, stem_text
from tests.test_index_utterances import make_speaker, make_video

NOW = "2026-09-21T00:00:00+00:00"


def stored_tokens(raw: str) -> list[tuple[str, str]]:
    """(surface, normalized) pairs, the way Part 3's `split_token` stores them.

    Contracts §4: a row holds exactly one token, and the tokens stored for
    a piece of text are exactly `normalize_text(text).split()`. So a
    hyphenated word is two rows. Seeding any other shape would test rows
    the real pipeline cannot produce.
    """
    pieces = re.split(r"[^\w]+", raw, flags=re.UNICODE)
    return [(piece, normalize_text(piece)) for piece in pieces if normalize_text(piece)]


def add_words(
    db: Database,
    video_id: int,
    text: str,
    *,
    start_ms: int = 0,
    first_ord: int = 0,
    speaker_id: int | None = None,
    source: str = "aligned",
    step_ms: int = 300,
    word_ms: int = 250,
) -> tuple[int, int]:
    """Append one words row per stored token. Returns (next_ord, next_start)."""
    ordinal = first_ord
    clock = start_ms
    for token, normalized in stored_tokens(text):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine, video_speaker_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'fake', ?)",
            (
                video_id,
                ordinal,
                clock,
                # Contracts §3: only caption rows have a null end.
                None if source == "caption" else clock + word_ms,
                token,
                normalized,
                stem_text(normalized),
                source,
                speaker_id,
            ),
        )
        ordinal += 1
        clock += step_ms
    return ordinal, clock


def corpus(db: Database, text: str, *, source: str = "aligned", title: str = "Sample") -> int:
    """One video holding `text`, indexed."""
    video_id = make_video(db, title)
    add_words(db, video_id, text, source=source)
    index_video(db, video_id)
    return video_id


def name_speaker(db: Database, video_speaker_id: int, label: str) -> int:
    """Attach a global roster entry to a per-video diarizer label."""
    speaker_id = int(
        db.conn.execute(
            "INSERT INTO speakers (label, created_at) VALUES (?, ?)", (label, NOW)
        ).lastrowid
    )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE id = ?",
        (speaker_id, video_speaker_id),
    )
    return speaker_id


# --- the two regressions this part exists to fix ---------------------


def test_a_two_word_phrase_whose_words_are_adjacent_returns_a_hit(db: Database) -> None:
    """The exact failure measured on the owner's corpus.

    The replaced implementation shadowed `words` with FTS, one word per
    row, so a two-term query was an implicit AND inside a single row:
    `добрый*` returned 1 hit and `добрый* вечер*` returned 0 with the two
    words adjacent in the text. If this test ever goes back to zero, the
    index has regressed to the old shape.
    """
    corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "добрый").hits) == 1
    result = search(db, "добрый вечер")
    assert len(result.hits) == 1
    assert result.tier is MatchTier.EXACT


def test_a_russian_inflection_is_found_through_the_stem_tier(db: Database) -> None:
    """The second measured failure: the old tokenizer was `porter`, which
    is English-only, so `ощущения*` never reached the corpus's `ощущение`."""
    corpus(db, "Сегодня было странное ощущение")
    exact = search(db, "ощущение")
    assert exact.tier is MatchTier.EXACT
    stemmed = search(db, "ощущения")
    assert stemmed.tier is MatchTier.STEM
    assert len(stemmed.hits) == 1
    assert stemmed.hits[0].text == "Сегодня было странное ощущение"


# --- phrase semantics ------------------------------------------------


def test_words_that_are_present_but_not_adjacent_do_not_match(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    assert search(db, "добрый друзья").hits == ()


def test_a_phrase_absent_from_the_corpus_returns_nothing_and_no_tier(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    result = search(db, "совершенно другое")
    assert result.hits == ()
    assert result.tier is None


def test_the_stem_tier_is_only_consulted_when_the_exact_tier_is_empty(db: Database) -> None:
    """An exact hit must never be reported as an inflection."""
    corpus(db, "Они сказали слово")
    result = search(db, "сказали слово")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1


def test_yo_folding_works_in_both_directions(db: Database) -> None:
    """normalize_text folds ё→е (contracts §4), so neither spelling loses."""
    corpus(db, "И ещё раз", title="WithYo")
    assert len(search(db, "еще раз").hits) == 1
    assert len(search(db, "ещё раз").hits) == 1


def test_yo_folding_also_works_when_the_corpus_has_the_bare_e(db: Database) -> None:
    corpus(db, "И еще раз", title="WithoutYo")
    assert len(search(db, "ещё раз").hits) == 1


def test_a_hyphen_and_a_yo_in_the_same_query(db: Database) -> None:
    """The owner's case, and the one that falls between the other two.

    `кто-то` exercises contracts §4's one-row-per-token rule and `ещё`
    exercises the ё→е fold, and both normalizations have to agree on the
    query side and the corpus side at once. Four of the five spellings
    below differ from what is stored; all five must find the same span.
    """
    video_id = corpus(db, "Кто-то ещё об этом спрашивал")
    result = search(db, "кто-то ещё")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    assert result.hits[0].video_id == video_id
    assert (result.hits[0].first_word_ord, result.hits[0].last_word_ord) == (0, 2)
    for spelling in ("Кто-то ещё", "кто-то еще", "кто то ещё", "кто то еще"):
        assert len(search(db, spelling).hits) == 1, spelling


def test_fts_operators_inside_a_query_are_literal_text(db: Database) -> None:
    """Tokens go inside one quoted phrase, so OR/NEAR/AND are not operators."""
    corpus(db, "Может быть или нет")
    assert search(db, "может OR нет").hits == ()
    assert len(search(db, "или нет").hits) == 1


def test_punctuation_and_quotes_in_a_query_cannot_break_the_fts_expression(
    db: Database,
) -> None:
    corpus(db, "Может быть или нет")
    assert len(search(db, '"может" быть!').hits) == 1
    assert len(search(db, "может - быть").hits) == 1


def test_a_single_token_query_matches(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "друзья").hits) == 1


def test_an_empty_or_punctuation_only_query_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError):
        search(db, "")
    with pytest.raises(InvalidInputError):
        search(db, " ... ")


# --- what a hit carries ----------------------------------------------


def test_a_hit_locates_the_phrase_rather_than_the_whole_sentence(db: Database) -> None:
    """Playback and export use these timings; the sentence would be wrong."""
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    hit = search(db, "дорогие друзья").hits[0]
    assert hit.video_id == video_id
    assert (hit.first_word_ord, hit.last_word_ord) == (2, 3)
    assert hit.start_ms == 600
    assert hit.end_ms == 1150
    assert hit.text == "Добрый вечер дорогие друзья"


def test_a_hyphenated_word_is_two_rows_and_a_hit_spans_both(db: Database) -> None:
    """Contracts §4: `кто-то` is stored as two rows, not one row of two
    tokens, so the span covers both ordinals and its boundary is measured."""
    video_id = corpus(db, "Кто-то сказал")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 3
    hit = search(db, "кто то").hits[0]
    assert (hit.first_word_ord, hit.last_word_ord) == (0, 1)


def test_a_hit_carries_the_video_title_and_the_speaker(db: Database) -> None:
    video_id = make_video(db, "Evening")
    local = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, local, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    hit = search(db, "добрый вечер").hits[0]
    assert hit.video_title == "Evening"
    assert hit.speaker == "Ведущий"


def test_an_unmapped_diarizer_label_shows_as_the_raw_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert search(db, "добрый вечер").hits[0].speaker == "SPEAKER_00"


def test_an_undiarized_hit_has_no_speaker(db: Database) -> None:
    corpus(db, "Добрый вечер")
    assert search(db, "добрый вечер").hits[0].speaker is None


def test_the_anchor_round_trips(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    hit = search(db, "добрый вечер").hits[0]
    assert hit.anchor == f"v{video_id}:0-1"
    assert parse_anchor(hit.anchor) == (video_id, 0, 1)


# --- filters ---------------------------------------------------------


def test_cuttable_is_true_only_for_aligned_words(db: Database) -> None:
    """Contracts §3: cuttable is `source = 'aligned'` and nothing else."""
    corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    corpus(db, "Добрый вечер", source="timed", title="Timed")
    hits = search(db, "добрый вечер").hits
    assert sorted(hit.cuttable for hit in hits) == [False, False, True]


def test_cuttable_only_keeps_aligned_and_drops_the_other_two_tiers(
    db: Database,
) -> None:
    """Contracts §3: cuttable is `aligned` and nothing else. `timed` is the
    easy one to get wrong — it has real end times, it just has no measured
    boundary to cut on."""
    aligned = corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    corpus(db, "Добрый вечер", source="timed", title="Timed")
    hits = search(db, "добрый вечер", cuttable_only=True).hits
    assert [hit.video_id for hit in hits] == [aligned]


def test_a_hit_carries_its_transcript_tier(db: Database) -> None:
    corpus(db, "Добрый вечер", source="timed")
    hit = search(db, "добрый вечер").hits[0]
    assert hit.source == "timed"
    assert hit.cuttable is False


def test_the_speaker_filter_restricts_to_the_resolved_ids(db: Database) -> None:
    """This layer takes `video_speakers.id` values, never a label: the two
    identifier spaces are resolved once, in `rytp.commands` (contracts §5)."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    name_speaker(db, host, "Ведущий")
    ordinal, clock = add_words(db, video_id, "Добрый вечер", speaker_id=host)
    add_words(
        db, video_id, "Добрый вечер", first_ord=ordinal, start_ms=clock, speaker_id=guest
    )
    index_video(db, video_id)
    assert len(search(db, "добрый вечер").hits) == 2
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host})).hits) == 1
    assert search(db, "добрый вечер", speaker_ids=frozenset({host})).hits[0].speaker == (
        "Ведущий"
    )
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({guest})).hits) == 1
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host, guest})).hits) == 2


def test_an_empty_speaker_set_matches_nothing_rather_than_everything(
    db: Database,
) -> None:
    """A resolver that found no labels for a real person must not widen
    the search — that is the silent-wrong-answer failure mode."""
    corpus(db, "Добрый вечер")
    assert search(db, "добрый вечер", speaker_ids=frozenset()).hits == ()
    assert search(db, "добрый вечер", speaker_ids=None).hits != ()


def test_the_video_filter_restricts_to_one_video(db: Database) -> None:
    first = corpus(db, "Добрый вечер", title="First")
    corpus(db, "Добрый вечер", title="Second")
    hits = search(db, "добрый вечер", video_id=first).hits
    assert [hit.video_id for hit in hits] == [first]


def test_the_limit_is_honoured_and_capped(db: Database) -> None:
    for index in range(5):
        corpus(db, "Добрый вечер", title=f"V{index}")
    assert len(search(db, "добрый вечер", limit=2).hits) == 2
    assert len(search(db, "добрый вечер", limit=C.SEARCH_MAX_LIMIT * 10).hits) == 5
    assert len(search(db, "добрый вечер", limit=0).hits) == 1


# --- helpers ---------------------------------------------------------


def test_query_tokens_normalizes_and_splits() -> None:
    assert query_tokens("  Добрый, ВЕЧЕР! ") == ("добрый", "вечер")


def test_fts_phrase_scopes_to_a_column_and_quotes_the_whole_run() -> None:
    assert fts_phrase("normalized_text", ("добрый", "вечер")) == (
        'normalized_text : "добрый вечер"'
    )


def test_anchor_helpers_are_windows_safe() -> None:
    anchor = anchor_for(12, 340, 341)
    assert anchor == "v12:340-341"
    assert ":" not in anchor_filename(anchor)
    assert anchor_filename(anchor) == "v12_340-341.wav"


def test_parse_anchor_rejects_rubbish() -> None:
    for bad in ("", "12:0-1", "v12:0", "vx:0-1", "v12:3-1"):
        with pytest.raises(InvalidInputError):
            parse_anchor(bad)


def test_span_for_anchor_returns_the_words_own_timings(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    span = span_for_anchor(db, anchor_for(video_id, 2, 3))
    assert (span.start_ms, span.end_ms) == (600, 1150)
    assert span.text == "дорогие друзья"
    assert span.cuttable is True


def test_span_for_anchor_raises_when_the_range_holds_no_words(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    with pytest.raises(NotFoundError):
        span_for_anchor(db, anchor_for(video_id, 900, 901))


# --- phrases that cross an utterance boundary ------------------------


def split_corpus(db: Database, title: str = "Split") -> int:
    """A video where `добрый` and `вечер` land in different utterances.

    The split is a silence longer than the threshold, which is an artifact
    of how this code chose to cut the transcript, not a fact about the
    recording — so the phrase must still be findable.
    """
    video_id = make_video(db, title)
    ordinal, clock = add_words(db, video_id, "Добрый")
    add_words(
        db,
        video_id,
        "вечер друзья",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    return video_id


def test_the_split_really_did_produce_two_utterances(db: Database) -> None:
    """Guard for the tests below: if this is one row they prove nothing."""
    video_id = split_corpus(db)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2


def test_a_phrase_split_by_a_silence_is_still_found(db: Database) -> None:
    video_id = split_corpus(db)
    result = search(db, "добрый вечер")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.video_id == video_id
    assert (hit.first_word_ord, hit.last_word_ord) == (0, 1)
    assert hit.crosses_utterances is True
    assert "Добрый" in hit.text
    assert "вечер" in hit.text


def test_a_phrase_split_by_a_speaker_change_is_not_a_hit(db: Database) -> None:
    """Two people each saying half of it is not a place it was said."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    ordinal, clock = add_words(db, video_id, "Добрый", speaker_id=host)
    add_words(db, video_id, "вечер", first_ord=ordinal, start_ms=clock, speaker_id=guest)
    index_video(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2
    assert search(db, "добрый вечер").hits == ()


def test_the_walk_never_bridges_two_videos(db: Database) -> None:
    corpus(db, "Добрый", title="Ends")
    corpus(db, "вечер", title="Begins")
    assert search(db, "добрый вечер").hits == ()


def test_a_hit_inside_one_utterance_does_not_cross(db: Database) -> None:
    corpus(db, "Добрый вечер")
    hit = search(db, "добрый вечер").hits[0]
    assert hit.crosses_utterances is False


def test_the_walk_does_not_run_when_the_fts_tier_answered(db: Database) -> None:
    """Known limitation, pinned: an in-utterance hit suppresses the walk, so
    a bridged occurrence elsewhere is not reported alongside it."""
    inline = corpus(db, "Добрый вечер", title="Inline")
    split_corpus(db, title="Split")
    hits = search(db, "добрый вечер").hits
    assert [hit.video_id for hit in hits] == [inline]


def test_a_bridged_phrase_is_reachable_through_the_stem_tier(db: Database) -> None:
    video_id = make_video(db, "SplitInflected")
    ordinal, clock = add_words(db, video_id, "странное")
    add_words(
        db,
        video_id,
        "ощущение",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    result = search(db, "странные ощущения")
    assert result.tier is MatchTier.STEM
    assert len(result.hits) == 1
    assert result.hits[0].video_id == video_id
    assert result.hits[0].crosses_utterances is True


def test_a_bridged_hit_honours_the_cuttable_filter(db: Database) -> None:
    video_id = make_video(db, "SplitCaptions")
    ordinal, clock = add_words(db, video_id, "Добрый", source="caption")
    # A caption word's implied end is capped at CAPTION_WORD_FALLBACK_MS
    # after its start, so the silence has to clear both to split.
    add_words(
        db,
        video_id,
        "вечер",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS + C.CAPTION_WORD_FALLBACK_MS,
        source="caption",
    )
    index_video(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2
    assert len(search(db, "добрый вечер").hits) == 1
    assert search(db, "добрый вечер").hits[0].cuttable is False
    assert search(db, "добрый вечер", cuttable_only=True).hits == ()


def test_a_bridged_hit_honours_the_speaker_filter(db: Database) -> None:
    video_id = make_video(db, "SplitOneSpeaker")
    host = make_speaker(db, video_id, "SPEAKER_00")
    other = make_speaker(db, video_id, "SPEAKER_01")
    ordinal, clock = add_words(db, video_id, "Добрый", speaker_id=host)
    add_words(
        db,
        video_id,
        "вечер",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
        speaker_id=host,
    )
    index_video(db, video_id)
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host})).hits) == 1
    assert search(db, "добрый вечер", speaker_ids=frozenset({other})).hits == ()


def test_a_bridged_hit_honours_the_video_and_limit_filters(db: Database) -> None:
    first = split_corpus(db, title="SplitOne")
    split_corpus(db, title="SplitTwo")
    assert len(search(db, "добрый вечер").hits) == 2
    assert [h.video_id for h in search(db, "добрый вечер", video_id=first).hits] == [first]
    assert len(search(db, "добрый вечер", limit=1).hits) == 1


def test_the_walk_does_not_answer_for_an_unindexed_video(db: Database) -> None:
    """The walk bridges splits; it is not a substitute for the index.

    Without this, `index.drop` would leave a video findable through the
    walk alone — a hit with no surrounding sentence — and whether a
    result had context would depend on whether anyone had indexed it.
    """
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "добрый вечер").hits) == 1
    db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    assert search(db, "добрый вечер").hits == ()
    assert search(db, "друзья").hits == ()
    index_video(db, video_id)
    assert len(search(db, "добрый вечер").hits) == 1


def test_the_walk_anchors_on_the_rarest_token(db: Database) -> None:
    """A common function word would make the walk scan the whole corpus."""
    for index in range(5):
        corpus(db, "и что то ещё", title=f"Common{index}")
    corpus(db, "и редкое слово", title="Rare")
    assert rarest_token_index(db, ("и", "редкое"), "normalized_text") == 1
    assert rarest_token_index(db, ("редкое", "и"), "normalized_text") == 0


def test_rarest_token_index_stops_at_a_token_that_is_absent(db: Database) -> None:
    corpus(db, "Добрый вечер")
    assert rarest_token_index(db, ("добрый", "отсутствует"), "normalized_text") == 1
