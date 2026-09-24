"""What Part 4 assumes about its neighbours, as executable assertions.

Every fact here belongs to another part. This module exists so that when
one of them moves, exactly one test fails and names the thing that moved
— rather than a search quietly returning the wrong rows.
"""

from __future__ import annotations

import dataclasses
import threading

import pytest

from rytp.db import Database
from rytp.models import normalize_text, stem_text

# --- Part 1: the stemmer (contracts §4) ------------------------------


def test_stem_text_is_part_ones_and_already_works() -> None:
    """Part 4 does not implement this. Part 3 calls it when it writes
    words.stem, which is what makes it Part 1's."""
    assert stem_text("ощущения") == stem_text("ощущение")


def test_stem_text_is_one_stem_per_token_in_the_same_order() -> None:
    """The only two properties Part 4 depends on.

    `utterances.stem_text` is a join of `words.stem`, and a phrase
    position in the stem FTS column has to mean the same word as the
    matching position in the exact column. A stemmer that dropped or
    reordered a token would break the stem tier silently.
    """
    assert stem_text("добрый вечер дорогие друзья").split() == [
        "добр",
        "вечер",
        "дорог",
        "друз",
    ]
    assert stem_text("") == ""
    assert stem_text("   ") == ""


def test_stem_text_sees_only_normalized_input() -> None:
    """normalize_text folds ё→е (contracts §4), so the stemmer never has
    to know about the two spellings."""
    assert stem_text(normalize_text("ещё")) == stem_text(normalize_text("еще"))


def test_stem_text_is_safe_to_call_from_several_threads() -> None:
    """Part 2's worker runs three pools as threads in one process, and the
    index job stems on the cpu pool while transcription stems on the gpu
    pool."""
    words = ["ощущения", "сказали", "добрым", "вечера", "дорогие", "мысли"]
    expected = [stem_text(word) for word in words]
    results: list[list[str]] = []
    lock = threading.Lock()

    def worker() -> None:
        got = [stem_text(word) for _ in range(200) for word in words]
        with lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [expected * 200] * 8


# --- Part 3: one row, one token (contracts §4) -----------------------


def test_a_stored_word_row_holds_exactly_one_token() -> None:
    """Contracts §4: "the tokens stored for a piece of text are exactly
    `normalize_text(text).split()`". Part 4's span locator is a plain
    subsequence search over rows because of this."""
    from rytp.transcribe.base import split_token

    assert [normalized for _, normalized in split_token("Кто-то")] == ["кто", "то"]
    for surface, normalized in split_token("Кто-то ещё"):
        assert len(normalized.split()) == 1, (surface, normalized)
    assert split_token("—") == []


# --- Part 1: the schema Part 4 writes into (contracts §3) ------------


def test_the_fts_tokenizer_is_unicode61_and_never_porter(db: Database) -> None:
    """Porter is English-only; on a Russian corpus it stems nothing at
    all, which is the second of the two measured failures this part
    exists to fix."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "unicode61" in sql
    assert "porter" not in sql


def test_the_fts_shadow_has_both_columns(db: Database) -> None:
    """Exact first, stems as the fallback (design §7) needs two columns."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "normalized_text" in sql
    assert "stem_text" in sql


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("video_speakers", "engine"),
        ("words", "engine"),
        ("words", "stem"),
        ("utterances", "stem_text"),
    ],
)
def test_these_columns_are_not_null_so_every_fixture_must_supply_them(
    db: Database, table: str, column: str
) -> None:
    """Contracts §3 makes supplying a NOT NULL column the inserting part's
    job. A fixture that omits one fails at insert time, which is fine —
    but it fails in whichever test happens to run first, so it is worth
    naming the columns here."""
    info = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    not_null = {row[1] for row in info if row[3]}
    assert column in not_null


# --- Part 1: the shared speaker resolver (contracts §5) --------------


def test_the_registries_part_four_writes_into_exist() -> None:
    """Contracts §5: job kinds, commands and health checks all register
    into `rytp.commands` / `rytp.jobs`; Part 4 adds to three of them."""
    import rytp.commands as commands

    assert hasattr(commands, "register_check")
    assert hasattr(commands, "HEALTH_CHECKS")


def test_the_shared_speaker_resolver_has_the_pinned_signature() -> None:
    """Contracts §5 fixes this exactly, because three parts each assumed a
    different shape when only the responsibility was named."""
    import inspect

    import rytp.commands as commands

    assert hasattr(commands, "SpeakerFilter")
    assert hasattr(commands, "resolve_speaker_filter")
    fields = {f.name for f in dataclasses.fields(commands.SpeakerFilter)}
    assert fields == {"video_speaker_ids", "description"}
    params = inspect.signature(commands.resolve_speaker_filter).parameters
    assert set(params) >= {"db", "speaker", "video_local_speaker", "video_id"}
