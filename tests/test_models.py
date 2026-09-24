"""Contracts §4 types, the error base class, and the text normalizer."""

from __future__ import annotations

import dataclasses
import re

import pytest

from rytp.models import (
    ChannelEntry,
    DiarSegment,
    Fragment,
    InvalidInputError,
    NotFoundError,
    RawWord,
    RytpError,
    Span,
    normalize_text,
    stem_text,
    utc_now_iso,
)


def test_every_core_type_is_a_frozen_dataclass() -> None:
    for cls in (RawWord, Span, DiarSegment, Fragment, ChannelEntry):
        assert dataclasses.is_dataclass(cls), cls
        assert cls.__dataclass_params__.frozen, cls


def test_core_type_fields_match_contracts_section_4() -> None:
    def names(cls: type) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    assert names(RawWord) == ["text", "start_ms", "end_ms", "confidence"]
    assert names(Span) == ["start_ms", "end_ms", "score"]
    assert names(DiarSegment) == ["start_ms", "end_ms", "local_label"]
    assert names(Fragment) == [
        "video_id",
        "first_word_ord",
        "last_word_ord",
        "start_ms",
        "end_ms",
        "text",
    ]
    assert names(ChannelEntry) == [
        "external_id",
        "title",
        "url",
        "duration_ms",
        "kind",
        "published_at",
    ]


def test_a_transcriber_may_emit_text_with_no_timings() -> None:
    """Contracts §4: every RawWord field but the text is optional."""
    word = RawWord(text="привет")
    assert word.start_ms is None
    assert word.end_ms is None
    assert word.confidence is None
    assert RawWord("привет", 0, 100, 0.9) == RawWord(
        text="привет", start_ms=0, end_ms=100, confidence=0.9
    )


def test_stem_text_is_importable_from_models() -> None:
    """Part 3 and part 4 both import it from here; nothing imports rytp.index."""
    assert stem_text.__module__ == "rytp.models"


def test_domain_errors_share_one_base() -> None:
    assert issubclass(NotFoundError, RytpError)
    assert issubclass(InvalidInputError, RytpError)
    assert issubclass(RytpError, Exception)


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize_text("  Привет   МИР \n") == "привет мир"


def test_normalize_drops_punctuation_and_leaves_a_word_boundary() -> None:
    assert normalize_text("Привет, мир!") == "привет мир"
    assert normalize_text("кто-то") == "кто то"


def test_normalize_folds_yo_to_ye() -> None:
    """Captions and ASR disagree about ё constantly; search must not care.

    The FTS tokenizer is `unicode61 remove_diacritics 0` precisely so it
    does NOT fold Cyrillic letters (it would wreck й), so the ё/е fold
    has to happen here, at write time.
    """
    assert normalize_text("ЁЖ и ещё") == "еж и еще"


def test_normalize_keeps_short_i_intact() -> None:
    """й must survive: it is a letter, not an и with a diacritic."""
    assert normalize_text("Мой") == "мой"


def test_normalize_is_idempotent() -> None:
    once = normalize_text("Привет, мир!")
    assert normalize_text(once) == once


def test_normalize_handles_empty_and_punctuation_only_input() -> None:
    assert normalize_text("") == ""
    assert normalize_text("  ...  ") == ""


def test_utc_now_iso_is_an_iso_utc_timestamp() -> None:
    stamp = utc_now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$", stamp), stamp


def test_stem_text_strips_russian_endings() -> None:
    """Contracts §4: real from the start — part 3 calls it to fill words.stem."""
    assert stem_text("сказали слово") == "сказа слов"
    assert stem_text("говорил") == "говор"
    assert stem_text("говорить") == "говор"


def test_stem_text_leaves_an_unstemmable_word_alone() -> None:
    assert stem_text("привет мир") == "привет мир"


def test_stem_text_handles_empty_input() -> None:
    assert stem_text("") == ""
    assert stem_text("   ") == ""


def test_stem_text_is_token_for_token() -> None:
    """One token in, one token out — utterances are joined, never re-stemmed."""
    normalized = normalize_text("Сказали кто-то слово")
    assert len(stem_text(normalized).split()) == len(normalized.split())


def test_stemming_is_not_idempotent() -> None:
    """Contracts §4 warns about this: never re-stem an already-stemmed string."""
    once = stem_text("сказали")
    assert stem_text(once) != once


def test_the_yo_fold_reaches_the_stem() -> None:
    """ещё and еще must land on the same stem, or search splits in two."""
    assert stem_text(normalize_text("ещё")) == stem_text(normalize_text("еще"))


def test_part_one_never_imports_the_index_package() -> None:
    """`rytp/index/` is part 4's. Checked in a subprocess so nothing else can mask it."""
    import subprocess
    import sys

    from tests.test_config import child_env

    probe = (
        "import sys, rytp, rytp.config, rytp.constants, rytp.models;"
        "assert not [n for n in sys.modules if n.startswith('rytp.index')], sys.modules.keys()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], env=child_env(), capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_frozen_types_reject_mutation() -> None:
    word = RawWord(start_ms=0, end_ms=100, text="да", confidence=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        word.text = "нет"  # type: ignore[misc]
