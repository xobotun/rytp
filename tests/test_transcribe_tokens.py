"""One word row holds exactly one token — the rule Parts 4 and 5 depend on."""
from __future__ import annotations

import subprocess
import sys

import pytest

from rytp.models import normalize_text
from rytp.transcribe.base import split_token


def test_importing_base_does_not_pull_in_snowballstemmer_at_module_level() -> None:
    """`rytp.transcribe.base` must stay standard-library-only (review findings).

    It is imported again inside the out-of-process seam under a foreign
    interpreter, where only the standard library and :mod:`rytp.models` are
    guaranteed to load. `rytp.models` keeps that true by importing
    ``snowballstemmer`` lazily, inside :func:`rytp.models.stem_text`, rather
    than at module level — this pins that both stay true, in a fresh
    subprocess so an import cached by an earlier test cannot hide a
    regression.
    """
    script = (
        "import sys\n"
        "import rytp.transcribe.base\n"
        "assert 'snowballstemmer' not in sys.modules, "
        "'snowballstemmer was imported at module level'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_a_plain_word_is_one_token() -> None:
    assert split_token("слово") == [("слово", "слово")]


def test_a_hyphenated_word_becomes_two_rows() -> None:
    assert split_token("кто-то") == [("кто", "кто"), ("то", "то")]
    assert split_token("из-за") == [("из", "из"), ("за", "за")]


def test_trailing_punctuation_does_not_create_an_empty_token() -> None:
    assert split_token("слово,") == [("слово", "слово")]
    assert split_token("«слово»") == [("слово", "слово")]


def test_a_punctuation_only_token_yields_nothing() -> None:
    assert split_token("—") == []
    assert split_token("  ") == []


@pytest.mark.parametrize(
    "text", ["слово", "кто-то", "из-за", "по-моему", "слово,", "два слова", "—", "ещё"]
)
def test_the_split_is_exactly_what_the_shared_normalizer_would_produce(text: str) -> None:
    # The invariant that keeps this consistent with Part 4's query-side
    # normalization: splitting a token here must give the same sequence as
    # normalizing the whole thing and splitting on whitespace. If Part 1 ever
    # changes normalize_text's punctuation handling, this fails loudly.
    assert [n for _surface, n in split_token(text)] == normalize_text(text).split()
