"""Voice vectors: a storage format and four pieces of arithmetic."""

from __future__ import annotations

import math

import pytest

from rytp.db import Database
from rytp.diarize.embed import (
    EMBEDDERS,
    EmbeddingError,
    centroid,
    comparable,
    cosine_similarity,
    l2_normalize,
    load_embedder,
    pack_embedding,
    resolve_embedder,
    unpack_embedding,
)
from tests.fake_speaker_engines import FakeEmbedder, registered

# -- the codec -------------------------------------------------------------


def test_a_vector_round_trips_through_the_blob() -> None:
    values = [0.5, -0.25, 0.125, 0.0]
    assert unpack_embedding(pack_embedding(values)) == values


def test_the_blob_is_four_bytes_per_value_and_little_endian() -> None:
    blob = pack_embedding([1.0, 2.0])
    assert len(blob) == 8
    assert blob[:4] == b"\x00\x00\x80\x3f"


def test_float64_precision_is_lost_but_the_value_survives() -> None:
    [restored] = unpack_embedding(pack_embedding([0.1]))
    assert restored == pytest.approx(0.1, abs=1e-7)


def test_an_empty_vector_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        pack_embedding([])


def test_a_truncated_blob_is_refused_rather_than_silently_shortened() -> None:
    with pytest.raises(EmbeddingError):
        unpack_embedding(b"\x00\x00\x80")


# -- the arithmetic --------------------------------------------------------


def test_normalising_gives_a_unit_vector() -> None:
    values = l2_normalize([3.0, 4.0])
    assert values == pytest.approx([0.6, 0.8])
    assert math.isclose(sum(v * v for v in values), 1.0)


def test_normalising_a_zero_vector_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        l2_normalize([0.0, 0.0])


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-2.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude() -> None:
    assert cosine_similarity([1.0, 1.0], [7.0, 7.0]) == pytest.approx(1.0)


def test_cosine_refuses_vectors_of_different_lengths() -> None:
    # Two embedders in one corpus produce two dimensionalities. Comparing
    # them is meaningless and must say so rather than fail somewhere else.
    with pytest.raises(EmbeddingError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_comparable_is_the_question_to_ask_before_comparing() -> None:
    assert comparable([1.0, 0.0], [0.0, 1.0]) is True
    assert comparable([1.0, 0.0], [1.0, 0.0, 0.0]) is False
    assert comparable([], []) is False


def test_a_centroid_is_the_normalised_mean_of_normalised_vectors() -> None:
    assert centroid([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_a_centroid_of_one_vector_is_that_vector_normalised() -> None:
    assert centroid([[3.0, 4.0]]) == pytest.approx([0.6, 0.8])


def test_a_centroid_of_nothing_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        centroid([])


def test_a_centroid_of_mismatched_vectors_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        centroid([[1.0, 0.0], [1.0, 0.0, 0.0]])


# -- the registry ----------------------------------------------------------


def test_register_and_resolve_returns_the_class(db: Database) -> None:
    with registered(FakeEmbedder):
        assert resolve_embedder("fake-embedder") is FakeEmbedder
        assert isinstance(load_embedder(db, "fake-embedder"), FakeEmbedder)


def test_resolving_an_unknown_embedder_names_the_available_ones() -> None:
    with registered(FakeEmbedder), pytest.raises(ValueError) as excinfo:
        resolve_embedder("nope")
    assert "fake-embedder" in str(excinfo.value)


def test_the_registry_holds_classes_not_instances() -> None:
    with registered(FakeEmbedder):
        assert EMBEDDERS["fake-embedder"] is FakeEmbedder
