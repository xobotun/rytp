"""Reconstructing a label's speech from its words, and embedding it."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import (
    EMBEDDERS,
    EmbeddingError,
    concat_wav_windows,
    unpack_embedding,
)
from rytp.diarize.pipeline import embed_video_speakers, embedder_name
from rytp.models import DiarSegment
from tests.fake_speaker_engines import ConstantEmbedder, FakeEmbedder
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def a_wav(path: Path, ms: int) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db)
    word_ids = add_words(
        db, video_id, [(0, 2_000), (2_000, 4_000), (10_000, 12_000), (12_000, 14_000)]
    )
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db,
        video_id,
        {word_ids[0]: host, word_ids[1]: host, word_ids[2]: guest, word_ids[3]: guest},
    )
    return video_id


# -- reconstructing segments from words ------------------------------------


def test_consecutive_words_of_one_label_become_one_segment(db: Database) -> None:
    video_id = a_diarized_video(db)
    assert store.label_segments(db, video_id) == [
        DiarSegment(start_ms=0, end_ms=4_000, local_label="SPEAKER_00"),
        DiarSegment(start_ms=10_000, end_ms=14_000, local_label="SPEAKER_01"),
    ]


def test_a_label_that_speaks_twice_gets_two_segments(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db, video_id, {word_ids[0]: host, word_ids[1]: guest, word_ids[2]: host}
    )
    assert [s.local_label for s in store.label_segments(db, video_id)] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]


def test_unlabelled_words_break_a_run_and_contribute_nothing(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[2]: host})
    assert store.label_segments(db, video_id) == [
        DiarSegment(start_ms=0, end_ms=1_000, local_label="SPEAKER_00"),
        DiarSegment(start_ms=4_000, end_ms=5_000, local_label="SPEAKER_00"),
    ]


# -- the stage -------------------------------------------------------------


def test_every_label_with_enough_speech_gets_a_vector(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    outcome = embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 20_000), embedder=FakeEmbedder()
    )
    assert (outcome.n_embedded, outcome.n_skipped) == (2, 0)
    assert all(row.has_embedding for row in store.label_rows(db, video_id))


def test_a_label_below_the_speech_floor_is_skipped_not_guessed(
    db: Database, tmp_path: Path
) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 500)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, {word_ids[0]: label_id})
    outcome = embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 2_000), embedder=FakeEmbedder()
    )
    assert (outcome.n_embedded, outcome.n_skipped) == (0, 1)
    assert store.get_label(db, label_id).has_embedding is False


def test_the_stored_vector_is_unit_length(db: Database, tmp_path: Path) -> None:
    video_id = a_diarized_video(db)
    embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 20_000), embedder=FakeEmbedder()
    )
    blob = db.conn.execute(
        "SELECT embedding FROM video_speakers ORDER BY id LIMIT 1"
    ).fetchone()[0]
    values = unpack_embedding(bytes(blob))
    assert sum(v * v for v in values) == pytest.approx(1.0)


def test_two_labels_with_the_same_voice_get_the_same_vector(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    embed_video_speakers(
        db,
        video_id,
        wav_path=a_wav(tmp_path / "a.wav", 20_000),
        embedder=ConstantEmbedder(),
    )
    blobs = {
        bytes(row[0])
        for row in db.conn.execute("SELECT embedding FROM video_speakers").fetchall()
    }
    assert len(blobs) == 1


def test_re_embedding_replaces_rather_than_accumulates(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    wav = a_wav(tmp_path / "a.wav", 20_000)
    first_row = "SELECT embedding FROM video_speakers ORDER BY id LIMIT 1"
    embed_video_speakers(db, video_id, wav_path=wav, embedder=FakeEmbedder())
    first = bytes(db.conn.execute(first_row).fetchone()[0])
    embed_video_speakers(db, video_id, wav_path=wav, embedder=ConstantEmbedder())
    second = bytes(db.conn.execute(first_row).fetchone()[0])
    assert first != second


def test_embedder_name_prefers_the_request_then_the_setting_then_the_default(
    db: Database,
) -> None:
    assert embedder_name(db, "") == C.DEFAULT_EMBEDDER
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        (C.SETTINGS_EMBEDDER, "redimnet"),
    )
    assert embedder_name(db, "") == "redimnet"
    assert embedder_name(db, "fake-embedder") == "fake-embedder"


# -- the audio helper and the adapter --------------------------------------


def test_concatenating_windows_produces_their_total_length(tmp_path: Path) -> None:
    source = a_wav(tmp_path / "src.wav", 10_000)
    out = concat_wav_windows(source, tmp_path / "out.wav", [(0, 1_000), (5_000, 7_000)])
    with wave.open(str(out), "rb") as reader:
        assert reader.getnframes() == 16_000 * 3
        assert reader.getframerate() == 16_000
        assert reader.getnchannels() == 1


def test_concatenating_no_windows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError):
        concat_wav_windows(a_wav(tmp_path / "src.wav", 100), tmp_path / "o.wav", [])


def test_the_redimnet_adapter_imports_and_registers_without_torch() -> None:
    assert importlib.util.find_spec("torch") is None, "the dev venv must have no torch"
    from rytp.diarize.embed import ReDimNetEmbedder

    assert EMBEDDERS["redimnet"] is ReDimNetEmbedder
    assert ReDimNetEmbedder.out_of_process is True
    assert ReDimNetEmbedder.requires_hf_token is False


def test_the_redimnet_adapter_sends_the_request_the_child_expects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import embed as embed_module
    from rytp.diarize.embed import ReDimNetEmbedder

    seen: dict[str, object] = {}

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"vector": [1.0, 0.0, 0.0]}

    monkeypatch.setattr(embed_module, "run_child", fake_run_child)
    engine = ReDimNetEmbedder(interpreter="/opt/redimnet/python")
    vector = engine.embed(tmp_path / "a.wav", [(0, 1_000)])

    assert vector == [1.0, 0.0, 0.0]
    assert seen["interpreter"] == "/opt/redimnet/python"
    assert seen["module"] == "rytp.diarize.embed"
    request = seen["request"]
    assert request["windows"] == [[0, 1_000]]
    assert request["model"] == C.REDIMNET_DEFAULT_MODEL
