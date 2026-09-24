"""Diarization end to end, with a fake engine and no audio decoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import DiarizeError
from rytp.diarize.pipeline import diarize_video
from rytp.jobs import Readiness
from rytp.models import DiarSegment
from tests.fake_speaker_engines import FakeDiarizer
from tests.fakes import make_video, temp_job_kind
from tests.test_speakers_labels import add_words

WAV = Path("does-not-need-to-exist.wav")


def a_video_with_words(db: Database) -> int:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    return video_id


def test_one_row_per_label_and_every_word_stamped(db: Database) -> None:
    video_id = a_video_with_words(db)
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert outcome.n_labels == 2
    assert outcome.n_words_labelled == 3
    assert [row.local_label for row in store.label_rows(db, video_id)] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert [row.n_words for row in store.label_rows(db, video_id)] == [2, 1]


def test_a_word_outside_every_segment_is_counted_not_dropped(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 500), (60_000, 60_500)])
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert (outcome.n_words_labelled, outcome.n_words_unlabelled) == (1, 1)


def test_a_diarizer_that_finds_nothing_is_an_error_not_an_empty_success(
    db: Database,
) -> None:
    video_id = a_video_with_words(db)
    with pytest.raises(DiarizeError, match="no speech"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer(script=()))


def test_diarizing_a_video_with_no_words_is_an_error(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(DiarizeError, match="no words"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())


def test_rediarizing_keeps_a_link_whose_label_came_back(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(
        db, store.find_label(db, video_id, "SPEAKER_00").video_speaker_id, speaker_id
    )
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"


def test_rediarizing_drops_a_label_the_new_run_did_not_produce(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    single = FakeDiarizer(script=((0, 3_000, "SPEAKER_00"),))
    diarize_video(db, video_id, wav_path=WAV, diarizer=single)
    assert [row.local_label for row in store.label_rows(db, video_id)] == ["SPEAKER_00"]


def test_diarizing_deletes_the_videos_utterances(db: Database) -> None:
    video_id = a_video_with_words(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 900, 0, 0, 'x', 'x', 'x')",
        (video_id,),
    )
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        == 0
    )


def test_diarizing_enqueues_the_index_job_when_that_kind_exists(db: Database) -> None:
    from rytp.jobs import queue as Q

    video_id = a_video_with_words(db)
    with temp_job_kind(
        "index", "cpu", lambda db_, t, p: None, readiness=lambda db_, t: Readiness.READY
    ):
        outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
        assert [job.kind for job in Q.list_jobs(db)] == ["index"]
    assert outcome.reindexed is True


def test_diarizing_without_an_index_kind_still_succeeds(db: Database) -> None:
    # In this codebase Part 4's `index` kind is always registered, so the
    # "not registered" branch is exercised by removing it for the duration
    # of this test rather than by running before Part 4 exists.
    from rytp.jobs import JOB_HANDLERS, JOB_KINDS

    video_id = a_video_with_words(db)
    saved_kind = JOB_KINDS.pop("index", None)
    saved_handler = JOB_HANDLERS.pop("index", None)
    try:
        outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
        assert outcome.reindexed is False
    finally:
        if saved_kind is not None:
            JOB_KINDS["index"] = saved_kind
        if saved_handler is not None:
            JOB_HANDLERS["index"] = saved_handler


def test_a_diarizer_emitting_a_backwards_segment_is_refused(db: Database) -> None:
    video_id = a_video_with_words(db)
    bad = FakeDiarizer(script=((900, 100, "SPEAKER_00"),))
    with pytest.raises(DiarizeError, match="start"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=bad)


def test_the_outcome_names_the_engine(db: Database) -> None:
    video_id = a_video_with_words(db)
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert outcome.diarizer == "fake-diarizer"
    assert "fake-diarizer" in outcome.message()


def test_every_label_records_the_diarizer_that_produced_it(db: Database) -> None:
    # contracts §3: video_speakers.engine is NOT NULL, for the same reason
    # words.engine is — design §11 forbids assuming one engine.
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert {row.engine for row in store.label_rows(db, video_id)} == {"fake-diarizer"}


def test_the_null_diarizer_records_itself_too(db: Database, tmp_path: Path) -> None:
    # Not an exception: "one voice, unmodelled" is a claim about the audio
    # like any other, and a later re-run with pyannote must be tellable from
    # it without guessing.
    import wave

    from rytp.diarize.none import NullDiarizer

    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16_000 * 5)
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=audio, diarizer=NullDiarizer())
    assert [row.engine for row in store.label_rows(db, video_id)] == ["none"]


def test_re_diarizing_with_another_engine_updates_the_provenance(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())

    class OtherDiarizer(FakeDiarizer):
        name = "fake-other"

    diarize_video(db, video_id, wav_path=WAV, diarizer=OtherDiarizer())
    assert {row.engine for row in store.label_rows(db, video_id)} == {"fake-other"}


def test_segments_are_accepted_in_any_order(db: Database) -> None:
    video_id = a_video_with_words(db)
    shuffled = FakeDiarizer(
        script=(
            (2_000, 3_000, "SPEAKER_00"),
            (0, 1_000, "SPEAKER_00"),
            (1_000, 2_000, "SPEAKER_01"),
        )
    )
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=shuffled)
    assert outcome.n_words_labelled == 3


def test_the_contract_type_field_is_local_label() -> None:
    # A regression guard: the field is `local_label`, not the old `speaker`.
    assert DiarSegment(start_ms=0, end_ms=1, local_label="SPEAKER_00").local_label
