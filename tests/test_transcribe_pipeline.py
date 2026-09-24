"""Transcriber text, aligner timings, measured boundaries, one write, one tier."""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe.pipeline import (
    AlignmentMismatchError,
    engine_tag,
    speaker_loss_warning,
    tier_for,
    transcribe_video,
)
from tests.fake_engines import (
    FakeAligner,
    FakeTranscriber,
    NoEndTimesTranscriber,
    ScorelessAligner,
    ShortAligner,
    TextOnlyTranscriber,
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav


def _make_video(db: Database) -> int:
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, title, external_id, url, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "youtube",
            "video",
            "Sample",
            "VIDEO_A",
            "https://example.invalid/v/VIDEO_A",
            "2026-09-21T00:00:00+00:00",
        ),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def _make_wav(tmp_path: Path) -> Path:
    samples = concat(
        tone(200),
        silence(120, amp=0.001, seed=11),
        tone(200),
        silence(120, amp=0.001, seed=12),
        tone(200),
    )
    return write_wav(tmp_path / "audio.wav", samples)


def test_engine_tag_records_every_stage() -> None:
    assert engine_tag("gigaam", "mfa", True) == "gigaam+mfa+energy"
    assert engine_tag("whisper", None, False) == "whisper"


def test_only_a_run_with_an_aligner_produces_a_cuttable_tier() -> None:
    assert tier_for("mfa") == "aligned"
    assert tier_for(None) == "timed"
    assert tier_for("") == "timed"


def test_without_an_aligner_the_words_are_timed_and_not_cuttable(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §3: the transcriber's own timestamps are 78.7% zero-gap on
    # real data. Energy refinement improves them but cannot place a boundary
    # that was never there, so these rows are searchable, not cuttable.
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        outcome = transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
    assert outcome.n_words == 3
    assert outcome.source == "timed"
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in rows] == [0, 1, 2]
    assert all(row[2] is not None and row[2] > row[1] for row in rows)
    assert {row[4] for row in rows} == {"timed"}
    assert {row[5] for row in rows} == {"fake+energy"}


def test_with_an_aligner_the_words_are_aligned_and_cuttable(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
        )
    assert outcome.source == "aligned"
    sources = {
        row[0]
        for row in db.conn.execute(
            "SELECT source FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    }
    assert sources == {"aligned"}


def test_replacing_a_transcript_reports_the_speaker_labels_it_destroyed(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §4 deletes them and Part 7's diarize is reopenable=False, so
    # nothing brings them back. Saying so is the least this can do.
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine) "
            "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
            (video_id,),
        )
        db.conn.commit()
        outcome = transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
    assert outcome.speakers_lost == 1
    warning = speaker_loss_warning(video_id, outcome.speakers_lost)
    assert warning is not None
    assert "speakers diarize" in warning
    assert speaker_loss_warning(video_id, 0) is None


def test_adjacent_words_share_a_measured_boundary(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake")
    rows = db.conn.execute(
        "SELECT start_ms, end_ms FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    for left, right in itertools.pairwise(rows):
        assert left[1] == right[0]


def test_an_aligner_supplies_the_timings_and_the_score(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
        )
    assert outcome.engine == "fake+fake-aligner+energy"
    scores = [
        row[0]
        for row in db.conn.execute(
            "SELECT align_score FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    ]
    assert scores == [pytest.approx(0.8)] * 3


def test_a_scoreless_aligner_gets_a_measured_boundary_quality(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, ScorelessAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-scoreless-aligner",
        )
    scores = [
        row[0]
        for row in db.conn.execute(
            "SELECT align_score FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    ]
    assert all(score is not None and 0.0 <= score <= 1.0 for score in scores)


def test_promotion_deletes_caption_words_utterances_and_speaker_labels(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, stem, "
        "source, engine) VALUES (?, 0, 0, 'старое', 'старое', 'стар', 'caption', 'captions:json3')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 100, 0, 0, 'x', 'x', 'x')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    db.conn.commit()
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND source = 'caption'", (video_id,)
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_a_hyphenated_word_becomes_two_rows_with_a_shared_boundary(
    db: Database, tmp_path: Path
) -> None:
    # The invariant Parts 4 and 5 rely on: one token per row, no spaces in
    # normalized_text. The split point is interpolated, then measured by
    # refinement like every other boundary.
    video_id = _make_video(db)
    script = ((0, 400, "кто-то"), (460, 700, "ещё"))
    with registered(FakeTranscriber):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": script},
        )
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, normalized_text FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[3] for row in rows] == ["кто", "то", "ещё"]
    assert [row[4] for row in rows] == ["кто", "то", "еще"]
    assert all(" " not in row[4] for row in rows)
    assert rows[0][2] == rows[1][1]
    assert rows[0][1] < rows[0][2] < rows[1][2]


def test_punctuation_only_tokens_never_get_an_ordinal(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    script = ((0, 200, "один"), (200, 260, "—"), (260, 460, "два"))
    with registered(FakeTranscriber):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": script},
        )
    texts = [
        row[0]
        for row in db.conn.execute(
            "SELECT text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
        ).fetchall()
    ]
    assert texts == ["один", "два"]


def test_a_transcriber_without_end_times_demands_an_aligner(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(NoEndTimesTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake-no-ends")
    assert "does not time every word" in str(excinfo.value)


def test_a_text_only_transcriber_demands_an_aligner(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(TextOnlyTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake-text-only")
    assert "does not time every word" in str(excinfo.value)


def test_a_text_only_transcriber_is_fully_usable_with_an_aligner(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §4 allows RawWord to carry text alone; the aligner supplies
    # every boundary and refinement measures them.
    video_id = _make_video(db)
    with registered(TextOnlyTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake-text-only",
            aligner="fake-aligner",
        )
    assert outcome.n_words == 3
    rows = db.conn.execute(
        "SELECT start_ms, end_ms, text FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[2] for row in rows] == ["один", "два", "три"]
    assert all(row[1] > row[0] for row in rows)


def test_an_aligner_returning_the_wrong_count_is_refused(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, ShortAligner), pytest.raises(AlignmentMismatchError):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-short-aligner",
        )


def test_a_transcriber_that_finds_nothing_is_reported(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": ()},
        )
    assert "no words" in str(excinfo.value)
