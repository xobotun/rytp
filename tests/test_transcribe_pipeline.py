"""Transcriber text, aligner timings, measured boundaries, one write, one tier."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import ClassVar

import pytest

import rytp.transcribe.subproc as subproc_module
from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError, Span, normalize_text, stem_text
from rytp.transcribe.pipeline import (
    engine_tag,
    realign_video,
    resolve_word_scale,
    speaker_loss_warning,
    tier_for,
    transcribe_video,
    unalign_video,
)
from tests.fake_engines import (
    FailingAligner,
    FakeAligner,
    FakeTranscriber,
    NoEndTimesTranscriber,
    ScorelessAligner,
    ShortAligner,
    TextOnlyTranscriber,
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_MODULE = "tests.subproc_engine_stub"


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


def _triples(db: Database, video_id: int) -> set[tuple[str, float | None, str | None]]:
    rows = db.conn.execute(
        "SELECT source, align_score, align_scale FROM words WHERE video_id = ?", (video_id,)
    ).fetchall()
    return {(row[0], row[1], row[2]) for row in rows}


def test_resolve_word_scale_follows_plan_1a_row_by_row() -> None:
    # An aligner that reports a score keeps its own scale, refined or not.
    assert resolve_word_scale(0.8, refine=True, aligner_scale=C.ALIGN_SCALE_LOGPROB) == (
        C.ALIGN_SCALE_LOGPROB
    )
    assert resolve_word_scale(0.8, refine=False, aligner_scale=C.ALIGN_SCALE_LOGPROB) == (
        C.ALIGN_SCALE_LOGPROB
    )
    # An aligner that reports none, refine off: `none` — a real aligner ran
    # and had nothing to say.
    assert resolve_word_scale(None, refine=False, aligner_scale=C.ALIGN_SCALE_NONE) == (
        C.ALIGN_SCALE_NONE
    )
    # No aligner, refine on: the energy measure filled it in.
    assert resolve_word_scale(None, refine=True, aligner_scale=None) == C.ALIGN_SCALE_ENERGY
    # No aligner, no refine: unscored.
    assert resolve_word_scale(None, refine=False, aligner_scale=None) is None


def test_no_aligner_with_refine_writes_the_energy_scale(
    db: Database, tmp_path: Path
) -> None:
    # Plan §1a row 3: "No aligner, --refine on" -> the 0-1 energy measure,
    # scale `energy`. BUGS.md entry 13's exact defect, now labelled. Each
    # word's own measured boundary quality differs, so only the source and
    # the scale are uniform — never the exact score.
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        outcome = transcribe_video(
            db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake", refine=True
        )
    assert outcome.align_scale == C.ALIGN_SCALE_ENERGY
    triples = _triples(db, video_id)
    assert {(source, scale) for source, _score, scale in triples} == {
        ("timed", C.ALIGN_SCALE_ENERGY)
    }
    assert all(score is not None and 0.0 <= score <= 1.0 for _source, score, _scale in triples)


def test_no_aligner_no_refine_writes_no_score_and_no_scale(
    db: Database, tmp_path: Path
) -> None:
    # Plan §1a row 4: "No aligner, no refine" -> NULL / NULL.
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        outcome = transcribe_video(
            db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake", refine=False
        )
    assert outcome.align_scale is None
    assert outcome.median_align_score is None
    assert _triples(db, video_id) == {("timed", None, None)}


def test_an_aligner_with_a_score_keeps_its_scale_even_when_refined(
    db: Database, tmp_path: Path
) -> None:
    # Plan §1a: "An aligner that reports a score AND was refined keeps the
    # aligner's score and scale" — the both-ran case. Refine defaults to True.
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
        )
    assert outcome.align_scale == C.ALIGN_SCALE_LOGPROB
    assert outcome.median_align_score == pytest.approx(0.8)
    # FakeAligner always scores exactly 0.8 — an ordinary equality, not an
    # approximation, and so hashable enough to compare as a set.
    assert _triples(db, video_id) == {("aligned", 0.8, C.ALIGN_SCALE_LOGPROB)}


def test_a_scoreless_aligner_with_no_refine_writes_the_none_scale(
    db: Database, tmp_path: Path
) -> None:
    # Plan §1a row 2: "An aligner that reports none (MFA), refine off" ->
    # NULL / `none` — a real aligner ran and had nothing to say. The stored
    # row names `none`, but the run *reported* no score at all, so the
    # outcome's summary (and, per `commands/transcribe.py`, the report's
    # heading) is the same as the no-score case: nothing to name.
    video_id = _make_video(db)
    with registered(FakeTranscriber, ScorelessAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-scoreless-aligner",
            refine=False,
        )
    assert outcome.align_scale is None
    assert outcome.median_align_score is None
    assert _triples(db, video_id) == {("aligned", None, C.ALIGN_SCALE_NONE)}


def test_a_scoreless_aligner_with_refine_writes_the_energy_scale(
    db: Database, tmp_path: Path
) -> None:
    # Not a literal §1a row, but its stated principle applied: the number on
    # the row came from the energy refiner, not the aligner, so the scale
    # names the energy refiner, and it clears entry 26/28's energy floor.
    video_id = _make_video(db)
    with registered(FakeTranscriber, ScorelessAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-scoreless-aligner",
            refine=True,
        )
    assert outcome.align_scale == C.ALIGN_SCALE_ENERGY
    triples = _triples(db, video_id)
    assert {(source, scale) for source, _score, scale in triples} == {
        ("aligned", C.ALIGN_SCALE_ENERGY)
    }
    assert all(score is not None and 0.0 <= score <= 1.0 for _source, score, _scale in triples)


def test_realign_video_writes_the_aligners_scale_too(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake", refine=False)
    with registered(FakeAligner):
        outcome = realign_video(db, video_id, wav_path=wav, aligner="fake-aligner")
    assert outcome.align_scale == C.ALIGN_SCALE_LOGPROB
    assert outcome.align_device == "cpu"
    assert outcome.median_align_score == pytest.approx(0.8)
    assert _triples(db, video_id) == {("aligned", 0.8, C.ALIGN_SCALE_LOGPROB)}


def test_an_invalid_scale_is_never_written(db: Database) -> None:
    # Contracts §3: SQLite cannot `CHECK` a column added by `ALTER TABLE`, so
    # every writer of `align_scale` validates in Python instead.
    from rytp.models import RawWord
    from rytp.transcribe.pipeline import replace_words

    video_id = _make_video(db)
    with pytest.raises(RytpError):
        replace_words(
            db,
            video_id,
            [RawWord(start_ms=0, end_ms=100, text="да")],
            [Span(start_ms=0, end_ms=100, score=0.5)],
            "fake",
            source="timed",
            align_scales=["not-a-real-scale"],
        )
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


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
    """The transcriber timed every word, so an aligner mismatch degrades the
    tier rather than destroying the transcript (the defect fix): the `timed`
    write survives, and the mismatch shows up as a note, not an exception."""
    video_id = _make_video(db)
    with registered(FakeTranscriber, ShortAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-short-aligner",
        )
    assert outcome.source == "timed"
    assert outcome.n_words == 3
    assert any("alignment failed" in note for note in outcome.align_notes)
    rows = db.conn.execute(
        "SELECT source, text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    assert [row[0] for row in rows] == ["timed", "timed", "timed"]
    assert [row[1] for row in rows] == ["один", "два", "три"]


def test_a_text_only_transcriber_still_raises_on_an_aligner_mismatch(
    db: Database, tmp_path: Path
) -> None:
    """When the transcriber left timing entirely to the aligner, there is no
    `timed` write to fall back to — this path still aligns inline and still
    raises outright, exactly as before the defect fix."""
    from rytp.transcribe.pipeline import AlignmentMismatchError

    video_id = _make_video(db)
    with registered(TextOnlyTranscriber, ShortAligner), pytest.raises(AlignmentMismatchError):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake-text-only",
            aligner="fake-short-aligner",
        )
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_a_failed_alignment_leaves_a_complete_timed_transcript_behind(
    db: Database, tmp_path: Path
) -> None:
    """The defect this whole task exists to fix: alignment used to run before
    `replace_words`, so an aligner exception threw away the transcription
    with it. Now the `timed` words are written first and survive; only the
    alignment attempt is lost, and it says so.
    """
    video_id = _make_video(db)
    with registered(FakeTranscriber, FailingAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-failing-aligner",
        )
    # Assert on the stored rows, not on the absence of an exception.
    assert outcome.source == "timed"
    assert outcome.n_words == 3
    rows = db.conn.execute(
        "SELECT ord, source, start_ms, end_ms, text FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert len(rows) == 3
    assert [row[1] for row in rows] == ["timed", "timed", "timed"]
    assert [row[0] for row in rows] == [0, 1, 2]
    assert all(row[3] > row[2] for row in rows)  # end after start: real timings
    assert len(outcome.align_notes) == 1
    assert "simulated alignment failure" in outcome.align_notes[0]
    assert "timed" in outcome.align_notes[0]
    # The transcript is indexable: an index job can be enqueued for it exactly
    # as for any other transcribed video (contracts §5's producer chain).
    utterance_rows = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND source = 'timed'",
        (video_id,),
    ).fetchone()
    assert utterance_rows[0] == 3


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


# -- the persistent worker seam: one spawn, one model load, many chunks -----
#
# The regression this whole task exists to prevent: `realign_video` used to
# call `engine.align(...)` once per chunk, and each call spawned a fresh
# child that reloaded the model from scratch (TODO.md's "pulsing CPU/GPU
# trace" observation). These tests exercise the real out-of-process seam
# (`rytp.transcribe.subproc.run_child`) against a stub engine module, not a
# monkeypatched fake, so they would fail exactly the way the pre-fix code
# did if the reload ever came back.


class _CountingOOPAligner:
    """A real out-of-process aligner, talking to `tests.subproc_engine_stub`.

    Deliberately not a `FakeAligner` subclass — those are in-process and
    would not touch `rytp.transcribe.subproc` at all. `required_module` and
    `required_binary` are both `None` so `check_available` never probes an
    interpreter for a dependency that does not exist; the stub module is
    pure standard library and always "available".
    """

    name = "stub-oop-aligner"
    requires_hf_token = False
    out_of_process = True
    required_module: ClassVar[str | None] = None
    required_binary: ClassVar[str | None] = None
    extra: ClassVar[str | None] = None
    score_scale = C.ALIGN_SCALE_NONE
    device = "auto"

    #: Every call's reported `model_loads` (the stub's own load counter,
    #: read back from the child), across every instance this test
    #: constructs. A class-level list rather than an instance attribute:
    #: `realign_video` builds exactly one instance per call and this test
    #: wants everything that instance saw, but reading a private instance
    #: back out of `realign_video` would mean reaching into the pipeline's
    #: internals instead of asking the registry for what actually happened.
    seen_model_loads: ClassVar[list[int]] = []

    def __init__(self, interpreter: str) -> None:
        self._interpreter = interpreter
        self.notes: list[str] = []

    def align(
        self, audio: Path, words: list[str] | tuple[str, ...], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        from rytp.transcribe.subproc import run_child

        result = run_child(
            interpreter=self._interpreter,
            module=STUB_MODULE,
            request={
                "align": True,
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
            repo_root=REPO_ROOT,
        )
        type(self).seen_model_loads.append(int(result["model_loads"]))
        return [
            Span(start_ms=int(item["start_ms"]), end_ms=int(item["end_ms"]), score=item["score"])
            for item in result["spans"]
        ]


def _make_multi_chunk_wav(tmp_path: Path) -> Path:
    """~23 s of audio as two well-separated speech runs.

    `plan_chunks`'s `target_ms` default is 20 000 ms (`VAD_CHUNK_TARGET_MS`),
    and `detect_speech`'s noise floor is a 10th-percentile estimate over the
    whole clip — a 3 s digitally-silent gap is both long enough for VAD to
    split the two 10 s tones into separate speech segments and a large
    enough share of the clip for the floor estimate to actually notice it,
    which a short, noisy gap (closer to a real recording) was not, in
    practice: two speech segments whose combined span exceeds `target_ms`
    are packed into two chunks rather than one, which is what makes this a
    genuine multi-chunk run rather than a single `align()` call in disguise.
    """
    samples = concat(
        tone(10_000),
        silence(3_000, amp=0.0),
        tone(10_000),
    )
    return write_wav(tmp_path / "multi_chunk.wav", samples)


def _insert_timed_words(db: Database, video_id: int, starts_ms: list[int]) -> None:
    """Directly-inserted `timed` words, one per given start, 300 ms long.

    `realign_video` reads existing words rather than transcribing, so the
    corpus is built by hand here instead of through `transcribe_video` — the
    only thing this test needs control over is *which chunk* each word's
    start lands in.
    """
    with db.transaction():
        for ord_, start_ms in enumerate(starts_ms):
            text = f"слово{ord_}"
            db.conn.execute(
                "INSERT INTO words (video_id, ord, start_ms, end_ms, text, "
                "normalized_text, stem, source, engine) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'timed', 'fake')",
                (
                    video_id,
                    ord_,
                    start_ms,
                    start_ms + 300,
                    text,
                    normalize_text(text),
                    stem_text(normalize_text(text)),
                ),
            )


def test_realign_reuses_one_resident_worker_across_chunks(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CountingOOPAligner.seen_model_loads = []
    spawn_count = 0
    original_spawn = subproc_module._Worker._spawn

    def _counting_spawn(self: subproc_module._Worker) -> None:
        nonlocal spawn_count
        spawn_count += 1
        original_spawn(self)

    monkeypatch.setattr(subproc_module._Worker, "_spawn", _counting_spawn)

    video_id = _make_video(db)
    wav = _make_multi_chunk_wav(tmp_path)
    # One word early in the first tone, one early in the second — the
    # exact VAD-detected chunk boundaries are asserted below, but these two
    # starts land well inside each of the two runs regardless. `realign_video`
    # buckets by start time, so each word lands in a different chunk and
    # `align()` is called at least twice.
    _insert_timed_words(db, video_id, [500, 13_500])

    try:
        with registered(_CountingOOPAligner):
            outcome = realign_video(db, video_id, wav_path=wav, aligner="stub-oop-aligner")
    finally:
        subproc_module.shutdown_workers()

    assert outcome.n_chunks >= 2, "the test setup must actually produce >1 chunk"
    assert len(_CountingOOPAligner.seen_model_loads) >= 2, "expected one align() call per chunk"
    # The regression this test exists to catch: before the persistent worker
    # seam, each align() call span its own child and reloaded the model, so
    # this would read [1, 1] loads from two *different* processes rather
    # than [1, 1] from the *same* one. Asserting only the loads themselves
    # would not tell those apart — the spawn count below does.
    assert all(loads == 1 for loads in _CountingOOPAligner.seen_model_loads)
    assert spawn_count == 1, (
        f"expected exactly one resident worker for {outcome.n_chunks} chunks, "
        f"got {spawn_count} spawns"
    )


def _orig_triples(db: Database, video_id: int) -> list[tuple[int, int | None, int | None]]:
    return [
        (row[0], row[1], row[2])
        for row in db.conn.execute(
            "SELECT ord, orig_start_ms, orig_end_ms FROM words "
            "WHERE video_id = ? ORDER BY ord",
            (video_id,),
        ).fetchall()
    ]


def test_combined_path_captures_the_transcribers_own_timing_as_original(
    db: Database, tmp_path: Path
) -> None:
    """Contracts amendment §7: even with an aligner overwriting `start_ms`/
    `end_ms`, the transcriber's own per-word timing survives in
    `orig_start_ms`/`orig_end_ms`, untouched by alignment or refinement."""
    video_id = _make_video(db)
    script = ((0, 200, "один"), (200, 460, "два"), (460, 700, "три"))
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
            transcriber_kwargs={"script": script},
        )
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, orig_start_ms, orig_end_ms, source FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[5] for row in rows] == ["aligned", "aligned", "aligned"]
    assert [(row[3], row[4]) for row in rows] == [(0, 200), (200, 460), (460, 700)]
    # The aligner's own timing landed on start_ms/end_ms, which is not the
    # transcriber's script above (FakeAligner spreads words evenly instead).
    assert [(row[1], row[2]) for row in rows] != [(0, 200), (200, 460), (460, 700)]


def test_timed_tier_also_captures_its_own_original(db: Database, tmp_path: Path) -> None:
    """No aligner: `start_ms`/`end_ms` are the transcriber's own timing
    already, but refinement can still move them, so the original is worth
    capturing (and captured) even here."""
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        transcribe_video(
            db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake", refine=False
        )
    rows = db.conn.execute(
        "SELECT start_ms, end_ms, orig_start_ms, orig_end_ms, source FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert all(row[4] == "timed" for row in rows)
    # No refine: nothing moved, so original and current agree exactly.
    assert [(row[0], row[1]) for row in rows] == [(row[2], row[3]) for row in rows]


def test_a_text_only_transcriber_leaves_no_original_to_capture(
    db: Database, tmp_path: Path
) -> None:
    """A transcriber that supplies no timing at all (contracts §4) has
    nothing to record as `orig_start_ms`/`orig_end_ms`; both stay NULL."""
    video_id = _make_video(db)
    with registered(TextOnlyTranscriber, FakeAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake-text-only",
            aligner="fake-aligner",
        )
    rows = db.conn.execute(
        "SELECT orig_start_ms, orig_end_ms FROM words WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert rows
    assert all(row[0] is None and row[1] is None for row in rows)


def test_a_hyphenated_word_splits_its_own_timing_too(db: Database, tmp_path: Path) -> None:
    """`_expand_tokens` interpolates the transcriber's own timing for a split
    piece independently of the aligner's span (contracts amendment §7)."""
    video_id = _make_video(db)
    script = ((0, 400, "кто-то"), (460, 700, "ещё"))
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
            transcriber_kwargs={"script": script},
            refine=False,
        )
    rows = db.conn.execute(
        "SELECT text, orig_start_ms, orig_end_ms FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in rows] == ["кто", "то", "ещё"]
    # "кто-то" was 0..400 in the transcriber's own script; the split shares it.
    assert rows[0][1] == 0
    assert rows[0][2] == rows[1][1]
    assert rows[1][2] == 400
    assert (rows[2][1], rows[2][2]) == (460, 700)


def test_realign_does_not_touch_the_recorded_original(db: Database, tmp_path: Path) -> None:
    """The ratchet: a second alignment must not become the new "original",
    or `unalign` would restore to the wrong thing after one round trip."""
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    script = ((0, 200, "один"), (200, 460, "два"), (460, 700, "три"))
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=wav,
            transcriber="fake",
            aligner="fake-aligner",
            transcriber_kwargs={"script": script},
        )
    before = _orig_triples(db, video_id)
    assert before == [(0, 0, 200), (1, 200, 460), (2, 460, 700)]

    class OtherAligner(FakeAligner):
        name = "fake-other-aligner"

        def align(self, audio: Path, words, *, start_ms: int, end_ms: int) -> list[Span]:
            spans = super().align(audio, words, start_ms=start_ms, end_ms=end_ms)
            # A visibly different result from FakeAligner's own spread.
            return [
                Span(start_ms=span.start_ms + 5, end_ms=span.end_ms + 5, score=0.1)
                for span in spans
            ]

    with registered(OtherAligner):
        realign_video(db, video_id, wav_path=wav, aligner="fake-other-aligner")

    after = _orig_triples(db, video_id)
    assert after == before, "a re-alignment must never overwrite the recorded original"
    # Sanity: the second alignment actually did move start_ms/end_ms.
    current = [
        (row[0], row[1], row[2])
        for row in db.conn.execute(
            "SELECT ord, start_ms, end_ms FROM words WHERE video_id = ? ORDER BY ord",
            (video_id,),
        ).fetchall()
    ]
    assert current != before


def test_unalign_restores_the_transcribers_timing_and_downgrades_the_tier(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    script = ((0, 200, "один"), (200, 460, "два"), (460, 700, "три"))
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=wav,
            transcriber="fake",
            aligner="fake-aligner",
            transcriber_kwargs={"script": script},
        )
    result = unalign_video(db, video_id)
    assert result.n_words == 3
    assert result.engine == "fake"
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, align_score, align_scale, source, engine "
        "FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [(row[1], row[2]) for row in rows] == [(0, 200), (200, 460), (460, 700)]
    assert all(row[3] is None and row[4] is None for row in rows)
    assert all(row[5] == "timed" for row in rows)
    assert all(row[6] == "fake" for row in rows)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_unalign_preserves_text_ordinals_and_speaker_labels(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake", aligner="fake-aligner")
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    speaker_row = db.conn.execute(
        "SELECT id FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()
    db.conn.execute(
        "UPDATE words SET video_speaker_id = ? WHERE video_id = ? AND ord = 0",
        (speaker_row[0], video_id),
    )
    db.conn.commit()
    before_text = [
        row[0]
        for row in db.conn.execute(
            "SELECT text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
        ).fetchall()
    ]
    unalign_video(db, video_id)
    after_text = [
        row[0]
        for row in db.conn.execute(
            "SELECT text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
        ).fetchall()
    ]
    assert after_text == before_text
    assert db.conn.execute(
        "SELECT video_speaker_id FROM words WHERE video_id = ? AND ord = 0", (video_id,)
    ).fetchone()[0] == speaker_row[0]
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_unalign_refuses_when_there_are_no_aligned_words(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake")
    with pytest.raises(RytpError, match="no aligned words"):
        unalign_video(db, video_id)


def test_unalign_refuses_when_the_original_was_never_recorded(
    db: Database, tmp_path: Path
) -> None:
    """A row written before migration 16 (or by a text-only transcriber
    under an aligner) has no honest timing to restore to."""
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
        "source, engine) VALUES (?, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'wav2vec2')",
        (video_id,),
    )
    db.conn.commit()
    with pytest.raises(RytpError, match="no recorded original timing"):
        unalign_video(db, video_id)
    # Refusal must not have touched anything.
    row = db.conn.execute(
        "SELECT source, start_ms, end_ms FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert (row[0], row[1], row[2]) == ("aligned", 0, 100)
