"""Part 3's four job kinds: readiness from the world, inputs from the payload."""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.audio.extract import wav_path
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.jobs import JOB_HANDLERS, JOB_KINDS, Readiness
from rytp.transcribe.base import EngineUnavailable
from rytp.transcribe.readiness import (
    align_readiness,
    caption_words_readiness,
    fingerprint_readiness,
    transcribe_readiness,
)
from tests.fake_engines import (
    FakeAligner,
    FakeTranscriber,
    MissingModuleTranscriber,
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav

KINDS = ("caption_words", "transcribe", "align", "fingerprint")


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


def _place_wav(video_id: int) -> Path:
    target = wav_path(video_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(
        target,
        concat(tone(200), silence(120, amp=0.001, seed=81), tone(200),
               silence(120, amp=0.001, seed=82), tone(200)),
    )


def _add_aligned_word(db: Database, video_id: int) -> None:
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
        "stem, source, engine) VALUES (?, 0, 0, 100, 'один', 'один', 'один', "
        "'aligned', 'fake+energy')",
        (video_id,),
    )
    db.conn.commit()


def test_all_three_kinds_are_registered_in_both_views() -> None:
    for kind in KINDS:
        assert kind in JOB_KINDS
        assert JOB_HANDLERS[kind] is JOB_KINDS[kind].handler


def test_the_pools_match_the_resource_each_kind_uses() -> None:
    assert JOB_KINDS["transcribe"].pool == "gpu"
    assert JOB_KINDS["align"].pool == "gpu"
    assert JOB_KINDS["fingerprint"].pool == "cpu"
    assert JOB_KINDS["caption_words"].pool == "cpu"


def test_only_the_one_shot_re_timing_is_not_reopenable() -> None:
    assert JOB_KINDS["align"].reopenable is False
    assert JOB_KINDS["transcribe"].reopenable is True
    assert JOB_KINDS["fingerprint"].reopenable is True
    assert JOB_KINDS["caption_words"].reopenable is True


def test_every_kind_targets_a_video_id() -> None:
    assert {JOB_KINDS[kind].target_kind for kind in KINDS} == {"video"}


def test_caption_words_waits_for_the_asset_and_stops_once_there_are_words(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    assert caption_words_readiness(db, video_id) is Readiness.BLOCKED
    captions = tmp_path / "captions.json3"
    captions.write_text("{}", encoding="utf-8")
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(captions), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    assert caption_words_readiness(db, video_id) is Readiness.READY
    _add_aligned_word(db, video_id)
    assert caption_words_readiness(db, video_id) is Readiness.SATISFIED


def test_the_caption_words_handler_writes_caption_rows(
    db: Database, tmp_path: Path
) -> None:
    import json

    video_id = _make_video(db)
    captions = tmp_path / "captions.json3"
    captions.write_text(
        json.dumps({"events": [{"tStartMs": 0, "segs": [{"utf8": "да", "tOffsetMs": 0}]}]}),
        encoding="utf-8",
    )
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(captions), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    JOB_HANDLERS["caption_words"](db, video_id, {})
    row = db.conn.execute(
        "SELECT text, source, end_ms FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert (row[0], row[1], row[2]) == ("да", "caption", None)


def test_transcribe_is_blocked_without_a_cached_wav(db: Database) -> None:
    assert transcribe_readiness(db, _make_video(db)) is Readiness.BLOCKED


def test_transcribe_is_ready_once_the_wav_is_there(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    assert transcribe_readiness(db, video_id) is Readiness.READY


def test_transcribe_is_satisfied_once_the_words_are_aligned(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    _add_aligned_word(db, video_id)
    assert transcribe_readiness(db, video_id) is Readiness.SATISFIED


def test_align_is_blocked_until_there_are_words_to_retime(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    assert align_readiness(db, video_id) is Readiness.BLOCKED
    _add_aligned_word(db, video_id)
    assert align_readiness(db, video_id) is Readiness.READY


def test_fingerprint_is_satisfied_once_the_row_exists(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    assert fingerprint_readiness(db, video_id) is Readiness.READY
    db.conn.execute(
        "INSERT INTO video_acoustics (video_id, noise_floor_db, computed_at) "
        "VALUES (?, -60.0, '2026-09-21T00:00:00+00:00')",
        (video_id,),
    )
    db.conn.commit()
    assert fingerprint_readiness(db, video_id) is Readiness.SATISFIED


def test_the_transcribe_handler_takes_its_engines_from_the_payload(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    with registered(FakeTranscriber, FakeAligner):
        JOB_HANDLERS["transcribe"](
            db, video_id, {"transcriber": "fake", "aligner": "fake-aligner"}
        )
    rows = db.conn.execute(
        "SELECT source, engine FROM words WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert len(rows) == 3
    assert {row[1] for row in rows} == {"fake+fake-aligner+energy"}


def test_a_payload_with_no_transcriber_uses_the_setting_and_says_so(
    db: Database,
) -> None:
    # contracts §3: a run that used the default must announce it. A job's
    # announcement is the note the worker stores on the row, which is the
    # only place a hand-made job can say what it picked.
    video_id = _make_video(db)
    _place_wav(video_id)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "fake")
    with registered(FakeTranscriber):
        note = JOB_HANDLERS["transcribe"](db, video_id, {})
    assert note is not None
    assert "fake" in note
    assert C.SETTING_DEFAULT_TRANSCRIBER in note


def test_a_payload_that_names_its_transcriber_leaves_no_note(db: Database) -> None:
    # Nothing was chosen on the job's behalf, so a note would be noise —
    # and `rytp jobs stats` counts notes.
    video_id = _make_video(db)
    _place_wav(video_id)
    with registered(FakeTranscriber):
        assert JOB_HANDLERS["transcribe"](db, video_id, {"transcriber": "fake"}) is None


def test_an_uninstalled_default_fails_the_job_rather_than_substituting(
    db: Database,
) -> None:
    # contracts §3: never fall back to another engine. `words.engine` records
    # what actually ran, which only helps if nothing lies about what it used.
    video_id = _make_video(db)
    _place_wav(video_id)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "fake-missing")
    with registered(FakeTranscriber, MissingModuleTranscriber), pytest.raises(
        EngineUnavailable
    ) as excinfo:
        JOB_HANDLERS["transcribe"](db, video_id, {})
    assert "pip install rytp[" in str(excinfo.value)
    # `fake` was registered and ready beside it, and wrote nothing.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_transcribing_asks_for_the_index_to_be_rebuilt(db: Database) -> None:
    # Part 2's chain stops at extract_wav, so this is the only thing that
    # makes a freshly transcribed video searchable without a hand-run command.
    video_id = _make_video(db)
    _place_wav(video_id)
    with registered(FakeTranscriber, FakeAligner):
        JOB_HANDLERS["transcribe"](db, video_id, {"transcriber": "fake"})
    queued = db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'index' AND target_id = ?", (video_id,)
    ).fetchone()[0]
    # Part 4 registers the `index` kind; before it lands there is nothing to
    # enqueue and the call is a no-op, which is why this tolerates both.
    assert queued == (1 if "index" in JOB_KINDS else 0)


def test_re_aligning_asks_for_the_index_to_be_rebuilt(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    with registered(FakeTranscriber, FakeAligner):
        JOB_HANDLERS["transcribe"](db, video_id, {"transcriber": "fake"})
        db.conn.execute("DELETE FROM jobs WHERE kind = 'index'")
        db.conn.commit()
        JOB_HANDLERS["align"](db, video_id, {"aligner": "fake-aligner"})
    queued = db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'index' AND target_id = ?", (video_id,)
    ).fetchone()[0]
    assert queued == (1 if "index" in JOB_KINDS else 0)


def test_the_align_handler_refuses_a_payload_with_no_aligner(db: Database) -> None:
    from rytp.models import RytpError

    video_id = _make_video(db)
    _place_wav(video_id)
    _add_aligned_word(db, video_id)
    with pytest.raises(RytpError) as excinfo:
        JOB_HANDLERS["align"](db, video_id, {})
    assert "aligner" in str(excinfo.value)


def test_the_fingerprint_handler_writes_the_row(db: Database) -> None:
    video_id = _make_video(db)
    _place_wav(video_id)
    JOB_HANDLERS["fingerprint"](db, video_id, {})
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1
