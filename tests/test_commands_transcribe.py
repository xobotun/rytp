"""The transcribe command group: registered once, used by both surfaces."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import rytp.commands.transcribe  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.commands import COMMANDS, CommandResult, resolve
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.transcribe.base import EngineUnavailable
from tests.fake_engines import (
    FakeAligner,
    FakeTranscriber,
    MissingModuleTranscriber,
    ScorelessAligner,
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav

NAMES = (
    "transcribe.captions",
    "transcribe.run",
    "transcribe.align",
    "transcribe.unalign",
    "transcribe.compare",
    "transcribe.engines",
    "transcribe.fingerprint",
)


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


def _wav(tmp_path: Path) -> Path:
    return write_wav(
        tmp_path / "audio.wav",
        concat(tone(200), silence(120, amp=0.001, seed=61), tone(200),
               silence(120, amp=0.001, seed=62), tone(200)),
    )


def test_every_command_is_registered_in_the_transcribe_group() -> None:
    for name in NAMES:
        assert name in COMMANDS
        assert COMMANDS[name].group == "transcribe"
        assert COMMANDS[name].summary


def test_the_long_running_commands_are_flagged() -> None:
    assert resolve("transcribe.run").long_running is True
    assert resolve("transcribe.align").long_running is True
    assert resolve("transcribe.compare").long_running is True
    assert resolve("transcribe.fingerprint").long_running is True
    assert resolve("transcribe.captions").long_running is False
    assert resolve("transcribe.engines").long_running is False
    # A single UPDATE plus a DELETE, cheap enough to run inline — not a job
    # kind, so it never appears in test_consistency_jobs.py's roster.
    assert resolve("transcribe.unalign").long_running is False


def test_captions_handler_reports_how_many_words_it_wrote(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    path = tmp_path / "captions.json3"
    path.write_text(
        json.dumps({"events": [{"tStartMs": 0, "segs": [{"utf8": "да", "tOffsetMs": 0}]}]}),
        encoding="utf-8",
    )
    result = resolve("transcribe.captions").handler(db, video=str(video_id), path=path)
    assert isinstance(result, CommandResult)
    assert "1" in (result.message or "")


def test_run_handler_transcribes_with_the_named_engines(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        result = resolve("transcribe.run").handler(
            db,
            video=str(video_id),
            transcriber="fake",
            aligner="fake-aligner",
            language="ru",
            refine=True,
            wav=_wav(tmp_path),
        )
    assert result.rows[0][1] == "3"
    assert result.rows[0][3] == "aligned"
    assert result.rows[0][4] == "fake+fake-aligner+energy"


def test_the_default_transcriber_is_announced_when_no_flag_named_one(
    db: Database, tmp_path: Path
) -> None:
    # contracts §3: "set a default and inform" was chosen over "refuse until
    # configured", so this message is the only thing standing between the
    # owner and 35-90 GPU-hours spent on an engine nobody picked.
    video_id = _make_video(db)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "fake")
    with registered(FakeTranscriber):
        result = resolve("transcribe.run").handler(
            db, video=str(video_id), wav=_wav(tmp_path)
        )
    message = result.message or ""
    assert "fake" in message
    assert C.SETTING_DEFAULT_TRANSCRIBER in message
    assert result.rows[0][4].startswith("fake")


def test_an_explicitly_named_transcriber_is_not_announced(
    db: Database, tmp_path: Path
) -> None:
    # Nothing was chosen on the owner's behalf, so there is nothing to report.
    video_id = _make_video(db)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "fake")
    with registered(FakeTranscriber):
        result = resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", wav=_wav(tmp_path)
        )
    assert C.SETTING_DEFAULT_TRANSCRIBER not in (result.message or "")


def test_an_uninstalled_default_fails_and_substitutes_nothing(
    db: Database, tmp_path: Path
) -> None:
    # contracts §3: never fall back to another engine. A silent substitution
    # is the same bug wearing a different hat, and worse — the corpus would
    # hold rows from two engines with nothing recording which, and
    # `words.engine` only helps if nothing lies about what it used.
    video_id = _make_video(db)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "fake-missing")
    with registered(FakeTranscriber, MissingModuleTranscriber), pytest.raises(
        EngineUnavailable
    ) as excinfo:
        resolve("transcribe.run").handler(db, video=str(video_id), wav=_wav(tmp_path))
    assert "pip install rytp[" in str(excinfo.value)
    # The zero rows are what proves nothing was substituted: `fake` was
    # registered and ready right beside it, and no word came from it.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_align_handler_retimes_existing_words(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    wav = _wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        resolve("transcribe.run").handler(db, video=str(video_id), transcriber="fake", wav=wav)
        result = resolve("transcribe.align").handler(
            db, video=str(video_id), aligner="fake-aligner", wav=wav
        )
    assert result.rows[0][3] == "aligned"
    assert result.rows[0][4] == "fake+fake-aligner+energy"


def test_unalign_handler_restores_the_timed_tier(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    wav = _wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", aligner="fake-aligner", wav=wav
        )
        result = resolve("transcribe.unalign").handler(db, video=str(video_id))
    assert result.rows[0][2] == "timed"
    assert result.rows[0][3] == "fake"
    assert "not cuttable" in (result.message or "")
    sources = {
        row[0]
        for row in db.conn.execute(
            "SELECT source FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    }
    assert sources == {"timed"}


def test_unalign_handler_refuses_a_video_with_no_aligned_words(
    db: Database, tmp_path: Path
) -> None:
    from rytp.models import RytpError

    video_id = _make_video(db)
    with registered(FakeTranscriber):
        resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", wav=_wav(tmp_path)
        )
    with pytest.raises(RytpError):
        resolve("transcribe.unalign").handler(db, video=str(video_id))


def test_engines_handler_lists_what_is_registered(db: Database) -> None:
    with registered(FakeTranscriber):
        result = resolve("transcribe.engines").handler(db)
    assert result.columns[0] == "name"
    assert any(row[0] == "fake" for row in result.rows)


def test_compare_handler_writes_a_report_when_asked(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    out = tmp_path / "report.md"
    with registered(FakeTranscriber, FakeAligner):
        result = resolve("transcribe.compare").handler(
            db,
            video=str(video_id),
            transcribers="fake",
            aligners="fake-aligner",
            start_ms=0,
            end_ms=0,
            out=out,
            wav=_wav(tmp_path),
        )
    assert out.exists()
    assert "## Boundaries" in out.read_text(encoding="utf-8")
    assert len(result.rows) == 2


def test_compare_handler_needs_at_least_one_transcriber(
    db: Database, tmp_path: Path
) -> None:
    from rytp.models import RytpError

    video_id = _make_video(db)
    with pytest.raises(RytpError):
        resolve("transcribe.compare").handler(
            db, video=str(video_id), transcribers="", wav=_wav(tmp_path)
        )


def test_fingerprint_handler_stores_the_row(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    result = resolve("transcribe.fingerprint").handler(
        db, video=str(video_id), wav=_wav(tmp_path)
    )
    assert result.columns[0] == "f0 mean"
    stored = db.conn.execute(
        "SELECT COUNT(*) FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert stored == 1


def test_the_score_heading_names_the_energy_scale_for_an_unaligned_run(
    db: Database, tmp_path: Path
) -> None:
    # BUGS.md entry 13: "median align score 1.00" on a `timed` run read as a
    # perfect alignment. The heading now says what actually produced it.
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        result = resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", refine=True, wav=_wav(tmp_path)
        )
    assert result.columns[-1] == "median energy score"
    assert result.rows[0][3] == "timed"


def test_the_score_heading_names_the_logprob_scale_for_an_aligned_run(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        result = resolve("transcribe.run").handler(
            db,
            video=str(video_id),
            transcriber="fake",
            aligner="fake-aligner",
            wav=_wav(tmp_path),
        )
    assert result.columns[-1] == "median logprob score"
    # The two headings must never be printed under one name (plan §1a).
    assert result.columns[-1] != "median energy score"


def test_the_score_heading_and_cell_are_blank_with_no_aligner_and_no_refine(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        result = resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", refine=False, wav=_wav(tmp_path)
        )
    assert result.columns[-1] == "median align score"
    assert result.rows[0][-1] == "-"


def test_align_handler_reports_the_logprob_heading_and_the_aligner_device(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    wav = _wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        resolve("transcribe.run").handler(db, video=str(video_id), transcriber="fake", wav=wav)
        result = resolve("transcribe.align").handler(
            db, video=str(video_id), aligner="fake-aligner", wav=wav
        )
    assert result.columns[-1] == "median logprob score"
    assert "aligner device: cpu" in (result.message or "")


def test_a_scoreless_aligner_run_leaves_the_heading_generic(
    db: Database, tmp_path: Path
) -> None:
    # ScorelessAligner never scores anything, and refine is off, so no word
    # ever gets a score at all — the `none` row of plan §1a's table.
    video_id = _make_video(db)
    with registered(FakeTranscriber, ScorelessAligner):
        result = resolve("transcribe.run").handler(
            db,
            video=str(video_id),
            transcriber="fake",
            aligner="fake-scoreless-aligner",
            refine=False,
            wav=_wav(tmp_path),
        )
    assert result.columns[-1] == "median align score"
    assert result.rows[0][-1] == "-"


def test_a_missing_wav_is_a_domain_error_not_a_traceback(db: Database) -> None:
    from rytp.models import RytpError

    video_id = _make_video(db)
    with registered(FakeTranscriber), pytest.raises(RytpError) as excinfo:
        resolve("transcribe.run").handler(
            db, video=str(video_id), transcriber="fake", wav=Path("nope.wav")
        )
    assert "nope.wav" in str(excinfo.value)
