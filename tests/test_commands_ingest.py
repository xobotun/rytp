"""Tests for the Part 2 commands (contracts §5, design §5)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.commands import ingest as I
from rytp.commands import resolve
from rytp.db import Database
from rytp.db.queries import asset_for, insert_asset, set_setting
from rytp.jobs import JOB_KINDS, Readiness
from rytp.jobs import queue as Q
from rytp.models import InvalidInputError, NotFoundError
from tests.fakes import (
    CHANNEL_ONE_URL,
    FakeYtDlpRunner,
    make_video,
    temp_job_kind,
    touch,
)


def _ingest(db: Database, **kwargs: object):
    """Call the ingest handler with the defaults its Params declare."""
    args: dict[str, object] = {
        "video": None, "channel_id": 0, "pending": False, "transcribe": False,
        "transcriber": "",
        "limit": C.INGEST_DEFAULT_LIMIT, "priority": 0, "dry_run": False,
    }
    args.update(kwargs)
    return resolve("ingest").handler(db, **args)


def test_ingest_enqueues_every_registered_kind_of_the_chain(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video=str(vid))
    queued = {j.kind for j in Q.list_jobs(db)}
    # Part 3 has now registered caption_words and fingerprint, so the chain
    # is Part 2's full remote chain with no change here — this is exactly
    # the growth the original comment predicted.
    assert queued == {k for k in C.INGEST_CHAIN_REMOTE if k in JOB_KINDS}
    assert queued == {"download", "captions", "caption_words", "extract_wav", "fingerprint"}


def test_ingest_enqueues_a_later_parts_kind_once_it_is_registered(
    db: Database, tmp_path: Path
) -> None:
    with temp_job_kind(
        "fingerprint", "cpu", lambda db_, t, p: None,
        readiness=lambda db_, t: Readiness.BLOCKED,
    ):
        vid = make_video(db)
        _ingest(db, video=str(vid))
        assert Q.list_jobs(db, kind="fingerprint") != []


def test_ingest_reports_the_state_each_kind_started_in(db: Database) -> None:
    result = _ingest(db, video=str(make_video(db)))
    by_kind = {row[0]: dict(zip(result.columns[1:], row[1:], strict=True))
               for row in result.rows}
    assert by_kind["download"]["pending"] == "1"
    assert by_kind["captions"]["pending"] == "1"
    # No audio yet, so there is nothing to decode.
    assert by_kind["extract_wav"]["blocked"] == "1"


def test_ingest_is_idempotent(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video=str(vid))
    _ingest(db, video=str(vid))
    assert len(Q.list_jobs(db)) == len(C.INGEST_CHAIN_REMOTE)


def test_transcribe_flag_adds_transcription_but_not_alignment_by_default(
    db: Database,
) -> None:
    # design §6: tier 2 is opt-in per video, so --transcribe is off by
    # default. And with no default_aligner set, contracts §3 says enqueue no
    # align job at all rather than one that fails five times.
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None), temp_job_kind(
        "align", "gpu", lambda db_, t, p: None, reopenable=False
    ):
        a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
        b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
        _ingest(db, video=str(a))
        assert Q.list_jobs(db, kind="transcribe") == []
        result = _ingest(db, video=str(b), transcribe=True)
        assert [j.target_id for j in Q.list_jobs(db, kind="transcribe")] == [b]
        assert Q.list_jobs(db, kind="align") == []
        assert "timed" in (result.message or "")


def test_a_default_aligner_stamps_the_payload_every_align_job_needs(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Part 3's align handler raises without payload["aligner"], so an align
    # job enqueued without one dies after JOB_MAX_ATTEMPTS retries.
    monkeypatch.setattr(I, "_reject_unknown_aligner", lambda name: None)
    set_setting(db, C.SETTING_DEFAULT_ALIGNER, "mfa")
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None), temp_job_kind(
        "align", "gpu", lambda db_, t, p: None, reopenable=False
    ):
        vid = make_video(db)
        _ingest(db, video=str(vid), transcribe=True)
        jobs = Q.list_jobs(db, kind="align")
        assert [j.target_id for j in jobs] == [vid]
        assert jobs[0].payload == {"aligner": "mfa"}


def test_a_misspelled_aligner_is_rejected_before_anything_is_enqueued(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # contracts §3: rejected at enqueue time, not after five failed retries.
    def explode(name: str) -> None:
        raise InvalidInputError(f"{name} is not a registered aligner")

    monkeypatch.setattr(I, "_reject_unknown_aligner", explode)
    set_setting(db, C.SETTING_DEFAULT_ALIGNER, "mfaa")
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None), temp_job_kind(
        "align", "gpu", lambda db_, t, p: None, reopenable=False
    ):
        make_video(db)
        with pytest.raises(InvalidInputError, match="mfaa"):
            _ingest(db, transcribe=True)
    # Nothing at all was enqueued: the check runs before the loop.
    assert Q.list_jobs(db) == []


def test_the_aligner_check_is_skipped_when_part_three_is_absent(
    db: Database,
) -> None:
    # No rytp.transcribe.registry yet, so there is nothing to validate
    # against and no align kind registered either.
    set_setting(db, C.SETTING_DEFAULT_ALIGNER, "mfa")
    I._reject_unknown_aligner("mfa")   # must not raise


def test_every_transcribe_job_carries_the_engine_that_will_run_it(
    db: Database,
) -> None:
    # contracts §3: a transcribe job with no engine named is a job that
    # resolves one at run time, where nothing announces the choice.
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        vid = make_video(db)
        _ingest(db, video=str(vid), transcribe=True)
        jobs = Q.list_jobs(db, kind="transcribe")
        assert [j.target_id for j in jobs] == [vid]
        assert jobs[0].payload == {"transcriber": C.DEFAULT_TRANSCRIBER_FALLBACK}


def test_the_setting_overrides_the_shipped_fallback(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(I, "_reject_unknown_transcriber", lambda name: None)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "whisper")
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        vid = make_video(db)
        _ingest(db, video=str(vid), transcribe=True)
        assert Q.list_jobs(db, kind="transcribe")[0].payload == {
            "transcriber": "whisper"
        }


def test_a_transcriber_taken_from_the_setting_is_announced(db: Database) -> None:
    # The whole mechanism keeping the choice visible instead of accidental:
    # the owner chose "set a default and inform" over "refuse until
    # configured", so this message is what makes 35-90 GPU-hours deliberate.
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        make_video(db)
        result = _ingest(db, transcribe=True)
    message = result.message or ""
    assert C.DEFAULT_TRANSCRIBER_FALLBACK in message
    assert C.SETTING_DEFAULT_TRANSCRIBER in message


def test_an_explicitly_named_transcriber_is_not_announced(db: Database) -> None:
    # Nothing was chosen on the owner's behalf, so there is nothing to report.
    named = C.DEFAULT_TRANSCRIBER_FALLBACK
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        make_video(db)
        result = _ingest(db, transcribe=True, transcriber=named)
    assert C.SETTING_DEFAULT_TRANSCRIBER not in (result.message or "")
    assert Q.list_jobs(db, kind="transcribe")[0].payload == {"transcriber": named}


def test_a_misspelled_transcriber_is_rejected_before_anything_is_enqueued(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # contracts §3: rejected at enqueue time, not after five failed retries.
    def explode(name: str) -> None:
        raise InvalidInputError(f"{name} is not a registered transcriber")

    monkeypatch.setattr(I, "_reject_unknown_transcriber", explode)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "gigaamm")
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        make_video(db)
        with pytest.raises(InvalidInputError, match="gigaamm"):
            _ingest(db, transcribe=True)
    # Resolved once, before the loop, so 1,600 jobs are not stamped with it.
    assert Q.list_jobs(db) == []


def test_an_empty_setting_falls_back_rather_than_disabling_transcription(
    db: Database,
) -> None:
    # Empty is meaningful for default_aligner and meaningless here: a
    # transcribe job with no engine does nothing, so there is no unset path.
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "   ")
    with temp_job_kind("transcribe", "gpu", lambda db_, t, p: None):
        vid = make_video(db)
        _ingest(db, video=str(vid), transcribe=True)
        assert Q.list_jobs(db, kind="transcribe")[0].payload == {
            "transcriber": C.DEFAULT_TRANSCRIBER_FALLBACK
        }


def test_the_transcriber_check_is_skipped_when_part_three_is_absent(
    db: Database,
) -> None:
    # Same shape as the aligner's: no rytp.transcribe.registry yet, so there
    # is nothing to validate against and no transcribe kind registered either.
    I._reject_unknown_transcriber(C.DEFAULT_TRANSCRIBER_FALLBACK)   # must not raise


def test_bulk_ingest_covers_a_whole_channel(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO channels (id, url, title) VALUES (1, ?, 'CHANNEL_ONE')",
        (CHANNEL_ONE_URL,),
    )
    ours = [
        make_video(db, channel_id=1, external_id=f"VIDEO_{i}",
                   url=f"https://example.invalid/{i}")
        for i in range(3)
    ]
    make_video(db, channel_id=None, external_id="VIDEO_X",
               url="https://example.invalid/x")
    result = _ingest(db, channel_id=1)
    assert "3 video(s)" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="download")} == set(ours)


def test_bulk_ingest_skips_videos_it_has_already_done(db: Database) -> None:
    a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    _ingest(db, video=str(a))
    b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    result = _ingest(db, pending=True)
    assert "1 video(s)" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="download")} == {a, b}


def test_bulk_ingest_honours_the_limit(db: Database) -> None:
    for i in range(5):
        make_video(db, external_id=f"VIDEO_{i}", url=f"https://example.invalid/{i}")
    _ingest(db, limit=2)
    assert len(Q.list_jobs(db, kind="download")) == 2


def test_bulk_ingest_dry_run_enqueues_nothing(db: Database) -> None:
    make_video(db)
    result = _ingest(db, dry_run=True)
    assert result.columns == ("video", "source")
    assert len(result.rows) == 1
    assert Q.list_jobs(db) == []


def test_bulk_ingest_survives_one_unreadable_local_file(
    db: Database, tmp_path: Path
) -> None:
    good = make_video(db, source=C.LOCAL_SOURCE,
                      external_id=str(touch(tmp_path / "ok.mkv")), url=None)
    make_video(db, source=C.LOCAL_SOURCE,
               external_id=str(tmp_path / "gone.mkv"), url=None)
    result = _ingest(db)
    assert "1 skipped" in (result.message or "")
    assert {j.target_id for j in Q.list_jobs(db, kind="extract_wav")} == {good}


def test_ingest_of_a_local_file_registers_it_and_skips_the_downloads(
    db: Database, tmp_path: Path
) -> None:
    clip = touch(tmp_path / "clip.mkv")
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(clip), url=None)
    result = _ingest(db, video=str(vid))
    assert {row[0] for row in result.rows} == set(C.INGEST_CHAIN_LOCAL)
    assert asset_for(db, vid, "container") is not None
    assert Q.list_jobs(db, kind="download") == []


def test_ingest_of_an_unknown_video_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        _ingest(db, video=str(4242))


def test_ingest_passes_priority_through(db: Database) -> None:
    _ingest(db, video=str(make_video(db)), priority=7)
    assert {j.priority for j in Q.list_jobs(db)} == {7}


def _writing_ffmpeg():
    """An ffmpeg replacement that writes the output file it is asked for.

    A real, minimal 16 kHz mono WAV rather than placeholder bytes — Part 3's
    fingerprint job actually reads this file (`rytp.audio.energy.read_wav_mono`),
    where earlier fixtures only ever asserted the path existed.
    """
    import subprocess
    import wave

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(C.WAV_SAMPLE_RATE_HZ)
            writer.writeframes(b"\x00\x00" * 1600)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run


def _fake_the_world(monkeypatch: pytest.MonkeyPatch, db: Database | None = None) -> None:
    """Replace yt-dlp and ffmpeg where the stage code looks them up.

    ``acquire_media`` and ``acquire_captions`` each did
    ``from ... import RealYtDlpRunner``, so the name to replace lives in
    *their* module, not in ``rytp.acquire.ytdlp``.
    """
    from rytp.audio import extract as E
    from rytp.db.queries import set_setting

    monkeypatch.setattr("rytp.acquire.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr("rytp.acquire.captions.RealYtDlpRunner", FakeYtDlpRunner)
    monkeypatch.setattr(E, "_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(E, "_run_ffmpeg", _writing_ffmpeg())
    if db is not None:
        for key in (C.SETTING_DOWNLOAD_DELAY_MIN, C.SETTING_DOWNLOAD_DELAY_MAX,
                    C.SETTING_DOWNLOAD_FILE_DELAY):
            set_setting(db, key, "0")


def test_fetch_video_runs_the_whole_chain_inline(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch)

    vid = make_video(db)
    result = resolve("fetch-video").handler(db, video=str(vid), captions=True)
    assert asset_for(db, vid, "audio") is not None
    assert asset_for(db, vid, "video") is not None
    assert asset_for(db, vid, "captions") is not None
    assert E.wav_path(vid).exists()
    assert result.message and "audio.m4a" in result.message


def test_fetch_video_can_skip_captions(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_the_world(monkeypatch)

    vid = make_video(db)
    resolve("fetch-video").handler(db, video=str(vid), captions=False)
    assert asset_for(db, vid, "captions") is None


def test_worker_once_reports_what_it_did(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch)
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    Q.enqueue(db, "extract_wav", vid)

    result = resolve("worker").handler(db, pool="cpu", once=True, max_jobs=0)
    assert result.message and "1 done" in result.message
    assert E.wav_path(vid).exists()


def test_every_part_two_command_is_registered() -> None:
    for name in (
        "ingest", "fetch-video", "worker",
        "jobs.list", "jobs.stats", "jobs.retry",
        "queue.pause", "queue.resume", "cache.prune",
    ):
        assert resolve(name).summary


def test_jobs_list_renders_rows_of_strings(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video=str(vid))
    result = resolve("jobs.list").handler(db, state="", pool="", kind="", limit=50)
    assert result.columns == (
        "id", "kind", "target", "state", "pool", "attempts", "not_before",
        "error", "note",
    )
    assert len(result.rows) == len(C.INGEST_CHAIN_REMOTE)
    assert all(isinstance(cell, str) for row in result.rows for cell in row)


def test_jobs_list_filters_by_state(db: Database) -> None:
    vid = make_video(db)
    _ingest(db, video=str(vid))
    result = resolve("jobs.list").handler(
        db, state="blocked", pool="", kind="", limit=50
    )
    # extract_wav has no audio yet, caption_words has no captions asset yet,
    # and fingerprint has no cached WAV yet — all three start blocked.
    assert {row[1] for row in result.rows} == {"extract_wav", "caption_words", "fingerprint"}


def test_jobs_stats_shows_throttling_not_a_mystery_stall(db: Database) -> None:
    from datetime import UTC, datetime, timedelta

    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    soon = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    Q.defer(db, job_id, not_before=soon, error="HTTP Error 429")
    result = resolve("jobs.stats").handler(db)
    flat = {row[0]: row[1] for row in result.rows}
    assert flat["throttled"] == "1"
    assert flat["next attempt"] == soon
    assert flat["with notes"] == "0"
    assert flat["paused"] == "no"


def test_jobs_retry_revives_failed_jobs(db: Database) -> None:
    vid = make_video(db)
    job_id = Q.enqueue(db, "download", vid)
    Q.fail(db, job_id, error="boom")
    result = resolve("jobs.retry").handler(db, job_id=0, kind="", state="failed")
    assert "1" in (result.message or "")
    assert Q.get_job(db, job_id).state == "pending"


def test_queue_pause_and_resume_round_trip(db: Database) -> None:
    resolve("queue.pause").handler(db)
    assert Q.is_paused(db) is True
    resolve("queue.resume").handler(db)
    assert Q.is_paused(db) is False


def test_cache_prune_frees_files_and_reopens_the_extract_jobs(
    db: Database, tmp_path: Path
) -> None:
    from rytp.audio import extract as E

    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    touch(E.wav_path(vid), b"\x00" * 200)
    job_id = Q.enqueue(db, "extract_wav", vid)
    assert Q.get_job(db, job_id).state == "done"

    result = resolve("cache.prune").handler(db, video=None, dry_run=False)
    assert not E.wav_path(vid).exists()
    assert "200" in (result.message or "")
    # This is the self-healing loop, end to end.
    assert Q.get_job(db, job_id).state == "pending"


def test_cache_prune_dry_run_changes_nothing(db: Database, tmp_path: Path) -> None:
    from rytp.audio import extract as E

    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    touch(E.wav_path(vid))
    job_id = Q.enqueue(db, "extract_wav", vid)
    resolve("cache.prune").handler(db, video=None, dry_run=True)
    assert E.wav_path(vid).exists()
    assert Q.get_job(db, job_id).state == "done"


def test_ingest_then_worker_once_acquires_everything(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline claim of Part 2, end to end, with yt-dlp and ffmpeg faked.

    It is the one test that exercises the blocked -> pending transition:
    `ingest` parks extract_wav as blocked because there is no audio yet, and
    only the finished download job can turn it into work.
    """
    from rytp.audio import extract as E

    _fake_the_world(monkeypatch, db)
    vid = make_video(db)

    _ingest(db, video=str(vid))
    assert Q.get_job(db, Q.list_jobs(db, kind="extract_wav")[0].id).state == "blocked"

    result = resolve("worker").handler(db, pool="all", once=True, max_jobs=0)

    assert asset_for(db, vid, "audio") is not None
    assert asset_for(db, vid, "video") is not None
    assert asset_for(db, vid, "captions") is not None
    assert E.wav_path(vid).exists()
    assert {j.state for j in Q.list_jobs(db)} == {"done"}
    # Part 4 registers "index"; caption_words now enqueues it for real
    # instead of it being a silent no-op, so this run also claims and
    # finishes that job. >= rather than == because a later part may add
    # another follow-on job the same way.
    done = int(re.search(r"(\d+) done", result.message or "")[1])
    assert done >= len(C.INGEST_CHAIN_REMOTE)
