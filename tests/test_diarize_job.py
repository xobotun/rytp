"""The diarize job kind: gpu pool, derived readiness, never auto-reopened."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.readiness import diarize_readiness
from rytp.jobs import JOB_HANDLERS, JOB_KINDS, Readiness
from rytp.jobs import queue as Q
from rytp.models import RytpError
from tests.fake_speaker_engines import FakeDiarizer, registered
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def cache_a_wav(video_id: int) -> Path:
    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")          # readiness only checks that it exists
    return path


def test_the_kind_is_registered_on_the_gpu_pool() -> None:
    assert JOB_KINDS["diarize"].pool == "gpu"
    assert JOB_KINDS["diarize"].target_kind == "video"
    assert "diarize" in JOB_HANDLERS


def test_the_kind_is_not_reopenable() -> None:
    # design §6: diarization is opt-in per video and the largest GPU cost in
    # the project. A reconcile that re-queued it across the corpus would be
    # a serious bug.
    assert JOB_KINDS["diarize"].reopenable is False


def test_readiness_is_blocked_for_an_unknown_video(db: Database, data_dir: Path) -> None:
    assert diarize_readiness(db, 4242) is Readiness.BLOCKED


def test_readiness_is_blocked_without_aligned_words(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    cache_a_wav(video_id)
    assert diarize_readiness(db, video_id) is Readiness.BLOCKED


def test_caption_tier_words_do_not_make_it_ready(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    cache_a_wav(video_id)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, stem, "
        "source, engine) VALUES (?, 0, 0, 'x', 'x', 'x', 'caption', 'captions:json3')",
        (video_id,),
    )
    assert diarize_readiness(db, video_id) is Readiness.BLOCKED


def test_readiness_is_blocked_without_the_cached_wav(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    assert diarize_readiness(db, video_id) is Readiness.BLOCKED


def test_readiness_is_ready_with_words_and_a_wav(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    cache_a_wav(video_id)
    assert diarize_readiness(db, video_id) is Readiness.READY


def test_readiness_is_satisfied_once_the_labels_exist(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    cache_a_wav(video_id)
    add_label(db, video_id, "SPEAKER_00")
    assert diarize_readiness(db, video_id) is Readiness.SATISFIED


def test_a_full_reconcile_never_re_queues_a_finished_diarization(
    db: Database, data_dir: Path
) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    cache_a_wav(video_id)
    job_id = Q.enqueue(db, "diarize", video_id)
    with registered(FakeDiarizer):
        JOB_HANDLERS["diarize"](db, video_id, {"diarizer": "fake-diarizer"})
    Q.finish(db, job_id)
    assert store.label_rows(db, video_id), "the job really did produce labels"

    # Re-transcribing deletes those labels (contracts §4), so the predicate
    # answers READY again — and a sweep must still leave the job alone.
    store.clear_video_speakers(db, video_id)
    assert diarize_readiness(db, video_id) is Readiness.READY
    assert Q.reconcile(db) == 0
    assert Q.get_job(db, job_id).state == "done"
    # Asking again explicitly still runs it.
    Q.enqueue(db, "diarize", video_id)
    assert Q.get_job(db, job_id).state == "pending"


def test_ingest_does_not_enqueue_diarization(db: Database, data_dir: Path) -> None:
    from rytp.commands import resolve

    video_id = make_video(db)
    resolve("ingest").handler(db, video=str(video_id), priority=0)
    assert "diarize" not in {job.kind for job in Q.list_jobs(db)}


def test_the_handler_runs_the_named_diarizer(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)
    with registered(FakeDiarizer):
        JOB_HANDLERS["diarize"](db, video_id, {"diarizer": "fake-diarizer"})
    assert len(store.label_rows(db, video_id)) == 2


def test_the_handler_defaults_to_the_setting_then_the_constant(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp import constants as C
    from rytp.diarize import pipeline

    seen: list[str] = []

    def spy(db_: Database, name: str) -> FakeDiarizer:
        seen.append(name)
        return FakeDiarizer()

    monkeypatch.setattr(pipeline, "load_diarizer", spy)
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)

    pipeline.run_diarize_job(db, video_id, {})
    assert seen[-1] == C.DEFAULT_DIARIZER

    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        (C.SETTINGS_DIARIZER, "fake-diarizer"),
    )
    pipeline.run_diarize_job(db, video_id, {})
    assert seen[-1] == "fake-diarizer"


def test_a_missing_cached_wav_names_the_step_that_makes_it(
    db: Database, data_dir: Path
) -> None:
    from rytp.diarize.pipeline import wav_for

    video_id = make_video(db)
    with pytest.raises(RytpError, match="extract"):
        wav_for(db, video_id)


def test_importing_the_jobs_package_pulls_in_nothing_heavy() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.jobs; "
            "assert 'torch' not in sys.modules and 'pyannote' not in sys.modules; "
            "print('clean')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
