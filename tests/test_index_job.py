"""The `index` job kind (contracts §5, design §5)."""

from __future__ import annotations

import subprocess
import sys

from rytp.db import Database
from rytp.index.utterances import index_readiness, index_video
from rytp.jobs import JOB_HANDLERS, JOB_KINDS, Readiness, resolve_job_kind
from rytp.jobs import queue as Q
from tests.test_index_utterances import make_speaker, make_video, seed_words


def test_the_index_kind_is_registered_on_the_cpu_pool() -> None:
    kind = resolve_job_kind("index")
    assert kind.pool == "cpu"
    assert kind.target_kind == "video"
    assert kind.reopenable is True
    assert kind.summary


def test_the_flat_handler_map_agrees_with_the_kind() -> None:
    """contracts §5 requires JOB_HANDLERS; register_job_kind writes both."""
    assert set(JOB_HANDLERS) == set(JOB_KINDS)
    assert JOB_HANDLERS["index"] is resolve_job_kind("index").handler


def test_a_video_with_no_words_is_blocked(db: Database) -> None:
    video_id = make_video(db)
    assert index_readiness(db, video_id) is Readiness.BLOCKED


def test_a_video_with_words_and_no_utterances_is_ready(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    assert index_readiness(db, video_id) is Readiness.READY


def test_an_indexed_video_is_satisfied(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    assert index_readiness(db, video_id) is Readiness.SATISFIED


def test_extending_the_words_makes_it_ready_again(db: Database) -> None:
    """Part 3 deletes utterances with the words (contracts §4); even if it
    did not, the ordinal range check would catch the change."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
        " stem, source, engine) VALUES (?, 2, 720, 1100, 'друзья', 'друзья',"
        " 'друз', 'aligned', 'fake')",
        (video_id,),
    )
    assert index_readiness(db, video_id) is Readiness.READY


def test_the_state_part_seven_leaves_behind_is_ready(db: Database) -> None:
    """Part 7 assigns speakers, deletes the utterances and enqueues this job.

    Rebuilding has to both re-fire and carry the speakers through, or
    design §7's speaker filter — which searches utterances, not words —
    goes on returning nothing.
    """
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    speaker = make_speaker(db, video_id, "SPEAKER_00")
    db.conn.execute(
        "UPDATE words SET video_speaker_id = ? WHERE video_id = ?", (speaker, video_id)
    )
    db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))

    assert index_readiness(db, video_id) is Readiness.READY
    JOB_HANDLERS["index"](db, video_id, {})
    assert index_readiness(db, video_id) is Readiness.SATISFIED
    assert db.conn.execute(
        "SELECT video_speaker_id FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()["video_speaker_id"] == speaker


def test_diarizing_after_indexing_makes_it_ready_again(db: Database) -> None:
    """Part 7 sets words.video_speaker_id in place, which does not trip the
    contracts §4 invariant, so readiness has to notice on its own."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    assert index_readiness(db, video_id) is Readiness.SATISFIED
    speaker = make_speaker(db, video_id, "SPEAKER_00")
    db.conn.execute(
        "UPDATE words SET video_speaker_id = ? WHERE video_id = ?", (speaker, video_id)
    )
    assert index_readiness(db, video_id) is Readiness.READY


def test_the_handler_builds_the_utterances(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    JOB_HANDLERS["index"](db, video_id, {})
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_the_handler_ignores_an_unexpected_payload(db: Database) -> None:
    """Handlers receive the payload; this one has no inputs beyond the id."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300)])
    JOB_HANDLERS["index"](db, video_id, {"nonsense": True})
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_the_handler_returns_none(db: Database) -> None:
    """contracts §5: Callable[[Database, int, dict], None]."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300)])
    assert JOB_HANDLERS["index"](db, video_id, {}) is None


def test_enqueueing_index_lands_pending_when_the_words_are_there(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300)])
    job_id = Q.enqueue(db, "index", video_id)
    row = db.conn.execute("SELECT state, pool FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert (row["state"], row["pool"]) == ("pending", "cpu")


def test_enqueueing_index_lands_blocked_without_words(db: Database) -> None:
    video_id = make_video(db)
    job_id = Q.enqueue(db, "index", video_id)
    assert db.conn.execute(
        "SELECT state FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["state"] == "blocked"


def test_importing_the_index_package_first_does_not_break_the_registry() -> None:
    """rytp.index.utterances is not a submodule of rytp.jobs, so Python does
    not import rytp.jobs first for it. The Readiness import therefore has to
    stay inside the function body."""
    probe = (
        "import rytp.index.utterances as u;"
        "import rytp.jobs as j;"
        "assert 'index' in j.JOB_KINDS;"
        "assert j.JOB_KINDS['index'].readiness is u.index_readiness;"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_importing_jobs_first_also_works() -> None:
    probe = (
        "import rytp.jobs as j;"
        "import rytp.index.utterances as u;"
        "assert j.JOB_KINDS['index'].readiness is u.index_readiness;"
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
