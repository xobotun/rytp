"""The queue's registry, read as a whole rather than one part at a time.

Five parts append registrations to the bottom of `rytp/jobs/__init__.py` and
nobody reads the result. Two of the reviews' findings came from exactly that:
a kind whose only producer was never written, and a hardcoded list of kinds
that was stale before there was any code to be stale about.
"""

from __future__ import annotations

import inspect

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.jobs import JOB_HANDLERS, JOB_KINDS, Readiness
from rytp.jobs import queue as Q
from rytp.jobs.worker import WorkerOptions, run_worker
from tests.consistency import PRODUCERS, enqueued_kinds_in_source, load_registries
from tests.fakes import make_video

COMMANDS, _ = load_registries()
KINDS = sorted(JOB_KINDS)

#: contracts §5 "Job handlers" assigns every kind to a part. Kept as a set so
#: a kind that appears without a contract entry is visible.
CONTRACTED_KINDS = {
    "download",
    "captions",
    "extract_wav",
    "caption_words",
    "transcribe",
    "align",
    "fingerprint",
    "index",
    "diarize",
    "render",
}


def test_the_registry_is_exactly_the_contracted_set() -> None:
    """contracts §5 lists the kinds by owner; `caption_words` was added to it
    by agreement between Parts 2 and 3 and belongs here too."""
    assert set(JOB_KINDS) == CONTRACTED_KINDS


def test_the_flat_handler_view_cannot_drift_from_the_registry() -> None:
    """contracts §5 requires `JOB_HANDLERS`; Part 2 writes it only through
    `register_job_kind`, which is what makes drift impossible. Pinned so a
    later part cannot start writing it directly."""
    assert set(JOB_HANDLERS) == set(JOB_KINDS)
    for name, kind in JOB_KINDS.items():
        assert JOB_HANDLERS[name] is kind.handler


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_has_a_handler_and_a_readiness_predicate(kind: str) -> None:
    entry = JOB_KINDS[kind]
    assert callable(entry.handler), kind
    assert callable(entry.readiness), kind
    assert entry.summary.strip(), kind


@pytest.mark.parametrize("kind", KINDS)
def test_a_handler_takes_the_database_the_target_and_the_payload(kind: str) -> None:
    """contracts §5 fixes the shape: `(Database, int, dict) -> str | None`."""
    params = list(inspect.signature(JOB_KINDS[kind].handler).parameters)
    assert len(params) == 3, f"{kind}: {params}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_readiness_predicate_never_sees_the_payload(kind: str) -> None:
    """design §5: readiness is "a statement about the world". A predicate that
    changed its mind based on how a job was enqueued would take the
    self-healing property with it."""
    params = list(inspect.signature(JOB_KINDS[kind].readiness).parameters)
    assert len(params) == 2, f"{kind}: {params}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_predicate_answers_blocked_for_a_target_that_does_not_exist(
    db: Database, kind: str
) -> None:
    """The worker calls every predicate against whatever ids the table holds,
    including one whose row was removed. None of them may raise."""
    assert JOB_KINDS[kind].readiness(db, 999_999) is Readiness.BLOCKED


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_runs_on_a_declared_pool(kind: str) -> None:
    assert JOB_KINDS[kind].pool in C.POOLS, f"{kind}: {JOB_KINDS[kind].pool}"


def test_the_gpu_pool_is_the_expensive_one_and_holds_only_what_belongs_there() -> None:
    """design §5's table. A cheap kind on the gpu pool serialises behind
    transcription for no reason."""
    gpu = {name for name, kind in JOB_KINDS.items() if kind.pool == "gpu"}
    assert gpu == {"transcribe", "align", "diarize"}


def test_target_kind_namespaces_the_only_kind_that_needs_it() -> None:
    """contracts §3: `render` targets a `renders` row, everything else a video.
    The namespace is what stops render 7 re-evaluating video 7's jobs."""
    by_target = {name: kind.target_kind for name, kind in JOB_KINDS.items()}
    assert by_target.pop("render") == "render"
    assert set(by_target.values()) == {"video"}


def test_only_the_one_shots_refuse_to_be_reopened() -> None:
    """`reconcile` turns a done job back into work when its output vanishes.
    That is right for anything derived from the corpus and wrong for work a
    person explicitly asked for once — re-timing, and hours of GPU time.

    `render` is deliberately *not* in this set. Part 2's original prose said
    a render "must not silently re-run on the next worker start", but Part 6
    raised that against the `renders` table it ended up adding: once a render
    has a real row, its predicate is a genuine question about the world
    (SATISFIED while `output.mp4` is on disk, READY once it is pruned), so
    "delete the output and it re-runs" is the same self-healing behaviour as
    a pruned WAV, not a silent re-run of user-directed work — see
    `docs/superpowers/plans/2026-09-21-part6-render.md` around line 4329.
    Part 6 owns `render` and made the later, reasoned call; `align` and
    `diarize` are the two kinds where "already has an answer" does not mean
    "the answer you just asked for", which is what `reopenable=False` is
    actually for.
    """
    not_reopenable = {name for name, kind in JOB_KINDS.items() if not kind.reopenable}
    assert not_reopenable == {"align", "diarize"}


# --- producers -------------------------------------------------------


def test_the_producer_table_covers_the_registry_exactly() -> None:
    """The staleness guard. A new kind must be classified here, and this is the
    assertion that makes forgetting impossible."""
    assert set(PRODUCERS) == set(JOB_KINDS), (
        f"unclassified kinds: {sorted(set(JOB_KINDS) - set(PRODUCERS))}; "
        f"kinds that no longer exist: {sorted(set(PRODUCERS) - set(JOB_KINDS))}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_every_registered_kind_has_at_least_one_producer(kind: str) -> None:
    """The review's finding, in one line: a kind with a handler and no producer
    is work the worker will never receive."""
    producers = PRODUCERS[kind]
    assert producers, f"nothing can create a {kind} job"
    for command in producers:
        assert command in COMMANDS, f"{kind}'s producer {command!r} is not registered"


def test_every_kind_enqueued_in_the_source_tree_is_registered() -> None:
    """A typo in an `enqueue(db, "fingerpint", …)` is a job nobody ever runs."""
    found = enqueued_kinds_in_source()
    unknown = {kind: files for kind, files in found.items() if kind not in JOB_KINDS}
    assert not unknown, f"enqueued but never registered: {unknown}"


@pytest.mark.parametrize(
    "chain",
    [C.INGEST_CHAIN_REMOTE, C.INGEST_CHAIN_LOCAL, C.INGEST_CHAIN_TRANSCRIBE],
)
def test_every_ingest_chain_names_registered_kinds(chain: tuple[str, ...]) -> None:
    assert set(chain) <= set(JOB_KINDS), sorted(set(chain) - set(JOB_KINDS))


def test_diarization_is_never_reached_by_accident() -> None:
    """design §6: "Diarization is opt-in per video… at full corpus it is the
    single largest GPU cost". This is the robust form of the assertion — it
    survives Part 3 adding two kinds to the chain, which the version that
    pinned the chain's exact contents did not."""
    for chain in (C.INGEST_CHAIN_REMOTE, C.INGEST_CHAIN_LOCAL, C.INGEST_CHAIN_TRANSCRIBE):
        assert "diarize" not in chain, chain


def test_promoting_a_video_does_not_make_its_captions_runnable_again(
    db: Database,
) -> None:
    """design §6: "Promoting a video from tier 1 to tier 2 deletes its caption
    words. One tier per video at a time."

    `reconcile` reopens a done job whose output has gone, which is what makes
    the queue self-healing — and a caption-words predicate that only looks for
    caption-tier rows sees exactly that after a promotion. The result would be
    caption words reappearing underneath an aligned transcript, which contracts
    §3 says cannot coexist. The predicate must read "this video has words",
    not "this video has caption words".
    """
    from rytp.db.queries import insert_asset

    video_id = make_video(db)
    insert_asset(db, video_id=video_id, role="captions", path="captions.json3")
    db.conn.execute(
        "INSERT INTO words"
        " (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        "  source, engine)"
        " VALUES (?, 0, 0, 100, 'раз', 'раз', 'раз', 'aligned', 'fake')",
        (video_id,),
    )
    assert JOB_KINDS["caption_words"].readiness(db, video_id) is Readiness.SATISFIED


def test_transcription_is_opt_in_too() -> None:
    """design §6: tier 2 is "for videos you actually want to cut from", and
    design §13 prices the corpus at 130-200 GPU hours."""
    assert "transcribe" not in C.INGEST_CHAIN_REMOTE
    assert "transcribe" not in C.INGEST_CHAIN_LOCAL
    assert "align" not in C.INGEST_CHAIN_REMOTE


# --- notes survive the worker (review finding 3.2) -------------------


def test_a_handler_note_reaches_the_job_row(db: Database) -> None:
    """contracts §5: "A handler may return a short note… The worker stores it
    in `jobs.note`". This is the mechanism; the next test is the case it was
    added for."""
    from tests.fakes import temp_job_kind

    def noted(db_: Database, target_id: int, payload: dict[str, object]) -> str:
        return "discarded 4 speaker labels"

    with temp_job_kind("t_note", pool="cpu", handler=noted):
        video_id = make_video(db)
        job_id = Q.enqueue(db, "t_note", video_id)
        run_worker(db, WorkerOptions(pools=("cpu",), once=True))
        job = Q.get_job(db, job_id)
    assert job.state == "done"
    row = db.conn.execute("SELECT note FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["note"] == "discarded 4 speaker labels"


def test_re_transcribing_a_mapped_video_leaves_the_warning_on_the_job(
    db: Database, data_dir: object
) -> None:
    """Review finding 3.2, on the path it was reported for.

    Re-transcribing deletes a video's speaker mapping. Part 3 computes the
    warning and shows it to whoever runs `transcribe run` by hand — and the
    worker, which is the bulk path, is the one that will do this across
    hundreds of videos with nobody reading stdout.

    The pre-existing word is caption-tier, not `timed`: `transcribe_readiness`
    (rytp/transcribe/readiness.py) reports SATISFIED once a video has any
    *non-caption* word, and `queue.enqueue` writes a SATISFIED job straight in
    as ``done`` (rytp/jobs/queue.py:_state_for) — the worker never claims it,
    so the handler never runs. A `timed` or `aligned` pre-existing word here
    would make this test pass vacuously (the row stays untouched, "done" with
    no note, for the wrong reason). A caption-tier word is exactly the tier
    `transcribe run` promotes, so it makes the job READY and the handler is
    what has to run and produce the note.
    """
    from rytp.audio.extract import wav_path
    from tests.consistency import write_test_wav
    from tests.fake_engines import FakeAligner, FakeTranscriber, registered

    video_id = make_video(db)
    write_test_wav(wav_path(video_id))
    db.conn.execute(
        "INSERT INTO words"
        " (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        "  source, engine)"
        " VALUES (?, 0, 0, 100, 'раз', 'раз', 'раз', 'caption', 'old')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (?, 'SPEAKER_00', 'pyannote')",
        (video_id,),
    )

    with registered(FakeTranscriber, FakeAligner):
        job_id = Q.enqueue(
            db,
            "transcribe",
            video_id,
            payload={"transcriber": "fake", "aligner": "fake-aligner"},
        )
        run_worker(db, WorkerOptions(pools=("gpu",), once=True))

    job = Q.get_job(db, job_id)
    assert job.state == "done", job.last_error
    note = db.conn.execute(
        "SELECT note FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["note"]
    assert note, (
        "re-transcribing discarded a speaker mapping and the worker recorded "
        "nothing; see rytp/jobs/__init__.py::_run_transcribe"
    )
    assert "speaker" in note.lower()
