"""Progress across the engine seam (Task 8b, BUGS.md entry 10).

Two things are asserted here that no other owned test file checks:

* the transcription and alignment pipelines report the chunk counter that
  already existed but nothing asked for, once per chunk;
* the out-of-process seam (`run_child`) streams a child's stderr line by
  line to the progress sink instead of buffering it whole, while the error
  path still carries the same stderr tail it always did.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rytp import constants as C
from rytp import progress
from rytp.db import Database
from rytp.transcribe.base import EngineSubprocessError
from rytp.transcribe.pipeline import realign_video, transcribe_video
from rytp.transcribe.subproc import run_child, shutdown_workers
from tests.fake_engines import FakeAligner, FakeTranscriber, registered
from tests.synth_audio import concat, silence, tone, write_wav

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = "tests.subproc_engine_stub"


@pytest.fixture(autouse=True)
def _no_leftover_workers() -> None:
    """Close every resident worker after each test — see the same fixture
    in `test_transcribe_subproc.py` for why this file needs it too (it
    calls the real `run_child` seam directly, not through a fake)."""
    yield
    shutdown_workers()


class _Recorder:
    """Collects every `report()` call made while installed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int | None, str]] = []

    def __call__(
        self, stage: str, done: int | None, total: int | None, detail: str
    ) -> None:
        self.calls.append((stage, done, total, detail))


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


# -- the chunk counter (BUGS.md entry 10) -----------------------------------


def test_transcribe_video_reports_one_chunk_at_a_time(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    recorder = _Recorder()
    with registered(FakeTranscriber), progress.install(recorder):
        outcome = transcribe_video(
            db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake"
        )
    transcribe_calls = [call for call in recorder.calls if call[0] == "transcribe"]
    assert transcribe_calls, "transcribe_video must report at least once"
    assert [call[1] for call in transcribe_calls] == list(
        range(1, outcome.n_chunks + 1)
    )
    assert all(call[2] == outcome.n_chunks for call in transcribe_calls)


def test_realign_video_reports_one_chunk_at_a_time(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake", refine=False)
    recorder = _Recorder()
    with registered(FakeAligner), progress.install(recorder):
        outcome = realign_video(db, video_id, wav_path=wav, aligner="fake-aligner")
    align_calls = [call for call in recorder.calls if call[0] == "align"]
    assert align_calls, "realign_video must report at least once"
    assert [call[1] for call in align_calls] == list(range(1, outcome.n_chunks + 1))
    assert all(call[2] == outcome.n_chunks for call in align_calls)


def test_nothing_is_emitted_when_no_sink_is_installed(db: Database, tmp_path: Path) -> None:
    # `progress.install(None)` silences report() entirely (plan §1c); the
    # pipeline must not care either way — it just calls report().
    video_id = _make_video(db)
    with registered(FakeTranscriber), progress.install(None):
        outcome = transcribe_video(
            db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake"
        )
    assert outcome.n_words == 3  # ran to completion, no exception from a silenced sink


# -- streaming a child's stderr (BUGS.md entry 10) --------------------------


def test_child_stderr_is_forwarded_to_the_progress_sink_line_by_line() -> None:
    recorder = _Recorder()
    with progress.install(recorder):
        result = run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"stderr_lines": ["downloading 10%", "downloading 100%"]},
            repo_root=REPO_ROOT,
        )
    assert result["words"][0]["text"] == "ok"
    # Entry 10: "which one" as well as "how far along" — the stage names the
    # engine module, not a bare "engine".
    engine_calls = [call for call in recorder.calls if call[0] == f"engine:{STUB.split('.')[-1]}"]
    details = [call[3] for call in engine_calls]
    assert "downloading 10%" in details
    assert "downloading 100%" in details
    # No unit is known for a download's lines, so no count is invented.
    assert all(call[1] is None and call[2] is None for call in engine_calls)


def test_child_stderr_forwarding_never_raises_with_no_sink_installed() -> None:
    with progress.install(None):
        result = run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"stderr_lines": ["loading weights"]},
            repo_root=REPO_ROOT,
        )
    assert result["words"][0]["text"] == "ok"


def test_a_piped_run_stays_silent(capsys: pytest.CaptureFixture[str]) -> None:
    # The default sink checks `isatty()` on every call; a redirected run (the
    # owner's own routine capture-to-file) must see nothing, ever. No sink is
    # installed here, so whatever is current (the module default `TtySink`)
    # is what a real, un-instrumented CLI run would get — and pytest's
    # captured stderr is never a tty.
    result = run_child(
        interpreter=sys.executable,
        module=STUB,
        request={"stderr_lines": ["downloading 50%"]},
        repo_root=REPO_ROOT,
    )
    assert result["words"][0]["text"] == "ok"
    assert capsys.readouterr().err == ""


def test_the_error_path_still_carries_the_stderr_tail(tmp_path: Path) -> None:
    # `boom` lands in the `ok: False` branch, whose message never carried a
    # stderr tail (before or after this change) — only "wrote no response"
    # does, because that is the only path with no structured error to report
    # instead. So a real crash that skips `_child_main`'s own exception
    # handler is needed: `os._exit` after writing to stderr, which leaves no
    # response file behind at all.
    crash = tmp_path / "crash_engine.py"
    crash.write_text(
        "import os, sys\n"
        "\n"
        "def child_main(request):\n"
        "    print('downloading 10%', file=sys.stderr, flush=True)\n"
        "    print('downloading 90%', file=sys.stderr, flush=True)\n"
        "    os._exit(3)\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="crash_engine",
            request={},
            repo_root=tmp_path,
        )
    message = str(excinfo.value)
    assert "wrote no response" in message
    assert "downloading 10%" in message
    assert "downloading 90%" in message


def test_stderr_forwarding_still_respects_the_tail_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `ENGINE_SUBPROCESS_STDERR_TAIL` trims the *tail* kept for an error
    # message; forwarding every line to the sink (asserted separately above)
    # is unaffected by that trim. Proven the same way as the tail test: a
    # crash with no response file, so the tail actually reaches the message.
    monkeypatch.setattr(C, "ENGINE_SUBPROCESS_STDERR_TAIL", 1)
    crash = tmp_path / "crash_engine.py"
    crash.write_text(
        "import os, sys\n"
        "\n"
        "def child_main(request):\n"
        "    for line in ('one', 'two', 'three'):\n"
        "        print(line, file=sys.stderr, flush=True)\n"
        "    os._exit(3)\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="crash_engine",
            request={},
            repo_root=tmp_path,
        )
    message = str(excinfo.value)
    assert "three" in message
    assert "one" not in message
