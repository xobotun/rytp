"""M1, walked: two URLs in, one rendered file out (design §11).

Seven parts, seven green suites, never once run in the same process. Every
external tool is faked at the seam its own plan published, so this runs on a
machine with no ffmpeg, no yt-dlp, no model and no network.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from rytp import constants as C
from rytp.commands import resolve
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.models import ChannelEntry, RawWord
from tests.consistency import write_test_wav
from tests.fake_engines import FakeAligner, registered
from tests.fakes import FakeYtDlpRunner

#: Two corpora with one shared run and one distinct one, so the assembler has
#: to cross a seam to satisfy the target. Russian, as the product is.
SCRIPTS: dict[int, tuple[str, ...]] = {}

VIDEO_A_WORDS = ("мы", "все", "хорошо", "понимаем")
VIDEO_B_WORDS = ("это", "будет", "совершенно", "неизбежно")
TARGET = "мы все совершенно неизбежно"

WORD_MS = 400
WORD_LENGTH_MS = 320

LOUDNORM_STDERR = json.dumps(
    {
        "input_i": "-27.24",
        "input_tp": "-8.51",
        "input_lra": "6.90",
        "input_thresh": "-37.51",
        "output_i": "-16.02",
        "output_tp": "-1.50",
        "output_lra": "6.70",
        "output_thresh": "-26.29",
        "normalization_type": "dynamic",
        "target_offset": "0.02",
    }
)

PROBE_STDOUT = json.dumps(
    {"streams": [{"width": 1280, "height": 720, "avg_frame_rate": "25/1"}]}
)


class CorpusTranscriber:
    """Says something different in each video, keyed by the WAV's filename."""

    name = "corpus"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        words = SCRIPTS[int(Path(audio).stem)]
        for index, text in enumerate(words):
            yield RawWord(
                text=text,
                start_ms=index * WORD_MS,
                end_ms=index * WORD_MS + WORD_LENGTH_MS,
                confidence=0.9,
            )


def _fake_ffmpeg_decode(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Write the WAV ffmpeg would have written. The output path is last."""
    write_test_wav(Path(cmd[-1]), duration_ms=4000)
    return subprocess.CompletedProcess(list(cmd), 0, "", "")


def _fake_render_runner(args: list[str]) -> Any:
    """One runner for ffprobe, the loudness passes and the encodes."""
    from rytp.render.ffmpeg import CompletedRun

    if "ffprobe" in Path(args[0]).name:
        return CompletedRun(tuple(args), 0, PROBE_STDOUT, "")
    if "loudnorm" in " ".join(args) and "null" in args:
        return CompletedRun(tuple(args), 0, "", LOUDNORM_STDERR)
    out = Path(args[-1])
    if out.suffix:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 2048)
    return CompletedRun(tuple(args), 0, "", "")


@pytest.fixture()
def pipeline(
    db: Database, data_dir: object, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Database]:
    """Every external tool, replaced at the seam its own part published."""
    import rytp.acquire as acquire
    import rytp.acquire.captions as captions
    import rytp.audio.extract as extract
    import rytp.commands.catalog as catalog
    import rytp.render.ffmpeg as render_ffmpeg

    runner = FakeYtDlpRunner()
    monkeypatch.setattr(acquire, "RealYtDlpRunner", lambda: runner)
    monkeypatch.setattr(captions, "RealYtDlpRunner", lambda: runner)
    monkeypatch.setattr(extract, "_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(extract, "_run_ffmpeg", _fake_ffmpeg_decode)
    monkeypatch.setattr(
        render_ffmpeg.Tools,
        "resolve",
        classmethod(lambda cls, runner=None: cls.faked(_fake_render_runner)),
    )

    probed: dict[str, ChannelEntry] = {
        "https://example.invalid/w/VIDEO_A": ChannelEntry(
            external_id="VIDEO_A",
            title="Первое видео",
            url="https://example.invalid/w/VIDEO_A",
            duration_ms=4000,
            kind="video",
            published_at="2026-01-01T00:00:00+00:00",
        ),
        "https://example.invalid/w/VIDEO_B": ChannelEntry(
            external_id="VIDEO_B",
            title="Второе видео",
            url="https://example.invalid/w/VIDEO_B",
            duration_ms=4000,
            kind="video",
            published_at="2026-01-02T00:00:00+00:00",
        ),
    }
    monkeypatch.setattr(catalog, "_probe_video", lambda url: probed[url])

    # design §5's randomised inter-request pause (20-60s real, plus a
    # per-file delay) is deliberate throttling against the real yt-dlp
    # network pool; the worker's own tests inject a no-op `sleep` by calling
    # `run_worker` directly, but the `worker` CLI/TUI command always uses
    # `time.sleep`. This walk goes through the registered command on
    # purpose (it is the surface both the CLI and the TUI actually use), so
    # the settings are the seam here: zeroing them keeps the walk a
    # composition test rather than a multi-minute sleep.
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MIN, "0")
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MAX, "0")
    set_setting(db, C.SETTING_DOWNLOAD_FILE_DELAY, "0")

    SCRIPTS.clear()
    with registered(CorpusTranscriber, FakeAligner):
        yield db
    SCRIPTS.clear()


def _run(db: Database, name: str, argline: str = "") -> Any:
    """Run a registered command the way a user types it.

    Through `parse_arguments`, not by calling the handler with keywords, for
    two reasons. It exercises the surface M1 is actually about — one
    definition, two surfaces — and it is immune to the flag renames Task 2
    mandates, because every value below is either positional or a flag whose
    spelling is settled.
    """
    from rytp.tui.palette import parse_arguments

    cmd = resolve(name)
    return cmd.handler(db, **parse_arguments(cmd, argline))


def _drain(db: Database, *, pool: str = "all") -> None:
    """Run the worker until the queue stops giving it anything."""
    result = _run(db, "worker", f"pool={pool} once=true max_jobs=200")
    assert result.message


def _state(db: Database, kind: str, target_id: int) -> str:
    row = db.conn.execute(
        "SELECT state, last_error FROM jobs WHERE kind = ? AND target_id = ?",
        (kind, target_id),
    ).fetchone()
    assert row is not None, f"no {kind} job for {target_id}"
    assert row["state"] == "done", f"{kind} for {target_id}: {row['last_error']}"
    return str(row["state"])


def test_m1_two_urls_in_one_rendered_file_out(pipeline: Database) -> None:
    db = pipeline

    # --- catalogue, by URL, as design §5 says ------------------------
    for url in (
        "https://example.invalid/w/VIDEO_A",
        "https://example.invalid/w/VIDEO_B",
    ):
        _run(db, "videos.add", url)
    ids = [
        int(row["id"])
        for row in db.conn.execute("SELECT id FROM videos ORDER BY external_id")
    ]
    assert len(ids) == 2
    SCRIPTS[ids[0]] = VIDEO_A_WORDS
    SCRIPTS[ids[1]] = VIDEO_B_WORDS

    # --- acquire: audio and a rendition as separate assets ----------
    for video_id in ids:
        _run(db, "ingest", str(video_id))
    _drain(db)
    for video_id in ids:
        _state(db, "download", video_id)
        _state(db, "extract_wav", video_id)
        roles = {
            str(row["role"])
            for row in db.conn.execute(
                "SELECT role FROM assets WHERE video_id = ?", (video_id,)
            )
        }
        assert {"audio", "video"} <= roles, roles

    # --- transcribe and align, queued from the command, run by the
    #     worker: the path the TUI uses (design §10) -----------------
    for video_id in ids:
        # Boundary refinement is Part 3's to test and needs real speech; this
        # test is about whether the parts compose.
        _run(
            db,
            "transcribe.run",
            f"{video_id} transcriber=corpus aligner=fake-aligner "
            f"refine=false enqueue=true",
        )
    _drain(db)
    for video_id in ids:
        _state(db, "transcribe", video_id)
    for video_id in ids:
        aligned = db.conn.execute(
            "SELECT COUNT(*) FROM words WHERE video_id = ? AND source = 'aligned'",
            (video_id,),
        ).fetchone()[0]
        assert aligned == len(SCRIPTS[video_id]), (
            f"video {video_id} has {aligned} cuttable words; contracts §3 makes "
            f"`aligned` the only tier that may be cut"
        )
    # Deliberately not `SELECT DISTINCT source FROM words == {'aligned'}`.
    # Promoting a video to tier 2 deletes its caption words, which makes
    # `caption_words` runnable again — so a later `reconcile` may legitimately
    # put them back. Whether it should is Part 3's question (see the defect
    # note below); what M1 needs is that every word of the target is cuttable,
    # which is what the assembler asserts for itself two steps down.

    # --- index: enqueued by the transcribe handler, not by hand -----
    _drain(db, pool="cpu")
    for video_id in ids:
        _state(db, "index", video_id)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] >= 2

    # --- search: the backward index, both directions of the corpus --
    hits = _run(db, "search.words", '"все хорошо" cuttable=true')
    assert hits.rows, "a phrase that was said is not findable"
    everywhere = _run(db, "search.words", '"совершенно неизбежно" cuttable=true')
    assert everywhere.rows

    # --- assemble: the point of M1 is more than one source ----------
    _run(db, "assemble.plan", f'"{TARGET}" name=m1 force=true')
    from rytp.assemble.cutlist import read_cutlist

    cutlist = read_cutlist("m1")
    sources = {slot.video_id for slot in cutlist.fragments}
    assert len(sources) >= 2, (
        f"M1 requires a sentence drawn from several source videos; got {sources}"
    )
    assert not cutlist.gaps, "every word of the target exists in the corpus"

    # --- render: a real file and a source list ----------------------
    # gap_ms=0 turns freeze-frame pauses off: design §9 makes them optional
    # and Part 6 measures them itself. M1 asks for a file and a source list.
    rendered = _run(db, "render.run", "m1 gap_ms=0")
    # render.run's own CommandResult carries the summary in its rows/columns
    # rather than a `message` line (message is None; the table is the report).
    assert rendered.rows, rendered
    row = db.conn.execute(
        "SELECT output_path, state FROM renders WHERE cutlist_name = 'm1'"
    ).fetchone()
    assert row is not None and row["state"] == "rendered", row
    output = Path(str(row["output_path"]))
    assert output.exists(), output
    report = output.parent / "report.md"
    assert report.exists(), "design §9: a report beside the output"
    text = report.read_text(encoding="utf-8")
    for video_id in sources:
        assert str(video_id) in text, "the report must list every source"


def test_the_whole_walk_touches_no_network_and_no_binary(
    pipeline: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that keeps this test honest as the pipeline grows.

    Every seam above is a monkeypatch on a specific module attribute; a new
    stage that shells out through `subprocess.run` directly would slip past
    them and quietly start needing ffmpeg on the machine running the suite.
    """
    import socket
    import subprocess as sp

    def no_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("the pipeline opened a socket")

    def no_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"the pipeline shelled out: {args!r}")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(sp, "run", no_subprocess)
    monkeypatch.setattr(sp, "Popen", no_subprocess)

    db = pipeline
    _run(db, "videos.add", "https://example.invalid/w/VIDEO_A")
    video_id = int(db.conn.execute("SELECT id FROM videos").fetchone()["id"])
    SCRIPTS[video_id] = VIDEO_A_WORDS
    _run(db, "ingest", str(video_id))
    _drain(db)
    _state(db, "download", video_id)
