# Part 6 — Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a cut list into `output/{render_id}/output.mp4` — one uploadable file, hard cuts, one canvas, matched loudness, freeze-frame gaps — and `output/{render_id}/report.md` beside it listing the assembled text, every fragment with its source and timings, every word that could not be found, and a paste-ready description block.

**Architecture:** Every ffmpeg and ffprobe invocation is *built* by a pure function returning a `list[str]` and *run* through an injected `Runner` seam, so the whole suite tests the argument lists without the binary. Rendering is two phases: `plan_render` reads the database and the sources and produces a `RenderPlan` — canvas geometry, per-source loudness gain, per-seam gap — with no subprocess; `render_cutlist` executes that plan, one intermediate per fragment plus one concat pass. Gaps are freeze-frames produced by `tpad=stop_mode=clone` on the *preceding* fragment rather than by separate filler segments, so a gap costs no extra ffmpeg run and cannot drift out of sync with its audio.

**Tech Stack:** Python 3.11 (stdlib only: `subprocess`, `pathlib`, `statistics`, `hashlib`, `json`, `dataclasses`), SQLite through Part 1's `Database`, ffmpeg and ffprobe as external binaries invoked with list arguments, pytest. No new dependency, optional or otherwise.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (§9 Render, §8 Assembly for the cut list, §4 Assets and Speakers, §5 pools) and the binding `docs/superpowers/specs/2026-09-21-rytp-contracts.md` (§2 layout, §3 schema, §5 registry and job handlers, §7 filesystem, §8 conventions).

> **Where the cut-list contract came from.** Part 5's plan was still being written when this one was, so the format in Task 6 is the one its author sent directly and called settled: `load_cutlist` / `write_cutlist` / `dumps_cutlist` / `CutlistError`, one ordered `[[slot]]` array whose `kind` is `"fragment"` or `"gap"`, a gap slot *being* a missing word with ranked `substitutions`, and fragment slots carrying the contracts §4 `Fragment` fields plus `align_score`, `cost`, `video_speaker_id`, `speaker_label`, optional `gap_before_ms` and nested `alternatives`. It is reproduced in Task 6's tests as a `SimpleNamespace`, and Part 6 touches it through exactly one adapter, so if Part 5's published plan differs, `request_from_cutlist` is the only function to change and nothing else in this part moves.


## Global Constraints

Copied from contracts §1 and §8. Every task's requirements implicitly include this section.

- Python >= 3.11. Target 3.11 syntax; `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- ffmpeg and ffprobe are required system binaries. Runs on Windows: no POSIX-only calls, **no shell pipelines**, no symlinks. `subprocess` with list arguments only; filter graphs are passed as one argument, never as shell text.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` on `rytp/`.
- Heavy dependencies are imported lazily inside functions, never at module import time. Part 6 adds none.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. The repository root currently holds media files named with real ids — never name one in a file, a test, or a commit message.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- **Errors.** Domain errors subclass `RytpError` from `rytp/models.py`. One line to stderr, exit 1, no traceback. Unexpected exceptions propagate.
- **Time.** Milliseconds, integers, everywhere. Database timestamps are ISO-8601 UTC strings from `datetime.now(UTC).isoformat()` (Part 1 exposes `utc_now_iso()`).
- **Database access.** Every stage takes an open `Database`; nothing opens its own connection. Multi-statement writes use `db.transaction()`.
- **Tests.** No test may touch the network, ever. **The suite must pass with no ffmpeg on `PATH`**: command construction is unit-tested, execution is faked. The two tests that genuinely run ffmpeg are marked `@pytest.mark.skipif(shutil.which("ffmpeg") is None, ...)`, generate their own inputs from `lavfi` sources, and commit no fixture media.
- **Commits.** One per task, conventional-commit prefix, present tense.

## How to run the tests

Activate the project's virtualenv and install the package in editable mode once; then every "run the test" step in this plan is exactly `python -m pytest <args>` from the repository root — no absolute interpreter paths, no developer-specific directories, because this ships on Windows (contracts §1).

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q
```

Scratch files go wherever `tests/conftest.py` puts them; `RYTP_TEST_TMP` overrides that root, and nothing in this plan requires setting it.

## Consumes from other parts

Confirmed with the authors of Parts 1, 2 and 5. Nothing here is re-implemented.

**Part 1** — `rytp.db.Database` (`.conn` with `row_factory = sqlite3.Row` and foreign keys on, `.transaction()`, re-entrant); `rytp.config.paths()` (a *function*, no singleton) with `.output_dir(render_id: str)`, `.cutlist(name: str)`, `.media_dir(video_id: int)`, and `rytp.config.ensure_dir(path)`; `rytp.models.RytpError`, `NotFoundError`, `InvalidInputError`, `Fragment`, `utc_now_iso()`; `rytp.commands.{REQUIRED, Param, Command, CommandResult, register, resolve, COMMANDS}`; `rytp.constants.MS_PER_SECOND`; `tests/conftest.py` fixtures `tmp_path`, `data_dir`, `db`.

**Part 1, continued** — the `renders` table of contracts §3 (`id, cutlist_name, output_path, canvas_mode, state, created_at, finished_at`, `state IN ('planned','rendered','failed')`) and its `renders_cutlist` index. Part 1 owns migrations; Part 6 only reads and writes rows. Also `rytp.db.queries.clamp_limit` and `rytp.constants.DEFAULT_LIST_LIMIT` for `render.list`.

**Part 2** — `rytp.db.queries.{asset_for, assets_for, has_usable_asset}`; `rytp.constants.FFMPEG_ERROR_TAIL_CHARS`; the job registry in `rytp/jobs/__init__.py`: `Readiness`, `JobKind(name, pool, readiness, handler, summary, target_kind="video", reopenable=True)`, `register_job_kind`, and `JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]` which `register_job_kind` fills from the same `JobKind`. Part 2's `fetch-video` command is the one named in Part 6's missing-media error messages. `tests/fakes.py::{make_video, touch}` is the shared factory.

**Part 5** — `rytp.assemble.cutlist.{load_cutlist, CutList, Slot, CutlistError}`, settled with the Part 5 author and reproduced in Task 6. Part 6 imports `load_cutlist` at exactly **two** sites, both lazy and both inside a function body: `rytp/commands/render.py::_load_cutlist` and `rytp/render/run.py::run_render_job`. Everything else in `rytp/render/` works on Part 6's own `RenderFragment` / `MissingWord` types, built from a cut list by `request_from_cutlist()`, which reads attributes off whatever object it is handed and imports nothing from `rytp.assemble`. **This is deliberate:** the whole Part 6 suite passes with no Part 5 on disk, and if the cut-list dataclasses move or gain fields, exactly one adapter changes.

## Produces

- `rytp/render/` — the render stage. `render_cutlist(db, request, options, *, tools=None) -> RenderResult`.
- Four registered commands: `render.run`, `render.pauses`, `render.list` and `render.remove`.
- One `JobKind` entry contributed to Part 2's registry (`kind="render"`, pool `cpu`, `target_kind="render"`) plus the light predicate module it imports.
- One appended section in `rytp/constants.py`.

## Decisions this plan makes, and why

Six choices the design left open. They are settled here so an executor does not have to re-derive them; each is called out again at the task that implements it.

1. **Intermediates are Matroska with `pcm_s16le` audio, and the final pass always re-encodes audio.** AAC carries ~1024 samples of encoder priming per file. The concat demuxer does not trim it, so with AAC intermediates the audio slides ~21 ms later at *every* seam — most of a second over forty fragments, plus a click at each. PCM has no priming. The final pass therefore copies video (`-c:v copy`, no re-encode, fast) and encodes audio once (`-c:a aac`), whether or not loudness normalization is on; `--no-loudnorm` only drops the `-af loudnorm=…` from that one command.
2. **Loudness is matched per *source video*, then the whole programme is normalized once.** Two-pass `loudnorm` on a 700 ms fragment is meaningless — EBU R128 integrated loudness needs seconds of material, and gating makes short measurements wander. So each distinct source video gets one gain in dB (target minus its measured LUFS, clamped), which is exactly what "volume doesn't lurch between sources" asks for, and the assembled programme — long enough to measure honestly — gets the classic two-pass `loudnorm` in the final command.
3. **Output frame rate is measured, not assumed.** A decade of Russian uploads is PAL territory; forcing 25 fps material to 30 duplicates every fifth frame and looks it. ffprobe runs once per *distinct source video* and the output rate is the commonest source rate, capped, overridable with `--fps`.
4. **Gaps are deterministic.** Design §8 requires that the same input give the same output. The old code sampled a Gaussian per seam; this one takes the **median** measured gap. No randomness, no seed.
5. **Missing media is fatal, missing words are not.** Design §8 says report a word gap plainly and render everything else — that is about *words*. A fragment whose video rendition is not on disk cannot be rendered at all, and rendering its audio over black would be a silent lie, so the render refuses, listing every unreadable fragment at once with the command that fetches each.
6. **`render_id` is derived, not stamped.** `sanitised cut-list name` + 8 hex of the plan hash. Re-rendering the same cut list with the same options lands in the same directory and overwrites; changing an option makes a new one. Deterministic, and it keeps `output/` navigable. It is the *directory* name; the render's database identity is its `renders.id`, which is also what a job targets.
7. **A render is a row before it is a file.** Contracts §3 gives Part 6 a `renders` table, so a render has an integer identity like every other job target: `rytp render.run --queue` opens a `planned` row and enqueues `target_id = renders.id`. The row is opened before any encoding and closed `rendered` or `failed` afterwards, which means a render that dies mid-encode leaves a trace, the readiness predicate can ask a real question about the world, and `render.list` is a one-line query rather than a feature.

## File Structure

| File | Responsibility |
|---|---|
| `rytp/render/__init__.py` | **Docstring only.** It must stay import-light: `rytp/jobs/__init__.py` imports `rytp.render.readiness` at module level, and importing a submodule executes the package `__init__` first — anything heavy here would land in every CLI invocation. |
| `rytp/render/ffmpeg.py` | The binaries and the seam: `Runner`, `Tools`, `run_command`, errors, every ffmpeg/ffprobe argument list, the audio filter chain, the concat list file, the loudnorm JSON parser. Knows nothing about the database. |
| `rytp/render/canvas.py` | Output geometry: probing a source, choosing the canvas and frame rate, the per-fragment scale/pad/fps/freeze filter chain. |
| `rytp/render/pauses.py` | Between-word gap statistics from aligned words, and the degeneracy guard that stops a zero-inflated transcript teaching us that the median pause is zero. |
| `rytp/render/report.py` | The structured report types and the markdown renderer. Every type is JSON-ready so JSON and subtitle output can be added later without rework. |
| `rytp/render/run.py` | Planning and execution: resolve sources, build the `RenderPlan`, run it, write `output.mp4` and `report.md`. Holds the job-handler entry point. |
| `rytp/render/readiness.py` | The `render` readiness predicate. Light on purpose — Part 2's registry imports it. |
| `rytp/commands/render.py` | `render.run` and `render.pauses`, thin over `run.py` and `pauses.py`. The one place that imports Part 5's cut-list loader. |
| `tests/render_fakes.py` | `RecordingRunner` (records argv, creates the output file each command names) and cut-list stand-ins. A plain module, not a conftest: five test modules import it. |
| `tests/test_render_ffmpeg.py` … `tests/test_commands_render.py` | One test module per task. |

**Four files beyond the contracts §2 tree** (`render/ ffmpeg.py canvas.py report.py`), with reasons, per the "do not create files outside this tree without saying why" rule:

- `readiness.py` — requested by the Part 2 author. Part 2's registry needs a predicate it can import at module level without dragging render's machinery into every `rytp` invocation.
- `run.py` — the orchestrator cannot live in `__init__.py` for the reason in the table above, and must not live in `ffmpeg.py`, which is the one module with no database and no policy in it.
- `pauses.py` — statistics over `words` is a separate responsibility from both geometry and command construction, and it is the piece the owner explicitly wants to be able to switch off, so it stays isolated behind one function.
- `tests/render_fakes.py` — shared by five test modules; Part 2 and Part 3 set the same precedent (`tests/fakes.py`, `tests/fake_engines.py`).

**Modified:** `rytp/constants.py` (one appended section, Task 1), `rytp/jobs/__init__.py` (one thunk plus one registry entry, Task 8), `rytp/commands/__init__.py` (one line in the sibling-import block, Task 9).

---

### Task 1: Render constants and the ffmpeg seam

Everything else in this part calls ffmpeg through one function, and every argument list is built by a pure function so the suite can assert on it without the binary. This task builds that seam and nothing else: no geometry, no database.

The runner returns a small record rather than `subprocess.CompletedProcess` so a fake is three lines, and `run_command` raises on a non-zero exit with the *tail* of stderr — ffmpeg puts the actual complaint last, after several screens of build configuration.

**Files:**
- Modify: `rytp/constants.py` (append a new section at the very end; touch nothing above it)
- Create: `rytp/render/__init__.py`, `rytp/render/ffmpeg.py`
- Create: `tests/render_fakes.py`
- Test: `tests/test_render_ffmpeg.py`

**Interfaces:**
- Consumes: `rytp.models.RytpError`; `rytp.constants.{MS_PER_SECOND, FFMPEG_ERROR_TAIL_CHARS}`.
- Produces:
  - `rytp.render.ffmpeg.{RenderError, FfmpegNotFoundError, FfmpegFailedError}`
  - `CompletedRun(args: tuple[str, ...], returncode: int, stdout: str, stderr: str)`
  - `Runner = Callable[[list[str]], CompletedRun]`
  - `ffmpeg_binary() -> str`, `ffprobe_binary() -> str`
  - `subprocess_runner(timeout_s: int) -> Runner`
  - `run_command(args: list[str], *, runner: Runner | None = None, what: str = "", timeout_s: int = C.RENDER_FFMPEG_TIMEOUT_S) -> CompletedRun`
  - `seconds(ms: int) -> str`
  - `probe_command(path: Path, *, binary: str | None = None) -> list[str]`
  - `parse_loudnorm_json(stderr: str) -> LoudnormMeasurement | None`, `LoudnormMeasurement(input_i, input_tp, input_lra, input_thresh, target_offset)`
  - `tests/render_fakes.RecordingRunner`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_ffmpeg.py`:

```python
"""The ffmpeg seam: argument lists are data, execution is injected (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import ffmpeg as F

from tests.render_fakes import RecordingRunner

LOUDNORM_STDERR = """\
[Parsed_loudnorm_0 @ 0x5599] 
{
        "input_i" : "-27.24",
        "input_tp" : "-8.51",
        "input_lra" : "6.90",
        "input_thresh" : "-37.51",
        "output_i" : "-16.02",
        "output_tp" : "-1.50",
        "output_lra" : "6.70",
        "output_thresh" : "-26.29",
        "normalization_type" : "dynamic",
        "target_offset" : "0.02"
}
"""


def test_seconds_renders_milliseconds_with_three_decimals() -> None:
    assert F.seconds(0) == "0.000"
    assert F.seconds(1) == "0.001"
    assert F.seconds(612_340) == "612.340"


def test_probe_command_asks_for_the_first_video_stream_as_json() -> None:
    args = F.probe_command(Path("media/3/audio.m4a"), binary="ffprobe")
    assert args[0] == "ffprobe"
    assert "-select_streams" in args and "v:0" in args
    assert args[-1] == str(Path("media/3/audio.m4a"))
    assert "json" in args


def test_parse_loudnorm_json_reads_the_five_measured_values() -> None:
    measured = F.parse_loudnorm_json(LOUDNORM_STDERR)
    assert measured is not None
    assert measured.input_i == pytest.approx(-27.24)
    assert measured.input_tp == pytest.approx(-8.51)
    assert measured.input_lra == pytest.approx(6.90)
    assert measured.input_thresh == pytest.approx(-37.51)
    assert measured.target_offset == pytest.approx(0.02)


def test_parse_loudnorm_json_of_silence_is_unknown() -> None:
    assert F.parse_loudnorm_json('{"input_i" : "-inf", "input_tp" : "-inf"}') is None
    assert F.parse_loudnorm_json("no json at all") is None


def test_run_command_returns_the_recorded_result() -> None:
    runner = RecordingRunner()
    result = F.run_command(["ffmpeg", "-version"], runner=runner)
    assert result.returncode == 0
    assert runner.calls == [["ffmpeg", "-version"]]


def test_run_command_raises_with_the_tail_of_stderr() -> None:
    runner = RecordingRunner(returncode=1, stderr="x" * 5000 + "Invalid argument")
    with pytest.raises(F.FfmpegFailedError) as excinfo:
        F.run_command(["ffmpeg", "-nope"], runner=runner)
    message = str(excinfo.value)
    assert "Invalid argument" in message
    assert len(message) < C.FFMPEG_ERROR_TAIL_CHARS + 200


def test_a_missing_binary_says_where_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    with pytest.raises(F.FfmpegNotFoundError, match="ffmpeg.org"):
        F.ffmpeg_binary()
    with pytest.raises(F.FfmpegNotFoundError, match="ffmpeg.org"):
        F.ffprobe_binary()


def test_render_errors_are_rytp_errors() -> None:
    from rytp.models import RytpError

    assert issubclass(F.RenderError, RytpError)
    assert issubclass(F.FfmpegFailedError, F.RenderError)
    assert issubclass(F.FfmpegNotFoundError, F.RenderError)
```

Create `tests/render_fakes.py`:

```python
"""Fakes shared by the Part 6 test modules.

Nothing here runs a binary or touches the network. ``RecordingRunner``
does the two things a fake ffmpeg must do besides recording: answer a
probe or a loudness scan the way the real one would, and create the file
the command says it writes, so the orchestrator's existence checks
behave the way they will in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from rytp.render.ffmpeg import CompletedRun


def probe_json(width: int, height: int, *, fps: str = "25/1", sar: str = "1:1") -> str:
    """One ffprobe reply, the shape `probe_command` asks for."""
    return json.dumps(
        {
            "streams": [
                {
                    "width": width,
                    "height": height,
                    "r_frame_rate": fps,
                    "sample_aspect_ratio": sar,
                }
            ]
        }
    )


def loudnorm_json(
    input_i: float,
    *,
    tp: float = -3.0,
    lra: float = 7.0,
    thresh: float = -30.0,
    offset: float = 0.1,
) -> str:
    """One measuring-pass stderr, chatter line included."""
    payload = {
        "input_i": f"{input_i:.2f}",
        "input_tp": f"{tp:.2f}",
        "input_lra": f"{lra:.2f}",
        "input_thresh": f"{thresh:.2f}",
        "target_offset": f"{offset:.2f}",
    }
    return "[Parsed_loudnorm_0 @ 0x1] \n" + json.dumps(payload)


class RecordingRunner:
    """A fake :data:`rytp.render.ffmpeg.Runner`.

    Args:
        returncode: what every invocation returns.
        stdout: canned stdout for any command not matched below.
        stderr: canned stderr for any command not matched below.
        geometry: stdout for ffprobe commands — one string, or a dict
            keyed by a substring of the path being probed, so a test
            with mixed-shape sources can answer differently per source.
            A dict is scanned in order, so put the specific key first.
        loudness: stderr for loudness-measuring commands, same keying.
        fail_when: when set, only commands containing this substring
            get ``returncode``; everything else succeeds. Lets a test
            fail one stage without failing the probe that precedes it.
        make_outputs: create the last argument as a file when it looks
            like an output path (no leading dash, has a suffix).
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        geometry: str | dict[str, str] | None = None,
        loudness: str | dict[str, str] | None = None,
        fail_when: str = "",
        make_outputs: bool = True,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.geometry = geometry
        self.loudness = loudness
        self.fail_when = fail_when
        self.make_outputs = make_outputs
        self.calls: list[list[str]] = []

    @staticmethod
    def is_probe(args: list[str]) -> bool:
        return "-show_entries" in args

    @staticmethod
    def is_measurement(args: list[str]) -> bool:
        return any("print_format=json" in arg for arg in args)

    @staticmethod
    def _pick(table: str | dict[str, str], args: list[str]) -> str:
        if isinstance(table, str):
            return table
        for needle, value in table.items():
            if any(needle in arg for arg in args):
                return value
        return ""

    def __call__(self, args: list[str]) -> CompletedRun:
        self.calls.append(list(args))
        out, err = self.stdout, self.stderr
        if self.geometry is not None and self.is_probe(args):
            out = self._pick(self.geometry, args)
        if self.loudness is not None and self.is_measurement(args):
            err = self._pick(self.loudness, args)
        code = self.returncode
        if self.fail_when and not any(self.fail_when in arg for arg in args):
            code = 0
        last = args[-1]
        if self.make_outputs and code == 0 and not last.startswith("-"):
            target = Path(last)
            if target.suffix and target.suffix != ".":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"\x00" * 64)
        return CompletedRun(args=tuple(args), returncode=code, stdout=out, stderr=err)

    def commands_containing(self, needle: str) -> list[list[str]]:
        """Every recorded call with ``needle`` in one of its arguments."""
        return [call for call in self.calls if any(needle in arg for arg in call)]

    @property
    def probes(self) -> list[list[str]]:
        return [call for call in self.calls if self.is_probe(call)]

    @property
    def measurements(self) -> list[list[str]]:
        return [call for call in self.calls if self.is_measurement(call)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_ffmpeg.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render'`.

- [ ] **Step 3: Append the render section to `rytp/constants.py`**

Append at the very end of the file. Do not reorder or edit anything above it.

```python
# ---------------------------------------------------------------------------
# Part 6 — render (design §9)
# ---------------------------------------------------------------------------

#: Seconds before one ffmpeg invocation is abandoned. A whole-programme
#: concat pass over a long cut list decodes minutes of video, and the
#: machine is also transcribing; an hour is generous rather than tight.
RENDER_FFMPEG_TIMEOUT_S: Final = 3600

#: Seconds before one ffprobe invocation is abandoned. It reads a header.
RENDER_FFPROBE_TIMEOUT_S: Final = 60

#: Canvas modes (design §9 "16:9 with pillarboxing by default. Configurable,
#: with 'smallest bounding box across all sources' as an alternative").
RENDER_CANVAS_MODES: Final = ("16:9", "bbox")
RENDER_DEFAULT_CANVAS_MODE: Final = "16:9"

#: The default canvas aspect, as a pair so the arithmetic stays exact.
RENDER_DEFAULT_ASPECT: Final = (16, 9)

#: Canvas height when nothing is given: the tallest source, capped here.
#: Upscaling a 360p archive to 1080p buys nothing but encode time, and a
#: 2160p source in a talking-head mix is not worth the hours.
RENDER_MAX_CANVAS_HEIGHT: Final = 1080

#: Floor for a derived canvas height, so a mis-probed source cannot
#: produce a 2-pixel output.
RENDER_MIN_CANVAS_HEIGHT: Final = 240

#: Colour of the pillar/letterbox bars (design §9 "pillarboxing").
RENDER_PAD_COLOR: Final = "black"

#: Output frame rate is the commonest source rate, capped here. Above 60
#: the file gets large for no benefit on archive footage.
RENDER_MAX_FPS: Final = 60

#: Frame rate used when no source could be probed for one. 25, not 30:
#: this is a European archive and PAL is the likelier majority.
RENDER_FALLBACK_FPS: Final = 25

#: Final video encode settings. CRF 20 is visually transparent for
#: talking-head content at these resolutions; "medium" is the preset
#: where x264 stops paying for itself on an i7 laptop.
RENDER_VIDEO_CODEC: Final = "libx264"
RENDER_VIDEO_CRF: Final = 20
RENDER_VIDEO_PRESET: Final = "medium"
RENDER_PIXEL_FORMAT: Final = "yuv420p"

#: Per-fragment intermediates. Matroska because it carries PCM, and PCM
#: because AAC's ~1024-sample encoder priming is not trimmed by the
#: concat demuxer and would slide the audio later at every seam.
RENDER_INTERMEDIATE_SUFFIX: Final = ".mkv"
RENDER_INTERMEDIATE_FORMAT: Final = "matroska"
RENDER_INTERMEDIATE_AUDIO_CODEC: Final = "pcm_s16le"

#: Final audio. 48 kHz stereo AAC is what every upload target wants, and
#: every fragment is resampled to it before concatenation so the demuxer
#: never sees a parameter change.
RENDER_AUDIO_CODEC: Final = "aac"
RENDER_AUDIO_BITRATE: Final = "192k"
RENDER_AUDIO_RATE_HZ: Final = 48_000
RENDER_AUDIO_CHANNELS: Final = 2

#: EBU R128 targets for the final programme pass (design §9 "loudness-
#: normalized by default"). -16 LUFS is the speech-content convention and
#: survives every platform's own normalization; -1.5 dBTP leaves room for
#: lossy-codec overshoot. LRA 13 rather than the broadcast 11: a
#: multi-source cut legitimately has a wider range than one recording,
#: and asking for 11 makes loudnorm abandon linear mode and start pumping.
RENDER_TARGET_LUFS: Final = -16.0
RENDER_TRUE_PEAK_DBTP: Final = -1.5
RENDER_LOUDNESS_RANGE_LU: Final = 13.0

#: Per-source gain is clamped to this many dB either way. A source that
#: needs more than 12 dB is broken, not quiet, and boosting it that far
#: would lift its noise floor into the mix.
RENDER_MAX_GAIN_DB: Final = 12.0

#: Sample-peak ceiling for the per-fragment limiter, linear. 0.891 is
#: -1.0 dBFS. It exists only to stop a clamped gain clipping the PCM
#: intermediate before the final true-peak limiter gets a say.
RENDER_LIMITER_CEILING: Final = 0.891

#: How much of a source to measure when `video_acoustics.loudness_lufs`
#: is not already there: two minutes from just before the video's first
#: fragment. Measuring a whole hour to place one gain is not worth the
#: minutes it costs.
RENDER_LOUDNESS_WINDOW_MS: Final = 120_000

#: Hex characters of the plan hash in a derived render id, and the cap on
#: the whole id — it becomes a directory name under output/.
RENDER_ID_HASH_CHARS: Final = 8
RENDER_ID_MAX_CHARS: Final = 64

#: Heading above the source list in the paste-ready description block.
#: The audience for that block reads Russian; the report around it is in
#: English like the rest of the CLI.
RENDER_DESCRIPTION_HEADER: Final = "Источники:"

# --- Pause statistics (design §9 "Gap length") -----------------------------

#: Gaps at or below this are "zero". Aligned boundaries land on measured
#: energy minima, so a 1 ms gap means the two words share a boundary.
PAUSE_ZERO_GAP_MS: Final = 1

#: Gaps longer than this are turn boundaries or edits, not between-word
#: pauses, and would drag the median upward.
PAUSE_MAX_GAP_MS: Final = 2_000

#: Fewer measured gaps than this and the distribution is not worth
#: believing; the constant fallback is used instead.
PAUSE_MIN_SAMPLES: Final = 50

#: Above this share of exactly-zero gaps the distribution is degenerate.
#: Design §6: a Whisper-timed transcript has 78.7% of its gaps at exactly
#: zero because the transcriber sets word[i].end == word[i+1].start, and
#: a naive median over that data is zero. Half is a generous line: real
#: aligned speech puts most between-word gaps above zero.
PAUSE_DEGENERATE_ZERO_FRACTION: Final = 0.5

#: Used whenever measurement is unavailable or degenerate. A short,
#: unobtrusive between-word pause.
PAUSE_FALLBACK_GAP_MS: Final = 180

#: Applied gaps are clamped here regardless of what was measured.
PAUSE_MIN_APPLIED_GAP_MS: Final = 40
PAUSE_MAX_APPLIED_GAP_MS: Final = 600

#: Upper bound on rows pulled for one statistic, so a speaker with a
#: hundred hours of aligned words cannot make the query eat the machine.
PAUSE_SAMPLE_LIMIT: Final = 200_000

#: contracts §3: "Cuttable is defined as source = 'aligned'". Only these
#: rows carry an end_ms, so only these can measure a pause at all.
#: If Part 3 has already added a constant for this literal, use theirs
#: and delete this one rather than keeping two.
ALIGNED_WORD_SOURCE: Final = "aligned"
```

- [ ] **Step 4: Create `rytp/render/__init__.py`**

```python
"""Render: cut list in, one uploadable file plus a report out. design §9.

Deliberately empty of imports. ``rytp/jobs/__init__.py`` imports
``rytp.render.readiness`` at module level, and importing a submodule
executes this file first — so anything imported here would be imported by
every CLI invocation, which contracts §1 forbids. The stage entry points
live in :mod:`rytp.render.run`.
"""

from __future__ import annotations
```

- [ ] **Step 5: Write `rytp/render/ffmpeg.py`**

```python
"""ffmpeg and ffprobe: argument lists as data, execution behind a seam.

design §9. Every command in this module is built by a pure function that
returns a ``list[str]``, and every invocation goes through :data:`Runner`.
That split is not ceremony: the bugs in a render pipeline are in the
argument lists, and a test suite that cannot run without the binary
cannot check them. Contracts §1 also rules out shell pipelines, so a
filter graph is one argument, never a string the shell sees.

This module knows nothing about the database, the canvas or the cut list.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.models import RytpError


class RenderError(RytpError):
    """Anything the render stage refuses to do. One line, no traceback."""


class FfmpegNotFoundError(RenderError):
    """ffmpeg or ffprobe is not on PATH."""


class FfmpegFailedError(RenderError):
    """An invocation exited non-zero, timed out, or wrote nothing."""


@dataclass(frozen=True)
class CompletedRun:
    """What a :data:`Runner` gives back."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


#: Injected at every call site so tests never need the binary.
Runner = Callable[[list[str]], CompletedRun]


@dataclass(frozen=True)
class LoudnormMeasurement:
    """The five numbers ffmpeg's measuring pass hands the applying pass."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


def _which(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise FfmpegNotFoundError(
            f"{name} not found on PATH; install it from https://ffmpeg.org/"
        )
    return found


def ffmpeg_binary() -> str:
    """Absolute path to ffmpeg, or a one-line error naming the download."""
    return _which("ffmpeg")


def ffprobe_binary() -> str:
    """Absolute path to ffprobe, or a one-line error naming the download."""
    return _which("ffprobe")


def seconds(ms: int) -> str:
    """Milliseconds as the fixed-point seconds ffmpeg's -ss/-t want."""
    return f"{ms / C.MS_PER_SECOND:.3f}"


def subprocess_runner(timeout_s: int) -> Runner:
    """The real runner. A factory because ffprobe and ffmpeg wait differently."""

    def run(args: list[str]) -> CompletedRun:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, never a shell string
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - guarded by _which
            raise FfmpegNotFoundError(
                f"{args[0]} disappeared between lookup and launch"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise FfmpegFailedError(f"{args[0]} timed out after {timeout_s}s") from exc
        return CompletedRun(
            args=tuple(args),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    return run


def run_command(
    args: list[str],
    *,
    runner: Runner | None = None,
    what: str = "",
    timeout_s: int = C.RENDER_FFMPEG_TIMEOUT_S,
) -> CompletedRun:
    """Run one command, raising :class:`FfmpegFailedError` on a bad exit.

    Only the tail of stderr is quoted: ffmpeg opens with several screens
    of build configuration and puts the actual complaint last.
    """
    result = (runner or subprocess_runner(timeout_s))(args)
    if result.returncode != 0:
        label = what or Path(args[0]).name
        tail = result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS :].strip()
        raise FfmpegFailedError(f"{label} failed (exit {result.returncode}): {tail}")
    return result


def probe_command(path: Path, *, binary: str | None = None) -> list[str]:
    """Ask ffprobe for the first video stream's geometry, as JSON."""
    return [
        binary or ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,sample_aspect_ratio",
        "-of",
        "json",
        str(path),
    ]


def parse_loudnorm_json(stderr: str) -> LoudnormMeasurement | None:
    """Pull the measuring pass's JSON block out of ffmpeg's stderr.

    ffmpeg surrounds it with filter chatter, so the last ``{...}`` pair is
    the block. Silence measures as ``-inf`` and is not a measurement;
    returning ``None`` lets the caller skip normalization rather than
    feed an infinity back into the filter.
    """
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stderr[start : end + 1])
    except ValueError:
        return None
    try:
        values = [
            float(payload[key])
            for key in (
                "input_i",
                "input_tp",
                "input_lra",
                "input_thresh",
                "target_offset",
            )
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if any(math.isinf(v) or math.isnan(v) for v in values):
        return None
    return LoudnormMeasurement(*values)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_ffmpeg.py -q`
Expected: PASS, 8 passed.

- [ ] **Step 7: Prove the package stays import-light**

Run: `python -c "import sys, rytp.render; print(sorted(m for m in sys.modules if m.startswith('rytp')))"`
Expected: `['rytp', 'rytp.render']` — no `rytp.render.ffmpeg`, no `rytp.db`.

- [ ] **Step 8: Commit**

```bash
git add rytp/constants.py rytp/render/__init__.py rytp/render/ffmpeg.py \
        tests/render_fakes.py tests/test_render_ffmpeg.py
git commit -m "feat: add the render constants and the ffmpeg command seam"
```

---

### Task 2: Canvas geometry

Design §9: *"16:9 with pillarboxing by default. Configurable, with 'smallest bounding box across all sources' as an alternative mode."* A decade of uploads guarantees 4:3 and 16:9 at different heights in the same render, so this is not a corner case.

**What "smallest bounding box" means here.** The canvas is the smallest rectangle that every source fits inside without being cropped: height is the tallest source (capped), and the aspect is the *widest* source's aspect. The widest source then fills the canvas edge to edge with no bars at all, and narrower ones are pillarboxed into it. The rejected reading is `max(width) × max(height)`, which for a 1440×1080 4:3 source and a 1280×720 16:9 source would invent a 1440×1080 canvas that neither source has the shape of, and pillarbox *and* letterbox both of them. When every source is 16:9, `bbox` and the default agree — which is the property that makes `bbox` safe to offer.

Frame rate is measured rather than assumed, for the reason in "Decisions" above. `r_frame_rate` arrives as a fraction (`"30000/1001"`), so it is parsed as one.

**Files:**
- Create: `rytp/render/canvas.py`
- Test: `tests/test_render_canvas.py`

**Interfaces:**
- Consumes: `rytp.render.ffmpeg.{Runner, run_command, probe_command, seconds, RenderError}`; `rytp.constants.RENDER_*`.
- Produces:
  - `SourceGeometry(width: int, height: int, fps: float, sar: float = 1.0)` with `.display_width` and `.aspect`
  - `Canvas(width: int, height: int, fps: int, mode: str)` with `.label`
  - `parse_probe_json(text: str) -> SourceGeometry`
  - `probe_geometry(path: Path, *, runner: Runner | None = None, binary: str | None = None) -> SourceGeometry`
  - `choose_fps(geometries: Sequence[SourceGeometry], *, override: int = 0) -> int`
  - `plan_canvas(geometries, *, mode: str, height: int = 0, fps: int = 0) -> Canvas`
  - `video_filter_chain(canvas: Canvas, *, gap_ms: int = 0) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_canvas.py`:

```python
"""Canvas geometry: one output shape for mixed sources (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import canvas as K
from rytp.render.ffmpeg import RenderError

from tests.render_fakes import RecordingRunner

PROBE_JSON = """\
{"streams": [{"width": 1280, "height": 720, "r_frame_rate": "30000/1001",
              "sample_aspect_ratio": "1:1"}]}
"""

WIDE = K.SourceGeometry(width=1280, height=720, fps=25.0)
FOUR_THREE = K.SourceGeometry(width=640, height=480, fps=25.0)
TALL_WIDE = K.SourceGeometry(width=1920, height=1080, fps=50.0)
CINEMA = K.SourceGeometry(width=1280, height=544, fps=25.0)


def test_parse_probe_json_reads_size_rate_and_pixel_aspect() -> None:
    geometry = K.parse_probe_json(PROBE_JSON)
    assert (geometry.width, geometry.height) == (1280, 720)
    assert geometry.fps == pytest.approx(29.97, abs=0.01)
    assert geometry.sar == pytest.approx(1.0)


def test_parse_probe_json_treats_an_unknown_pixel_aspect_as_square() -> None:
    geometry = K.parse_probe_json(
        '{"streams": [{"width": 720, "height": 576, "r_frame_rate": "25/1",'
        ' "sample_aspect_ratio": "0:1"}]}'
    )
    assert geometry.sar == pytest.approx(1.0)


def test_parse_probe_json_honours_an_anamorphic_pixel_aspect() -> None:
    geometry = K.parse_probe_json(
        '{"streams": [{"width": 720, "height": 576, "r_frame_rate": "25/1",'
        ' "sample_aspect_ratio": "16:11"}]}'
    )
    assert geometry.display_width == 1047
    assert geometry.aspect == pytest.approx(1047 / 576)


def test_parse_probe_json_without_a_video_stream_is_an_error() -> None:
    with pytest.raises(RenderError, match="no video stream"):
        K.parse_probe_json('{"streams": []}')


def test_probe_geometry_runs_ffprobe_once_and_parses_its_stdout() -> None:
    runner = RecordingRunner(stdout=PROBE_JSON, make_outputs=False)
    geometry = K.probe_geometry(Path("media/3/video-720.mp4"), runner=runner, binary="ffprobe")
    assert geometry.height == 720
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "ffprobe"


def test_the_default_canvas_is_sixteen_by_nine() -> None:
    canvas = K.plan_canvas([FOUR_THREE, WIDE], mode="16:9")
    assert (canvas.width, canvas.height) == (1280, 720)
    assert canvas.mode == "16:9"


def test_the_default_canvas_height_is_the_tallest_source() -> None:
    canvas = K.plan_canvas([WIDE, TALL_WIDE], mode="16:9")
    assert canvas.height == 1080


def test_the_canvas_height_is_capped() -> None:
    huge = K.SourceGeometry(width=3840, height=2160, fps=25.0)
    assert K.plan_canvas([huge], mode="16:9").height == C.RENDER_MAX_CANVAS_HEIGHT


def test_an_explicit_height_wins_and_is_made_even() -> None:
    canvas = K.plan_canvas([WIDE], mode="16:9", height=721)
    assert canvas.height == 722
    assert canvas.width % 2 == 0


def test_bbox_takes_the_widest_aspect_so_that_source_fills_the_frame() -> None:
    canvas = K.plan_canvas([FOUR_THREE, CINEMA], mode="bbox")
    # Tallest source is the 4:3 one at 480; widest aspect is 1280x544.
    assert canvas.height == 480
    assert canvas.width == 2 * round(480 * (1280 / 544) / 2)
    assert canvas.mode == "bbox"


def test_bbox_and_the_default_agree_when_every_source_is_sixteen_by_nine() -> None:
    assert K.plan_canvas([WIDE, TALL_WIDE], mode="bbox") == K.plan_canvas(
        [WIDE, TALL_WIDE], mode="16:9"
    )


def test_an_unknown_canvas_mode_names_the_known_ones() -> None:
    with pytest.raises(RenderError, match="bbox"):
        K.plan_canvas([WIDE], mode="square")


def test_no_sources_is_an_error_rather_than_a_default_canvas() -> None:
    with pytest.raises(RenderError, match="no sources"):
        K.plan_canvas([], mode="16:9")


def test_the_frame_rate_is_the_commonest_source_rate() -> None:
    assert K.choose_fps([WIDE, FOUR_THREE, TALL_WIDE]) == 25


def test_a_tie_on_frame_rate_takes_the_higher() -> None:
    assert K.choose_fps([WIDE, TALL_WIDE]) == 50


def test_ntsc_rates_round_to_whole_frames() -> None:
    ntsc = K.SourceGeometry(width=640, height=480, fps=29.97)
    film = K.SourceGeometry(width=640, height=480, fps=23.976)
    assert K.choose_fps([ntsc]) == 30
    assert K.choose_fps([film]) == 24


def test_the_frame_rate_is_capped_and_has_a_fallback() -> None:
    fast = K.SourceGeometry(width=640, height=480, fps=240.0)
    assert K.choose_fps([fast]) == C.RENDER_MAX_FPS
    assert K.choose_fps([]) == C.RENDER_FALLBACK_FPS
    assert K.choose_fps([WIDE], override=60) == 60


def test_the_video_filter_scales_inside_the_canvas_and_pads_the_rest() -> None:
    chain = K.video_filter_chain(K.Canvas(width=1280, height=720, fps=25, mode="16:9"))
    assert "scale=1280:720:force_original_aspect_ratio=decrease" in chain
    assert "force_divisible_by=2" in chain
    assert f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:color={C.RENDER_PAD_COLOR}" in chain
    assert "setsar=1" in chain
    assert "fps=25" in chain
    assert f"format={C.RENDER_PIXEL_FORMAT}" in chain
    assert "tpad" not in chain


def test_a_gap_freezes_the_last_frame_after_the_rate_conversion() -> None:
    chain = K.video_filter_chain(
        K.Canvas(width=1280, height=720, fps=25, mode="16:9"), gap_ms=240
    )
    assert chain.endswith("tpad=stop_mode=clone:stop_duration=0.240")
    assert chain.index("fps=25") < chain.index("tpad")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_canvas.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render.canvas'`.

- [ ] **Step 3: Write `rytp/render/canvas.py`**

```python
"""Output geometry for a mixed-shape archive. design §9.

A decade of uploads means 4:3 and 16:9 at several heights inside one
render, so every fragment is scaled into one canvas and padded. The two
modes differ only in how the canvas aspect is chosen; both keep every
pixel of every source (``force_original_aspect_ratio=decrease`` never
crops) and both pad with a flat colour.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.render.ffmpeg import RenderError, Runner, probe_command, run_command, seconds


@dataclass(frozen=True)
class SourceGeometry:
    """One source video's shape, as ffprobe reports it."""

    width: int
    height: int
    fps: float
    sar: float = 1.0

    @property
    def display_width(self) -> int:
        """Width after the pixel aspect ratio, which is what the eye sees."""
        return max(2, round(self.width * self.sar))

    @property
    def aspect(self) -> float:
        return self.display_width / self.height


@dataclass(frozen=True)
class Canvas:
    """The one shape every fragment is scaled and padded into."""

    width: int
    height: int
    fps: int
    mode: str

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps} fps ({self.mode})"


def _even(value: float) -> int:
    """Nearest even integer >= 2. yuv420p needs both dimensions even."""
    return max(2, 2 * round(value / 2))


def _ratio(text: str | None, *, default: float) -> float:
    """Parse ffprobe's ``"30000/1001"`` / ``"16:11"`` fractions."""
    if not text:
        return default
    for separator in ("/", ":"):
        if separator in text:
            head, _, tail = text.partition(separator)
            try:
                numerator, denominator = float(head), float(tail)
            except ValueError:
                return default
            if denominator <= 0 or numerator <= 0:
                return default
            return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return default


def parse_probe_json(text: str) -> SourceGeometry:
    """Turn one ffprobe JSON reply into a :class:`SourceGeometry`."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise RenderError(f"ffprobe returned no usable JSON: {text[:120]!r}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise RenderError("ffprobe found no video stream in the source")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError("ffprobe reported no video stream size") from exc
    if width <= 0 or height <= 0:
        raise RenderError(f"ffprobe reported a {width}x{height} video stream")
    return SourceGeometry(
        width=width,
        height=height,
        fps=_ratio(stream.get("r_frame_rate"), default=float(C.RENDER_FALLBACK_FPS)),
        # "0:1" means "unknown"; square pixels are the only safe reading.
        sar=_ratio(stream.get("sample_aspect_ratio"), default=1.0),
    )


def probe_geometry(
    path: Path, *, runner: Runner | None = None, binary: str | None = None
) -> SourceGeometry:
    """Probe one source file. Call once per source video, not per fragment."""
    result = run_command(
        probe_command(path, binary=binary),
        runner=runner,
        what=f"ffprobe {path.name}",
        timeout_s=C.RENDER_FFPROBE_TIMEOUT_S,
    )
    return parse_probe_json(result.stdout)


def choose_fps(geometries: Sequence[SourceGeometry], *, override: int = 0) -> int:
    """The output frame rate: the commonest source rate, ties to the higher.

    Measured rather than assumed. This archive is European, so 25 is the
    likely majority, and forcing it to 30 duplicates every fifth frame —
    visible on a talking head, and not worth the file size.
    """
    if override > 0:
        return min(override, C.RENDER_MAX_FPS)
    if not geometries:
        return C.RENDER_FALLBACK_FPS
    counts = Counter(max(1, round(g.fps)) for g in geometries)
    best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return min(best, C.RENDER_MAX_FPS)


def plan_canvas(
    geometries: Sequence[SourceGeometry],
    *,
    mode: str = C.RENDER_DEFAULT_CANVAS_MODE,
    height: int = 0,
    fps: int = 0,
) -> Canvas:
    """Decide the output shape for a set of sources.

    ``16:9`` — the default — is a fixed 16:9 canvas; 4:3 sources are
    pillarboxed into it. ``bbox`` is the smallest rectangle every source
    fits inside without cropping: the tallest source's height, the widest
    source's aspect. The widest source then fills the frame exactly, and
    with an all-16:9 corpus the two modes give the same canvas.
    """
    if mode not in C.RENDER_CANVAS_MODES:
        raise RenderError(
            f"unknown canvas mode {mode!r}; available: {', '.join(C.RENDER_CANVAS_MODES)}"
        )
    if not geometries:
        raise RenderError("cannot plan a canvas with no sources")
    if height > 0:
        canvas_height = _even(height)
    else:
        tallest = max(g.height for g in geometries)
        canvas_height = _even(min(tallest, C.RENDER_MAX_CANVAS_HEIGHT))
    canvas_height = max(canvas_height, C.RENDER_MIN_CANVAS_HEIGHT)
    if mode == "bbox":
        aspect = max(g.aspect for g in geometries)
    else:
        wide, tall = C.RENDER_DEFAULT_ASPECT
        aspect = wide / tall
    return Canvas(
        width=_even(canvas_height * aspect),
        height=canvas_height,
        fps=choose_fps(geometries, override=fps),
        mode=mode,
    )


def video_filter_chain(canvas: Canvas, *, gap_ms: int = 0) -> str:
    """The per-fragment video chain: fit, pad, normalise, optionally freeze.

    ``force_divisible_by=2`` matters: an odd intermediate width with
    yuv420p is rejected by the encoder, and a 4:3 source scaled into a
    16:9 canvas hits odd widths constantly.

    A gap is a freeze-frame (design §9), produced by ``tpad`` cloning the
    last frame of *this* fragment rather than by a separate filler
    segment — one fewer encode per seam, and the frozen video cannot
    drift away from the silence padded onto the same fragment's audio.
    ``tpad`` comes after ``fps`` so the held frames are at the output rate.
    """
    parts = [
        f"scale={canvas.width}:{canvas.height}"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2",
        f"pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2"
        f":color={C.RENDER_PAD_COLOR}",
        "setsar=1",
        f"fps={canvas.fps}",
        f"format={C.RENDER_PIXEL_FORMAT}",
    ]
    if gap_ms > 0:
        parts.append(f"tpad=stop_mode=clone:stop_duration={seconds(gap_ms)}")
    return ",".join(parts)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_canvas.py -q`
Expected: PASS, 18 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/canvas.py tests/test_render_canvas.py
git commit -m "feat: choose one canvas for mixed-shape sources"
```

---

### Task 3: Fragment, gap and concat command construction

The rest of the ffmpeg surface: one command per fragment, one list file, and the final pass. Six things here are easy to get subtly wrong, so each has its own test:

- **Video comes from the rendition asset and audio from the audio asset** (design §4: they are deliberately never merged), so a fragment normally has *two* inputs seeked independently to the same time range. A local file registered as a single `container` asset is the exception: both paths are the same file, one input, and the map indices shift.
- **The intermediate is Matroska with PCM audio.** AAC priming would slide the audio later at every seam; see "Decisions" at the top.
- **The concat demuxer's list file has its own escaping.** Backslashes are escape characters to it, so paths go in as `as_posix()`, and a single quote in a filename has to become `'\''`. `-safe 0` is required for absolute paths.
- **`alimiter` auto-levels by default.** `alimiter=limit=X` on its own *raises* quiet audio to the ceiling, which would undo the per-source gain matching entirely. `level=disabled` is not optional.
- **The measuring passes must not be quiet.** Every encoding command runs at `-loglevel error`, but ffmpeg prints loudnorm's `print_format=json` block at INFO — silence the log and the two-pass recipe silently measures nothing. The measuring commands therefore use their own base arguments.
- **Binaries are resolved once**, into a `Tools` record, so a render fails before it starts rather than four fragments in — and so the orchestration tests can run on a machine with no ffmpeg at all.

**Files:**
- Modify: `rytp/render/ffmpeg.py` (append; do not edit Task 1's part)
- Test: `tests/test_render_ffmpeg_commands.py`

**Interfaces:**
- Consumes: Task 1's `seconds`, `ffmpeg_binary`, `ffprobe_binary`, `LoudnormMeasurement`, `RenderError`.
- Produces:
  - `Tools(ffmpeg: str, ffprobe: str, runner: Runner | None = None)` with `Tools.resolve(runner=None)` and `Tools.faked(runner)`
  - `audio_filter_chain(*, gain_db: float | None = None, gap_ms: int = 0) -> str`
  - `fragment_command(*, video_path: Path, audio_path: Path, start_ms: int, end_ms: int, video_filter: str, out_path: Path, gain_db: float | None = None, gap_ms: int = 0, preset: str = C.RENDER_VIDEO_PRESET, crf: int = C.RENDER_VIDEO_CRF, binary: str | None = None) -> list[str]`
  - `concat_list_text(paths: Sequence[Path]) -> str`, `write_concat_list(list_file: Path, paths: Sequence[Path]) -> Path`
  - `source_loudness_command(*, path: Path, start_ms: int, duration_ms: int, binary: str | None = None) -> list[str]`
  - `programme_loudness_command(*, list_file: Path, binary: str | None = None) -> list[str]`
  - `concat_command(*, list_file: Path, out_path: Path, measured: LoudnormMeasurement | None = None, binary: str | None = None) -> list[str]`
  - `gain_for(measured_lufs: float | None) -> float | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_ffmpeg_commands.py`:

```python
"""Every ffmpeg argument list Part 6 builds (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.render import ffmpeg as F

VIDEO = Path("/media/3/video-720.mp4")
AUDIO = Path("/media/3/audio.m4a")
OUT = Path("/out/r1/fragments/0000.mkv")
VFILTER = "scale=1280:720:force_original_aspect_ratio=decrease,setsar=1,fps=25"


def _pairs(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_a_fragment_seeks_both_inputs_to_the_same_range() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=612_340, end_ms=613_100,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-i") == [str(VIDEO), str(AUDIO)]
    assert _pairs(args, "-ss") == ["612.340", "612.340"]
    assert _pairs(args, "-t") == ["0.760", "0.760"]


def test_a_fragment_takes_video_from_the_rendition_and_audio_from_the_audio_asset() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    graph = args[args.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]")
    assert "[1:a]" in graph
    assert _pairs(args, "-map") == ["[v]", "[a]"]


def test_a_container_source_is_opened_once_and_mapped_twice() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=VIDEO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-i") == [str(VIDEO)]
    graph = args[args.index("-filter_complex") + 1]
    assert "[0:a]" in graph and "[1:a]" not in graph


def test_the_intermediate_is_matroska_with_pcm_audio() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(args, "-c:a") == [C.RENDER_INTERMEDIATE_AUDIO_CODEC]
    assert _pairs(args, "-f") == [C.RENDER_INTERMEDIATE_FORMAT]
    assert _pairs(args, "-ar") == [str(C.RENDER_AUDIO_RATE_HZ)]
    assert _pairs(args, "-ac") == [str(C.RENDER_AUDIO_CHANNELS)]
    assert args[-1] == str(OUT)


def test_encoder_settings_are_overridable_per_render() -> None:
    args = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=1_000,
        video_filter=VFILTER, out_path=OUT, preset="ultrafast", crf=30,
        binary="ffmpeg",
    )
    assert _pairs(args, "-preset") == ["ultrafast"]
    assert _pairs(args, "-crf") == ["30"]


def test_an_empty_fragment_is_refused() -> None:
    with pytest.raises(F.RenderError, match="not positive"):
        F.fragment_command(
            video_path=VIDEO, audio_path=AUDIO, start_ms=1_000, end_ms=1_000,
            video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
        )


def test_tools_can_be_faked_without_either_binary_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    tools = F.Tools.faked(lambda args: F.CompletedRun(tuple(args), 0, "", ""))
    assert (tools.ffmpeg, tools.ffprobe) == ("ffmpeg", "ffprobe")
    assert tools.runner is not None


def test_resolving_tools_without_ffmpeg_is_a_one_line_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(F.shutil, "which", lambda _name: None)
    with pytest.raises(F.FfmpegNotFoundError, match="ffmpeg.org"):
        F.Tools.resolve()


def test_the_audio_chain_is_a_plain_resample_when_nothing_is_asked_of_it() -> None:
    chain = F.audio_filter_chain()
    assert chain == (
        f"aresample={C.RENDER_AUDIO_RATE_HZ},"
        "aformat=sample_fmts=s16:channel_layouts=stereo"
    )


def test_a_gain_is_applied_with_a_limiter_that_does_not_auto_level() -> None:
    chain = F.audio_filter_chain(gain_db=3.25)
    assert "volume=3.25dB" in chain
    assert f"alimiter=limit={C.RENDER_LIMITER_CEILING}:level=disabled" in chain


def test_a_zero_gain_adds_no_filter() -> None:
    assert "volume" not in F.audio_filter_chain(gain_db=0.0)


def test_a_gap_pads_the_audio_by_exactly_the_freeze_duration() -> None:
    assert F.audio_filter_chain(gap_ms=240).endswith("apad=pad_dur=0.240")


def test_gain_for_is_the_distance_to_target_clamped() -> None:
    assert F.gain_for(None) is None
    assert F.gain_for(-16.0) == pytest.approx(0.0)
    assert F.gain_for(-23.0) == pytest.approx(7.0)
    assert F.gain_for(-60.0) == pytest.approx(C.RENDER_MAX_GAIN_DB)
    assert F.gain_for(0.0) == pytest.approx(-C.RENDER_MAX_GAIN_DB)


def test_the_concat_list_uses_forward_slashes_and_escapes_quotes() -> None:
    text = F.concat_list_text([Path("/out/r1/a b.mkv"), Path("/out/r1/it's.mkv")])
    assert text.splitlines()[0] == "file '/out/r1/a b.mkv'"
    assert text.splitlines()[1] == "file '/out/r1/it'\\''s.mkv'"


def test_write_concat_list_creates_parents_and_ends_with_a_newline(
    tmp_path: Path,
) -> None:
    listing = F.write_concat_list(tmp_path / "deep" / "concat.txt", [tmp_path / "a.mkv"])
    assert listing.read_text(encoding="utf-8").endswith("\n")


def test_the_source_loudness_scan_reads_one_window_with_no_video() -> None:
    args = F.source_loudness_command(
        path=AUDIO, start_ms=600_000, duration_ms=120_000, binary="ffmpeg"
    )
    assert _pairs(args, "-ss") == ["600.000"]
    assert _pairs(args, "-t") == ["120.000"]
    assert "-vn" in args
    assert "print_format=json" in args[args.index("-af") + 1]
    assert args[-3:] == ["-f", "null", "-"]


def test_the_programme_scan_reads_the_concat_list() -> None:
    args = F.programme_loudness_command(list_file=Path("/out/r1/concat.txt"),
                                        binary="ffmpeg")
    assert _pairs(args, "-f")[0] == "concat"
    assert "-safe" in args and args[args.index("-safe") + 1] == "0"
    assert "print_format=json" in args[args.index("-af") + 1]


def test_the_measuring_passes_are_not_silenced() -> None:
    """loudnorm prints its JSON at INFO; -loglevel error would eat it."""
    for args in (
        F.source_loudness_command(
            path=AUDIO, start_ms=0, duration_ms=1_000, binary="ffmpeg"
        ),
        F.programme_loudness_command(list_file=Path("/c.txt"), binary="ffmpeg"),
    ):
        assert _pairs(args, "-loglevel") == ["info"]
    encode = F.fragment_command(
        video_path=VIDEO, audio_path=AUDIO, start_ms=0, end_ms=500,
        video_filter=VFILTER, out_path=OUT, binary="ffmpeg",
    )
    assert _pairs(encode, "-loglevel") == ["error"]


def test_the_final_pass_copies_video_and_always_re_encodes_audio() -> None:
    args = F.concat_command(
        list_file=Path("/out/r1/concat.txt"),
        out_path=Path("/out/r1/output.mp4"),
        binary="ffmpeg",
    )
    assert _pairs(args, "-c:v") == ["copy"]
    assert _pairs(args, "-c:a") == [C.RENDER_AUDIO_CODEC]
    assert "-af" not in args
    assert _pairs(args, "-movflags") == ["+faststart"]
    assert args[-1] == str(Path("/out/r1/output.mp4"))


def test_the_final_pass_feeds_the_measurement_back_in_for_the_second_pass() -> None:
    measured = F.LoudnormMeasurement(
        input_i=-27.24, input_tp=-8.51, input_lra=6.90,
        input_thresh=-37.51, target_offset=0.02,
    )
    args = F.concat_command(
        list_file=Path("/out/r1/concat.txt"),
        out_path=Path("/out/r1/output.mp4"),
        measured=measured,
        binary="ffmpeg",
    )
    graph = args[args.index("-af") + 1]
    assert f"I={C.RENDER_TARGET_LUFS}" in graph
    assert f"TP={C.RENDER_TRUE_PEAK_DBTP}" in graph
    assert f"LRA={C.RENDER_LOUDNESS_RANGE_LU}" in graph
    assert "measured_I=-27.24" in graph
    assert "measured_TP=-8.51" in graph
    assert "measured_LRA=6.9" in graph
    assert "measured_thresh=-37.51" in graph
    assert "offset=0.02" in graph
    assert "linear=true" in graph
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_ffmpeg_commands.py -q`
Expected: collection error, `AttributeError: module 'rytp.render.ffmpeg' has no attribute 'fragment_command'`.

- [ ] **Step 3: Append to `rytp/render/ffmpeg.py`**

Widen the existing import to `from collections.abc import Callable, Sequence`.

```python
# ---------------------------------------------------------------------------
# Command construction. Everything below returns a list[str] and runs nothing.
# ---------------------------------------------------------------------------

#: Quiet, non-interactive, overwrite. Used by everything that encodes.
_BASE: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "error", "-y")

#: The measuring passes must NOT be quiet. ffmpeg prints loudnorm's
#: ``print_format=json`` block at INFO level, so ``-loglevel error``
#: would silence the very thing those commands exist to read.
_MEASURE_BASE: tuple[str, ...] = ("-nostdin", "-hide_banner", "-loglevel", "info", "-y")


@dataclass(frozen=True)
class Tools:
    """The two binaries plus the runner, resolved once per render.

    Looking them up once matters for the test suite as much as for
    speed: :meth:`faked` produces a ``Tools`` that never calls
    ``shutil.which``, which is what lets the whole orchestration suite
    run on a machine with no ffmpeg installed.
    """

    ffmpeg: str
    ffprobe: str
    runner: Runner | None = None

    @classmethod
    def resolve(cls, *, runner: Runner | None = None) -> Tools:
        """Find both binaries now, so a missing one fails before any work."""
        return cls(ffmpeg=ffmpeg_binary(), ffprobe=ffprobe_binary(), runner=runner)

    @classmethod
    def faked(cls, runner: Runner) -> Tools:
        """Tools that name the binaries but never look for them."""
        return cls(ffmpeg="ffmpeg", ffprobe="ffprobe", runner=runner)


def gain_for(measured_lufs: float | None) -> float | None:
    """How many dB to move one source to the programme target.

    ``None`` in, ``None`` out: an unmeasurable source is left alone
    rather than guessed at. The clamp is design-level, not cosmetic — a
    source needing more than :data:`C.RENDER_MAX_GAIN_DB` is damaged, and
    lifting it that far lifts its noise floor into the mix with it.
    """
    if measured_lufs is None:
        return None
    delta = C.RENDER_TARGET_LUFS - measured_lufs
    return max(-C.RENDER_MAX_GAIN_DB, min(C.RENDER_MAX_GAIN_DB, delta))


def audio_filter_chain(*, gain_db: float | None = None, gap_ms: int = 0) -> str:
    """The per-fragment audio chain.

    Always resamples to the programme's rate and layout, so the concat
    demuxer never meets a parameter change. ``alimiter`` carries
    ``level=disabled`` deliberately: its default auto-level *raises*
    quiet material to the ceiling, which would throw away the
    source-to-source matching the gain just established.

    ``apad`` matches the video chain's ``tpad``, so a freeze-frame gap
    and its silence are the same length by construction.
    """
    parts = [
        f"aresample={C.RENDER_AUDIO_RATE_HZ}",
        "aformat=sample_fmts=s16:channel_layouts=stereo",
    ]
    if gain_db:
        parts.append(f"volume={gain_db:g}dB")
        parts.append(f"alimiter=limit={C.RENDER_LIMITER_CEILING}:level=disabled")
    if gap_ms > 0:
        parts.append(f"apad=pad_dur={seconds(gap_ms)}")
    return ",".join(parts)


def fragment_command(
    *,
    video_path: Path,
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    video_filter: str,
    out_path: Path,
    gain_db: float | None = None,
    gap_ms: int = 0,
    preset: str = C.RENDER_VIDEO_PRESET,
    crf: int = C.RENDER_VIDEO_CRF,
    binary: str | None = None,
) -> list[str]:
    """Cut one fragment into a concat-safe intermediate.

    Video comes from a rendition asset and audio from the audio asset —
    design §4 keeps them apart so a rendition can be upgraded later — so
    there are normally two inputs, seeked identically. A local file
    registered as a single ``container`` asset passes the same path
    twice and is opened once.

    ``-ss`` precedes each ``-i``, which is both the fast seek and an
    accurate one here: the output is re-encoded, so ffmpeg decodes from
    the preceding keyframe and discards the excess.
    """
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        raise RenderError(f"fragment duration is not positive: {start_ms}..{end_ms} ms")
    single_input = video_path == audio_path
    args = [binary or ffmpeg_binary(), *_BASE]
    args += ["-ss", seconds(start_ms), "-t", seconds(duration_ms), "-i", str(video_path)]
    if not single_input:
        args += [
            "-ss", seconds(start_ms), "-t", seconds(duration_ms), "-i", str(audio_path)
        ]
    audio_input = 0 if single_input else 1
    graph = (
        f"[0:v]{video_filter}[v];"
        f"[{audio_input}:a]{audio_filter_chain(gain_db=gain_db, gap_ms=gap_ms)}[a]"
    )
    args += [
        "-filter_complex", graph,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", C.RENDER_VIDEO_CODEC,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", C.RENDER_PIXEL_FORMAT,
        "-c:a", C.RENDER_INTERMEDIATE_AUDIO_CODEC,
        "-ar", str(C.RENDER_AUDIO_RATE_HZ),
        "-ac", str(C.RENDER_AUDIO_CHANNELS),
        "-f", C.RENDER_INTERMEDIATE_FORMAT,
        str(out_path),
    ]
    return args


def concat_list_text(paths: Sequence[Path]) -> str:
    """The concat demuxer's list file.

    Forward slashes even on Windows — the demuxer treats a backslash as
    an escape character — and a literal single quote in a filename has
    to leave and re-enter the quoting, which is the ``'\\''`` dance.
    """
    lines = []
    for path in paths:
        quoted = path.as_posix().replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    return "\n".join(lines) + "\n"


def write_concat_list(list_file: Path, paths: Sequence[Path]) -> Path:
    """Write the list file, creating its directory. Returns the path."""
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text(concat_list_text(paths), encoding="utf-8")
    return list_file


def _measure_filter() -> str:
    return (
        f"loudnorm=I={C.RENDER_TARGET_LUFS}"
        f":TP={C.RENDER_TRUE_PEAK_DBTP}"
        f":LRA={C.RENDER_LOUDNESS_RANGE_LU}"
        ":print_format=json"
    )


def _apply_filter(measured: LoudnormMeasurement) -> str:
    """The second pass, with the first pass's numbers fed back in.

    ``linear=true`` asks for one static gain over the whole programme,
    which is what keeps the cut sounding like a recording rather than a
    compressor. ffmpeg silently falls back to dynamic mode when the
    measured range exceeds the target LRA — which is why
    :data:`C.RENDER_LOUDNESS_RANGE_LU` is 13 rather than the broadcast 11.
    """
    return (
        f"loudnorm=I={C.RENDER_TARGET_LUFS}"
        f":TP={C.RENDER_TRUE_PEAK_DBTP}"
        f":LRA={C.RENDER_LOUDNESS_RANGE_LU}"
        f":measured_I={measured.input_i:g}"
        f":measured_TP={measured.input_tp:g}"
        f":measured_LRA={measured.input_lra:g}"
        f":measured_thresh={measured.input_thresh:g}"
        f":offset={measured.target_offset:g}"
        ":linear=true"
    )


def source_loudness_command(
    *, path: Path, start_ms: int, duration_ms: int, binary: str | None = None
) -> list[str]:
    """Measure one window of one source, to place its gain."""
    return [
        binary or ffmpeg_binary(),
        *_MEASURE_BASE,
        "-ss", seconds(max(0, start_ms)),
        "-t", seconds(duration_ms),
        "-i", str(path),
        "-vn",
        "-af", _measure_filter(),
        "-f", "null",
        "-",
    ]


def programme_loudness_command(
    *, list_file: Path, binary: str | None = None
) -> list[str]:
    """Measure the assembled programme — the first of the two passes."""
    return [
        binary or ffmpeg_binary(),
        *_MEASURE_BASE,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-vn",
        "-af", _measure_filter(),
        "-f", "null",
        "-",
    ]


def concat_command(
    *,
    list_file: Path,
    out_path: Path,
    measured: LoudnormMeasurement | None = None,
    binary: str | None = None,
) -> list[str]:
    """Join the intermediates into the deliverable.

    Video is copied — every intermediate was already encoded to the same
    canvas, rate and pixel format, so there is nothing left to do to it.
    Audio is always re-encoded, with or without ``loudnorm``: PCM is what
    made the seams safe, and an MP4 wants AAC.
    """
    args = [
        binary or ffmpeg_binary(),
        *_BASE,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c:v", "copy",
    ]
    if measured is not None:
        args += ["-af", _apply_filter(measured)]
    args += [
        "-c:a", C.RENDER_AUDIO_CODEC,
        "-b:a", C.RENDER_AUDIO_BITRATE,
        "-ar", str(C.RENDER_AUDIO_RATE_HZ),
        "-ac", str(C.RENDER_AUDIO_CHANNELS),
        "-movflags", "+faststart",
        str(out_path),
    ]
    return args
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_ffmpeg_commands.py -q`
Expected: PASS, 20 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/ffmpeg.py tests/test_render_ffmpeg_commands.py
git commit -m "feat: build the fragment, loudness and concat commands"
```

---

### Task 4: Pause statistics from aligned words

Design §9: *"Gap length: configurable. Defaults to natural between-word pauses measured from aligned transcripts — per speaker where the video is diarized, per video otherwise. Disableable if it doesn't sound right."* The owner called this a tall ask and asked that it be switchable, so it lives behind one function and one flag.

**The trap this task exists to avoid.** Design §6, measured on the owner's real data: **78.7% of Whisper's word gaps are exactly zero**, because the transcriber sets `word[i].end == word[i+1].start` and absorbs every pause into an adjacent word. A naive median over that learns that the median pause is zero, and the feature then does nothing while appearing to work. Two defences, both explicit:

1. Only `source = 'aligned'` rows are measured. Caption rows have no `end_ms` at all (contracts §3), so they cannot contribute; aligned rows have boundaries placed at measured energy minima (design §6) and are the only rows where a gap means anything.
2. The distribution is *inspected* before it is believed. Too few samples, a majority of exact zeros, or a zero median all mark it degenerate, and a degenerate distribution falls back to a constant and **says so** in the report rather than silently rendering nothing.

Scoping follows the design: pooled across the corpus for a named speaker, per diarized label when the video has labels but no roster entry, per video otherwise.

**Files:**
- Create: `rytp/render/pauses.py`
- Test: `tests/test_render_pauses.py`

**Interfaces:**
- Consumes: `rytp.db.Database`; `rytp.constants.{PAUSE_*, ALIGNED_WORD_SOURCE}`.
- Produces:
  - `PauseStats(scope: str, key: str, n_samples: int, median_ms: int, zero_fraction: float, degenerate: bool, reason: str)` with `.description`
  - `summarize_gaps(samples: Sequence[int], *, scope: str, key: str) -> PauseStats`
  - `gaps_for_video(db, video_id) -> list[int]`, `gaps_for_video_speaker(db, video_speaker_id) -> list[int]`, `gaps_for_speaker_label(db, label) -> list[int]`
  - `measure_pause_stats(db, *, video_id: int, video_speaker_id: int | None = None, speaker_label: str | None = None) -> PauseStats`
  - `applied_gap_ms(stats: PauseStats) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_pauses.py`:

```python
"""Between-word pause statistics and the zero-inflation guard (design §9)."""

from __future__ import annotations

from rytp import constants as C
from rytp.db import Database
from rytp.render import pauses as P

from tests.fakes import make_video


def add_words(
    db: Database,
    video_id: int,
    spans: list[tuple[int, int]],
    *,
    source: str = "aligned",
    video_speaker_id: int | None = None,
    first_ord: int = 0,
) -> None:
    """Insert words at the given (start_ms, end_ms) spans, ords contiguous."""
    rows = [
        (
            video_id, first_ord + i, start, end if source == "aligned" else None,
            "х", "х", "х", source, "fake", video_speaker_id,
        )
        for i, (start, end) in enumerate(spans)
    ]
    db.conn.executemany(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
        "stem, source, engine, video_speaker_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.conn.commit()


def add_speaker(db: Database, video_id: int, *, label: str, local: str = "SPEAKER_00") -> int:
    """Map one diarized label of one video to a roster person, idempotently."""
    db.conn.execute(
        "INSERT OR IGNORE INTO speakers (label, created_at) "
        "VALUES (?, '2026-01-01T00:00:00+00:00')",
        (label,),
    )
    speaker_id = db.conn.execute(
        "SELECT id FROM speakers WHERE label = ?", (label,)
    ).fetchone()[0]
    cursor = db.conn.execute(
        # `engine` is NOT NULL (contracts §3): a label always knows which
        # diarizer produced it, fixtures included.
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine) "
        "VALUES (?, ?, ?, 'fake')",
        (video_id, local, speaker_id),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def evenly_spaced(n: int, *, word_ms: int = 300, gap_ms: int = 200) -> list[tuple[int, int]]:
    spans = []
    clock = 0
    for _ in range(n):
        spans.append((clock, clock + word_ms))
        clock += word_ms + gap_ms
    return spans


def zero_gapped(n: int, *, word_ms: int = 300) -> list[tuple[int, int]]:
    return [(i * word_ms, (i + 1) * word_ms) for i in range(n)]


def test_a_healthy_distribution_takes_the_median(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(100, gap_ms=200))
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is False
    assert stats.median_ms == 200
    assert stats.scope == "video"
    assert stats.n_samples == 99


def test_a_whisper_shaped_distribution_is_refused(db: Database) -> None:
    """Design §6: 78.7% of gaps at exactly zero. The median must not be believed."""
    vid = make_video(db)
    tail_base = 80 * 300
    spans = zero_gapped(80) + [
        (tail_base + 250 * i, tail_base + 250 * i + 200) for i in range(20)
    ]
    add_words(db, vid, spans)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is True
    assert stats.zero_fraction > C.PAUSE_DEGENERATE_ZERO_FRACTION
    assert "zero" in stats.reason
    assert P.applied_gap_ms(stats) == C.PAUSE_FALLBACK_GAP_MS


def test_too_few_samples_is_degenerate_and_says_so(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(5))
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.degenerate is True
    assert "samples" in stats.reason
    assert P.applied_gap_ms(stats) == C.PAUSE_FALLBACK_GAP_MS


def test_caption_words_are_never_measured(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(100), source="caption")
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 0
    assert stats.degenerate is True


def test_turn_length_silences_are_excluded(db: Database) -> None:
    vid = make_video(db)
    spans = evenly_spaced(60, gap_ms=150)
    tail_start = spans[-1][1] + 30_000
    spans.append((tail_start, tail_start + 300))
    add_words(db, vid, spans)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 59  # the 30 s silence is not a between-word pause
    assert stats.median_ms == 150


def test_a_gap_in_the_ordinals_is_not_a_pause(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(60, gap_ms=150))
    add_words(db, vid, evenly_spaced(60, gap_ms=150), first_ord=500)
    stats = P.measure_pause_stats(db, video_id=vid)
    assert stats.n_samples == 118  # 59 within each run, none across the break


def test_a_named_speaker_pools_across_videos(db: Database) -> None:
    first = make_video(db)
    second = make_video(
        db, external_id="VIDEO_B", url="https://example.invalid/watch/VIDEO_B"
    )
    vs_one = add_speaker(db, first, label="host")
    vs_two = add_speaker(db, second, label="host", local="SPEAKER_01")
    add_words(db, first, evenly_spaced(40, gap_ms=120), video_speaker_id=vs_one)
    add_words(db, second, evenly_spaced(40, gap_ms=120), video_speaker_id=vs_two)
    stats = P.measure_pause_stats(db, video_id=first, speaker_label="host")
    assert stats.scope == "speaker"
    assert stats.key == "host"
    assert stats.n_samples == 78
    assert stats.degenerate is False
    assert stats.median_ms == 120


def test_a_diarized_label_with_no_roster_entry_scopes_to_that_label(db: Database) -> None:
    vid = make_video(db)
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake')",
        (vid,),
    )
    db.conn.commit()
    label_id = int(cursor.lastrowid)
    add_words(db, vid, evenly_spaced(60, gap_ms=90), video_speaker_id=label_id)
    add_words(db, vid, evenly_spaced(60, gap_ms=400), first_ord=500)
    stats = P.measure_pause_stats(db, video_id=vid, video_speaker_id=label_id)
    assert stats.scope == "video_speaker"
    assert stats.median_ms == 90


def test_an_unknown_speaker_falls_back_to_the_video(db: Database) -> None:
    vid = make_video(db)
    add_words(db, vid, evenly_spaced(80, gap_ms=210))
    stats = P.measure_pause_stats(db, video_id=vid, speaker_label="nobody")
    assert stats.scope == "video"
    assert stats.median_ms == 210


def test_the_applied_gap_is_clamped_both_ways() -> None:
    tiny = P.PauseStats(
        scope="video", key="1", n_samples=999, median_ms=3,
        zero_fraction=0.0, degenerate=False, reason="",
    )
    huge = P.PauseStats(
        scope="video", key="1", n_samples=999, median_ms=1_900,
        zero_fraction=0.0, degenerate=False, reason="",
    )
    assert P.applied_gap_ms(tiny) == C.PAUSE_MIN_APPLIED_GAP_MS
    assert P.applied_gap_ms(huge) == C.PAUSE_MAX_APPLIED_GAP_MS


def test_summarize_is_deterministic() -> None:
    samples = [100, 200, 300, 400]
    first = P.summarize_gaps(samples, scope="video", key="1")
    second = P.summarize_gaps(samples, scope="video", key="1")
    assert first == second
    assert first.median_ms == 250
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_pauses.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render.pauses'`.

- [ ] **Step 3: Write `rytp/render/pauses.py`**

```python
"""Between-word pause statistics. design §9.

The gap the render inserts between two fragments defaults to how long
this speaker actually pauses between words. That is only measurable on
aligned-tier words: caption rows have no end time at all (contracts §3),
and a transcriber's own timings are worse than useless here — design §6
measured 78.7% of Whisper's word gaps at exactly zero, because it sets
``word[i].end == word[i+1].start`` and absorbs every pause into the
neighbouring word.

So this module does not trust its own input. It measures, then asks
whether the distribution it measured could possibly be real, and says
plainly when it could not. A degenerate distribution falls back to a
constant and the reason reaches the report.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database

_COLUMNS = "video_id, ord, start_ms, end_ms, video_speaker_id"


@dataclass(frozen=True)
class PauseStats:
    """What the corpus says about one speaker's between-word pauses."""

    scope: str  # "speaker" | "video_speaker" | "video"
    key: str  # roster label, label row id, or video id — for the report
    n_samples: int
    median_ms: int
    zero_fraction: float
    degenerate: bool
    reason: str  # empty when the distribution is usable

    @property
    def description(self) -> str:
        """One line for the report's notes section."""
        if self.degenerate:
            return (
                f"{self.scope} {self.key}: {self.reason}; "
                f"fell back to {C.PAUSE_FALLBACK_GAP_MS} ms"
            )
        return (
            f"{self.scope} {self.key}: median {self.median_ms} ms "
            f"from {self.n_samples} gaps"
        )


def _pairwise_gaps(rows: Sequence[sqlite3.Row]) -> list[int]:
    """Gaps between consecutive words of one speaker, in milliseconds.

    A pair counts only when the two words are adjacent ordinals in the
    same video and carry the same speaker label — otherwise the "gap" is
    a turn boundary or the seam of an unrelated passage. Gaps longer
    than :data:`C.PAUSE_MAX_GAP_MS` are silences, not pauses, and
    negative ones are overlapping boundaries; both are dropped.
    """
    gaps: list[int] = []
    previous: sqlite3.Row | None = None
    for row in rows:
        if (
            previous is not None
            and row["video_id"] == previous["video_id"]
            and row["ord"] == previous["ord"] + 1
            and row["video_speaker_id"] == previous["video_speaker_id"]
            and previous["end_ms"] is not None
        ):
            gap = int(row["start_ms"]) - int(previous["end_ms"])
            if 0 <= gap <= C.PAUSE_MAX_GAP_MS:
                gaps.append(gap)
        previous = row
    return gaps


def gaps_for_video(db: Database, video_id: int) -> list[int]:
    """Every between-word gap in one video's aligned words."""
    rows = db.conn.execute(
        f"SELECT {_COLUMNS} FROM words "
        "WHERE video_id = ? AND source = ? ORDER BY ord LIMIT ?",
        (video_id, C.ALIGNED_WORD_SOURCE, C.PAUSE_SAMPLE_LIMIT),
    ).fetchall()
    return _pairwise_gaps(rows)


def gaps_for_video_speaker(db: Database, video_speaker_id: int) -> list[int]:
    """Gaps for one diarized label of one video."""
    rows = db.conn.execute(
        f"SELECT {_COLUMNS} FROM words "
        "WHERE video_speaker_id = ? AND source = ? ORDER BY video_id, ord LIMIT ?",
        (video_speaker_id, C.ALIGNED_WORD_SOURCE, C.PAUSE_SAMPLE_LIMIT),
    ).fetchall()
    return _pairwise_gaps(rows)


def gaps_for_speaker_label(db: Database, label: str) -> list[int]:
    """Gaps for one person, pooled over every video they are mapped in.

    Pooling is what makes the statistic usable: one video rarely holds
    enough aligned words for a speaker to clear
    :data:`C.PAUSE_MIN_SAMPLES`, and a person's speaking rhythm is more
    theirs than the recording's.
    """
    rows = db.conn.execute(
        "SELECT w.video_id, w.ord, w.start_ms, w.end_ms, w.video_speaker_id "
        "FROM words w "
        "JOIN video_speakers vs ON vs.id = w.video_speaker_id "
        "JOIN speakers s ON s.id = vs.speaker_id "
        "WHERE s.label = ? AND w.source = ? "
        "ORDER BY w.video_id, w.ord LIMIT ?",
        (label, C.ALIGNED_WORD_SOURCE, C.PAUSE_SAMPLE_LIMIT),
    ).fetchall()
    return _pairwise_gaps(rows)


def summarize_gaps(samples: Sequence[int], *, scope: str, key: str) -> PauseStats:
    """Reduce measured gaps to one number, refusing when they are junk."""
    n = len(samples)
    if n == 0:
        return PauseStats(
            scope=scope, key=key, n_samples=0, median_ms=0, zero_fraction=1.0,
            degenerate=True, reason="no aligned words to measure",
        )
    zeros = sum(1 for gap in samples if gap <= C.PAUSE_ZERO_GAP_MS)
    zero_fraction = zeros / n
    median_ms = int(round(statistics.median(samples)))
    reason = ""
    if n < C.PAUSE_MIN_SAMPLES:
        reason = f"only {n} samples, fewer than {C.PAUSE_MIN_SAMPLES}"
    elif zero_fraction > C.PAUSE_DEGENERATE_ZERO_FRACTION:
        reason = (
            f"{zero_fraction:.0%} of gaps are exactly zero — the transcript's "
            "boundaries absorbed the pauses"
        )
    elif median_ms <= C.PAUSE_ZERO_GAP_MS:
        reason = "the median gap is zero"
    return PauseStats(
        scope=scope, key=key, n_samples=n, median_ms=median_ms,
        zero_fraction=zero_fraction, degenerate=bool(reason), reason=reason,
    )


def measure_pause_stats(
    db: Database,
    *,
    video_id: int,
    video_speaker_id: int | None = None,
    speaker_label: str | None = None,
) -> PauseStats:
    """Pause statistics at the narrowest scope with data behind it.

    design §9: per speaker where the video is diarized, per video
    otherwise. A named speaker pools across the corpus; a diarized label
    with no roster entry is still a distinct voice and is scoped to
    itself; everything else falls back to the video. An empty result at
    a narrow scope falls through to the wider one rather than reporting
    "no data" for a video that plainly has words.
    """
    if speaker_label:
        samples = gaps_for_speaker_label(db, speaker_label)
        if samples:
            return summarize_gaps(samples, scope="speaker", key=speaker_label)
    if video_speaker_id is not None:
        samples = gaps_for_video_speaker(db, video_speaker_id)
        if samples:
            return summarize_gaps(
                samples, scope="video_speaker", key=str(video_speaker_id)
            )
    return summarize_gaps(gaps_for_video(db, video_id), scope="video", key=str(video_id))


def applied_gap_ms(stats: PauseStats) -> int:
    """The gap to actually insert: the median, clamped, or the fallback."""
    if stats.degenerate:
        return C.PAUSE_FALLBACK_GAP_MS
    return max(
        C.PAUSE_MIN_APPLIED_GAP_MS, min(C.PAUSE_MAX_APPLIED_GAP_MS, stats.median_ms)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_pauses.py -q`
Expected: PASS, 11 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/pauses.py tests/test_render_pauses.py
git commit -m "feat: measure between-word pauses and refuse degenerate ones"
```

---

### Task 5: The report

Design §9: *"The report is a markdown file beside the output containing the assembled text, every fragment with its source and timings, any words that couldn't be found, and a paste-ready block for a video description. The underlying data stays structured so JSON and subtitle output can be added later."*

So the markdown is a *rendering* of a dataclass tree, not a pile of f-strings. Every field is a scalar, a string or a tuple of those, which is what makes `dataclasses.asdict` JSON-serialisable — the test asserts it, so the day `--format json` is wanted it is one function and no refactor. `FragmentReport` already carries output-timeline timings for the same reason: a subtitle writer needs exactly those.

This task comes before the planner because the planner builds these types. `report.py` imports `rytp.constants` and nothing else of Part 6's.

**Files:**
- Create: `rytp/render/report.py`
- Test: `tests/test_render_report.py`

**Interfaces:**
- Consumes: `rytp.constants.{MS_PER_SECOND, RENDER_DESCRIPTION_HEADER}`.
- Produces:
  - `SubstitutionRef(text, reason, video_id, start_ms, end_ms, occurrences)`
  - `MissingWord(text, position, substitutions)`
  - `FragmentReport(ord, video_id, video_title, video_url, speaker, source_start_ms, source_end_ms, output_start_ms, output_end_ms, gap_after_ms, gap_origin, text)`
  - `SourceReport(video_id, title, url, fragment_count, used_ms, measured_lufs, gain_db, first_output_ms, geometry)`
  - `RenderReport(render_id, cutlist, created_at, output_path, duration_ms, canvas, loudnorm, gap_policy, target_text, assembled_text, fragments, sources, missing, notes)`
  - `format_timecode(ms: int) -> str`, `format_clock(ms: int) -> str`
  - `description_block(report: RenderReport) -> str`
  - `render_markdown(report: RenderReport) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_report.py`:

````python
"""The render report: structured first, markdown second (design §9)."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from rytp import constants as C
from rytp.render import report as R

FRAGMENTS = (
    R.FragmentReport(
        ord=0, video_id=3, video_title="Разговор о выборах", video_url=None,
        speaker="host", source_start_ms=612_340, source_end_ms=613_100,
        output_start_ms=0, output_end_ms=760, gap_after_ms=180,
        gap_origin="measured", text="мы всё",
    ),
    R.FragmentReport(
        ord=1, video_id=9, video_title="Другой | разговор",
        video_url="https://example.invalid/watch/VIDEO_B", speaker=None,
        source_start_ms=220_100, source_end_ms=220_780,
        output_start_ms=940, output_end_ms=1_620, gap_after_ms=0,
        gap_origin="measured", text="исправит",
    ),
)

SOURCES = (
    R.SourceReport(
        video_id=3, title="Разговор о выборах", url=None, fragment_count=1,
        used_ms=760, measured_lufs=-23.4, gain_db=7.4, first_output_ms=0,
        geometry="1280x720 @ 25 fps",
    ),
    R.SourceReport(
        video_id=9, title="Другой | разговор",
        url="https://example.invalid/watch/VIDEO_B", fragment_count=1,
        used_ms=680, measured_lufs=None, gain_db=None, first_output_ms=940,
        geometry="640x480 @ 25 fps",
    ),
)

REPORT = R.RenderReport(
    render_id="demo-1a2b3c4d",
    cutlist="demo",
    created_at="2026-09-21T09:00:00+00:00",
    output_path="data/output/demo-1a2b3c4d/output.mp4",
    duration_ms=1_620,
    canvas="1280x720 @ 25 fps (16:9)",
    loudnorm=True,
    gap_policy="measured from aligned transcripts",
    target_text="мы всё исправим",
    assembled_text="мы всё исправит",
    fragments=FRAGMENTS,
    sources=SOURCES,
    missing=(
        R.MissingWord(
            text="исправим",
            position=2,
            substitutions=(
                R.SubstitutionRef(
                    text="исправит", reason="edit", video_id=9,
                    start_ms=220_100, end_ms=220_780, occurrences=3,
                ),
            ),
        ),
    ),
    notes=("speaker host: 81% of gaps are exactly zero; fell back to 180 ms",),
)


def test_timecodes_are_hours_minutes_seconds_and_milliseconds() -> None:
    assert R.format_timecode(0) == "0:00:00.000"
    assert R.format_timecode(612_340) == "0:10:12.340"
    assert R.format_timecode(3_661_001) == "1:01:01.001"


def test_the_clock_form_drops_the_hour_when_there_is_none() -> None:
    assert R.format_clock(0) == "0:00"
    assert R.format_clock(612_340) == "10:12"
    assert R.format_clock(3_661_001) == "1:01:01"


def test_the_markdown_opens_with_the_render_and_carries_the_text() -> None:
    text = R.render_markdown(REPORT)
    assert text.startswith("# Render demo-1a2b3c4d")
    assert "мы всё исправит" in text
    assert "1280x720 @ 25 fps (16:9)" in text


def test_every_fragment_appears_with_its_source_and_both_timelines() -> None:
    text = R.render_markdown(REPORT)
    assert "0:10:12.340" in text  # source in-point
    assert "0:00:00.940" in text  # output in-point of the second fragment
    assert "video 3" in text and "video 9" in text


def test_a_pipe_in_a_title_does_not_break_the_table() -> None:
    text = R.render_markdown(REPORT)
    row = next(line for line in text.splitlines() if "Другой" in line and "|" in line)
    assert "Другой \\| разговор" in row


def test_missing_words_are_listed_with_their_substitutions() -> None:
    text = R.render_markdown(REPORT)
    assert "## Words not found" in text
    assert "исправим" in text
    assert "исправит" in text
    assert "edit" in text


def test_a_render_with_nothing_missing_says_so() -> None:
    complete = replace(REPORT, missing=())
    assert "every word was found" in R.render_markdown(complete).lower()


def test_the_description_block_lists_each_source_once_in_appearance_order() -> None:
    block = R.description_block(REPORT)
    assert C.RENDER_DESCRIPTION_HEADER in block
    assert block.count("Разговор о выборах") == 1
    assert block.index("Разговор о выборах") < block.index("Другой | разговор")
    assert block.splitlines()[0] == "мы всё исправит"


def test_the_description_block_is_fenced_inside_the_markdown() -> None:
    text = R.render_markdown(REPORT)
    assert "## Description" in text
    assert "```" in text[text.index("## Description") :]


def test_notes_are_rendered_so_a_degenerate_fallback_is_visible() -> None:
    assert "fell back to 180 ms" in R.render_markdown(REPORT)


def test_the_whole_report_is_json_serialisable() -> None:
    """Design §9: JSON and subtitle output must be addable without rework."""
    payload = json.dumps(asdict(REPORT), ensure_ascii=False)
    round_tripped = json.loads(payload)
    assert round_tripped["fragments"][1]["output_start_ms"] == 940
    assert round_tripped["missing"][0]["substitutions"][0]["reason"] == "edit"
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_report.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render.report'`.

- [ ] **Step 3: Write `rytp/render/report.py`**

````python
"""The report that ships beside the output file. design §9.

The owner's actual deliverable is two files: something to upload and a
list of where every second of it came from. This module owns the second
one.

It is a dataclass tree with a markdown renderer bolted on, rather than
markdown assembled inline, because design §9 asks that JSON and subtitle
output be addable later without rework. Every field here is a scalar, a
string, or a tuple of those, so ``dataclasses.asdict`` is already
JSON-serialisable, and ``FragmentReport`` already carries output-timeline
timings — which is exactly what a subtitle writer needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from rytp import constants as C

_MS_PER_MINUTE = 60_000
_MS_PER_HOUR = 3_600_000


@dataclass(frozen=True)
class SubstitutionRef:
    """A ranked stand-in for a word the corpus does not contain."""

    text: str
    reason: str  # "stem" | "edit"
    video_id: int
    start_ms: int
    end_ms: int
    occurrences: int = 0


@dataclass(frozen=True)
class MissingWord:
    """A word of the target that no cuttable fragment covers."""

    text: str
    position: int  # index into the target's token sequence
    substitutions: tuple[SubstitutionRef, ...] = ()


@dataclass(frozen=True)
class FragmentReport:
    """One cut, with both timelines: where it came from and where it landed."""

    ord: int
    video_id: int
    video_title: str
    video_url: str | None
    speaker: str | None
    source_start_ms: int
    source_end_ms: int
    output_start_ms: int
    output_end_ms: int
    gap_after_ms: int
    gap_origin: str
    text: str


@dataclass(frozen=True)
class SourceReport:
    """One source video's whole contribution."""

    video_id: int
    title: str
    url: str | None
    fragment_count: int
    used_ms: int
    measured_lufs: float | None
    gain_db: float | None
    first_output_ms: int
    geometry: str


@dataclass(frozen=True)
class RenderReport:
    """Everything the render did, structured."""

    render_id: str
    cutlist: str
    created_at: str
    output_path: str
    duration_ms: int
    canvas: str
    loudnorm: bool
    gap_policy: str
    target_text: str
    assembled_text: str
    fragments: tuple[FragmentReport, ...]
    sources: tuple[SourceReport, ...]
    missing: tuple[MissingWord, ...]
    notes: tuple[str, ...]


def format_timecode(ms: int) -> str:
    """``H:MM:SS.mmm`` — precise enough to seek to in a player."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs, millis = divmod(rest, C.MS_PER_SECOND)
    return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_clock(ms: int) -> str:
    """``M:SS``, or ``H:MM:SS`` past an hour — the form a description wants."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    secs = rest // C.MS_PER_SECOND
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _cell(text: str | None) -> str:
    """Make a value safe inside a markdown table cell."""
    if text is None:
        return "—"
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _number(value: float | None, *, suffix: str = "") -> str:
    return "—" if value is None else f"{value:+.1f}{suffix}"


def description_block(report: RenderReport) -> str:
    """The paste-ready block for a video description.

    The assembled text, then each source once, in the order it first
    appears, with the output timestamp where it does. One line per
    source rather than per fragment: a description listing forty
    entries is not a description.
    """
    lines = [report.assembled_text.strip(), "", C.RENDER_DESCRIPTION_HEADER]
    for source in sorted(report.sources, key=lambda s: s.first_output_ms):
        tail = f" — {source.url}" if source.url else ""
        lines.append(f"{format_clock(source.first_output_ms)} — {source.title}{tail}")
    return "\n".join(lines).strip() + "\n"


def render_markdown(report: RenderReport) -> str:
    """The whole report as one markdown document."""
    out: list[str] = [
        f"# Render {report.render_id}",
        "",
        f"Cut list `{report.cutlist}` · rendered {report.created_at} · "
        f"{format_timecode(report.duration_ms)} long",
        "",
        f"- **Output** `{report.output_path}`",
        f"- **Canvas** {report.canvas}",
        f"- **Loudness** {'normalized' if report.loudnorm else 'left alone'}",
        f"- **Gaps** {report.gap_policy}",
        "",
        "## Text",
        "",
        f"> {report.assembled_text.strip() or '—'}",
        "",
    ]
    if report.target_text.strip() != report.assembled_text.strip():
        out += ["Asked for:", "", f"> {report.target_text.strip()}", ""]

    out += [
        "## Fragments",
        "",
        "| # | output | source | speaker | source in → out | gap after | text |",
        "|--:|---|---|---|---|--:|---|",
    ]
    for fragment in report.fragments:
        out.append(
            f"| {fragment.ord + 1} "
            f"| {format_timecode(fragment.output_start_ms)} → "
            f"{format_timecode(fragment.output_end_ms)} "
            f"| video {fragment.video_id} — {_cell(fragment.video_title)} "
            f"| {_cell(fragment.speaker)} "
            f"| {format_timecode(fragment.source_start_ms)} → "
            f"{format_timecode(fragment.source_end_ms)} "
            f"| {fragment.gap_after_ms} ms ({fragment.gap_origin}) "
            f"| {_cell(fragment.text)} |"
        )

    out += ["", "## Words not found", ""]
    if not report.missing:
        out.append("None — every word was found in the corpus.")
    else:
        out += ["| word | position | suggested instead |", "|---|--:|---|"]
        for missing in report.missing:
            suggestions = (
                "; ".join(
                    f"{_cell(s.text)} ({s.reason}, {s.occurrences}×, "
                    f"video {s.video_id} @ {format_timecode(s.start_ms)})"
                    for s in missing.substitutions
                )
                or "—"
            )
            out.append(f"| {_cell(missing.text)} | {missing.position} | {suggestions} |")

    out += [
        "",
        "## Sources",
        "",
        "| video | title | fragments | used | geometry | measured | gain |",
        "|--:|---|--:|--:|---|---|---|",
    ]
    for source in report.sources:
        out.append(
            f"| {source.video_id} "
            f"| {_cell(source.title)} "
            f"| {source.fragment_count} "
            f"| {format_timecode(source.used_ms)} "
            f"| {_cell(source.geometry)} "
            f"| {_number(source.measured_lufs, suffix=' LUFS')} "
            f"| {_number(source.gain_db, suffix=' dB')} |"
        )

    if report.notes:
        out += ["", "## Notes", ""]
        out += [f"- {note}" for note in report.notes]

    out += [
        "",
        "## Description",
        "",
        "```",
        description_block(report).rstrip("\n"),
        "```",
        "",
    ]
    return "\n".join(out)
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_report.py -q`
Expected: PASS, 11 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/report.py tests/test_render_report.py
git commit -m "feat: write the render report as structured data plus markdown"
```

---

### Task 6: The render plan — sources, geometry, gaps

Everything decided before a single frame is encoded. `plan_render` reads the cut list and the database and answers: which files each fragment is cut from, what shape the output is, how loud each source is, how long each seam is, and where it all lands on the output timeline. It writes nothing and encodes nothing, which is what makes `--dry-run` free and the whole thing testable.

**The Part 5 seam.** Part 5's settled cut-list format is one ordered `[[slot]]` array where `kind` is `"fragment"` or `"gap"`; a gap slot *is* a missing word, carrying ranked `substitutions`, each with a concrete cuttable occurrence so the hole can be filled by copy-paste. Fragment slots carry the contracts §4 `Fragment` fields plus `align_score`, `cost`, `video_speaker_id`, `speaker_label`, an optional hand-written `gap_before_ms`, and nested `alternatives`. `request_from_cutlist` is the single adapter, and it reads attributes rather than importing `rytp.assemble` — so this module and its tests work with Part 5 absent, and a change to Part 5's dataclasses is a one-function fix here.

**Missing media is fatal** (design §9, and the brief: never silently render audio over black). Every unreadable fragment is collected and reported in one message, each naming `rytp fetch-video <id>`, rather than failing on the first and making the owner discover them one render at a time.

**Gap precedence**, decided here and documented in the docstring: `--gap-ms 0` is the off switch and wins over everything, because a switch that something else can override is not a switch; otherwise a hand-written `gap_before_ms` wins, because design §8 makes the cut list the durable, hand-editable representation; otherwise a positive `--gap-ms` applies everywhere; otherwise the gap is measured for the speaker of the fragment that just *finished* — the pause belongs to the person who stopped talking.

**Files:**
- Create: `rytp/render/run.py`
- Test: `tests/test_render_plan.py`

**Interfaces:**
- Consumes: `rytp.config.paths`; `rytp.db.Database`; `rytp.db.queries.assets_for`; `rytp.render.canvas.{Canvas, SourceGeometry, plan_canvas, probe_geometry}`; `rytp.render.ffmpeg.{Tools, RenderError, gain_for, parse_loudnorm_json, run_command, source_loudness_command}`; `rytp.render.pauses.{PauseStats, applied_gap_ms, measure_pause_stats}`; `rytp.render.report.{MissingWord, SubstitutionRef}`.
- Produces:
  - `MissingMediaError(RenderError)`
  - `RenderOptions(canvas_mode, height, fps, gap_ms, loudnorm, preset, crf, keep_intermediates, render_id)` with `.gap_policy`
  - `RenderFragment(video_id, first_word_ord, last_word_ord, start_ms, end_ms, text, video_speaker_id, speaker_label, gap_before_ms)` with `.duration_ms`
  - `RenderRequest(name, fragments, target_text, missing)` with `.assembled_text`
  - `request_from_cutlist(cutlist: object) -> RenderRequest`
  - `SourceMedia(video_id, title, url, video_path, audio_path, geometry, measured_lufs, gain_db)`
  - `resolve_sources(db, fragments, *, tools) -> tuple[SourceMedia, ...]`
  - `sanitize_id(text) -> str`, `plan_fingerprint(request, options) -> str`, `derive_render_id(request, options) -> str`
  - `SeamGap(gap_ms, origin, note)`, `PlannedFragment(ord, fragment, source, gap_after_ms, gap_origin, output_start_ms, output_end_ms, intermediate)`, `RenderPlan(render_id, request, options, canvas, fragments, sources, output_dir, output_path, report_path, list_file, duration_ms, notes)` with `.fragments_dir`
  - `plan_render(db, request, options=None, *, tools=None) -> RenderPlan`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_plan.py`:

```python
"""Planning a render: sources, canvas, gains, gaps (design §9)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.render import run as RUN
from rytp.render.ffmpeg import Tools

from tests.fakes import make_video, touch
from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_pauses import add_words, evenly_spaced

WIDE = probe_json(1280, 720)
NARROW = probe_json(640, 480)


def fragment(video_id: int, start_ms: int, end_ms: int, **kw: object) -> RUN.RenderFragment:
    return RUN.RenderFragment(
        video_id=video_id,
        first_word_ord=kw.pop("first_word_ord", 0),  # type: ignore[arg-type]
        last_word_ord=kw.pop("last_word_ord", 1),  # type: ignore[arg-type]
        start_ms=start_ms,
        end_ms=end_ms,
        text=kw.pop("text", "текст"),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def source_video(db: Database, tmp_path: Path, name: str, **kw: object) -> int:
    """A catalogued video with a rendition and an audio asset on disk."""
    vid = make_video(db, external_id=name, url=f"https://example.invalid/watch/{name}", **kw)
    insert_asset(
        db, video_id=vid, role="video", format_id="720",
        path=str(touch(tmp_path / f"{name}-video.mp4")), width=1280, height=720,
    )
    insert_asset(
        db, video_id=vid, role="audio", path=str(touch(tmp_path / f"{name}-audio.m4a"))
    )
    return vid


def tools(**kw: object) -> tuple[Tools, RecordingRunner]:
    runner = RecordingRunner(geometry=WIDE, loudness=loudnorm_json(-23.0), **kw)  # type: ignore[arg-type]
    return Tools.faked(runner), runner


def test_a_cut_list_becomes_a_request_without_importing_part_five() -> None:
    cutlist = SimpleNamespace(
        name="demo",
        target="мы всё исправим",
        slots=(
            SimpleNamespace(
                kind="fragment", target_first=0, target_last=1, text="мы всё",
                video_id=3, first_word_ord=1204, last_word_ord=1205,
                start_ms=612_340, end_ms=613_100, align_score=0.81, cost=1.42,
                video_speaker_id=11, speaker_label="host", alternatives=(),
            ),
            SimpleNamespace(
                kind="gap", target_first=2, target_last=2, text="исправим",
                substitutions=(
                    SimpleNamespace(
                        text="исправит", reason="edit", distance=1, occurrences=3,
                        video_id=9, first_word_ord=502, last_word_ord=502,
                        start_ms=220_100, end_ms=220_780,
                    ),
                ),
            ),
        ),
    )
    request = RUN.request_from_cutlist(cutlist)
    assert request.name == "demo"
    assert request.target_text == "мы всё исправим"
    assert len(request.fragments) == 1
    assert request.fragments[0].speaker_label == "host"
    assert request.fragments[0].gap_before_ms is None
    assert request.missing[0].text == "исправим"
    assert request.missing[0].substitutions[0].video_id == 9
    assert request.assembled_text == "мы всё"


def test_a_missing_rendition_names_the_command_that_fetches_it(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db)
    insert_asset(db, video_id=vid, role="audio", path=str(touch(tmp_path / "a.m4a")))
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 1_000),))
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError) as excinfo:
        RUN.plan_render(db, request, tools=kit)
    message = str(excinfo.value)
    assert "no video rendition" in message
    assert f"rytp fetch-video {vid}" in message


def test_every_unreadable_source_is_reported_at_once(db: Database, tmp_path: Path) -> None:
    first = make_video(db, external_id="VIDEO_A")
    second = make_video(db, external_id="VIDEO_B", url="https://example.invalid/watch/VIDEO_B")
    insert_asset(db, video_id=second, role="video", format_id="720",
                 path=str(touch(tmp_path / "b.mp4")))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(first, 0, 500), fragment(second, 0, 500)),
    )
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError) as excinfo:
        RUN.plan_render(db, request, tools=kit)
    assert str(first) in str(excinfo.value)
    assert str(second) in str(excinfo.value)


def test_a_deleted_file_is_as_missing_as_a_missing_row(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    Path(tmp_path / "VIDEO_A-video.mp4").unlink()
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    with pytest.raises(RUN.MissingMediaError, match="no video rendition"):
        RUN.plan_render(db, request, tools=kit)


def test_a_local_container_serves_as_both_video_and_audio(
    db: Database, tmp_path: Path
) -> None:
    vid = make_video(db, source=C.LOCAL_SOURCE, external_id=str(tmp_path / "clip.mkv"),
                     url=None)
    insert_asset(db, video_id=vid, role="container",
                 path=str(touch(tmp_path / "clip.mkv")))
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    source = plan.sources[0]
    assert source.video_path == source.audio_path


def test_each_source_is_probed_exactly_once(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 500), fragment(vid, 900, 1_400)),
    )
    kit, runner = tools()
    RUN.plan_render(db, request, tools=kit)
    assert len(runner.probes) == 1


def test_the_canvas_covers_every_source(db: Database, tmp_path: Path) -> None:
    first = source_video(db, tmp_path, "VIDEO_A")
    second = source_video(db, tmp_path, "VIDEO_B")
    request = RUN.RenderRequest(
        name="demo", fragments=(fragment(first, 0, 500), fragment(second, 0, 500))
    )
    runner = RecordingRunner(
        geometry={"VIDEO_A-video": WIDE, "VIDEO_B-video": NARROW},
        loudness=loudnorm_json(-23.0),
    )
    plan = RUN.plan_render(db, request, tools=Tools.faked(runner))
    assert (plan.canvas.width, plan.canvas.height) == (1280, 720)
    assert plan.canvas.fps == 25


def test_output_offsets_accumulate_the_gaps(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 5_000, 5_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(gap_ms=200), tools=kit)
    first, second = plan.fragments
    assert (first.output_start_ms, first.output_end_ms) == (0, 1_000)
    assert first.gap_after_ms == 200
    assert (second.output_start_ms, second.output_end_ms) == (1_200, 1_700)
    assert second.gap_after_ms == 0  # nothing follows the last fragment
    assert plan.duration_ms == 1_700


def test_zero_gaps_switches_the_feature_off_including_hand_written_ones(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(
            fragment(vid, 0, 1_000),
            fragment(vid, 5_000, 5_500, gap_before_ms=750),
        ),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(gap_ms=0), tools=kit)
    assert plan.fragments[0].gap_after_ms == 0
    assert plan.duration_ms == 1_500


def test_a_hand_written_gap_wins_over_measurement(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, evenly_spaced(100, gap_ms=200))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(
            fragment(vid, 0, 1_000),
            fragment(vid, 5_000, 5_500, gap_before_ms=750),
        ),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == 750
    assert plan.fragments[0].gap_origin == "override"


def test_measured_gaps_use_the_speaker_who_just_stopped(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, evenly_spaced(100, gap_ms=240))
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 5_000, 5_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == 240
    assert plan.fragments[0].gap_origin == "measured"


def test_a_degenerate_distribution_falls_back_and_leaves_a_note(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, [(i * 300, (i + 1) * 300) for i in range(120)])
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 0, 1_000), fragment(vid, 60_000, 60_500)),
    )
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.fragments[0].gap_after_ms == C.PAUSE_FALLBACK_GAP_MS
    assert plan.fragments[0].gap_origin == "fallback"
    assert any("zero" in note for note in plan.notes)


def test_the_gain_comes_from_video_acoustics_without_running_anything(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    db.conn.execute(
        "INSERT INTO video_acoustics (video_id, loudness_lufs, computed_at) "
        "VALUES (?, ?, '2026-01-01T00:00:00+00:00')",
        (vid, -26.0),
    )
    db.conn.commit()
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, runner = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.sources[0].measured_lufs == pytest.approx(-26.0)
    assert plan.sources[0].gain_db == pytest.approx(10.0)
    assert runner.measurements == []


def test_an_unmeasured_source_is_scanned_once_around_its_first_fragment(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo",
        fragments=(fragment(vid, 600_000, 600_500), fragment(vid, 700_000, 700_500)),
    )
    kit, runner = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.sources[0].measured_lufs == pytest.approx(-23.0)
    assert plan.sources[0].gain_db == pytest.approx(7.0)
    assert len(runner.measurements) == 1
    scan = runner.measurements[0]
    assert scan[scan.index("-ss") + 1] == "540.000"  # 120 s window, centred


def test_an_unmeasurable_source_is_left_alone_and_noted(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    runner = RecordingRunner(geometry=WIDE, loudness='{"input_i" : "-inf"}')
    plan = RUN.plan_render(db, request, tools=Tools.faked(runner))
    assert plan.sources[0].gain_db is None
    assert any("loudness" in note for note in plan.notes)


def test_no_loudnorm_skips_every_scan(db: Database, tmp_path: Path) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, runner = tools()
    plan = RUN.plan_render(db, request, RUN.RenderOptions(loudnorm=False), tools=kit)
    assert runner.measurements == []
    assert plan.sources[0].gain_db is None


def test_the_render_id_is_deterministic_and_option_sensitive() -> None:
    request = RUN.RenderRequest(name="demo", fragments=(fragment(3, 0, 500),))
    first = RUN.derive_render_id(request, RUN.RenderOptions())
    again = RUN.derive_render_id(request, RUN.RenderOptions())
    other = RUN.derive_render_id(request, RUN.RenderOptions(canvas_mode="bbox"))
    assert first == again
    assert first.startswith("demo-")
    assert first != other
    assert RUN.derive_render_id(request, RUN.RenderOptions(render_id="mine")) == "mine"


def test_dropping_provenance_does_not_move_the_output() -> None:
    """Word ordinals never reach ffmpeg, so they are not in the hash."""
    options = RUN.RenderOptions()
    with_ords = RUN.RenderRequest(
        name="demo", fragments=(fragment(3, 0, 500, first_word_ord=12,
                                         last_word_ord=15),)
    )
    without = RUN.RenderRequest(
        name="demo", fragments=(fragment(3, 0, 500, first_word_ord=0,
                                         last_word_ord=0),)
    )
    assert RUN.derive_render_id(with_ords, options) == RUN.derive_render_id(
        without, options
    )
    moved = RUN.RenderRequest(name="demo", fragments=(fragment(3, 0, 600),))
    assert RUN.derive_render_id(moved, options) != RUN.derive_render_id(
        with_ords, options
    )


def test_a_render_id_is_a_safe_directory_name() -> None:
    request = RUN.RenderRequest(name="../не безопасно/имя", fragments=(fragment(3, 0, 5),))
    render_id = RUN.derive_render_id(request, RUN.RenderOptions())
    assert "/" not in render_id and ".." not in render_id
    assert "не" in render_id  # Cyrillic survives; only path syntax is stripped
    assert len(render_id) <= C.RENDER_ID_MAX_CHARS


def test_an_empty_cut_list_is_refused(db: Database) -> None:
    kit, _ = tools()
    with pytest.raises(RUN.RenderError, match="no fragments"):
        RUN.plan_render(db, RUN.RenderRequest(name="demo", fragments=()), tools=kit)


def test_the_plan_points_at_the_contracted_output_paths(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 500),))
    kit, _ = tools()
    plan = RUN.plan_render(db, request, tools=kit)
    assert plan.output_path.name == "output.mp4"
    assert plan.report_path.name == "report.md"
    assert plan.output_path.parent.name == plan.render_id
    assert plan.fragments[0].intermediate.suffix == C.RENDER_INTERMEDIATE_SUFFIX
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_plan.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render.run'`.

- [ ] **Step 3: Write `rytp/render/run.py`**

```python
"""Planning and running a render. design §9.

Two phases, deliberately separated. :func:`plan_render` reads the cut
list, the catalog and the sources and decides everything — which files,
what shape, how loud, how long each seam is, where each fragment lands
on the output timeline — without encoding a frame. :func:`render_cutlist`
then executes that plan. The split is what makes ``--dry-run`` free and
what lets the whole of this module be tested with a fake runner.

The cut list arrives as Part 5's ``CutList``, but nothing here imports
``rytp.assemble``: :func:`request_from_cutlist` reads attributes off
whatever it is handed. One adapter, one place to change.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import assets_for
from rytp.render.canvas import Canvas, SourceGeometry, plan_canvas, probe_geometry
from rytp.render.ffmpeg import (
    RenderError,
    Tools,
    gain_for,
    parse_loudnorm_json,
    run_command,
    source_loudness_command,
)
from rytp.render.pauses import PauseStats, applied_gap_ms, measure_pause_stats
from rytp.render.report import MissingWord, SubstitutionRef

#: Everything that is not a letter, digit or underscore becomes a dash.
#: ``\w`` is Unicode-aware, so a Cyrillic cut-list name survives intact,
#: while "..", "/" and a backslash cannot reach a path.
_UNSAFE = re.compile(r"[^\w-]+", re.UNICODE)


class MissingMediaError(RenderError):
    """A fragment's source file is not on disk. Names the fetch command."""


@dataclass(frozen=True)
class RenderOptions:
    """Every knob, with the defaults design §9 asks for."""

    canvas_mode: str = C.RENDER_DEFAULT_CANVAS_MODE
    height: int = 0  # 0 = the tallest source, capped
    fps: int = 0  # 0 = the commonest source rate
    gap_ms: int = -1  # -1 = measure, 0 = no gaps, >0 = this many
    loudnorm: bool = True
    preset: str = C.RENDER_VIDEO_PRESET
    crf: int = C.RENDER_VIDEO_CRF
    keep_intermediates: bool = False
    render_id: str = ""

    @property
    def gap_policy(self) -> str:
        """One phrase for the report."""
        if self.gap_ms == 0:
            return "off"
        if self.gap_ms > 0:
            return f"fixed at {self.gap_ms} ms"
        return "measured from aligned transcripts"


@dataclass(frozen=True)
class RenderFragment:
    """One cut, as the render needs it. Part 5's fragment slot, flattened."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    video_speaker_id: int | None = None
    speaker_label: str | None = None
    gap_before_ms: int | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class RenderRequest:
    """What to render: the fragments, plus what the report needs to say."""

    name: str
    fragments: tuple[RenderFragment, ...]
    target_text: str = ""
    missing: tuple[MissingWord, ...] = ()

    @property
    def assembled_text(self) -> str:
        return " ".join(f.text.strip() for f in self.fragments if f.text.strip())


def request_from_cutlist(cutlist: object) -> RenderRequest:
    """Adapt Part 5's ``CutList`` — by attribute, never by import.

    Part 5's format is one ordered ``slots`` sequence; ``kind`` is
    ``"fragment"`` or ``"gap"``, and a gap *is* a missing word carrying
    its ranked substitutions. Document order is the output timeline, so
    the order here is preserved exactly.
    """
    fragments: list[RenderFragment] = []
    missing: list[MissingWord] = []
    for slot in getattr(cutlist, "slots", ()):
        if getattr(slot, "kind", "") == "fragment":
            fragments.append(
                RenderFragment(
                    video_id=int(slot.video_id),
                    first_word_ord=int(slot.first_word_ord),
                    last_word_ord=int(slot.last_word_ord),
                    start_ms=int(slot.start_ms),
                    end_ms=int(slot.end_ms),
                    text=str(slot.text),
                    video_speaker_id=getattr(slot, "video_speaker_id", None),
                    speaker_label=getattr(slot, "speaker_label", None),
                    gap_before_ms=getattr(slot, "gap_before_ms", None),
                )
            )
            continue
        missing.append(
            MissingWord(
                text=str(slot.text),
                position=int(getattr(slot, "target_first", 0)),
                substitutions=tuple(
                    SubstitutionRef(
                        text=str(sub.text),
                        reason=str(getattr(sub, "reason", "")),
                        video_id=int(getattr(sub, "video_id", 0)),
                        start_ms=int(getattr(sub, "start_ms", 0)),
                        end_ms=int(getattr(sub, "end_ms", 0)),
                        occurrences=int(getattr(sub, "occurrences", 0)),
                    )
                    for sub in getattr(slot, "substitutions", ())
                ),
            )
        )
    return RenderRequest(
        name=str(getattr(cutlist, "name", "")),
        fragments=tuple(fragments),
        target_text=str(getattr(cutlist, "target", "")),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class SourceMedia:
    """One source video, resolved to files and measured."""

    video_id: int
    title: str
    url: str | None
    video_path: Path
    audio_path: Path
    geometry: SourceGeometry
    measured_lufs: float | None = None
    gain_db: float | None = None


def _video_row(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT id, title, url FROM videos WHERE id = ?", (video_id,)
    ).fetchone()


def _newest_existing(db: Database, video_id: int, role: str) -> Path | None:
    """The newest asset of a role whose file is actually there.

    Newest wins because design §4 says upgrading a rendition is just an
    insert; existence is checked because an asset row whose file was
    deleted outside the tool must not look usable.
    """
    for row in reversed(assets_for(db, video_id, role)):
        path = Path(row["path"])
        if path.exists():
            return path
    return None


def resolve_sources(
    db: Database, fragments: Sequence[RenderFragment], *, tools: Tools
) -> tuple[SourceMedia, ...]:
    """Resolve every distinct source, or refuse with all the reasons.

    Video comes from a rendition asset and audio from the audio asset
    (design §4). A local file registered as a single ``container`` asset
    stands in for either. Every problem is collected before raising, so
    one render tells the owner about all of them instead of one per run.
    """
    ordered: list[int] = []
    for fragment in fragments:
        if fragment.video_id not in ordered:
            ordered.append(fragment.video_id)
    problems: list[str] = []
    sources: list[SourceMedia] = []
    for video_id in ordered:
        row = _video_row(db, video_id)
        if row is None:
            problems.append(f"video {video_id} is not in the catalog")
            continue
        container = _newest_existing(db, video_id, "container")
        video_path = _newest_existing(db, video_id, "video") or container
        audio_path = _newest_existing(db, video_id, "audio") or container
        if video_path is None:
            problems.append(
                f"video {video_id} has no video rendition on disk — "
                f"run: rytp fetch-video {video_id}"
            )
            continue
        if audio_path is None:
            problems.append(
                f"video {video_id} has no audio asset on disk — "
                f"run: rytp fetch-video {video_id}"
            )
            continue
        sources.append(
            SourceMedia(
                video_id=video_id,
                title=str(row["title"]),
                url=row["url"],
                video_path=video_path,
                audio_path=audio_path,
                geometry=probe_geometry(
                    video_path, runner=tools.runner, binary=tools.ffprobe
                ),
            )
        )
    if problems:
        raise MissingMediaError("cannot render: " + "; ".join(problems))
    return tuple(sources)


def _stored_loudness(db: Database, video_id: int) -> float | None:
    row = db.conn.execute(
        "SELECT loudness_lufs FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None or row["loudness_lufs"] is None:
        return None
    return float(row["loudness_lufs"])


def measure_source_loudness(
    db: Database, source: SourceMedia, *, around_ms: int, tools: Tools
) -> float | None:
    """This source's integrated loudness, in LUFS, or ``None``.

    Part 3's fingerprint already measures the whole video, so that value
    is used when it is there. Otherwise one bounded window is scanned —
    two minutes around the first fragment this render takes from the
    source. Scanning a whole hour to place a single gain is not worth
    the minutes, and a two-minute window of continuous speech is plenty
    for an R128 integrated measurement.
    """
    stored = _stored_loudness(db, source.video_id)
    if stored is not None:
        return stored
    window_start = max(0, around_ms - C.RENDER_LOUDNESS_WINDOW_MS // 2)
    result = run_command(
        source_loudness_command(
            path=source.audio_path,
            start_ms=window_start,
            duration_ms=C.RENDER_LOUDNESS_WINDOW_MS,
            binary=tools.ffmpeg,
        ),
        runner=tools.runner,
        what=f"loudness scan of video {source.video_id}",
    )
    measured = parse_loudnorm_json(result.stderr)
    return None if measured is None else measured.input_i


@dataclass(frozen=True)
class SeamGap:
    """How long the freeze-frame between two fragments lasts, and why."""

    gap_ms: int
    origin: str  # "off" | "override" | "fixed" | "measured" | "fallback" | "end"
    note: str = ""


def plan_gap(
    db: Database,
    *,
    previous: RenderFragment,
    following: RenderFragment,
    options: RenderOptions,
    cache: dict[tuple[str, str], PauseStats],
) -> SeamGap:
    """Decide one seam.

    Precedence, most specific last except for the off switch:

    1. ``gap_ms == 0`` — off, and it wins over everything. A switch that
       something else can override is not a switch.
    2. ``gap_before_ms`` on the *following* fragment — a human wrote it
       into the cut list, which design §8 makes the durable artifact.
    3. ``gap_ms > 0`` — one fixed length everywhere.
    4. Measured (the default). The pause belongs to the speaker who just
       stopped talking, so it is measured for ``previous``.
    """
    if options.gap_ms == 0:
        return SeamGap(0, "off")
    if following.gap_before_ms is not None:
        return SeamGap(max(0, int(following.gap_before_ms)), "override")
    if options.gap_ms > 0:
        return SeamGap(options.gap_ms, "fixed")
    key = (
        previous.speaker_label or "",
        ""
        if previous.video_speaker_id is None
        else str(previous.video_speaker_id),
    )
    cache_key = (f"{previous.video_id}", f"{key[0]}|{key[1]}")
    stats = cache.get(cache_key)
    if stats is None:
        stats = measure_pause_stats(
            db,
            video_id=previous.video_id,
            video_speaker_id=previous.video_speaker_id,
            speaker_label=previous.speaker_label,
        )
        cache[cache_key] = stats
    return SeamGap(
        applied_gap_ms(stats),
        "fallback" if stats.degenerate else "measured",
        stats.description if stats.degenerate else "",
    )


@dataclass(frozen=True)
class PlannedFragment:
    """One fragment with everything the encoder needs decided."""

    ord: int
    fragment: RenderFragment
    source: SourceMedia
    gap_after_ms: int
    gap_origin: str
    output_start_ms: int
    output_end_ms: int
    intermediate: Path


@dataclass(frozen=True)
class RenderPlan:
    """A whole render, decided but not yet executed."""

    render_id: str
    request: RenderRequest
    options: RenderOptions
    canvas: Canvas
    fragments: tuple[PlannedFragment, ...]
    sources: tuple[SourceMedia, ...]
    output_dir: Path
    output_path: Path
    report_path: Path
    list_file: Path
    duration_ms: int
    notes: tuple[str, ...]

    @property
    def fragments_dir(self) -> Path:
        """Where the per-fragment intermediates go, under the output dir."""
        return self.output_dir / "fragments"


def sanitize_id(text: str) -> str:
    """A path-safe, still-readable directory name."""
    return _UNSAFE.sub("-", text.strip()).strip("-")[: C.RENDER_ID_MAX_CHARS]


def plan_fingerprint(request: RenderRequest, options: RenderOptions) -> str:
    """A hash over exactly what reaches ffmpeg, and nothing else.

    Word ordinals are deliberately **not** in it. They are provenance —
    they never reach a filter graph, and Part 5 allows a hand-edit to
    drop them. Hashing them would move byte-identical output to a new
    ``render_id`` because somebody tidied a TOML file, which is the
    opposite of what the hash is for.
    """
    payload = {
        "name": request.name,
        "target": request.target_text,
        "fragments": [
            [f.video_id, f.start_ms, f.end_ms, f.gap_before_ms]
            for f in request.fragments
        ],
        "canvas_mode": options.canvas_mode,
        "height": options.height,
        "fps": options.fps,
        "gap_ms": options.gap_ms,
        "loudnorm": options.loudnorm,
        "preset": options.preset,
        "crf": options.crf,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_render_id(request: RenderRequest, options: RenderOptions) -> str:
    """``{cut list name}-{8 hex}``, stable for the same inputs.

    Deterministic on purpose: re-rendering the same cut list with the
    same options overwrites the same directory instead of littering
    ``output/`` with near-identical copies, and changing any option that
    affects the output makes a new one.
    """
    if options.render_id:
        return sanitize_id(options.render_id) or "render"
    room = C.RENDER_ID_MAX_CHARS - C.RENDER_ID_HASH_CHARS - 1
    stem = sanitize_id(request.name)[:room] or "render"
    digest = plan_fingerprint(request, options)[: C.RENDER_ID_HASH_CHARS]
    return f"{stem}-{digest}"


def plan_render(
    db: Database,
    request: RenderRequest,
    options: RenderOptions | None = None,
    *,
    tools: Tools | None = None,
) -> RenderPlan:
    """Decide the whole render. Reads; writes and encodes nothing."""
    options = options or RenderOptions()
    if not request.fragments:
        raise RenderError(f"cut list {request.name!r} has no fragments to render")
    for fragment in request.fragments:
        if fragment.duration_ms <= 0:
            raise RenderError(
                f"fragment from video {fragment.video_id} has a non-positive "
                f"duration: {fragment.start_ms}..{fragment.end_ms} ms"
            )
    tools = tools or Tools.resolve()
    notes: list[str] = []

    sources = list(resolve_sources(db, request.fragments, tools=tools))
    canvas = plan_canvas(
        [source.geometry for source in sources],
        mode=options.canvas_mode,
        height=options.height,
        fps=options.fps,
    )

    if options.loudnorm:
        first_use: dict[int, int] = {}
        for fragment in request.fragments:
            first_use.setdefault(fragment.video_id, fragment.start_ms)
        measured_sources = []
        for source in sources:
            lufs = measure_source_loudness(
                db, source, around_ms=first_use[source.video_id], tools=tools
            )
            if lufs is None:
                notes.append(
                    f"video {source.video_id}: loudness could not be measured; "
                    "its level was left alone"
                )
            measured_sources.append(
                replace(source, measured_lufs=lufs, gain_db=gain_for(lufs))
            )
        sources = measured_sources

    by_id = {source.video_id: source for source in sources}
    render_id = derive_render_id(request, options)
    output_dir = config.paths().output_dir(render_id)
    fragments_dir = output_dir / "fragments"

    planned: list[PlannedFragment] = []
    cache: dict[tuple[str, str], PauseStats] = {}
    clock = 0
    for index, fragment in enumerate(request.fragments):
        following = (
            request.fragments[index + 1] if index + 1 < len(request.fragments) else None
        )
        seam = (
            SeamGap(0, "end")
            if following is None
            else plan_gap(
                db,
                previous=fragment,
                following=following,
                options=options,
                cache=cache,
            )
        )
        if seam.note and seam.note not in notes:
            notes.append(seam.note)
        planned.append(
            PlannedFragment(
                ord=index,
                fragment=fragment,
                source=by_id[fragment.video_id],
                gap_after_ms=seam.gap_ms,
                gap_origin=seam.origin,
                output_start_ms=clock,
                output_end_ms=clock + fragment.duration_ms,
                intermediate=fragments_dir
                / f"{index:04d}{C.RENDER_INTERMEDIATE_SUFFIX}",
            )
        )
        clock += fragment.duration_ms + seam.gap_ms

    return RenderPlan(
        render_id=render_id,
        request=request,
        options=options,
        canvas=canvas,
        fragments=tuple(planned),
        sources=tuple(sources),
        output_dir=output_dir,
        output_path=output_dir / "output.mp4",
        report_path=output_dir / "report.md",
        list_file=output_dir / "concat.txt",
        duration_ms=clock,
        notes=tuple(notes),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_plan.py -q`
Expected: PASS, 21 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/run.py tests/test_render_plan.py
git commit -m "feat: plan a render from a cut list without touching ffmpeg"
```

---

### Task 7: Executing the plan

The part that makes files. One ffmpeg run per fragment, one list file, one measuring pass, one concat pass, then the report. Nothing here decides anything — every decision was made in Task 6, which is why this function is short and why a `--dry-run` that returns the same `RenderResult` without touching the disk is three lines rather than a parallel code path.

Three behaviours worth stating:

- **The report is written last and includes what execution learned.** If the programme could not be measured, or a source's loudness could not be, the report says so. A report that claims a normalization that did not happen is worse than no report.
- **Intermediates are deleted on success, kept on failure.** The output directory is the thing the owner opens; forty stray `.mkv` files beside `output.mp4` are clutter. On failure they survive, because that is when they are worth looking at, and `--keep-intermediates` keeps them always.
- **Deleting a render deletes files, so it is planned before it is done.** Contracts §5: anything that removes a file offers `--dry-run` and requires `--yes`, and jobs — which have no foreign key — must be cancelled in the same transaction. Rendered output is minutes of encoding, so the dry run counts bytes and names the cut list that produced it. Only a directory inside the data tree's `output/` is ever removed: `renders.output_path` is a text column a human can edit, and recursively deleting whatever it happens to say is not a risk worth carrying.
- **A render is a `renders` row first.** `jobs.target_id` is an integer, and contracts §3 now gives a render its own row to be the integer — so the job addresses `renders.id` the way a download addresses `videos.id`, and nothing has to be hashed, re-derived or verified. The row opens `planned` before any encoding and closes `rendered` or `failed`, so a render killed mid-encode leaves a trace, and the history the redesign lost when it dropped `splice_runs` has somewhere to live.

**Files:**
- Modify: `rytp/render/run.py` (append)
- Test: `tests/test_render_run.py`

**Interfaces:**
- Consumes: everything from Task 6; `rytp.render.ffmpeg.{concat_command, fragment_command, programme_loudness_command, write_concat_list, FfmpegFailedError}`; `rytp.render.canvas.video_filter_chain`; `rytp.render.report.{FragmentReport, RenderReport, SourceReport, render_markdown}`; `rytp.models.utc_now_iso`; `rytp.config.ensure_dir`.
- Produces:
  - `RenderResult(render_id, output_path, report_path, duration_ms, fragment_count, source_count, plan)`
  - `build_report(plan: RenderPlan, *, notes: Sequence[str], created_at: str) -> RenderReport`
  - `render_cutlist(db, request, options=None, *, tools=None, dry_run=False, render_row_id=None) -> RenderResult`
  - `create_render(db, *, cutlist_name: str, options: RenderOptions, now: str | None = None) -> int`
  - `get_render(db, render_id: int) -> sqlite3.Row`, `finish_render(db, render_id, *, state: str, output_path: Path | None = None, now: str | None = None) -> None`
  - `payload_for(options: RenderOptions) -> dict[str, object]`, `options_from_payload(payload: dict) -> RenderOptions`
  - `run_render_job(db: Database, target_id: int, payload: dict) -> None`
  - `RenderRemoval(render_id, cutlist_name, cutlist_present, output_dir, files, total_bytes, jobs_cancelled, removed)`
  - `plan_removal(db, render_id: int) -> RenderRemoval`, `remove_render(db, render_id: int, *, dry_run: bool = False) -> RenderRemoval`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_run.py`:

```python
"""Executing a render plan, with a fake ffmpeg (design §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.db import Database
from rytp.render import run as RUN
from rytp.render.ffmpeg import FfmpegFailedError, Tools
from rytp.render.report import MissingWord, SubstitutionRef

from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_plan import fragment, source_video

WIDE = probe_json(1280, 720)


def kit(**kw: object) -> tuple[Tools, RecordingRunner]:
    runner = RecordingRunner(geometry=WIDE, loudness=loudnorm_json(-23.0), **kw)  # type: ignore[arg-type]
    return Tools.faked(runner), runner


def two_fragment_request(vid: int) -> RUN.RenderRequest:
    return RUN.RenderRequest(
        name="demo",
        target_text="мы всё исправим",
        fragments=(fragment(vid, 0, 1_000, text="мы всё"),
                   fragment(vid, 5_000, 5_500, text="исправит")),
        missing=(
            MissingWord(
                text="исправим",
                position=2,
                substitutions=(
                    SubstitutionRef(
                        text="исправит", reason="edit", video_id=vid,
                        start_ms=5_000, end_ms=5_500, occurrences=3,
                    ),
                ),
            ),
        ),
    )


def test_a_render_runs_one_command_per_fragment_then_measures_then_concats(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    encodes = [c for c in runner.calls if "-filter_complex" in c]
    assert len(encodes) == 2
    assert len(runner.commands_containing("concat")) == 2  # measure, then join
    assert result.output_path.exists()
    assert result.fragment_count == 2


def test_each_fragment_command_carries_the_canvas_the_gap_and_the_gain(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(gap_ms=200), tools=tools
    )
    first = next(c for c in runner.calls if "-filter_complex" in c)
    graph = first[first.index("-filter_complex") + 1]
    assert "scale=1280:720" in graph
    assert "tpad=stop_mode=clone:stop_duration=0.200" in graph
    assert "apad=pad_dur=0.200" in graph
    assert "volume=7dB" in graph  # -23 LUFS measured against a -16 target


def test_the_last_fragment_has_no_freeze_frame(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(gap_ms=200), tools=tools
    )
    last = [c for c in runner.calls if "-filter_complex" in c][-1]
    graph = last[last.index("-filter_complex") + 1]
    assert "tpad" not in graph and "apad" not in graph


def test_the_concat_list_holds_the_intermediates_in_order(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(keep_intermediates=True),
        tools=tools,
    )
    listing = (result.plan.list_file).read_text(encoding="utf-8").splitlines()
    assert listing[0].endswith("0000.mkv'")
    assert listing[1].endswith("0001.mkv'")


def test_the_report_lands_beside_the_output_and_names_everything(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    assert result.report_path.parent == result.output_path.parent
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Fragments" in text
    assert "## Words not found" in text and "исправим" in text
    assert "## Description" in text
    assert f"video {vid}" in text


def test_a_dry_run_writes_nothing_and_runs_no_encode(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools, dry_run=True)
    assert not result.output_path.exists()
    assert not result.report_path.exists()
    assert [c for c in runner.calls if "-filter_complex" in c] == []
    assert result.plan.duration_ms > 0


def test_intermediates_are_removed_on_success(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=tools)
    assert not result.plan.fragments_dir.exists()
    assert result.output_path.exists()


def test_keep_intermediates_leaves_them_for_inspection(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(keep_intermediates=True),
        tools=tools,
    )
    assert sorted(p.name for p in result.plan.fragments_dir.iterdir()) == [
        "0000.mkv", "0001.mkv"
    ]


def test_no_loudnorm_skips_both_scans_and_the_filter(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    tools, runner = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), RUN.RenderOptions(loudnorm=False), tools=tools
    )
    assert runner.measurements == []
    final = runner.calls[-1]
    assert "-af" not in final
    assert "-c:a" in final  # audio is still re-encoded; see the plan's decisions
    assert "left alone" in result.report_path.read_text(encoding="utf-8")


def test_an_unmeasurable_programme_still_renders_and_says_so(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    runner = RecordingRunner(
        geometry=WIDE,
        # Scanned in order: the programme scan is the one reading concat.txt.
        loudness={"concat.txt": '{"input_i" : "-inf"}', "-i": loudnorm_json(-23.0)},
    )
    result = RUN.render_cutlist(db, two_fragment_request(vid), tools=Tools.faked(runner))
    assert result.output_path.exists()
    assert "could not be measured" in result.report_path.read_text(encoding="utf-8")


def test_a_failed_fragment_stops_the_render_with_ffmpegs_complaint(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    runner = RecordingRunner(
        returncode=1, stderr="Invalid argument", geometry=WIDE,
        loudness=loudnorm_json(-23.0), fail_when="-filter_complex",
    )
    with pytest.raises(FfmpegFailedError, match="Invalid argument"):
        RUN.render_cutlist(db, two_fragment_request(vid), tools=Tools.faked(runner))


def test_the_job_payload_carries_the_options_and_nothing_else() -> None:
    options = RUN.RenderOptions(canvas_mode="bbox", gap_ms=0, loudnorm=False, crf=24)
    payload = RUN.payload_for(options)
    assert "cutlist" not in payload  # that lives in the renders row
    assert RUN.options_from_payload(payload) == options


def test_a_render_row_opens_planned_and_closes_rendered(db: Database) -> None:
    render_id = RUN.create_render(
        db, cutlist_name="demo", options=RUN.RenderOptions(canvas_mode="bbox")
    )
    row = RUN.get_render(db, render_id)
    assert (row["cutlist_name"], row["state"], row["canvas_mode"]) == (
        "demo", "planned", "bbox"
    )
    assert row["created_at"] and row["finished_at"] is None
    RUN.finish_render(db, render_id, state="rendered", output_path=Path("/out/o.mp4"))
    row = RUN.get_render(db, render_id)
    assert row["state"] == "rendered"
    assert row["output_path"] == "/out/o.mp4"
    assert row["finished_at"]


def test_an_unknown_render_is_a_not_found_error(db: Database) -> None:
    from rytp.models import NotFoundError

    with pytest.raises(NotFoundError, match="404404"):
        RUN.get_render(db, 404404)


def test_a_finished_render_records_its_output(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    row = RUN.get_render(db, render_id)
    assert row["state"] == "rendered"
    assert row["output_path"] == str(result.output_path)


def test_a_failed_render_marks_its_row_failed(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    runner = RecordingRunner(
        returncode=1, stderr="Invalid argument", geometry=WIDE,
        loudness=loudnorm_json(-23.0), fail_when="-filter_complex",
    )
    with pytest.raises(FfmpegFailedError):
        RUN.render_cutlist(
            db, two_fragment_request(vid), tools=Tools.faked(runner),
            render_row_id=render_id,
        )
    assert RUN.get_render(db, render_id)["state"] == "failed"


def test_a_dry_run_removal_counts_the_bytes_and_deletes_nothing(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    removal = RUN.remove_render(db, render_id, dry_run=True)
    assert removal.removed is False
    assert removal.cutlist_name == "demo"
    assert removal.total_bytes > 0
    assert any(path.endswith("output.mp4") for path, _ in removal.files)
    assert any(path.endswith("report.md") for path, _ in removal.files)
    assert result.output_path.exists()
    assert RUN.get_render(db, render_id)["id"] == render_id


def test_removal_takes_the_directory_the_row_and_any_queued_job(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    from rytp.jobs.queue import enqueue
    from rytp.models import NotFoundError

    vid = source_video(db, tmp_path, "VIDEO_A")
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    tools, _ = kit()
    result = RUN.render_cutlist(
        db, two_fragment_request(vid), tools=tools, render_row_id=render_id
    )
    enqueue(db, "render", render_id, payload=RUN.payload_for(RUN.RenderOptions()))
    removal = RUN.remove_render(db, render_id)
    assert removal.removed is True
    assert removal.jobs_cancelled == 1
    assert not result.output_path.parent.exists()
    assert db.conn.execute(
        "SELECT state FROM jobs WHERE kind = 'render' AND target_id = ?", (render_id,)
    ).fetchone()["state"] == "cancelled"
    with pytest.raises(NotFoundError):
        RUN.get_render(db, render_id)


def test_removal_notices_a_cut_list_that_is_already_gone(
    db: Database, data_dir: None
) -> None:
    """Part 5's `assemble.remove` warns and leaves the row; this is the mirror."""
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    assert RUN.plan_removal(db, render_id).cutlist_present is False


def test_removal_refuses_to_delete_outside_the_data_tree(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    """`output_path` is text a human can edit; it is not a licence to rmtree."""
    render_id = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    elsewhere = tmp_path / "precious"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("mine", encoding="utf-8")
    db.conn.execute(
        "UPDATE renders SET output_path = ? WHERE id = ?",
        (str(elsewhere / "output.mp4"), render_id),
    )
    db.conn.commit()
    removal = RUN.remove_render(db, render_id)
    assert removal.output_dir is None
    assert removal.files == ()
    assert (elsewhere / "keep.txt").exists()  # untouched
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0


def test_the_job_reads_its_cut_list_from_the_row(db: Database, data_dir: None) -> None:
    render_id = RUN.create_render(db, cutlist_name="ghost", options=RUN.RenderOptions())
    with pytest.raises(RUN.RenderError, match="not found"):
        RUN.run_render_job(db, render_id, RUN.payload_for(RUN.RenderOptions()))
    assert RUN.get_render(db, render_id)["state"] == "failed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_run.py -q`
Expected: collection error, `ImportError: cannot import name 'render_cutlist'` (or `AttributeError` on `RUN.render_cutlist`).

- [ ] **Step 3: Append to `rytp/render/run.py`**

Extend the imports at the top of the file with `shutil`, `rytp.models.{NotFoundError, utc_now_iso}`, `rytp.render.canvas.video_filter_chain`, `rytp.render.ffmpeg.{FfmpegFailedError, concat_command, fragment_command, programme_loudness_command, write_concat_list}` and `rytp.render.report.{FragmentReport, RenderReport, SourceReport, render_markdown}`.

```python
# ---------------------------------------------------------------------------
# Execution. Everything above decided; this part only does.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderResult:
    """What a finished render hands back to a command or a job."""

    render_id: str
    output_path: Path
    report_path: Path
    duration_ms: int
    fragment_count: int
    source_count: int
    plan: RenderPlan


def _speaker_of(fragment: RenderFragment) -> str | None:
    """A human-readable speaker, or the diarized label, or nothing."""
    if fragment.speaker_label:
        return fragment.speaker_label
    if fragment.video_speaker_id is not None:
        return f"label {fragment.video_speaker_id}"
    return None


def build_report(
    plan: RenderPlan, *, notes: Sequence[str], created_at: str
) -> RenderReport:
    """Turn a plan plus what execution learned into the report tree."""
    fragments = tuple(
        FragmentReport(
            ord=planned.ord,
            video_id=planned.source.video_id,
            video_title=planned.source.title,
            video_url=planned.source.url,
            speaker=_speaker_of(planned.fragment),
            source_start_ms=planned.fragment.start_ms,
            source_end_ms=planned.fragment.end_ms,
            output_start_ms=planned.output_start_ms,
            output_end_ms=planned.output_end_ms,
            gap_after_ms=planned.gap_after_ms,
            gap_origin=planned.gap_origin,
            text=planned.fragment.text,
        )
        for planned in plan.fragments
    )
    sources = []
    for source in plan.sources:
        mine = [p for p in plan.fragments if p.source.video_id == source.video_id]
        geometry = source.geometry
        sources.append(
            SourceReport(
                video_id=source.video_id,
                title=source.title,
                url=source.url,
                fragment_count=len(mine),
                used_ms=sum(p.fragment.duration_ms for p in mine),
                measured_lufs=source.measured_lufs,
                gain_db=source.gain_db,
                first_output_ms=min(p.output_start_ms for p in mine),
                geometry=f"{geometry.width}x{geometry.height} @ {round(geometry.fps)} fps",
            )
        )
    return RenderReport(
        render_id=plan.render_id,
        cutlist=plan.request.name,
        created_at=created_at,
        output_path=str(plan.output_path),
        duration_ms=plan.duration_ms,
        canvas=plan.canvas.label,
        loudnorm=plan.options.loudnorm,
        gap_policy=plan.options.gap_policy,
        target_text=plan.request.target_text,
        assembled_text=plan.request.assembled_text,
        fragments=fragments,
        sources=tuple(sources),
        missing=plan.request.missing,
        notes=tuple(notes),
    )


def render_cutlist(
    db: Database,
    request: RenderRequest,
    options: RenderOptions | None = None,
    *,
    tools: Tools | None = None,
    dry_run: bool = False,
    render_row_id: int | None = None,
) -> RenderResult:
    """Plan, then execute: ``output/{render_id}/output.mp4`` and its report.

    One encode per fragment into a concat-safe intermediate, then one
    joining pass. ``dry_run`` stops after planning — same answer, no
    files, no encoder — which is how you check the canvas, the gaps and
    the sources before committing an hour of CPU to them.

    ``render_row_id`` is a ``renders`` row opened by the caller; when
    given it is closed ``rendered`` or ``failed`` here, so a crash
    mid-encode leaves the row saying so rather than saying nothing.
    """
    options = options or RenderOptions()
    tools = tools or Tools.resolve()
    try:
        plan = plan_render(db, request, options, tools=tools)
    except Exception:
        if render_row_id is not None:
            finish_render(db, render_row_id, state="failed")
        raise
    notes = list(plan.notes)

    if dry_run:
        return RenderResult(
            render_id=plan.render_id,
            output_path=plan.output_path,
            report_path=plan.report_path,
            duration_ms=plan.duration_ms,
            fragment_count=len(plan.fragments),
            source_count=len(plan.sources),
            plan=plan,
        )

    try:
        _encode(plan, options, tools, notes)
    except Exception:
        if render_row_id is not None:
            finish_render(db, render_row_id, state="failed")
        raise
    if render_row_id is not None:
        finish_render(
            db, render_row_id, state="rendered", output_path=plan.output_path
        )

    return RenderResult(
        render_id=plan.render_id,
        output_path=plan.output_path,
        report_path=plan.report_path,
        duration_ms=plan.duration_ms,
        fragment_count=len(plan.fragments),
        source_count=len(plan.sources),
        plan=plan,
    )


def _encode(
    plan: RenderPlan, options: RenderOptions, tools: Tools, notes: list[str]
) -> None:
    """Everything that touches the disk: fragments, join, report, cleanup."""
    config.ensure_dir(plan.fragments_dir)
    total = len(plan.fragments)
    for planned in plan.fragments:
        run_command(
            fragment_command(
                video_path=planned.source.video_path,
                audio_path=planned.source.audio_path,
                start_ms=planned.fragment.start_ms,
                end_ms=planned.fragment.end_ms,
                video_filter=video_filter_chain(
                    plan.canvas, gap_ms=planned.gap_after_ms
                ),
                out_path=planned.intermediate,
                gain_db=planned.source.gain_db,
                gap_ms=planned.gap_after_ms,
                preset=options.preset,
                crf=options.crf,
                binary=tools.ffmpeg,
            ),
            runner=tools.runner,
            what=f"fragment {planned.ord + 1}/{total}",
        )

    write_concat_list(plan.list_file, [p.intermediate for p in plan.fragments])

    measured = None
    if options.loudnorm:
        scan = run_command(
            programme_loudness_command(
                list_file=plan.list_file, binary=tools.ffmpeg
            ),
            runner=tools.runner,
            what="loudness scan of the assembled programme",
        )
        measured = parse_loudnorm_json(scan.stderr)
        if measured is None:
            notes.append(
                "the assembled programme could not be measured; its level was "
                "left alone"
            )

    run_command(
        concat_command(
            list_file=plan.list_file,
            out_path=plan.output_path,
            measured=measured,
            binary=tools.ffmpeg,
        ),
        runner=tools.runner,
        what="joining the fragments",
    )
    if not plan.output_path.exists():
        raise FfmpegFailedError(
            f"ffmpeg reported success but wrote no {plan.output_path.name}"
        )

    plan.report_path.write_text(
        render_markdown(build_report(plan, notes=notes, created_at=utc_now_iso())),
        encoding="utf-8",
    )

    if not options.keep_intermediates:
        # Kept on failure, because that is when they are worth looking at.
        shutil.rmtree(plan.fragments_dir, ignore_errors=True)
        plan.list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The job entry point (contracts §5 "Job handlers"; Part 2 owns the registry)
# ---------------------------------------------------------------------------


def create_render(
    db: Database,
    *,
    cutlist_name: str,
    options: RenderOptions,
    now: str | None = None,
) -> int:
    """Open a ``renders`` row and return its id — the job's ``target_id``.

    Contracts §3 gives a render its own identity, so a job addresses it
    the way every other kind addresses its target: an integer primary
    key, no hashing of names into integers and nothing to verify later.
    The row is opened before any work, so a render that dies mid-encode
    is still on record as ``planned``.
    """
    with db.transaction():
        cursor = db.conn.execute(
            "INSERT INTO renders (cutlist_name, output_path, canvas_mode, state, "
            "created_at) VALUES (?, NULL, ?, 'planned', ?)",
            (cutlist_name, options.canvas_mode, now or utc_now_iso()),
        )
        return int(cursor.lastrowid)


def get_render(db: Database, render_id: int) -> sqlite3.Row:
    """One render row, or :class:`NotFoundError`."""
    row = db.conn.execute(
        "SELECT * FROM renders WHERE id = ?", (render_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"no render {render_id}")
    return row


def finish_render(
    db: Database,
    render_id: int,
    *,
    state: str,
    output_path: Path | None = None,
    now: str | None = None,
) -> None:
    """Close a render row as ``rendered`` or ``failed``."""
    with db.transaction():
        db.conn.execute(
            "UPDATE renders SET state = ?, "
            "output_path = COALESCE(?, output_path), finished_at = ? WHERE id = ?",
            (
                state,
                None if output_path is None else str(output_path),
                now or utc_now_iso(),
                render_id,
            ),
        )


def payload_for(options: RenderOptions) -> dict[str, object]:
    """The ``jobs.payload_json`` body for one render.

    Options only: which cut list it is lives in the ``renders`` row the
    job targets, so there is one place for it rather than two that can
    disagree.
    """
    return {
        "canvas_mode": options.canvas_mode,
        "height": options.height,
        "fps": options.fps,
        "gap_ms": options.gap_ms,
        "loudnorm": options.loudnorm,
        "preset": options.preset,
        "crf": options.crf,
        "keep_intermediates": options.keep_intermediates,
        "render_id": options.render_id,
    }


def options_from_payload(payload: dict) -> RenderOptions:
    """Rebuild the options a queued render was enqueued with."""
    defaults = RenderOptions()
    return RenderOptions(
        canvas_mode=str(payload.get("canvas_mode", defaults.canvas_mode)),
        height=int(payload.get("height", defaults.height)),
        fps=int(payload.get("fps", defaults.fps)),
        gap_ms=int(payload.get("gap_ms", defaults.gap_ms)),
        loudnorm=bool(payload.get("loudnorm", defaults.loudnorm)),
        preset=str(payload.get("preset", defaults.preset)),
        crf=int(payload.get("crf", defaults.crf)),
        keep_intermediates=bool(
            payload.get("keep_intermediates", defaults.keep_intermediates)
        ),
        render_id=str(payload.get("render_id", defaults.render_id)),
    )


def run_render_job(db: Database, target_id: int, payload: dict) -> None:
    """The ``render`` job kind's handler. contracts §5.

    ``target_id`` is a ``renders.id``: the row says which cut list this
    is, so nothing is re-derived from the payload and nothing needs
    checking against it. Idempotent — the output path is derived from
    the cut list and the options, so a repeat overwrites the same file.
    """
    row = get_render(db, target_id)
    name = str(row["cutlist_name"])
    path = config.paths().cutlist(name)
    if not path.exists():
        finish_render(db, target_id, state="failed")
        raise RenderError(f"cut list {name!r} not found at {path}")
    # Lazy, and one of only two import sites for Part 5 (contracts §1).
    from rytp.assemble.cutlist import load_cutlist

    render_cutlist(
        db,
        request_from_cutlist(load_cutlist(path)),
        options_from_payload(payload),
        render_row_id=target_id,
    )


# ---------------------------------------------------------------------------
# Removal (contracts §5 "Deletion")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderRemoval:
    """What removing one render would take, or did take."""

    render_id: int
    cutlist_name: str
    cutlist_present: bool
    output_dir: Path | None
    files: tuple[tuple[str, int], ...]  # (path, bytes)
    total_bytes: int
    jobs_cancelled: int
    removed: bool


def _removable_output_dir(row: sqlite3.Row) -> Path | None:
    """The directory this render owns, if it is safely inside the tree.

    ``renders.output_path`` is text a human can edit, so the path is
    checked against the data tree's ``output/`` root before anything
    recursive happens to it. Outside the tree — or empty — means there
    is nothing this command is willing to delete, and only the row goes.
    """
    raw = row["output_path"]
    if not raw:
        return None
    candidate = Path(str(raw)).parent
    # Paths() has no output root accessor; every render directory is a
    # child of this one, which is the same thing.
    root = config.paths().output_dir("_").parent
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def plan_removal(db: Database, render_id: int) -> RenderRemoval:
    """Exactly what :func:`remove_render` would do. Touches nothing."""
    row = get_render(db, render_id)
    name = str(row["cutlist_name"])
    directory = _removable_output_dir(row)
    files: list[tuple[str, int]] = []
    if directory is not None and directory.is_dir():
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            files.append((str(path), path.stat().st_size))
    queued = db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'render' AND target_id = ? "
        "AND state IN ('pending', 'blocked', 'running')",
        (render_id,),
    ).fetchone()[0]
    return RenderRemoval(
        render_id=render_id,
        cutlist_name=name,
        cutlist_present=config.paths().cutlist(name).exists(),
        output_dir=directory,
        files=tuple(files),
        total_bytes=sum(size for _, size in files),
        jobs_cancelled=int(queued),
        removed=False,
    )


def remove_render(
    db: Database, render_id: int, *, dry_run: bool = False
) -> RenderRemoval:
    """Delete a render's output directory and its row. contracts §5.

    Synchronous, never a job: a half-deleted render recovered from a
    crashed queue is worse than a slow command. The job cancellation and
    the row deletion share one transaction because ``jobs`` has no
    foreign key to lean on — a surviving job would target a render that
    no longer exists.
    """
    plan = plan_removal(db, render_id)
    if dry_run:
        return plan
    if plan.output_dir is not None:
        shutil.rmtree(plan.output_dir, ignore_errors=True)
    with db.transaction():
        db.conn.execute(
            "UPDATE jobs SET state = 'cancelled', finished_at = ? "
            "WHERE kind = 'render' AND target_id = ? "
            "AND state IN ('pending', 'blocked', 'running')",
            (utc_now_iso(), render_id),
        )
        db.conn.execute("DELETE FROM renders WHERE id = ?", (render_id,))
    return replace(plan, removed=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_run.py -q`
Expected: PASS, 23 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/render/run.py tests/test_render_run.py
git commit -m "feat: render a cut list into output.mp4 and report.md"
```

---

### Task 8: The `render` job kind

Design §5 puts `render` on the `cpu` pool; contracts §5 makes the job-kind → handler mapping itself a contract. Part 2 owns `rytp/jobs/`, and its author specified exactly how another part contributes a kind, so this task follows that shape rather than inventing one.

**Two things that are not obvious:**

- **The predicate is a real one, because `target_id` is a real row.** Readiness predicates never see `payload_json` — a predicate that changed its answer based on how a job was enqueued would stop being a statement about the world, and design §5's self-healing property would go with it. With `target_id = renders.id` the predicate needs no payload: the row names the cut list, so it can answer SATISFIED when the output file is on disk, BLOCKED when the cut list has been deleted, READY otherwise. Delete `output.mp4` and the job becomes runnable again, exactly like pruning a cached WAV — which is why `reopenable` stays at its default rather than being switched off.
- **`target_kind="render"`.** The worker calls `unblock(target_id=..., target_kind=...)` after each finished job; the namespace is what stops render id 7 re-evaluating video id 7's jobs.

Registration lives **inside** `rytp/jobs/__init__.py` with a lazy thunk, like Part 2's own three kinds: if that module imported `rytp.render.run` at import time, every CLI invocation would pull in the whole render stage.

**Files:**
- Create: `rytp/render/readiness.py`
- Modify: `rytp/jobs/__init__.py` (one thunk, one import line in the bottom block, one `register_job_kind` call)
- Test: `tests/test_render_job.py`

**Interfaces:**
- Consumes: `rytp.jobs.{Readiness, JobKind, register_job_kind, JOB_HANDLERS, resolve_job_kind}`; `rytp.render.run.run_render_job`.
- Consumes also: `rytp.config.paths` and the `renders` table.
- Produces: `rytp.render.readiness.render_readiness(db: Database, target_id: int) -> Readiness`; the registered `render` kind.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_job.py`:

```python
"""The render job kind, contributed to Part 2's registry (contracts §5)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rytp.config import paths
from rytp.db import Database
from rytp.jobs import JOB_HANDLERS, Readiness, resolve_job_kind
from rytp.render.readiness import render_readiness

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_render_kind_is_registered_on_the_cpu_pool() -> None:
    kind = resolve_job_kind("render")
    assert kind.pool == "cpu"
    assert kind.target_kind == "render"
    assert kind.summary


def test_the_handler_is_in_the_contracted_flat_mapping() -> None:
    assert "render" in JOB_HANDLERS
    assert callable(JOB_HANDLERS["render"])


def _render_row(db: Database, name: str = "demo") -> int:
    cursor = db.conn.execute(
        "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at) "
        "VALUES (?, '16:9', 'planned', '2026-01-01T00:00:00+00:00')",
        (name,),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def test_an_unknown_render_is_blocked(db: Database) -> None:
    assert render_readiness(db, 999_999) is Readiness.BLOCKED


def test_a_planned_render_with_its_cut_list_on_disk_is_ready(
    db: Database, data_dir: None
) -> None:
    render_id = _render_row(db)
    cutlist = paths().cutlist("demo")
    cutlist.parent.mkdir(parents=True, exist_ok=True)
    cutlist.write_text("schema_version = 1\n", encoding="utf-8")
    assert render_readiness(db, render_id) is Readiness.READY


def test_a_render_whose_cut_list_is_gone_is_blocked(
    db: Database, data_dir: None
) -> None:
    assert render_readiness(db, _render_row(db)) is Readiness.BLOCKED


def test_a_finished_render_is_satisfied_until_its_output_disappears(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    render_id = _render_row(db)
    cutlist = paths().cutlist("demo")
    cutlist.parent.mkdir(parents=True, exist_ok=True)
    cutlist.write_text("schema_version = 1\n", encoding="utf-8")
    output = tmp_path / "output.mp4"
    output.write_bytes(b"\x00")
    db.conn.execute(
        "UPDATE renders SET state = 'rendered', output_path = ? WHERE id = ?",
        (str(output), render_id),
    )
    db.conn.commit()
    assert render_readiness(db, render_id) is Readiness.SATISFIED
    output.unlink()
    # Self-healing, design §5: the world changed, so the job is work again.
    assert render_readiness(db, render_id) is Readiness.READY


def test_the_handler_delegates_to_the_stage(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, dict]] = []
    import rytp.render.run as run_module

    monkeypatch.setattr(
        run_module,
        "run_render_job",
        lambda _db, target_id, payload: seen.append((target_id, payload)),
    )
    result = JOB_HANDLERS["render"](db, 7, {"cutlist": "demo"})
    assert result is None  # contracts §5: handlers return None
    assert seen == [(7, {"cutlist": "demo"})]


def test_the_predicate_module_imports_cold(tmp_path: Path) -> None:
    """`rytp.render` imports nothing, so this module must not either."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import rytp.render.readiness as m; print(m.render_readiness.__name__)"],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "render_readiness" in proc.stdout


def test_importing_the_job_registry_does_not_import_the_render_stage() -> None:
    """A lazy thunk, so `rytp <anything>` does not pull in the render stage."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.jobs; "
            "assert 'rytp.render.run' not in sys.modules, sorted(sys.modules); "
            "print('clean')",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_render_job.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.render.readiness'`.

- [ ] **Step 3: Write `rytp/render/readiness.py`**

```python
"""Readiness for the ``render`` job kind. design §5.

Import-light on purpose: ``rytp/jobs/__init__.py`` imports this at
module level, so it may not reach for ffmpeg, the canvas, or anything
else the render stage needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rytp.config import paths
from rytp.db import Database

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rytp.jobs import Readiness


def render_readiness(db: Database, target_id: int) -> "Readiness":
    """What the database and the disk say about one render right now.

    ``target_id`` is a ``renders.id``, so this is an ordinary predicate
    over the world, with no need for the payload design §5 forbids it
    from reading: the row names the cut list, the row records the
    output. Delete the output file and the job becomes runnable again —
    the same self-healing behaviour as pruning a cached WAV.

    The ``Readiness`` import is **inside** the function on purpose, and
    must stay there. Part 2's own ``rytp/jobs/readiness.py`` can import
    it at module level because its parent package is the importer —
    ``rytp.jobs`` is always initialised first. This module's parent is
    ``rytp.render``, which imports nothing, so a cold
    ``import rytp.render.readiness`` would run ``rytp.jobs``' bottom
    block, which imports this half-initialised module straight back and
    fails. Do not let a cleanup pass hoist it.
    """
    from rytp.jobs import Readiness

    row = db.conn.execute(
        "SELECT cutlist_name, output_path, state FROM renders WHERE id = ?",
        (target_id,),
    ).fetchone()
    if row is None:
        return Readiness.BLOCKED
    output = row["output_path"]
    if row["state"] == "rendered" and output and Path(output).exists():
        return Readiness.SATISFIED
    if not paths().cutlist(str(row["cutlist_name"])).exists():
        return Readiness.BLOCKED
    return Readiness.READY
```

- [ ] **Step 4: Register the kind in `rytp/jobs/__init__.py`**

Add the thunk beside Part 2's `_run_download` / `_run_captions` / `_run_extract_wav`:

```python
def _run_render(db: Database, target_id: int, payload: dict) -> None:
    from rytp.render.run import run_render_job

    run_render_job(db, target_id, payload)
```

Add `render_readiness` to the bottom import block, next to Part 2's predicates:

```python
from rytp.render.readiness import render_readiness  # noqa: E402
```

And register it with the other kinds:

```python
register_job_kind(
    JobKind(
        name="render",
        pool="cpu",
        readiness=render_readiness,
        handler=_run_render,
        summary="cut, normalise and join a cut list into one file",
        target_kind="render",
    )
)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_render_job.py tests/test_jobs_readiness.py tests/test_jobs_queue.py -q`
Expected: PASS — the new module plus Part 2's suites still green.

- [ ] **Step 6: Commit**

```bash
git add rytp/render/readiness.py rytp/jobs/__init__.py tests/test_render_job.py
git commit -m "feat: register the render job kind on the cpu pool"
```

---

### Task 9: The `render.run`, `render.pauses`, `render.list` and `render.remove` commands

Contracts §5: one definition, both surfaces. Handlers take an open `Database` first, take keyword arguments matching `Param.name`, never print, never exit, and raise for an expected failure.

`render.run` is `long_running=True` so the TUI does not try to run it inline (contracts §5), and it can either render in the foreground or drop a job on the queue with `--queue`.

`render.pauses` exists because of the tall-ask caveat in design §9: before trusting a measured gap, the owner should be able to look at the distribution and see whether it is real. It prints the sample count, the median, the share of exact zeros, and the verdict — and it is also how you find out that a video needs aligning before its pauses mean anything.

`render.remove` is contracts §5's deletion for this group: the output directory and the row, `--dry-run` to see what would go with byte counts and which cut list made it, `--yes` to mean it. **Agreed with the Part 5 author** (we proposed the same thing independently): removing a *cut list* warns and leaves `renders` rows alone rather than refusing, because `renders.cutlist_name` is a name and not a foreign key precisely so history outlives its source — the same shape as `channel.remove` orphaning videos and `speakers.remove` nulling labels. So `render.remove` is the only thing that deletes a render, `assemble.remove` is the only thing that deletes a cut list, and neither cascades into the other. A render whose cut list is gone still lists, still removes, and if it is queued the readiness predicate parks it as BLOCKED rather than failing; `render.remove --dry-run` says so.

`render.list` is one query over the `renders` table. It exists because the table does: the redesign dropped `splice_runs` and left render history nowhere, and "what did I make, from what, and where is it" is the question the owner will actually ask. No pagination beyond `--limit`, no filters — that is what the table is for if it is ever wanted.

**Files:**
- Create: `rytp/commands/render.py`
- Modify: `rytp/commands/__init__.py` (one line in the existing sibling-import block)
- Test: `tests/test_commands_render.py`

**Interfaces:**
- Consumes: `rytp.commands.{REQUIRED, Param, Command, CommandResult, register}`; `rytp.config.paths`; `rytp.models.{InvalidInputError, NotFoundError}`; `rytp.render.run.*`; `rytp.render.pauses.*`; `rytp.jobs.queue.enqueue`.
- Consumes also: `rytp.db.queries.clamp_limit`; `rytp.render.run.{create_render, payload_for, remove_render}`.
- Produces: registered commands `render.run`, `render.pauses`, `render.list` and `render.remove`; `rytp.commands.render._load_request(name) -> RenderRequest` (the second and last Part 5 import site).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_commands_render.py`:

```python
"""The render commands: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.render  # noqa: F401 - importing registers the commands
from rytp.commands import COMMANDS, resolve
from rytp.db import Database
from rytp.models import RytpError
from rytp.render import run as RUN
from rytp.render.ffmpeg import Tools

from tests.render_fakes import RecordingRunner, loudnorm_json, probe_json
from tests.test_render_pauses import add_speaker, add_words, evenly_spaced
from tests.test_render_plan import fragment, source_video


def test_every_command_is_registered_in_the_render_group() -> None:
    for name in ("render.run", "render.pauses", "render.list", "render.remove"):
        assert name in COMMANDS
        assert COMMANDS[name].group == "render"
        assert COMMANDS[name].summary


def test_render_run_is_marked_long_running() -> None:
    assert COMMANDS["render.run"].long_running is True


def test_render_run_exposes_every_design_knob() -> None:
    names = {p.name for p in COMMANDS["render.run"].params}
    assert {
        "name", "canvas", "height", "fps", "gap_ms", "loudnorm",
        "preset", "crf", "keep_intermediates", "render_id", "dry_run", "queue",
    } <= names
    canvas = next(p for p in COMMANDS["render.run"].params if p.name == "canvas")
    assert canvas.choices == ("16:9", "bbox")


def test_render_run_renders_and_reports(
    db: Database, tmp_path: Path, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(
        name="demo", target_text="мы всё",
        fragments=(fragment(vid, 0, 1_000, text="мы всё"),),
    )
    runner = RecordingRunner(geometry=probe_json(1280, 720),
                             loudness=loudnorm_json(-23.0))
    monkeypatch.setattr(
        "rytp.commands.render._load_request", lambda _name: request
    )
    monkeypatch.setattr(
        "rytp.commands.render._tools", lambda: Tools.faked(runner)
    )
    result = resolve("render.run").handler(db, name="demo")
    assert result.rows
    assert "output.mp4" in " ".join(result.rows[0])
    assert Path(result.rows[0][1]).exists()


def test_render_run_dry_run_writes_nothing(
    db: Database, tmp_path: Path, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    request = RUN.RenderRequest(name="demo", fragments=(fragment(vid, 0, 1_000),))
    runner = RecordingRunner(geometry=probe_json(1280, 720),
                             loudness=loudnorm_json(-23.0))
    monkeypatch.setattr("rytp.commands.render._load_request", lambda _name: request)
    monkeypatch.setattr("rytp.commands.render._tools", lambda: Tools.faked(runner))
    result = resolve("render.run").handler(db, name="demo", dry_run=True)
    assert "dry run" in (result.message or "")
    assert not Path(result.rows[0][1]).exists()


def test_render_run_can_queue_instead_of_rendering(
    db: Database, data_dir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rytp.commands.render._load_request",
        lambda _name: RUN.RenderRequest(name="demo", fragments=(fragment(1, 0, 10),)),
    )
    result = resolve("render.run").handler(db, name="demo", queue=True)
    assert "queued" in (result.message or "")
    render_row = db.conn.execute("SELECT id, cutlist_name, state FROM renders").fetchone()
    assert (render_row["cutlist_name"], render_row["state"]) == ("demo", "planned")
    job = db.conn.execute("SELECT kind, target_id, pool FROM jobs").fetchone()
    assert job["kind"] == "render"
    assert job["target_id"] == render_row["id"]
    assert job["pool"] == "cpu"


def test_render_list_shows_what_has_been_made(db: Database) -> None:
    first = RUN.create_render(db, cutlist_name="demo", options=RUN.RenderOptions())
    RUN.finish_render(db, first, state="rendered", output_path=Path("/out/demo/o.mp4"))
    RUN.create_render(db, cutlist_name="другой", options=RUN.RenderOptions())
    result = resolve("render.list").handler(db)
    assert len(result.rows) == 2
    assert result.rows[0][1] == "другой"  # newest first
    assert "rendered" in " ".join(result.rows[1])


def test_an_unknown_cut_list_is_a_one_line_error(db: Database, data_dir: None) -> None:
    with pytest.raises(RytpError, match="demo"):
        resolve("render.run").handler(db, name="demo")


def test_render_pauses_reports_per_speaker_and_per_video(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    label = add_speaker(db, vid, label="host")
    add_words(db, vid, evenly_spaced(80, gap_ms=220), video_speaker_id=label)
    result = resolve("render.pauses").handler(db, video_id=vid)
    flat = [" ".join(row) for row in result.rows]
    assert any("host" in row for row in flat)
    assert any("220" in row for row in flat)
    assert any("video" in row for row in flat)


def test_render_pauses_flags_a_degenerate_distribution(
    db: Database, tmp_path: Path
) -> None:
    vid = source_video(db, tmp_path, "VIDEO_A")
    add_words(db, vid, [(i * 300, (i + 1) * 300) for i in range(120)])
    result = resolve("render.pauses").handler(db, video_id=vid)
    assert any("zero" in " ".join(row) for row in result.rows)


def test_render_pauses_needs_something_to_look_at(db: Database) -> None:
    with pytest.raises(RytpError, match="video-id"):
        resolve("render.pauses").handler(db)


def _render_with_output(db: Database, *, name: str = "demo", size: int = 2_048) -> int:
    from rytp.config import paths

    render_id = RUN.create_render(db, cutlist_name=name, options=RUN.RenderOptions())
    output_dir = paths().output_dir(f"{name}-1a2b3c4d")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output.mp4").write_bytes(b"\x00" * size)
    (output_dir / "report.md").write_text("# Render", encoding="utf-8")
    db.conn.execute(
        "UPDATE renders SET output_path = ? WHERE id = ?",
        (str(output_dir / "output.mp4"), render_id),
    )
    db.conn.commit()
    return render_id


def test_render_remove_refuses_without_yes(db: Database, data_dir: None) -> None:
    render_id = _render_with_output(db)
    with pytest.raises(RytpError, match="--yes"):
        resolve("render.remove").handler(db, render_id=render_id)
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 1


def test_render_remove_dry_run_shows_the_files_and_the_cut_list(
    db: Database, data_dir: None
) -> None:
    render_id = _render_with_output(db)
    result = resolve("render.remove").handler(db, render_id=render_id, dry_run=True)
    assert "would remove" in (result.message or "")
    assert "demo" in (result.message or "")
    assert len(result.rows) == 2
    assert any("2,048" in cell for row in result.rows for cell in row)
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 1


def test_render_remove_with_yes_takes_the_directory_and_the_row(
    db: Database, data_dir: None
) -> None:
    from rytp.config import paths

    render_id = _render_with_output(db)
    result = resolve("render.remove").handler(db, render_id=render_id, yes=True)
    assert "removed render" in (result.message or "")
    assert not paths().output_dir("demo-1a2b3c4d").exists()
    assert db.conn.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_commands_render.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'rytp.commands.render'`.

- [ ] **Step 3: Write `rytp/commands/render.py`**

```python
"""The render commands. design §9, contracts §5.

Thin over :mod:`rytp.render.run`. The two seams that tests replace —
loading a cut list and resolving the binaries — are module-level
functions rather than inline calls, so the suite never needs Part 5 on
disk or ffmpeg on PATH.
"""

from __future__ import annotations

from rytp import config
from rytp import constants as C
from rytp.commands import REQUIRED, Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db.queries import clamp_limit
from rytp.models import InvalidInputError, RytpError
from rytp.render.ffmpeg import Tools
from rytp.render.pauses import (
    PauseStats,
    applied_gap_ms,
    gaps_for_speaker_label,
    gaps_for_video,
    gaps_for_video_speaker,
    summarize_gaps,
)
from rytp.render.report import format_timecode
from rytp.render.run import (
    RenderOptions,
    RenderRequest,
    create_render,
    payload_for,
    remove_render,
    render_cutlist,
    request_from_cutlist,
)


def _tools() -> Tools:
    """Resolve ffmpeg and ffprobe. A seam so tests can fake both."""
    return Tools.resolve()


def _load_request(name: str) -> RenderRequest:
    """Read a cut list by name. The last of two Part 5 import sites."""
    path = config.paths().cutlist(name)
    if not path.exists():
        raise RytpError(
            f"no cut list named {name!r} at {path}; "
            f"make one with `rytp assemble` first"
        )
    from rytp.assemble.cutlist import load_cutlist

    return request_from_cutlist(load_cutlist(path))


def _options(
    *,
    canvas: str,
    height: int,
    fps: int,
    gap_ms: int,
    loudnorm: bool,
    preset: str,
    crf: int,
    keep_intermediates: bool,
    render_id: str,
) -> RenderOptions:
    return RenderOptions(
        canvas_mode=canvas,
        height=height,
        fps=fps,
        gap_ms=gap_ms,
        loudnorm=loudnorm,
        preset=preset,
        crf=crf,
        keep_intermediates=keep_intermediates,
        render_id=render_id,
    )


def _run_handler(
    db: Database,
    *,
    name: str,
    canvas: str = C.RENDER_DEFAULT_CANVAS_MODE,
    height: int = 0,
    fps: int = 0,
    gap_ms: int = -1,
    loudnorm: bool = True,
    preset: str = C.RENDER_VIDEO_PRESET,
    crf: int = C.RENDER_VIDEO_CRF,
    keep_intermediates: bool = False,
    render_id: str = "",
    dry_run: bool = False,
    queue: bool = False,
) -> CommandResult:
    options = _options(
        canvas=canvas, height=height, fps=fps, gap_ms=gap_ms, loudnorm=loudnorm,
        preset=preset, crf=crf, keep_intermediates=keep_intermediates,
        render_id=render_id,
    )
    request = _load_request(name)
    if queue:
        from rytp.jobs.queue import enqueue

        render_id = create_render(db, cutlist_name=name, options=options)
        job_id = enqueue(db, "render", render_id, payload=payload_for(options))
        return CommandResult(
            columns=("job", "render", "cut list"),
            rows=((str(job_id), str(render_id), name),),
            message=f"queued as job {job_id}; run `rytp worker` to drain it",
        )
    # A dry run decides nothing durable, so it opens no row.
    render_id = None if dry_run else create_render(
        db, cutlist_name=name, options=options
    )
    result = render_cutlist(
        db, request, options, tools=_tools(), dry_run=dry_run, render_row_id=render_id
    )
    return CommandResult(
        columns=("render", "output", "report", "fragments", "sources", "duration"),
        rows=(
            (
                result.render_id,
                str(result.output_path),
                str(result.report_path),
                str(result.fragment_count),
                str(result.source_count),
                format_timecode(result.duration_ms),
            ),
        ),
        message="dry run: nothing was written" if dry_run else None,
    )


def _pause_row(stats: PauseStats) -> tuple[str, ...]:
    return (
        stats.scope,
        stats.key,
        str(stats.n_samples),
        str(stats.median_ms),
        f"{stats.zero_fraction:.0%}",
        str(applied_gap_ms(stats)),
        stats.reason or "usable",
    )


def _pauses_handler(
    db: Database, *, video_id: int = 0, speaker: str = ""
) -> CommandResult:
    if not video_id and not speaker:
        raise InvalidInputError("pass --video-id, --speaker, or both")
    rows: list[tuple[str, ...]] = []
    if speaker:
        rows.append(
            _pause_row(
                summarize_gaps(
                    gaps_for_speaker_label(db, speaker), scope="speaker", key=speaker
                )
            )
        )
    if video_id:
        labels = db.conn.execute(
            "SELECT vs.id, vs.local_label, s.label AS roster "
            "FROM video_speakers vs LEFT JOIN speakers s ON s.id = vs.speaker_id "
            "WHERE vs.video_id = ? ORDER BY vs.id",
            (video_id,),
        ).fetchall()
        for label in labels:
            key = label["roster"] or label["local_label"]
            rows.append(
                _pause_row(
                    summarize_gaps(
                        gaps_for_video_speaker(db, int(label["id"])),
                        scope="video_speaker",
                        key=str(key),
                    )
                )
            )
        rows.append(
            _pause_row(
                summarize_gaps(
                    gaps_for_video(db, video_id), scope="video", key=str(video_id)
                )
            )
        )
    return CommandResult(
        columns=(
            "scope", "key", "samples", "median ms", "zeros", "would insert", "verdict"
        ),
        rows=tuple(rows),
    )


register(
    Command(
        name="render.run",
        group="render",
        summary="Cut, normalise and join a cut list into one uploadable file.",
        params=(
            Param(
                name="name",
                type=str,
                help="Cut list name, as in cutlists/{name}.toml.",
                default=REQUIRED,
                positional=True,
            ),
            Param(
                name="canvas",
                type=str,
                help="Output shape: a fixed 16:9 canvas, or the smallest box "
                "every source fits in.",
                default=C.RENDER_DEFAULT_CANVAS_MODE,
                choices=C.RENDER_CANVAS_MODES,
            ),
            Param(
                name="height",
                type=int,
                help="Canvas height in pixels; 0 takes the tallest source.",
                default=0,
            ),
            Param(
                name="fps",
                type=int,
                help="Output frame rate; 0 takes the commonest source rate.",
                default=0,
            ),
            Param(
                name="gap_ms",
                type=int,
                help="Freeze-frame gap between fragments: -1 measures it from "
                "the aligned transcripts, 0 turns gaps off, >0 sets it.",
                default=-1,
            ),
            Param(
                name="loudnorm",
                type=bool,
                help="Match each source's loudness and normalise the result.",
                default=True,
            ),
            Param(
                name="preset",
                type=str,
                help="x264 preset.",
                default=C.RENDER_VIDEO_PRESET,
            ),
            Param(
                name="crf", type=int, help="x264 CRF.", default=C.RENDER_VIDEO_CRF
            ),
            Param(
                name="keep_intermediates",
                type=bool,
                help="Keep the per-fragment files next to the output.",
                default=False,
            ),
            Param(
                name="render_id",
                type=str,
                help="Output directory name; derived from the cut list by default.",
                default="",
            ),
            Param(
                name="dry_run",
                type=bool,
                help="Plan the render and print it without encoding anything.",
                default=False,
            ),
            Param(
                name="queue",
                type=bool,
                help="Enqueue the render for the worker instead of running it now.",
                default=False,
            ),
        ),
        handler=_run_handler,
        long_running=True,
    )
)

def _list_handler(db: Database, *, limit: int = C.DEFAULT_LIST_LIMIT) -> CommandResult:
    rows = db.conn.execute(
        "SELECT id, cutlist_name, state, canvas_mode, output_path, created_at "
        "FROM renders ORDER BY id DESC LIMIT ?",
        (clamp_limit(limit),),
    ).fetchall()
    return CommandResult(
        columns=("render", "cut list", "state", "canvas", "output", "created"),
        rows=tuple(
            (
                str(row["id"]),
                str(row["cutlist_name"]),
                str(row["state"]),
                str(row["canvas_mode"]),
                str(row["output_path"] or C.NULL_CELL),
                str(row["created_at"]),
            )
            for row in rows
        ),
    )


register(
    Command(
        name="render.pauses",
        group="render",
        summary="Show the measured between-word pauses a render would insert.",
        params=(
            Param(
                name="video_id",
                type=int,
                help="Show every diarized label of this video, plus the video itself.",
                default=0,
            ),
            Param(
                name="speaker",
                type=str,
                help="Show one roster speaker, pooled across every video.",
                default="",
            ),
        ),
        handler=_pauses_handler,
    )
)

def _remove_handler(
    db: Database, *, render_id: int, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    if not dry_run and not yes:
        raise InvalidInputError(
            f"removing render {render_id} deletes its output directory; "
            "pass --yes to confirm, or --dry-run to see what would go"
        )
    removal = remove_render(db, render_id, dry_run=dry_run)
    detail = [
        f"cut list {removal.cutlist_name!r}",
        f"{len(removal.files)} file(s), {removal.total_bytes:,} bytes",
    ]
    if not removal.cutlist_present:
        detail.append("whose cut list is already gone")
    if removal.jobs_cancelled:
        detail.append(
            f"{removal.jobs_cancelled} queued job(s) "
            f"{'cancelled' if removal.removed else 'would be cancelled'}"
        )
    return CommandResult(
        columns=("file", "bytes"),
        rows=tuple((path, f"{size:,}") for path, size in removal.files),
        message=(
            f"{'removed' if removal.removed else 'would remove'} "
            f"render {render_id} — " + "; ".join(detail)
        ),
    )


register(
    Command(
        name="render.list",
        group="render",
        summary="List past renders: what was made, from which cut list, and where.",
        params=(
            Param(
                name="limit",
                type=int,
                help="How many to show, newest first.",
                default=C.DEFAULT_LIST_LIMIT,
            ),
        ),
        handler=_list_handler,
    )
)

register(
    Command(
        name="render.remove",
        group="render",
        summary="Delete a render's output directory and its row.",
        params=(
            Param(
                name="render_id",
                type=int,
                help="Render id, as `render list` shows it.",
                default=REQUIRED,
                positional=True,
            ),
            Param(
                name="dry_run",
                type=bool,
                help="List what would be deleted, with byte counts, and stop.",
                default=False,
            ),
            Param(
                name="yes",
                type=bool,
                help="Confirm. Required, because this deletes files.",
                default=False,
            ),
        ),
        handler=_remove_handler,
    )
)
```

Then add one line to the sibling-import block at the bottom of `rytp/commands/__init__.py`, keeping it alphabetical with the others:

```python
from rytp.commands import render  # noqa: E402,F401
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_commands_render.py -q`
Expected: PASS, 15 passed.

- [ ] **Step 5: Check both surfaces actually show the commands**

Run: `python -m rytp render --help`
Expected: `run`, `pauses`, `list` and `remove` listed with their summaries.

Run: `python -m rytp render run --help`
Expected: `--canvas` shows the two choices; `--loudnorm/--no-loudnorm`, `--dry-run` and `--queue` are present. If Part 1's generator renders a `bool` default-True parameter as a bare `--loudnorm` with no way to turn it off, that is a Part 1 bug — report it rather than adding a second `--no-loudnorm` parameter here.

- [ ] **Step 6: Commit**

```bash
git add rytp/commands/render.py rytp/commands/__init__.py tests/test_commands_render.py
git commit -m "feat: add the render.run and render.pauses commands"
```

---

### Task 10: Real ffmpeg, lint, types, full suite

Everything so far asserted argument lists. This task runs two of them for real, then closes the gate.

Two tests, no more, and they build their own inputs: `lavfi` sources at two different aspect ratios, a fraction of a second each, `ultrafast`. They are the only place that would catch a filter graph ffmpeg refuses, a `tpad` option that does not exist in the installed build, or a loudnorm JSON block that does not parse — none of which a fake can tell you. No media is committed: contracts §1 forbids fixtures with real ids, and generated tones cost nothing.

**Files:**
- Test: `tests/test_render_real_ffmpeg.py`
- Modify: nothing

**Interfaces:**
- Consumes: everything built above.
- Produces: nothing importable.

- [ ] **Step 1: Write the real-ffmpeg tests**

Create `tests/test_render_real_ffmpeg.py`:

```python
"""The two tests that actually run ffmpeg. Everything else fakes it.

Inputs are synthesised here — a colour pattern and a tone, two aspect
ratios, well under a second each. Nothing is committed and nothing is
downloaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.render import run as RUN

from tests.fakes import make_video

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(
    not HAS_FFMPEG, reason="ffmpeg and ffprobe are not both on PATH"
)


def synth_source(
    db: Database, tmp_path: Path, name: str, *, size: str, freq: int
) -> int:
    """A catalogued video whose rendition and audio are generated here."""
    video = tmp_path / f"{name}.mp4"
    audio = tmp_path / f"{name}.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size={size}:rate=25:duration=5", "-pix_fmt", "yuv420p",
         "-preset", "ultrafast", "-an", str(video)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:duration=5", "-ac", "1", str(audio)],
        check=True, capture_output=True,
    )
    vid = make_video(
        db, external_id=name, url=f"https://example.invalid/watch/{name}"
    )
    insert_asset(db, video_id=vid, role="video", format_id="src", path=str(video))
    insert_asset(db, video_id=vid, role="audio", path=str(audio))
    return vid


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_two_shapes_join_into_one_canvas_with_a_freeze_frame_between(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    narrow = synth_source(db, tmp_path, "VIDEO_A", size="160x120", freq=440)
    wide = synth_source(db, tmp_path, "VIDEO_B", size="256x144", freq=220)
    request = RUN.RenderRequest(
        name="smoke",
        target_text="раз два",
        fragments=(
            RUN.RenderFragment(
                video_id=narrow, first_word_ord=0, last_word_ord=0,
                start_ms=200, end_ms=600, text="раз",
            ),
            RUN.RenderFragment(
                video_id=wide, first_word_ord=0, last_word_ord=0,
                start_ms=300, end_ms=700, text="два",
            ),
        ),
    )
    result = RUN.render_cutlist(
        db,
        request,
        RUN.RenderOptions(gap_ms=150, loudnorm=False, preset="ultrafast", crf=30),
    )
    assert result.output_path.exists()
    assert result.report_path.exists()

    probed = probe(result.output_path)
    stream = probed["streams"][0]
    assert (stream["width"], stream["height"]) == (
        result.plan.canvas.width,
        result.plan.canvas.height,
    )
    # 400 ms + a 150 ms freeze + 400 ms, within a container's rounding.
    assert float(probed["format"]["duration"]) == pytest.approx(0.95, abs=0.25)
    assert "## Fragments" in result.report_path.read_text(encoding="utf-8")


def test_the_two_pass_loudnorm_survives_a_real_ffmpeg(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    """The measuring pass's JSON is the fragile part; parse a real one."""
    vid = synth_source(db, tmp_path, "VIDEO_A", size="160x120", freq=440)
    request = RUN.RenderRequest(
        name="loud",
        fragments=(
            # Four seconds, not one: EBU R128's loudness range is a
            # short-term statistic over 3 s windows, and ffmpeg can
            # legitimately answer -inf for a shorter programme — which
            # the parser reads, correctly, as "not measured".
            RUN.RenderFragment(
                video_id=vid, first_word_ord=0, last_word_ord=0,
                start_ms=500, end_ms=4_000, text="раз",
            ),
        ),
    )
    result = RUN.render_cutlist(
        db, request, RUN.RenderOptions(loudnorm=True, preset="ultrafast", crf=30)
    )
    assert result.output_path.exists()
    report = result.report_path.read_text(encoding="utf-8")
    assert "normalized" in report
    assert "could not be measured" not in report
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_render_real_ffmpeg.py -q`
Expected: 2 passed where ffmpeg is installed, 2 skipped where it is not. **Both outcomes are a pass for this step** — but if they skip, say so in the commit message rather than claiming they ran.

If `test_the_two_pass_loudnorm_survives_a_real_ffmpeg` fails on the measurement, read the ffmpeg stderr it quotes before changing anything: a synthesised sine is unusually easy to measure, so a failure there is a real parsing or filter problem, not a fixture artefact.

- [ ] **Step 3: Prove the suite does not need ffmpeg**

The whole point of the seam. Run the rest of Part 6's suite with the binaries hidden:

Run the command below with ffmpeg and ffprobe unreachable — a POSIX shell prefixes it with `PATH=/nonexistent`, a Windows shell runs `set PATH=` first:

`python -m pytest tests/test_render_ffmpeg.py tests/test_render_ffmpeg_commands.py tests/test_render_canvas.py tests/test_render_pauses.py tests/test_render_report.py tests/test_render_plan.py tests/test_render_run.py tests/test_render_job.py tests/test_commands_render.py -q`

Expected: every test passes; none errors on a missing binary. If one does, it resolved a binary outside a `Tools` — find that call and route it through the seam.

- [ ] **Step 4: Lint**

Run: `ruff check rytp tests`
Expected: no findings. Fix what it reports; do not add `noqa` except the two already justified in this plan (`E402` on the deliberate bottom imports, `S603` on the fixed-argv subprocess call).

- [ ] **Step 5: Type-check**

Run: `mypy rytp`
Expected: no errors. Two likely complaints and their correct fixes: `sqlite3.Row` indexing returns `Any`, so wrap in `int(...)`/`str(...)` at the boundary as the code above already does; and `Tools.runner` is `Runner | None`, which is what `run_command` expects — do not "fix" that by making it non-optional.

- [ ] **Step 6: Full suite**

Run: `python -m pytest -q`
Expected: every part's tests pass together. Part 6 adds nine test modules; nothing it touches is shared except `rytp/constants.py` (appended), `rytp/jobs/__init__.py` (one kind) and `rytp/commands/__init__.py` (one import), so a failure elsewhere is a real integration problem, not a merge artefact.

- [ ] **Step 7: Run it against a real file, by hand**

A green suite says the argument lists are right, not that the pipeline is. CLAUDE.md is explicit about this. With ffmpeg installed:

```bash
# Register a local media file as a container asset. Use any file you have;
# never put a real video id, channel name or URL into a committed file.
python -m rytp videos add --local "<a local media file>"
python -m rytp videos list
```

Then, if Part 5 is implemented, hand-write `data/cutlists/smoke.toml` with two `[[slot]]` fragments pointing at that video id and two time ranges you know contain speech, and run:

```bash
python -m rytp render run smoke --dry-run
python -m rytp render run smoke
```

If Part 5 is **not** implemented yet, drive the stage directly — this is the honest end-to-end check of everything Part 6 owns:

```bash
python -c "
from rytp.config import paths
from rytp.db import Database
from rytp.render.run import RenderFragment, RenderRequest, render_cutlist
db = Database(paths().db); db.migrate()
request = RenderRequest(name='smoke', target_text='проверка', fragments=(
    RenderFragment(video_id=1, first_word_ord=0, last_word_ord=0,
                   start_ms=10000, end_ms=12000, text='проверка'),
    RenderFragment(video_id=1, first_word_ord=1, last_word_ord=1,
                   start_ms=30000, end_ms=32000, text='связи'),
))
print(render_cutlist(db, request).output_path)
"
```

Watch for four things, which are the four ways this stage can be wrong while every test passes: the cut lands on the words you meant; the freeze-frame between them looks like a held frame rather than a stutter or a black flash; the volume does not jump at the seam; and `report.md` names the right source and timestamps.

- [ ] **Step 8: Commit**

```bash
git add tests/test_render_real_ffmpeg.py
git commit -m "test: run the render pipeline against a real ffmpeg"
```

---

## End state a reviewer can check

- `python -m rytp render run <name>` turns `data/cutlists/<name>.toml` into `data/output/<render_id>/output.mp4` plus `report.md`, and prints the two paths.
- The output has one canvas, one frame rate, hard cuts, freeze-frame gaps, and no volume step between sources.
- `report.md` contains the assembled text, a row per fragment with source video, source in/out and output in/out, a row per source with its measured loudness and applied gain, every word that could not be found with its ranked substitutions, any notes about measurements that failed, and a fenced description block.
- A fragment whose rendition is missing fails with `video N has no video rendition on disk — run: rytp fetch-video N`, before anything is encoded, listing every such fragment at once.
- `python -m rytp render pauses --video-id N` shows, per speaker and for the video, how many gaps were measured, their median, the share that are exactly zero, what the render would insert, and whether the distribution is usable.
- `--gap-ms 0` renders with no gaps; `--gap-ms 250` uses 250 ms everywhere; a `gap_before_ms` in the cut list overrides the measurement for that seam.
- `--canvas bbox` changes the output shape only when the sources are not all 16:9.
- `--no-loudnorm` skips both measuring passes and the filter, and the report says the loudness was left alone.
- `--dry-run` prints the same summary and writes nothing.
- `render` appears in `rytp jobs stats` as a `cpu`-pool kind; `--queue` opens a `planned` row in `renders`, enqueues a job against its id, and `rytp worker` renders it and closes the row `rendered`. Delete the output file and the job is runnable again; delete the cut list and it is blocked.
- `python -m rytp render list` shows every render, newest first, with its cut list, state and output path.
- `python -m rytp render remove <id> --dry-run` lists every file that would go with its byte count and names the cut list that produced it; without `--yes` the real command refuses; with it, the output directory, the `renders` row and any queued job for that render go together. It never touches the cut list file, and it will not delete a directory outside the data tree's `output/`.
- `ruff check rytp tests` and `mypy rytp` are clean; `pytest -q` is green, and green **with `PATH=/nonexistent`** for every module except `tests/test_render_real_ffmpeg.py`, which skips.

## Self-review

Run against design §9 and the contracts after the plan was written.

**Spec coverage.** Design §9's five rows: audio loudness-normalized by default with a flag to disable — Tasks 3, 6, 7, `--no-loudnorm`. Hard cuts, no crossfades — there is no transition filter anywhere in Task 3. Freeze-frame over a gap — `tpad=stop_mode=clone` in Task 2, paired with `apad` in Task 3. Gap length configurable, defaulting to measured between-word pauses per speaker or per video, disableable — Tasks 4 and 6. Canvas 16:9 pillarboxed by default, configurable, with a smallest-bounding-box alternative — Task 2. The report with assembled text, fragments, missing words and a paste-ready block, structured for later JSON and subtitles — Task 5. Output paths from contracts §7 — Task 6. The `render` job kind from contracts §5 — Task 8.

**Contracts.** Copied verbatim: the `Command`/`Param`/`CommandResult` shapes, the handler convention, `JOB_HANDLERS`' signature, `RytpError` subclassing, milliseconds everywhere, `output/{render_id}/output.mp4` and `report.md`, list-argument subprocess, no real ids. Four files added beyond the §2 tree, each justified in the File Structure section.

**Placeholders.** None. Every step names its files and carries the code; no "add error handling", no "similar to Task N".

**Deletion (contracts §5, added after this plan's first draft).** `render.remove` is in Task 7 (`plan_removal` / `remove_render`) and Task 9 (the command). It honours all three rules: files are removed explicitly rather than by cascade, the queued job is cancelled in the same transaction as the row deletion because `jobs` has no foreign key, `--dry-run` counts rows and bytes, `--yes` is required, and it is synchronous rather than a job. One rule of its own: only a directory under the data tree's `output/` is ever removed, because `renders.output_path` is text a human can edit.

**Cross-part, settled with the Part 5 author.** Deleting a cut list *warns* about the renders that reference it and leaves their rows alone; it does not refuse. `renders.cutlist_name` is a name rather than a foreign key so history outlives its source, renders accumulate so refusing would make tidying harder the more the tool is used, and warn-and-orphan is already the house pattern (`channel.remove`, `speakers.remove`). Both plans carry the same decision and the same reasons. Nothing in Part 6 re-reads a cut list after a successful render, so a missing one costs reproducibility, not an artifact; a *queued* render of a deleted cut list parks as BLOCKED, which is why Part 5's warning says so.

**The `renders` table (contracts §3, added after this plan's first draft).** Consumed in Task 7 (`create_render` / `get_render` / `finish_render`, and `render_cutlist(render_row_id=...)`), Task 8 (the predicate reads the row), Task 9 (`--queue` opens the row, `render.list` reads them). Part 1 owns the migration; Part 6 creates no schema. The CRC-derived `target_id` and the payload-name check that the first draft used as a workaround are gone, along with `zlib`.

**Type consistency.** Checked across tasks: `Tools`, `Runner`, `CompletedRun`, `LoudnormMeasurement` (Tasks 1, 3) are used with the same signatures in Tasks 6 and 7. `SourceGeometry` / `Canvas` (Task 2) are consumed by `plan_render` and `build_report` with the fields they declare. `PauseStats.description` (Task 4) is what Task 6 puts in `plan.notes`. `MissingWord` / `SubstitutionRef` (Task 5) are built by `request_from_cutlist` (Task 6) and read by `render_markdown` (Task 5) — same field names. `RenderPlan.fragments_dir` (Task 6) is used by Task 7's cleanup. `payload_for` / `options_from_payload` (Task 7) round-trip every `RenderOptions` field, and `_options()` in Task 9 builds the same dataclass. `create_render` / `finish_render` / `get_render` (Task 7) are used with those signatures by Task 9's handlers and asserted on by Task 8's predicate tests.

**Two things a reviewer should push on.**

1. *Contracts §5 versus Part 2's registry.* The contract specifies `JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]`; Part 2 originally shipped `JobKind.handler(db, target_id) -> str` with no payload. The Part 2 author has since amended their plan to expose both, and added `target_kind` and `reopenable` at Part 6's request. This plan is written against the amended version. If the executor finds `rytp/jobs/__init__.py` without `JOB_HANDLERS`, `target_kind` or `reopenable`, stop and reconcile with Part 2 rather than working around it — a render registered as `target_kind="video"` would let the worker's `unblock` cross namespaces.
2. *The `renders` table, and what it bought.* The first draft of this plan hashed a cut-list name into `jobs.target_id` with a CRC and had the handler verify the name in the payload to catch collisions. That was raised as a contract gap rather than lived with, and contracts §3 now carries a `renders` table. Everything downstream got simpler, not just safer: the readiness predicate went from "always READY, because it is forbidden to know anything" to a genuine question about the world with working self-healing, `reopenable` went back to its default, the payload carries options only, and `render.list` fell out for free. Worth remembering as an argument for raising this kind of thing.

3. *The intermediates decision is the one to preserve.* Per-fragment intermediates are Matroska with `pcm_s16le`, and the final pass always re-encodes audio, because AAC carries roughly 1024 samples of encoder priming per file that the concat demuxer does not trim — about 21 ms of drift at **every** seam, most of a second over forty fragments, plus a click at each. It is invisible in any test that checks argument lists and obvious in the finished file. Do not "optimise" the intermediates to AAC.
