# Part 7 — Speakers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the backward index from "where was this word said" into "where did *this person* say this word" — diarize a video into local speaker labels, let a human map those labels onto a global roster through a two-pane TUI, and suggest (never apply) cross-video links from voice embeddings scoped to comparable recordings.

**Architecture:** Speaker identity is two levels and the distinction is the whole design. `video_speakers` holds one row per diarizer label per video — `SPEAKER_04` in some video is a real, citable entity even when nobody ever named it. `speakers` is the roster of actual people, and `video_speakers.speaker_id` is the nullable link between them. Diarization stamps `words.video_speaker_id` once, at diarize time, from segment overlap; *linking a label to a person afterwards updates exactly one row* and never walks the word table. Everything a human touches goes through a headless `MappingSession` that knows nothing about terminals, so the Textual screen is a thin shell and the logic is unit-tested without a TTY. Cross-video linking produces ranked suggestions with a three-way verdict and writes nothing.

**Tech Stack:** Python 3.11+, SQLite, stdlib `wave` / `struct` for audio slicing and embedding vectors, Textual for the mapper screen. Optional extras, all lazily imported and none required by the test suite: `pyannote.audio` (diarization, out of process, HF-token gated), `torch` + ReDimNet (speaker embeddings, out of process). No numpy is required by anything in this part.

**Spec:**
- `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` — the approved design; §4 (speakers, acoustics), §6 ("Diarization is opt-in per video", "Cross-video speaker linking", "Engines run out of process"), §10 ("TUI is required for speaker assignment") and §11 (milestone M3) are this part's mandate.
- `docs/superpowers/specs/2026-09-21-rytp-contracts.md` — binding shared interfaces. Schema DDL (§3), core types (§4) including the cross-part invariants, command registry and job handlers (§5), engine protocols (§6), filesystem layout (§7) and conventions (§8) are copied verbatim into this plan. **Never vary them.** Six other plans depend on the same file.

## Global Constraints

Every task's requirements implicitly include this section.

- Python >= 3.11. Target 3.11 syntax. `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. `pathlib` everywhere; `subprocess` with **list** arguments only.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` on `rytp/`.
- Heavy dependencies (diarizers, embedders) are optional extras, **imported lazily inside functions, never at module import time**.
- **No test may download a model or touch the network.** Every engine is faked, every embedding is a hand-written vector. The suite must pass with none of the optional extras installed.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. Speaker names in tests are obviously fictional: `Host One`, `Guest Two`, `Каспар Хаузер`.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- Time is integer milliseconds everywhere. Database timestamps are `datetime.now(UTC).isoformat()` (`rytp.models.utc_now_iso`).
- Domain errors subclass `RytpError` (`rytp/models.py`). The CLI prints one line to stderr and exits 1; no traceback for an expected failure.
- Every stage takes an **open `Database`**; nothing opens its own connection. Multi-statement writes use `db.transaction()`, which is re-entrant (contracts §8).
- One commit per task, conventional-commit prefix, present tense.

**Test invocation is portable** (contracts §1). Every run step in this plan is

```bash
python -m pytest <args>
```

with the project's virtualenv already activated — no absolute interpreter
paths, no developer-specific directories, no `/tmp`. The project ships on
Windows, and a plan that only runs on one machine is a plan that cannot be
followed. `RYTP_TEST_TMP` may override the scratch root but the suite must
pass without it set.

The development environment deliberately has **no ML libraries and no
textual-testing extras** — pytest, ruff, mypy, numpy, scipy and `textual`,
and nothing else. If a test fails there because something heavy is missing,
the test is wrong, not the environment. Textual screens are driven with
`asyncio.run(app.run_test())`, which is core Textual and needs no plugin.

---

## Prerequisites (owned by other plan parts)

Part 7 consumes these. Each is named here so an executor knows exactly what to check, and each has few import sites so a rename is a small fix.

| Needed | Owner | Used by |
|---|---|---|
| `rytp/models.py`: `DiarSegment(start_ms, end_ms, local_label)`, `RytpError`, `NotFoundError`, `InvalidInputError`, `utc_now_iso` (contracts §4) | Part 1 | everywhere |
| `rytp/db/__init__.py`: `Database` with `.conn` (`sqlite3.Connection`, `row_factory = sqlite3.Row`, foreign keys on) and a re-entrant `.transaction()` | Part 1 | every DB task |
| `rytp/db/schema.py`: `MIGRATIONS` creating `speakers`, `video_speakers`, `words`, `utterances`, `video_acoustics`, `jobs`, `settings` exactly as contracts §3 spells them, with `speakers` **before** `video_speakers` | Part 1 | every DB task |
| `video_speakers.engine TEXT NOT NULL` — which diarizer produced this label (contracts §3). Part 1 creates the column; Part 7 is the only thing that writes it | Part 1 | Tasks 5, 7, 12 |
| `rytp/commands/__init__.py`: `REQUIRED`, `Param`, `Command`, `CommandResult`, `register`, `resolve`, `COMMANDS`, and the `cli_only: bool = False` field of `Command` (contracts §5). The CLI must expose a `cli_only` command and `rytp.tui.palette.palette_entries()` must leave it out | Part 1 | Tasks 13, 16 |
| `rytp/commands/__init__.py` (or wherever Part 1 puts them): `HealthCheck(name, summary, run, required=True)`, `HealthResult(ok, detail, remedy=None)`, `HEALTH_CHECKS`, `register_check(check)`, and the `doctor` command, which exits non-zero only for a `required=True` check returning `ok=False` (contracts §5) | Part 1 | Task 15 |
| `rytp/constants.py` importable with **stdlib imports only** | Part 1 | everywhere |
| `tests/conftest.py` fixtures `tmp_path`, `data_dir`, `db` with their contracted names and semantics; `tmp_path` rooted at `$RYTP_TEST_TMP` | Part 1 | all tests |
| `rytp/tui/app.py`: `RytpApp(db, commands=None)` with `BINDINGS: ClassVar[list[BindingType]]` and `set_status(text)` | Part 1 | Task 14 |
| `rytp.config.paths().cache_wav(video_id) -> Path` returning `<RYTP_DATA>/cache/wav/{video_id}.wav` (contracts §7) | Part 1 | `readiness.py`, `pipeline.py` |
| `tests/fakes.py::make_video(db, **overrides) -> int` | Part 2 | all tests |
| `rytp/jobs/__init__.py`: `Readiness`, `JobKind(name, pool, readiness, handler, summary, target_kind="video", reopenable=True)`, `register_job_kind`, `JOB_KINDS`, `JOB_HANDLERS`; `tests/fakes.py::temp_job_kind` | Part 2 | Task 8 |
| `rytp/jobs/queue.py`: `enqueue(db, kind, target_id, *, priority=0, payload=None, now=None) -> int`, `list_jobs`, `reconcile`, `finish`, `get_job` | Part 2 | Tasks 8, 13 |
| `rytp/audio/extract.py::ensure_wav(db, video_id, *, overwrite=False) -> Path` and the `extract_wav` job kind | Part 2 | the message a missing WAV produces |
| `rytp/transcribe/base.py`: `EngineUnavailable`, `EngineSubprocessError`, `slice_wav_window(src, dst, start_ms, end_ms) -> Path` | Part 3 | Tasks 1, 10, 11 |
| `rytp/transcribe/registry.py`: `check_available(cls)`, `interpreter_for(db, name)`, `setting(db, key, default=None)`, `availability(db, cls)` | Part 3 | Tasks 1, 9 |
| `rytp/transcribe/subproc.py::run_child(*, interpreter, module, request, timeout_s=None, repo_root=None) -> dict` | Part 3 | Tasks 10, 11 |
| `video_acoustics` rows written by `rytp.audio.acoustics.store_acoustics` / the `fingerprint` job | Part 3 | Task 12 |
| `words` rows with `source='aligned'`, `start_ms`, `end_ms` and `video_speaker_id` defaulting to NULL | Part 3 | Tasks 5, 7 |
| Part 3 deletes a video's `utterances` **and** `video_speakers` in the same transaction that replaces its words (contracts §4) | Part 3 | assumed, never guarded against |
| The `index` job kind, whose handler **deletes and rebuilds** a video's `utterances` from its current `words`, and whose readiness is READY when the video has words and no utterances | Part 4 | Task 7 |

**Two resolutions made here, because the contracts left them open** (raise them if you disagree rather than varying them silently):

1. **Registration of the `diarize` job kind edits `rytp/jobs/__init__.py`, which Part 2 owns.** Contracts §5 says Part 7 owns the kind; Part 2's "Produces for Parts 3-7: registering a job kind" section says the thunk and the `register_job_kind` call go at the bottom of that file, with the readiness predicate in a light module of the owning package. This plan follows Part 2's recipe exactly and puts the predicate in `rytp/diarize/readiness.py`.
2. **`rytp/diarize/` gets seven modules beyond the four contracts §2 lists.** The reasons are in the File Structure table. Contracts §2 permits this ("Do not create files outside this tree without saying why"), and Part 3 set the precedent with `pipeline.py` and `subproc.py`.

---

## File Structure

**Created by this plan**

| File | Responsibility |
|---|---|
| `rytp/diarize/__init__.py` | Imports the engine modules — the import is what makes a name resolvable |
| `rytp/diarize/base.py` | `Diarizer` protocol, the `DIARIZERS` registry, gating, `wav_duration_ms` (contracts §6) |
| `rytp/diarize/none.py` | `NullDiarizer` — one label for the whole file, no model, no token |
| `rytp/diarize/pyannote.py` | `PyannoteDiarizer` and its `child_main`, out of process, HF-token gated |
| `rytp/diarize/embed.py` | `SpeakerEmbedder` protocol and registry, the float32 BLOB codec, cosine/centroid maths, `ReDimNetEmbedder` and its `child_main` |
| `rytp/diarize/segments.py` | Pure segment arithmetic: label a word list from segments, speech totals, embedding window choice. No database, no engine — which is why it is separate and why its tests are the sharpest in this part |
| `rytp/diarize/store.py` | Every SQL statement about `speakers` and `video_speakers`, plus the row types both the commands and the TUI render. Separate so the TUI never imports `rytp.commands` and the commands never write SQL |
| `rytp/diarize/readiness.py` | `diarize_readiness` — deliberately a light module, because `rytp/jobs/__init__.py` imports it at load time |
| `rytp/diarize/pipeline.py` | `diarize_video` and `embed_video_speakers` — the two stages, the only writers of `words.video_speaker_id` |
| `rytp/diarize/link.py` | Eras, acoustic comparability, enrolment centroids, the two-threshold band. Suggestion only: it never writes |
| `rytp/diarize/mapper.py` | `MappingSession` and `match_roster` — the whole mapper, headless |
| `rytp/diarize/health.py` | The two `doctor` checks Part 7 contributes: which diarizers could run, and whether `HF_TOKEN` is set |
| `rytp/commands/speakers.py` | The thirteen registered commands |
| `rytp/tui/screens/speakers.py` | `SpeakerVideosScreen`, `SpeakerMapperScreen`, `run_mapper` — a thin shell over `MappingSession` |
| `tests/fake_speaker_engines.py` | Fake diarizers and embedders, and a `registered` context manager |
| `tests/test_diarize_registry.py` … `tests/test_speakers_integration.py` | One test module per task |

**Modified by this plan** (all additive, all shared with other parts)

- `rytp/constants.py` — one Part 7 section appended at the end (Task 1).
- `rytp/commands/__init__.py` — one line in the bottom import block (Task 13).
- `rytp/jobs/__init__.py` — one thunk, one predicate import and one `register_job_kind` call (Task 8).
- `rytp/tui/app.py` — one binding and one action that pushes the speakers screen (Task 14).
- `pyproject.toml` — the `pyannote` and `redimnet` extras (Task 10).

---

## Tasks

### Task 1: Constants and the diarizer registry

**Files:**
- Modify: `rytp/constants.py` (append one delimited Part 7 block at the end)
- Create: `rytp/diarize/__init__.py` (empty for now; Task 2 fills it)
- Create: `rytp/diarize/base.py`
- Create: `rytp/diarize/embed.py` (a placeholder; Task 9 replaces it)
- Create: `tests/fake_speaker_engines.py`
- Test: `tests/test_diarize_registry.py`

Part 1 deleted the old `rytp/diarize/` tree, marker included. Create the empty
`__init__.py` first: an implicit namespace package would import, but `mypy` and
the packaging metadata both expect a real one.

**Interfaces:**
- Consumes: `rytp.models.{DiarSegment, RytpError}`; `rytp.transcribe.base.EngineUnavailable`; `rytp.transcribe.registry.{check_available, interpreter_for, availability}`; `Database.conn`.
- Produces:
  - `rytp.diarize.base`: `Diarizer` (Protocol), `DiarizeError`, `DIARIZERS`, `register_diarizer`, `resolve_diarizer`, `load_diarizer(db, name, **kw)`, `diarizer_rows(db)`, `wav_duration_ms(path) -> int`.
  - `tests/fake_speaker_engines.py`: `FakeDiarizer`, `SingleLabelDiarizer`, `GatedDiarizer`, `OutOfProcessDiarizer`, `FakeEmbedder`, `ConstantEmbedder`, `registered(*classes)`.

- [ ] **Step 1: Append the Part 7 constants block to `rytp/constants.py`**

Every later task reads these. Adding them once keeps a Part-1-owned file from being edited fifteen times.

```python
# ---------------------------------------------------------------------------
# Part 7 — speakers, diarization and cross-video linking (design §4, §6, §11)
# ---------------------------------------------------------------------------

#: The local label a diarizer with no speaker model assigns to everything.
#: pyannote's own labels look like this, so one convention covers both.
NULL_DIARIZER_LABEL: str = "SPEAKER_00"

#: Default engine names. The default diarizer is the null one: diarization is
#: opt-in per video (design §6) and the default must never cost GPU time.
DEFAULT_DIARIZER: str = "none"

#: Empty means "do not embed". Embedding is useful only once a second video
#: of the same person exists, so it is off until asked for (design §6).
DEFAULT_EMBEDDER: str = ""

#: Models behind the two real engines (design §6, "Recommended stack").
#: community-1 has roughly half the speaker confusion of pyannote 3.x, which
#: is the host-plus-guest failure mode this corpus is made of.
PYANNOTE_DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
REDIMNET_DEFAULT_MODEL: str = "b6"

#: Settings keys holding the default engine names, so a machine is configured
#: once rather than on every command line.
SETTINGS_DIARIZER: str = "speakers.diarizer"
SETTINGS_EMBEDDER: str = "speakers.embedder"

# --- Embedding extraction (design §6, "Speaker embeddings") ---

#: A label with less speech than this is not embedded at all. Three seconds
#: is the usual floor below which a speaker embedding is dominated by the
#: phonetic content rather than the voice.
EMBED_MIN_SPEECH_MS: int = 3_000

#: At most this much of one label's speech is fed to the embedder. Thirty
#: seconds saturates every published speaker-verification curve; an hour of
#: the host would cost sixty times as much for no gain.
EMBED_MAX_SPEECH_MS: int = 30_000

#: Segments shorter than this are skipped when choosing embedding windows:
#: a one-second turn is mostly onset and offset.
EMBED_MIN_SEGMENT_MS: int = 1_000

#: Embeddings are stored in video_speakers.embedding as little-endian
#: float32, which is half the size of float64 and further below the noise
#: floor of the model than any downstream comparison can notice. The width
#: is what the codec needs: the format string is built from the vector
#: length at pack time.
EMBEDDING_BYTES_PER_VALUE: int = 4

# --- Cross-video linking (design §6, "Cross-video speaker linking") ---

#: Era bucket width in years. The corpus spans more than a decade and the
#: microphones, rooms and the main speaker's voice all changed; comparing
#: within a bucket is what keeps an enrolment centroid meaningful.
SPEAKER_ERA_YEARS: int = 2

#: Bucket for a video with no published_at — a local file, usually.
SPEAKER_ERA_UNKNOWN: str = "unknown"

#: The two-threshold band (design §6). At or above HI is a confident match,
#: at or below LO a confident non-match, and the space between goes to the
#: human. UNVALIDATED: no measurement of this corpus exists, these are the
#: usual starting points for a cosine speaker-verification score. Re-tune
#: them once real suggestions have been accepted and rejected for a while.
SPEAKER_MATCH_HI: float = 0.70
SPEAKER_MATCH_LO: float = 0.45

#: Across eras, embedding error roughly triples (design §6), so a
#: cross-era comparison needs a much higher score before it may be called
#: confident. Below this it is a suggestion for a human, never a match.
#: UNVALIDATED, same caveat as above.
SPEAKER_MATCH_HI_CROSS_ERA: float = 0.85

#: How far apart two recordings may be acoustically and still be treated as
#: comparable. The distance is the mean of the per-feature differences
#: below, each divided by its scale, so 1.0 means "one typical spread apart
#: on average".
ACOUSTIC_MAX_DISTANCE: float = 1.0

#: Per-feature scales for that distance, in the feature's own units: a
#: plausible spread across recordings of one person (design §4, acoustics).
#: UNVALIDATED — order-of-magnitude guesses until the corpus has been
#: fingerprinted and the real spread can be measured.
ACOUSTIC_FEATURE_SCALES: tuple[tuple[str, float], ...] = (
    ("f0_mean", 20.0),          # Hz
    ("f0_std", 15.0),           # Hz
    ("spectral_tilt", 6.0),     # dB per decade
    ("noise_floor_db", 10.0),   # dB
    ("reverb_proxy", 0.15),     # dimensionless ratio
)

#: How many roster candidates one suggestion call returns per local label.
SPEAKER_SUGGEST_LIMIT: int = 5

#: `speakers remove` unlinks every label naming that person. Below this many
#: links it just does it; at or above, it wants `--yes`. Contracts §5 only
#: *requires* confirmation for commands that delete files, and this deletes
#: none — but undoing an afternoon of mapping by mistyping a name is the
#: kind of loss worth one extra word (design §3: erase and replace, but
#: deliberately).
SPEAKER_REMOVE_CONFIRM_LINKS: int = 5

# --- Surfaces ---

#: Rows the speaker listing commands return before the user must filter.
SPEAKER_LIST_LIMIT: int = 200

#: Titles are truncated harder in the speakers panes than in the video list,
#: because the mapper shows two tables side by side.
SPEAKER_TITLE_TRUNCATE_CHARS: int = 40
```

- [ ] **Step 2: Write `tests/fake_speaker_engines.py`**

Not a test itself — the fakes every later task uses. Written now because Task 1's test needs them.

```python
"""Fake diarizers and embedders: no model, no network, no optional extra.

Registering an engine mutates module-level dicts, so always go through
:func:`registered`, which undoes exactly what it did. Snapshotting the whole
registry and restoring it would be wrong: a real engine registers itself at
import time and Python imports a module once, so wiping its name here would
leave it unregistered for every later test file.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rytp import constants as C
from rytp.diarize import base, embed
from rytp.models import DiarSegment


class FakeDiarizer:
    """Replays a fixed script of (start_ms, end_ms, local_label) triples."""

    name = "fake-diarizer"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    script: tuple[tuple[int, int, str], ...] = (
        (0, 1_000, "SPEAKER_00"),
        (1_000, 2_000, "SPEAKER_01"),
        (2_000, 3_000, "SPEAKER_00"),
    )

    def __init__(self, script: Sequence[tuple[int, int, str]] | None = None) -> None:
        self._script = tuple(script) if script is not None else self.script

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        for start, end, label in self._script:
            yield DiarSegment(start_ms=start, end_ms=end, local_label=label)


class SingleLabelDiarizer(FakeDiarizer):
    """Everything is one voice — what the null diarizer does, without a WAV."""

    name = "fake-single"
    script = ((0, 5_000, C.NULL_DIARIZER_LABEL),)


class GatedDiarizer(FakeDiarizer):
    name = "fake-gated-diarizer"
    requires_hf_token = True


class OutOfProcessDiarizer(FakeDiarizer):
    name = "fake-remote-diarizer"
    out_of_process = True

    def __init__(
        self,
        interpreter: str,
        script: Sequence[tuple[int, int, str]] | None = None,
    ) -> None:
        super().__init__(script)
        self.interpreter = interpreter


class FakeEmbedder:
    """Returns a vector derived from the windows, so tests can predict it.

    The first component is the total window length in seconds and the rest
    are fixed, which makes two labels with different amounts of speech land
    in different directions without any model involved.
    """

    name = "fake-embedder"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    dim = 4

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> list[float]:
        total_s = sum(end - start for start, end in windows) / 1000.0
        return [total_s, 1.0, 0.0, 0.0]


class ConstantEmbedder(FakeEmbedder):
    """Always the same vector — two labels then look like the same person."""

    name = "fake-constant-embedder"
    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> list[float]:
        return list(self.vector)


_MISSING = object()


@contextmanager
def registered(*classes: type) -> Iterator[None]:
    """Register engines for the duration of a test, then undo exactly that."""
    undo: list[tuple[dict[str, type], str, object]] = []
    try:
        for cls in classes:
            table = base.DIARIZERS if hasattr(cls, "diarize") else embed.EMBEDDERS
            undo.append((table, cls.name, table.get(cls.name, _MISSING)))
            table[cls.name] = cls
        yield
    finally:
        for table, name, previous in reversed(undo):
            if previous is _MISSING:
                table.pop(name, None)
            else:
                table[name] = previous  # type: ignore[assignment]
```

- [ ] **Step 3: Create `rytp/diarize/embed.py` as a placeholder**

`tests/fake_speaker_engines.py` imports `EMBEDDERS`, and Task 9 replaces this
whole file. It exists now only so that import resolves.

```python
"""Speaker embeddings. Filled in by Task 9."""

from __future__ import annotations

from typing import Any

EMBEDDERS: dict[str, type[Any]] = {}
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_diarize_registry.py`:

```python
"""The diarizer registry: classes in, classes out, gated before construction."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.diarize import base
from rytp.transcribe.base import EngineUnavailable
from tests.fake_speaker_engines import (
    FakeDiarizer,
    GatedDiarizer,
    OutOfProcessDiarizer,
    registered,
)


def test_register_and_resolve_returns_the_class_not_an_instance() -> None:
    with registered(FakeDiarizer):
        assert base.resolve_diarizer("fake-diarizer") is FakeDiarizer


def test_resolve_unknown_diarizer_names_the_available_ones() -> None:
    with registered(FakeDiarizer), pytest.raises(ValueError) as excinfo:
        base.resolve_diarizer("nope")
    assert "nope" in str(excinfo.value)
    assert "fake-diarizer" in str(excinfo.value)


def test_gate_rejects_a_token_engine_before_construction(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with registered(GatedDiarizer), pytest.raises(EngineUnavailable) as excinfo:
        base.load_diarizer(db, "fake-gated-diarizer")
    assert "HF_TOKEN" in str(excinfo.value)


def test_gate_accepts_a_token_engine_when_the_token_is_set(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    with registered(GatedDiarizer):
        assert isinstance(base.load_diarizer(db, "fake-gated-diarizer"), GatedDiarizer)


def test_load_diarizer_injects_the_interpreter_out_of_process(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote-diarizer", "/opt/remote/python"),
    )
    with registered(OutOfProcessDiarizer):
        engine = base.load_diarizer(db, "fake-remote-diarizer")
    assert engine.interpreter == "/opt/remote/python"


def test_load_diarizer_does_not_inject_an_interpreter_in_process(db: Database) -> None:
    with registered(FakeDiarizer):
        engine = base.load_diarizer(db, "fake-diarizer")
    assert not hasattr(engine, "interpreter")


def test_interpreter_falls_back_to_this_interpreter(db: Database) -> None:
    from rytp.transcribe.registry import interpreter_for

    assert interpreter_for(db, "fake-remote-diarizer") == sys.executable


def test_diarizer_rows_report_token_process_and_availability(db: Database) -> None:
    with registered(FakeDiarizer, GatedDiarizer):
        rows = {row[0]: row for row in base.diarizer_rows(db)}
    assert rows["fake-diarizer"][1] == "no"
    assert rows["fake-diarizer"][2] == "no"
    assert rows["fake-diarizer"][3] == "ready"
    assert rows["fake-gated-diarizer"][1] == "yes"


def test_wav_duration_ms_reads_the_header(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 8000)      # 0.5 s
    assert base.wav_duration_ms(path) == 500
```

- [ ] **Step 5: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_registry.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.base'`.

- [ ] **Step 6: Write `rytp/diarize/base.py`**

```python
"""The diarizer protocol and registry (contracts §6, design §6).

Classes, never instances — the Hugging Face token gate has to be answerable
*before* anything is constructed, because building a pyannote pipeline
without a token downloads two gigabytes only to fail on the token.

The gate itself, the per-engine interpreter lookup and the availability
summary are Part 3's (`rytp.transcribe.registry`) and are reused rather
than duplicated: an engine is an engine, and two implementations of "can
this run here" is how the two answers drift apart. Those helpers are
stdlib-only, so importing them keeps this module import-light — which
matters, because `rytp/diarize/pyannote.py` is imported inside a *foreign*
interpreter where only the standard library and its own dependency load.
"""

from __future__ import annotations

import wave
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rytp.models import DiarSegment, RytpError
from rytp.transcribe.registry import check_available, interpreter_for

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = [
    "DIARIZERS",
    "DiarizeError",
    "Diarizer",
    "diarizer_rows",
    "load_diarizer",
    "register_diarizer",
    "resolve_diarizer",
    "wav_duration_ms",
]


class DiarizeError(RytpError):
    """Diarization could not produce usable labels for this video."""


class Diarizer(Protocol):
    """Audio in, labelled time ranges out. Absolute milliseconds."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def diarize(self, audio: Path) -> Iterable[DiarSegment]: ...


DIARIZERS: dict[str, type[Diarizer]] = {}
"""Registered diarizer classes, by ``cls.name``."""


def register_diarizer(cls: type[Diarizer]) -> type[Diarizer]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    DIARIZERS[cls.name] = cls
    return cls


def resolve_diarizer(name: str) -> type[Diarizer]:
    """Look up a diarizer class. Raises ``ValueError`` naming the alternatives."""
    try:
        return DIARIZERS[name]
    except KeyError:
        available = ", ".join(sorted(DIARIZERS)) or "(none registered)"
        raise ValueError(f"unknown diarizer {name!r}; available: {available}") from None


def load_diarizer(db: Database, name: str, **kwargs: Any) -> Diarizer:
    """Resolve, gate, and construct a diarizer."""
    cls = resolve_diarizer(name)
    check_available(cls)
    if getattr(cls, "out_of_process", False):
        kwargs.setdefault("interpreter", interpreter_for(db, name))
    return cls(**kwargs)  # type: ignore[call-arg]


def diarizer_rows(db: Database) -> list[tuple[str, str, str, str]]:
    """Rows for ``rytp speakers engines``: name, token, out-of-process, state."""
    from rytp.transcribe.registry import availability

    return [
        (
            name,
            "yes" if getattr(cls, "requires_hf_token", False) else "no",
            "yes" if getattr(cls, "out_of_process", False) else "no",
            availability(db, cls),
        )
        for name, cls in sorted(DIARIZERS.items())
    ]


def wav_duration_ms(path: Path) -> int:
    """Length of a PCM WAV in milliseconds, from its header alone.

    Standard library only: the null diarizer needs a duration and must not
    pull numpy or :mod:`rytp.audio` into an import path that a foreign
    interpreter may follow.
    """
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
    if rate <= 0:
        raise DiarizeError(f"{path} reports a sample rate of {rate}")
    return frames * 1000 // rate
```

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_registry.py -v
```
Expected: 9 passed.

- [ ] **Step 8: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings. Fix import order with `ruff check --fix` if rule `I` complains.

- [ ] **Step 9: Commit**

```bash
git add rytp/constants.py rytp/diarize/__init__.py rytp/diarize/base.py \
  rytp/diarize/embed.py tests/fake_speaker_engines.py tests/test_diarize_registry.py
git commit -m "feat(diarize): diarizer protocol and the name-to-class registry"
```

---

### Task 2: The null diarizer and the package that makes names resolvable

A diarizer that puts every word in one bucket is not a toy: it is what a
single-speaker video actually needs, and it is the default, because
diarization is opt-in per video (design §6) and the default may never cost
GPU time.

**Files:**
- Create: `rytp/diarize/none.py`
- Create: `rytp/diarize/__init__.py`
- Test: `tests/test_diarize_none.py`

**Interfaces:**
- Consumes: `rytp.diarize.base.{register_diarizer, wav_duration_ms}`, `rytp.constants.NULL_DIARIZER_LABEL`.
- Produces: `rytp.diarize.none.NullDiarizer` (name `none`). Importing `rytp.diarize` registers every built-in engine.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_none.py`:

```python
"""The null diarizer: one label, the real duration, no model."""

from __future__ import annotations

import wave
from pathlib import Path

from rytp import constants as C
from rytp.diarize import base
from rytp.diarize.none import NullDiarizer


def _wav(path: Path, ms: int) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * (16 * ms))
    return path


def test_one_segment_covering_the_whole_file(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "a.wav", 2_500)
    segments = list(NullDiarizer().diarize(audio))
    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 2_500
    assert segments[0].local_label == C.NULL_DIARIZER_LABEL


def test_importing_the_package_registers_the_built_in_names() -> None:
    import rytp.diarize  # noqa: F401 - the import is the point

    assert base.resolve_diarizer("none") is NullDiarizer
    assert "pyannote" in base.DIARIZERS


def test_the_default_diarizer_constant_names_a_registered_engine() -> None:
    import rytp.diarize  # noqa: F401

    assert C.DEFAULT_DIARIZER in base.DIARIZERS
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_none.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.none'`.

- [ ] **Step 3: Write `rytp/diarize/none.py`**

```python
"""The null diarizer: every word belongs to one voice (design §6).

Not a placeholder. Most videos in this corpus are one person talking, and
for those this is the *correct* answer, produced for free. It is also the
default, which is what keeps diarization opt-in.

The segment covers exactly the audio, read from the WAV header, rather than
some sentinel large number: a segment that claims to run past the end of the
file would make every "how much did this label speak" figure a lie.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rytp import constants as C
from rytp.diarize.base import register_diarizer, wav_duration_ms
from rytp.models import DiarSegment


@register_diarizer
class NullDiarizer:
    """One label for the whole file."""

    name = "none"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        yield DiarSegment(
            start_ms=0,
            end_ms=wav_duration_ms(audio),
            local_label=C.NULL_DIARIZER_LABEL,
        )
```

- [ ] **Step 4: Fill `rytp/diarize/__init__.py`**

Task 1 created it empty. The finished file is:

```python
"""Speaker diarization and the speaker roster.

Importing this package is what makes an engine's name resolvable: each
module registers its class at import time, so a name that is never imported
is a name that does not exist.

Only the engine modules are imported here. `store`, `pipeline`, `link` and
`mapper` are imported by their callers, because `pyannote` and `embed` are
loaded inside a *foreign* interpreter by the out-of-process seam and must
not drag the database layer along with them.
"""

from rytp.diarize import embed, none, pyannote  # noqa: F401
```

**`rytp/diarize/pyannote.py` does not exist until Task 11.** Until then, write
the import line as

```python
from rytp.diarize import embed, none  # noqa: F401
```

and change the test's `assert "pyannote" in base.DIARIZERS` to
`assert "none" in base.DIARIZERS`. **Task 11, Steps 4 and 5 restore both
lines.** This is the only forward reference in the plan and it is called out
in both places.

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_none.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Prove the package stays import-light**

Nothing heavy may be pulled in by importing the diarize package — the
out-of-process child imports through it.

```bash
python -c "import sys, rytp.diarize; assert 'numpy' not in sys.modules and 'torch' not in sys.modules; print('clean')"
```
Expected: `clean`.

- [ ] **Step 7: Commit**

```bash
git add rytp/diarize/__init__.py rytp/diarize/none.py tests/test_diarize_none.py
git commit -m "feat(diarize): the null diarizer and engine registration on import"
```

---

### Task 3: Segment arithmetic, with no database and no engine in sight

Everything interesting about diarization output is arithmetic on time
ranges: which label owns a word, how much each label actually spoke, and
which slices of audio are worth feeding an embedder. None of it needs a
database, an engine, or a file, so none of it is mixed with any of those.
These are the sharpest tests in this part — write them first and the
pipeline task becomes plumbing.

**Files:**
- Create: `rytp/diarize/segments.py`
- Test: `tests/test_diarize_segments.py`

**Interfaces:**
- Consumes: `rytp.models.DiarSegment`, `rytp.constants.{EMBED_MIN_SEGMENT_MS, EMBED_MAX_SPEECH_MS}`.
- Produces: `WordSpan = tuple[int, int, int | None]` (word id, start_ms, end_ms), `merge_segments(segments) -> list[DiarSegment]`, `local_labels(segments) -> tuple[str, ...]`, `speech_ms_by_label(segments) -> dict[str, int]`, `label_words(words, segments) -> dict[int, str]`, `windows_for_label(segments, label, *, min_segment_ms=None, max_total_ms=None) -> list[tuple[int, int]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_segments.py`:

```python
"""Pure arithmetic over diarizer segments: no database, no audio, no engine."""

from __future__ import annotations

from rytp import constants as C
from rytp.diarize.segments import (
    label_words,
    local_labels,
    merge_segments,
    speech_ms_by_label,
    windows_for_label,
)
from rytp.models import DiarSegment


def seg(start: int, end: int, label: str = "SPEAKER_00") -> DiarSegment:
    return DiarSegment(start_ms=start, end_ms=end, local_label=label)


# -- merging ---------------------------------------------------------------


def test_touching_segments_of_one_label_merge() -> None:
    assert merge_segments([seg(0, 100), seg(100, 250)]) == [seg(0, 250)]


def test_overlapping_segments_of_one_label_merge() -> None:
    assert merge_segments([seg(0, 200), seg(150, 400)]) == [seg(0, 400)]


def test_segments_of_different_labels_never_merge() -> None:
    merged = merge_segments([seg(0, 200, "A"), seg(100, 400, "B")])
    assert merged == [seg(0, 200, "A"), seg(100, 400, "B")]


def test_merging_is_stable_and_sorted_by_start() -> None:
    merged = merge_segments([seg(500, 600, "B"), seg(0, 100, "A"), seg(200, 300, "A")])
    assert [(s.start_ms, s.local_label) for s in merged] == [
        (0, "A"),
        (200, "A"),
        (500, "B"),
    ]


# -- totals ----------------------------------------------------------------


def test_speech_totals_do_not_double_count_an_overlap() -> None:
    assert speech_ms_by_label([seg(0, 1_000, "A"), seg(500, 1_500, "A")]) == {"A": 1_500}


def test_local_labels_are_unique_and_sorted() -> None:
    assert local_labels([seg(0, 1, "B"), seg(2, 3, "A"), seg(4, 5, "B")]) == ("A", "B")


# -- labelling words -------------------------------------------------------


def test_each_word_takes_the_label_it_overlaps_most() -> None:
    words = [(1, 0, 400), (2, 400, 900), (3, 900, 1_400)]
    segments = [seg(0, 500, "A"), seg(500, 1_500, "B")]
    assert label_words(words, segments) == {1: "A", 2: "B", 3: "B"}


def test_a_word_straddling_a_boundary_goes_to_the_larger_share() -> None:
    # 100 ms in A, 300 ms in B.
    assert label_words([(1, 400, 800)], [seg(0, 500, "A"), seg(500, 900, "B")]) == {1: "B"}


def test_a_word_overlapping_nothing_is_left_unlabelled() -> None:
    # A gap in diarization is not an error: it is a stretch nobody labelled.
    assert label_words([(1, 5_000, 5_400)], [seg(0, 500, "A")]) == {}


def test_a_tie_goes_to_the_earlier_segment() -> None:
    assert label_words([(1, 400, 600)], [seg(0, 500, "A"), seg(500, 1_000, "B")]) == {1: "A"}


def test_a_zero_length_word_is_placed_by_its_start() -> None:
    # Caption-tier rows have no end; an aligned row can still collapse.
    assert label_words([(1, 600, 600), (2, 700, None)], [seg(500, 1_000, "A")]) == {
        1: "A",
        2: "A",
    }


def test_words_may_arrive_in_any_order() -> None:
    words = [(3, 900, 1_400), (1, 0, 400), (2, 400, 900)]
    segments = [seg(0, 500, "A"), seg(500, 1_500, "B")]
    assert label_words(words, segments) == {1: "A", 2: "B", 3: "B"}


def test_no_segments_labels_nothing() -> None:
    assert label_words([(1, 0, 100)], []) == {}


def test_labelling_ten_thousand_words_is_not_quadratic() -> None:
    # A real hour: ~10k words against ~2k segments. If this takes more than
    # a moment the implementation went back to scanning every segment.
    words = [(i, i * 350, i * 350 + 300) for i in range(10_000)]
    segments = [seg(i * 1_750, (i + 1) * 1_750, f"SPEAKER_{i % 3:02d}") for i in range(2_000)]
    assert len(label_words(words, segments)) == 10_000


# -- embedding windows -----------------------------------------------------


def test_windows_prefer_the_longest_segments_and_stay_chronological() -> None:
    segments = [
        seg(0, 2_000, "A"),
        seg(10_000, 15_000, "A"),
        seg(20_000, 21_500, "A"),
        seg(5_000, 6_000, "B"),
    ]
    assert windows_for_label(segments, "A", max_total_ms=7_000) == [
        (0, 2_000),
        (10_000, 15_000),
    ]


def test_windows_truncate_the_last_segment_to_the_budget() -> None:
    assert windows_for_label([seg(0, 60_000, "A")], "A", max_total_ms=30_000) == [
        (0, 30_000)
    ]


def test_windows_skip_segments_below_the_minimum_length() -> None:
    segments = [seg(0, 400, "A"), seg(1_000, 6_000, "A")]
    assert windows_for_label(segments, "A", min_segment_ms=1_000) == [(1_000, 6_000)]


def test_windows_fall_back_to_short_segments_when_there_is_nothing_else() -> None:
    # A label made entirely of interjections still deserves a vector; the
    # "is there enough speech at all" decision belongs to the caller.
    assert windows_for_label([seg(0, 400, "A"), seg(900, 1_200, "A")], "A") == [
        (0, 400),
        (900, 1_200),
    ]


def test_windows_for_an_unknown_label_are_empty() -> None:
    assert windows_for_label([seg(0, 1_000, "A")], "B") == []


def test_window_defaults_come_from_constants() -> None:
    assert windows_for_label([seg(0, 10_000_000, "A")], "A") == [
        (0, C.EMBED_MAX_SPEECH_MS)
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_segments.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.segments'`.

- [ ] **Step 3: Write `rytp/diarize/segments.py`**

```python
"""Time arithmetic over diarizer output (design §4, §6).

No database, no engine, no file. Everything here is a function of a list of
:class:`~rytp.models.DiarSegment` and, at most, a list of word timings —
which is what makes it the part of diarization that can be tested to death.

Three ideas carry the rest of this plan:

* **A word belongs to the label it overlaps most.** Not the label at its
  midpoint, which mis-assigns a long word at a turn boundary, and not the
  first match, which depends on segment order.
* **A word that overlaps no segment stays unlabelled.** A gap in the
  diarizer's output means "nobody was labelled here", which is information,
  not an error. It is exactly the case `video_speakers.speaker_id IS NULL`
  exists to describe one level up.
* **Overlap totals are computed on merged segments.** A diarizer that emits
  two overlapping turns for one voice must not make that voice look like it
  spoke for twice as long.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rytp import constants as C
from rytp.models import DiarSegment

#: One word, as the labeller needs it: its row id and its timings. ``end_ms``
#: is optional because a caption-tier row has none (contracts §3).
WordSpan = tuple[int, int, "int | None"]

__all__ = [
    "WordSpan",
    "label_words",
    "local_labels",
    "merge_segments",
    "speech_ms_by_label",
    "windows_for_label",
]


def merge_segments(segments: Iterable[DiarSegment]) -> list[DiarSegment]:
    """Merge touching and overlapping segments **of the same label**.

    Returned sorted by start, then by label, so the result is a stable
    function of the input set rather than of its order.
    """
    by_label: dict[str, list[DiarSegment]] = {}
    for segment in segments:
        by_label.setdefault(segment.local_label, []).append(segment)

    merged: list[DiarSegment] = []
    for label, group in by_label.items():
        group.sort(key=lambda s: (s.start_ms, s.end_ms))
        start, end = group[0].start_ms, group[0].end_ms
        for segment in group[1:]:
            if segment.start_ms <= end:
                end = max(end, segment.end_ms)
                continue
            merged.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))
            start, end = segment.start_ms, segment.end_ms
        merged.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))

    merged.sort(key=lambda s: (s.start_ms, s.local_label))
    return merged


def local_labels(segments: Iterable[DiarSegment]) -> tuple[str, ...]:
    """Every distinct label the diarizer emitted, sorted."""
    return tuple(sorted({segment.local_label for segment in segments}))


def speech_ms_by_label(segments: Iterable[DiarSegment]) -> dict[str, int]:
    """How long each label actually spoke, overlaps counted once."""
    totals: dict[str, int] = {}
    for segment in merge_segments(segments):
        totals[segment.local_label] = (
            totals.get(segment.local_label, 0) + segment.end_ms - segment.start_ms
        )
    return totals


def label_words(
    words: Iterable[WordSpan], segments: Sequence[DiarSegment]
) -> dict[int, str]:
    """Map word id to local label by maximum overlap.

    A sweep, not a search: both sides are sorted once and a small active set
    is carried forward, so an hour of words against thousands of segments
    stays linear. Words unlabelled by every segment are simply absent from
    the result, which is what the caller writes as NULL.
    """
    ordered = sorted(segments, key=lambda s: (s.start_ms, s.end_ms, s.local_label))
    if not ordered:
        return {}

    assigned: dict[int, str] = {}
    active: list[DiarSegment] = []
    cursor = 0

    for word_id, raw_start, raw_end in sorted(words, key=lambda w: (w[1], w[0])):
        # A zero-length or end-less word behaves as one millisecond wide, so
        # "which segment contains its start" falls out of the same maths.
        start = raw_start
        end = max(raw_end if raw_end is not None else start, start + 1)

        while cursor < len(ordered) and ordered[cursor].start_ms <= end:
            active.append(ordered[cursor])
            cursor += 1
        active = [s for s in active if s.end_ms > start]

        best_label: str | None = None
        best_overlap = 0
        for segment in active:
            overlap = min(segment.end_ms, end) - max(segment.start_ms, start)
            # Strictly greater: a tie keeps the earlier segment, because
            # `active` is in start order.
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = segment.local_label
        if best_label is not None:
            assigned[word_id] = best_label

    return assigned


def windows_for_label(
    segments: Iterable[DiarSegment],
    label: str,
    *,
    min_segment_ms: int | None = None,
    max_total_ms: int | None = None,
) -> list[tuple[int, int]]:
    """Choose up to ``max_total_ms`` of this label's speech to embed.

    Longest segments first, because a long turn is a cleaner sample of a
    voice than a scatter of interjections, then returned in chronological
    order so the concatenated audio sounds like speech. Deterministic: ties
    on length are broken by start time.

    If every segment is shorter than ``min_segment_ms`` the filter is
    dropped rather than returning nothing — whether there is *enough* speech
    to embed at all is the caller's decision (`C.EMBED_MIN_SPEECH_MS`).
    """
    floor = C.EMBED_MIN_SEGMENT_MS if min_segment_ms is None else min_segment_ms
    budget = C.EMBED_MAX_SPEECH_MS if max_total_ms is None else max_total_ms

    mine = [s for s in merge_segments(segments) if s.local_label == label]
    if not mine:
        return []
    long_enough = [s for s in mine if s.end_ms - s.start_ms >= floor]
    candidates = long_enough or mine

    chosen: list[tuple[int, int]] = []
    remaining = budget
    for segment in sorted(candidates, key=lambda s: (-(s.end_ms - s.start_ms), s.start_ms)):
        if remaining <= 0:
            break
        take = min(segment.end_ms - segment.start_ms, remaining)
        chosen.append((segment.start_ms, segment.start_ms + take))
        remaining -= take

    chosen.sort()
    return chosen
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_segments.py -v
```
Expected: 19 passed, the ten-thousand-word case in well under a second.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/segments.py tests/test_diarize_segments.py
git commit -m "feat(diarize): overlap-based word labelling and embedding window choice"
```

---

### Task 4: The roster store

`speakers` is the list of real people. It is small — tens of rows, not
thousands — which is why alias matching can be done in Python and why
nothing here needs an index beyond the unique label.

**Files:**
- Create: `rytp/diarize/store.py`
- Test: `tests/test_speakers_roster.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.{InvalidInputError, NotFoundError, utc_now_iso}`.
- Produces:
  - `Speaker(id, label, aliases: tuple[str, ...], notes: str | None, created_at: str)` and `Speaker.from_row(row)`
  - `RosterRow(speaker_id, label, aliases, notes, n_labels, n_videos)`
  - `parse_aliases(text: str) -> tuple[str, ...]`
  - `add_speaker(db, label, *, aliases=(), notes=None) -> tuple[int, bool]`
  - `update_speaker(db, speaker_id, *, label=None, aliases=None, notes=None) -> None`
  - `add_aliases(db, speaker_id, aliases) -> tuple[str, ...]`
  - `list_speakers(db) -> list[Speaker]`, `roster_rows(db) -> list[RosterRow]`
  - `get_speaker(db, speaker_id) -> Speaker`, `find_speaker(db, ref) -> Speaker | None`, `resolve_speaker(db, ref) -> Speaker`
  - `find_alias_collisions(db, aliases, *, exclude_speaker_id=None) -> list[tuple[str, int]]`
  - `remove_speaker(db, speaker_id) -> int` — deletes the roster entry, returns how many local labels it left unnamed

- [ ] **Step 1: Write the failing test**

Create `tests/test_speakers_roster.py`:

```python
"""The global roster: one row per real person, addressable by alias."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.models import InvalidInputError, NotFoundError
from tests.fakes import make_video


def test_add_speaker_returns_the_id_and_says_it_was_created(db: Database) -> None:
    speaker_id, created = store.add_speaker(db, "Host One", aliases=("host", "h1"))
    assert created is True
    speaker = store.get_speaker(db, speaker_id)
    assert speaker.label == "Host One"
    assert speaker.aliases == ("host", "h1")


def test_adding_the_same_label_twice_is_idempotent(db: Database) -> None:
    first, _ = store.add_speaker(db, "Host One")
    second, created_second = store.add_speaker(db, "Host One", aliases=("ignored",))
    assert second == first
    assert created_second is False
    # The second call must not silently rewrite the aliases.
    assert store.get_speaker(db, first).aliases == ()


def test_an_empty_label_is_refused(db: Database) -> None:
    with pytest.raises(InvalidInputError):
        store.add_speaker(db, "   ")


def test_get_speaker_names_the_missing_id(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        store.get_speaker(db, 4242)


def test_aliases_round_trip_through_json(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Каспар Хаузер", aliases=("каспар", "кх"))
    assert store.get_speaker(db, speaker_id).aliases == ("каспар", "кх")


def test_add_aliases_appends_without_duplicating(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One", aliases=("host",))
    assert store.add_aliases(db, speaker_id, ("HOST", "boss")) == ("host", "boss")


def test_update_speaker_can_rename_and_renotate(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.update_speaker(db, speaker_id, label="Host Uno", notes="the one in the hat")
    speaker = store.get_speaker(db, speaker_id)
    assert (speaker.label, speaker.notes) == ("Host Uno", "the one in the hat")


def test_find_speaker_accepts_an_id_a_label_or_an_alias(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Guest Two", aliases=("g2",))
    assert store.find_speaker(db, str(speaker_id)).id == speaker_id
    assert store.find_speaker(db, "Guest Two").id == speaker_id
    assert store.find_speaker(db, "guest two").id == speaker_id
    assert store.find_speaker(db, "G2").id == speaker_id
    assert store.find_speaker(db, "nobody") is None


def test_resolve_speaker_raises_rather_than_returning_none(db: Database) -> None:
    with pytest.raises(NotFoundError, match="nobody"):
        store.resolve_speaker(db, "nobody")


def test_alias_collisions_report_another_speakers_label_and_aliases(db: Database) -> None:
    one, _ = store.add_speaker(db, "Host One", aliases=("boss",))
    two, _ = store.add_speaker(db, "Guest Two")
    collisions = store.find_alias_collisions(db, ("boss", "guest two", "fresh"))
    assert sorted(alias for alias, _ in collisions) == ["boss", "guest two"]
    assert {sid for _, sid in collisions} == {one, two}


def test_removing_a_speaker_leaves_their_labels_as_unnamed_voices(
    db: Database,
) -> None:
    # contracts §5: ON DELETE SET NULL, not cascade. Losing the roster entry
    # must not lose the knowledge that this video had two distinct speakers.
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_id = make_video(db)
    for local in ("SPEAKER_00", "SPEAKER_01"):
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine) "
            "VALUES (?, ?, 'fake-diarizer')",
            (video_id, local),
        )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE local_label = 'SPEAKER_00'",
        (speaker_id,),
    )

    assert store.remove_speaker(db, speaker_id) == 1
    assert store.find_speaker(db, "Host One") is None
    rows = db.conn.execute(
        "SELECT local_label, speaker_id FROM video_speakers WHERE video_id = ? "
        "ORDER BY local_label",
        (video_id,),
    ).fetchall()
    assert [(r["local_label"], r["speaker_id"]) for r in rows] == [
        ("SPEAKER_00", None),
        ("SPEAKER_01", None),
    ]


def test_removing_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError, match="4242"):
        store.remove_speaker(db, 4242)


def test_a_speaker_does_not_collide_with_itself(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One", aliases=("boss",))
    assert store.find_alias_collisions(db, ("boss",), exclude_speaker_id=speaker_id) == []


def test_parse_aliases_splits_and_trims_a_comma_separated_flag() -> None:
    assert store.parse_aliases(" host , , h1 ,host ") == ("host", "h1")
    assert store.parse_aliases("") == ()


def test_roster_rows_count_labels_and_videos(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_a = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    video_b = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    for video_id in (video_a, video_b):
        for label in ("SPEAKER_00", "SPEAKER_01"):
            db.conn.execute(
                "INSERT INTO video_speakers (video_id, local_label) VALUES (?, ?)",
                (video_id, label),
            )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE local_label = 'SPEAKER_00'",
        (speaker_id,),
    )
    row = store.roster_rows(db)[0]
    assert (row.label, row.n_labels, row.n_videos) == ("Host One", 2, 2)


def test_roster_rows_are_ordered_by_label(db: Database) -> None:
    store.add_speaker(db, "Guest Two")
    store.add_speaker(db, "Host One")
    assert [row.label for row in store.roster_rows(db)] == ["Guest Two", "Host One"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_speakers_roster.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.store'`.

- [ ] **Step 3: Write the roster half of `rytp/diarize/store.py`**

```python
"""Every SQL statement about speakers and their per-video labels.

Two levels, and the distinction is the design (design §4):

``speakers``
    The roster of real people. Small — tens of rows — which is why alias
    matching happens in Python rather than in an index.

``video_speakers``
    One row per diarizer label per video. ``SPEAKER_04`` in some video is a
    real, citable entity whether or not anybody ever named it, and
    ``speaker_id`` is nullable precisely to say "a distinct voice I never
    bothered to name".

Linking a label to a person updates **one row**. The old implementation
back-filled `words.speaker_id` across thousands of rows on every mapping;
the new schema points `words.video_speaker_id` at the label instead, so the
person can be decided, changed and undecided for the cost of one UPDATE.

This module holds the queries and nothing else, so that `rytp/commands/`
never writes SQL and `rytp/tui/` never imports `rytp/commands/`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database
from rytp.diarize.segments import WordSpan
from rytp.models import DiarSegment, InvalidInputError, NotFoundError, utc_now_iso


@dataclass(frozen=True)
class Speaker:
    """One row of the global roster."""

    id: int
    label: str
    aliases: tuple[str, ...]
    notes: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Speaker:
        return cls(
            id=int(row["id"]),
            label=row["label"],
            aliases=_decode_aliases(row["aliases_json"]),
            notes=row["notes"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class RosterRow:
    """A roster entry with the two counts the mapper's right pane shows."""

    speaker_id: int
    label: str
    aliases: tuple[str, ...]
    notes: str | None
    n_labels: int
    n_videos: int


def _decode_aliases(raw: str | None) -> tuple[str, ...]:
    """Tolerate a malformed aliases_json rather than failing a whole listing."""
    try:
        values = json.loads(raw or "[]")
    except ValueError:
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def parse_aliases(text: str) -> tuple[str, ...]:
    """Split a comma-separated flag into aliases (contracts §5).

    `Param.type` is a scalar, so a multi-valued flag is a string the handler
    parses. Trims, drops empties, and de-duplicates case-insensitively while
    keeping the first spelling the user typed.
    """
    out: list[str] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        alias = chunk.strip()
        if not alias or alias.casefold() in seen:
            continue
        seen.add(alias.casefold())
        out.append(alias)
    return tuple(out)


def add_speaker(
    db: Database,
    label: str,
    *,
    aliases: Iterable[str] = (),
    notes: str | None = None,
) -> tuple[int, bool]:
    """Insert a roster entry. Returns ``(speaker_id, was_created)``.

    Idempotent on ``label``: an existing speaker is returned untouched
    rather than having its aliases silently rewritten, and the ``False``
    lets the caller say so.
    """
    clean = label.strip()
    if not clean:
        raise InvalidInputError("a speaker needs a label")
    existing = db.conn.execute(
        "SELECT id FROM speakers WHERE label = ?", (clean,)
    ).fetchone()
    if existing is not None:
        return int(existing["id"]), False
    cursor = db.conn.execute(
        "INSERT INTO speakers (label, aliases_json, notes, created_at) VALUES (?, ?, ?, ?)",
        (clean, json.dumps(list(aliases), ensure_ascii=False), notes, utc_now_iso()),
    )
    return int(cursor.lastrowid), True


def update_speaker(
    db: Database,
    speaker_id: int,
    *,
    label: str | None = None,
    aliases: Sequence[str] | None = None,
    notes: str | None = None,
) -> None:
    """Change the mutable fields of an existing roster entry."""
    get_speaker(db, speaker_id)
    sets: list[str] = []
    params: list[object] = []
    if label is not None:
        clean = label.strip()
        if not clean:
            raise InvalidInputError("a speaker needs a label")
        sets.append("label = ?")
        params.append(clean)
    if aliases is not None:
        sets.append("aliases_json = ?")
        params.append(json.dumps(list(aliases), ensure_ascii=False))
    if notes is not None:
        sets.append("notes = ?")
        params.append(notes)
    if not sets:
        return
    params.append(speaker_id)
    db.conn.execute(f"UPDATE speakers SET {', '.join(sets)} WHERE id = ?", params)


def add_aliases(db: Database, speaker_id: int, aliases: Iterable[str]) -> tuple[str, ...]:
    """Append aliases, case-insensitively de-duplicated. Returns the new set."""
    current = get_speaker(db, speaker_id).aliases
    merged = parse_aliases(",".join([*current, *aliases]))
    update_speaker(db, speaker_id, aliases=merged)
    return merged


def list_speakers(db: Database) -> list[Speaker]:
    """The whole roster, ordered by label."""
    rows = db.conn.execute("SELECT * FROM speakers ORDER BY label").fetchall()
    return [Speaker.from_row(row) for row in rows]


def roster_rows(db: Database) -> list[RosterRow]:
    """The roster plus how many labels and videos each person is linked to."""
    rows = db.conn.execute(
        """
        SELECT s.id, s.label, s.aliases_json, s.notes,
               COUNT(vs.id)                AS n_labels,
               COUNT(DISTINCT vs.video_id) AS n_videos
        FROM speakers s
        LEFT JOIN video_speakers vs ON vs.speaker_id = s.id
        GROUP BY s.id
        ORDER BY s.label
        """
    ).fetchall()
    return [
        RosterRow(
            speaker_id=int(row["id"]),
            label=row["label"],
            aliases=_decode_aliases(row["aliases_json"]),
            notes=row["notes"],
            n_labels=int(row["n_labels"]),
            n_videos=int(row["n_videos"]),
        )
        for row in rows
    ]


def get_speaker(db: Database, speaker_id: int) -> Speaker:
    """One roster entry by id. Raises :class:`NotFoundError`."""
    row = db.conn.execute("SELECT * FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"speaker {speaker_id} is not in the roster")
    return Speaker.from_row(row)


def find_speaker(db: Database, ref: str) -> Speaker | None:
    """Resolve a user-typed reference: an id, a label, or an alias.

    Case-insensitive on label and alias, because nobody types a roster entry
    the same way twice. Labels are tried before aliases so an exact name
    always wins.
    """
    needle = ref.strip()
    if not needle:
        return None
    if needle.isdigit():
        row = db.conn.execute(
            "SELECT * FROM speakers WHERE id = ?", (int(needle),)
        ).fetchone()
        if row is not None:
            return Speaker.from_row(row)
    folded = needle.casefold()
    roster = list_speakers(db)
    for speaker in roster:
        if speaker.label.casefold() == folded:
            return speaker
    for speaker in roster:
        if any(alias.casefold() == folded for alias in speaker.aliases):
            return speaker
    return None


def resolve_speaker(db: Database, ref: str) -> Speaker:
    """:func:`find_speaker`, raising :class:`NotFoundError` instead of returning None."""
    speaker = find_speaker(db, ref)
    if speaker is None:
        raise NotFoundError(f"no speaker matches {ref!r}")
    return speaker


def remove_speaker(db: Database, speaker_id: int) -> int:
    """Delete a roster entry. Returns how many local labels it left unnamed.

    `video_speakers.speaker_id` is `ON DELETE SET NULL` (contracts §3), and
    that is the whole design rather than a convenience: removing a person
    from the roster must not destroy the knowledge that some video had four
    distinct speakers. The labels survive as voices nobody has named, which
    is exactly what they were before anybody named them.

    No files are involved, so contracts §5 asks for no `--dry-run` and no
    `--yes`; the caller decides whether the number returned is large enough
    to be worth confirming first.
    """
    get_speaker(db, speaker_id)
    linked = db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE speaker_id = ?", (speaker_id,)
    ).fetchone()[0]
    db.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
    return int(linked)


def find_alias_collisions(
    db: Database,
    aliases: Iterable[str],
    *,
    exclude_speaker_id: int | None = None,
) -> list[tuple[str, int]]:
    """``[(alias, other_speaker_id), …]`` for aliases already spoken for.

    An alias that matches another person's *label* collides too: the roster
    is looked up by either, so a duplicate would make lookups ambiguous.
    Not an error by itself — the caller decides whether to warn or refuse.
    """
    wanted = {alias.casefold(): alias for alias in aliases if alias.strip()}
    if not wanted:
        return []
    found: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for speaker in list_speakers(db):
        if speaker.id == exclude_speaker_id:
            continue
        for candidate in (speaker.label, *speaker.aliases):
            typed = wanted.get(candidate.casefold())
            if typed is not None and (typed, speaker.id) not in seen:
                seen.add((typed, speaker.id))
                found.append((typed, speaker.id))
    return found
```

**Import block:** write only the imports this half uses — `json`, `sqlite3`,
`Iterable`, `Sequence`, `dataclass`, `Database`, `InvalidInputError`,
`NotFoundError`, `utc_now_iso`. Task 5 adds `Mapping`, `rytp.constants as C`
and `WordSpan`; Task 10 adds `DiarSegment`. Importing them early would leave
`ruff` failing on unused imports at this commit, and every task in this plan
ends green.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_speakers_roster.py -v
```
Expected: 16 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/store.py tests/test_speakers_roster.py
git commit -m "feat(speakers): the global roster, addressable by id, label or alias"
```

---

### Task 5: Local labels and one-row linking

This is the task the whole part turns on. **Read both sentences:**

1. `words.video_speaker_id` **is** written, once, at diarize time, by
   `stamp_words`. Without it the left pane cannot say "SPEAKER_04 — 340
   words, 12 minutes", the search filter has nothing to filter on, and
   `utterances` cannot be split on speaker change.
2. Linking a label to a *person* afterwards **never** touches `words`. It
   is one `UPDATE video_speakers SET speaker_id = ? WHERE id = ?`. The old
   implementation back-filled thousands of word rows on every mapping; the
   new schema exists so that it does not have to, and a test below asserts
   that exactly one row changes.

**Files:**
- Modify: `rytp/diarize/store.py` (append)
- Test: `tests/test_speakers_labels.py`

**Interfaces:**
- Consumes: Task 4's module; `rytp.diarize.segments.WordSpan`; `rytp.constants.SPEAKER_LIST_LIMIT`.
- Produces:
  - `LabelRow(video_speaker_id, video_id, local_label, speaker_id, speaker_label, engine, n_words, speech_ms, has_embedding)`
  - `VideoRow(video_id, title, published_at, n_labels, n_unmapped)`
  - `upsert_video_speaker(db, video_id, local_label, *, engine: str) -> int` — `engine` is the diarizer that produced the label (contracts §3)
  - `label_rows(db, video_id) -> list[LabelRow]`, `get_label(db, video_speaker_id) -> LabelRow`, `find_label(db, video_id, local_label) -> LabelRow`
  - `link_video_speaker(db, video_speaker_id, speaker_id: int | None) -> None`
  - `set_embedding(db, video_speaker_id, blob: bytes | None) -> None`
  - `clear_video_speakers(db, video_id) -> int`
  - `word_spans(db, video_id) -> list[WordSpan]`, `stamp_words(db, video_id, by_word_id: Mapping[int, int]) -> int`
  - `diarized_videos(db, *, only_unmapped=False, limit=C.SPEAKER_LIST_LIMIT) -> list[VideoRow]`
  - `linked_labels(db, speaker_id) -> list[LabelRow]`
- Produces for every later task: `tests/test_speakers_labels.py::DIARIZER` and `add_label(db, video_id, local_label, *, engine=DIARIZER) -> int`. Six other test modules import them alongside `add_words`, so a label in a test never has to spell the engine out.
- **Not** produced here: between-word pause statistics. Design §9 needs them for render gap defaults, but Part 6 owns that measurement — `rytp/render/pauses.py::measure_pause_stats`, which scopes by video, by `video_speaker_id` and by roster label, and guards against the zero-inflation this corpus is full of (design §6: 78.7% of the existing transcript's word gaps are exactly zero, because the transcriber absorbed every pause into an adjacent word, so a naive median is 0 ms and the renderer inserts nothing). Two definitions of "natural pause" in one codebase is how they drift. Part 7 provides the speaker dimension those queries read — `words.video_speaker_id` and `video_speakers.speaker_id` — and nothing more.

- [ ] **Step 1: Write the failing test**

Create `tests/test_speakers_labels.py`:

```python
"""Per-video labels: stamped onto words once, linked to a person one row at a time."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.models import NotFoundError
from tests.fakes import make_video

ENGINE = "fake:test"

#: The diarizer name a test label claims to come from. `video_speakers.engine`
#: is NOT NULL (contracts §3), so every label needs one; only the tests that
#: are *about* provenance care which.
DIARIZER = "fake-diarizer"


def add_label(
    db: Database, video_id: int, local_label: str, *, engine: str = DIARIZER
) -> int:
    """`store.upsert_video_speaker` with the test engine filled in."""
    return store.upsert_video_speaker(db, video_id, local_label, engine=engine)


def add_words(db: Database, video_id: int, spans: list[tuple[int, int]]) -> list[int]:
    """Insert aligned words at the given (start_ms, end_ms) and return their ids."""
    ids: list[int] = []
    for ordinal, (start, end) in enumerate(spans):
        cursor = db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
            "stem, source, engine) VALUES (?, ?, ?, ?, ?, ?, ?, 'aligned', ?)",
            (video_id, ordinal, start, end, f"w{ordinal}", f"w{ordinal}", f"w{ordinal}", ENGINE),
        )
        ids.append(int(cursor.lastrowid))
    return ids


def test_upsert_is_idempotent_on_video_and_label(db: Database) -> None:
    video_id = make_video(db)
    first = add_label(db, video_id, "SPEAKER_00")
    second = add_label(db, video_id, "SPEAKER_00")
    assert first == second
    assert len(store.label_rows(db, video_id)) == 1


def test_a_label_starts_unlinked(db: Database) -> None:
    video_id = make_video(db)
    add_label(db, video_id, "SPEAKER_04")
    row = store.label_rows(db, video_id)[0]
    assert row.local_label == "SPEAKER_04"
    assert row.speaker_id is None
    assert row.speaker_label is None


def test_a_label_records_which_diarizer_produced_it(db: Database) -> None:
    # contracts §3, design §11: a corpus built with more than one engine has
    # to stay interpretable, which is why words.engine exists and why this
    # column does too.
    video_id = make_video(db)
    store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="pyannote")
    assert store.label_rows(db, video_id)[0].engine == "pyannote"


def test_re_running_with_another_diarizer_overwrites_the_provenance(
    db: Database,
) -> None:
    video_id = make_video(db)
    first = store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="none")
    second = store.upsert_video_speaker(db, video_id, "SPEAKER_00", engine="pyannote")
    assert first == second
    assert store.get_label(db, first).engine == "pyannote"


def test_stamp_words_writes_the_label_onto_each_word(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800), (800, 1_200)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    assert store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id)) == 3
    row = store.label_rows(db, video_id)[0]
    assert (row.n_words, row.speech_ms) == (3, 1_200)


def test_stamp_words_clears_words_left_out_of_the_assignment(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    store.stamp_words(db, video_id, {word_ids[0]: label_id})
    remaining = db.conn.execute(
        "SELECT video_speaker_id FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    assert [row[0] for row in remaining] == [label_id, None]


def test_linking_a_label_to_a_person_changes_exactly_one_row(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(i * 400, i * 400 + 300) for i in range(50)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    speaker_id, _ = store.add_speaker(db, "Host One")

    before = db.conn.total_changes
    store.link_video_speaker(db, label_id, speaker_id)
    assert db.conn.total_changes - before == 1

    assert store.get_label(db, label_id).speaker_label == "Host One"


def test_unlinking_also_changes_exactly_one_row(db: Database) -> None:
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)

    before = db.conn.total_changes
    store.link_video_speaker(db, label_id, None)
    assert db.conn.total_changes - before == 1
    assert store.get_label(db, label_id).speaker_id is None


def test_linking_never_touches_the_word_rows(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    speaker_id, _ = store.add_speaker(db, "Host One")

    def fingerprint() -> str:
        return db.conn.execute(
            "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
        ).fetchone()[0]

    before = fingerprint()
    store.link_video_speaker(db, label_id, speaker_id)
    assert fingerprint() == before


def test_deleting_a_speaker_leaves_the_label_but_unlinks_it(db: Database) -> None:
    # ON DELETE SET NULL (contracts §3): the local voice survives the roster.
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)
    db.conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
    assert store.get_label(db, label_id).speaker_id is None


def test_find_label_names_the_missing_one(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(NotFoundError, match="SPEAKER_09"):
        store.find_label(db, video_id, "SPEAKER_09")


def test_clear_video_speakers_removes_the_rows_and_nulls_the_words(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    assert store.clear_video_speakers(db, video_id) == 1
    assert store.label_rows(db, video_id) == []
    assert db.conn.execute("SELECT video_speaker_id FROM words").fetchone()[0] is None


def test_word_spans_returns_ids_and_timings_in_ordinal_order(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 400), (400, 800)])
    assert store.word_spans(db, video_id) == [
        (word_ids[0], 0, 400),
        (word_ids[1], 400, 800),
    ]


def test_diarized_videos_lists_only_videos_with_labels(db: Database) -> None:
    mapped = make_video(db, external_id="VIDEO_A", url="https://example.invalid/a")
    bare = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    add_label(db, mapped, "SPEAKER_00")
    rows = store.diarized_videos(db)
    assert [row.video_id for row in rows] == [mapped]
    assert bare not in [row.video_id for row in rows]
    assert rows[0].n_unmapped == 1


def test_diarized_videos_can_show_only_the_ones_still_unmapped(db: Database) -> None:
    video_id = make_video(db)
    label_id = add_label(db, video_id, "SPEAKER_00")
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(db, label_id, speaker_id)
    assert store.diarized_videos(db, only_unmapped=True) == []
    assert len(store.diarized_videos(db)) == 1


def test_linked_labels_finds_every_label_of_one_person(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    for external_id in ("VIDEO_A", "VIDEO_B"):
        video_id = make_video(
            db, external_id=external_id, url=f"https://example.invalid/{external_id}"
        )
        label_id = add_label(db, video_id, "SPEAKER_00")
        store.link_video_speaker(db, label_id, speaker_id)
    assert len(store.linked_labels(db, speaker_id)) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_speakers_labels.py -v
```
Expected: `AttributeError: module 'rytp.diarize.store' has no attribute 'upsert_video_speaker'`.

- [ ] **Step 3: Append the label half to `rytp/diarize/store.py`**

```python
@dataclass(frozen=True)
class LabelRow:
    """One diarizer label of one video, with everything a pane needs to show it."""

    video_speaker_id: int
    video_id: int
    local_label: str
    speaker_id: int | None
    speaker_label: str | None
    engine: str
    n_words: int
    speech_ms: int
    has_embedding: bool


@dataclass(frozen=True)
class VideoRow:
    """A diarized video, for the mapper's video chooser."""

    video_id: int
    title: str
    published_at: str | None
    n_labels: int
    n_unmapped: int


_LABEL_SELECT = """
    SELECT vs.id, vs.video_id, vs.local_label, vs.speaker_id, vs.engine,
           s.label                  AS speaker_label,
           vs.embedding IS NOT NULL AS has_embedding,
           COUNT(w.id)              AS n_words,
           COALESCE(SUM(COALESCE(w.end_ms, w.start_ms) - w.start_ms), 0) AS speech_ms
    FROM video_speakers vs
    LEFT JOIN speakers s ON s.id = vs.speaker_id
    LEFT JOIN words w    ON w.video_speaker_id = vs.id
"""


def _label_row(row: sqlite3.Row) -> LabelRow:
    return LabelRow(
        video_speaker_id=int(row["id"]),
        video_id=int(row["video_id"]),
        local_label=row["local_label"],
        speaker_id=None if row["speaker_id"] is None else int(row["speaker_id"]),
        speaker_label=row["speaker_label"],
        engine=row["engine"],
        n_words=int(row["n_words"]),
        speech_ms=int(row["speech_ms"]),
        has_embedding=bool(row["has_embedding"]),
    )


def upsert_video_speaker(
    db: Database, video_id: int, local_label: str, *, engine: str
) -> int:
    """Create (or find) the row for one diarizer label of one video.

    ``engine`` is which diarizer produced the label (contracts §3). It is
    recorded for the same reason `words.engine` is: design §11 says no stage
    may assume a particular engine, so a corpus built with more than one has
    to stay interpretable. A re-run with a *different* diarizer overwrites
    it, because the label it is describing is that run's.

    Never clears an existing `speaker_id`: re-running diarization on a video
    whose labels happen to come out the same should not throw the mapping
    away. Re-*transcribing* does throw it away, and that is Part 3's job in
    the same transaction that replaces the words (contracts §4) — not this
    function's.
    """
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) VALUES (?, ?, ?) "
        "ON CONFLICT (video_id, local_label) DO UPDATE SET engine = excluded.engine",
        (video_id, local_label, engine),
    )
    row = db.conn.execute(
        "SELECT id FROM video_speakers WHERE video_id = ? AND local_label = ?",
        (video_id, local_label),
    ).fetchone()
    return int(row["id"])


def label_rows(db: Database, video_id: int) -> list[LabelRow]:
    """Every local label of one video, ordered by label name."""
    rows = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.video_id = ? GROUP BY vs.id ORDER BY vs.local_label",
        (video_id,),
    ).fetchall()
    return [_label_row(row) for row in rows]


def get_label(db: Database, video_speaker_id: int) -> LabelRow:
    """One local label by its id. Raises :class:`NotFoundError`."""
    row = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.id = ? GROUP BY vs.id", (video_speaker_id,)
    ).fetchone()
    if row is None or row["id"] is None:
        raise NotFoundError(f"video speaker {video_speaker_id} does not exist")
    return _label_row(row)


def find_label(db: Database, video_id: int, local_label: str) -> LabelRow:
    """One local label by video and name. Raises :class:`NotFoundError`."""
    row = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.video_id = ? AND vs.local_label = ? GROUP BY vs.id",
        (video_id, local_label),
    ).fetchone()
    if row is None or row["id"] is None:
        raise NotFoundError(f"video {video_id} has no label {local_label!r}")
    return _label_row(row)


def link_video_speaker(
    db: Database, video_speaker_id: int, speaker_id: int | None
) -> None:
    """Say who a local label is — or, with ``None``, say that nobody knows.

    **One row.** This is the whole point of the two-level model: the words
    already point at the label, so deciding the person is a single UPDATE
    rather than a back-fill across thousands of rows.
    """
    get_label(db, video_speaker_id)
    if speaker_id is not None:
        get_speaker(db, speaker_id)
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE id = ?",
        (speaker_id, video_speaker_id),
    )


def set_embedding(db: Database, video_speaker_id: int, blob: bytes | None) -> None:
    """Store (or clear) the voice vector for one local label."""
    db.conn.execute(
        "UPDATE video_speakers SET embedding = ? WHERE id = ?", (blob, video_speaker_id)
    )


def clear_video_speakers(db: Database, video_id: int) -> int:
    """Drop every label of one video. `words.video_speaker_id` follows.

    `ON DELETE SET NULL` on `words.video_speaker_id` (contracts §3) does the
    second half, so there is no second statement to forget.
    """
    cursor = db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))
    return cursor.rowcount


def word_spans(db: Database, video_id: int) -> list[WordSpan]:
    """Every word of one video as ``(id, start_ms, end_ms)``, in ordinal order."""
    rows = db.conn.execute(
        "SELECT id, start_ms, end_ms FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    return [(int(r["id"]), int(r["start_ms"]), r["end_ms"]) for r in rows]


def stamp_words(db: Database, video_id: int, by_word_id: Mapping[int, int]) -> int:
    """Point each word at its local label; clear every word not in the map.

    This is the *only* place `words.video_speaker_id` is written, and it is
    written once per diarization run. Clearing the rest matters: a second
    run that finds fewer speakers must not leave words pointing at labels
    the new run did not produce.
    """
    with db.transaction():
        db.conn.execute(
            "UPDATE words SET video_speaker_id = NULL WHERE video_id = ?", (video_id,)
        )
        db.conn.executemany(
            "UPDATE words SET video_speaker_id = ? WHERE id = ?",
            [(label_id, word_id) for word_id, label_id in by_word_id.items()],
        )
    return len(by_word_id)


def diarized_videos(
    db: Database, *, only_unmapped: bool = False, limit: int = C.SPEAKER_LIST_LIMIT
) -> list[VideoRow]:
    """Videos that have local labels, newest first — the mapper's front door."""
    having = "HAVING SUM(vs.speaker_id IS NULL) > 0" if only_unmapped else ""
    rows = db.conn.execute(
        f"""
        SELECT v.id, v.title, v.published_at,
               COUNT(vs.id)               AS n_labels,
               SUM(vs.speaker_id IS NULL) AS n_unmapped
        FROM videos v
        JOIN video_speakers vs ON vs.video_id = v.id
        GROUP BY v.id
        {having}
        ORDER BY v.published_at DESC, v.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        VideoRow(
            video_id=int(row["id"]),
            title=row["title"],
            published_at=row["published_at"],
            n_labels=int(row["n_labels"]),
            n_unmapped=int(row["n_unmapped"]),
        )
        for row in rows
    ]


def linked_labels(db: Database, speaker_id: int) -> list[LabelRow]:
    """Every local label linked to one person, across every video."""
    rows = db.conn.execute(
        _LABEL_SELECT + " WHERE vs.speaker_id = ? GROUP BY vs.id ORDER BY vs.video_id",
        (speaker_id,),
    ).fetchall()
    return [_label_row(row) for row in rows]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_speakers_labels.py -v
```
Expected: 16 passed. If `test_linking_a_label_to_a_person_changes_exactly_one_row`
reports more than one change, something is back-filling `words` — find it and
delete it.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/store.py tests/test_speakers_labels.py
git commit -m "feat(speakers): per-video labels and one-row linking"
```

---

### Task 6: Alias matching and the headless mapping session

The mapper is the one genuinely interactive part of the product, and the
owner named it as the thing he most needs from the TUI. So none of it lives
in the TUI. `MappingSession` is the whole mapper — two panes, a cursor in
each, a filter that matches aliases, and four verbs — as an object with no
terminal anywhere near it. Task 14's screen draws this object and forwards
keystrokes to it, and holds no state of its own.

**Files:**
- Create: `rytp/diarize/mapper.py`
- Test: `tests/test_speakers_mapper.py`

**Interfaces:**
- Consumes: `rytp.diarize.store` in full.
- Produces:
  - `format_ms(ms: int) -> str`
  - `match_roster(rows: Sequence[RosterRow], query: str) -> list[RosterRow]`
  - `MappingSession(db, video_id)` with fields `labels`, `roster`, `matches`, `query`, `status`, `label_index`, `roster_index`, and methods `reload()`, `set_query(text)`, `select_label(i)`, `select_roster(i)`, `move_label(delta)`, `move_roster(delta)`, `selected_label`, `selected_roster`, `assign()`, `assign_to(ref)`, `unassign()`, `create_and_assign(label)`, `n_unmapped()`, `label_table()`, `roster_table()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_speakers_mapper.py`:

```python
"""The mapper, with no terminal in sight."""

from __future__ import annotations

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.mapper import MappingSession, format_ms, match_roster
from rytp.models import NotFoundError
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def roster(**kwargs: object) -> store.RosterRow:
    base: dict[str, object] = {
        "speaker_id": 1,
        "label": "Host One",
        "aliases": (),
        "notes": None,
        "n_labels": 0,
        "n_videos": 0,
    }
    base.update(kwargs)
    return store.RosterRow(**base)  # type: ignore[arg-type]


# -- pure matching ---------------------------------------------------------


def test_an_empty_query_matches_everything_in_order() -> None:
    rows = [roster(speaker_id=1, label="Guest Two"), roster(speaker_id=2, label="Host One")]
    assert match_roster(rows, "") == rows


def test_a_label_prefix_beats_an_alias_prefix() -> None:
    rows = [
        roster(speaker_id=1, label="Guest Two", aliases=("ho",)),
        roster(speaker_id=2, label="Host One"),
    ]
    assert [r.speaker_id for r in match_roster(rows, "ho")] == [2, 1]


def test_an_alias_prefix_beats_a_label_substring() -> None:
    rows = [
        roster(speaker_id=1, label="The Host One"),
        roster(speaker_id=2, label="Guest Two", aliases=("hostile",)),
    ]
    assert [r.speaker_id for r in match_roster(rows, "host")] == [2, 1]


def test_matching_ignores_case_on_both_sides() -> None:
    rows = [roster(speaker_id=1, label="Каспар Хаузер", aliases=("КХ",))]
    assert match_roster(rows, "кх") == rows
    assert match_roster(rows, "КАСПАР") == rows


def test_a_query_matching_nothing_returns_nothing() -> None:
    assert match_roster([roster()], "zzz") == []


def test_ties_are_broken_by_label_so_the_order_is_stable() -> None:
    rows = [roster(speaker_id=1, label="Host Two"), roster(speaker_id=2, label="Host One")]
    assert [r.label for r in match_roster(rows, "host")] == ["Host One", "Host Two"]


def test_format_ms_is_minutes_and_seconds() -> None:
    assert format_ms(0) == "0:00"
    assert format_ms(63_000) == "1:03"
    assert format_ms(3_723_000) == "62:03"


# -- the session -----------------------------------------------------------


@pytest.fixture
def session(db: Database) -> MappingSession:
    video_id = make_video(db, title="An Interview")
    word_ids = add_words(db, video_id, [(0, 400), (500, 900), (1_000, 1_400)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db, video_id, {word_ids[0]: host, word_ids[1]: host, word_ids[2]: guest}
    )
    store.add_speaker(db, "Host One", aliases=("h1",))
    store.add_speaker(db, "Guest Two", aliases=("g2",))
    return MappingSession(db, video_id)


def test_the_left_pane_is_the_videos_labels_with_their_weight(
    session: MappingSession,
) -> None:
    columns, rows = session.label_table()
    assert columns[0] == "label"
    assert [row[0] for row in rows] == ["SPEAKER_00", "SPEAKER_01"]
    assert rows[0][1] == "2"          # two words
    assert rows[0][3] == "—"          # nobody yet


def test_the_right_pane_is_the_roster(session: MappingSession) -> None:
    _columns, rows = session.roster_table()
    assert [row[0] for row in rows] == ["Guest Two", "Host One"]


def test_filtering_narrows_the_right_pane_only(session: MappingSession) -> None:
    session.set_query("h1")
    assert [row.label for row in session.matches] == ["Host One"]
    assert len(session.labels) == 2


def test_filtering_resets_the_roster_cursor(session: MappingSession) -> None:
    session.move_roster(1)
    session.set_query("h1")
    assert session.roster_index == 0
    assert session.selected_roster.label == "Host One"


def test_assign_links_the_selected_label_to_the_selected_person(
    session: MappingSession,
) -> None:
    session.set_query("Host One")
    message = session.assign()
    assert "SPEAKER_00" in message and "Host One" in message
    assert session.labels[0].speaker_label == "Host One"


def test_assign_advances_to_the_next_unmapped_label(session: MappingSession) -> None:
    session.set_query("Host One")
    session.assign()
    assert session.selected_label.local_label == "SPEAKER_01"


def test_assign_with_nothing_selected_says_so_and_writes_nothing(db: Database) -> None:
    empty = MappingSession(db, make_video(db))
    before = db.conn.total_changes
    assert "no label" in empty.assign().lower()
    assert db.conn.total_changes == before


def test_assign_to_accepts_an_alias(session: MappingSession) -> None:
    session.assign_to("g2")
    assert session.labels[0].speaker_label == "Guest Two"


def test_assign_to_an_unknown_name_raises(session: MappingSession) -> None:
    with pytest.raises(NotFoundError):
        session.assign_to("nobody at all")


def test_unassign_puts_a_label_back_to_nobody(session: MappingSession) -> None:
    session.assign_to("Host One")
    session.select_label(0)
    session.unassign()
    assert session.labels[0].speaker_id is None


def test_create_and_assign_adds_a_roster_entry_and_links_it(
    session: MappingSession,
) -> None:
    message = session.create_and_assign("Каспар Хаузер")
    assert "Каспар Хаузер" in message
    assert session.labels[0].speaker_label == "Каспар Хаузер"
    assert [row.label for row in session.roster] == [
        "Guest Two",
        "Host One",
        "Каспар Хаузер",
    ]


def test_create_and_assign_reuses_an_existing_label(session: MappingSession) -> None:
    session.create_and_assign("Host One")
    assert len(session.roster) == 2


def test_n_unmapped_counts_down_as_work_is_done(session: MappingSession) -> None:
    assert session.n_unmapped() == 2
    session.assign_to("Host One")
    assert session.n_unmapped() == 1


def test_the_cursors_never_leave_the_list(session: MappingSession) -> None:
    session.move_label(-5)
    assert session.label_index == 0
    session.move_label(99)
    assert session.label_index == 1
    session.move_roster(99)
    assert session.roster_index == 1


def test_a_session_over_a_video_with_no_labels_is_usable(db: Database) -> None:
    empty = MappingSession(db, make_video(db))
    assert empty.label_table()[1] == []
    assert empty.selected_label is None
    assert empty.n_unmapped() == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_speakers_mapper.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.mapper'`.

- [ ] **Step 3: Write `rytp/diarize/mapper.py`**

```python
"""The two-pane speaker mapper, with no terminal in it (design §10).

Design §10 calls speaker assignment "the one genuinely interactive task",
which is exactly why the interaction lives here rather than in a Textual
screen: a session is an object with two lists, two cursors, a filter and
four verbs, and every one of those can be asserted on without a TTY. The
screen in `rytp/tui/screens/speakers.py` draws this object and forwards
keystrokes; it holds no state of its own.

Both panes matter and they are not symmetrical. The left pane is this
video's diarizer labels — local, citable, possibly nameless. The right pane
is the global roster of real people. Assignment writes a single
`video_speakers.speaker_id`, and the session then re-reads, so what the
panes show is always what the database says rather than what the last
keystroke intended.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.store import LabelRow, RosterRow

if TYPE_CHECKING:
    from rytp.diarize.link import Suggestion

#: The dash shown where a column has no value. Two spellings of "unknown"
#: in one table would read as two different states.
_EMPTY = "—"


def format_ms(ms: int) -> str:
    """Milliseconds as ``m:ss`` — the mapper's only unit of time."""
    seconds = max(0, ms) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


def _rank(row: RosterRow, folded: str) -> int | None:
    """How well one roster entry matches a query: lower is better, None is no."""
    label = row.label.casefold()
    aliases = [alias.casefold() for alias in row.aliases]
    if label.startswith(folded):
        return 0
    if any(alias.startswith(folded) for alias in aliases):
        return 1
    if folded in label:
        return 2
    if any(folded in alias for alias in aliases):
        return 3
    return None


def match_roster(rows: Sequence[RosterRow], query: str) -> list[RosterRow]:
    """Rank the roster against a typed query, aliases included.

    Prefixes beat substrings and the label beats an alias, so typing the
    first letters of somebody's name puts them at the top even when an
    unrelated alias happens to contain those letters. Ties go to the
    alphabetically earlier label, which keeps the order stable between
    keystrokes — a list that reshuffles under the cursor is unusable.
    """
    folded = query.strip().casefold()
    if not folded:
        return list(rows)
    scored: list[tuple[int, str, RosterRow]] = []
    for row in rows:
        rank = _rank(row, folded)
        if rank is not None:
            scored.append((rank, row.label.casefold(), row))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _rank_, _label, row in scored]


@dataclass
class MappingSession:
    """Everything the mapper does, as state plus verbs."""

    db: Database
    video_id: int

    labels: list[LabelRow] = field(default_factory=list, init=False)
    roster: list[RosterRow] = field(default_factory=list, init=False)
    matches: list[RosterRow] = field(default_factory=list, init=False)
    query: str = field(default="", init=False)
    status: str = field(default="", init=False)
    label_index: int = field(default=0, init=False)
    roster_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.reload()

    # -- state -----------------------------------------------------------

    def reload(self) -> None:
        """Re-read both panes from the database and clamp the cursors."""
        self.labels = store.label_rows(self.db, self.video_id)
        self.roster = store.roster_rows(self.db)
        self.matches = match_roster(self.roster, self.query)
        self.label_index = _clamp(self.label_index, len(self.labels))
        self.roster_index = _clamp(self.roster_index, len(self.matches))

    def set_query(self, text: str) -> None:
        """Filter the right pane and put its cursor back at the top."""
        self.query = text
        self.matches = match_roster(self.roster, text)
        self.roster_index = 0

    def select_label(self, index: int) -> None:
        self.label_index = _clamp(index, len(self.labels))

    def select_roster(self, index: int) -> None:
        self.roster_index = _clamp(index, len(self.matches))

    def move_label(self, delta: int) -> None:
        self.select_label(self.label_index + delta)

    def move_roster(self, delta: int) -> None:
        self.select_roster(self.roster_index + delta)

    @property
    def selected_label(self) -> LabelRow | None:
        return self.labels[self.label_index] if self.labels else None

    @property
    def selected_roster(self) -> RosterRow | None:
        return self.matches[self.roster_index] if self.matches else None

    def n_unmapped(self) -> int:
        """How many local labels still belong to nobody."""
        return sum(1 for row in self.labels if row.speaker_id is None)

    # -- verbs -----------------------------------------------------------

    def assign(self) -> str:
        """Link the highlighted label to the highlighted person."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        person = self.selected_roster
        if person is None:
            return self._say("no speaker selected")
        return self._link(label, person.speaker_id, person.label)

    def assign_to(self, ref: str) -> str:
        """Link the highlighted label to a person named by id, label or alias."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        person = store.resolve_speaker(self.db, ref)
        return self._link(label, person.id, person.label)

    def unassign(self) -> str:
        """Put the highlighted label back to "a voice nobody named"."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        with self.db.transaction():
            store.link_video_speaker(self.db, label.video_speaker_id, None)
        self.reload()
        return self._say(f"{label.local_label} is nobody again")

    def create_and_assign(self, label_text: str) -> str:
        """Add a roster entry (or find it) and link the highlighted label to it."""
        label = self.selected_label
        if label is None:
            return self._say("no label selected")
        with self.db.transaction():
            speaker_id, _created = store.add_speaker(self.db, label_text)
            store.link_video_speaker(self.db, label.video_speaker_id, speaker_id)
        self._after_assign(label)
        return self._say(f"{label.local_label} is {label_text.strip()}")

    # -- rendering -------------------------------------------------------

    def label_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The left pane: this video's voices, heaviest information first."""
        rows = [
            (
                row.local_label,
                str(row.n_words),
                format_ms(row.speech_ms),
                row.speaker_label or _EMPTY,
                "yes" if row.has_embedding else "no",
            )
            for row in self.labels
        ]
        return ("label", "words", "speech", "person", "voice"), rows

    def roster_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The right pane: the roster, filtered by the current query."""
        rows = [
            (row.label, ", ".join(row.aliases) or _EMPTY, str(row.n_videos))
            for row in self.matches
        ]
        return ("person", "aliases", "videos"), rows

    # -- internals -------------------------------------------------------

    def _link(self, label: LabelRow, speaker_id: int, speaker_label: str) -> str:
        with self.db.transaction():
            store.link_video_speaker(self.db, label.video_speaker_id, speaker_id)
        self._after_assign(label)
        return self._say(f"{label.local_label} is {speaker_label}")

    def _after_assign(self, label: LabelRow) -> None:
        """Re-read, then jump to the next label that still needs a person.

        Moving the cursor on is what makes the keyboard flow work: two to
        five labels per video, assigned one after another, without reaching
        for the arrow keys in between.
        """
        self.reload()
        for index, row in enumerate(self.labels):
            if row.speaker_id is None and row.video_speaker_id != label.video_speaker_id:
                self.label_index = index
                return

    def _say(self, message: str) -> str:
        self.status = message
        return message


def _clamp(index: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(index, length - 1))
```

`Suggestion` is imported under `TYPE_CHECKING` for Task 12, which appends two
methods to this class.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_speakers_mapper.py -v
```
Expected: 22 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/mapper.py tests/test_speakers_mapper.py
git commit -m "feat(speakers): the two-pane mapper as a headless session"
```

---

### Task 7: The diarize pipeline

One function: run a diarizer over a video's cached WAV, write one
`video_speakers` row per label it found — each stamped with the name of the
diarizer that produced it — stamp every word with the label it overlaps
most, and then tell the index to rebuild.

That last part is easy to skip and important. Design §4 splits utterances
"on speaker change where speakers are known", and design §7's speaker
filter searches `utterances`, not `words`. A video whose words are stamped
but whose utterances were built before diarization is a video you cannot
search by speaker — half the owner's goal, silently missing. So this stage
deletes the video's utterances and enqueues its `index` job, which Part 4
re-derives from the now-labelled words.

**Files:**
- Create: `rytp/diarize/pipeline.py`
- Test: `tests/test_diarize_pipeline.py`

**Interfaces:**
- Consumes: `rytp.diarize.base.{DiarizeError, Diarizer}`, `rytp.diarize.segments.{label_words, local_labels}`, `rytp.diarize.store` in full, `rytp.jobs.JOB_KINDS`, `rytp.jobs.queue.enqueue`.
- Produces: `DiarizeOutcome(video_id, diarizer, n_labels, n_segments, n_words_labelled, n_words_unlabelled, reindexed)` with `.message()`, `diarize_video(db, video_id, *, wav_path, diarizer) -> DiarizeOutcome`, `request_reindex(db, video_id) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_pipeline.py`:

```python
"""Diarization end to end, with a fake engine and no audio decoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import DiarizeError
from rytp.diarize.pipeline import diarize_video
from rytp.jobs import Readiness
from rytp.models import DiarSegment
from tests.fake_speaker_engines import FakeDiarizer
from tests.fakes import make_video, temp_job_kind
from tests.test_speakers_labels import add_words

WAV = Path("does-not-need-to-exist.wav")


def a_video_with_words(db: Database) -> int:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    return video_id


def test_one_row_per_label_and_every_word_stamped(db: Database) -> None:
    video_id = a_video_with_words(db)
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert outcome.n_labels == 2
    assert outcome.n_words_labelled == 3
    assert [row.local_label for row in store.label_rows(db, video_id)] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert [row.n_words for row in store.label_rows(db, video_id)] == [2, 1]


def test_a_word_outside_every_segment_is_counted_not_dropped(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 500), (60_000, 60_500)])
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert (outcome.n_words_labelled, outcome.n_words_unlabelled) == (1, 1)


def test_a_diarizer_that_finds_nothing_is_an_error_not_an_empty_success(
    db: Database,
) -> None:
    video_id = a_video_with_words(db)
    with pytest.raises(DiarizeError, match="no speech"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer(script=()))


def test_diarizing_a_video_with_no_words_is_an_error(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(DiarizeError, match="no words"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())


def test_rediarizing_keeps_a_link_whose_label_came_back(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    speaker_id, _ = store.add_speaker(db, "Host One")
    store.link_video_speaker(
        db, store.find_label(db, video_id, "SPEAKER_00").video_speaker_id, speaker_id
    )
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"


def test_rediarizing_drops_a_label_the_new_run_did_not_produce(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    single = FakeDiarizer(script=((0, 3_000, "SPEAKER_00"),))
    diarize_video(db, video_id, wav_path=WAV, diarizer=single)
    assert [row.local_label for row in store.label_rows(db, video_id)] == ["SPEAKER_00"]


def test_diarizing_deletes_the_videos_utterances(db: Database) -> None:
    video_id = a_video_with_words(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 900, 0, 0, 'x', 'x', 'x')",
        (video_id,),
    )
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        == 0
    )


def test_diarizing_enqueues_the_index_job_when_that_kind_exists(db: Database) -> None:
    from rytp.jobs import queue as Q

    video_id = a_video_with_words(db)
    with temp_job_kind(
        "index", "cpu", lambda db_, t, p: None, readiness=lambda db_, t: Readiness.READY
    ):
        outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
        assert [job.kind for job in Q.list_jobs(db)] == ["index"]
    assert outcome.reindexed is True


def test_diarizing_without_an_index_kind_still_succeeds(db: Database) -> None:
    video_id = a_video_with_words(db)
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert outcome.reindexed is False


def test_a_diarizer_emitting_a_backwards_segment_is_refused(db: Database) -> None:
    video_id = a_video_with_words(db)
    bad = FakeDiarizer(script=((900, 100, "SPEAKER_00"),))
    with pytest.raises(DiarizeError, match="start"):
        diarize_video(db, video_id, wav_path=WAV, diarizer=bad)


def test_the_outcome_names_the_engine(db: Database) -> None:
    video_id = a_video_with_words(db)
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert outcome.diarizer == "fake-diarizer"
    assert "fake-diarizer" in outcome.message()


def test_every_label_records_the_diarizer_that_produced_it(db: Database) -> None:
    # contracts §3: video_speakers.engine is NOT NULL, for the same reason
    # words.engine is — design §11 forbids assuming one engine.
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())
    assert {row.engine for row in store.label_rows(db, video_id)} == {"fake-diarizer"}


def test_the_null_diarizer_records_itself_too(db: Database, tmp_path: Path) -> None:
    # Not an exception: "one voice, unmodelled" is a claim about the audio
    # like any other, and a later re-run with pyannote must be tellable from
    # it without guessing.
    import wave

    from rytp.diarize.none import NullDiarizer

    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16_000 * 5)
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=audio, diarizer=NullDiarizer())
    assert [row.engine for row in store.label_rows(db, video_id)] == ["none"]


def test_re_diarizing_with_another_engine_updates_the_provenance(db: Database) -> None:
    video_id = a_video_with_words(db)
    diarize_video(db, video_id, wav_path=WAV, diarizer=FakeDiarizer())

    class OtherDiarizer(FakeDiarizer):
        name = "fake-other"

    diarize_video(db, video_id, wav_path=WAV, diarizer=OtherDiarizer())
    assert {row.engine for row in store.label_rows(db, video_id)} == {"fake-other"}


def test_segments_are_accepted_in_any_order(db: Database) -> None:
    video_id = a_video_with_words(db)
    shuffled = FakeDiarizer(
        script=(
            (2_000, 3_000, "SPEAKER_00"),
            (0, 1_000, "SPEAKER_00"),
            (1_000, 2_000, "SPEAKER_01"),
        )
    )
    outcome = diarize_video(db, video_id, wav_path=WAV, diarizer=shuffled)
    assert outcome.n_words_labelled == 3


def test_the_contract_type_field_is_local_label() -> None:
    # A regression guard: the field is `local_label`, not the old `speaker`.
    assert DiarSegment(start_ms=0, end_ms=1, local_label="SPEAKER_00").local_label
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_pipeline.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.pipeline'`.

- [ ] **Step 3: Write `rytp/diarize/pipeline.py`**

```python
"""Diarize one video: labels in the database, labels on the words (design §6).

Two things happen here and both are deliberate.

**`words.video_speaker_id` is written.** Once, at diarize time, from
overlap. Everything downstream — the mapper's left pane, the search
filter, Part 6's per-speaker pause measurement, utterance splitting — reads
that column. This is the *only* stage that writes it, and it is not the same
thing as linking a label to a person: that happens later and touches one
row of `video_speakers`, never the words.

**The utterances are rebuilt.** Design §4 splits utterances on speaker
change, and design §7's speaker filter searches utterances rather than
words. Stamping the words without re-deriving the utterances would leave
the video searchable by word and unsearchable by speaker, which is exactly
the gap this part exists to close. So the video's utterances are deleted
and its `index` job enqueued; Part 4 rebuilds them from the labelled words.
The deletion is unconditional (design §3: "erase and replace, don't
reconcile"); the enqueue happens only when the `index` kind is registered,
so Part 7 can be built and tested before Part 4 exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import DiarizeError, Diarizer, load_diarizer
from rytp.diarize.segments import label_words, local_labels
from rytp.models import DiarSegment, RytpError


@dataclass(frozen=True)
class DiarizeOutcome:
    """What one diarization run did, for the command line and the job log."""

    video_id: int
    diarizer: str
    n_labels: int
    n_segments: int
    n_words_labelled: int
    n_words_unlabelled: int
    reindexed: bool

    def message(self) -> str:
        tail = "" if self.reindexed else " (no index job kind registered)"
        return (
            f"video {self.video_id}: {self.n_labels} speakers from "
            f"{self.n_segments} segments via {self.diarizer}; "
            f"{self.n_words_labelled} words labelled, "
            f"{self.n_words_unlabelled} left unlabelled{tail}"
        )


def _validated(segments: list[DiarSegment]) -> list[DiarSegment]:
    for segment in segments:
        if segment.end_ms <= segment.start_ms:
            raise DiarizeError(
                f"diarizer returned a segment whose end ({segment.end_ms} ms) is not "
                f"after its start ({segment.start_ms} ms)"
            )
        if not segment.local_label:
            raise DiarizeError("diarizer returned a segment with an empty label")
    return segments


def request_reindex(db: Database, video_id: int) -> bool:
    """Drop the video's utterances and ask for them to be rebuilt.

    Returns whether an `index` job was enqueued. Part 4 owns that kind; the
    imports are inside the function so Part 7 depends on neither Part 4's
    import order nor the queue being in the diarize package's import path.
    """
    db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    from rytp.jobs import JOB_KINDS

    if "index" not in JOB_KINDS:
        return False
    from rytp.jobs.queue import enqueue

    enqueue(db, "index", video_id)
    return True


def diarize_video(
    db: Database, video_id: int, *, wav_path: Path, diarizer: Diarizer
) -> DiarizeOutcome:
    """Run a diarizer over one video and record who spoke when."""
    words = store.word_spans(db, video_id)
    if not words:
        raise DiarizeError(f"video {video_id} has no words to label; transcribe it first")

    segments = _validated(list(diarizer.diarize(wav_path)))
    if not segments:
        raise DiarizeError(f"diarizer {diarizer.name!r} found no speech in {wav_path}")

    labels = local_labels(segments)
    by_word_label = label_words(words, segments)

    with db.transaction():
        # Labels first: every id the stamp needs must already exist.
        label_ids = {
            label: store.upsert_video_speaker(db, video_id, label, engine=diarizer.name)
            for label in labels
        }
        # A label from a previous run that this run did not produce goes,
        # and its words are released by ON DELETE SET NULL.
        for existing in store.label_rows(db, video_id):
            if existing.local_label not in label_ids:
                db.conn.execute(
                    "DELETE FROM video_speakers WHERE id = ?",
                    (existing.video_speaker_id,),
                )
        store.stamp_words(
            db,
            video_id,
            {word_id: label_ids[label] for word_id, label in by_word_label.items()},
        )
        reindexed = request_reindex(db, video_id)

    return DiarizeOutcome(
        video_id=video_id,
        diarizer=diarizer.name,
        n_labels=len(labels),
        n_segments=len(segments),
        n_words_labelled=len(by_word_label),
        n_words_unlabelled=len(words) - len(by_word_label),
        reindexed=reindexed,
    )
```

**Import block:** write only what this task uses — `dataclass`, `Path`,
`Database`, `store`, `DiarizeError`, `Diarizer`, `label_words`,
`local_labels`, `DiarSegment`. Task 8 adds `rytp.constants as C`,
`rytp.config.paths`, `load_diarizer` and `RytpError`; Task 10 adds the
`TYPE_CHECKING` import of `SpeakerEmbedder`. Adding them early would leave
`ruff` failing on unused imports at this commit.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_pipeline.py -v
```
Expected: 16 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/pipeline.py tests/test_diarize_pipeline.py
git commit -m "feat(diarize): label a video's words and ask for its index to be rebuilt"
```

---

### Task 8: The diarize job kind, opt-in by construction

At full corpus, diarization is the single largest GPU cost — larger than
transcription (design §6, §13) — and speakers are only needed for videos
being mined. So `diarize` must never happen by accident, and there are
exactly two ways it could:

- **A default ingest enqueuing it.** It cannot: Part 2's `ingest` enqueues
  `download`, `captions` and `extract_wav` and nothing else, and a test
  below pins that.
- **A `reconcile` sweep re-deriving it.** It would, by default:
  `reconcile` turns a `done` job back into work when its output is gone,
  and deleting a video's words (a re-transcribe) deletes its
  `video_speakers` rows, so the readiness predicate would answer READY
  again — quietly re-queuing hours of GPU time across the corpus. That is
  what `reopenable=False` is for, and it is the single most important line
  in this task.

**Files:**
- Create: `rytp/diarize/readiness.py`
- Modify: `rytp/diarize/pipeline.py` (append the job entry point)
- Modify: `rytp/jobs/__init__.py` (one thunk, one predicate import, one registration — Part 2's recipe)
- Test: `tests/test_diarize_job.py`

**Interfaces:**
- Consumes: `rytp.jobs.{Readiness, JobKind, register_job_kind}`, `rytp.config.paths`, `rytp.transcribe.registry.setting`, `rytp.diarize.base.load_diarizer`.
- Produces:
  - `rytp.diarize.readiness.diarize_readiness(db, video_id) -> Readiness`
  - `rytp.diarize.pipeline.wav_for(db, video_id) -> Path`
  - `rytp.diarize.pipeline.diarizer_name(db, requested) -> str`
  - `rytp.diarize.pipeline.run_diarize_job(db, video_id, payload) -> None`
  - The registered job kind `diarize`: pool `gpu`, `target_kind="video"`, **`reopenable=False`**.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_job.py`:

```python
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
    resolve("ingest").handler(db, video_id=video_id, priority=0)
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_job.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.readiness'`.

- [ ] **Step 3: Write `rytp/diarize/readiness.py`**

Part 2's recipe requires this to be a **light** module: `rytp/jobs/__init__.py`
imports it at load time, so it may import `pathlib`, `rytp.config`,
`rytp.constants`, `rytp.db` and `rytp.jobs` and nothing heavier. In
particular it must not import `rytp.diarize.base` — that would pull the engine
registry into every `rytp jobs` invocation.

```python
"""When a video is ready to be diarized (design §5).

A predicate over the database and the filesystem as they are *now*. It
never sees the job's payload, so which engine somebody asked for cannot
change the answer — a predicate that did would stop being a statement about
the world and the self-healing property would go with it.

Deliberately light: `rytp/jobs/__init__.py` imports this at module load, so
nothing from the rest of `rytp.diarize` may be imported here.
"""

from __future__ import annotations

from rytp.config import paths
from rytp.db import Database
from rytp.jobs import Readiness


def diarize_readiness(db: Database, video_id: int) -> Readiness:
    """READY once there are aligned words and a cached WAV to measure them against.

    Aligned words, not any words: caption-tier rows have no end times
    (contracts §3), so their overlap with a diarizer segment would be
    guesswork, and a caption-tier video is not one anybody is cutting from.

    SATISFIED as soon as the video has labels. Re-transcribing deletes them
    (contracts §4), which puts this back to READY — but the kind is
    registered `reopenable=False`, so that only matters when a human asks
    for diarization again.
    """
    exists = db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone()
    if exists is None:
        return Readiness.BLOCKED
    labelled = db.conn.execute(
        "SELECT 1 FROM video_speakers WHERE video_id = ? LIMIT 1", (video_id,)
    ).fetchone()
    if labelled is not None:
        return Readiness.SATISFIED
    aligned = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source = 'aligned' LIMIT 1",
        (video_id,),
    ).fetchone()
    if aligned is None:
        return Readiness.BLOCKED
    if not paths().cache_wav(video_id).exists():
        return Readiness.BLOCKED
    return Readiness.READY
```

- [ ] **Step 4: Append the job entry point to `rytp/diarize/pipeline.py`**

```python
def wav_for(db: Database, video_id: int) -> Path:
    """The cached WAV this video is diarized from (contracts §7).

    The cache is regenerable and prunable, so a missing file is a normal
    state rather than corruption — say which command puts it back.
    """
    path = paths().cache_wav(video_id)
    if not path.exists():
        raise RytpError(
            f"no cached audio for video {video_id} at {path}; "
            f"run the extract_wav job, or `rytp ingest {video_id}` and let the "
            f"worker decode it"
        )
    return path


def diarizer_name(db: Database, requested: str | None) -> str:
    """What to diarize with: the request, then the setting, then the default."""
    if requested:
        return requested
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_DIARIZER) or C.DEFAULT_DIARIZER


def run_diarize_job(db: Database, video_id: int, payload: dict[str, object]) -> None:
    """The `diarize` job kind's body (contracts §5).

    Reached from two directions — `rytp speakers diarize` by hand and the
    worker draining the queue — which is why the orchestration lives here
    rather than in either caller.
    """
    name = diarizer_name(db, str(payload.get("diarizer") or ""))
    diarize_video(
        db, video_id, wav_path=wav_for(db, video_id), diarizer=load_diarizer(db, name)
    )
```

Task 10 extends `run_diarize_job` with the optional embedding pass. Note that
`load_diarizer` is referenced through the module namespace at call time, which
is what lets the test monkeypatch `pipeline.load_diarizer`.

- [ ] **Step 5: Register the kind in `rytp/jobs/__init__.py`**

Part 2 owns this file. Add the thunk next to Part 2's other thunks:

```python
def _run_diarize(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.diarize.pipeline import run_diarize_job

    run_diarize_job(db, video_id, payload)
```

Add the predicate to the bottom import block, alongside Part 2's:

```python
from rytp.diarize.readiness import diarize_readiness  # noqa: E402
```

And add the registration at the end of the registration block:

```python
register_job_kind(
    JobKind(
        name="diarize",
        pool="gpu",
        readiness=diarize_readiness,
        handler=_run_diarize,
        summary="work out who spoke when, and label this video's words",
        # design §6: diarization is opt-in per video and the largest GPU
        # cost in the project. A reconcile sweep must never re-derive it.
        reopenable=False,
    )
)
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_job.py -v
```
Expected: 14 passed.

- [ ] **Step 7: Run the whole suite — a new job kind changes what `jobs stats` lists**

Run:
```bash
python -m pytest -q
```
Expected: everything passes. If a Part 2 test asserts the exact set of
registered kinds, it now needs `diarize` in it — fix the assertion, not the
registration.

- [ ] **Step 8: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 9: Commit**

```bash
git add rytp/diarize/readiness.py rytp/diarize/pipeline.py rytp/jobs/__init__.py \
  tests/test_diarize_job.py
git commit -m "feat(diarize): register the diarize job kind, opt-in and non-reopenable"
```

---

### Task 9: Embedding vectors — codec, similarity, centroids

A speaker embedding is a numeric fingerprint of a voice (design §2). This
task is the arithmetic and the storage format, with no model anywhere near
it: vectors are lists of floats, the column is a little-endian float32
BLOB, and every test writes its vectors by hand.

This replaces the placeholder `rytp/diarize/embed.py` from Task 1, Step 3.

**Files:**
- Modify: `rytp/diarize/embed.py` (replace the placeholder entirely)
- Test: `tests/test_diarize_embed.py`

**Interfaces:**
- Consumes: `rytp.constants.EMBEDDING_BYTES_PER_VALUE`, `rytp.models.RytpError`, `rytp.transcribe.registry.{check_available, interpreter_for, availability}`.
- Produces:
  - `SpeakerEmbedder` (Protocol), `EmbeddingError(RytpError)`
  - `EMBEDDERS`, `register_embedder`, `resolve_embedder`, `load_embedder(db, name, **kw)`, `embedder_rows(db)`
  - `pack_embedding(values) -> bytes`, `unpack_embedding(blob) -> list[float]`
  - `l2_normalize(values) -> list[float]`, `cosine_similarity(a, b) -> float`, `comparable(a, b) -> bool`, `centroid(vectors) -> list[float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_embed.py`:

```python
"""Voice vectors: a storage format and four pieces of arithmetic."""

from __future__ import annotations

import math

import pytest

from rytp.db import Database
from rytp.diarize.embed import (
    EMBEDDERS,
    EmbeddingError,
    centroid,
    comparable,
    cosine_similarity,
    l2_normalize,
    load_embedder,
    pack_embedding,
    resolve_embedder,
    unpack_embedding,
)
from tests.fake_speaker_engines import FakeEmbedder, registered


# -- the codec -------------------------------------------------------------


def test_a_vector_round_trips_through_the_blob() -> None:
    values = [0.5, -0.25, 0.125, 0.0]
    assert unpack_embedding(pack_embedding(values)) == values


def test_the_blob_is_four_bytes_per_value_and_little_endian() -> None:
    blob = pack_embedding([1.0, 2.0])
    assert len(blob) == 8
    assert blob[:4] == b"\x00\x00\x80\x3f"


def test_float64_precision_is_lost_but_the_value_survives() -> None:
    [restored] = unpack_embedding(pack_embedding([0.1]))
    assert restored == pytest.approx(0.1, abs=1e-7)


def test_an_empty_vector_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        pack_embedding([])


def test_a_truncated_blob_is_refused_rather_than_silently_shortened() -> None:
    with pytest.raises(EmbeddingError):
        unpack_embedding(b"\x00\x00\x80")


# -- the arithmetic --------------------------------------------------------


def test_normalising_gives_a_unit_vector() -> None:
    values = l2_normalize([3.0, 4.0])
    assert values == pytest.approx([0.6, 0.8])
    assert math.isclose(sum(v * v for v in values), 1.0)


def test_normalising_a_zero_vector_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        l2_normalize([0.0, 0.0])


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-2.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude() -> None:
    assert cosine_similarity([1.0, 1.0], [7.0, 7.0]) == pytest.approx(1.0)


def test_cosine_refuses_vectors_of_different_lengths() -> None:
    # Two embedders in one corpus produce two dimensionalities. Comparing
    # them is meaningless and must say so rather than fail somewhere else.
    with pytest.raises(EmbeddingError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_comparable_is_the_question_to_ask_before_comparing() -> None:
    assert comparable([1.0, 0.0], [0.0, 1.0]) is True
    assert comparable([1.0, 0.0], [1.0, 0.0, 0.0]) is False
    assert comparable([], []) is False


def test_a_centroid_is_the_normalised_mean_of_normalised_vectors() -> None:
    assert centroid([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx([2 ** -0.5, 2 ** -0.5])


def test_a_centroid_of_one_vector_is_that_vector_normalised() -> None:
    assert centroid([[3.0, 4.0]]) == pytest.approx([0.6, 0.8])


def test_a_centroid_of_nothing_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        centroid([])


def test_a_centroid_of_mismatched_vectors_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        centroid([[1.0, 0.0], [1.0, 0.0, 0.0]])


# -- the registry ----------------------------------------------------------


def test_register_and_resolve_returns_the_class(db: Database) -> None:
    with registered(FakeEmbedder):
        assert resolve_embedder("fake-embedder") is FakeEmbedder
        assert isinstance(load_embedder(db, "fake-embedder"), FakeEmbedder)


def test_resolving_an_unknown_embedder_names_the_available_ones() -> None:
    with registered(FakeEmbedder), pytest.raises(ValueError) as excinfo:
        resolve_embedder("nope")
    assert "fake-embedder" in str(excinfo.value)


def test_the_registry_holds_classes_not_instances() -> None:
    with registered(FakeEmbedder):
        assert EMBEDDERS["fake-embedder"] is FakeEmbedder
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_embed.py -v
```
Expected: `ImportError: cannot import name 'EmbeddingError' from 'rytp.diarize.embed'`.

- [ ] **Step 3: Replace `rytp/diarize/embed.py`**

```python
"""Speaker embeddings: the protocol, the registry, the codec and the maths.

A speaker embedding is a numeric fingerprint of a voice (design §2), used
to judge whether two recordings are the same person. Everything in this
module except the adapter at the bottom is plain arithmetic on lists of
floats — no numpy, no torch — for two reasons: the vectors are a couple of
hundred values and the roster is tens of people, so vectorising buys
nothing; and this module is imported inside a *foreign* interpreter by the
out-of-process seam, where only the standard library is guaranteed.

Storage is `video_speakers.embedding`, a little-endian float32 BLOB.
float32 is half the size of float64 and further below the noise floor of
any speaker model than a cosine comparison can notice.

Vectors are stored **already L2-normalised**, so a cosine similarity is a
dot product and a centroid is a mean. Nothing outside this module needs to
remember that.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rytp import constants as C
from rytp.models import RytpError
from rytp.transcribe.registry import check_available, interpreter_for

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = [
    "EMBEDDERS",
    "EmbeddingError",
    "SpeakerEmbedder",
    "centroid",
    "comparable",
    "cosine_similarity",
    "embedder_rows",
    "l2_normalize",
    "load_embedder",
    "pack_embedding",
    "register_embedder",
    "resolve_embedder",
    "unpack_embedding",
]


class EmbeddingError(RytpError):
    """A vector is empty, malformed, or being compared with an incompatible one."""


class SpeakerEmbedder(Protocol):
    """Audio plus the windows to listen to, one vector out."""

    name: str
    requires_hf_token: bool
    out_of_process: bool
    dim: int

    def embed(
        self, audio: Path, windows: Sequence[tuple[int, int]]
    ) -> Sequence[float]: ...


EMBEDDERS: dict[str, type[SpeakerEmbedder]] = {}
"""Registered embedder classes, by ``cls.name``."""


def register_embedder(cls: type[SpeakerEmbedder]) -> type[SpeakerEmbedder]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    EMBEDDERS[cls.name] = cls
    return cls


def resolve_embedder(name: str) -> type[SpeakerEmbedder]:
    """Look up an embedder class. Raises ``ValueError`` naming the alternatives."""
    try:
        return EMBEDDERS[name]
    except KeyError:
        available = ", ".join(sorted(EMBEDDERS)) or "(none registered)"
        raise ValueError(f"unknown embedder {name!r}; available: {available}") from None


def load_embedder(db: Database, name: str, **kwargs: Any) -> SpeakerEmbedder:
    """Resolve, gate, and construct an embedder."""
    cls = resolve_embedder(name)
    check_available(cls)
    if getattr(cls, "out_of_process", False):
        kwargs.setdefault("interpreter", interpreter_for(db, name))
    return cls(**kwargs)  # type: ignore[call-arg]


def embedder_rows(db: Database) -> list[tuple[str, str, str]]:
    """Rows for the engine listing: name, out-of-process, availability."""
    from rytp.transcribe.registry import availability

    return [
        (
            name,
            "yes" if getattr(cls, "out_of_process", False) else "no",
            availability(db, cls),
        )
        for name, cls in sorted(EMBEDDERS.items())
    ]


# -- codec -----------------------------------------------------------------


def pack_embedding(values: Sequence[float]) -> bytes:
    """A vector as the little-endian float32 BLOB the column stores."""
    if not values:
        raise EmbeddingError("refusing to store an empty embedding")
    return struct.pack(f"<{len(values)}f", *(float(v) for v in values))


def unpack_embedding(blob: bytes) -> list[float]:
    """A stored BLOB back into a vector.

    A blob whose length is not a whole number of floats is corruption, not
    a shorter vector — say so rather than truncating silently.
    """
    if not blob:
        raise EmbeddingError("refusing to read an empty embedding")
    per_value = C.EMBEDDING_BYTES_PER_VALUE
    if len(blob) % per_value:
        raise EmbeddingError(
            f"embedding blob is {len(blob)} bytes, not a multiple of {per_value}"
        )
    return list(struct.unpack(f"<{len(blob) // per_value}f", blob))


# -- arithmetic ------------------------------------------------------------


def l2_normalize(values: Sequence[float]) -> list[float]:
    """Scale a vector to unit length."""
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm == 0.0:
        raise EmbeddingError("cannot normalise a zero-length embedding")
    return [float(v) / norm for v in values]


def comparable(left: Sequence[float], right: Sequence[float]) -> bool:
    """Whether two vectors can be compared at all.

    Two embedders in one corpus produce two dimensionalities. Asking this
    first is how the suggestion pass skips such a pair instead of dying on
    it halfway through a listing.
    """
    return bool(left) and len(left) == len(right)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1 identical, -1 opposite."""
    if not comparable(left, right):
        raise EmbeddingError(
            f"cannot compare embeddings of length {len(left)} and {len(right)}"
        )
    unit_left = l2_normalize(left)
    unit_right = l2_normalize(right)
    return sum(a * b for a, b in zip(unit_left, unit_right, strict=True))


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """The average direction of several vectors, as a unit vector.

    This is the enrolment centroid of design §6: several recordings of one
    person, reduced to one point to compare a new voice against.
    """
    if not vectors:
        raise EmbeddingError("cannot take the centroid of no embeddings")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise EmbeddingError("cannot take the centroid of embeddings of different lengths")
    units = [l2_normalize(vector) for vector in vectors]
    mean = [sum(unit[i] for unit in units) / len(units) for i in range(width)]
    return l2_normalize(mean)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_embed.py -v
```
Expected: 19 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/diarize/embed.py tests/test_diarize_embed.py
git commit -m "feat(diarize): voice-vector codec, cosine similarity and enrolment centroids"
```

---

### Task 10: The ReDimNet embedder, out of process

Two things here, and the second is the interesting one.

The adapter is straightforward: ReDimNet B6 is substantially better than
pyannote's internal embedding model (design §6), it needs torch, and
torch's pins are the reason engines run out of process — so it declares
`out_of_process = True` and follows Part 3's seam exactly.

The stage around it does not re-run the diarizer. Diarizer segments are not
stored anywhere (the schema has no table for them), but the *words* are,
and every word carries the label that spoke it. So a label's speech can be
reconstructed from consecutive words — which makes re-embedding a cheap,
standalone operation rather than a reason to pay the corpus's largest GPU
cost again.

**Files:**
- Modify: `rytp/diarize/embed.py` (append the audio helper and the adapter)
- Modify: `rytp/diarize/store.py` (append `label_segments`)
- Modify: `rytp/diarize/pipeline.py` (append `embed_video_speakers`, extend `run_diarize_job`)
- Modify: `pyproject.toml`
- Test: `tests/test_diarize_embed_stage.py`

**Interfaces:**
- Consumes: `rytp.transcribe.subproc.run_child`, `rytp.diarize.segments.{speech_ms_by_label, windows_for_label}`.
- Produces:
  - `rytp.diarize.embed.ReDimNetEmbedder` (name `redimnet`), `concat_wav_windows(src, dst, windows) -> Path`, `child_main(request) -> dict`
  - `rytp.diarize.store.label_segments(db, video_id) -> list[DiarSegment]`
  - `rytp.diarize.pipeline.EmbedOutcome(video_id, embedder, n_embedded, n_skipped)` with `.message()`, `embed_video_speakers(db, video_id, *, wav_path, embedder) -> EmbedOutcome`, `embedder_name(db, requested) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_embed_stage.py`:

```python
"""Reconstructing a label's speech from its words, and embedding it."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import (
    EMBEDDERS,
    EmbeddingError,
    concat_wav_windows,
    unpack_embedding,
)
from rytp.diarize.pipeline import embed_video_speakers, embedder_name
from rytp.models import DiarSegment
from tests.fake_speaker_engines import ConstantEmbedder, FakeEmbedder
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def a_wav(path: Path, ms: int) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db)
    word_ids = add_words(
        db, video_id, [(0, 2_000), (2_000, 4_000), (10_000, 12_000), (12_000, 14_000)]
    )
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db,
        video_id,
        {word_ids[0]: host, word_ids[1]: host, word_ids[2]: guest, word_ids[3]: guest},
    )
    return video_id


# -- reconstructing segments from words ------------------------------------


def test_consecutive_words_of_one_label_become_one_segment(db: Database) -> None:
    video_id = a_diarized_video(db)
    assert store.label_segments(db, video_id) == [
        DiarSegment(start_ms=0, end_ms=4_000, local_label="SPEAKER_00"),
        DiarSegment(start_ms=10_000, end_ms=14_000, local_label="SPEAKER_01"),
    ]


def test_a_label_that_speaks_twice_gets_two_segments(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(
        db, video_id, {word_ids[0]: host, word_ids[1]: guest, word_ids[2]: host}
    )
    assert [s.local_label for s in store.label_segments(db, video_id)] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]


def test_unlabelled_words_break_a_run_and_contribute_nothing(db: Database) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[2]: host})
    assert store.label_segments(db, video_id) == [
        DiarSegment(start_ms=0, end_ms=1_000, local_label="SPEAKER_00"),
        DiarSegment(start_ms=4_000, end_ms=5_000, local_label="SPEAKER_00"),
    ]


# -- the stage -------------------------------------------------------------


def test_every_label_with_enough_speech_gets_a_vector(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    outcome = embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 20_000), embedder=FakeEmbedder()
    )
    assert (outcome.n_embedded, outcome.n_skipped) == (2, 0)
    assert all(row.has_embedding for row in store.label_rows(db, video_id))


def test_a_label_below_the_speech_floor_is_skipped_not_guessed(
    db: Database, tmp_path: Path
) -> None:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 500)])
    label_id = add_label(db, video_id, "SPEAKER_00")
    store.stamp_words(db, video_id, {word_ids[0]: label_id})
    outcome = embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 2_000), embedder=FakeEmbedder()
    )
    assert (outcome.n_embedded, outcome.n_skipped) == (0, 1)
    assert store.get_label(db, label_id).has_embedding is False


def test_the_stored_vector_is_unit_length(db: Database, tmp_path: Path) -> None:
    video_id = a_diarized_video(db)
    embed_video_speakers(
        db, video_id, wav_path=a_wav(tmp_path / "a.wav", 20_000), embedder=FakeEmbedder()
    )
    blob = db.conn.execute(
        "SELECT embedding FROM video_speakers ORDER BY id LIMIT 1"
    ).fetchone()[0]
    values = unpack_embedding(bytes(blob))
    assert sum(v * v for v in values) == pytest.approx(1.0)


def test_two_labels_with_the_same_voice_get_the_same_vector(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    embed_video_speakers(
        db,
        video_id,
        wav_path=a_wav(tmp_path / "a.wav", 20_000),
        embedder=ConstantEmbedder(),
    )
    blobs = {
        bytes(row[0])
        for row in db.conn.execute("SELECT embedding FROM video_speakers").fetchall()
    }
    assert len(blobs) == 1


def test_re_embedding_replaces_rather_than_accumulates(
    db: Database, tmp_path: Path
) -> None:
    video_id = a_diarized_video(db)
    wav = a_wav(tmp_path / "a.wav", 20_000)
    first_row = "SELECT embedding FROM video_speakers ORDER BY id LIMIT 1"
    embed_video_speakers(db, video_id, wav_path=wav, embedder=FakeEmbedder())
    first = bytes(db.conn.execute(first_row).fetchone()[0])
    embed_video_speakers(db, video_id, wav_path=wav, embedder=ConstantEmbedder())
    second = bytes(db.conn.execute(first_row).fetchone()[0])
    assert first != second


def test_embedder_name_prefers_the_request_then_the_setting_then_the_default(
    db: Database,
) -> None:
    assert embedder_name(db, "") == C.DEFAULT_EMBEDDER
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        (C.SETTINGS_EMBEDDER, "redimnet"),
    )
    assert embedder_name(db, "") == "redimnet"
    assert embedder_name(db, "fake-embedder") == "fake-embedder"


# -- the audio helper and the adapter --------------------------------------


def test_concatenating_windows_produces_their_total_length(tmp_path: Path) -> None:
    source = a_wav(tmp_path / "src.wav", 10_000)
    out = concat_wav_windows(source, tmp_path / "out.wav", [(0, 1_000), (5_000, 7_000)])
    with wave.open(str(out), "rb") as reader:
        assert reader.getnframes() == 16_000 * 3
        assert reader.getframerate() == 16_000
        assert reader.getnchannels() == 1


def test_concatenating_no_windows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError):
        concat_wav_windows(a_wav(tmp_path / "src.wav", 100), tmp_path / "o.wav", [])


def test_the_redimnet_adapter_imports_and_registers_without_torch() -> None:
    assert importlib.util.find_spec("torch") is None, "the dev venv must have no torch"
    from rytp.diarize.embed import ReDimNetEmbedder

    assert EMBEDDERS["redimnet"] is ReDimNetEmbedder
    assert ReDimNetEmbedder.out_of_process is True
    assert ReDimNetEmbedder.requires_hf_token is False


def test_the_redimnet_adapter_sends_the_request_the_child_expects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import embed as embed_module
    from rytp.diarize.embed import ReDimNetEmbedder

    seen: dict[str, object] = {}

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"vector": [1.0, 0.0, 0.0]}

    monkeypatch.setattr(embed_module, "run_child", fake_run_child)
    engine = ReDimNetEmbedder(interpreter="/opt/redimnet/python")
    vector = engine.embed(tmp_path / "a.wav", [(0, 1_000)])

    assert vector == [1.0, 0.0, 0.0]
    assert seen["interpreter"] == "/opt/redimnet/python"
    assert seen["module"] == "rytp.diarize.embed"
    request = seen["request"]
    assert request["windows"] == [[0, 1_000]]
    assert request["model"] == C.REDIMNET_DEFAULT_MODEL
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_embed_stage.py -v
```
Expected: `ImportError: cannot import name 'concat_wav_windows' from 'rytp.diarize.embed'`.

- [ ] **Step 3: Append `label_segments` to `rytp/diarize/store.py`**

Every word is selected, not only the labelled ones, because an *unlabelled*
word between two runs of one label must break the run rather than be skipped
over — otherwise the reconstructed segment would swallow speech that belongs
to nobody.

```python
def label_segments(db: Database, video_id: int) -> list[DiarSegment]:
    """Reconstruct who-spoke-when from the labelled words.

    Diarizer segments are not stored — the schema has no table for them and
    does not need one, because every word already carries the label that
    spoke it. Maximal runs of consecutive words with the same label become
    one segment, and an unlabelled word breaks a run, which is correct: the
    stretch around a voice nobody labelled is not this speaker's.

    This is what makes re-embedding cheap. Re-running the diarizer is the
    largest GPU cost in the project (design §6); re-running an embedder
    over thirty seconds of reconstructed speech is nothing.
    """
    rows = db.conn.execute(
        """
        SELECT w.start_ms, w.end_ms, w.video_speaker_id, vs.local_label
        FROM words w
        LEFT JOIN video_speakers vs ON vs.id = w.video_speaker_id
        WHERE w.video_id = ?
        ORDER BY w.ord
        """,
        (video_id,),
    ).fetchall()

    segments: list[DiarSegment] = []
    current: int | None = None
    start = end = 0
    label = ""

    def close() -> None:
        nonlocal current
        if current is not None:
            segments.append(DiarSegment(start_ms=start, end_ms=end, local_label=label))
            current = None

    for row in rows:
        if row["video_speaker_id"] is None:
            close()
            continue
        if int(row["video_speaker_id"]) != current:
            close()
            current = int(row["video_speaker_id"])
            label = row["local_label"]
            start = int(row["start_ms"])
        end = int(row["end_ms"] if row["end_ms"] is not None else row["start_ms"])
    close()
    return segments
```

- [ ] **Step 4: Append the audio helper and the adapter to `rytp/diarize/embed.py`**

Add `from rytp.transcribe.subproc import run_child` to the module's imports.

```python
def concat_wav_windows(
    src: Path, dst: Path, windows: Sequence[tuple[int, int]]
) -> Path:
    """Copy several time ranges of a PCM WAV into one file, in order.

    Standard library only: this also runs inside the embedder's own
    interpreter, which has torch but need not have anything of ours beyond
    :mod:`rytp.models`.
    """
    import wave

    if not windows:
        raise EmbeddingError("refusing to embed with no audio selected")
    with wave.open(str(src), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        total = reader.getnframes()
        chunks: list[bytes] = []
        for start_ms, end_ms in windows:
            first = min(max(0, start_ms * rate // 1000), total)
            last = min(max(first, end_ms * rate // 1000), total)
            reader.setpos(first)
            chunks.append(reader.readframes(last - first))
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(b"".join(chunks))
    return dst


@register_embedder
class ReDimNetEmbedder:
    """ReDimNet B6 under its own interpreter (design §6).

    Substantially better than pyannote's internal embedding model, and it
    needs torch — whose pins are the reason engines run out of process at
    all. :func:`child_main` below is the only place the library is
    imported, and it runs inside the interpreter named by the
    `engine.interpreter.redimnet` setting.

    **The library call is unverified.** Nobody here has run ReDimNet. Its
    *contract* — the dictionary `child_main` returns — is what the rest of
    this part depends on, and that is covered by tests. When you install
    the real thing, expect to rewrite the body of one function and nothing
    else.
    """

    name = "redimnet"
    requires_hf_token = False
    out_of_process = True
    required_module = "torch"
    extra = "redimnet"
    dim = 192

    def __init__(self, interpreter: str, model: str = C.REDIMNET_DEFAULT_MODEL) -> None:
        self._interpreter = interpreter
        self._model = model

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> Sequence[float]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.diarize.embed",
            request={
                "audio": str(audio),
                "model": self._model,
                "windows": [[int(start), int(end)] for start, end in windows],
            },
        )
        return [float(value) for value in result["vector"]]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside ReDimNet's own interpreter. The only import of torch."""
    import tempfile

    import redimnet  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import torchaudio  # type: ignore[import-not-found]

    windows = [(int(start), int(end)) for start, end in request["windows"]]
    with tempfile.TemporaryDirectory(prefix="rytp-embed-") as tmp:
        clip = concat_wav_windows(Path(request["audio"]), Path(tmp) / "clip.wav", windows)
        waveform, _rate = torchaudio.load(str(clip))
        model = redimnet.ReDimNet.from_pretrained(request["model"])
        model.eval()
        with torch.no_grad():
            vector = model(waveform).squeeze().tolist()
    return {"vector": [float(value) for value in vector]}
```

- [ ] **Step 5: Append the embedding stage to `rytp/diarize/pipeline.py`**

```python
@dataclass(frozen=True)
class EmbedOutcome:
    """What one embedding pass did."""

    video_id: int
    embedder: str
    n_embedded: int
    n_skipped: int

    def message(self) -> str:
        return (
            f"video {self.video_id}: {self.n_embedded} voices embedded with "
            f"{self.embedder}, {self.n_skipped} skipped for too little speech"
        )


def embedder_name(db: Database, requested: str | None) -> str:
    """What to embed with: the request, then the setting, then the default."""
    if requested:
        return requested
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_EMBEDDER) or C.DEFAULT_EMBEDDER


def embed_video_speakers(
    db: Database, video_id: int, *, wav_path: Path, embedder: SpeakerEmbedder
) -> EmbedOutcome:
    """Give every sufficiently talkative label of one video a voice vector.

    The speech is reconstructed from the labelled words, not from a stored
    segment list, so this can be re-run on its own at any time — and it
    should be, whenever a better embedder arrives. A label with less than
    `C.EMBED_MIN_SPEECH_MS` of speech is skipped rather than embedded
    badly: a vector from two seconds of audio is a fingerprint of the
    phonemes, not of the voice, and a wrong suggestion is worse than none.
    """
    from rytp.diarize.embed import l2_normalize, pack_embedding
    from rytp.diarize.segments import speech_ms_by_label, windows_for_label

    segments = store.label_segments(db, video_id)
    totals = speech_ms_by_label(segments)

    embedded = skipped = 0
    with db.transaction():
        for row in store.label_rows(db, video_id):
            if totals.get(row.local_label, 0) < C.EMBED_MIN_SPEECH_MS:
                store.set_embedding(db, row.video_speaker_id, None)
                skipped += 1
                continue
            windows = windows_for_label(segments, row.local_label)
            vector = list(embedder.embed(wav_path, windows))
            store.set_embedding(
                db, row.video_speaker_id, pack_embedding(l2_normalize(vector))
            )
            embedded += 1

    return EmbedOutcome(
        video_id=video_id,
        embedder=embedder.name,
        n_embedded=embedded,
        n_skipped=skipped,
    )
```

Add `SpeakerEmbedder` to the `TYPE_CHECKING` imports at the top of
`pipeline.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rytp.diarize.embed import SpeakerEmbedder
```

Then extend `run_diarize_job` so a job may embed in the same pass:

```python
def run_diarize_job(db: Database, video_id: int, payload: dict[str, object]) -> None:
    """The `diarize` job kind's body (contracts §5)."""
    wav = wav_for(db, video_id)
    name = diarizer_name(db, str(payload.get("diarizer") or ""))
    diarize_video(db, video_id, wav_path=wav, diarizer=load_diarizer(db, name))

    embedder = embedder_name(db, str(payload.get("embedder") or ""))
    if embedder:
        from rytp.diarize.embed import load_embedder

        embed_video_speakers(
            db, video_id, wav_path=wav, embedder=load_embedder(db, embedder)
        )
```

- [ ] **Step 6: Add the extras to `pyproject.toml`**

```toml
pyannote = ["pyannote.audio>=4.0"]
redimnet = ["torch>=2.1", "torchaudio>=2.1", "redimnet>=0.1"]
```

and add both to the `all` extra. They are separate from Part 3's transcriber
extras on purpose: design §6 says the recommended transcriber and diarizer pin
incompatible versions of their shared dependencies, so installing them into
one environment is precisely what the out-of-process seam exists to avoid.
Point `engine.interpreter.pyannote` and `engine.interpreter.redimnet` at the
Python inside their own virtual environments.

**One thing to check when you build those environments.** The child imports
`rytp.diarize.embed` (or `.pyannote`), which pulls in `rytp/diarize/__init__.py`
→ `base` → `rytp.transcribe.registry` → `rytp.transcribe.base` → `rytp.models`.
So each engine venv must be able to import `rytp.models`, including whatever
`rytp/models.py` imports at module level — `snowballstemmer`, if Part 1 put the
stemmer there rather than behind a lazy import. From the repo root:

```bash
<engine-venv>/bin/python -c "import rytp.models, rytp.diarize.embed; print('child imports ok')"
```

If that fails, `pip install snowballstemmer` into the engine venv. It is a pure
Python package of about a hundred kilobytes and cannot conflict with anything.

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_embed_stage.py -v
```
Expected: 12 passed.

- [ ] **Step 8: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 9: Commit**

```bash
git add rytp/diarize/embed.py rytp/diarize/store.py rytp/diarize/pipeline.py \
  pyproject.toml tests/test_diarize_embed_stage.py
git commit -m "feat(diarize): ReDimNet embeddings, reconstructed from words"
```

---

### Task 11: The pyannote diarizer, out of process

`speaker-diarization-community-1` has roughly half the speaker confusion of
pyannote 3.x (design §6), which matters because host-versus-guest confusion
is exactly the failure mode of this corpus. It is gated on a Hugging Face
token and its pins conflict with the transcriber's, so it is both
`requires_hf_token = True` and `out_of_process = True`.

**Files:**
- Create: `rytp/diarize/pyannote.py`
- Modify: `rytp/diarize/__init__.py` (restore the `pyannote` import deferred in Task 2)
- Modify: `tests/test_diarize_none.py` (restore the assertion deferred in Task 2)
- Test: `tests/test_diarize_pyannote.py`

**Interfaces:**
- Consumes: `rytp.diarize.base.register_diarizer`, `rytp.transcribe.subproc.run_child`, `rytp.models.DiarSegment`.
- Produces: `PyannoteDiarizer` (name `pyannote`), the pure helper `segments_from_tracks(tracks) -> list[dict]`, and `child_main(request) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_pyannote.py`:

```python
"""The pyannote adapter: importable without pyannote, gated, out of process."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import base
from rytp.diarize.pyannote import PyannoteDiarizer, segments_from_tracks
from rytp.transcribe.base import EngineUnavailable


def test_the_adapter_imports_and_registers_without_pyannote() -> None:
    assert importlib.util.find_spec("pyannote") is None, "the dev venv must be clean"
    assert base.DIARIZERS["pyannote"] is PyannoteDiarizer
    assert PyannoteDiarizer.requires_hf_token is True
    assert PyannoteDiarizer.out_of_process is True


def test_loading_it_without_a_token_is_refused_before_construction(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable, match="HF_TOKEN"):
        base.load_diarizer(db, "pyannote")


def test_segments_from_tracks_converts_seconds_to_milliseconds() -> None:
    tracks = [(0.0, 1.25, "SPEAKER_00"), (1.25, 3.5, "SPEAKER_01")]
    assert segments_from_tracks(tracks) == [
        {"start_ms": 0, "end_ms": 1_250, "local_label": "SPEAKER_00"},
        {"start_ms": 1_250, "end_ms": 3_500, "local_label": "SPEAKER_01"},
    ]


def test_segments_from_tracks_drops_a_zero_length_turn() -> None:
    assert segments_from_tracks([(1.0, 1.0, "SPEAKER_00")]) == []


def test_the_adapter_sends_the_request_the_child_expects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import pyannote as adapter

    seen: dict[str, object] = {}

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"segments": [{"start_ms": 0, "end_ms": 1_000, "local_label": "SPEAKER_00"}]}

    monkeypatch.setattr(adapter, "run_child", fake_run_child)
    engine = PyannoteDiarizer(interpreter="/opt/pyannote/python", hf_token="t")
    segments = list(engine.diarize(tmp_path / "a.wav"))

    assert [s.local_label for s in segments] == ["SPEAKER_00"]
    assert segments[0].end_ms == 1_000
    assert seen["interpreter"] == "/opt/pyannote/python"
    assert seen["module"] == "rytp.diarize.pyannote"
    request = seen["request"]
    assert request["model"] == C.PYANNOTE_DIARIZATION_MODEL
    assert request["hf_token"] == "t"


def test_the_adapter_reads_the_token_from_the_environment_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.diarize import pyannote as adapter

    seen: dict[str, object] = {}
    monkeypatch.setenv("HF_TOKEN", "from-env")

    def fake_run_child(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"segments": []}

    monkeypatch.setattr(adapter, "run_child", fake_run_child)
    list(PyannoteDiarizer(interpreter="/opt/p/python").diarize(tmp_path / "a.wav"))
    assert seen["request"]["hf_token"] == "from-env"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_pyannote.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.pyannote'`.

- [ ] **Step 3: Write `rytp/diarize/pyannote.py`**

```python
"""pyannote `speaker-diarization-community-1`, out of process (design §6).

Roughly half the speaker confusion of pyannote 3.x, which is worth having:
the confusion 3.x makes is host-versus-guest, and this corpus is mostly
host and guest.

Two properties shape the adapter. It needs a Hugging Face token, checked
against the *class* before anything is constructed, because building the
pipeline without one downloads two gigabytes and then fails. And it pins
versions that conflict with the transcriber's, so it runs under the
interpreter named by the `engine.interpreter.pyannote` setting and never
imports its dependency in this process at all.

:func:`child_main` runs inside that interpreter and is the only place
`pyannote.audio` is imported. **The library call is unverified** — nobody
here has run community-1. Its *contract*, the dictionary it returns, is
what the rest of this part depends on, and that is covered by tests.
Expect to rewrite the body of one function when you install the real thing.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.diarize.base import register_diarizer
from rytp.models import DiarSegment
from rytp.transcribe.subproc import run_child


def segments_from_tracks(
    tracks: Sequence[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    """pyannote's ``(start_s, end_s, label)`` triples to the wire shape.

    Pure, so the conversion is tested without the library. A zero-length
    turn is dropped: pyannote emits them at the edges of some pipelines and
    a segment with no duration cannot own a word.
    """
    out: list[dict[str, Any]] = []
    for start_s, end_s, label in tracks:
        start_ms = int(round(float(start_s) * 1000))
        end_ms = int(round(float(end_s) * 1000))
        if end_ms <= start_ms:
            continue
        out.append({"start_ms": start_ms, "end_ms": end_ms, "local_label": str(label)})
    return out


@register_diarizer
class PyannoteDiarizer:
    """Diarization under pyannote's own interpreter."""

    name = "pyannote"
    requires_hf_token = True
    out_of_process = True
    required_module = "pyannote.audio"
    extra = "pyannote"

    def __init__(
        self,
        interpreter: str,
        model: str = C.PYANNOTE_DIARIZATION_MODEL,
        hf_token: str | None = None,
    ) -> None:
        self._interpreter = interpreter
        self._model = model
        self._hf_token = (
            hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        )

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        """Whole file in, labelled turns out.

        Always the whole file, never a chunk: speaker labels are
        context-dependent and chunking makes them inconsistent between
        chunks. Only the transcriber chunks (design §6).
        """
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.diarize.pyannote",
            request={
                "audio": str(audio),
                "model": self._model,
                "hf_token": self._hf_token,
            },
        )
        return [
            DiarSegment(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                local_label=str(item["local_label"]),
            )
            for item in result["segments"]
        ]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside pyannote's own interpreter. The only import of the library."""
    from pyannote.audio import Pipeline  # type: ignore[import-not-found]

    pipeline = Pipeline.from_pretrained(request["model"], token=request["hf_token"])
    annotation = pipeline(request["audio"])
    tracks = [
        (segment.start, segment.end, label)
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    return {"segments": segments_from_tracks(tracks)}
```

- [ ] **Step 4: Restore the deferred import in `rytp/diarize/__init__.py`**

```python
from rytp.diarize import embed, none, pyannote  # noqa: F401
```

- [ ] **Step 5: Restore the deferred assertion in `tests/test_diarize_none.py`**

In `test_importing_the_package_registers_the_built_in_names`, change
`assert "none" in base.DIARIZERS` back to:

```python
    assert "pyannote" in base.DIARIZERS
```

- [ ] **Step 6: Run both test files to verify they pass**

Run:
```bash
python -m pytest tests/test_diarize_pyannote.py tests/test_diarize_none.py -v
```
Expected: 9 passed.

- [ ] **Step 7: Prove the package is still import-light**

```bash
python -c "import sys, rytp.diarize; assert 'torch' not in sys.modules and 'pyannote' not in sys.modules; print('clean')"
```
Expected: `clean`.

- [ ] **Step 8: Lint, type-check, commit**

```bash
ruff check rytp tests
mypy rytp/diarize
git add rytp/diarize/pyannote.py rytp/diarize/__init__.py \
  tests/test_diarize_pyannote.py tests/test_diarize_none.py
git commit -m "feat(diarize): the pyannote community-1 diarizer behind the subprocess seam"
```

---

### Task 12: Eras, acoustic scoping and two-threshold suggestions

The corpus spans more than a decade. Microphones changed, rooms changed,
and the main speaker aged — which roughly triples voice-embedding error
(design §6). Automatic linking across that is not trustworthy, so this
task produces **suggestions and nothing else**. No code path here writes a
`speaker_id`, and a test asserts that generating suggestions performs zero
database writes.

Three mechanisms, all from design §6:

- **Per-era enrolment centroids.** A person's linked labels are grouped
  into era buckets and each bucket is averaged, so a 2013 recording is
  compared against how they sounded in 2013.
- **Acoustic scoping.** `video_acoustics` (Part 3) says which recordings
  resemble each other. This is its second consumer, after the assembler.
- **A two-threshold band.** Confident match, confident non-match, and a
  middle that goes to the human. Across eras — or when the acoustics are
  unknown, which is the same kind of ignorance — only a much higher score
  counts as confident.

And one hard gate in front of all three: **labels from different diarizers
are never compared.** `video_speakers.engine` (contracts §3) records which
engine drew each label's turn boundaries, and a vector measured over one
engine's segments is not a measurement of the same thing as a vector
measured over another's. Such a pair is skipped, and `engines_in_play`
exists so a caller can say *why* a list came back short instead of leaving
it a mystery.

**Files:**
- Create: `rytp/diarize/link.py`
- Modify: `rytp/diarize/mapper.py` (append `suggestions()` and `suggestion_table()`)
- Test: `tests/test_diarize_link.py`

**Interfaces:**
- Consumes: `rytp.diarize.embed.{centroid, comparable, cosine_similarity, unpack_embedding}`, `rytp.diarize.store` in full, the `C.SPEAKER_*` and `C.ACOUSTIC_*` constants.
- Produces:
  - `MATCH`, `REVIEW`, `NO_MATCH` (str constants), `EnrolmentSample(video_id, engine, vector)`, `Suggestion(video_speaker_id, speaker_id, speaker_label, similarity, era, same_era, acoustically_close, engine, verdict)`
  - `era_of(published_at: str | None) -> str`
  - `acoustic_distance(db, video_a: int, video_b: int) -> float | None`
  - `enrolment(db, speaker_id) -> dict[str, list[EnrolmentSample]]`, `engines_in_play(db) -> set[str]`
  - `verdict_for(similarity: float, *, strict: bool) -> str`
  - `suggest_for_label(db, video_speaker_id, *, limit=C.SPEAKER_SUGGEST_LIMIT) -> list[Suggestion]`
  - `suggest_for_video(db, video_id, *, limit=C.SPEAKER_SUGGEST_LIMIT) -> dict[int, list[Suggestion]]`
  - `MappingSession.suggestions() -> list[Suggestion]` and `MappingSession.suggestion_table()`

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_link.py`:

```python
"""Cross-video suggestions: scoped, banded, and never applied."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import pack_embedding
from rytp.diarize.link import (
    MATCH,
    NO_MATCH,
    REVIEW,
    acoustic_distance,
    engines_in_play,
    enrolment,
    era_of,
    suggest_for_label,
    suggest_for_video,
    verdict_for,
)
from tests.fakes import make_video
from tests.test_speakers_labels import DIARIZER, add_label, add_words

# Unit vectors whose cosine with HOST_VOICE is exactly the first component,
# so every expected verdict below can be read off the thresholds by eye.
HOST_VOICE = [1.0, 0.0, 0.0, 0.0]
NEAR_HOST = [0.95, 0.3122499, 0.0, 0.0]     # 0.95: over HI and over HI_CROSS_ERA
MID_HOST = [0.75, 0.6614378, 0.0, 0.0]      # 0.75: over HI, under HI_CROSS_ERA
HALF_WAY = [0.6, 0.8, 0.0, 0.0]             # 0.60: in the band either way
OTHER_VOICE = [0.0, 1.0, 0.0, 0.0]          # 0.00: under LO


def a_labelled_video(
    db: Database,
    *,
    external_id: str,
    published_at: str,
    vector: list[float] | None = None,
    speaker_id: int | None = None,
    acoustics: dict[str, float] | None = None,
    engine: str = DIARIZER,
) -> tuple[int, int]:
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    label_id = add_label(db, video_id, "SPEAKER_00", engine=engine)
    store.stamp_words(db, video_id, dict.fromkeys(word_ids, label_id))
    if vector is not None:
        store.set_embedding(db, label_id, pack_embedding(vector))
    if speaker_id is not None:
        store.link_video_speaker(db, label_id, speaker_id)
    if acoustics is not None:
        columns = ", ".join(acoustics)
        marks = ", ".join("?" for _ in acoustics)
        db.conn.execute(
            f"INSERT INTO video_acoustics (video_id, {columns}, computed_at) "
            f"VALUES (?, {marks}, '2026-01-01T00:00:00+00:00')",
            (video_id, *acoustics.values()),
        )
    return video_id, label_id


# -- eras ------------------------------------------------------------------


def test_eras_are_two_year_buckets() -> None:
    assert era_of("2014-03-02T00:00:00+00:00") == era_of("2015-11-30")
    assert era_of("2014-03-02") != era_of("2016-01-01")


def test_an_era_label_names_its_range() -> None:
    assert era_of("2015-06-01") == "2014-2015"


def test_a_video_with_no_date_lands_in_the_unknown_bucket() -> None:
    assert era_of(None) == C.SPEAKER_ERA_UNKNOWN
    assert era_of("") == C.SPEAKER_ERA_UNKNOWN
    assert era_of("not a date") == C.SPEAKER_ERA_UNKNOWN


# -- acoustic scoping ------------------------------------------------------


def test_two_similar_recordings_are_close(db: Database) -> None:
    a, _ = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    b, _ = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-06-01",
        acoustics={"f0_mean": 124.0, "noise_floor_db": -53.0},
    )
    assert acoustic_distance(db, a, b) < C.ACOUSTIC_MAX_DISTANCE


def test_two_different_recordings_are_far(db: Database) -> None:
    a, _ = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        acoustics={"f0_mean": 110.0, "noise_floor_db": -70.0},
    )
    b, _ = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-06-01",
        acoustics={"f0_mean": 200.0, "noise_floor_db": -30.0},
    )
    assert acoustic_distance(db, a, b) > C.ACOUSTIC_MAX_DISTANCE


def test_distance_is_unknown_when_a_video_was_never_fingerprinted(db: Database) -> None:
    a, _ = a_labelled_video(db, external_id="VIDEO_A", published_at="2020-01-01")
    b, _ = a_labelled_video(db, external_id="VIDEO_B", published_at="2020-06-01")
    assert acoustic_distance(db, a, b) is None


# -- the band --------------------------------------------------------------


def test_a_high_score_within_an_era_is_a_match() -> None:
    assert verdict_for(0.9, strict=False) == MATCH


def test_the_same_score_across_eras_only_earns_a_review() -> None:
    # design §6: a decade of changing microphones roughly triples the error.
    assert verdict_for(0.75, strict=False) == MATCH
    assert verdict_for(0.75, strict=True) == REVIEW


def test_a_low_score_is_a_confident_non_match() -> None:
    assert verdict_for(0.1, strict=False) == NO_MATCH
    assert verdict_for(0.1, strict=True) == NO_MATCH


def test_the_middle_of_the_band_goes_to_the_human() -> None:
    middle = (C.SPEAKER_MATCH_LO + C.SPEAKER_MATCH_HI) / 2
    assert verdict_for(middle, strict=False) == REVIEW


# -- enrolment -------------------------------------------------------------


def test_enrolment_groups_a_persons_labels_by_era(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2014-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2022-01-01",
        vector=NEAR_HOST, speaker_id=speaker_id,
    )
    grouped = enrolment(db, speaker_id)
    assert sorted(grouped) == ["2014-2015", "2022-2023"]
    assert len(grouped["2014-2015"]) == 1


def test_enrolment_carries_the_engine_each_sample_came_from(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2014-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    [sample] = enrolment(db, speaker_id)["2014-2015"]
    assert sample.engine == "pyannote"


def test_engines_in_play_reports_only_engines_with_embeddings(db: Database) -> None:
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, engine="pyannote",
    )
    # No vector, so this engine is not in play for comparison purposes.
    a_labelled_video(db, external_id="VIDEO_B", published_at="2020-02-01", engine="none")
    assert engines_in_play(db) == {"pyannote"}


# -- suggestions -----------------------------------------------------------


def test_a_near_identical_voice_in_the_same_era_is_suggested_as_a_match(
    db: Database,
) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01",
        vector=NEAR_HOST, acoustics={"f0_mean": 122.0, "noise_floor_db": -54.0},
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.speaker_label == "Host One"
    assert suggestion.verdict == MATCH
    assert suggestion.same_era is True
    assert suggestion.acoustically_close is True
    assert suggestion.similarity == pytest.approx(0.95, abs=0.01)


def test_the_same_voice_a_decade_apart_only_earns_a_review(db: Database) -> None:
    # 0.75 would be a confident match inside one era. Across fourteen years
    # it is only a suggestion, which is the whole point of the second bar.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2010-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
        acoustics={"f0_mean": 120.0, "noise_floor_db": -55.0},
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2024-01-01",
        vector=MID_HOST, acoustics={"f0_mean": 122.0, "noise_floor_db": -54.0},
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.same_era is False
    assert suggestion.verdict == REVIEW


def test_an_unfingerprinted_pair_is_treated_as_strictly_as_a_cross_era_one(
    db: Database,
) -> None:
    # Same era, same 0.75 as the test above, but nobody ran `fingerprint`:
    # not knowing whether the recordings resemble each other is the same
    # kind of ignorance as knowing they were made a decade apart.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01", vector=MID_HOST
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.same_era is True
    assert suggestion.acoustically_close is False
    assert suggestion.verdict == REVIEW


def test_a_different_voice_is_reported_as_a_non_match_not_hidden(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-09-01", vector=OTHER_VOICE
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.verdict == NO_MATCH


def test_suggestions_are_ranked_best_first(db: Database) -> None:
    near, _ = store.add_speaker(db, "Host One")
    far, _ = store.add_speaker(db, "Guest Two")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=near,
    )
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HALF_WAY, speaker_id=far,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_C", published_at="2020-09-01", vector=NEAR_HOST
    )
    assert [s.speaker_label for s in suggest_for_label(db, query_label)] == [
        "Host One",
        "Guest Two",
    ]


def test_a_person_already_on_another_label_of_this_video_is_not_suggested(
    db: Database,
) -> None:
    # One person cannot be two voices in the same recording.
    speaker_id, _ = store.add_speaker(db, "Host One")
    video_id = make_video(db, published_at="2020-01-01")
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    taken = add_label(db, video_id, "SPEAKER_00")
    query = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: taken, word_ids[1]: query})
    store.set_embedding(db, query, pack_embedding(NEAR_HOST))
    store.link_video_speaker(db, taken, speaker_id)
    a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    assert suggest_for_label(db, query) == []


def test_a_label_with_no_embedding_gets_no_suggestions(db: Database) -> None:
    store.add_speaker(db, "Host One")
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01"
    )
    assert suggest_for_label(db, query_label) == []


def test_a_roster_entry_with_no_embedded_labels_is_skipped(db: Database) -> None:
    store.add_speaker(db, "Host One")
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01", vector=HOST_VOICE
    )
    assert suggest_for_label(db, query_label) == []


def test_labels_from_different_diarizers_are_never_compared(db: Database) -> None:
    # Different engines draw different turn boundaries, so a vector measured
    # over one engine's segments is not a measurement of the same thing.
    # Identical vectors, and still no suggestion.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, engine="none",
    )
    assert suggest_for_label(db, query_label) == []
    # …and the caller can find out why rather than guessing.
    assert engines_in_play(db) == {"none", "pyannote"}


def test_the_same_diarizer_on_both_sides_does_compare(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id, engine="pyannote",
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01",
        vector=HOST_VOICE, engine="pyannote",
    )
    [suggestion] = suggest_for_label(db, query_label)
    assert suggestion.engine == "pyannote"
    assert suggestion.verdict == MATCH


def test_a_vector_of_another_dimension_is_skipped_not_fatal(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=[1.0, 0.0, 0.0], speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=HOST_VOICE
    )
    assert suggest_for_label(db, query_label) == []


def test_generating_suggestions_writes_nothing(db: Database) -> None:
    # design §6: cross-video linking is never applied silently.
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    _video, query_label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    before = db.conn.total_changes
    assert suggest_for_label(db, query_label)
    assert db.conn.total_changes == before


def test_suggest_for_video_covers_every_unlinked_label(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    video_id, label_id = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    assert list(suggest_for_video(db, video_id)) == [label_id]


def test_the_mapper_exposes_suggestions_for_the_highlighted_label(db: Database) -> None:
    from rytp.diarize.mapper import MappingSession

    speaker_id, _ = store.add_speaker(db, "Host One")
    a_labelled_video(
        db, external_id="VIDEO_A", published_at="2020-01-01",
        vector=HOST_VOICE, speaker_id=speaker_id,
    )
    video_id, _label = a_labelled_video(
        db, external_id="VIDEO_B", published_at="2020-02-01", vector=NEAR_HOST
    )
    session = MappingSession(db, video_id)
    _columns, rows = session.suggestion_table()
    assert rows[0][0] == "Host One"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_link.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.link'`.

- [ ] **Step 3: Write `rytp/diarize/link.py`**

```python
"""Suggesting, never applying, cross-video speaker links (design §6).

The corpus spans more than a decade: microphones changed, rooms changed,
the main speaker aged. That roughly triples voice-embedding error, so
automatic linking across eras is untrustworthy and **must never be applied
silently**. Nothing in this module writes to the database. The only writer
is `store.link_video_speaker`, and the only things that call it are a
command a human ran and a keystroke a human pressed.

Three mechanisms, all from design §6:

**Per-era enrolment centroids.** A person's linked labels are bucketed by
the publication date of their video and each bucket is averaged, so a 2013
voice is compared against how that person sounded in 2013 rather than
against a smear of their whole life.

**Acoustic scoping.** `video_acoustics` (Part 3) is one cheap row per video
saying what the recording sounds like. It exists to tell the assembler
which sources blend; this is its second consumer, deciding which
recordings resemble each other enough for a voice comparison to mean
anything.

**A two-threshold band.** Above HI is a confident match, at or below LO a
confident non-match, and the space between goes to the human. Across eras
— or when the acoustics are unknown, which is the same kind of ignorance —
only a much higher score counts as confident. Every one of those numbers
is a guess until somebody measures this corpus; they are marked
UNVALIDATED in `constants.py` and are meant to be re-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import centroid, comparable, cosine_similarity, unpack_embedding

#: The three answers. Strings rather than an enum because they are rendered
#: straight into a `CommandResult` row and a TUI cell.
MATCH = "match"
REVIEW = "review"
NO_MATCH = "no-match"


@dataclass(frozen=True)
class EnrolmentSample:
    """One embedded label of one person: where it came from and what it sounds like."""

    video_id: int
    engine: str
    vector: list[float]


@dataclass(frozen=True)
class Suggestion:
    """One roster candidate for one local label. Advisory, always."""

    video_speaker_id: int
    speaker_id: int
    speaker_label: str
    similarity: float
    era: str
    same_era: bool
    acoustically_close: bool
    engine: str
    verdict: str


def era_of(published_at: str | None) -> str:
    """Which era bucket a publication date falls in.

    Buckets are `C.SPEAKER_ERA_YEARS` wide and named by their range, so a
    label in a listing explains itself. A video with no usable date — a
    local file, usually — gets its own bucket rather than being silently
    filed under some year.
    """
    if not published_at:
        return C.SPEAKER_ERA_UNKNOWN
    head = published_at[:4]
    if not head.isdigit():
        return C.SPEAKER_ERA_UNKNOWN
    year = int(head)
    start = year - (year % C.SPEAKER_ERA_YEARS)
    return f"{start}-{start + C.SPEAKER_ERA_YEARS - 1}"


def acoustic_distance(db: Database, video_a: int, video_b: int) -> float | None:
    """How unlike two recordings sound, or ``None`` if nobody measured.

    The mean of the per-feature differences, each divided by its scale, so
    1.0 means "one typical spread apart on average". Features missing from
    either row are skipped; if none survive, the answer is ignorance rather
    than zero — and ignorance is treated strictly by the caller.
    """
    columns = ", ".join(name for name, _scale in C.ACOUSTIC_FEATURE_SCALES)
    rows = {
        int(row["video_id"]): row
        for row in db.conn.execute(
            f"SELECT video_id, {columns} FROM video_acoustics WHERE video_id IN (?, ?)",
            (video_a, video_b),
        ).fetchall()
    }
    left, right = rows.get(video_a), rows.get(video_b)
    if left is None or right is None:
        return None
    ratios: list[float] = []
    for name, scale in C.ACOUSTIC_FEATURE_SCALES:
        if left[name] is None or right[name] is None:
            continue
        ratios.append(abs(float(left[name]) - float(right[name])) / scale)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def enrolment(db: Database, speaker_id: int) -> dict[str, list[EnrolmentSample]]:
    """One person's embedded labels, grouped by era."""
    grouped: dict[str, list[EnrolmentSample]] = {}
    for label in store.linked_labels(db, speaker_id):
        row = db.conn.execute(
            "SELECT vs.embedding, v.published_at FROM video_speakers vs "
            "JOIN videos v ON v.id = vs.video_id WHERE vs.id = ?",
            (label.video_speaker_id,),
        ).fetchone()
        if row is None or row["embedding"] is None:
            continue
        grouped.setdefault(era_of(row["published_at"]), []).append(
            EnrolmentSample(
                video_id=label.video_id,
                engine=label.engine,
                vector=unpack_embedding(bytes(row["embedding"])),
            )
        )
    return grouped


def engines_in_play(db: Database) -> set[str]:
    """Every diarizer that produced a label currently carrying an embedding.

    More than one means some pairs are simply not comparable, which is a
    thing to *say* rather than to leave as a mysteriously short list.
    """
    rows = db.conn.execute(
        "SELECT DISTINCT engine FROM video_speakers WHERE embedding IS NOT NULL"
    ).fetchall()
    return {row["engine"] for row in rows}


def verdict_for(similarity: float, *, strict: bool) -> str:
    """Place a score in the two-threshold band (design §6)."""
    high = C.SPEAKER_MATCH_HI_CROSS_ERA if strict else C.SPEAKER_MATCH_HI
    if similarity >= high:
        return MATCH
    if similarity <= C.SPEAKER_MATCH_LO:
        return NO_MATCH
    return REVIEW


def suggest_for_label(
    db: Database, video_speaker_id: int, *, limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> list[Suggestion]:
    """Rank the roster against one local label's voice. Writes nothing."""
    label = store.get_label(db, video_speaker_id)
    row = db.conn.execute(
        "SELECT vs.embedding, v.published_at FROM video_speakers vs "
        "JOIN videos v ON v.id = vs.video_id WHERE vs.id = ?",
        (video_speaker_id,),
    ).fetchone()
    if row is None or row["embedding"] is None:
        return []
    query = unpack_embedding(bytes(row["embedding"]))
    query_era = era_of(row["published_at"])
    query_engine = label.engine

    # One person cannot be two voices in one recording.
    taken = {
        other.speaker_id
        for other in store.label_rows(db, label.video_id)
        if other.speaker_id is not None and other.video_speaker_id != video_speaker_id
    }

    found: list[Suggestion] = []
    for person in store.roster_rows(db):
        if person.speaker_id in taken:
            continue
        best: Suggestion | None = None
        for era, members in enrolment(db, person.speaker_id).items():
            # Two rules for "can these be compared at all", and neither is
            # negotiable. Different diarizers draw different turn
            # boundaries, so a vector measured over one engine's segments
            # says nothing about a vector measured over another's; and two
            # embedders produce two dimensionalities. Both are skipped
            # rather than scored, and `engines_in_play` is how a caller
            # explains a suspiciously short list.
            usable = [
                sample
                for sample in members
                if sample.engine == query_engine and comparable(sample.vector, query)
            ]
            if not usable:
                continue
            similarity = cosine_similarity(
                query, centroid([sample.vector for sample in usable])
            )
            distances = [
                distance
                for sample in usable
                if (distance := acoustic_distance(db, label.video_id, sample.video_id))
                is not None
            ]
            close = bool(distances) and min(distances) <= C.ACOUSTIC_MAX_DISTANCE
            same_era = era == query_era and era != C.SPEAKER_ERA_UNKNOWN
            candidate = Suggestion(
                video_speaker_id=video_speaker_id,
                speaker_id=person.speaker_id,
                speaker_label=person.label,
                similarity=similarity,
                era=era,
                same_era=same_era,
                acoustically_close=close,
                engine=query_engine,
                verdict=verdict_for(similarity, strict=not (same_era and close)),
            )
            if best is None or candidate.similarity > best.similarity:
                best = candidate
        if best is not None:
            found.append(best)

    found.sort(key=lambda s: (-s.similarity, s.speaker_label))
    return found[:limit]


def suggest_for_video(
    db: Database, video_id: int, *, limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> dict[int, list[Suggestion]]:
    """Suggestions for every label of one video that nobody has named yet."""
    out: dict[int, list[Suggestion]] = {}
    for label in store.label_rows(db, video_id):
        if label.speaker_id is not None:
            continue
        found = suggest_for_label(db, label.video_speaker_id, limit=limit)
        if found:
            out[label.video_speaker_id] = found
    return out
```

- [ ] **Step 4: Append suggestions to `rytp/diarize/mapper.py`**

Two methods on `MappingSession`. The import is inside the method so a session
over a video with no embeddings pays nothing and so `mapper.py` and `link.py`
do not import each other at module level.

```python
    def suggestions(self) -> list[Suggestion]:
        """Roster candidates for the highlighted label. Advisory only."""
        label = self.selected_label
        if label is None:
            return []
        from rytp.diarize.link import suggest_for_label

        return suggest_for_label(self.db, label.video_speaker_id)

    def suggestion_table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """The suggestion pane: who it might be, how sure, and why only that sure."""
        rows = [
            (
                item.speaker_label,
                f"{item.similarity:.2f}",
                item.verdict,
                item.era,
                "same era" if item.same_era else "other era",
                "similar sound" if item.acoustically_close else "unlike or unmeasured",
            )
            for item in self.suggestions()
        ]
        return ("person", "score", "verdict", "era", "when", "sound"), rows
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_link.py -v
```
Expected: 26 passed.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/diarize
```
Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add rytp/diarize/link.py rytp/diarize/mapper.py tests/test_diarize_link.py
git commit -m "feat(diarize): per-era, acoustically scoped speaker suggestions"
```

---

### Task 13: Commands

Thirteen commands, one registry, two surfaces. Handlers take an open
`Database` plus keyword arguments, return a `CommandResult`, never print
and never exit (contracts §5). The two long ones are marked
`long_running=True` so the TUI refuses to run them inline, and
`speakers.map` is marked `cli_only=True` because it *launches* the
interactive surface and cannot sensibly be launched from inside it.

**Two commands with confusable names, deliberately spelled apart.**
`speakers.unlink` detaches *one label of one video* from its person and
leaves the roster alone. `speakers.remove` deletes *the person*, and every
label naming them goes back to being a voice nobody named — the labels
themselves survive, because contracts §5 makes `video_speakers.speaker_id`
`ON DELETE SET NULL` rather than a cascade. Both summaries say which, since
`unlink` and `remove` are close enough that a tired reader will pick the
wrong one.

**Files:**
- Create: `rytp/commands/speakers.py`
- Modify: `rytp/commands/__init__.py` (one line in the bottom import block)
- Test: `tests/test_commands_speakers.py`

**Interfaces:**
- Consumes: `rytp.commands.{Command, CommandResult, Param, REQUIRED, register}`; every Part 7 module above; `rytp.jobs.queue.enqueue`.
- Produces: thirteen registered commands — `speakers.add`, `speakers.list`, `speakers.alias`, `speakers.remove`, `speakers.labels`, `speakers.link`, `speakers.unlink`, `speakers.diarize`, `speakers.embed`, `speakers.enqueue`, `speakers.suggest`, `speakers.engines`, `speakers.map`; and the monkeypatchable seam `rytp.commands.speakers._run_mapper(db, video_id) -> None`.

**Job-kind mapping.** The worker reaches diarization through
`JOB_HANDLERS["diarize"]` (Task 8), not through a command. `speakers.diarize`
is the by-hand path and `speakers.enqueue` is the queue path; both end up in
`rytp.diarize.pipeline`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_commands_speakers.py`:

```python
"""The speakers command group: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.speakers  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.commands import COMMANDS, REQUIRED, resolve
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.embed import pack_embedding
from rytp.models import NotFoundError, RytpError
from tests.fake_speaker_engines import FakeDiarizer, FakeEmbedder, registered
from tests.fakes import make_video
from tests.test_speakers_labels import DIARIZER, add_label, add_words

GROUP = (
    "speakers.add",
    "speakers.alias",
    "speakers.diarize",
    "speakers.remove",
    "speakers.embed",
    "speakers.engines",
    "speakers.enqueue",
    "speakers.labels",
    "speakers.link",
    "speakers.list",
    "speakers.map",
    "speakers.suggest",
    "speakers.unlink",
)


def cache_a_wav(video_id: int, ms: int = 20_000) -> Path:
    import wave

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db)
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[1]: guest})
    return video_id


# -- registration ----------------------------------------------------------


def test_every_command_is_registered_in_the_speakers_group() -> None:
    for name in GROUP:
        assert COMMANDS[name].group == "speakers"


def test_the_long_ones_are_marked_long_running() -> None:
    assert COMMANDS["speakers.diarize"].long_running is True
    assert COMMANDS["speakers.embed"].long_running is True
    assert COMMANDS["speakers.list"].long_running is False


def test_the_mapper_launcher_is_cli_only() -> None:
    # It opens the interactive surface, so it cannot be run from inside it.
    assert COMMANDS["speakers.map"].cli_only is True
    assert COMMANDS["speakers.list"].cli_only is False


def test_every_parameter_has_help_and_a_scalar_type() -> None:
    for name in GROUP:
        for param in COMMANDS[name].params:
            assert param.help
            assert param.type in (str, int, float, bool, Path)


def test_required_parameters_are_positional() -> None:
    link = {p.name: p for p in COMMANDS["speakers.link"].params}
    assert link["video_id"].default is REQUIRED
    assert link["video_id"].positional is True


# -- the roster ------------------------------------------------------------


def test_add_creates_a_speaker_and_reports_the_id(db: Database) -> None:
    result = resolve("speakers.add").handler(
        db, label="Host One", aliases="host, h1", notes=None
    )
    assert "Host One" in (result.message or "")
    assert store.find_speaker(db, "h1") is not None


def test_add_says_when_the_speaker_was_already_there(db: Database) -> None:
    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    result = resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    assert "already" in (result.message or "").lower()


def test_add_warns_about_an_alias_another_speaker_owns(db: Database) -> None:
    resolve("speakers.add").handler(db, label="Host One", aliases="boss", notes=None)
    result = resolve("speakers.add").handler(db, label="Guest Two", aliases="boss", notes=None)
    assert "boss" in (result.message or "")


def test_list_returns_the_roster_as_rows_of_strings(db: Database) -> None:
    store.add_speaker(db, "Host One", aliases=("h1",))
    result = resolve("speakers.list").handler(db)
    assert result.columns[0] == "id"
    assert all(isinstance(cell, str) for row in result.rows for cell in row)
    assert result.rows[0][1] == "Host One"


def test_alias_appends_to_an_existing_speaker(db: Database) -> None:
    store.add_speaker(db, "Host One", aliases=("h1",))
    resolve("speakers.alias").handler(db, speaker="Host One", aliases="boss")
    assert store.find_speaker(db, "boss").label == "Host One"


def test_alias_on_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("speakers.alias").handler(db, speaker="nobody", aliases="x")


def test_remove_deletes_the_person_and_keeps_their_labels(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One", aliases=("h1",))
    resolve("speakers.link").handler(
        db, video_id=video_id, label="SPEAKER_00", speaker="h1"
    )
    result = resolve("speakers.remove").handler(db, speaker="h1", yes=False)
    assert "Host One" in (result.message or "")
    assert "1 local label" in (result.message or "")
    assert store.find_speaker(db, "Host One") is None
    # Both voices are still there, both unnamed.
    rows = store.label_rows(db, video_id)
    assert len(rows) == 2
    assert all(row.speaker_id is None for row in rows)


def test_remove_asks_before_undoing_a_lot_of_mapping(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    for index in range(C.SPEAKER_REMOVE_CONFIRM_LINKS):
        other = make_video(
            db, external_id=f"VIDEO_{index}", url=f"https://example.invalid/{index}"
        )
        label_id = add_label(db, other, "SPEAKER_00")
        store.link_video_speaker(db, label_id, speaker_id)

    with pytest.raises(RytpError, match="--yes"):
        resolve("speakers.remove").handler(db, speaker="Host One", yes=False)
    assert store.find_speaker(db, "Host One") is not None

    result = resolve("speakers.remove").handler(db, speaker="Host One", yes=True)
    assert f"{C.SPEAKER_REMOVE_CONFIRM_LINKS} local labels" in (result.message or "")
    assert store.find_speaker(db, "Host One") is None


def test_remove_and_unlink_describe_different_things(db: Database) -> None:
    # The names are one letter of intent apart; the summaries must not be.
    remove = COMMANDS["speakers.remove"].summary.lower()
    unlink = COMMANDS["speakers.unlink"].summary.lower()
    assert "roster" in remove and "label" in unlink
    assert remove != unlink


def test_remove_an_unknown_speaker_raises(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("speakers.remove").handler(db, speaker="nobody", yes=True)


# -- labels and linking ----------------------------------------------------


def test_labels_lists_a_videos_voices(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.labels").handler(db, video_id=video_id)
    assert [row[0] for row in result.rows] == ["SPEAKER_00", "SPEAKER_01"]


def test_labels_shows_which_diarizer_produced_each_one(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.labels").handler(db, video_id=video_id)
    assert result.columns[-1] == "diarizer"
    assert {row[-1] for row in result.rows} == {DIARIZER}


def test_labels_on_an_undiarized_video_says_so_rather_than_returning_nothing(
    db: Database,
) -> None:
    video_id = make_video(db)
    result = resolve("speakers.labels").handler(db, video_id=video_id)
    assert result.rows == ()
    assert "diariz" in (result.message or "").lower()


def test_link_names_a_local_label(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One", aliases=("h1",))
    result = resolve("speakers.link").handler(
        db, video_id=video_id, label="SPEAKER_00", speaker="h1"
    )
    assert "Host One" in (result.message or "")
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"


def test_link_to_an_unknown_speaker_raises(db: Database) -> None:
    video_id = a_diarized_video(db)
    with pytest.raises(NotFoundError):
        resolve("speakers.link").handler(
            db, video_id=video_id, label="SPEAKER_00", speaker="nobody"
        )


def test_link_to_an_unknown_label_raises(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One")
    with pytest.raises(NotFoundError):
        resolve("speakers.link").handler(
            db, video_id=video_id, label="SPEAKER_99", speaker="Host One"
        )


def test_unlink_puts_a_label_back_to_nobody(db: Database) -> None:
    video_id = a_diarized_video(db)
    store.add_speaker(db, "Host One")
    resolve("speakers.link").handler(
        db, video_id=video_id, label="SPEAKER_00", speaker="Host One"
    )
    resolve("speakers.unlink").handler(db, video_id=video_id, label="SPEAKER_00")
    assert store.find_label(db, video_id, "SPEAKER_00").speaker_id is None


# -- the two long ones -----------------------------------------------------


def test_diarize_runs_the_named_engine(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)
    with registered(FakeDiarizer):
        result = resolve("speakers.diarize").handler(
            db, video_id=video_id, diarizer="fake-diarizer", embedder=""
        )
    assert "2 speakers" in (result.message or "")
    assert len(store.label_rows(db, video_id)) == 2


def test_diarize_can_embed_in_the_same_pass(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    cache_a_wav(video_id)
    with registered(FakeDiarizer, FakeEmbedder):
        resolve("speakers.diarize").handler(
            db, video_id=video_id, diarizer="fake-diarizer", embedder="fake-embedder"
        )
    assert any(row.has_embedding for row in store.label_rows(db, video_id))


def test_diarize_without_a_cached_wav_says_which_step_makes_it(
    db: Database, data_dir: Path
) -> None:
    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    with pytest.raises(RytpError, match="extract"):
        resolve("speakers.diarize").handler(db, video_id=video_id, diarizer="none", embedder="")


def test_embed_is_runnable_on_its_own(db: Database, data_dir: Path) -> None:
    video_id = a_diarized_video(db)
    cache_a_wav(video_id)
    with registered(FakeEmbedder):
        result = resolve("speakers.embed").handler(
            db, video_id=video_id, embedder="fake-embedder"
        )
    assert "embedded" in (result.message or "")
    assert all(row.has_embedding for row in store.label_rows(db, video_id))


def test_embed_with_no_embedder_configured_refuses_rather_than_doing_nothing(
    db: Database, data_dir: Path
) -> None:
    video_id = a_diarized_video(db)
    cache_a_wav(video_id)
    with pytest.raises(RytpError, match="embedder"):
        resolve("speakers.embed").handler(db, video_id=video_id, embedder="")


# -- the queue -------------------------------------------------------------


def test_enqueue_adds_exactly_one_diarize_job(db: Database, data_dir: Path) -> None:
    from rytp.jobs import queue as Q

    video_id = make_video(db)
    add_words(db, video_id, [(0, 400)])
    cache_a_wav(video_id)
    resolve("speakers.enqueue").handler(db, video_id=video_id, priority=5)
    jobs = Q.list_jobs(db)
    assert [(job.kind, job.state, job.priority) for job in jobs] == [
        ("diarize", "pending", 5)
    ]


# -- suggestions -----------------------------------------------------------


def test_suggest_reports_the_verdict_and_writes_nothing(db: Database) -> None:
    speaker_id, _ = store.add_speaker(db, "Host One")
    other = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    other_words = add_words(db, other, [(0, 4_000)])
    other_label = add_label(db, other, "SPEAKER_00")
    store.stamp_words(db, other, dict.fromkeys(other_words, other_label))
    store.set_embedding(db, other_label, pack_embedding([1.0, 0.0, 0.0, 0.0]))
    store.link_video_speaker(db, other_label, speaker_id)

    video_id = a_diarized_video(db)
    label = store.find_label(db, video_id, "SPEAKER_00")
    store.set_embedding(db, label.video_speaker_id, pack_embedding([1.0, 0.0, 0.0, 0.0]))

    before = db.conn.total_changes
    result = resolve("speakers.suggest").handler(
        db, video_id=video_id, label="", limit=C.SPEAKER_SUGGEST_LIMIT
    )
    assert db.conn.total_changes == before
    assert "Host One" in {row[1] for row in result.rows}
    assert "match" in {row[3] for row in result.rows}


def test_suggest_says_plainly_when_there_is_nothing_to_go_on(db: Database) -> None:
    video_id = a_diarized_video(db)
    result = resolve("speakers.suggest").handler(db, video_id=video_id, label="", limit=5)
    assert result.rows == ()
    assert "embed" in (result.message or "").lower()


def test_suggest_names_a_diarizer_mismatch_rather_than_going_quiet(
    db: Database,
) -> None:
    # An empty list because two engines are in play must not look like an
    # empty list because nothing was embedded.
    speaker_id, _ = store.add_speaker(db, "Host One")
    other = make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")
    other_words = add_words(db, other, [(0, 4_000)])
    other_label = add_label(db, other, "SPEAKER_00", engine="pyannote")
    store.stamp_words(db, other, dict.fromkeys(other_words, other_label))
    store.set_embedding(db, other_label, pack_embedding([1.0, 0.0, 0.0, 0.0]))
    store.link_video_speaker(db, other_label, speaker_id)

    video_id = a_diarized_video(db)          # labels come from DIARIZER
    label = store.find_label(db, video_id, "SPEAKER_00")
    store.set_embedding(db, label.video_speaker_id, pack_embedding([1.0, 0.0, 0.0, 0.0]))

    result = resolve("speakers.suggest").handler(db, video_id=video_id, label="", limit=5)
    assert result.rows == ()
    assert "diarizer" in (result.message or "")
    assert "pyannote" in (result.message or "")


# -- engines and the mapper launcher --------------------------------------


def test_engines_lists_both_registries(db: Database) -> None:
    result = resolve("speakers.engines").handler(db)
    names = {row[0] for row in result.rows}
    assert {"none", "pyannote", "redimnet"} <= names
    kinds = {row[1] for row in result.rows}
    assert kinds == {"diarizer", "embedder"}


def test_map_launches_the_mapper_for_that_video(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rytp.commands import speakers as commands_speakers

    seen: list[int] = []
    monkeypatch.setattr(
        commands_speakers, "_run_mapper", lambda db_, video_id: seen.append(video_id)
    )
    video_id = a_diarized_video(db)
    resolve("speakers.map").handler(db, video_id=video_id)
    assert seen == [video_id]


def test_map_refuses_a_video_with_no_labels(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(RytpError, match="diariz"):
        resolve("speakers.map").handler(db, video_id=video_id)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_commands_speakers.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.commands.speakers'`.

- [ ] **Step 3: Write `rytp/commands/speakers.py`**

```python
"""The speakers command group (contracts §5, design §6, §10).

Every operation is defined once here and both surfaces are generated from
it, so the CLI and the TUI cannot drift. Handlers hold no SQL — that is
`rytp.diarize.store` — and no interaction — that is
`rytp.diarize.mapper`. What is left is argument shaping and rendering.

`speakers.map` is the exception that proves the rule: it is `cli_only`,
because its job is to *open* the interactive surface and a surface cannot
sensibly launch itself. Everything it then does is a `MappingSession`,
which the TUI test drives directly.
"""

from __future__ import annotations

from rytp import constants as C
from rytp.commands import REQUIRED, Command, CommandResult, Param, register
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.base import diarizer_rows, load_diarizer
from rytp.diarize.embed import embedder_rows, load_embedder
from rytp.diarize.link import engines_in_play, suggest_for_label, suggest_for_video
from rytp.diarize.mapper import format_ms
from rytp.diarize.pipeline import (
    diarize_video,
    diarizer_name,
    embed_video_speakers,
    embedder_name,
    wav_for,
)
from rytp.models import NotFoundError, RytpError

_EMPTY = "—"


# -- the roster ------------------------------------------------------------


def speakers_add(
    db: Database, *, label: str, aliases: str = "", notes: str | None = None
) -> CommandResult:
    """Add one person to the global roster."""
    parsed = store.parse_aliases(aliases)
    with db.transaction():
        speaker_id, created = store.add_speaker(db, label, aliases=parsed, notes=notes)
    if not created:
        return CommandResult(
            message=f"{label} is already speaker {speaker_id}; "
            f"use `speakers alias` to add aliases"
        )
    collisions = store.find_alias_collisions(
        db, parsed, exclude_speaker_id=speaker_id
    )
    warning = ""
    if collisions:
        taken = ", ".join(sorted({alias for alias, _sid in collisions}))
        warning = f" (warning: {taken} also resolves to another speaker)"
    return CommandResult(message=f"added {label} as speaker {speaker_id}{warning}")


def speakers_list(db: Database) -> CommandResult:
    """Show the roster and how much of the corpus each person accounts for."""
    rows = tuple(
        (
            str(row.speaker_id),
            row.label,
            ", ".join(row.aliases) or _EMPTY,
            str(row.n_videos),
            str(row.n_labels),
        )
        for row in store.roster_rows(db)
    )
    return CommandResult(
        columns=("id", "person", "aliases", "videos", "labels"),
        rows=rows,
        message=None if rows else "the roster is empty; add someone with `speakers add`",
    )


def speakers_alias(db: Database, *, speaker: str, aliases: str) -> CommandResult:
    """Add aliases to a person already on the roster."""
    person = store.resolve_speaker(db, speaker)
    parsed = store.parse_aliases(aliases)
    if not parsed:
        raise RytpError("give at least one alias, comma-separated")
    with db.transaction():
        merged = store.add_aliases(db, person.id, parsed)
    return CommandResult(message=f"{person.label} now answers to {', '.join(merged)}")


def speakers_remove(db: Database, *, speaker: str, yes: bool = False) -> CommandResult:
    """Delete a person from the roster (contracts §5).

    Not `speakers unlink`. That detaches one label of one video; this
    deletes the person, and every label naming them goes back to being a
    voice nobody named. The labels survive — `video_speakers` rows are
    `ON DELETE SET NULL` — so the knowledge that a video had four distinct
    speakers outlives the roster entry.

    No files are touched, so contracts §5 asks for neither `--dry-run` nor
    `--yes`. It does silently undo mapping work, though, so past
    `C.SPEAKER_REMOVE_CONFIRM_LINKS` links it asks first.
    """
    person = store.resolve_speaker(db, speaker)
    labels = store.linked_labels(db, person.id)
    if len(labels) >= C.SPEAKER_REMOVE_CONFIRM_LINKS and not yes:
        videos = len({row.video_id for row in labels})
        raise RytpError(
            f"{person.label} is named on {len(labels)} labels across {videos} "
            f"videos; pass --yes to undo that mapping work "
            f"(to detach just one label, use `speakers unlink`)"
        )
    with db.transaction():
        unlinked = store.remove_speaker(db, person.id)
    plural = "" if unlinked == 1 else "s"
    return CommandResult(
        message=f"removed {person.label} from the roster; {unlinked} local "
        f"label{plural} kept, now unnamed"
    )


# -- labels and linking ----------------------------------------------------


def speakers_labels(db: Database, *, video_id: int) -> CommandResult:
    """Show one video's diarizer labels and who each of them is."""
    rows = store.label_rows(db, video_id)
    if not rows:
        return CommandResult(
            message=f"video {video_id} has no labels; diarize it first "
            f"(`rytp speakers diarize {video_id}`)"
        )
    return CommandResult(
        columns=("label", "words", "speech", "person", "voice", "diarizer"),
        rows=tuple(
            (
                row.local_label,
                str(row.n_words),
                format_ms(row.speech_ms),
                row.speaker_label or _EMPTY,
                "yes" if row.has_embedding else "no",
                row.engine,
            )
            for row in rows
        ),
        message=f"{sum(1 for r in rows if r.speaker_id is None)} still unnamed",
    )


def speakers_link(
    db: Database, *, video_id: int, label: str, speaker: str
) -> CommandResult:
    """Say which real person one of a video's local labels is."""
    row = store.find_label(db, video_id, label)
    person = store.resolve_speaker(db, speaker)
    with db.transaction():
        store.link_video_speaker(db, row.video_speaker_id, person.id)
    return CommandResult(
        message=f"video {video_id} {label} is {person.label} "
        f"({row.n_words} words follow along)"
    )


def speakers_unlink(db: Database, *, video_id: int, label: str) -> CommandResult:
    """Undo a link, leaving the local label as a voice nobody named."""
    row = store.find_label(db, video_id, label)
    with db.transaction():
        store.link_video_speaker(db, row.video_speaker_id, None)
    return CommandResult(message=f"video {video_id} {label} is nobody again")


# -- the two long ones -----------------------------------------------------


def speakers_diarize(
    db: Database, *, video_id: int, diarizer: str = "", embedder: str = ""
) -> CommandResult:
    """Work out who spoke when, and label this video's words.

    Opt-in per video: the largest GPU cost in the project (design §6), so
    nothing runs this on your behalf.
    """
    wav = wav_for(db, video_id)
    name = diarizer_name(db, diarizer)
    outcome = diarize_video(db, video_id, wav_path=wav, diarizer=load_diarizer(db, name))
    message = outcome.message()

    wanted = embedder_name(db, embedder)
    if wanted:
        embedded = embed_video_speakers(
            db, video_id, wav_path=wav, embedder=load_embedder(db, wanted)
        )
        message = f"{message}; {embedded.message()}"
    return CommandResult(message=message)


def speakers_embed(db: Database, *, video_id: int, embedder: str = "") -> CommandResult:
    """Give this video's voices their fingerprints, for cross-video suggestions.

    Runnable on its own, and cheap: the speech is reconstructed from the
    already-labelled words rather than by diarizing again.
    """
    wanted = embedder_name(db, embedder)
    if not wanted:
        raise RytpError(
            "no embedder configured; pass --embedder or set the "
            f"{C.SETTINGS_EMBEDDER} setting"
        )
    if not store.label_rows(db, video_id):
        raise NotFoundError(f"video {video_id} has no labels; diarize it first")
    outcome = embed_video_speakers(
        db, video_id, wav_path=wav_for(db, video_id), embedder=load_embedder(db, wanted)
    )
    return CommandResult(message=outcome.message())


# -- the queue -------------------------------------------------------------


def speakers_enqueue(db: Database, *, video_id: int, priority: int = 0) -> CommandResult:
    """Ask the worker to diarize this video when it next has the GPU."""
    from rytp.jobs.queue import enqueue, get_job

    job_id = enqueue(db, "diarize", video_id, priority=priority)
    job = get_job(db, job_id)
    return CommandResult(
        message=f"diarize job {job_id} for video {video_id} is {job.state}"
    )


# -- suggestions -----------------------------------------------------------


def speakers_suggest(
    db: Database, *, video_id: int, label: str = "", limit: int = C.SPEAKER_SUGGEST_LIMIT
) -> CommandResult:
    """Who each unnamed voice might be. A suggestion, never an action.

    Nothing here writes. Design §6 is explicit that a decade of changing
    microphones makes automatic linking untrustworthy, so the verdict
    column is advice and the `link` command is the only way to act on it.
    """
    if label:
        row = store.find_label(db, video_id, label)
        grouped = {
            row.video_speaker_id: suggest_for_label(
                db, row.video_speaker_id, limit=limit
            )
        }
    else:
        grouped = suggest_for_video(db, video_id, limit=limit)

    by_id = {row.video_speaker_id: row for row in store.label_rows(db, video_id)}
    rows = tuple(
        (
            by_id[video_speaker_id].local_label,
            item.speaker_label,
            f"{item.similarity:.2f}",
            item.verdict,
            item.era,
            "same era" if item.same_era else "other era",
            "similar sound" if item.acoustically_close else "unlike or unmeasured",
        )
        for video_speaker_id, items in grouped.items()
        for item in items
    )
    return CommandResult(
        columns=("label", "person", "score", "verdict", "era", "when", "sound"),
        rows=rows,
        message=None if rows else _nothing_to_suggest(db),
    )


def _nothing_to_suggest(db: Database) -> str:
    """Say which of the two reasons an empty suggestion list has.

    An empty list because nobody embedded anything and an empty list because
    every candidate came from a different diarizer look identical, and the
    second one would otherwise be a mystery (design §6: nothing is applied
    silently, which also means nothing is *withheld* silently).
    """
    engines = engines_in_play(db)
    if len(engines) > 1:
        return (
            "nothing to suggest: the embedded voices come from more than one "
            f"diarizer ({', '.join(sorted(engines))}) and vectors from different "
            "diarizers are not comparable — re-diarize with one engine"
        )
    return (
        "nothing to suggest: embed this video's voices and at least one "
        "named video (`rytp speakers embed <video-id>`)"
    )


# -- engines and the mapper launcher --------------------------------------


def speakers_engines(db: Database) -> CommandResult:
    """Which diarizers and embedders this machine could actually run."""
    rows = [
        (name, "diarizer", token, out_of_process, state)
        for name, token, out_of_process, state in diarizer_rows(db)
    ]
    rows += [
        (name, "embedder", "no", out_of_process, state)
        for name, out_of_process, state in embedder_rows(db)
    ]
    return CommandResult(
        columns=("name", "kind", "token", "subprocess", "state"), rows=tuple(rows)
    )


def _run_mapper(db: Database, video_id: int) -> None:
    """Open the interactive mapper. A seam, so the test never starts Textual."""
    from rytp.tui.screens.speakers import run_mapper

    run_mapper(db, video_id)


def speakers_map(db: Database, *, video_id: int) -> CommandResult:
    """Open the two-pane mapper for one video (design §10)."""
    if not store.label_rows(db, video_id):
        raise NotFoundError(
            f"video {video_id} has no labels to map; diarize it first "
            f"(`rytp speakers diarize {video_id}`)"
        )
    _run_mapper(db, video_id)
    remaining = sum(1 for row in store.label_rows(db, video_id) if row.speaker_id is None)
    return CommandResult(message=f"{remaining} labels left unnamed")


# -- registration ----------------------------------------------------------

_VIDEO_ID = Param("video_id", int, "Video id.", positional=True)
_LABEL = Param("label", str, "Diarizer label, for example SPEAKER_00.", positional=True)

register(
    Command(
        name="speakers.add",
        group="speakers",
        summary="Add one person to the global roster.",
        params=(
            Param("label", str, "The person's name.", positional=True),
            Param("aliases", str, "Comma-separated alternative names.", default=""),
            Param("notes", str, "Free text.", default=None),
        ),
        handler=speakers_add,
    )
)
register(
    Command(
        name="speakers.list",
        group="speakers",
        summary="Show the global roster.",
        params=(),
        handler=speakers_list,
    )
)
register(
    Command(
        name="speakers.alias",
        group="speakers",
        summary="Add aliases to somebody already on the roster.",
        params=(
            Param("speaker", str, "Roster id, name or alias.", positional=True),
            Param("aliases", str, "Comma-separated names to add.", positional=True),
        ),
        handler=speakers_alias,
    )
)
register(
    Command(
        name="speakers.remove",
        group="speakers",
        summary="Delete a person from the roster. Their labels stay, unnamed.",
        params=(
            Param("speaker", str, "Roster id, name or alias.", positional=True),
            Param("yes", bool, "Confirm when many labels name them.", default=False),
        ),
        handler=speakers_remove,
    )
)
register(
    Command(
        name="speakers.labels",
        group="speakers",
        summary="Show one video's diarizer labels and who they are.",
        params=(_VIDEO_ID,),
        handler=speakers_labels,
    )
)
register(
    Command(
        name="speakers.link",
        group="speakers",
        summary="Say which person one of a video's labels is.",
        params=(
            _VIDEO_ID,
            _LABEL,
            Param("speaker", str, "Roster id, name or alias.", positional=True),
        ),
        handler=speakers_link,
    )
)
register(
    Command(
        name="speakers.unlink",
        group="speakers",
        summary="Detach one label of one video from its person. The roster is kept.",
        params=(_VIDEO_ID, _LABEL),
        handler=speakers_unlink,
    )
)
register(
    Command(
        name="speakers.diarize",
        group="speakers",
        summary="Work out who spoke when in one video. Opt-in; uses the GPU.",
        params=(
            _VIDEO_ID,
            Param("diarizer", str, "Engine name; default from settings.", default=""),
            Param("embedder", str, "Also embed the voices, with this engine.", default=""),
        ),
        handler=speakers_diarize,
        long_running=True,
    )
)
register(
    Command(
        name="speakers.embed",
        group="speakers",
        summary="Fingerprint one video's voices for cross-video suggestions.",
        params=(
            _VIDEO_ID,
            Param("embedder", str, "Engine name; default from settings.", default=""),
        ),
        handler=speakers_embed,
        long_running=True,
    )
)
register(
    Command(
        name="speakers.enqueue",
        group="speakers",
        summary="Queue one video for diarization by the worker.",
        params=(
            _VIDEO_ID,
            Param("priority", int, "Higher runs sooner.", default=0),
        ),
        handler=speakers_enqueue,
    )
)
register(
    Command(
        name="speakers.suggest",
        group="speakers",
        summary="Who each unnamed voice might be. Advice only; changes nothing.",
        params=(
            _VIDEO_ID,
            Param("label", str, "One label only; default every unnamed one.", default=""),
            Param("limit", int, "Candidates per label.", default=C.SPEAKER_SUGGEST_LIMIT),
        ),
        handler=speakers_suggest,
    )
)
register(
    Command(
        name="speakers.engines",
        group="speakers",
        summary="Which diarizers and embedders could run on this machine.",
        params=(),
        handler=speakers_engines,
    )
)
register(
    Command(
        name="speakers.map",
        group="speakers",
        summary="Open the two-pane mapper for one video.",
        params=(_VIDEO_ID,),
        handler=speakers_map,
        cli_only=True,
    )
)
```

- [ ] **Step 4: Add the import to the bottom of `rytp/commands/__init__.py`**

A command module nobody imports registers nothing. Add one line to the
existing block:

```python
from rytp.commands import speakers as _speakers  # noqa: E402,F401
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_commands_speakers.py -v
```
Expected: 32 passed.

- [ ] **Step 6: Run the whole suite — the registry grew, and the surfaces are generated from it**

Run:
```bash
python -m pytest -q
```
Expected: everything passes. Part 1's surface-parity test walks `COMMANDS`, so
thirteen new commands must appear in the Typer app and (all but `speakers.map`)
in the TUI palette without any further work. If the parity test fails, the
registry entry is wrong — not the test.

- [ ] **Step 7: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 8: Commit**

```bash
git add rytp/commands/speakers.py rytp/commands/__init__.py \
  tests/test_commands_speakers.py
git commit -m "feat(speakers): the thirteen speaker commands, one registry, two surfaces"
```

---

### Task 14: The TUI mapper screen

Design §10: "TUI is required for speaker assignment — the one genuinely
interactive task", and the owner named the mapper as the thing he most
needs from the TUI. Everything it does already exists and is already
tested: `MappingSession` from Task 6 and `suggestion_table()` from Task 12.
What is left is a shell — two tables, a filter line, a status line, and
keys wired to the session's verbs. The shell holds **no state**: it reads
`self.session` and writes keystrokes back into it.

Two screens, because the mapper needs a video and the owner should not have
to remember an id: `SpeakerVideosScreen` lists the diarized videos and
`SpeakerMapperScreen` maps one of them. `F3` on the main app opens the
first; `rytp speakers map <video-id>` (Task 13) opens the second directly.

Keys avoid every bare letter, because the filter `Input` has focus by
default and must keep receiving typed text. Typing a name and pressing
`ctrl+n` creates that person and links them in one gesture, which is the
motion this screen exists for.

**Files:**
- Create: `rytp/tui/screens/speakers.py`
- Modify: `rytp/tui/app.py` (one binding, one action)
- Test: `tests/test_tui_speakers.py`

**Interfaces:**
- Consumes: `rytp.diarize.mapper.MappingSession` (Task 6) and its `suggestion_table()` (Task 12), `rytp.diarize.store.diarized_videos`, `rytp.constants.SPEAKER_TITLE_TRUNCATE_CHARS`, Part 1's `RytpApp` with its `_db` attribute and `BINDINGS`.
- Produces: `SpeakerMapperScreen(db, video_id)` with `.session` and the actions `assign`, `unassign`, `new_speaker`, `toggle_suggestions`, `next_label`, `previous_label`; `SpeakerVideosScreen(db)`; `run_mapper(db, video_id) -> None`; `RytpApp.action_speakers()`. `run_mapper` is what `rytp/commands/speakers.py::_run_mapper` (Task 13) imports.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_speakers.py`. It mirrors Part 1's `drive()` helper:
`asyncio.run` around `app.run_test()`, which is core Textual and needs no
plugin. Three or four smoke tests only — everything else about the mapper is
already covered headlessly in `tests/test_speakers_mapper.py`, and a screen
test that re-asserts business rules is a screen test that breaks when the
layout changes.

```python
"""The mapper screen: a shell over MappingSession, driven headlessly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from textual.widgets import DataTable, Input

from rytp.db import Database
from rytp.diarize import store
from rytp.tui.app import RytpApp
from rytp.tui.screens.speakers import SpeakerMapperScreen, SpeakerVideosScreen
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db, title="An Interview")
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[1]: guest})
    store.add_speaker(db, "Host One", aliases=("h1",))
    store.add_speaker(db, "Guest Two", aliases=("g2",))
    return video_id


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    """Run `body` against a mounted RytpApp, headlessly."""

    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_the_mapper_shows_both_panes(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#labels", DataTable).row_count == 2
        assert screen.query_one("#roster", DataTable).row_count == 2

    drive(db, body)


def test_typing_in_the_filter_narrows_the_roster_pane_only(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "h1"
        await pilot.pause()
        assert screen.query_one("#roster", DataTable).row_count == 1
        assert screen.query_one("#labels", DataTable).row_count == 2

    drive(db, body)


def test_assigning_names_the_label_and_says_so(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "Host One"
        await pilot.pause()
        screen.action_assign()
        await pilot.pause()
        assert "Host One" in screen.session.status
        assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"

    drive(db, body)


def test_typing_a_new_name_and_creating_links_in_one_gesture(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "Каспар Хаузер"
        await pilot.pause()
        screen.action_new_speaker()
        await pilot.pause()
        assert (
            store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Каспар Хаузер"
        )

    drive(db, body)


def test_the_suggestion_pane_is_hidden_until_asked_for(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        suggestions = screen.query_one("#suggestions", DataTable)
        assert suggestions.display is False
        screen.action_toggle_suggestions()
        await pilot.pause()
        assert suggestions.display is True

    drive(db, body)


def test_the_video_chooser_lists_diarized_videos(db: Database) -> None:
    a_diarized_video(db)
    make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerVideosScreen(db)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#videos", DataTable).row_count == 1

    drive(db, body)


def test_f3_opens_the_speakers_screen(db: Database) -> None:
    a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f3")
        await pilot.pause()
        assert isinstance(app.screen, SpeakerVideosScreen)

    drive(db, body)


def test_a_video_with_no_labels_still_mounts(db: Database) -> None:
    video_id = make_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#labels", DataTable).row_count == 0
        screen.action_assign()
        await pilot.pause()
        assert "no label" in screen.session.status

    drive(db, body)


def test_importing_the_screen_creates_no_directories(tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.test_config import child_env

    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.tui.screens.speakers"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_tui_speakers.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.screens.speakers'`.

- [ ] **Step 3: Write `rytp/tui/screens/speakers.py`**

```python
"""The speaker mapper screen (design §10).

Design §10 calls speaker assignment "the one genuinely interactive task",
and this is the screen for it. It is deliberately thin: every decision
lives in :class:`~rytp.diarize.mapper.MappingSession`, which is tested
without a terminal, and this file only draws that object and forwards
keystrokes into it. If you find yourself adding an `if` here, it probably
belongs in the session.

Two screens. :class:`SpeakerVideosScreen` lists the videos that have
diarizer labels, because nobody remembers a video id; picking one opens
:class:`SpeakerMapperScreen`, which is the two panes.

**Key choices.** The filter `Input` holds focus, so every binding avoids a
bare letter — typing must reach the filter. The motion the screen exists
for is: type part of a name, press enter to link, and the cursor moves to
the next unnamed voice. Typing a name nobody has and pressing ctrl+n
creates that person and links them in the same gesture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.diarize import store
from rytp.diarize.mapper import MappingSession

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["SpeakerMapperScreen", "SpeakerVideosScreen", "run_mapper"]


def _fill(table: DataTable, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Replace a table's contents. Columns are re-added: they can change."""
    table.clear(columns=True)
    table.add_columns(*columns)
    for row in rows:
        table.add_row(*row)


class SpeakerMapperScreen(Screen[None]):
    """Local labels on the left, the global roster on the right."""

    DEFAULT_CSS = """
    #panes { height: 1fr; }
    #labels { width: 1fr; }
    #roster { width: 1fr; }
    #suggestions { height: auto; max-height: 10; }
    #mapper-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("enter", "assign", "Name this voice"),
        Binding("ctrl+n", "new_speaker", "New person from filter"),
        Binding("ctrl+u", "unassign", "Unname"),
        Binding("f5", "toggle_suggestions", "Suggestions"),
        Binding("ctrl+down", "next_label", "Next voice"),
        Binding("ctrl+up", "previous_label", "Previous voice"),
    ]

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self.session = MappingSession(db, video_id)
        self._syncing = False

    # -- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter the roster: name or alias", id="roster-filter")
        with Horizontal(id="panes"):
            yield DataTable(id="labels")
            yield DataTable(id="roster")
        yield DataTable(id="suggestions")
        yield Static("", id="mapper-status")
        yield Footer()

    def on_mount(self) -> None:
        for table_id in ("#labels", "#roster", "#suggestions"):
            self.query_one(table_id, DataTable).cursor_type = "row"
        self.query_one("#suggestions", DataTable).display = False
        self.refresh_panes()
        self.query_one("#roster-filter", Input).focus()

    # -- drawing ---------------------------------------------------------

    def refresh_panes(self) -> None:
        """Redraw both tables from the session, cursors included."""
        self._syncing = True
        try:
            labels = self.query_one("#labels", DataTable)
            _fill(labels, *self.session.label_table())
            if self.session.labels:
                labels.move_cursor(row=self.session.label_index)

            roster = self.query_one("#roster", DataTable)
            _fill(roster, *self.session.roster_table())
            if self.session.matches:
                roster.move_cursor(row=self.session.roster_index)
        finally:
            self._syncing = False
        self._show_status()

    def _show_status(self, message: str | None = None) -> None:
        text = message if message is not None else self.session.status
        remaining = self.session.n_unmapped()
        suffix = f"{remaining} voice{'' if remaining == 1 else 's'} still unnamed"
        self.query_one("#mapper-status", Static).update(
            f"{text} — {suffix}" if text else suffix
        )

    # -- events ----------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "roster-filter":
            self.session.set_query(event.value)
            self.refresh_panes()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "roster-filter":
            self.action_assign()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Let the mouse and arrow keys move the session's cursors too."""
        if self._syncing:
            return
        if event.data_table.id == "labels":
            self.session.select_label(event.cursor_row)
            if self.query_one("#suggestions", DataTable).display:
                self._fill_suggestions()
        elif event.data_table.id == "roster":
            self.session.select_roster(event.cursor_row)

    # -- actions ---------------------------------------------------------

    def action_assign(self) -> None:
        message = self.session.assign()
        self.refresh_panes()
        self._show_status(message)

    def action_unassign(self) -> None:
        message = self.session.unassign()
        self.refresh_panes()
        self._show_status(message)

    def action_new_speaker(self) -> None:
        """Create the person named in the filter box and link them."""
        text = self.query_one("#roster-filter", Input).value.strip()
        if not text:
            self._show_status("type a name in the filter box first")
            return
        message = self.session.create_and_assign(text)
        self.query_one("#roster-filter", Input).value = ""
        self.session.set_query("")
        self.refresh_panes()
        self._show_status(message)

    def action_next_label(self) -> None:
        self.session.move_label(1)
        self.refresh_panes()

    def action_previous_label(self) -> None:
        self.session.move_label(-1)
        self.refresh_panes()

    def action_toggle_suggestions(self) -> None:
        """Show or hide the advisory pane. Showing it changes nothing."""
        table = self.query_one("#suggestions", DataTable)
        table.display = not table.display
        if table.display:
            self._fill_suggestions()

    def _fill_suggestions(self) -> None:
        self._syncing = True
        try:
            _fill(
                self.query_one("#suggestions", DataTable),
                *self.session.suggestion_table(),
            )
        finally:
            self._syncing = False


class SpeakerVideosScreen(Screen[None]):
    """The videos that have labels, so nobody has to remember an id."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._video_ids: list[int] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="videos")
        yield Static("", id="videos-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#videos", DataTable)
        table.cursor_type = "row"
        rows = store.diarized_videos(self._db)
        self._video_ids = [row.video_id for row in rows]
        _fill(
            table,
            ("id", "title", "published", "voices", "unnamed"),
            [
                (
                    str(row.video_id),
                    (row.title or "")[: C.SPEAKER_TITLE_TRUNCATE_CHARS],
                    (row.published_at or "—")[:10],
                    str(row.n_labels),
                    str(row.n_unmapped),
                )
                for row in rows
            ],
        )
        self.query_one("#videos-status", Static).update(
            f"{len(rows)} diarized video{'' if len(rows) == 1 else 's'}"
            if rows
            else "nothing diarized yet — `rytp speakers diarize <video-id>`"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "videos" or not self._video_ids:
            return
        self.app.push_screen(
            SpeakerMapperScreen(self._db, self._video_ids[event.cursor_row])
        )


class _MapperApp(App[None]):
    """The one-screen app `rytp speakers map <video-id>` launches."""

    TITLE = "rytp — speakers"

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self._db = db
        self._video_id = video_id

    def on_mount(self) -> None:
        self.push_screen(SpeakerMapperScreen(self._db, self._video_id))


def run_mapper(db: Database, video_id: int) -> None:
    """Open the mapper for one video. Called only by `rytp speakers map`."""
    _MapperApp(db, video_id).run()
```

Note what is *not* here: no formatting, no counting, no decision about which
label comes next. `MappingSession.label_table()` already returns strings and
`MappingSession._after_assign` already moves the cursor. The screen's whole
job is `_fill` plus six one-line actions, which is why the nine tests below
are enough for it.

- [ ] **Step 4: Add the binding to `rytp/tui/app.py`**

Two edits to Part 1's file. In `BINDINGS`:

```python
        Binding("f3", "speakers", "Speakers"),
```

and one action, with the import inside the body so the main app never imports
the diarize package just to start:

```python
    def action_speakers(self) -> None:
        """Open the speaker mapper (design §10: the one interactive task)."""
        from rytp.tui.screens.speakers import SpeakerVideosScreen

        self.push_screen(SpeakerVideosScreen(self._db))
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_tui_speakers.py -v
```
Expected: 9 passed.

If `await app.push_screen(screen)` complains that it is not awaitable, the
installed Textual returns the screen rather than an awaitable — call it
without `await` and follow with `await pilot.pause()`. That is the only
version-sensitive line in the file.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add rytp/tui/screens/speakers.py rytp/tui/app.py tests/test_tui_speakers.py
git commit -m "feat(tui): the two-pane speaker mapper screen"
```

---

### Task 15: Health checks for `doctor`

`doctor` is one command owned by Part 1, and each part contributes checks
rather than Part 1 knowing about them (contracts §5). Part 7 answers two
questions, both cheaply and neither by importing a model: **which diarizers
could run here**, and **is `HF_TOKEN` set**.

The token check is the one that earns its place. pyannote's
`speaker-diarization-community-1` is a gated model. Without a token the
failure arrives deep inside pipeline construction, after a download has
begun, as an HTTP error naming nothing useful — the exact shape of problem
a `doctor` check exists to pre-empt. So the remedy names the page to visit
and the variable to set, in that order, because accepting the terms is the
step people skip.

**Neither check imports an engine and neither downloads anything.**
`importlib.util.find_spec` answers "is this installed" without executing a
module, and an out-of-process engine's dependency is not in this
interpreter at all, so its check is whether the configured interpreter
exists. Both reuse Part 3's `availability`, which already does exactly that.

**Both are `required=False`, and both tell the truth anyway.** Contracts §5
keeps those two ideas apart on purpose: `ok` says what was actually found,
and `required` says whether a false `ok` should fail the command. So a
missing pyannote is `ok=False` — it *is* missing — and an unset `HF_TOKEN`
is `ok=False` with a remedy, and neither makes `doctor` exit non-zero,
because the null diarizer always works and nobody has to use pyannote. The
tempting alternative — returning `ok=True` to protect the exit code — makes
the output lie about the one thing the command exists to tell you.

**Files:**
- Create: `rytp/diarize/health.py`
- Modify: `rytp/commands/speakers.py` (one import at the bottom, so the checks register)
- Test: `tests/test_diarize_health.py`

**Interfaces:**
- Consumes: Part 1's `HealthCheck` (including its `required: bool = True` field), `HealthResult`, `register_check`; `rytp.diarize.base.DIARIZERS`; `rytp.transcribe.registry.{availability, setting}`; `rytp.constants.{SETTINGS_DIARIZER, DEFAULT_DIARIZER, PYANNOTE_DIARIZATION_MODEL, HF_TOKEN_ENV_VARS}`.
- Produces: `usable(state) -> bool`, `configured_diarizer(db) -> str`, `hf_token() -> str | None`, `check_diarizers(db) -> HealthResult`, `check_hf_token(db) -> HealthResult`, and the two registered checks `diarizers` and `hf-token`, **both `required=False`**.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_health.py`:

```python
"""What `doctor` says about speakers. No imports of engines, no network."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize.health import (
    check_diarizers,
    check_hf_token,
    configured_diarizer,
    usable,
)
from tests.fake_speaker_engines import GatedDiarizer, registered


@pytest.fixture(autouse=True)
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's own token decide a test's outcome."""
    for name in C.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def set_diarizer(db: Database, name: str) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (C.SETTINGS_DIARIZER, name),
    )


# -- the registry ----------------------------------------------------------


def test_both_checks_are_registered() -> None:
    import rytp.commands.speakers  # noqa: F401 - importing registers them

    from rytp.commands import HEALTH_CHECKS

    assert {"diarizers", "hf-token"} <= set(HEALTH_CHECKS)


def test_both_checks_are_advisory() -> None:
    # contracts §5: `ok` tells the truth, `required` decides the exit code.
    # A base install has no pyannote and no token, and `doctor` must still
    # exit zero — the null diarizer needs neither.
    import rytp.commands.speakers  # noqa: F401

    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["diarizers"].required is False
    assert HEALTH_CHECKS["hf-token"].required is False


def test_usable_reads_part_threes_availability_words() -> None:
    assert usable("ready") is True
    assert usable("interpreter ok") is True
    assert usable("needs pyannote.audio") is False
    assert usable("no HF_TOKEN") is False


def test_the_configured_diarizer_is_the_setting_then_the_default(
    db: Database,
) -> None:
    assert configured_diarizer(db) == C.DEFAULT_DIARIZER
    set_diarizer(db, "pyannote")
    assert configured_diarizer(db) == "pyannote"


# -- the diarizer check ----------------------------------------------------


def test_a_missing_engine_is_reported_as_missing(db: Database) -> None:
    # The dev environment has no pyannote, so the honest answer is False.
    # `required=False` is what keeps that from failing `doctor`.
    result = check_diarizers(db)
    assert result.ok is False
    assert "pyannote" in result.detail
    assert "none" in result.detail


def test_the_remedy_says_nothing_is_broken_when_the_configured_one_works(
    db: Database,
) -> None:
    result = check_diarizers(db)
    assert "nothing is broken" in (result.remedy or "")
    assert "rytp[pyannote]" in (result.remedy or "")


def test_the_remedy_leads_with_the_configured_engine_when_that_is_the_gap(
    db: Database,
) -> None:
    # This machine is set up to use pyannote and cannot. Still advisory,
    # but the remedy should be about pyannote, not about the others.
    set_diarizer(db, "pyannote")
    result = check_diarizers(db)
    assert result.ok is False
    remedy = result.remedy or ""
    assert remedy.startswith("the configured diarizer 'pyannote'")
    assert C.SETTINGS_DIARIZER in remedy


def test_everything_usable_is_the_only_way_to_pass(db: Database) -> None:
    from rytp.diarize import health

    monkey = {"none": "ready", "pyannote": "interpreter ok"}
    original = health._states
    health._states = lambda _db: monkey          # type: ignore[assignment]
    try:
        result = check_diarizers(db)
    finally:
        health._states = original                # type: ignore[assignment]
    assert result.ok is True
    assert result.remedy is None


def test_a_configured_engine_that_is_not_registered_at_all_is_a_failure(
    db: Database,
) -> None:
    set_diarizer(db, "nonesuch")
    result = check_diarizers(db)
    assert result.ok is False
    assert "nonesuch" in result.detail


def test_the_check_never_raises_on_a_broken_registry(db: Database) -> None:
    class Exploding:
        name = "fake-exploding"
        requires_hf_token = False
        out_of_process = False

        @property
        def required_module(self) -> str:
            raise RuntimeError("engines must not be able to break doctor")

        def diarize(self, audio: object) -> list[object]:
            return []

    with registered(Exploding):
        result = check_diarizers(db)
    assert isinstance(result.ok, bool)


# -- the token check -------------------------------------------------------


def test_a_missing_token_is_reported_as_missing(db: Database) -> None:
    # Honest, and advisory: `required=False` keeps it out of the exit code.
    result = check_hf_token(db)
    assert result.ok is False
    assert "not set" in result.detail
    assert C.PYANNOTE_DIARIZATION_MODEL in (result.remedy or "")
    assert "HF_TOKEN" in (result.remedy or "")


def test_the_detail_says_when_nothing_configured_needs_the_token(
    db: Database,
) -> None:
    assert "nothing configured needs it yet" in check_hf_token(db).detail


def test_that_reassurance_disappears_once_pyannote_is_configured(
    db: Database,
) -> None:
    set_diarizer(db, "pyannote")
    assert "nothing configured needs it yet" not in check_hf_token(db).detail


def test_a_present_token_is_reported_without_being_printed(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_thisisasecret")
    result = check_hf_token(db)
    assert result.ok is True
    assert "hf_thisisasecret" not in result.detail
    assert "HF_TOKEN" in result.detail


def test_the_alternative_variable_name_counts_too(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "t")
    assert check_hf_token(db).ok is True


def test_the_token_check_names_which_engines_are_gated(db: Database) -> None:
    with registered(GatedDiarizer):
        result = check_hf_token(db)
    assert "fake-gated-diarizer" in result.detail
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_diarize_health.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.diarize.health'`.

- [ ] **Step 3: Write `rytp/diarize/health.py`**

```python
"""What `doctor` reports about speakers (contracts §5).

Two questions, answered without importing an engine, constructing a
pipeline or touching the network: which diarizers could run on this
machine, and whether the Hugging Face token the gated one needs is set.

The second is the one that earns its place. pyannote's community-1 is a
gated model, and without a token the failure arrives deep inside pipeline
construction — after a download has begun — as an HTTP error that names
nothing useful. Two people will hit that and conclude the tool is broken.
Saying it up front, with the page to accept the terms on and the variable
to set, is what this check is for.

Neither check raises and neither is fatal by default. The null diarizer has
no dependencies and is the default, so a missing optional engine is a note.
`ok` goes false only when the diarizer this machine is *configured* to use
cannot run here — a real breakage on this machine, and the only thing in
this file worth a non-zero exit from `doctor`.
"""

from __future__ import annotations

import os

from rytp import constants as C
from rytp.commands import HealthCheck, HealthResult, register_check
from rytp.db import Database

#: The two words Part 3's `availability` uses for "this could run now".
#: Everything else it returns is a reason it could not.
_USABLE_STATES = frozenset({"ready", "interpreter ok"})


def usable(state: str) -> bool:
    """Whether one of `availability`'s answers means the engine can run."""
    return state in _USABLE_STATES


def hf_token() -> str | None:
    """The Hugging Face token, under either accepted name. Never logged."""
    for name in C.HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def configured_diarizer(db: Database) -> str:
    """Which diarizer this machine would use if asked right now.

    The same precedence `rytp.diarize.pipeline.diarizer_name` applies, read
    straight from settings so that importing this module does not drag the
    pipeline — and with it the store — into every `doctor` run.
    """
    from rytp.transcribe.registry import setting

    return setting(db, C.SETTINGS_DIARIZER) or C.DEFAULT_DIARIZER


def _states(db: Database) -> dict[str, str]:
    """Each registered diarizer's availability, tolerating a broken engine.

    An engine class is third-party-shaped code; a check that a badly
    written one can crash is a check that makes `doctor` useless exactly
    when it is needed.
    """
    from rytp.diarize.base import DIARIZERS
    from rytp.transcribe.registry import availability

    out: dict[str, str] = {}
    for name, cls in sorted(DIARIZERS.items()):
        try:
            out[name] = availability(db, cls)
        except Exception as exc:  # noqa: BLE001 - a check never raises
            out[name] = f"unusable: {type(exc).__name__}"
    return out


def check_diarizers(db: Database) -> HealthResult:
    """Which diarizers could run here, and whether the configured one can.

    `ok` is the plain truth — false when any registered diarizer cannot run,
    because one of them cannot. The check is registered `required=False`, so
    that is a note rather than a failure: `none` needs nothing and is the
    default, and nobody is obliged to install pyannote.
    """
    states = _states(db)
    ready = [name for name, state in states.items() if usable(state)]
    missing = {name: state for name, state in states.items() if not usable(state)}
    wanted = configured_diarizer(db)

    detail = "usable: " + (", ".join(ready) or "none")
    if missing:
        detail += "; unusable: " + ", ".join(
            f"{name} ({state})" for name, state in missing.items()
        )
    detail += f"; configured: {wanted}"

    if wanted not in states:
        return HealthResult(
            ok=False,
            detail=f"{detail}; {wanted!r} is not a registered diarizer",
            remedy=(
                f"set the {C.SETTINGS_DIARIZER} setting to one of: "
                f"{', '.join(sorted(states)) or '(none)'}"
            ),
        )
    if not missing:
        return HealthResult(ok=True, detail=detail)

    # Lead with the engine this machine is actually set up to use: that is
    # the one whose absence will bite today.
    extras = ", ".join(f"pip install rytp[{name}]" for name in sorted(missing))
    if wanted in missing:
        remedy = (
            f"the configured diarizer {wanted!r} cannot run here: "
            f"pip install rytp[{wanted}] into its own environment, point "
            f"engine.interpreter.{wanted} at that python, or set "
            f"{C.SETTINGS_DIARIZER} to 'none'"
        )
    else:
        remedy = f"{wanted!r} works, so nothing is broken; for the others: {extras}"
    return HealthResult(ok=False, detail=detail, remedy=remedy)


def check_hf_token(db: Database) -> HealthResult:
    """Whether the token the gated diarizer needs is set.

    `ok` is simply whether a token exists — not whether anybody currently
    needs one. The check is `required=False`, so an absent token is a note
    with a remedy rather than a failed `doctor`, and somebody who never
    intends to touch pyannote can ignore it forever.

    Reports that a token *exists*, never what it is: `doctor` output is the
    first thing anybody pastes into a bug report.
    """
    from rytp.diarize.base import DIARIZERS

    gated = sorted(
        name
        for name, cls in DIARIZERS.items()
        if getattr(cls, "requires_hf_token", False)
    )
    variables = " or ".join(C.HF_TOKEN_ENV_VARS)

    if hf_token() is not None:
        return HealthResult(ok=True, detail=f"{variables} is set")

    detail = f"{variables} is not set"
    if gated:
        detail += f"; gated diarizers: {', '.join(gated)}"
    if configured_diarizer(db) not in gated:
        detail += "; nothing configured needs it yet"
    return HealthResult(
        ok=False,
        detail=detail,
        remedy=(
            f"open https://huggingface.co/{C.PYANNOTE_DIARIZATION_MODEL}, accept the "
            f"model terms, then set HF_TOKEN to a token from "
            f"https://huggingface.co/settings/tokens"
        ),
    )


# Both advisory (contracts §5). The null diarizer needs nothing and is the
# default, and nobody is obliged to use a gated model — so a false `ok` here
# is information, never a reason for `doctor` to exit non-zero.
register_check(
    HealthCheck(
        name="diarizers",
        summary="Which diarizers can run on this machine.",
        run=check_diarizers,
        required=False,
    )
)
register_check(
    HealthCheck(
        name="hf-token",
        summary="Whether the Hugging Face token the gated diarizer needs is set.",
        run=check_hf_token,
        required=False,
    )
)
```

- [ ] **Step 4: Import it from `rytp/commands/speakers.py`**

A check nobody imports is registered nowhere, exactly like a command. One
line at the very bottom of the file, after the `register(...)` block:

```python
# Importing this registers Part 7's `doctor` checks (contracts §5). It sits
# here rather than in `rytp/diarize/__init__.py` because the engine package
# is imported inside a foreign interpreter, which has no `rytp.commands`.
from rytp.diarize import health as _health  # noqa: E402,F401
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_diarize_health.py -v
```
Expected: 16 passed.

- [ ] **Step 6: Run `doctor` for real and read it**

```bash
python -m rytp doctor
```
Expected: two speaker rows, both marked advisory, and on a base install
both reporting a miss — no pyannote, no token — with a remedy each, while
`doctor` still exits zero. Check that exit code: `echo $?` (PowerShell:
`$LASTEXITCODE`). Then:

```bash
python -m rytp settings set speakers.diarizer pyannote
python -m rytp doctor
```
Expected: the same two rows, with remedies now aimed at pyannote — and
still a zero exit, because neither check is required. Set the setting back
to `none` afterwards. (If Part 1 named the settings command something
else, use that — the point is to see a real failure once.)

- [ ] **Step 7: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 8: Commit**

```bash
git add rytp/diarize/health.py rytp/commands/speakers.py tests/test_diarize_health.py
git commit -m "feat(speakers): doctor checks for diarizer availability and HF_TOKEN"
```

---

### Task 16: Integration, lint, type-check, full suite

One test that walks the whole part the way the owner will: a transcribed
video arrives, somebody asks for diarization, the voices get named, a
second video of the same person is suggested rather than linked, and a
search by that person finds the words. Everything faked, nothing
downloaded.

**Files:**
- Test: `tests/test_speakers_integration.py`

**Interfaces:**
- Consumes: every task above. Produces nothing new.

- [ ] **Step 1: Write the integration test**

Create `tests/test_speakers_integration.py`:

```python
"""One video, transcript to named speakers, with nothing heavy installed."""

from __future__ import annotations

import wave
from pathlib import Path

import rytp.commands.speakers  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.config import paths
from rytp.db import Database
from rytp.diarize import store
from rytp.diarize.link import MATCH
from rytp.jobs import JOB_HANDLERS, JOB_KINDS
from rytp.jobs import queue as Q
from tests.fake_speaker_engines import ConstantEmbedder, FakeDiarizer, registered
from tests.fakes import make_video
from tests.test_speakers_labels import add_words


class LongDiarizer(FakeDiarizer):
    """Two voices, each with more speech than the embedding floor.

    The default fake script gives each label under a second, which is below
    `C.EMBED_MIN_SPEECH_MS` — correct behaviour, and useless for testing
    suggestions, because nothing would be embedded and the assertion would
    pass on an empty result.
    """

    name = "fake-long"
    script = ((0, 5_000, "SPEAKER_00"), (5_000, 8_000, "SPEAKER_01"))


def cache_a_wav(video_id: int, ms: int = 20_000) -> Path:
    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x01\x00" * (16 * ms))
    return path


def a_transcribed_video(db: Database, external_id: str, published_at: str) -> int:
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    add_words(db, video_id, [(0, 900), (1_000, 1_900), (2_000, 2_900)])
    cache_a_wav(video_id)
    return video_id


def a_long_video(db: Database, external_id: str, published_at: str) -> int:
    """Same, with words long enough that :class:`LongDiarizer` gives each
    label more than `C.EMBED_MIN_SPEECH_MS` of speech to embed."""
    video_id = make_video(
        db,
        external_id=external_id,
        url=f"https://example.invalid/{external_id}",
        published_at=published_at,
    )
    add_words(db, video_id, [(0, 2_500), (2_500, 5_000), (5_000, 8_000)])
    cache_a_wav(video_id)
    return video_id


def test_from_a_transcript_to_named_speakers_and_back(db: Database, data_dir: Path) -> None:
    first = a_transcribed_video(db, "VIDEO_A", "2020-01-01")

    # Nothing has been diarized, so nothing is queued for the GPU.
    assert Q.list_jobs(db) == []

    # The owner opts one video in, by hand.
    with registered(FakeDiarizer, ConstantEmbedder):
        resolve("speakers.diarize").handler(
            db,
            video_id=first,
            diarizer="fake-diarizer",
            embedder="fake-constant-embedder",
        )

    # Two citable voices, both nameless, both with the words to prove it,
    # and both recording which engine drew them (contracts §3).
    labels = store.label_rows(db, first)
    assert [row.local_label for row in labels] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(row.speaker_id is None for row in labels)
    assert all(row.engine == "fake-diarizer" for row in labels)
    assert sum(row.n_words for row in labels) == 3

    # Naming one is one row, and the words are untouched.
    resolve("speakers.add").handler(db, label="Host One", aliases="h1", notes=None)
    words_before = db.conn.execute(
        "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
    ).fetchone()[0]
    resolve("speakers.link").handler(
        db, video_id=first, label="SPEAKER_00", speaker="h1"
    )
    assert (
        db.conn.execute(
            "SELECT group_concat(id || ':' || COALESCE(video_speaker_id, 'x')) FROM words"
        ).fetchone()[0]
        == words_before
    )

    # The forward index now answers "who said this word".
    row = db.conn.execute(
        """
        SELECT s.label FROM words w
        JOIN video_speakers vs ON vs.id = w.video_speaker_id
        JOIN speakers s        ON s.id = vs.speaker_id
        WHERE w.video_id = ? ORDER BY w.ord LIMIT 1
        """,
        (first,),
    ).fetchone()
    assert row["label"] == "Host One"

    # And the backward index answers "where did this person say it".
    count = db.conn.execute(
        """
        SELECT COUNT(*) FROM words w
        JOIN video_speakers vs ON vs.id = w.video_speaker_id
        WHERE vs.speaker_id = (SELECT id FROM speakers WHERE label = 'Host One')
          AND w.normalized_text = 'w0'
        """
    ).fetchone()[0]
    assert count == 1


def test_a_second_video_is_suggested_never_linked(db: Database, data_dir: Path) -> None:
    first = a_long_video(db, "VIDEO_A", "2020-01-01")
    second = a_long_video(db, "VIDEO_B", "2020-06-01")
    with registered(LongDiarizer, ConstantEmbedder):
        for video_id in (first, second):
            resolve("speakers.diarize").handler(
                db,
                video_id=video_id,
                diarizer="fake-long",
                embedder="fake-constant-embedder",
            )
    # Both videos really did get vectors — otherwise the assertions below
    # would pass on an empty result and prove nothing.
    assert all(row.has_embedding for row in store.label_rows(db, second))

    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    resolve("speakers.link").handler(
        db, video_id=first, label="SPEAKER_00", speaker="Host One"
    )

    before = db.conn.total_changes
    result = resolve("speakers.suggest").handler(db, video_id=second, label="", limit=5)
    assert db.conn.total_changes == before, "suggestions must never write"

    # Identical vectors in the same era: as confident as this ever gets.
    assert result.rows
    assert {row[3] for row in result.rows} == {MATCH}
    # And still nobody is linked. That is the whole point.
    assert store.find_label(db, second, "SPEAKER_00").speaker_id is None
    assert store.find_label(db, second, "SPEAKER_01").speaker_id is None


def test_the_queue_path_reaches_the_same_place(db: Database, data_dir: Path) -> None:
    video_id = a_transcribed_video(db, "VIDEO_A", "2020-01-01")
    resolve("speakers.enqueue").handler(db, video_id=video_id, priority=0)
    job = Q.list_jobs(db)[0]
    assert (job.kind, job.pool, job.state) == ("diarize", "gpu", "pending")
    with registered(FakeDiarizer):
        JOB_HANDLERS["diarize"](db, job.target_id, {"diarizer": "fake-diarizer"})
    assert len(store.label_rows(db, video_id)) == 2
    # And the finished job is never re-derived behind the owner's back.
    assert JOB_KINDS["diarize"].reopenable is False


def test_re_transcribing_wipes_the_mapping_and_that_is_fine(
    db: Database, data_dir: Path
) -> None:
    # Contracts §4: whatever replaces a video's words deletes its
    # video_speakers rows in the same transaction. Part 7 assumes that and
    # adds no preservation mechanism (design §3, "erase and replace").
    video_id = a_transcribed_video(db, "VIDEO_A", "2020-01-01")
    with registered(FakeDiarizer):
        resolve("speakers.diarize").handler(
            db, video_id=video_id, diarizer="fake-diarizer", embedder=""
        )
    resolve("speakers.add").handler(db, label="Host One", aliases="", notes=None)
    resolve("speakers.link").handler(
        db, video_id=video_id, label="SPEAKER_00", speaker="Host One"
    )

    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))

    assert store.label_rows(db, video_id) == []
    # The roster survives — re-mapping two to five labels is the whole price.
    assert store.find_speaker(db, "Host One") is not None


def test_every_speakers_command_reaches_both_surfaces(db: Database) -> None:
    from rytp.cli import build_app, command_paths
    from rytp.commands import COMMANDS
    from rytp.tui.palette import palette_entries

    registered_names = {name for name in COMMANDS if name.startswith("speakers.")}
    cli = command_paths(build_app())
    assert registered_names <= cli

    palette = {entry.name for entry in palette_entries()}
    # Every one but the launcher, which cannot be launched from inside the
    # surface it launches (contracts §5, `cli_only`).
    assert registered_names - {"speakers.map"} <= palette
    assert "speakers.map" not in palette


def test_the_group_can_undo_everything_it_creates(db: Database) -> None:
    # contracts §5: no entity you can create but not get rid of.
    from rytp.commands import COMMANDS

    assert "speakers.remove" in COMMANDS


def test_doctor_reports_on_speakers_without_raising(db: Database, data_dir: Path) -> None:
    # contracts §5: a check never raises and a missing optional engine is
    # reported, not fatal.
    from rytp.commands import HEALTH_CHECKS

    for name in ("diarizers", "hf-token"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail
```

- [ ] **Step 2: Run the integration test**

Run:
```bash
python -m pytest tests/test_speakers_integration.py -v
```
Expected: 7 passed.

- [ ] **Step 3: Run the whole suite**

Run:
```bash
python -m pytest -q
```
Expected: everything passes, with no optional extra installed.

- [ ] **Step 4: Prove the suite needs no model and no network**

The dev venv has no ML libraries at all, so a pass there is already the
proof — but make it explicit, because it is the constraint most easily lost:

```bash
python -c "import importlib.util as u; assert all(u.find_spec(m) is None for m in ('torch', 'pyannote', 'redimnet')); print('no ML libraries, as intended')"
```
Expected: `no ML libraries, as intended`.

- [ ] **Step 5: Lint and type-check the whole package**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 6: Check the import graph one last time**

Three properties this part keeps, each a one-liner:

```bash
python -c "import sys, rytp.diarize; assert 'torch' not in sys.modules and 'pyannote' not in sys.modules; print('engines lazy')"
python -c "import sys, rytp.jobs; assert 'rytp.diarize.pipeline' not in sys.modules; print('jobs light')"
python -c "import sys, rytp.tui.screens.speakers; assert 'rytp.commands' not in sys.modules; print('tui does not import commands')"
```
Expected: three lines, all printed.

- [ ] **Step 7: Run the real thing once, against a real file**

A green suite says almost nothing here: every engine is faked. Do this on the
Windows machine with a real video already transcribed, and expect to fix
something in `child_main`:

```
python -m rytp speakers engines
python -m rytp speakers diarize <video-id> --diarizer none
python -m rytp speakers labels <video-id>
python -m rytp speakers add "<a real name>"
python -m rytp speakers link <video-id> SPEAKER_00 "<a real name>"
python -m rytp speakers map <video-id>
```

With `HF_TOKEN` set and a pyannote environment configured:

```
python -m rytp speakers diarize <video-id> --diarizer pyannote --embedder redimnet
python -m rytp speakers suggest <video-id>
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_speakers_integration.py
git commit -m "test(speakers): an end-to-end pass from transcript to named voices"
```

---

## Self-review notes for the executor

Seven things are worth reading before you start, because they are where this
plan is most likely to be wrong in practice.

1. **The thresholds are guesses.** `SPEAKER_MATCH_HI`, `SPEAKER_MATCH_LO`,
   `SPEAKER_MATCH_HI_CROSS_ERA` and every entry of `ACOUSTIC_FEATURE_SCALES`
   are marked UNVALIDATED in `constants.py` and they mean it: nobody has
   measured this corpus. The *shape* — two thresholds, a middle band for the
   human, a stricter bar across eras — is what design §6 commits to, and the
   tests pin the shape rather than the numbers. Re-tune the numbers once real
   suggestions have been accepted and rejected for a while; the tests will
   still pass.

2. **Two engine adapters are written against documented APIs nobody here has
   run.** `rytp.diarize.pyannote.child_main` and
   `rytp.diarize.embed.child_main` are the whole of that exposure. Their
   *contracts* — the dictionaries they return — are covered by tests, and
   every other line of this part depends on the contract rather than the
   library. Expect to rewrite one function body per engine and nothing else.

3. **Provenance is recorded twice and compared once.** `words.engine`
   (Part 3) says which transcriber produced a word; `video_speakers.engine`
   says which diarizer produced a label. Design §11 requires both: no stage
   may assume a particular engine, so a corpus built with more than one has
   to stay interpretable. The consequence that is easy to miss is in
   `link.py` — labels from different diarizers are never compared, because
   their vectors were measured over different turn boundaries, and
   `engines_in_play` exists so an empty suggestion list can say so.

4. **`words.video_speaker_id` is written exactly once, and linking never
   touches it.** Those are two separate rules and it is easy to conflate
   them. `stamp_words` (Task 5) is the only writer of the column;
   `link_video_speaker` is the only writer of `video_speakers.speaker_id` and
   changes exactly one row, asserted by
   `test_linking_a_label_to_a_person_changes_exactly_one_row`. If a future
   change makes that test report two changes, something has re-introduced the
   back-fill the old schema needed.

5. **The utterance rebuild is the quiet dependency.** Design §7's speaker
   filter searches `utterances`, not `words`, so Task 7's
   `request_reindex` is what actually makes "find where *this person* said
   this" work. It needs Part 4's `index` handler to delete and rebuild a
   video's utterances, and Part 4's `index` readiness to answer READY when a
   video has words and no utterances. If Part 4 does either differently, this
   is the seam to renegotiate — not something to work around here.

6. **One forward reference, in two places.** Task 2 writes
   `rytp/diarize/__init__.py` without `pyannote` and weakens one assertion in
   `tests/test_diarize_none.py`, because `rytp/diarize/pyannote.py` does not
   exist until Task 11. Task 11, Steps 4 and 5 restore both. If you execute
   the tasks out of order, that is the pair to reconcile.

7. **`doctor`'s `ok` is a fact; `required` is the policy.** Contracts §5
   separates them deliberately, and Part 7's two checks are the easiest
   place to collapse them again by mistake. `ok=False` means the thing is
   genuinely absent — no pyannote, no token — and both checks are
   registered `required=False`, which is the *only* reason `doctor` still
   exits zero on a base install. Never "fix" a red row here by returning
   `ok=True`: that makes the output lie about the one thing the command
   exists to report.

### Deviations from the contracts, stated rather than varied

- Contracts §2 lists four files under `rytp/diarize/`; this plan adds
  `segments.py`, `store.py`, `readiness.py`, `pipeline.py`, `link.py` and
  `mapper.py`. Each has one responsibility and the File Structure table says
  which. Part 3 set the precedent with `pipeline.py` and `subproc.py`.
- Contracts §5 says Part 7 owns the `diarize` job kind, but registering it
  means editing `rytp/jobs/__init__.py`, which Part 2 owns. This is working
  as intended: Part 2 owns the file and each part appends its own
  registration through the "Produces for Parts 3-7" recipe, with the
  predicate in Part 7's own light module as that section requires.
- `video_speakers.engine` (contracts §3) is consumed, never created: Part 1
  owns the migrations. Part 7 is the only thing that writes the column, and
  the only thing that reads it — in `link.py`, where it gates whether two
  vectors may be compared at all.
- `rytp/diarize/base.py` reuses `check_available`, `interpreter_for` and
  `availability` from `rytp/transcribe/registry.py` rather than
  re-implementing the Hugging Face gate. Contracts §6 puts `DIARIZERS` in
  `rytp/diarize/base.py` and says nothing about the gate helpers; two
  implementations of "can this engine run here" would drift.
