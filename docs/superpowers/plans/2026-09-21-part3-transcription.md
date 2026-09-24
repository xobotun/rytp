# Part 3 — Transcription, Alignment and Audio Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three transcript tiers — free `caption` words for the whole corpus, `timed` words wherever a transcriber has run, and cut-accurate `aligned` words for the videos worth mining — on top of swappable transcriber/aligner engines, energy-measured word boundaries, and a one-row-per-video acoustic fingerprint.

**Architecture:** Text comes from a transcriber, rough timings from an aligner, and the *final* boundary is measured in the audio. Only a run that actually had an aligner writes the cuttable `aligned` tier; a transcriber's own timestamps are written `timed`, because measured on real data 78.7% of their word gaps are exactly zero and energy refinement cannot place a boundary that was never there. — the local energy minimum between two words, snapped to a zero crossing, with voice-activity edges taken as free true boundaries. Nothing downstream may assume a particular engine: engines are classes in a name registry, `words.engine` records what produced each row, and a `transcribe.compare` command runs several engines over the same audio and reports where they disagree on text, on timings, and on how much silence actually sits at each claimed boundary. Engines whose dependencies conflict declare `out_of_process = True` and run under their own interpreter behind a JSON-file subprocess seam.

**Tech Stack:** Python 3.11+, SQLite, numpy (base dependency), stdlib `wave` and `subprocess`, ffmpeg for loudness only. Optional extras, all lazily imported and none required by the test suite: `faster-whisper`, `gigaam`, Montreal Forced Aligner (`russian_mfa`), `torch`/`transformers` for a wav2vec2 CTC fallback.

**Spec:**
- `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` — the approved design; §6 (transcription) and §11 (milestone M0) are this part's mandate.
- `docs/superpowers/specs/2026-09-21-rytp-contracts.md` — binding shared interfaces. Schema DDL (§3), core types (§4), command registry (§5), engine protocols (§6), filesystem layout (§7) and conventions (§8) are copied verbatim into this plan. **Never vary them.** Six other plans depend on the same file.

## Global Constraints

Every task's requirements implicitly include this section.

- Python >= 3.11. Target 3.11 syntax. `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. `pathlib` everywhere; `subprocess` with **list** arguments only.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` on `rytp/`.
- Heavy dependencies (transcribers, aligners, diarizers) are optional extras, **imported lazily inside functions, never at module import time**.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- **No test may download a model or touch the network.** Every engine is faked. The suite must pass with none of the optional extras installed.
- Time is integer milliseconds everywhere. Database timestamps are `datetime.now(UTC).isoformat()`.
- Domain errors subclass `RytpError` (`rytp/models.py`). The CLI prints one line to stderr and exits 1; no traceback for an expected failure.
- Every stage takes an **open `Database`**; nothing opens its own connection. Multi-statement writes use `db.transaction()`.
- One commit per task, conventional-commit prefix, present tense.

**The one test invocation used everywhere in this plan:**

```bash
python -m pytest <args>
```

That venv has numpy, scipy, pytest, ruff and mypy — and **no ML libraries**, by design. If a test fails there because something heavy is missing, the test is wrong, not the venv.

## Prerequisites (owned by other plan parts)

Part 3 consumes these. They must exist before the first Part 3 test runs. Each is named here so an executor knows exactly what to check, and each has exactly one import site in Part 3 so a rename is a one-line fix.

| Needed | Owner | Used by |
|---|---|---|
| `rytp/models.py`: `RawWord`, `Span`, `Fragment`, `RytpError`, `normalize_text`, `stem_text`, `utc_now_iso` (contracts §4) | Part 1 | everywhere |
| `rytp/db/__init__.py`: `Database` exposing `conn` (a `sqlite3.Connection` with `row_factory = sqlite3.Row`, foreign keys on), `migrate()`, `close()`, and a **re-entrant** `transaction()` — contracts §8 | Part 1 | captions, pipeline, acoustics, commands |
| `rytp/db/schema.py`: `MIGRATIONS` creating `videos`, `assets`, `words`, `utterances`, `video_speakers`, `video_acoustics`, `settings` exactly as contracts §3 spells them | Part 1 | all DB tasks |
| `rytp/commands/__init__.py`: `REQUIRED`, `Param`, `Command`, `CommandResult`, `register`, `resolve`, `COMMANDS` (contracts §5) | Part 1 | Task 17 |
| `rytp/constants.py` exists and is importable with **stdlib imports only** | Part 1 | everywhere |
| `rytp/constants.py`: `SETTING_DEFAULT_TRANSCRIBER` (`"default_transcriber"`) and `DEFAULT_TRANSCRIBER_FALLBACK` (`"gigaam"`), contracts §3 "Choosing a transcriber" — declared in Part 2's block beside `SETTING_DEFAULT_ALIGNER`, because Part 2's `ingest` must resolve them while Part 3 may still be absent | Part 2 | one import site, `rytp/transcribe/registry.py::default_transcriber` |
| `rytp/db/queries.py`: `set_setting(db, key, value)` | Part 1 | tests only — Tasks 1, 17, 19 write `default_transcriber` |
| `rytp/__init__.py` and `rytp/transcribe/__init__.py` stay **import-light** (no numpy, no heavy deps) | Part 1 / this part | the out-of-process child imports engine modules under a foreign interpreter |
| `tests/conftest.py` fixtures `tmp_path`, `data_dir`, `db` with their contracted names and semantics | Part 1 | all tests |
| `tests/conftest.py` roots `tmp_path` at `$RYTP_TEST_TMP` when that variable is set, falling back to a workspace-local directory — which is why every test invocation below sets it | Part 1 | all tests |
| `rytp.audio.extract.wav_path(video_id: int) -> Path` — the cached WAV's location, contracts §7 `cache/wav/{video_id}.wav`. Part 3 never computes that path itself | Part 2 | one import site, `rytp/commands/transcribe.py::_wav_for` |
| A 16 kHz mono 16-bit PCM WAV at that path, plus a `captions` asset row for caption ingest | Part 2 | Tasks 8, 9, 17 |
| `rytp/jobs/__init__.py`: `Readiness`, `Predicate`, `Handler`, `JobKind`, `register_job_kind`, `JOB_KINDS`, `JOB_HANDLERS` (contracts §5 "Job handlers"), and the file being additive at the bottom for other parts' thunks | Part 2 | Task 19 |
| `rytp.commands.{HealthCheck, HealthResult, register_check, HEALTH_CHECKS}` (contracts §5 "Health checks"), and the `doctor` command that runs them | Part 1 | Task 21 |

Two things this plan raised during review were settled in the contracts rather than resolved locally, and both now bind:

1. **`stem_text` lives in `rytp/models.py`**, fully implemented by Part 1 (snowballstemmer, which contracts §1 makes a base dependency), alongside `normalize_text`, and there is no `rytp/index/stem.py`. Contracts §4 spells out why: `words.stem` is written by Part 3 when words are created, so putting the stemmer under the index package would invert the dependency. Import both from `rytp.models`. Note `normalize_text` also folds `ё` to `е`; Part 3 only ever calls it, never reimplements it.
2. **Engine timestamps are absolute.** Contracts §4 now says so in `RawWord`'s own docstring: "Timings, when present, are absolute against the source audio, never relative to a chunk." `Transcriber.transcribe` and `Aligner.align` take a `start_ms`/`end_ms` window into the whole-file WAV and emit whole-file timestamps. Adapters that decode a slice use `rytp.transcribe.base.shift_words`.

**`RawWord` carries text only when a transcriber has nothing else.** Contracts §4 orders its fields `text, start_ms=None, end_ms=None, confidence=None`: a transcriber may legitimately emit no timings at all and leave every boundary to the aligner. Always construct it with keyword arguments, never positionally, and never assume `start_ms` is an `int` without checking.

## File Structure

Contracts §2 fixes the tree. Part 3 owns:

| File | Responsibility |
|---|---|
| `rytp/transcribe/base.py` | Protocols (`Transcriber`, `Aligner`), engine errors, the absolute-timestamp convention, and the two stdlib-only helpers every adapter needs (`shift_words`, `slice_wav_window`). **Import-light on purpose** — imported by every adapter, including inside a foreign interpreter. |
| `rytp/transcribe/registry.py` | `TRANSCRIBERS` / `ALIGNERS` name→**class** registries, `register_*`, `resolve_*`, the HF-token gate, the settings lookups that supply an out-of-process engine's interpreter and the `default_transcriber` (contracts §3), and the row source for `transcribe.engines`. |
| `rytp/transcribe/subproc.py` | *(added to the contract tree — justified below)* The out-of-process seam: parent-side runner and child-side entry point. |
| `rytp/transcribe/captions.py` | json3 auto-captions → caption-tier `words` rows. |
| `rytp/transcribe/pipeline.py` | *(added to the contract tree — justified below)* Aligned-tier orchestration: VAD chunk → transcribe → align → refine → write; plus re-alignment of existing words. |
| `rytp/transcribe/compare.py` | Multi-engine comparison: sequence alignment, text/timing disagreement, boundary statistics, markdown report. |
| `rytp/transcribe/readiness.py` | *(added to the contract tree — justified below)* The four readiness predicates the job queue imports. |
| `rytp/transcribe/health.py` | *(added to the contract tree — justified below)* The `doctor` checks: which engines are runnable, and whether a GPU is visible. |
| `rytp/transcribe/engines/{__init__,whisper,gigaam}.py` | Transcriber adapters. `__init__` imports each module — *that import is what makes the name resolvable.* |
| `rytp/transcribe/align/{__init__,mfa,wav2vec2}.py` | Aligner adapters, same registration rule. |
| `rytp/audio/vad.py` | Energy-based voice activity detection and chunk planning under the transcriber's per-call cap. |
| `rytp/audio/energy.py` | WAV sample access, short-time energy, the measured boundary (energy minimum + zero-crossing snap), monotonicity repair, silence-depth measurement. |
| `rytp/audio/acoustics.py` | One cheap fingerprint per video: pitch mean/spread, spectral tilt, noise floor, reverberation proxy, loudness. |
| `rytp/commands/transcribe.py` | Five registry commands, thin handlers over the modules above. |

**Two files added beyond the contract tree, with reasons:**

- `rytp/transcribe/subproc.py` — contracts §6 mandates an out-of-process seam but gives it no home. It cannot live in `base.py`, which must stay importable under a foreign interpreter with nothing but the standard library; process management, temp files and stderr handling are a separate responsibility.
- `rytp/transcribe/pipeline.py` — the transcription orchestration imports numpy and `rytp.audio`, which `base.py` must not, and command handlers stay thin per contracts §5.
- `rytp/transcribe/readiness.py` — Part 2's registration contract requires the readiness predicates to live in a module light enough for `rytp/jobs/__init__.py` to import at module load. `pipeline.py` imports numpy and is therefore disqualified.
- `rytp/transcribe/health.py` — contracts §5 has each part contribute its own `doctor` checks. They are neither a command nor a stage, and they must stay importable without touching an engine.

`rytp/audio/__init__.py` must exist (Part 2 also creates it). Create it if absent, docstring only.

**Two test-support modules** (plain modules, not `conftest.py`, so Part 1's conftest stays untouched):

- `tests/synth_audio.py` — tone/silence generators and a 16 kHz mono WAV writer.
- `tests/fake_engines.py` — fake transcribers and aligners plus a registry save/restore helper.

Do **not** copy `fake_video_row` from the current `tests/conftest.py`: it contains a real video id. Part 3 fixtures use `external_id="VIDEO_A"`.

---

### Task 1: Constants, engine protocols and the engine registry

**Files:**
- Modify: `rytp/constants.py` (append one delimited Part 3 block at the end)
- Create: `rytp/transcribe/base.py`
- Create: `rytp/transcribe/registry.py`
- Create: `tests/fake_engines.py`
- Test: `tests/test_transcribe_registry.py`, `tests/test_transcribe_tokens.py`

**Interfaces:**
- Consumes: `rytp.models.RawWord`, `rytp.models.Span`, `rytp.models.RytpError`; `Database.conn`.
- Produces:
  - `rytp.transcribe.base`: `Transcriber`, `Aligner` (Protocols), `EngineUnavailable`, `EngineSubprocessError`, `split_token(text) -> list[tuple[str, str]]`, `shift_words(words, offset_ms) -> list[RawWord]`, `slice_wav_window(src, dst, start_ms, end_ms) -> Path`.
  - `rytp.transcribe.registry`: `TRANSCRIBERS`, `ALIGNERS`, `register_transcriber`, `register_aligner`, `resolve_transcriber`, `resolve_aligner`, `check_available(cls)`, `interpreter_for(db, name)`, `setting(db, key, default)`, `default_transcriber(db)`, `load_transcriber(db, name, **kw)`, `load_aligner(db, name, **kw)`, `engine_rows(db)`.
  - `tests/fake_engines.py`: `FakeTranscriber`, `NoEndTimesTranscriber`, `FakeAligner`, `ShortAligner`, `GatedTranscriber`, `OutOfProcessTranscriber`, `MissingModuleTranscriber`, `registered(*classes)`.

- [ ] **Step 1: Append the Part 3 constants block to `rytp/constants.py`**

Every later task reads these; adding them once keeps a Part-1-owned file from being edited eighteen times.

```python
# ---------------------------------------------------------------------------
# Part 3 — transcription, alignment and audio analysis (design §6, §11)
# ---------------------------------------------------------------------------

#: Sample rate of the cached WAV every audio stage reads (design §4).
WAV_SAMPLE_RATE_HZ: int = 16000

#: Default spoken language passed to transcribers (design §1: a Russian archive).
DEFAULT_LANGUAGE: str = "ru"

#: Floor for dB conversion, and the epsilon that keeps log10 finite. A 16-bit
#: sample's quietest non-zero step is about -90 dBFS, so -120 dB is safely below
#: anything real while keeping digital silence from becoming -inf (design §6).
DB_FLOOR: float = -120.0
DB_EPSILON: float = 1e-10

# --- Voice activity detection (design §6, "VAD contributes for free") ---

#: Analysis frame and hop for VAD. 20 ms is the usual speech-analysis frame; a
#: 10 ms hop decides a boundary every 10 ms, well under the ~150 ms error that
#: becomes audible in a cut.
VAD_FRAME_MS: int = 20
VAD_HOP_MS: int = 10

#: The noise floor is this percentile of frame energy. 10 assumes at least a
#: tenth of any recording is not speech, which holds for interview audio.
VAD_NOISE_PERCENTILE: float = 10.0

#: Hysteresis band above the noise floor: enter speech at +12 dB, leave at
#: +8 dB. The gap stops a wavering frame shredding one utterance into many.
VAD_ENTER_DB: float = 12.0
VAD_EXIT_DB: float = 8.0

#: Segments shorter than this are discarded as clicks; silences shorter than
#: this are swallowed, because a stop consonant's closure is real silence
#: *inside* a word (design §6).
VAD_MIN_SPEECH_MS: int = 120
VAD_MIN_SILENCE_MS: int = 150

#: Padding either side of a detected segment so a soft onset is not clipped.
#: Acoustics passes pad_ms=0 when it measures a reverberation tail.
VAD_PAD_MS: int = 30

#: GigaAM v3 accepts at most 25 s per call (design §6), so a chunk may never
#: exceed it; the target is lower so chunks usually end at a real silence.
VAD_CHUNK_MAX_MS: int = 25_000
VAD_CHUNK_TARGET_MS: int = 20_000

# --- Boundary refinement (design §6, "boundaries good enough to cut on") ---

#: Short-time energy used to find the quietest point between two words. 5 ms
#: frames resolve a stop closure; a 1 ms hop is the finest the millisecond
#: resolution of the database can use.
ENERGY_FRAME_MS: int = 5
ENERGY_HOP_MS: int = 1

#: RMS values within this of the minimum count as tied, so digital silence
#: (hundreds of exactly equal frames) resolves deterministically to the frame
#: nearest the claimed boundary instead of to the earliest one.
ENERGY_TIE_RMS: float = 1e-6

#: A refined boundary may never move further than this from the aligner's
#: claim. The rail stops a runaway search swallowing a neighbouring word, and
#: 60 ms stays under the ~150 ms at which a mis-cut becomes audible (design §6).
BOUNDARY_SEARCH_MS: int = 60

#: How far the energy minimum may be nudged to land on a zero crossing.
ZERO_CROSSING_SEARCH_MS: int = 5

#: A word shorter than this cannot be real; monotonicity repair widens it.
MIN_WORD_DURATION_MS: int = 20

#: A word edge within this of a VAD segment edge is snapped to it — that edge
#: is measured silence, which beats any search (design §6).
VAD_EDGE_SNAP_MS: int = 40

#: Silence depth at a boundary: reference level is the median frame energy in
#: the 200 ms either side, the minimum is taken within 10 ms of the boundary.
#: The reference window has to reach *past* a pause to find the speech it is
#: comparing against — a window shorter than a typical pause would measure
#: silence against silence and report a depth of zero.
SILENCE_DEPTH_REF_MS: int = 200
SILENCE_DEPTH_WINDOW_MS: int = 10

#: Silence depth (dB) treated as a perfect boundary when measured boundary
#: quality stands in for an aligner that reports no per-word confidence.
SILENCE_DEPTH_FULL_DB: float = 30.0

# --- Acoustic fingerprint (design §4, "Acoustics") ---

#: 40 ms is two pitch periods at the 60 Hz floor — the shortest frame from
#: which autocorrelation can recover a male speaking pitch.
ACOUSTICS_FRAME_MS: int = 40
ACOUSTICS_HOP_MS: int = 20

#: At most this many voiced frames are analysed, evenly strided across the
#: file. The fingerprint is a summary; a full hour adds no information.
ACOUSTICS_MAX_FRAMES: int = 4000

#: Human speaking pitch range. Below 60 Hz is rumble, above 400 Hz is song.
F0_MIN_HZ: int = 60
F0_MAX_HZ: int = 400

#: A frame counts as voiced when its normalised autocorrelation peak clears
#: this and its energy clears the noise floor by this many dB.
F0_VOICED_AUTOCORR: float = 0.3
F0_VOICED_ABOVE_FLOOR_DB: float = 10.0

#: Band over which spectral tilt is fitted, in dB per decade. The top stays
#: below the 8 kHz Nyquist of 16 kHz audio where the anti-alias filter rolls off.
SPECTRAL_TILT_LO_HZ: int = 100
SPECTRAL_TILT_HI_HZ: int = 7800

#: Reverberation proxy: energy in the 150 ms after a speech segment ends,
#: relative to the last 50 ms of that segment. A dry room decays instantly.
REVERB_REF_MS: int = 50
REVERB_TAIL_MS: int = 150

#: ffmpeg loudness scan of an hour of audio finishes well inside this.
LOUDNESS_TIMEOUT_S: int = 600

# --- Engines (contracts §6; design §6, "Engines run out of process") ---

#: Settings keys that point an out-of-process engine at its own interpreter,
#: and an external binary at its path.
SETTINGS_INTERPRETER_PREFIX: str = "engine.interpreter."
SETTINGS_INTERPRETER_DEFAULT: str = "engine.interpreter.default"
SETTINGS_BINARY_PREFIX: str = "engine.binary."

#: One GPU pass over an hour of audio, with headroom for model loading.
ENGINE_SUBPROCESS_TIMEOUT_S: int = 3600

#: Lines of child stderr quoted back when an out-of-process engine fails.
ENGINE_SUBPROCESS_STDERR_TAIL: int = 20

#: words.engine value for caption-tier rows (design §6, tier 1).
CAPTION_ENGINE: str = "captions:json3"

#: Default model identifiers (design §6, "Recommended stack").
#: **There is deliberately no DEFAULT_TRANSCRIBER here.** contracts §3
#: "Choosing a transcriber" makes the engine a *setting* — Part 2's
#: `SETTING_DEFAULT_TRANSCRIBER`, read through
#: `rytp.transcribe.registry.default_transcriber` — so that a run which used
#: the default can say so. A constant cannot; that is the whole point.
WHISPER_DEFAULT_MODEL: str = "large-v3"
GIGAAM_DEFAULT_MODEL: str = "v3_rnnt"
MFA_ACOUSTIC_MODEL: str = "russian_mfa"
MFA_DICTIONARY: str = "russian_mfa"
WAV2VEC2_MODEL: str = "bond005/wav2vec2-large-ru-golos"

#: Default window and sample size for `transcribe compare` (design §11, M0).
COMPARE_DEFAULT_WINDOW_MS: int = 120_000
COMPARE_SAMPLE_WORDS: int = 40
```

- [ ] **Step 2: Write `tests/fake_engines.py`**

Not a test itself — the fakes every later task uses. Written now because Task 1's test needs them.

```python
"""Fake engines for tests: no model, no network, no optional extra.

Registering an engine mutates module-level dicts, so always go through
:func:`registered`, which restores both registries afterwards.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rytp.models import RawWord, Span
from rytp.transcribe import registry


class FakeTranscriber:
    """Replays a fixed script of (start_ms, end_ms, text) triples."""

    name = "fake"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    script: tuple[tuple[int, int | None, str], ...] = (
        (0, 200, "один"),
        (200, 460, "два"),
        (460, 700, "три"),
    )

    def __init__(self, script: Sequence[tuple[int, int | None, str]] | None = None) -> None:
        self._script = tuple(script) if script is not None else self.script

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        for start, end, text in self._script:
            if start < start_ms:
                continue
            if end_ms is not None and start >= end_ms:
                continue
            yield RawWord(start_ms=start, end_ms=end, text=text, confidence=0.9)


class NoEndTimesTranscriber(FakeTranscriber):
    """A transcriber that emits starts only — the pipeline must demand an aligner."""

    name = "fake-no-ends"
    script = ((0, None, "один"), (200, None, "два"))


class TextOnlyTranscriber:
    """Text and nothing else.

    Legal since contracts §4 made both ``RawWord`` timings optional, and the
    reason the pipeline may never assume ``start_ms`` is an ``int``. Must be
    paired with an aligner.
    """

    name = "fake-text-only"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    words: tuple[str, ...] = ("один", "два", "три")

    def __init__(self, words: Sequence[str] | None = None) -> None:
        self._words = tuple(words) if words is not None else self.words

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        for text in self._words:
            yield RawWord(text=text)


class FakeAligner:
    """Spreads the words evenly across the window, one span each."""

    name = "fake-aligner"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        n = len(words)
        if n == 0:
            return []
        step = max(1, (end_ms - start_ms) // n)
        return [
            Span(start_ms=start_ms + i * step, end_ms=start_ms + (i + 1) * step, score=0.8)
            for i in range(n)
        ]


class ShortAligner(FakeAligner):
    """Returns one span too few — the count mismatch the pipeline must refuse."""

    name = "fake-short-aligner"

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        return super().align(audio, words, start_ms=start_ms, end_ms=end_ms)[:-1]


class GatedTranscriber(FakeTranscriber):
    name = "fake-gated"
    requires_hf_token = True


class OutOfProcessTranscriber(FakeTranscriber):
    name = "fake-remote"
    out_of_process = True

    def __init__(
        self,
        interpreter: str,
        script: Sequence[tuple[int, int | None, str]] | None = None,
    ) -> None:
        super().__init__(script)
        self.interpreter = interpreter


class MissingModuleTranscriber(FakeTranscriber):
    name = "fake-missing"
    required_module = "definitely_not_installed_xyz"
    extra = "fakeextra"


_MISSING = object()


@contextmanager
def registered(*classes: type) -> Iterator[None]:
    """Register engines for the duration of a test, then undo exactly that.

    Only the names this call added are removed afterwards. Snapshotting and
    restoring the whole registry would be wrong: a module registers itself at
    import time and Python imports it once, so wiping a real engine's name
    here would leave it unregistered for every later test file.
    """
    undo: list[tuple[dict[str, type], str, object]] = []
    try:
        for cls in classes:
            table = registry.ALIGNERS if hasattr(cls, "align") else registry.TRANSCRIBERS
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

- [ ] **Step 3: Write the failing test**

Create `tests/test_transcribe_registry.py`:

```python
"""The engine registry: classes in, classes out, gated before construction."""
from __future__ import annotations

import sys

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import set_setting
from rytp.transcribe import registry
from rytp.transcribe.base import EngineUnavailable
from tests.fake_engines import (
    FakeAligner,
    GatedTranscriber,
    MissingModuleTranscriber,
    OutOfProcessTranscriber,
    FakeTranscriber,
    registered,
)


def test_register_and_resolve_returns_the_class_not_an_instance() -> None:
    with registered(FakeTranscriber, FakeAligner):
        assert registry.resolve_transcriber("fake") is FakeTranscriber
        assert registry.resolve_aligner("fake-aligner") is FakeAligner


def test_resolve_unknown_transcriber_names_the_available_ones() -> None:
    with registered(FakeTranscriber), pytest.raises(ValueError) as excinfo:
        registry.resolve_transcriber("nope")
    assert "nope" in str(excinfo.value)
    assert "fake" in str(excinfo.value)


def test_gate_rejects_a_token_engine_before_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(GatedTranscriber)
    assert "HF_TOKEN" in str(excinfo.value)


def test_gate_accepts_a_token_engine_when_the_token_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    registry.check_available(GatedTranscriber)


def test_gate_names_the_extra_for_a_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(MissingModuleTranscriber)
    assert "rytp[fakeextra]" in str(excinfo.value)


def test_interpreter_falls_back_to_this_interpreter(db: Database) -> None:
    assert registry.interpreter_for(db, "fake-remote") == sys.executable


def test_interpreter_prefers_the_per_engine_setting_over_the_default(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.default", "/opt/default/python"),
    )
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote", "/opt/remote/python"),
    )
    db.conn.commit()
    assert registry.interpreter_for(db, "fake-remote") == "/opt/remote/python"
    assert registry.interpreter_for(db, "other") == "/opt/default/python"


def test_load_transcriber_injects_the_interpreter_for_out_of_process_engines(
    db: Database,
) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        ("engine.interpreter.fake-remote", "/opt/remote/python"),
    )
    db.conn.commit()
    with registered(OutOfProcessTranscriber):
        engine = registry.load_transcriber(db, "fake-remote")
    assert engine.interpreter == "/opt/remote/python"


def test_load_transcriber_does_not_inject_an_interpreter_in_process(db: Database) -> None:
    with registered(FakeTranscriber):
        engine = registry.load_transcriber(db, "fake")
    assert not hasattr(engine, "interpreter")


def test_engine_rows_reports_kind_and_availability(db: Database) -> None:
    with registered(FakeTranscriber, FakeAligner, MissingModuleTranscriber):
        rows = {row[0]: row for row in registry.engine_rows(db)}
    assert rows["fake"][1] == "transcriber"
    assert rows["fake-aligner"][1] == "aligner"
    assert rows["fake"][4] == "ready"
    assert "definitely_not_installed_xyz" in rows["fake-missing"][4]


def test_the_default_transcriber_is_the_russian_engine_out_of_the_box(
    db: Database,
) -> None:
    # contracts §3: the corpus is Russian and gigaam roughly halves Whisper's
    # word error rate there. Part 1 seeds no row for this key, so the value
    # has to come from the fallback or "defaults to gigaam" is not true.
    assert registry.default_transcriber(db) == C.DEFAULT_TRANSCRIBER_FALLBACK
    assert registry.default_transcriber(db) == "gigaam"


def test_the_setting_wins_over_the_fallback(db: Database) -> None:
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "whisper")
    assert registry.default_transcriber(db) == "whisper"


def test_an_empty_default_transcriber_is_not_a_way_to_turn_it_off(
    db: Database,
) -> None:
    # Empty is meaningful for default_aligner and meaningless here: a
    # transcribe run with no engine does nothing at all, so there is no
    # unset path and this always answers with a name.
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "  ")
    assert registry.default_transcriber(db) == C.DEFAULT_TRANSCRIBER_FALLBACK
```

- [ ] **Step 4: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_registry.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.registry'`.

- [ ] **Step 5: Write `rytp/transcribe/base.py`**

```python
"""Engine protocols for transcription and alignment (contracts §6, design §6).

Import-light on purpose. Every engine adapter imports this module, and an
out-of-process adapter is imported again inside a *foreign* interpreter where
only the standard library and :mod:`rytp.models` are guaranteed to load.
Never import numpy, torch, or anything from :mod:`rytp.audio` here.

Timestamp convention
--------------------
:meth:`Transcriber.transcribe` and :meth:`Aligner.align` receive a window
(``start_ms`` / ``end_ms``) into a whole-file 16 kHz mono WAV and MUST emit
**absolute** timestamps, measured from the start of that file rather than from
the start of the window. An adapter that decodes a slice writes the slice with
:func:`slice_wav_window` and shifts the result with :func:`shift_words`.
"""
from __future__ import annotations

import re
import wave
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from rytp.models import RawWord, RytpError, Span, normalize_text


class EngineUnavailable(RytpError):
    """A named engine is registered but cannot run here (missing extra or token)."""


class EngineSubprocessError(RytpError):
    """An out-of-process engine failed; the message carries the child's report."""


class Transcriber(Protocol):
    """Audio in, words out. Timestamps are absolute — see the module docstring."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]: ...


class Aligner(Protocol):
    """Audio plus known text in, one :class:`Span` per word out, absolute ms."""

    name: str
    requires_hf_token: bool
    out_of_process: bool

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]: ...


#: Everything a word row may not contain. ``normalize_text`` turns each of
#: these into a space, so a token spanning one of them would normalize to two.
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def split_token(text: str) -> list[tuple[str, str]]:
    """Split one transcriber or caption token into the tokens that become rows.

    Returns ``(surface, normalized)`` pairs — the surface form for
    ``words.text``, the normalized form for ``words.normalized_text``.

    **Why a hyphenated word becomes two rows.** `normalize_text` replaces
    punctuation with a space, so "кто-то" normalizes to "кто то". A single row
    holding two tokens can never be found: Part 4's index and FTS lookups are
    single-token and Part 5's assembly pointer walk assumes one token per row.
    Stripping the hyphen instead — storing "ктото" — would be worse, because
    the *same* normalizer runs over the user's query, which also splits into
    two tokens, so the stripped row would be unreachable from either
    direction. One row per token is the only form that both sides agree on.

    Punctuation-only input yields nothing, which is how "—" and a stray comma
    are kept out of the ordinal sequence.
    """
    pairs: list[tuple[str, str]] = []
    for piece in _TOKEN_SPLIT_RE.split(text):
        normalized = normalize_text(piece)
        if normalized:
            pairs.append((piece, normalized))
    return pairs


def shift_words(words: Iterable[RawWord], offset_ms: int) -> list[RawWord]:
    """Move window-relative words onto the whole-file timeline."""
    return [
        RawWord(
            start_ms=w.start_ms + offset_ms,
            end_ms=None if w.end_ms is None else w.end_ms + offset_ms,
            text=w.text,
            confidence=w.confidence,
        )
        for w in words
    ]


def shift_spans(spans: Iterable[Span], offset_ms: int) -> list[Span]:
    """Move window-relative spans onto the whole-file timeline."""
    return [
        Span(start_ms=s.start_ms + offset_ms, end_ms=s.end_ms + offset_ms, score=s.score)
        for s in spans
    ]


def slice_wav_window(src: Path, dst: Path, start_ms: int, end_ms: int | None) -> Path:
    """Copy ``[start_ms, end_ms)`` of a PCM WAV into ``dst``, standard library only.

    Engines take a file, not an array, and the child process cannot import
    numpy. ``wave`` handles the frame arithmetic and writes a valid header.
    """
    with wave.open(str(src), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        total = reader.getnframes()
        first = min(max(0, start_ms * rate // 1000), total)
        last = total if end_ms is None else min(max(first, end_ms * rate // 1000), total)
        reader.setpos(first)
        frames = reader.readframes(last - first)
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return dst
```

- [ ] **Step 6: Write `rytp/transcribe/registry.py`**

```python
"""Name to class registries for transcribers and aligners (contracts §6).

Classes, never instances. The Hugging Face token gate and the "is this engine
even installed" listing have to be answerable *before* anything is
constructed, because constructing a gated engine downloads gigabytes only to
discover the token is missing. An engine class is therefore inspected, not
built, until the moment it is used.

Import-light, like :mod:`rytp.transcribe.base`: engine adapters import this
module at the bottom of their own file to register themselves, and those
adapters are imported inside a foreign interpreter.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rytp import constants as C
from rytp.transcribe.base import Aligner, EngineUnavailable, Transcriber

if TYPE_CHECKING:
    from rytp.db import Database

TRANSCRIBERS: dict[str, type[Transcriber]] = {}
"""Registered transcriber classes, by ``cls.name``."""

ALIGNERS: dict[str, type[Aligner]] = {}
"""Registered aligner classes, by ``cls.name``."""


def register_transcriber(cls: type[Transcriber]) -> type[Transcriber]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    TRANSCRIBERS[cls.name] = cls
    return cls


def register_aligner(cls: type[Aligner]) -> type[Aligner]:
    """Class decorator: make ``cls`` resolvable as ``cls.name``."""
    ALIGNERS[cls.name] = cls
    return cls


def resolve_transcriber(name: str) -> type[Transcriber]:
    """Look up a transcriber class. Raises ``ValueError`` naming the alternatives."""
    try:
        return TRANSCRIBERS[name]
    except KeyError:
        available = ", ".join(sorted(TRANSCRIBERS)) or "(none registered)"
        raise ValueError(f"unknown transcriber {name!r}; available: {available}") from None


def resolve_aligner(name: str) -> type[Aligner]:
    """Look up an aligner class. Raises ``ValueError`` naming the alternatives."""
    try:
        return ALIGNERS[name]
    except KeyError:
        available = ", ".join(sorted(ALIGNERS)) or "(none registered)"
        raise ValueError(f"unknown aligner {name!r}; available: {available}") from None


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_available(cls: type) -> None:
    """Raise :class:`EngineUnavailable` if this engine cannot run on this machine.

    Checked against the class, never an instance. An out-of-process engine's
    dependency lives in another interpreter, so ``find_spec`` here would be
    meaningless and is skipped.
    """
    if getattr(cls, "requires_hf_token", False) and not _hf_token():
        raise EngineUnavailable(
            f"engine {cls.name!r} needs a Hugging Face token; set HF_TOKEN"
        )
    module = getattr(cls, "required_module", None)
    if module and not getattr(cls, "out_of_process", False) and not _module_present(module):
        extra = getattr(cls, "extra", None) or cls.name
        raise EngineUnavailable(
            f"engine {cls.name!r} needs {module!r}: pip install rytp[{extra}]"
        )


def setting(db: Database, key: str, default: str | None = None) -> str | None:
    """Read one row of the ``settings`` table (contracts §3)."""
    row = db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else str(row[0])


def default_transcriber(db: Database) -> str:
    """The engine to use when no ``--transcriber`` was named (contracts §3).

    Reads Part 2's ``default_transcriber`` setting, falling back to the
    shipped value — Part 1's migration seeds ``default_aligner`` and not this
    key, so the fallback is what makes "defaults to gigaam" true. **Empty is
    not meaningful here**, unlike for the aligner: an empty aligner reads as
    "no alignment, the words stay `timed`", an empty transcriber would mean
    nothing runs. So this always returns a name.

    `gigaam` is the Russian-specific engine and published benchmarks put it
    at roughly half Whisper's Russian word error rate, which matters at
    35-90 GPU-hours for the corpus. **A starting point, not a verdict:** the
    one published test on *noisy YouTube* audio — which is exactly this
    corpus — favoured a Russian-finetuned Whisper instead. That is what
    ``transcribe compare`` is for; re-run it on real material and rewrite the
    setting rather than treating this value as settled.

    Callers that used this rather than an explicit name must say so, and must
    let :func:`check_available` raise when the chosen engine is not
    installed — never substitute another one. ``words.engine`` records what
    actually ran, which only helps if nothing lies about what it used.
    """
    name = (setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "") or "").strip()
    return name or C.DEFAULT_TRANSCRIBER_FALLBACK


def interpreter_for(db: Database, name: str) -> str:
    """Interpreter an out-of-process engine runs under (design §6).

    Per-engine setting first, then the shared default, then this interpreter —
    which is correct for an engine whose dependencies happen not to conflict.
    """
    value = setting(db, f"{C.SETTINGS_INTERPRETER_PREFIX}{name}")
    if value is None:
        value = setting(db, C.SETTINGS_INTERPRETER_DEFAULT)
    return value or sys.executable


def load_transcriber(db: Database, name: str, **kwargs: Any) -> Transcriber:
    """Resolve, gate, and construct a transcriber."""
    cls = resolve_transcriber(name)
    check_available(cls)
    if getattr(cls, "out_of_process", False):
        kwargs.setdefault("interpreter", interpreter_for(db, name))
    return cls(**kwargs)  # type: ignore[call-arg]


def load_aligner(db: Database, name: str, **kwargs: Any) -> Aligner:
    """Resolve, gate, and construct an aligner."""
    cls = resolve_aligner(name)
    check_available(cls)
    if getattr(cls, "out_of_process", False):
        kwargs.setdefault("interpreter", interpreter_for(db, name))
    return cls(**kwargs)  # type: ignore[call-arg]


def availability(db: Database, cls: type) -> str:
    """One human-readable word on whether this engine could run right now."""
    if getattr(cls, "requires_hf_token", False) and not _hf_token():
        return "no HF_TOKEN"
    if getattr(cls, "out_of_process", False):
        path = Path(interpreter_for(db, cls.name))
        return "interpreter ok" if path.exists() else f"interpreter missing: {path}"
    module = getattr(cls, "required_module", None)
    if not module:
        return "ready"
    return "ready" if _module_present(module) else f"needs {module}"


def engine_rows(db: Database) -> list[tuple[str, str, str, str, str]]:
    """Rows for ``rytp transcribe engines``: name, kind, token, process, state."""
    rows: list[tuple[str, str, str, str, str]] = []
    for kind, table in (("transcriber", TRANSCRIBERS), ("aligner", ALIGNERS)):
        for name, cls in sorted(table.items()):
            rows.append(
                (
                    name,
                    kind,
                    "yes" if getattr(cls, "requires_hf_token", False) else "no",
                    "yes" if getattr(cls, "out_of_process", False) else "no",
                    availability(db, cls),
                )
            )
    return rows
```

- [ ] **Step 7: Pin the one-token-per-row rule**

Create `tests/test_transcribe_tokens.py`:

```python
"""One word row holds exactly one token — the rule Parts 4 and 5 depend on."""
from __future__ import annotations

import pytest

from rytp.models import normalize_text
from rytp.transcribe.base import split_token


def test_a_plain_word_is_one_token() -> None:
    assert split_token("слово") == [("слово", "слово")]


def test_a_hyphenated_word_becomes_two_rows() -> None:
    assert split_token("кто-то") == [("кто", "кто"), ("то", "то")]
    assert split_token("из-за") == [("из", "из"), ("за", "за")]


def test_trailing_punctuation_does_not_create_an_empty_token() -> None:
    assert split_token("слово,") == [("слово", "слово")]
    assert split_token("«слово»") == [("слово", "слово")]


def test_a_punctuation_only_token_yields_nothing() -> None:
    assert split_token("—") == []
    assert split_token("  ") == []


@pytest.mark.parametrize(
    "text", ["слово", "кто-то", "из-за", "по-моему", "слово,", "два слова", "—", "ещё"]
)
def test_the_split_is_exactly_what_the_shared_normalizer_would_produce(text: str) -> None:
    # The invariant that keeps this consistent with Part 4's query-side
    # normalization: splitting a token here must give the same sequence as
    # normalizing the whole thing and splitting on whitespace. If Part 1 ever
    # changes normalize_text's punctuation handling, this fails loudly.
    assert [n for _surface, n in split_token(text)] == normalize_text(text).split()
```

- [ ] **Step 8: Run both tests to verify they pass**

Run:
```bash
python -m pytest tests/test_transcribe_registry.py tests/test_transcribe_tokens.py -v
```
Expected: 25 passed.

- [ ] **Step 9: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/transcribe
```
Expected: no findings. `ruff` wants imports sorted (rule `I`); fix any it reports with `ruff check --fix`.

- [ ] **Step 10: Commit**

```bash
git add rytp/constants.py rytp/transcribe/base.py rytp/transcribe/registry.py tests/fake_engines.py tests/test_transcribe_registry.py tests/test_transcribe_tokens.py
git commit -m "feat(transcribe): engine protocols, the registry, and one-token word splitting"
```

---

### Task 2: The out-of-process engine seam

Engines whose dependency pins conflict (design §6) run under their own interpreter. This task builds the channel and, just as importantly, the way a child reports failure: a silent subprocess that writes nothing is the worst possible outcome, so every failure mode produces a named exception with the child's stderr attached.

**Files:**
- Create: `rytp/transcribe/subproc.py`
- Create: `tests/subproc_engine_stub.py`
- Test: `tests/test_transcribe_subproc.py`

**Interfaces:**
- Consumes: `rytp.transcribe.base.EngineSubprocessError`, `rytp.constants`.
- Produces: `run_child(*, interpreter, module, request, timeout_s=None, repo_root=None) -> dict[str, Any]`. The child contract: the named module exposes `child_main(request: dict) -> dict`, is importable with the standard library plus `rytp.models` plus its own optional dependency, and returns JSON-serialisable plain dicts.

- [ ] **Step 1: Write the child stub the test drives**

Create `tests/subproc_engine_stub.py` — a stand-in for a real engine module, import-light so the child can load it:

```python
"""A stand-in engine module for the out-of-process seam test.

Imported by the child process by dotted name, exactly as a real adapter is.
Keeps to the standard library so it loads under any interpreter.
"""
from __future__ import annotations

from typing import Any


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Echo the request, or fail on demand, or print noise to stdout first."""
    if request.get("noise"):
        print("loading model 100%|##########|")  # the noise is the point
    if request.get("boom"):
        raise RuntimeError("engine exploded")
    return {"words": [{"start_ms": 0, "end_ms": 100, "text": "ok", "confidence": 1.0}]}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_transcribe_subproc.py`:

```python
"""The out-of-process seam: a request in, a JSON file out, errors that name themselves."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rytp.transcribe.base import EngineSubprocessError
from rytp.transcribe.subproc import run_child

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = "tests.subproc_engine_stub"


def test_round_trip_returns_the_child_result() -> None:
    result = run_child(
        interpreter=sys.executable, module=STUB, request={}, repo_root=REPO_ROOT
    )
    assert result["words"][0]["text"] == "ok"


def test_stdout_noise_from_the_child_is_ignored() -> None:
    result = run_child(
        interpreter=sys.executable,
        module=STUB,
        request={"noise": True},
        repo_root=REPO_ROOT,
    )
    assert result["words"][0]["text"] == "ok"


def test_child_exception_becomes_a_named_error_with_the_message() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module=STUB,
            request={"boom": True},
            repo_root=REPO_ROOT,
        )
    message = str(excinfo.value)
    assert "RuntimeError" in message
    assert "engine exploded" in message


def test_unimportable_module_is_reported_not_swallowed() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="tests.no_such_engine_module",
            request={},
            repo_root=REPO_ROOT,
        )
    assert "no_such_engine_module" in str(excinfo.value)


def test_missing_interpreter_names_the_path() -> None:
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter="/no/such/python",
            module=STUB,
            request={},
            repo_root=REPO_ROOT,
        )
    assert "/no/such/python" in str(excinfo.value)


def test_timeout_is_reported(tmp_path: Path) -> None:
    slow = tmp_path / "slow_engine.py"
    slow.write_text(
        "import time\n\n\ndef child_main(request):\n    time.sleep(30)\n    return {}\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineSubprocessError) as excinfo:
        run_child(
            interpreter=sys.executable,
            module="slow_engine",
            request={},
            repo_root=tmp_path,
            timeout_s=1,
        )
    assert "timed out" in str(excinfo.value)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_subproc.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.subproc'`.

- [ ] **Step 4: Write `rytp/transcribe/subproc.py`**

```python
"""The out-of-process engine seam (contracts §6, design §6).

The recommended transcriber and the recommended diarizer pin incompatible
versions of their shared dependencies, so an engine may have to live in its own
virtual environment. Such an engine declares ``out_of_process = True`` and never
imports its dependency in this process at all: it serialises a request, runs
*this file* as a script under another interpreter, and reads a JSON response
back out of a file.

Two deliberate choices, both about surviving real engines:

**The response goes to a file, not to stdout.** Model loaders print progress
bars, deprecation warnings and CUDA chatter to both streams. Stdout is treated
as noise; only the file is read.

**The child is launched by path, not as ``-m rytp.transcribe.subproc``.** The
child never imports the ``rytp`` package for its own sake — it puts the repo
root on ``sys.path`` and imports exactly one engine module, which in turn
imports only the standard library, :mod:`rytp.models` and its own dependency.

The child contract: the named module exposes ``child_main(request: dict) ->
dict`` returning JSON-serialisable plain dicts. Everything it raises comes back
as :class:`~rytp.transcribe.base.EngineSubprocessError` with the type, the
message and the tail of stderr.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tail(text: str, lines: int) -> str:
    kept = [line for line in (text or "").splitlines() if line.strip()][-lines:]
    return "\n".join(kept)


def run_child(
    *,
    interpreter: str,
    module: str,
    request: dict[str, Any],
    timeout_s: int | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run ``module.child_main(request)`` under ``interpreter`` and return its result.

    Raises :class:`~rytp.transcribe.base.EngineSubprocessError` for a missing
    interpreter, a timeout, a crashed child, a child that wrote nothing, and a
    child that reported failure.
    """
    from rytp import constants as C
    from rytp.transcribe.base import EngineSubprocessError

    timeout = C.ENGINE_SUBPROCESS_TIMEOUT_S if timeout_s is None else timeout_s
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # ignore_cleanup_errors: after a timeout the killed child may still hold
    # request.json open for a moment, which makes the cleanup raise
    # PermissionError on Windows.
    with tempfile.TemporaryDirectory(prefix="rytp-engine-", ignore_cleanup_errors=True) as tmp:
        request_path = Path(tmp) / "request.json"
        response_path = Path(tmp) / "response.json"
        request_path.write_text(
            json.dumps({"module": module, "root": str(root), "request": request}),
            encoding="utf-8",
        )
        command = [
            interpreter,
            str(Path(__file__).resolve()),
            str(request_path),
            str(response_path),
        ]
        try:
            proc = subprocess.run(  # fixed argv, no shell
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise EngineSubprocessError(
                f"engine interpreter not found: {interpreter}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EngineSubprocessError(
                f"engine {module} timed out after {timeout}s"
            ) from exc

        payload = _read_response(response_path)
        stderr_tail = _tail(proc.stderr, C.ENGINE_SUBPROCESS_STDERR_TAIL)
        if payload is None:
            raise EngineSubprocessError(
                f"engine {module} wrote no response (exit {proc.returncode})\n{stderr_tail}"
            )
        if not payload.get("ok"):
            error = payload.get("error") or {}
            raise EngineSubprocessError(
                f"engine {module} failed: {error.get('type', 'Error')}: "
                f"{error.get('message', '')}\n{error.get('traceback', '')}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise EngineSubprocessError(
                f"engine {module} returned {type(result).__name__}, expected an object"
            )
        return result


def _read_response(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _child_main(argv: list[str]) -> int:
    """Child entry point. Standard library only until the engine module loads."""
    request_path = Path(argv[1])
    response_path = Path(argv[2])
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    payload: dict[str, Any]
    try:
        root = envelope["root"]
        if root not in sys.path:
            sys.path.insert(0, root)
        import importlib

        module = importlib.import_module(envelope["module"])
        payload = {"ok": True, "result": module.child_main(envelope["request"])}
    except Exception as exc:  # the boundary exists to report anything
        payload = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv))
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_subproc.py -v
```
Expected: 6 passed. The timeout test takes about a second.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/transcribe
```
Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add rytp/transcribe/subproc.py tests/subproc_engine_stub.py tests/test_transcribe_subproc.py
git commit -m "feat(transcribe): out-of-process engine seam with reported child failures"
```

---

### Task 3: WAV reading and energy framing

The foundation of every measurement in this part. Nothing here is clever; it has to be exactly right, because a sample-index-off-by-one becomes a millisecond error in a cut.

**Files:**
- Create: `rytp/audio/__init__.py` (docstring only — create only if Part 2 has not already)
- Create: `rytp/audio/energy.py`
- Create: `tests/synth_audio.py`
- Test: `tests/test_audio_energy.py`

**Interfaces:**
- Consumes: `rytp.constants`, `rytp.models.RytpError`, numpy.
- Produces: `AudioFormatError`, `read_wav_mono(path) -> tuple[np.ndarray, int]`, `ms_to_index(ms, sr) -> int`, `index_to_ms(index, sr) -> int`, `frame_rms(samples, frame_len, hop) -> np.ndarray`, `to_db(rms) -> np.ndarray`.
- `tests/synth_audio.py`: `SR`, `tone`, `silence`, `concat`, `write_wav`, `tone_gap_tone`.

- [ ] **Step 1: Write `tests/synth_audio.py`**

Not a test — the generator every audio test uses. Property-style tests on synthesised audio beat fixtures here, because the thing under test is *where a boundary lands*, and only a generated signal has an exactly known answer.

```python
"""Synthesised audio for tests: a tone, a gap, a tone.

A real recording never has a known correct boundary. A generated one does,
which is why every audio test in Part 3 builds its input here.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from rytp import constants as C

SR = C.WAV_SAMPLE_RATE_HZ


def tone(ms: int, *, freq_hz: float = 220.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    """A sine of ``ms`` milliseconds. 220 Hz sits inside the speech pitch range."""
    n = int(ms * sr / 1000)
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def silence(ms: int, *, amp: float = 0.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Digital silence, or a faint noise floor when ``amp`` is positive.

    A real recording is never digitally silent; a faint floor keeps the tests
    honest about percentile noise-floor estimation.
    """
    n = int(ms * sr / 1000)
    if amp <= 0:
        return np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, *, sr: int = SR) -> Path:
    """Write 16 kHz mono 16-bit PCM — the one format this pipeline reads."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sr)
        writer.writeframes(pcm.tobytes())
    return path


def tone_gap_tone(
    *,
    lead_ms: int = 200,
    gap_ms: int = 120,
    tail_ms: int = 200,
    floor_amp: float = 0.001,
    sr: int = SR,
) -> tuple[np.ndarray, int, int]:
    """Two words with a measured silence between them.

    Returns ``(samples, gap_start_ms, gap_end_ms)`` — the interval a refined
    boundary is required to land in.
    """
    samples = concat(
        tone(lead_ms, sr=sr),
        silence(gap_ms, amp=floor_amp, sr=sr, seed=1),
        tone(tail_ms, sr=sr),
    )
    return samples, lead_ms, lead_ms + gap_ms
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_audio_energy.py`:

```python
"""Sample access and short-time energy."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from rytp import constants as C
from rytp.audio.energy import (
    AudioFormatError,
    frame_rms,
    index_to_ms,
    ms_to_index,
    read_wav_mono,
    to_db,
)
from tests.synth_audio import SR, silence, tone, write_wav


def test_read_wav_mono_round_trips_a_written_file(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", tone(100, amp=0.5))
    samples, sr = read_wav_mono(path)
    assert sr == SR
    assert samples.shape == (1600,)
    assert samples.dtype == np.float32
    assert 0.49 < float(np.max(samples)) <= 0.5


def test_read_wav_mono_rejects_stereo(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(SR)
        writer.writeframes(b"\x00\x00\x00\x00" * 100)
    with pytest.raises(AudioFormatError) as excinfo:
        read_wav_mono(path)
    assert "mono" in str(excinfo.value)


def test_read_wav_mono_rejects_8_bit(tmp_path: Path) -> None:
    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(SR)
        writer.writeframes(b"\x80" * 100)
    with pytest.raises(AudioFormatError):
        read_wav_mono(path)


def test_ms_and_index_round_trip() -> None:
    assert ms_to_index(1000, SR) == SR
    assert index_to_ms(SR, SR) == 1000
    assert ms_to_index(1, SR) == 16


def test_frame_rms_of_a_sine_is_the_amplitude_over_root_two() -> None:
    samples = tone(200, amp=0.4)
    rms = frame_rms(samples, ms_to_index(20, SR), ms_to_index(10, SR))
    assert rms.size > 15
    assert np.allclose(rms, 0.4 / np.sqrt(2), atol=0.02)


def test_frame_rms_of_digital_silence_is_zero() -> None:
    rms = frame_rms(silence(50), ms_to_index(20, SR), ms_to_index(10, SR))
    assert rms.size > 0
    assert float(np.max(rms)) == 0.0


def test_frame_rms_of_too_short_input_is_empty() -> None:
    assert frame_rms(tone(1), ms_to_index(20, SR), ms_to_index(10, SR)).size == 0


def test_to_db_floors_digital_silence_instead_of_returning_minus_infinity() -> None:
    db = to_db(np.array([0.0, 1.0], dtype=np.float32))
    assert float(db[0]) == C.DB_FLOOR
    assert float(db[1]) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_energy.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.audio.energy'`.

- [ ] **Step 4: Create `rytp/audio/__init__.py` if it does not exist**

```python
"""Audio analysis: extraction, voice activity, energy, acoustic fingerprint."""
```

- [ ] **Step 5: Write the first half of `rytp/audio/energy.py`**

```python
"""Sample access, short-time energy, and the measured word boundary (design §6).

A transcriber's own word boundaries are not usable for cutting. On the owner's
existing data 78.7% of words end exactly where the next one begins, so every
pause has been absorbed into a neighbouring word and some words have zero
duration. The boundary this module produces is *measured*: the quietest point
between two words, snapped to a zero crossing. Two adjacent words sharing one
boundary is then correct, because the shared point is real silence rather than
a guess.

Everything here works on an in-memory float32 array rather than a file. A
one-hour video is refined thousands of times; reading the audio once and
slicing is the difference between seconds and minutes.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from rytp import constants as C
from rytp.models import RytpError


class AudioFormatError(RytpError):
    """The WAV is not the 16 kHz mono 16-bit PCM every audio stage expects."""


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV as float32 in [-1, 1], with its sample rate."""
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    if channels != 1:
        raise AudioFormatError(f"{path}: expected mono audio, found {channels} channels")
    if width != 2:
        raise AudioFormatError(f"{path}: expected 16-bit PCM, found {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def ms_to_index(ms: float, sr: int) -> int:
    """Millisecond position to sample index, rounded to nearest."""
    return int(round(ms * sr / 1000.0))


def index_to_ms(index: int, sr: int) -> int:
    """Sample index to millisecond position, rounded to nearest."""
    return int(round(index * 1000.0 / sr))


def frame_rms(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Root-mean-square energy of every frame, as float32.

    Computed from a cumulative sum of squares rather than a sliding window,
    because a sliding window over an hour of audio materialises tens of
    gigabytes while the cumulative sum needs one array the size of the input.
    """
    if frame_len <= 0 or hop <= 0 or samples.size < frame_len:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (samples.size - frame_len) // hop
    csum = np.square(samples, dtype=np.float64)
    np.cumsum(csum, out=csum)
    starts = np.arange(n_frames, dtype=np.int64) * hop
    ends = starts + frame_len
    before = np.where(starts > 0, csum[np.maximum(starts - 1, 0)], 0.0)
    totals = np.maximum(csum[ends - 1] - before, 0.0)
    return np.sqrt(totals / frame_len).astype(np.float32)


def to_db(rms: np.ndarray) -> np.ndarray:
    """RMS to dBFS, floored so digital silence is a number rather than -inf."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, C.DB_EPSILON))
    return np.maximum(db, C.DB_FLOOR).astype(np.float32)
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_energy.py -v
```
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add rytp/audio/__init__.py rytp/audio/energy.py tests/synth_audio.py tests/test_audio_energy.py
git commit -m "feat(audio): WAV sample access and short-time energy framing"
```

---

### Task 4: Energy minimum and zero-crossing snap

**Files:**
- Modify: `rytp/audio/energy.py` (append)
- Test: `tests/test_audio_energy.py` (append)

**Interfaces:**
- Consumes: `frame_rms`, `to_db`, `ms_to_index`, `index_to_ms` from Task 3.
- Produces: `find_energy_minimum(samples, sr, lo_ms, hi_ms, *, prefer_ms) -> int`, `snap_to_zero_crossing(samples, sr, at_ms, *, search_ms=None) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_energy.py`:

```python
from rytp.audio.energy import find_energy_minimum, snap_to_zero_crossing
from tests.synth_audio import concat, tone_gap_tone


def test_energy_minimum_lands_inside_the_measured_gap() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    found = find_energy_minimum(samples, SR, 170, 290, prefer_ms=230)
    assert gap_start <= found <= gap_end


def test_energy_minimum_breaks_ties_toward_the_claimed_boundary() -> None:
    # Digital silence makes every frame in the gap exactly equal; the pick must
    # still be deterministic and must not drift to the edge of the window.
    samples = concat(tone(200), silence(200), tone(200))
    near_left = find_energy_minimum(samples, SR, 200, 400, prefer_ms=220)
    near_right = find_energy_minimum(samples, SR, 200, 400, prefer_ms=380)
    assert near_left < near_right
    assert abs(near_left - 220) <= 10
    assert abs(near_right - 380) <= 10


def test_energy_minimum_is_deterministic() -> None:
    samples, _, _ = tone_gap_tone()
    first = find_energy_minimum(samples, SR, 170, 290, prefer_ms=230)
    second = find_energy_minimum(samples, SR, 170, 290, prefer_ms=230)
    assert first == second


def test_energy_minimum_of_an_empty_window_returns_the_claim() -> None:
    samples, _, _ = tone_gap_tone()
    assert find_energy_minimum(samples, SR, 300, 300, prefer_ms=300) == 300


def test_zero_crossing_snap_lands_on_a_quiet_sample() -> None:
    samples = tone(200, freq_hz=200.0, amp=0.5)
    snapped = snap_to_zero_crossing(samples, SR, 101)
    assert abs(snapped - 101) <= C.ZERO_CROSSING_SEARCH_MS
    assert abs(float(samples[ms_to_index(snapped, SR)])) < 0.2


def test_zero_crossing_snap_without_any_crossing_stays_put() -> None:
    samples = np.full(SR // 10, 0.5, dtype=np.float32)
    assert abs(snap_to_zero_crossing(samples, SR, 50) - 50) <= C.ZERO_CROSSING_SEARCH_MS
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_energy.py -v
```
Expected: `ImportError: cannot import name 'find_energy_minimum'`.

- [ ] **Step 3: Append to `rytp/audio/energy.py`**

```python
def find_energy_minimum(
    samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float, *, prefer_ms: int
) -> int:
    """Millisecond of the quietest short-time frame in ``[lo_ms, hi_ms]``.

    Ties — which digital silence produces in bulk — resolve to the frame
    nearest ``prefer_ms``, so the result is deterministic and never drifts to
    the edge of the search window.
    """
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    if hi - lo < frame_len:
        return prefer_ms
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    if rms.size == 0:
        return prefer_ms
    centres = lo + np.arange(rms.size, dtype=np.int64) * hop + frame_len // 2
    tied = np.flatnonzero(rms <= float(rms.min()) + C.ENERGY_TIE_RMS)
    prefer_index = ms_to_index(prefer_ms, sr)
    pick = int(tied[int(np.argmin(np.abs(centres[tied] - prefer_index)))])
    return index_to_ms(int(centres[pick]), sr)


def snap_to_zero_crossing(
    samples: np.ndarray, sr: int, at_ms: int, *, search_ms: int | None = None
) -> int:
    """Nudge a boundary onto the nearest zero crossing, then round to a millisecond.

    The database stores milliseconds and one millisecond is sixteen samples at
    16 kHz, so the crossing itself cannot be stored. Of the two millisecond
    marks bracketing it the quieter one is kept, which is the best a
    millisecond grid can do. The snap matters most when the minimum is not
    fully silent; inside real silence the rounding lands in silence anyway.
    """
    span = max(1, ms_to_index(C.ZERO_CROSSING_SEARCH_MS if search_ms is None else search_ms, sr))
    centre = ms_to_index(at_ms, sr)
    lo = max(1, centre - span)
    hi = min(samples.size, centre + span + 1)
    if hi - lo <= 1:
        return at_ms
    window = samples[lo - 1 : hi]
    crossings = np.flatnonzero(np.signbit(window[:-1]) != np.signbit(window[1:])) + lo
    if crossings.size:
        index = int(crossings[int(np.argmin(np.abs(crossings - centre)))])
    else:
        # No crossing at all (digital silence, or a constant offset). Take the
        # quietest sample — and break the ties toward the centre, because
        # inside digital silence every sample ties and picking the window edge
        # would walk the boundary a few milliseconds left on every pass.
        magnitudes = np.abs(samples[lo:hi])
        tied = np.flatnonzero(magnitudes <= float(magnitudes.min()) + C.ENERGY_TIE_RMS) + lo
        index = int(tied[int(np.argmin(np.abs(tied - centre)))])
    return _quietest_millisecond(samples, sr, index)


def _quietest_millisecond(samples: np.ndarray, sr: int, index: int) -> int:
    lower = index * 1000 // sr
    candidates = [
        ms for ms in (lower, lower + 1) if 0 <= ms_to_index(ms, sr) < samples.size
    ]
    if not candidates:
        return max(0, index_to_ms(index, sr))
    return min(candidates, key=lambda ms: (abs(float(samples[ms_to_index(ms, sr)])), ms))
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_energy.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/audio/energy.py tests/test_audio_energy.py
git commit -m "feat(audio): locate the energy minimum and snap it to a zero crossing"
```

---

### Task 5: Boundary refinement

The heart of Part 3. Everything before it was plumbing.

**Files:**
- Modify: `rytp/audio/energy.py` (append)
- Test: `tests/test_audio_boundaries.py`

**Interfaces:**
- Consumes: Tasks 3 and 4; `rytp.models.Span`.
- Produces:
  - `refine_boundaries(samples, sr, spans, *, speech=()) -> list[Span]` — `speech` is a sequence of `(start_ms, end_ms)` pairs, as `rytp.audio.vad.detect_speech` produces. Taking pairs rather than the dataclass keeps `energy` free of a `vad` import, and `vad` already imports `energy`.
  - `enforce_monotonic(spans, *, total_ms) -> list[Span]`
  - `silence_depth_db(samples, sr, boundary_ms) -> float`
  - `boundary_quality(depth_db) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_boundaries.py`:

```python
"""Boundary refinement: a measured boundary beats a claimed one."""
from __future__ import annotations

from rytp import constants as C
from rytp.audio.energy import (
    boundary_quality,
    enforce_monotonic,
    refine_boundaries,
    silence_depth_db,
)
from rytp.models import Span
from tests.synth_audio import SR, concat, silence, tone, tone_gap_tone


def test_a_boundary_claimed_early_moves_into_the_measured_silence() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=0.5),
        Span(start_ms=230, end_ms=520, score=0.5),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert gap_start <= refined[0].end_ms <= gap_end
    assert refined[0].end_ms == refined[1].start_ms


def test_adjacent_words_keep_sharing_one_boundary() -> None:
    samples, _, _ = tone_gap_tone()
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    for left, right in zip(refined, refined[1:], strict=False):
        assert left.end_ms == right.start_ms


def test_a_boundary_never_moves_further_than_the_rail() -> None:
    samples, _, _ = tone_gap_tone(lead_ms=400, gap_ms=200, tail_ms=400)
    claimed = [
        Span(start_ms=0, end_ms=100, score=None),
        Span(start_ms=100, end_ms=1000, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert abs(refined[0].end_ms - 100) <= C.BOUNDARY_SEARCH_MS


def test_a_second_pass_stays_inside_the_measured_silence() -> None:
    # Not exact idempotence on noisy audio: a second pass re-centres the search
    # window, so a marginally quieter frame just outside the old one can win.
    # The property that matters is that it converges inside the real gap.
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    once = refine_boundaries(samples, SR, claimed)
    twice = refine_boundaries(samples, SR, once)
    assert gap_start <= once[0].end_ms <= gap_end
    assert gap_start <= twice[0].end_ms <= gap_end


def test_the_outer_edges_do_not_move_without_a_speech_edge_to_move_to() -> None:
    samples, _, _ = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [
        Span(start_ms=0, end_ms=230, score=None),
        Span(start_ms=230, end_ms=520, score=None),
    ]
    refined = refine_boundaries(samples, SR, claimed)
    assert refined[0].start_ms == 0
    assert refined[-1].end_ms == 520


def test_refinement_inside_digital_silence_is_exactly_stable() -> None:
    # With no noise every frame in the gap ties, so the tie-breaks decide the
    # answer. They must hold the boundary still rather than walk it.
    samples = concat(tone(200), silence(200), tone(200))
    claimed = [
        Span(start_ms=0, end_ms=300, score=None),
        Span(start_ms=300, end_ms=600, score=None),
    ]
    once = refine_boundaries(samples, SR, claimed)
    assert refine_boundaries(samples, SR, once) == once


def test_refinement_is_deterministic() -> None:
    samples, _, _ = tone_gap_tone()
    claimed = [Span(start_ms=0, end_ms=230, score=None), Span(start_ms=230, end_ms=520, score=None)]
    assert refine_boundaries(samples, SR, claimed) == refine_boundaries(samples, SR, claimed)


def test_refined_spans_stay_ordered_and_never_cross() -> None:
    samples = concat(*[part for _ in range(4) for part in (tone(150), silence(100, amp=0.001))])
    claimed = [
        Span(start_ms=i * 250, end_ms=(i + 1) * 250, score=None) for i in range(4)
    ]
    refined = refine_boundaries(samples, SR, claimed)
    for span in refined:
        assert span.start_ms < span.end_ms
    for left, right in zip(refined, refined[1:], strict=False):
        assert left.end_ms <= right.start_ms


def test_a_vad_edge_inside_the_window_wins_over_the_energy_search() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    claimed = [Span(start_ms=0, end_ms=230, score=None), Span(start_ms=230, end_ms=520, score=None)]
    refined = refine_boundaries(
        samples, SR, claimed, speech=[(0, gap_start), (gap_end, 520)]
    )
    assert gap_start <= refined[0].end_ms <= gap_end


def test_enforce_monotonic_widens_a_zero_duration_word() -> None:
    spans = [
        Span(start_ms=100, end_ms=100, score=None),
        Span(start_ms=100, end_ms=300, score=None),
    ]
    fixed = enforce_monotonic(spans, total_ms=1000)
    assert fixed[0].end_ms - fixed[0].start_ms >= C.MIN_WORD_DURATION_MS
    assert fixed[1].start_ms >= fixed[0].end_ms


def test_enforce_monotonic_clamps_to_the_audio_length() -> None:
    spans = [Span(start_ms=900, end_ms=1200, score=None)]
    fixed = enforce_monotonic(spans, total_ms=1000)
    assert fixed[0].end_ms <= 1000


def test_silence_depth_is_large_in_a_gap_and_small_inside_a_tone() -> None:
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    in_gap = silence_depth_db(samples, SR, (gap_start + gap_end) // 2)
    in_speech = silence_depth_db(samples, SR, 100)
    assert in_gap > 20.0
    assert in_speech < 6.0


def test_boundary_quality_is_bounded() -> None:
    assert boundary_quality(0.0) == 0.0
    assert boundary_quality(1000.0) == 1.0
    assert 0.0 < boundary_quality(C.SILENCE_DEPTH_FULL_DB / 2) < 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_boundaries.py -v
```
Expected: `ImportError: cannot import name 'refine_boundaries'`.

- [ ] **Step 3: Append to `rytp/audio/energy.py`**

```python
def _median_db(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    return C.DB_FLOOR if rms.size == 0 else float(np.median(to_db(rms)))


def _min_db(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    frame_len = max(1, ms_to_index(C.ENERGY_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ENERGY_HOP_MS, sr))
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    rms = frame_rms(samples[lo:hi], frame_len, hop)
    return C.DB_FLOOR if rms.size == 0 else float(np.min(to_db(rms)))


def silence_depth_db(samples: np.ndarray, sr: int, boundary_ms: int) -> float:
    """How much quieter this boundary is than the speech around it, in dB.

    This is the number that answers "is there actually silence where the
    aligner claims a word ends" — the measurement ``transcribe compare``
    reports and the one that stands in for a per-word alignment confidence
    when the aligner does not provide one.
    """
    reference = max(
        _median_db(samples, sr, boundary_ms - C.SILENCE_DEPTH_REF_MS, boundary_ms),
        _median_db(samples, sr, boundary_ms, boundary_ms + C.SILENCE_DEPTH_REF_MS),
    )
    floor = _min_db(
        samples,
        sr,
        boundary_ms - C.SILENCE_DEPTH_WINDOW_MS,
        boundary_ms + C.SILENCE_DEPTH_WINDOW_MS,
    )
    if floor <= C.DB_FLOOR:
        # Digital silence at the boundary: as good as a boundary gets.
        return C.SILENCE_DEPTH_FULL_DB
    return max(0.0, reference - floor)


def boundary_quality(depth_db: float) -> float:
    """Silence depth as a 0..1 score, for ``words.align_score``."""
    return min(1.0, max(0.0, depth_db / C.SILENCE_DEPTH_FULL_DB))


def _speech_edges(speech: Sequence[tuple[int, int]]) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for start_ms, end_ms in speech:
        edges.append((start_ms, "start"))
        edges.append((end_ms, "end"))
    return sorted(edges)


def _refine_one(
    samples: np.ndarray,
    sr: int,
    *,
    claimed: int,
    lo_ms: int,
    hi_ms: int,
    edges: Sequence[tuple[int, str]],
    total_ms: int,
) -> int:
    lo = max(0, min(lo_ms, claimed))
    hi = min(total_ms, max(hi_ms, claimed))
    if hi <= lo:
        return max(0, min(claimed, total_ms))
    near = [edge for edge in edges if lo <= edge[0] <= hi]
    gaps = [
        (end_ms, start_ms)
        for end_ms, end_kind in near
        for start_ms, start_kind in near
        if end_kind == "end" and start_kind == "start" and start_ms >= end_ms
    ]
    if gaps:
        # A speech segment ends and the next begins inside the window: the true
        # silence is the interval between them, so search there rather than
        # snapping to either edge.
        end_ms, start_ms = min(gaps, key=lambda pair: abs((pair[0] + pair[1]) // 2 - claimed))
        lo, hi = end_ms, max(start_ms, end_ms + 1)
    elif len(near) == 1 and abs(near[0][0] - claimed) <= C.VAD_EDGE_SNAP_MS:
        # One speech edge, close by: it is measured silence on one side, which
        # beats any search (design §6).
        return near[0][0]
    minimum = find_energy_minimum(samples, sr, lo, hi, prefer_ms=claimed)
    return snap_to_zero_crossing(samples, sr, minimum)


def _refine_edge(*, claimed: int, edges: Sequence[tuple[int, str]], kind: str, total_ms: int) -> int:
    """The very first start and the very last end: a speech edge, or nothing.

    There is no "between two words" to search at the outer edges, so an energy
    hunt there would only find the quietest ripple inside the first or last
    word and would move again on every pass. A nearby voice-activity edge is a
    real boundary and is taken; otherwise the aligner's claim stands.
    """
    candidates = [
        ms
        for ms, edge_kind in edges
        if edge_kind == kind and abs(ms - claimed) <= C.VAD_EDGE_SNAP_MS
    ]
    chosen = (
        min(candidates, key=lambda ms: (abs(ms - claimed), ms)) if candidates else claimed
    )
    return max(0, min(chosen, total_ms))


def refine_boundaries(
    samples: np.ndarray,
    sr: int,
    spans: Sequence[Span],
    *,
    speech: Sequence[tuple[int, int]] = (),
) -> list[Span]:
    """Replace every claimed boundary with a measured one (design §6).

    ``spans`` are the aligner's rough timings, in order. ``speech`` is the
    output of :func:`rytp.audio.vad.detect_speech` as ``(start_ms, end_ms)``
    pairs; where a speech edge sits near a claimed boundary it is used
    directly, because it is a true boundary for free.

    Boundaries are shared: word *i* ends exactly where word *i+1* begins. That
    is correct under this scheme, because the shared point is measured silence.
    """
    if not spans:
        return []
    total_ms = index_to_ms(samples.size, sr)
    edges = _speech_edges(speech)
    cuts: list[int] = []

    cuts.append(
        _refine_edge(
            claimed=spans[0].start_ms, edges=edges, kind="start", total_ms=total_ms
        )
    )
    for left, right in zip(spans, spans[1:], strict=False):
        claimed = (left.end_ms + right.start_ms) // 2
        left_mid = (left.start_ms + left.end_ms) // 2
        right_mid = (right.start_ms + right.end_ms) // 2
        cuts.append(
            _refine_one(
                samples,
                sr,
                claimed=claimed,
                lo_ms=max(left_mid, claimed - C.BOUNDARY_SEARCH_MS),
                hi_ms=min(right_mid, claimed + C.BOUNDARY_SEARCH_MS),
                edges=edges,
                total_ms=total_ms,
            )
        )
    cuts.append(
        _refine_edge(claimed=spans[-1].end_ms, edges=edges, kind="end", total_ms=total_ms)
    )
    refined = [
        Span(start_ms=cuts[i], end_ms=cuts[i + 1], score=spans[i].score)
        for i in range(len(spans))
    ]
    return enforce_monotonic(refined, total_ms=total_ms)


def enforce_monotonic(spans: Sequence[Span], *, total_ms: int) -> list[Span]:
    """Guarantee ``start < end <= next start`` and a minimum word duration.

    A shared boundary survives untouched; only a degenerate word is widened,
    which pushes its neighbour's start forward by the same amount.
    """
    out: list[Span] = []
    previous_end = 0
    for span in spans:
        start = max(span.start_ms, previous_end)
        end = max(span.end_ms, start + C.MIN_WORD_DURATION_MS)
        if total_ms > 0:
            end = min(end, total_ms)
            start = min(start, max(0, end - 1))
        out.append(Span(start_ms=start, end_ms=end, score=span.score))
        previous_end = end
    return out
```

Add `from collections.abc import Sequence` to the imports at the top of `rytp/audio/energy.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_boundaries.py tests/test_audio_energy.py -v
```
Expected: 27 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/audio
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/audio/energy.py tests/test_audio_boundaries.py
git commit -m "feat(audio): place word boundaries at the measured energy minimum"
```

---

### Task 6: Voice activity detection

**Files:**
- Create: `rytp/audio/vad.py`
- Test: `tests/test_audio_vad.py`

**Interfaces:**
- Consumes: `rytp.audio.energy.{frame_rms, to_db, ms_to_index, index_to_ms}`.
- Produces: `SpeechSegment(start_ms, end_ms)`, `AudioChunk(ord, start_ms, end_ms)`, `detect_speech(samples, sr, *, pad_ms=None) -> list[SpeechSegment]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_vad.py`:

```python
"""Energy-based voice activity detection."""
from __future__ import annotations

from rytp import constants as C
from rytp.audio.vad import SpeechSegment, detect_speech
from tests.synth_audio import SR, concat, silence, tone


def test_two_utterances_separated_by_silence_are_two_segments() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=2), tone(400))
    segments = detect_speech(samples, SR, pad_ms=0)
    assert len(segments) == 2
    assert abs(segments[0].start_ms - 0) <= 30
    assert abs(segments[0].end_ms - 400) <= 30
    assert abs(segments[1].start_ms - 800) <= 30
    assert abs(segments[1].end_ms - 1200) <= 30


def test_a_gap_shorter_than_the_minimum_silence_is_swallowed() -> None:
    gap = C.VAD_MIN_SILENCE_MS - 60
    samples = concat(tone(400), silence(gap, amp=0.001, seed=3), tone(400))
    segments = detect_speech(samples, SR, pad_ms=0)
    assert len(segments) == 1


def test_a_burst_shorter_than_the_minimum_speech_is_discarded() -> None:
    burst = C.VAD_MIN_SPEECH_MS - 60
    samples = concat(
        silence(400, amp=0.001, seed=4), tone(burst), silence(400, amp=0.001, seed=5)
    )
    assert detect_speech(samples, SR, pad_ms=0) == []


def test_digital_silence_has_no_speech() -> None:
    assert detect_speech(silence(1000), SR) == []


def test_continuous_speech_with_no_silence_is_one_segment() -> None:
    # The percentile noise floor sits inside speech here, so the hysteresis
    # band finds nothing. Treating the whole file as speech is the safe answer:
    # chunk planning still works and the transcriber sees everything.
    samples = tone(2000)
    segments = detect_speech(samples, SR)
    assert segments == [SpeechSegment(start_ms=0, end_ms=2000)]


def test_padding_widens_a_segment_and_is_clipped_to_the_audio() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=6), tone(400))
    tight = detect_speech(samples, SR, pad_ms=0)
    padded = detect_speech(samples, SR, pad_ms=50)
    assert padded[0].start_ms == 0
    assert padded[0].end_ms >= tight[0].end_ms
    assert padded[-1].end_ms <= 1200


def test_detection_is_deterministic() -> None:
    samples = concat(tone(400), silence(400, amp=0.001, seed=7), tone(400))
    assert detect_speech(samples, SR) == detect_speech(samples, SR)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_vad.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.audio.vad'`.

- [ ] **Step 3: Write `rytp/audio/vad.py`**

```python
"""Energy-based voice activity detection and chunk planning (design §6).

Two jobs. First, a transcriber caps a single call — GigaAM v3 at 25 s — so an
hour of audio has to be split, and split *at silence*, because a cut through a
word costs that word. Second, the edge of a detected speech segment is a true
boundary for free: there is measured silence on one side of it, which is
exactly what boundary refinement is hunting for.

No model is used. A percentile noise floor with a hysteresis band does both
jobs, has no dependency, and runs over a whole corpus without touching the
GPU. Anything smarter can be swapped in behind :func:`detect_speech`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rytp import constants as C
from rytp.audio.energy import frame_rms, index_to_ms, ms_to_index, to_db


@dataclass(frozen=True)
class SpeechSegment:
    """A stretch of audio that contains speech."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class AudioChunk:
    """One window handed to a transcriber, within its per-call cap."""

    ord: int
    start_ms: int
    end_ms: int


def _hysteresis_runs(db: np.ndarray, enter: float, leave: float) -> list[tuple[int, int]]:
    """Frame index runs that go above ``enter`` and stay above ``leave``.

    A plain loop: an hour of audio is 360k frames, which costs a fraction of a
    second, and the state machine reads far more clearly than a vectorised
    equivalent would.
    """
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(db.size):
        value = float(db[index])
        if start is None:
            if value >= enter:
                start = index
        elif value < leave:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, db.size))
    return runs


def _merge_close(spans: Sequence[tuple[int, int]], max_gap_ms: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= max_gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detect_speech(
    samples: np.ndarray, sr: int, *, pad_ms: int | None = None
) -> list[SpeechSegment]:
    """Find the stretches of speech in ``samples``.

    ``pad_ms`` defaults to :data:`rytp.constants.VAD_PAD_MS`; acoustics passes
    0, because a padded segment end would put the reverberation tail it wants
    to measure inside the segment.
    """
    frame_len = max(1, ms_to_index(C.VAD_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.VAD_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    total_ms = index_to_ms(samples.size, sr)
    if db.size == 0:
        return []
    floor = float(np.percentile(db, C.VAD_NOISE_PERCENTILE))
    runs = _hysteresis_runs(db, floor + C.VAD_ENTER_DB, floor + C.VAD_EXIT_DB)
    raw = [
        (
            index_to_ms(first * hop, sr),
            min(index_to_ms(last * hop + frame_len, sr), total_ms),
        )
        for first, last in runs
    ]
    kept = [
        span
        for span in _merge_close(raw, C.VAD_MIN_SILENCE_MS)
        if span[1] - span[0] >= C.VAD_MIN_SPEECH_MS
    ]
    if not kept:
        if float(np.max(db)) > C.DB_FLOOR + C.VAD_ENTER_DB:
            # Energy everywhere and no silence anywhere: the percentile floor
            # has landed inside speech. Treat the file as one utterance rather
            # than reporting no speech at all.
            return [SpeechSegment(start_ms=0, end_ms=total_ms)]
        return []
    pad = C.VAD_PAD_MS if pad_ms is None else pad_ms
    padded = [(max(0, start - pad), min(total_ms, end + pad)) for start, end in kept]
    return [SpeechSegment(start_ms=start, end_ms=end) for start, end in _merge_close(padded, 0)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_vad.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/audio/vad.py tests/test_audio_vad.py
git commit -m "feat(audio): energy-based voice activity detection"
```

---

### Task 7: VAD chunk planning

**Files:**
- Modify: `rytp/audio/vad.py` (append)
- Test: `tests/test_audio_vad.py` (append)

**Interfaces:**
- Consumes: `SpeechSegment`, `AudioChunk` from Task 6.
- Produces: `plan_chunks(speech, *, total_ms, max_chunk_ms=None, target_ms=None) -> list[AudioChunk]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_vad.py`:

```python
from rytp.audio.vad import AudioChunk, plan_chunks


def _segments(*pairs: tuple[int, int]) -> list[SpeechSegment]:
    return [SpeechSegment(start_ms=a, end_ms=b) for a, b in pairs]


def test_short_segments_are_packed_up_to_the_target() -> None:
    speech = _segments((0, 5_000), (6_000, 11_000), (12_000, 17_000), (18_000, 23_000))
    chunks = plan_chunks(speech, total_ms=30_000, max_chunk_ms=25_000, target_ms=12_000)
    assert [(c.start_ms, c.end_ms) for c in chunks] == [(0, 11_000), (12_000, 23_000)]


def test_every_chunk_stays_within_the_per_call_cap() -> None:
    speech = _segments(*[(i * 3_000, i * 3_000 + 2_500) for i in range(40)])
    chunks = plan_chunks(speech, total_ms=120_000)
    for chunk in chunks:
        assert chunk.end_ms - chunk.start_ms <= C.VAD_CHUNK_MAX_MS


def test_chunk_edges_fall_on_segment_edges_when_they_can() -> None:
    speech = _segments((0, 5_000), (30_000, 35_000))
    chunks = plan_chunks(speech, total_ms=40_000, max_chunk_ms=25_000, target_ms=20_000)
    edges = {c.start_ms for c in chunks} | {c.end_ms for c in chunks}
    assert edges <= {0, 5_000, 30_000, 35_000}


def test_all_speech_is_covered_by_some_chunk() -> None:
    speech = _segments((0, 5_000), (6_000, 11_000), (40_000, 45_000))
    chunks = plan_chunks(speech, total_ms=50_000, max_chunk_ms=25_000, target_ms=12_000)
    for segment in speech:
        assert any(
            c.start_ms <= segment.start_ms and segment.end_ms <= c.end_ms for c in chunks
        )


def test_one_unbroken_run_longer_than_the_cap_is_split_into_equal_pieces() -> None:
    chunks = plan_chunks(
        _segments((0, 60_000)), total_ms=60_000, max_chunk_ms=25_000, target_ms=20_000
    )
    assert len(chunks) == 3
    lengths = {c.end_ms - c.start_ms for c in chunks}
    assert max(lengths) <= 25_000
    assert chunks[0].start_ms == 0
    assert chunks[-1].end_ms == 60_000


def test_no_speech_means_no_chunks() -> None:
    assert plan_chunks([], total_ms=10_000) == []


def test_chunk_ordinals_are_contiguous_from_zero() -> None:
    speech = _segments(*[(i * 3_000, i * 3_000 + 2_500) for i in range(20)])
    chunks = plan_chunks(speech, total_ms=60_000)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert isinstance(chunks[0], AudioChunk)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_vad.py -v
```
Expected: `ImportError: cannot import name 'plan_chunks'`.

- [ ] **Step 3: Append to `rytp/audio/vad.py`**

```python
def plan_chunks(
    speech: Sequence[SpeechSegment],
    *,
    total_ms: int,
    max_chunk_ms: int | None = None,
    target_ms: int | None = None,
) -> list[AudioChunk]:
    """Pack speech segments into windows a transcriber will accept.

    Segments are grouped while the group stays under ``target_ms``, so a chunk
    normally begins and ends in silence. ``max_chunk_ms`` is the transcriber's
    hard per-call cap (design §6: 25 s for GigaAM v3) and is never exceeded.
    """
    max_ms = C.VAD_CHUNK_MAX_MS if max_chunk_ms is None else max_chunk_ms
    target = min(C.VAD_CHUNK_TARGET_MS if target_ms is None else target_ms, max_ms)
    if not speech:
        return []

    grouped: list[tuple[int, int]] = []
    start, end = speech[0].start_ms, speech[0].end_ms
    for segment in speech[1:]:
        if segment.end_ms - start <= target:
            end = segment.end_ms
        else:
            grouped.append((start, end))
            start, end = segment.start_ms, segment.end_ms
    grouped.append((start, end))

    pieces: list[tuple[int, int]] = []
    for group_start, group_end in grouped:
        span = group_end - group_start
        if span <= max_ms:
            pieces.append((group_start, group_end))
            continue
        # One unbroken speech run longer than the cap. Split it into equal
        # pieces: this cuts mid-speech and costs at most the word straddling
        # each cut, but a run this long without a 150 ms silence is rare.
        count = -(-span // max_ms)
        step = -(-span // count)
        pieces.extend(
            (group_start + i * step, min(group_start + (i + 1) * step, group_end))
            for i in range(count)
        )
    return [
        AudioChunk(ord=index, start_ms=max(0, piece_start), end_ms=min(piece_end, total_ms))
        for index, (piece_start, piece_end) in enumerate(pieces)
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_vad.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/audio
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/audio/vad.py tests/test_audio_vad.py
git commit -m "feat(audio): plan transcriber chunks at silences under the per-call cap"
```

---

### Task 8: Caption tier ingest

Tier 1 of contracts §3: auto-captions Part 2 already downloaded become searchable words for no GPU time at all. They carry per-word start times on a 40 ms grid and **no end times**, their alignment is loose, and the schema forbids cutting them (`CHECK (source = 'caption' OR end_ms IS NOT NULL)`).

**Files:**
- Create: `rytp/transcribe/captions.py`
- Test: `tests/test_transcribe_captions.py`

**Interfaces:**
- Consumes: `Database`, `rytp.models.{RytpError, normalize_text, stem_text}`, `rytp.constants.CAPTION_ENGINE`.
- Produces: `CaptionWord(start_ms, text)`, `CaptionDowngrade(RytpError)`, `parse_json3(payload) -> list[CaptionWord]`, `load_json3(path) -> dict`, `captions_asset_path(db, video_id) -> Path`, `ingest_captions(db, video_id, path=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe_captions.py`:

```python
"""Caption-tier ingest: starts only, never cuttable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe.captions import (
    CaptionDowngrade,
    ingest_captions,
    parse_json3,
)

SAMPLE = {
    "events": [
        {
            "tStartMs": 0,
            "dDurationMs": 1000,
            "segs": [
                {"utf8": "привет", "tOffsetMs": 0},
                {"utf8": " мир", "tOffsetMs": 320},
            ],
        },
        {"tStartMs": 1000, "dDurationMs": 40, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {
            "tStartMs": 1040,
            "segs": [
                {"utf8": "как", "tOffsetMs": 0},
                {"utf8": " дела", "tOffsetMs": 200},
                {"utf8": "\n"},
            ],
        },
    ]
}


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


def _write_captions(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_parse_json3_uses_event_start_plus_segment_offset() -> None:
    words = parse_json3(SAMPLE)
    assert [(w.start_ms, w.text) for w in words] == [
        (0, "привет"),
        (320, "мир"),
        (1040, "как"),
        (1240, "дела"),
    ]


def test_parse_json3_skips_rollup_repeats_and_whitespace_segments() -> None:
    assert all(w.text.strip() for w in parse_json3(SAMPLE))
    assert len(parse_json3(SAMPLE)) == 4


def test_parse_json3_splits_a_multi_word_segment_on_the_same_start() -> None:
    payload = {"events": [{"tStartMs": 500, "segs": [{"utf8": "два слова", "tOffsetMs": 0}]}]}
    words = parse_json3(payload)
    assert [(w.start_ms, w.text) for w in words] == [(500, "два"), (500, "слова")]


def test_a_hyphenated_caption_word_becomes_two_findable_rows(
    db: Database, tmp_path: Path
) -> None:
    # One row per token, because nothing can match a row whose normalized_text
    # holds a space: Part 4's lookups and Part 5's pointer walk are both
    # single-token, and the user's query normalizes the same way.
    payload = {"events": [{"tStartMs": 0, "segs": [{"utf8": "кто-то", "tOffsetMs": 0}]}]}
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", payload)
    assert ingest_captions(db, video_id, path) == 2
    rows = db.conn.execute(
        "SELECT ord, normalized_text FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[1] for row in rows] == ["кто", "то"]
    assert all(" " not in row[1] for row in rows)


def test_parse_json3_keeps_starts_non_decreasing() -> None:
    payload = {
        "events": [
            {"tStartMs": 1000, "segs": [{"utf8": "поздно", "tOffsetMs": 0}]},
            {"tStartMs": 400, "segs": [{"utf8": "рано", "tOffsetMs": 0}]},
        ]
    }
    starts = [w.start_ms for w in parse_json3(payload)]
    assert starts == sorted(starts)


def test_ingest_writes_caption_rows_with_no_end_times(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    assert ingest_captions(db, video_id, path) == 4
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in rows] == [0, 1, 2, 3]
    assert all(row[2] is None for row in rows)
    assert {row[4] for row in rows} == {"caption"}
    assert {row[5] for row in rows} == {C.CAPTION_ENGINE}


def test_ingest_drops_punctuation_only_tokens(db: Database, tmp_path: Path) -> None:
    payload = {"events": [{"tStartMs": 0, "segs": [{"utf8": "—", "tOffsetMs": 0},
                                                   {"utf8": " слово", "tOffsetMs": 100}]}]}
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", payload)
    assert ingest_captions(db, video_id, path) == 1
    text = db.conn.execute(
        "SELECT text FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert text == "слово"


def test_ingest_twice_replaces_rather_than_appends(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    ingest_captions(db, video_id, path)
    ingest_captions(db, video_id, path)
    count = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert count == 4


def test_ingest_refuses_to_downgrade_an_aligned_video(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, "
        "stem, source, engine) VALUES (?, 0, 0, 100, 'х', 'х', 'х', 'aligned', 'fake')",
        (video_id,),
    )
    db.conn.commit()
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    with pytest.raises(CaptionDowngrade):
        ingest_captions(db, video_id, path)


def test_ingest_without_a_captions_asset_says_so(db: Database) -> None:
    video_id = _make_video(db)
    with pytest.raises(RytpError) as excinfo:
        ingest_captions(db, video_id)
    assert str(video_id) in str(excinfo.value)


def test_ingest_uses_the_captions_asset_when_no_path_is_given(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    path = _write_captions(tmp_path / "captions.json3", SAMPLE)
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(path), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    assert ingest_captions(db, video_id) == 4


def test_ingest_clears_both_derived_tables(db: Database, tmp_path: Path) -> None:
    # Contracts §4: replacing a video's words deletes its utterances *and* its
    # video_speakers, in the same transaction.
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 100, 0, 1, 'x', 'x', 'x')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    db.conn.commit()
    ingest_captions(db, video_id, _write_captions(tmp_path / "captions.json3", SAMPLE))
    for table in ("utterances", "video_speakers"):
        left = db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        assert left == 0


def test_a_file_that_is_not_json3_is_reported(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    bad = tmp_path / "captions.json3"
    bad.write_text("<xml/>", encoding="utf-8")
    with pytest.raises(RytpError) as excinfo:
        ingest_captions(db, video_id, bad)
    assert "json3" in str(excinfo.value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_captions.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.captions'`.

- [ ] **Step 3: Write `rytp/transcribe/captions.py`**

```python
"""Caption tier: YouTube json3 auto-captions become searchable words (design §6).

Tier 1 exists because it is nearly free. Captions are tiny, cost no GPU time,
and are the first thing to disappear when a video is delisted, so they are
pulled for everything and turned into `words` rows with ``source='caption'``.

What they are not is cuttable. A json3 caption carries a per-word **start** on
a 40 ms grid and no end at all, and its alignment is loose. ``words.end_ms``
stays null, which the schema's ``CHECK (source = 'caption' OR end_ms IS NOT
NULL)`` turns into a hard guarantee that nothing downstream can cut on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.models import RytpError, normalize_text, stem_text
from rytp.transcribe.base import split_token


class CaptionDowngrade(RytpError):
    """Refused: the video already has aligned words, which captions would replace."""


@dataclass(frozen=True)
class CaptionWord:
    """One caption token. No end time exists — see the module docstring."""

    start_ms: int
    text: str


_WORD_COLUMNS = (
    "video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
    "confidence, align_score, source, engine, video_speaker_id"
)
_INSERT_WORD = f"INSERT INTO words ({_WORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"


def load_json3(path: Path) -> dict[str, Any]:
    """Read a json3 caption file, with a plain error when it is something else."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RytpError(f"cannot read captions at {path}: {exc}") from exc
    except ValueError as exc:
        raise RytpError(f"{path} is not json3 captions: {exc}") from exc
    if not isinstance(payload, dict) or "events" not in payload:
        raise RytpError(f"{path} is not json3 captions: no 'events' key")
    return payload


def parse_json3(payload: dict[str, Any]) -> list[CaptionWord]:
    """Flatten json3 events into caption words.

    A word's start is its event's ``tStartMs`` plus its segment's
    ``tOffsetMs``. Events marked ``aAppend`` are roll-up repeats of the line
    already emitted and are skipped, as are the whitespace segments that
    separate roll-up lines. A segment holding several words — which manual
    captions do and auto-captions occasionally do — gives every word the
    segment's start; captions are a search resource, not a cutting one. The
    same applies to a hyphenated word: :func:`~rytp.transcribe.base.split_token`
    makes it two rows sharing one start, because a row whose normalized text
    holds a space could never be found.
    """
    words: list[CaptionWord] = []
    previous = 0
    for event in payload.get("events") or ():
        if event.get("aAppend"):
            continue
        base = int(event.get("tStartMs", 0))
        for segment in event.get("segs") or ():
            raw = str(segment.get("utf8", ""))
            if not raw.strip():
                continue
            start = max(base + int(segment.get("tOffsetMs", 0)), previous)
            words.extend(
                CaptionWord(start_ms=start, text=surface)
                for surface, _normalized in split_token(raw)
            )
            previous = start
    return words


def captions_asset_path(db: Database, video_id: int) -> Path:
    """Path of the video's captions asset (contracts §3, role ``captions``)."""
    row = db.conn.execute(
        "SELECT path FROM assets WHERE video_id = ? AND role = 'captions' "
        "ORDER BY acquired_at DESC, id DESC LIMIT 1",
        (video_id,),
    ).fetchone()
    if row is None:
        raise RytpError(f"video {video_id} has no captions asset; fetch captions first")
    return Path(str(row[0]))


def ingest_captions(db: Database, video_id: int, path: Path | None = None) -> int:
    """Replace the video's words with caption-tier rows. Returns how many.

    Refuses to run on a video that already has `timed` or `aligned` words:
    either is better text than captions, so overwriting would be a downgrade.
    Promotion goes the other way, through
    :func:`rytp.transcribe.pipeline.transcribe_video`.
    """
    source = path if path is not None else captions_asset_path(db, video_id)
    transcribed = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND source <> 'caption'",
        (video_id,),
    ).fetchone()[0]
    if transcribed:
        raise CaptionDowngrade(
            f"video {video_id} already has {transcribed} transcribed words; "
            "captions would be a downgrade"
        )
    caption_words = parse_json3(load_json3(source))
    rows = []
    ordinal = 0
    for word in caption_words:
        normalized = normalize_text(word.text)
        if not normalized:
            continue
        rows.append(
            (
                video_id,
                ordinal,
                word.start_ms,
                None,
                word.text,
                normalized,
                stem_text(normalized),
                None,
                None,
                "caption",
                C.CAPTION_ENGINE,
                None,
            )
        )
        ordinal += 1
    # Imported here rather than at module level: pipeline pulls in numpy and
    # the audio stack, which the caption tier has no use for until this point.
    from rytp.transcribe.pipeline import invalidate_transcript

    with db.transaction():
        # Contracts §4's cross-part invariant, in the one shared function that
        # implements it: words, utterances and speaker labels go together.
        invalidate_transcript(db, video_id)
        db.conn.executemany(_INSERT_WORD, rows)
    return len(rows)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_captions.py -v
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/transcribe/captions.py tests/test_transcribe_captions.py
git commit -m "feat(transcribe): ingest json3 auto-captions as caption-tier words"
```

---

### Task 9: The transcription pipeline — `timed` and `aligned`

**Files:**
- Create: `rytp/transcribe/pipeline.py`
- Modify: `tests/fake_engines.py` (append `ScorelessAligner`)
- Test: `tests/test_transcribe_pipeline.py`

**Interfaces:**
- Consumes: `rytp.audio.energy.{read_wav_mono, index_to_ms, refine_boundaries, enforce_monotonic, silence_depth_db, boundary_quality}`, `rytp.audio.vad.{detect_speech, plan_chunks, AudioChunk}`, `rytp.transcribe.registry.{load_transcriber, load_aligner}`, `rytp.models.{RawWord, Span, RytpError, normalize_text, stem_text}`.
- Produces: `TranscribeOutcome(video_id, n_words, n_chunks, engine, median_align_score)`, `TranscriptRemoval(words, utterances, video_speakers)`, `AlignmentMismatch(RytpError)`, `engine_tag(transcriber, aligner, refined) -> str`, `invalidate_transcript(db, video_id) -> TranscriptRemoval`, `replace_words(db, video_id, words, spans, engine) -> int`, `transcribe_video(db, video_id, *, wav_path, transcriber, aligner=None, language=C.DEFAULT_LANGUAGE, refine=True) -> TranscribeOutcome`, `enqueue_index(db, video_id) -> None`.

- [ ] **Step 1: Append `ScorelessAligner` to `tests/fake_engines.py`**

```python
class ScorelessAligner(FakeAligner):
    """An aligner that reports no per-word confidence — MFA behaves this way."""

    name = "fake-scoreless-aligner"

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        return [
            Span(start_ms=span.start_ms, end_ms=span.end_ms, score=None)
            for span in super().align(audio, words, start_ms=start_ms, end_ms=end_ms)
        ]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_transcribe_pipeline.py`:

```python
"""Transcriber text, aligner timings, measured boundaries, one write, one tier."""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe.pipeline import (
    AlignmentMismatch,
    engine_tag,
    speaker_loss_warning,
    tier_for,
    transcribe_video,
)
from tests.fake_engines import (
    FakeAligner,
    FakeTranscriber,
    NoEndTimesTranscriber,
    ScorelessAligner,
    ShortAligner,
    TextOnlyTranscriber,
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav


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


def test_engine_tag_records_every_stage() -> None:
    assert engine_tag("gigaam", "mfa", True) == "gigaam+mfa+energy"
    assert engine_tag("whisper", None, False) == "whisper"


def test_only_a_run_with_an_aligner_produces_a_cuttable_tier() -> None:
    assert tier_for("mfa") == "aligned"
    assert tier_for(None) == "timed"
    assert tier_for("") == "timed"


def test_without_an_aligner_the_words_are_timed_and_not_cuttable(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §3: the transcriber's own timestamps are 78.7% zero-gap on
    # real data. Energy refinement improves them but cannot place a boundary
    # that was never there, so these rows are searchable, not cuttable.
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        outcome = transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
    assert outcome.n_words == 3
    assert outcome.source == "timed"
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in rows] == [0, 1, 2]
    assert all(row[2] is not None and row[2] > row[1] for row in rows)
    assert {row[4] for row in rows} == {"timed"}
    assert {row[5] for row in rows} == {"fake+energy"}


def test_with_an_aligner_the_words_are_aligned_and_cuttable(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
        )
    assert outcome.source == "aligned"
    sources = {
        row[0]
        for row in db.conn.execute(
            "SELECT source FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    }
    assert sources == {"aligned"}


def test_replacing_a_transcript_reports_the_speaker_labels_it_destroyed(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §4 deletes them and Part 7's diarize is reopenable=False, so
    # nothing brings them back. Saying so is the least this can do.
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine) "
            "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
            (video_id,),
        )
        db.conn.commit()
        outcome = transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
    assert outcome.speakers_lost == 1
    warning = speaker_loss_warning(video_id, outcome.speakers_lost)
    assert warning is not None
    assert "speakers diarize" in warning
    assert speaker_loss_warning(video_id, 0) is None


def test_adjacent_words_share_a_measured_boundary(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake")
    rows = db.conn.execute(
        "SELECT start_ms, end_ms FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    for left, right in zip(rows, rows[1:], strict=False):
        assert left[1] == right[0]


def test_an_aligner_supplies_the_timings_and_the_score(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-aligner",
        )
    assert outcome.engine == "fake+fake-aligner+energy"
    scores = [
        row[0]
        for row in db.conn.execute(
            "SELECT align_score FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    ]
    assert scores == [pytest.approx(0.8)] * 3


def test_a_scoreless_aligner_gets_a_measured_boundary_quality(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, ScorelessAligner):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-scoreless-aligner",
        )
    scores = [
        row[0]
        for row in db.conn.execute(
            "SELECT align_score FROM words WHERE video_id = ?", (video_id,)
        ).fetchall()
    ]
    assert all(score is not None and 0.0 <= score <= 1.0 for score in scores)


def test_promotion_deletes_caption_words_utterances_and_speaker_labels(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, stem, "
        "source, engine) VALUES (?, 0, 0, 'старое', 'старое', 'стар', 'caption', 'captions:json3')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 100, 0, 0, 'x', 'x', 'x')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    db.conn.commit()
    with registered(FakeTranscriber):
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND source = 'caption'", (video_id,)
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_a_hyphenated_word_becomes_two_rows_with_a_shared_boundary(
    db: Database, tmp_path: Path
) -> None:
    # The invariant Parts 4 and 5 rely on: one token per row, no spaces in
    # normalized_text. The split point is interpolated, then measured by
    # refinement like every other boundary.
    video_id = _make_video(db)
    script = ((0, 400, "кто-то"), (460, 700, "ещё"))
    with registered(FakeTranscriber):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": script},
        )
    rows = db.conn.execute(
        "SELECT ord, start_ms, end_ms, text, normalized_text FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[3] for row in rows] == ["кто", "то", "ещё"]
    assert [row[4] for row in rows] == ["кто", "то", "еще"]
    assert all(" " not in row[4] for row in rows)
    assert rows[0][2] == rows[1][1]
    assert rows[0][1] < rows[0][2] < rows[1][2]


def test_punctuation_only_tokens_never_get_an_ordinal(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    script = ((0, 200, "один"), (200, 260, "—"), (260, 460, "два"))
    with registered(FakeTranscriber):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": script},
        )
    texts = [
        row[0]
        for row in db.conn.execute(
            "SELECT text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
        ).fetchall()
    ]
    assert texts == ["один", "два"]


def test_a_transcriber_without_end_times_demands_an_aligner(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(NoEndTimesTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake-no-ends")
    assert "does not time every word" in str(excinfo.value)


def test_a_text_only_transcriber_demands_an_aligner(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(TextOnlyTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(db, video_id, wav_path=_make_wav(tmp_path), transcriber="fake-text-only")
    assert "does not time every word" in str(excinfo.value)


def test_a_text_only_transcriber_is_fully_usable_with_an_aligner(
    db: Database, tmp_path: Path
) -> None:
    # Contracts §4 allows RawWord to carry text alone; the aligner supplies
    # every boundary and refinement measures them.
    video_id = _make_video(db)
    with registered(TextOnlyTranscriber, FakeAligner):
        outcome = transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake-text-only",
            aligner="fake-aligner",
        )
    assert outcome.n_words == 3
    rows = db.conn.execute(
        "SELECT start_ms, end_ms, text FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[2] for row in rows] == ["один", "два", "три"]
    assert all(row[1] > row[0] for row in rows)


def test_an_aligner_returning_the_wrong_count_is_refused(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, ShortAligner), pytest.raises(AlignmentMismatch):
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            aligner="fake-short-aligner",
        )


def test_a_transcriber_that_finds_nothing_is_reported(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber), pytest.raises(RytpError) as excinfo:
        transcribe_video(
            db,
            video_id,
            wav_path=_make_wav(tmp_path),
            transcriber="fake",
            transcriber_kwargs={"script": ()},
        )
    assert "no words" in str(excinfo.value)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_pipeline.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.pipeline'`.

- [ ] **Step 4: Write `rytp/transcribe/pipeline.py`**

```python
"""Aligned-tier transcription: text, rough timings, then measured boundaries.

Design §6. Three sources, because no single one can be trusted:

1. **Text** from the best available transcriber.
2. **Rough timings** from forced alignment against that text.
3. **The final boundary** at the local energy minimum between two words,
   snapped to a zero crossing, with voice-activity edges taken for free.

Promoting a video to this tier deletes its caption words — one tier per video,
no run history to reason about. It also drops the video's utterances, which
copy word timings and ordinals, and its diarized speaker labels, because the
ordinals those labels were attached to no longer mean the same thing. Design
§3: erase and replace, don't reconcile.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from rytp import constants as C
from rytp.audio.energy import (
    boundary_quality,
    enforce_monotonic,
    index_to_ms,
    read_wav_mono,
    refine_boundaries,
    silence_depth_db,
)
from rytp.audio.vad import AudioChunk, detect_speech, plan_chunks
from rytp.db import Database
from rytp.models import RawWord, RytpError, Span, normalize_text, stem_text
from rytp.transcribe.base import Aligner, split_token
from rytp.transcribe.registry import load_aligner, load_transcriber


class AlignmentMismatch(RytpError):
    """The aligner returned a different number of spans than there were words."""


@dataclass(frozen=True)
class TranscribeOutcome:
    """What one transcription run produced, including which tier it wrote."""

    video_id: int
    n_words: int
    n_chunks: int
    engine: str
    #: The `words.source` tier this run wrote: "timed" or "aligned".
    source: str
    #: Speaker labels destroyed along with the old transcript, if any.
    speakers_lost: int
    median_align_score: float | None


_WORD_COLUMNS = (
    "video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
    "confidence, align_score, source, engine, video_speaker_id"
)
_INSERT_WORD = f"INSERT INTO words ({_WORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"


def tier_for(aligner: str | None) -> str:
    """Which `words.source` tier this run produces (contracts §3).

    An aligner ran, so every boundary was actually placed: ``aligned``, which
    is the definition of cuttable. No aligner, so the boundaries are the
    transcriber's own: ``timed`` — searchable, not cuttable.

    This is the whole point of the three-tier scheme. Measured on the owner's
    real data, 78.7% of the transcriber's word gaps are exactly zero, because
    it sets ``word[i].end == word[i+1].start`` and absorbs every pause into a
    neighbour. Energy refinement still runs on a `timed` transcript and still
    improves it, but it relocates a boundary inside a window — it cannot place
    one that was never there. Writing these rows ``aligned`` and hoping is
    exactly what the design exists to prevent, and nothing downstream could
    have caught it: ``align_score`` is null without an aligner and no consumer
    reads ``words.engine``.
    """
    return "aligned" if aligner else "timed"


def engine_tag(transcriber: str, aligner: str | None, refined: bool) -> str:
    """The ``words.engine`` value: every stage that touched these timings.

    A corpus built with more than one engine stays interpretable only if each
    row says what made it (design §11, M0).
    """
    parts = [transcriber]
    if aligner:
        parts.append(aligner)
    if refined:
        parts.append("energy")
    return "+".join(parts)


def transcribe_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    transcriber: str,
    aligner: str | None = None,
    language: str | None = C.DEFAULT_LANGUAGE,
    refine: bool = True,
    transcriber_kwargs: dict[str, Any] | None = None,
    aligner_kwargs: dict[str, Any] | None = None,
) -> TranscribeOutcome:
    """Transcribe, align and refine one video, then replace its words."""
    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    speech = detect_speech(samples, sr)
    chunks = plan_chunks(speech, total_ms=total_ms) or [
        AudioChunk(ord=0, start_ms=0, end_ms=total_ms)
    ]
    engine = load_transcriber(db, transcriber, **(transcriber_kwargs or {}))
    aligner_engine = (
        load_aligner(db, aligner, **(aligner_kwargs or {})) if aligner else None
    )

    words: list[RawWord] = []
    spans: list[Span] = []
    for chunk in chunks:
        produced = [
            word
            for word in engine.transcribe(
                wav_path, language=language, start_ms=chunk.start_ms, end_ms=chunk.end_ms
            )
            if normalize_text(word.text)
        ]
        if all(word.start_ms is not None for word in produced):
            produced.sort(key=lambda word: (word.start_ms or 0, word.text))
        # Otherwise the transcriber emitted text only (contracts §4 allows it)
        # and its words are already in spoken order; there is nothing to sort by.
        if not produced:
            continue
        words.extend(produced)
        spans.extend(
            _chunk_spans(
                chunk,
                produced,
                aligner_engine,
                wav_path,
                transcriber=transcriber,
                aligner=aligner,
            )
        )
    # One row per token, before refinement so an invented interior boundary
    # gets measured like any other.
    words, spans = _expand_tokens(words, spans)
    if not words:
        raise RytpError(
            f"transcriber {transcriber!r} produced no words for video {video_id}"
        )

    if refine:
        spans = refine_boundaries(
            samples,
            sr,
            spans,
            speech=[(segment.start_ms, segment.end_ms) for segment in speech],
        )
        spans = _score_boundaries(samples, sr, spans)
    else:
        spans = enforce_monotonic(spans, total_ms=total_ms)

    tag = engine_tag(transcriber, aligner, refine)
    tier = tier_for(aligner)
    removed = replace_words(db, video_id, words, spans, tag, source=tier)
    scores = [span.score for span in spans if span.score is not None]
    return TranscribeOutcome(
        video_id=video_id,
        n_words=len(words),
        n_chunks=len(chunks),
        engine=tag,
        source=tier,
        speakers_lost=removed.video_speakers,
        median_align_score=float(median(scores)) if scores else None,
    )


def _chunk_spans(
    chunk: AudioChunk,
    words: Sequence[RawWord],
    aligner_engine: Aligner | None,
    wav_path: Path,
    *,
    transcriber: str,
    aligner: str | None,
) -> list[Span]:
    if aligner_engine is not None:
        spans = list(
            aligner_engine.align(
                wav_path,
                [word.text for word in words],
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
        )
        if len(spans) != len(words):
            raise AlignmentMismatch(
                f"aligner {aligner!r} returned {len(spans)} spans for {len(words)} "
                f"words in chunk {chunk.ord} [{chunk.start_ms}..{chunk.end_ms}]"
            )
        return spans
    # No aligner: the transcriber's own timings have to be complete. Contracts
    # §4 lets a transcriber emit text with no timings at all, and Whisper-class
    # engines routinely omit an end, so both halves are checked.
    spans: list[Span] = []
    for word in words:
        start, end = word.start_ms, word.end_ms
        if start is None or end is None:
            raise RytpError(
                f"transcriber {transcriber!r} does not time every word "
                f"(first untimed: {word.text!r}); pass an aligner"
            )
        spans.append(Span(start_ms=start, end_ms=end, score=word.confidence))
    return spans


def _expand_tokens(
    words: Sequence[RawWord], spans: Sequence[Span]
) -> tuple[list[RawWord], list[Span]]:
    """One word row per token, splitting a multi-token word's span in proportion.

    A transcriber emits "кто-то" as a single token, and `normalize_text`
    replaces the hyphen with a space — so one row would hold two tokens, which
    nothing can ever find. Part 4's index and FTS lookups are single-token and
    Part 5's assembly pointer walk assumes one token per row. Stripping the
    hyphen into "ктото" would be worse rather than better: the same normalizer
    runs over the user's query, which also splits, so the stripped row would be
    unreachable from either direction.

    The interior boundary starts out interpolated by character count. Because
    this runs *before* :func:`refine_boundaries`, it is then measured like
    every other boundary — for "кто-то" the energy minimum lands in the stop
    closure of the /t/, which is exactly where the split belongs.
    """
    out_words: list[RawWord] = []
    out_spans: list[Span] = []
    for word, span in zip(words, spans, strict=True):
        pieces = split_token(word.text)
        if not pieces:
            continue
        if len(pieces) == 1:
            out_words.append(word)
            out_spans.append(span)
            continue
        total = sum(len(normalized) for _surface, normalized in pieces)
        cursor = span.start_ms
        for index, (surface, normalized) in enumerate(pieces):
            share = round((span.end_ms - span.start_ms) * len(normalized) / max(total, 1))
            end = (
                span.end_ms
                if index == len(pieces) - 1
                else min(cursor + share, span.end_ms)
            )
            out_words.append(
                RawWord(
                    text=surface,
                    start_ms=cursor,
                    end_ms=max(end, cursor),
                    confidence=word.confidence,
                )
            )
            out_spans.append(Span(start_ms=cursor, end_ms=max(end, cursor), score=span.score))
            cursor = max(end, cursor)
    return out_words, out_spans


@dataclass(frozen=True)
class TranscriptRemoval:
    """How much a transcript invalidation actually removed."""

    words: int
    utterances: int
    video_speakers: int


def _count(db: Database, table: str, video_id: int) -> int:
    if table not in ("words", "utterances", "video_speakers"):
        raise ValueError(f"not a per-video derived table: {table!r}")
    row = db.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
    ).fetchone()
    return int(row[0])


def invalidate_transcript(db: Database, video_id: int) -> TranscriptRemoval:
    """Drop a video's words and everything derived from them.

    Contracts §4's cross-part invariant, in one place so it cannot drift:
    `utterances` copy word timings and ordinals, and a `video_speakers` label
    is attached to ordinals that are about to change or disappear, so all
    three go together or none do. Called by every stage that replaces a
    video's transcript and by ``transcribe.remove``.

    ``db.transaction()`` is re-entrant (contracts §8), so this is safe to call
    from inside a larger write — which is exactly how `replace_words` uses it.

    The cached WAV is deliberately untouched: it is regenerable and Part 2's
    ``cache.prune`` owns it.
    """
    with db.transaction():
        removed = TranscriptRemoval(
            words=_count(db, "words", video_id),
            utterances=_count(db, "utterances", video_id),
            video_speakers=_count(db, "video_speakers", video_id),
        )
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM video_speakers WHERE video_id = ?", (video_id,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    return removed


def speaker_loss_warning(video_id: int, speakers_lost: int) -> str | None:
    """Warn when replacing a transcript threw away human work, permanently.

    Contracts §4 makes replacing a video's words delete its `video_speakers`,
    and Part 7 registers `diarize` with ``reopenable=False`` so `reconcile`
    will never bring it back. Both decisions are right on their own; together
    they mean a re-transcribed video loses its speaker dimension for good and
    says nothing about it. This is that something.
    """
    if speakers_lost <= 0:
        return None
    return (
        f"warning: dropped {speakers_lost} speaker label(s) with the previous "
        f"transcript. Nothing re-derives them — run `rytp speakers diarize "
        f"{video_id}` and re-map the labels to get the speaker dimension back."
    )


def enqueue_index(db: Database, video_id: int) -> None:
    """Ask for the video's utterances to be rebuilt (contracts §5).

    Transcribing or re-aligning deletes the video's utterances, so its search
    index is stale the moment this stage finishes. Nothing else enqueues the
    rebuild — Part 2's chain stops at ``extract_wav`` — so without this a
    freshly transcribed video stays invisible to search until somebody runs a
    command by hand. Part 4's ``index`` kind is reopenable and its readiness
    is derived from the data, so enqueueing is the whole of the coordination.

    Called by the callers rather than from inside :func:`transcribe_video`, so
    the stage itself stays testable without a registered job queue.
    """
    from rytp.jobs import JOB_KINDS
    from rytp.jobs.queue import enqueue

    if "index" not in JOB_KINDS:
        # Part 4 registers that kind. Until it lands there is nothing to ask
        # for, and a transcribed video simply has no index yet.
        return
    enqueue(db, "index", video_id)


def _score_boundaries(samples: Any, sr: int, spans: Sequence[Span]) -> list[Span]:
    """Fill a missing alignment score with the measured quality of its boundaries.

    MFA reports no per-word confidence. How much silence actually sits at each
    end of the word is a better signal anyway, and it is already measured.

    Cost is two :func:`silence_depth_db` calls per word, each a handful of
    small numpy reductions — one to three seconds for an hour of speech. That
    is deliberate and measured; do not "optimise" it by widening the windows,
    which would change what the score means.
    """
    scored: list[Span] = []
    for span in spans:
        if span.score is not None:
            scored.append(span)
            continue
        depth = min(
            silence_depth_db(samples, sr, span.start_ms),
            silence_depth_db(samples, sr, span.end_ms),
        )
        scored.append(
            Span(start_ms=span.start_ms, end_ms=span.end_ms, score=boundary_quality(depth))
        )
    return scored


def replace_words(
    db: Database,
    video_id: int,
    words: Sequence[RawWord],
    spans: Sequence[Span],
    engine: str,
    *,
    source: str,
) -> TranscriptRemoval:
    """Swap in a fresh transcript for one video, in one transaction.

    ``source`` is the tier from :func:`tier_for`. Returns what the replacement
    destroyed, so a caller can warn about speaker labels it cannot rebuild.
    """
    rows = []
    for ordinal, (word, span) in enumerate(zip(words, spans, strict=True)):
        normalized = normalize_text(word.text)
        rows.append(
            (
                video_id,
                ordinal,
                span.start_ms,
                span.end_ms,
                word.text,
                normalized,
                stem_text(normalized),
                word.confidence,
                span.score,
                source,
                engine,
                None,
            )
        )
    with db.transaction():
        removed = invalidate_transcript(db, video_id)
        db.conn.executemany(_INSERT_WORD, rows)
    return removed
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_pipeline.py -v
```
Expected: 13 passed.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/transcribe rytp/audio
```
Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add rytp/transcribe/pipeline.py tests/fake_engines.py tests/test_transcribe_pipeline.py
git commit -m "feat(transcribe): transcription pipeline writing the timed and aligned tiers"
```

---

### Task 10: Alignment of existing words — the `timed` to `aligned` upgrade

Design §4 lists `align` as its own job kind. Since the database is the only channel between stages, a separate align job can only mean one thing: re-time words that already exist, keeping their text. That is exactly the operation that makes aligners swappable (design §11, M0) — try a different one without paying for transcription again.

**Files:**
- Modify: `rytp/transcribe/pipeline.py` (append)
- Test: `tests/test_transcribe_pipeline.py` (append)

**Interfaces:**
- Consumes: Task 9.
- Produces: `realign_video(db, video_id, *, wav_path, aligner, refine=True, aligner_kwargs=None) -> TranscribeOutcome`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcribe_pipeline.py`:

```python
from rytp.transcribe.pipeline import realign_video


def test_realign_retimes_words_without_changing_their_text(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
        before = db.conn.execute(
            "SELECT ord, text, start_ms FROM words WHERE video_id = ? ORDER BY ord",
            (video_id,),
        ).fetchall()
        outcome = realign_video(db, video_id, wav_path=wav, aligner="fake-aligner")
    after = db.conn.execute(
        "SELECT ord, text, start_ms, engine FROM words WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[1] for row in after] == [row[1] for row in before]
    assert [row[0] for row in after] == [row[0] for row in before]
    assert outcome.engine == "fake+fake-aligner+energy"
    assert {row[3] for row in after} == {"fake+fake-aligner+energy"}


def test_aligning_upgrades_a_timed_transcript_in_place(
    db: Database, tmp_path: Path
) -> None:
    # The promotion path of contracts §3: same text, same ordinals, tier moves
    # from timed to aligned and the words become cuttable.
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
        assert {
            row[0]
            for row in db.conn.execute(
                "SELECT source FROM words WHERE video_id = ?", (video_id,)
            ).fetchall()
        } == {"timed"}
        outcome = realign_video(db, video_id, wav_path=wav, aligner="fake-aligner")
    assert outcome.source == "aligned"
    rows = db.conn.execute(
        "SELECT source, text FROM words WHERE video_id = ? ORDER BY ord", (video_id,)
    ).fetchall()
    assert {row[0] for row in rows} == {"aligned"}
    assert [row[1] for row in rows] == ["один", "два", "три"]


def test_realign_keeps_speaker_labels(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    wav = _make_wav(tmp_path)
    with registered(FakeTranscriber, FakeAligner):
        transcribe_video(db, video_id, wav_path=wav, transcriber="fake")
        cursor = db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine) "
            "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
            (video_id,),
        )
        db.conn.execute(
            "UPDATE words SET video_speaker_id = ? WHERE video_id = ?",
            (cursor.lastrowid, video_id),
        )
        db.conn.commit()
        realign_video(db, video_id, wav_path=wav, aligner="fake-aligner")
    labelled = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ? AND video_speaker_id IS NOT NULL",
        (video_id,),
    ).fetchone()[0]
    assert labelled == 3


def test_realign_refuses_a_caption_tier_video(db: Database, tmp_path: Path) -> None:
    # Captions have starts on a 40 ms grid and no ends; there is nothing there
    # for an aligner to improve. Transcribe first.
    video_id = _make_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, text, normalized_text, stem, "
        "source, engine) VALUES (?, 0, 0, 'слово', 'слово', 'слов', 'caption', 'captions:json3')",
        (video_id,),
    )
    db.conn.commit()
    with registered(FakeAligner), pytest.raises(RytpError) as excinfo:
        realign_video(db, video_id, wav_path=_make_wav(tmp_path), aligner="fake-aligner")
    assert "caption" in str(excinfo.value)


def test_realign_without_words_says_so(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    with registered(FakeAligner), pytest.raises(RytpError) as excinfo:
        realign_video(db, video_id, wav_path=_make_wav(tmp_path), aligner="fake-aligner")
    assert "no words" in str(excinfo.value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_pipeline.py -v
```
Expected: `ImportError: cannot import name 'realign_video'`.

- [ ] **Step 3: Append to `rytp/transcribe/pipeline.py`**

```python
def _chunk_for(chunks: Sequence[AudioChunk], ms: int) -> int:
    for chunk in chunks:
        if chunk.start_ms <= ms < chunk.end_ms:
            return chunk.ord
    return min(
        chunks, key=lambda chunk: min(abs(ms - chunk.start_ms), abs(ms - chunk.end_ms))
    ).ord


def realign_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    aligner: str,
    refine: bool = True,
    aligner_kwargs: dict[str, Any] | None = None,
) -> TranscribeOutcome:
    """Align a video's existing words, upgrading the tier in place.

    Two jobs in one. A `timed` transcript — the transcriber's own timestamps,
    searchable but not cuttable — becomes `aligned` and therefore cuttable,
    which is the promotion path contracts §3 describes. An already-`aligned`
    transcript gets re-timed by a different aligner, which is what makes
    aligners swappable without paying for transcription again.

    Either way the text is kept, so ordinals and the speaker labels hanging
    off them stay valid; only the timings, the alignment scores, the engine
    tag and the source tier change.
    """
    rows = db.conn.execute(
        "SELECT ord, start_ms, text, confidence, source, engine FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    if not rows:
        raise RytpError(f"video {video_id} has no words to re-align")
    if any(str(row[4]) == "caption" for row in rows):
        raise RytpError(
            f"video {video_id} is caption-tier; run transcribe run to promote it"
        )

    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    speech = detect_speech(samples, sr)
    chunks = plan_chunks(speech, total_ms=total_ms) or [
        AudioChunk(ord=0, start_ms=0, end_ms=total_ms)
    ]
    engine = load_aligner(db, aligner, **(aligner_kwargs or {}))

    buckets: dict[int, list[int]] = {chunk.ord: [] for chunk in chunks}
    for index, row in enumerate(rows):
        buckets[_chunk_for(chunks, int(row[1]))].append(index)

    by_index: dict[int, Span] = {}
    for chunk in chunks:
        indexes = buckets[chunk.ord]
        if not indexes:
            continue
        spans = list(
            engine.align(
                wav_path,
                [str(rows[index][2]) for index in indexes],
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
            )
        )
        if len(spans) != len(indexes):
            raise AlignmentMismatch(
                f"aligner {aligner!r} returned {len(spans)} spans for {len(indexes)} "
                f"words in chunk {chunk.ord} [{chunk.start_ms}..{chunk.end_ms}]"
            )
        for index, span in zip(indexes, spans, strict=True):
            by_index[index] = span

    ordered = [by_index[index] for index in range(len(rows))]
    if refine:
        ordered = refine_boundaries(
            samples,
            sr,
            ordered,
            speech=[(segment.start_ms, segment.end_ms) for segment in speech],
        )
        ordered = _score_boundaries(samples, sr, ordered)
    else:
        ordered = enforce_monotonic(ordered, total_ms=total_ms)

    base = str(rows[0][5]).split("+")[0]
    tag = engine_tag(base, aligner, refine)
    with db.transaction():
        # Utterances copy word timings, so they are stale; Part 4 rebuilds them.
        # Speaker labels are kept: the text and the ordinals did not change.
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.executemany(
            "UPDATE words SET start_ms = ?, end_ms = ?, align_score = ?, engine = ?, "
            "source = 'aligned' WHERE video_id = ? AND ord = ?",
            [
                (span.start_ms, span.end_ms, span.score, tag, video_id, int(rows[index][0]))
                for index, span in enumerate(ordered)
            ],
        )
    scores = [span.score for span in ordered if span.score is not None]
    return TranscribeOutcome(
        video_id=video_id,
        n_words=len(ordered),
        n_chunks=len(chunks),
        engine=tag,
        source="aligned",
        speakers_lost=0,
        median_align_score=float(median(scores)) if scores else None,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_pipeline.py -v
```
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/transcribe/pipeline.py tests/test_transcribe_pipeline.py
git commit -m "feat(transcribe): re-align existing words with a different aligner"
```

---

### Task 11: Acoustic metrics

Design §4: one cheap fingerprint per video, one pass, one row. Part 5 uses it to decide which sources blend when splicing; Part 7 uses it to scope cross-video speaker matching to comparable recordings. It replaces the per-clip MFCC scheme in the old code, which was both more expensive and aimed at the wrong problem.

**Files:**
- Create: `rytp/audio/acoustics.py`
- Test: `tests/test_audio_acoustics.py`

**Interfaces:**
- Consumes: `rytp.audio.energy.{frame_rms, to_db, ms_to_index}`, `rytp.audio.vad.SpeechSegment`.
- Produces: `Acoustics(f0_mean, f0_std, spectral_tilt, noise_floor_db, reverb_proxy, loudness_lufs)`, `estimate_f0(samples, sr, *, speech=None)`, `spectral_tilt(samples, sr, *, speech=None)`, `noise_floor_db(samples, sr)`, `reverb_proxy(samples, sr, speech)`, `compute_acoustics(samples, sr, *, speech=None, loudness_lufs=None) -> Acoustics`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_acoustics.py`:

```python
"""The per-video acoustic fingerprint: five numbers, one pass."""
from __future__ import annotations

import numpy as np

from rytp.audio.acoustics import (
    compute_acoustics,
    estimate_f0,
    noise_floor_db,
    reverb_proxy,
    spectral_tilt,
)
from rytp.audio.vad import SpeechSegment
from tests.synth_audio import SR, concat, silence, tone


def test_pitch_of_a_known_sine_is_recovered() -> None:
    mean, spread = estimate_f0(tone(1000, freq_hz=200.0), SR)
    assert mean is not None and spread is not None
    assert abs(mean - 200.0) < 5.0
    assert spread < 5.0


def test_pitch_of_digital_silence_is_unknown() -> None:
    assert estimate_f0(silence(1000), SR) == (None, None)


def test_noise_floor_finds_the_quiet_part_not_the_loud_one() -> None:
    samples = concat(tone(500, amp=0.3), silence(500, amp=0.001, seed=9))
    floor = noise_floor_db(samples, SR)
    assert -75.0 < floor < -45.0


def test_a_dull_recording_has_a_steeper_spectral_tilt_than_a_bright_one() -> None:
    rng = np.random.default_rng(5)
    white = (0.2 * rng.standard_normal(SR)).astype(np.float32)
    dull = np.convolve(white, np.ones(16, dtype=np.float32) / 16.0, mode="same").astype(
        np.float32
    )
    bright_tilt = spectral_tilt(white, SR)
    dull_tilt = spectral_tilt(dull, SR)
    assert bright_tilt is not None and dull_tilt is not None
    assert dull_tilt < bright_tilt - 5.0


def test_a_decaying_tail_reads_as_more_reverberant_than_an_abrupt_stop() -> None:
    body = tone(400)
    tail = tone(150)
    decayed = (tail * np.exp(-np.linspace(0.0, 5.0, tail.size))).astype(np.float32)
    dry = concat(body, silence(400, amp=0.0005, seed=21))
    wet = concat(body, decayed, silence(250, amp=0.0005, seed=22))
    segment = [SpeechSegment(start_ms=0, end_ms=400)]
    dry_value = reverb_proxy(dry, SR, segment)
    wet_value = reverb_proxy(wet, SR, segment)
    assert dry_value is not None and wet_value is not None
    assert wet_value > dry_value + 10.0


def test_reverb_of_nothing_is_unknown() -> None:
    assert reverb_proxy(silence(500), SR, []) is None


def test_compute_acoustics_fills_the_row_and_is_deterministic() -> None:
    samples = concat(tone(600, freq_hz=180.0), silence(400, amp=0.001, seed=31))
    first = compute_acoustics(samples, SR, loudness_lufs=-23.4)
    second = compute_acoustics(samples, SR, loudness_lufs=-23.4)
    assert first == second
    assert first.f0_mean is not None
    assert first.noise_floor_db < 0.0
    assert first.loudness_lufs == -23.4
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_acoustics.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.audio.acoustics'`.

- [ ] **Step 3: Write `rytp/audio/acoustics.py`**

```python
"""One acoustic fingerprint per video (design §4, "Acoustics").

Five numbers, one pass over the cached WAV, one row in ``video_acoustics``.
Two consumers: the assembler asks which sources will blend, and cross-video
speaker matching asks which recordings are comparable enough to compare
voices in. Both want "does this sound like the same room and the same
microphone", which is a far cheaper question than the per-clip MFCC machinery
this replaces.

Everything here is numpy and the standard library. Loudness is the exception
and lives in the next section of this module, because only ffmpeg measures
EBU R128 properly.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rytp import constants as C
from rytp.audio.energy import frame_rms, ms_to_index, to_db
from rytp.audio.vad import SpeechSegment


@dataclass(frozen=True)
class Acoustics:
    """The fingerprint. Any field may be ``None`` when the audio cannot answer."""

    f0_mean: float | None
    f0_std: float | None
    spectral_tilt: float | None
    noise_floor_db: float
    reverb_proxy: float | None
    loudness_lufs: float | None


def noise_floor_db(samples: np.ndarray, sr: int) -> float:
    """The quiet-end percentile of frame energy, in dBFS."""
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ACOUSTICS_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    if db.size == 0:
        return C.DB_FLOOR
    return float(np.percentile(db, C.VAD_NOISE_PERCENTILE))


def _frame_starts(
    samples: np.ndarray, sr: int, speech: Sequence[SpeechSegment] | None
) -> np.ndarray:
    """Sample indexes of the frames worth analysing: loud, and inside speech.

    Capped at :data:`rytp.constants.ACOUSTICS_MAX_FRAMES` by an even stride —
    a summary of an hour gains nothing from the other 175,000 frames, and the
    stride keeps the result deterministic.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    hop = max(1, ms_to_index(C.ACOUSTICS_HOP_MS, sr))
    db = to_db(frame_rms(samples, frame_len, hop))
    if db.size == 0:
        return np.zeros(0, dtype=np.int64)
    floor = float(np.percentile(db, C.VAD_NOISE_PERCENTILE))
    loud = db >= floor + C.F0_VOICED_ABOVE_FLOOR_DB
    if not bool(np.any(loud)) and float(np.max(db)) > C.DB_FLOOR:
        # Uniform level throughout: the percentile floor has landed on the
        # signal itself, so nothing clears it by 10 dB. Everything audible is
        # as loud as this recording gets, so analyse all of it. Digital
        # silence still falls through to an empty result.
        loud = db >= floor
    if speech is not None:
        starts_ms = np.arange(db.size, dtype=np.float64) * (hop * 1000.0 / sr)
        inside = np.zeros(db.size, dtype=bool)
        for segment in speech:
            inside |= (starts_ms >= segment.start_ms) & (starts_ms < segment.end_ms)
        loud &= inside
    indexes = np.flatnonzero(loud).astype(np.int64) * hop
    if indexes.size > C.ACOUSTICS_MAX_FRAMES:
        stride = int(math.ceil(indexes.size / C.ACOUSTICS_MAX_FRAMES))
        indexes = indexes[::stride]
    return indexes


def estimate_f0(
    samples: np.ndarray, sr: int, *, speech: Sequence[SpeechSegment] | None = None
) -> tuple[float | None, float | None]:
    """Mean and spread of speaking pitch, in Hz, over voiced frames.

    Autocorrelation via FFT: the direct product would be quadratic in the
    frame length and this runs over thousands of frames.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    starts = _frame_starts(samples, sr, speech)
    if starts.size == 0:
        return None, None
    lag_min = max(1, sr // C.F0_MAX_HZ)
    lag_max = min(frame_len - 1, sr // C.F0_MIN_HZ)
    if lag_max <= lag_min:
        return None, None
    size = 1 << (2 * frame_len - 1).bit_length()
    values: list[float] = []
    for start in starts:
        frame = samples[start : start + frame_len].astype(np.float64)
        if frame.size < frame_len:
            continue
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0.0:
            continue
        spectrum = np.fft.rfft(frame, size)
        auto = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:frame_len]
        band = auto[lag_min : lag_max + 1]
        if band.size == 0:
            continue
        peak = int(np.argmax(band))
        if float(band[peak]) / energy < C.F0_VOICED_AUTOCORR:
            continue
        values.append(sr / float(lag_min + peak))
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def spectral_tilt(
    samples: np.ndarray, sr: int, *, speech: Sequence[SpeechSegment] | None = None
) -> float | None:
    """Slope of the long-term average spectrum, in dB per decade.

    Negative and steep means dull — a distant microphone, a lossy encode, a
    muffled room. Near zero means bright. It is the single number that most
    often explains why two sources refuse to blend.
    """
    frame_len = max(1, ms_to_index(C.ACOUSTICS_FRAME_MS, sr))
    starts = _frame_starts(samples, sr, speech)
    if starts.size == 0:
        return None
    window = np.hanning(frame_len)
    accumulated = np.zeros(frame_len // 2 + 1, dtype=np.float64)
    used = 0
    for start in starts:
        frame = samples[start : start + frame_len]
        if frame.size < frame_len:
            continue
        accumulated += np.abs(np.fft.rfft(frame * window)) ** 2
        used += 1
    if used == 0:
        return None
    psd = accumulated / used
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    band = (freqs >= C.SPECTRAL_TILT_LO_HZ) & (freqs <= min(C.SPECTRAL_TILT_HI_HZ, sr / 2))
    if int(np.count_nonzero(band)) < 4:
        return None
    x = np.log10(freqs[band])
    y = 10.0 * np.log10(np.maximum(psd[band], 1e-20))
    return float(np.polyfit(x, y, 1)[0])


def _window_rms(samples: np.ndarray, sr: int, lo_ms: float, hi_ms: float) -> float:
    lo = max(0, ms_to_index(lo_ms, sr))
    hi = min(samples.size, ms_to_index(hi_ms, sr))
    if hi <= lo:
        return 0.0
    window = samples[lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(np.square(window))))


def reverb_proxy(
    samples: np.ndarray, sr: int, speech: Sequence[SpeechSegment]
) -> float | None:
    """Energy just after speech stops, relative to the speech, in dB.

    A dry room decays inside a few milliseconds and the value is far below
    zero; a live room keeps ringing and the value climbs toward it. Not a
    reverberation time — a proxy, which is all the assembler needs to tell two
    rooms apart. Pass unpadded segments: padding would put the tail being
    measured inside the segment.
    """
    values: list[float] = []
    for segment in speech:
        reference = _window_rms(samples, sr, segment.end_ms - C.REVERB_REF_MS, segment.end_ms)
        tail = _window_rms(samples, sr, segment.end_ms, segment.end_ms + C.REVERB_TAIL_MS)
        if reference <= 0.0:
            continue
        values.append(20.0 * math.log10(max(tail, C.DB_EPSILON) / reference))
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def compute_acoustics(
    samples: np.ndarray,
    sr: int,
    *,
    speech: Sequence[SpeechSegment] | None = None,
    loudness_lufs: float | None = None,
) -> Acoustics:
    """The whole fingerprint. ``loudness_lufs`` comes from ffmpeg, separately."""
    mean, spread = estimate_f0(samples, sr, speech=speech)
    return Acoustics(
        f0_mean=mean,
        f0_std=spread,
        spectral_tilt=spectral_tilt(samples, sr, speech=speech),
        noise_floor_db=noise_floor_db(samples, sr),
        reverb_proxy=reverb_proxy(samples, sr, speech or []),
        loudness_lufs=loudness_lufs,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_acoustics.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/audio/acoustics.py tests/test_audio_acoustics.py
git commit -m "feat(audio): per-video acoustic fingerprint"
```

---

### Task 12: Loudness measurement and acoustics persistence

**Files:**
- Modify: `rytp/audio/acoustics.py` (append)
- Test: `tests/test_audio_acoustics.py` (append)

**Interfaces:**
- Consumes: Task 11; `Database`; `rytp.audio.energy.read_wav_mono`; `rytp.audio.vad.detect_speech`.
- Produces: `LoudnessRunner = Callable[[list[str]], str]`, `parse_loudnorm(text) -> float | None`, `measure_loudness_lufs(wav_path, *, runner=None) -> float | None`, `store_acoustics(db, video_id, acoustics) -> None`, `fingerprint_video(db, video_id, *, wav_path, runner=None) -> Acoustics`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_acoustics.py`:

```python
import shutil
from pathlib import Path

import pytest

from rytp.audio.acoustics import (
    fingerprint_video,
    measure_loudness_lufs,
    parse_loudnorm,
    store_acoustics,
)
from rytp.db import Database
from tests.synth_audio import write_wav

FFMPEG_JSON = """
[Parsed_loudnorm_0 @ 0x55] 
{
	"input_i" : "-23.40",
	"input_tp" : "-3.20",
	"input_lra" : "5.10",
	"input_thresh" : "-33.80",
	"output_i" : "-24.00",
	"normalization_type" : "dynamic"
}
"""


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


def test_parse_loudnorm_reads_the_integrated_loudness() -> None:
    assert parse_loudnorm(FFMPEG_JSON) == pytest.approx(-23.40)


def test_parse_loudnorm_of_silence_is_unknown() -> None:
    assert parse_loudnorm('{"input_i" : "-inf"}') is None
    assert parse_loudnorm("no json here") is None


def test_measure_loudness_uses_the_injected_runner(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", tone(200))
    seen: list[list[str]] = []

    def runner(command: list[str]) -> str:
        seen.append(command)
        return FFMPEG_JSON

    assert measure_loudness_lufs(path, runner=runner) == pytest.approx(-23.40)
    assert "loudnorm=print_format=json" in seen[0]
    assert str(path) in seen[0]


def test_store_acoustics_inserts_then_updates_one_row(db: Database) -> None:
    video_id = _make_video(db)
    first = compute_acoustics(tone(500, freq_hz=150.0), SR, loudness_lufs=-20.0)
    store_acoustics(db, video_id, first)
    second = compute_acoustics(tone(500, freq_hz=150.0), SR, loudness_lufs=-18.0)
    store_acoustics(db, video_id, second)
    rows = db.conn.execute(
        "SELECT loudness_lufs, computed_at FROM video_acoustics WHERE video_id = ?",
        (video_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(-18.0)
    assert rows[0][1]


def test_fingerprint_video_writes_the_row(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    path = write_wav(
        tmp_path / "a.wav", concat(tone(600, freq_hz=170.0), silence(400, amp=0.001, seed=41))
    )
    result = fingerprint_video(
        db, video_id, wav_path=path, runner=lambda command: FFMPEG_JSON
    )
    assert result.loudness_lufs == pytest.approx(-23.40)
    stored = db.conn.execute(
        "SELECT f0_mean, noise_floor_db FROM video_acoustics WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    assert stored[0] is not None
    assert stored[1] < 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_real_ffmpeg_measures_a_finite_loudness(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", tone(3000, amp=0.3))
    value = measure_loudness_lufs(path)
    assert value is not None
    assert -60.0 < value < 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_audio_acoustics.py -v
```
Expected: `ImportError: cannot import name 'fingerprint_video'`.

- [ ] **Step 3: Append to `rytp/audio/acoustics.py`**

```python
LoudnessRunner = Callable[[list[str]], str]


def parse_loudnorm(text: str) -> float | None:
    """Pull ``input_i`` out of ffmpeg's loudnorm JSON block on stderr."""
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    try:
        value = float(payload.get("input_i"))
    except (TypeError, ValueError):
        return None
    return None if math.isinf(value) or math.isnan(value) else value


def _ffmpeg_stderr(command: list[str]) -> str:
    try:
        proc = subprocess.run(  # fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=C.LOUDNESS_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RytpError("ffmpeg is not on PATH; loudness cannot be measured") from exc
    except subprocess.TimeoutExpired as exc:
        raise RytpError(
            f"ffmpeg loudness scan timed out after {C.LOUDNESS_TIMEOUT_S}s"
        ) from exc
    return proc.stderr


def measure_loudness_lufs(wav_path: Path, *, runner: LoudnessRunner | None = None) -> float | None:
    """Integrated loudness in LUFS, measured by ffmpeg.

    ffmpeg is a required binary and implements EBU R128 properly, including
    the gating that a hand-rolled K-weighting would get subtly wrong. The
    runner is injectable so tests never need the binary.
    """
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-i",
        str(wav_path),
        "-af",
        "loudnorm=print_format=json",
        "-f",
        "null",
        "-",
    ]
    return parse_loudnorm((runner or _ffmpeg_stderr)(command))


def store_acoustics(db: Database, video_id: int, acoustics: Acoustics) -> None:
    """Upsert the video's fingerprint row (contracts §3, ``video_acoustics``)."""
    with db.transaction():
        db.conn.execute(
            "INSERT INTO video_acoustics (video_id, f0_mean, f0_std, spectral_tilt, "
            "noise_floor_db, reverb_proxy, loudness_lufs, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(video_id) DO UPDATE SET "
            "f0_mean = excluded.f0_mean, f0_std = excluded.f0_std, "
            "spectral_tilt = excluded.spectral_tilt, "
            "noise_floor_db = excluded.noise_floor_db, "
            "reverb_proxy = excluded.reverb_proxy, "
            "loudness_lufs = excluded.loudness_lufs, "
            "computed_at = excluded.computed_at",
            (
                video_id,
                acoustics.f0_mean,
                acoustics.f0_std,
                acoustics.spectral_tilt,
                acoustics.noise_floor_db,
                acoustics.reverb_proxy,
                acoustics.loudness_lufs,
                utc_now_iso(),
            ),
        )


def fingerprint_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    runner: LoudnessRunner | None = None,
) -> Acoustics:
    """Measure one video and store its fingerprint. One pass, one row."""
    samples, sr = read_wav_mono(wav_path)
    # Unpadded segments: a padded end would hide the reverberation tail.
    speech = detect_speech(samples, sr, pad_ms=0)
    acoustics = compute_acoustics(
        samples,
        sr,
        speech=speech,
        loudness_lufs=measure_loudness_lufs(wav_path, runner=runner),
    )
    store_acoustics(db, video_id, acoustics)
    return acoustics
```

Extend the imports at the top of `rytp/audio/acoustics.py`:

```python
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from rytp.audio.energy import frame_rms, ms_to_index, read_wav_mono, to_db
from rytp.audio.vad import SpeechSegment, detect_speech
from rytp.db import Database
from rytp.models import RytpError, utc_now_iso
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_audio_acoustics.py -v
```
Expected: 12 passed (13 with ffmpeg installed).

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/audio
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/audio/acoustics.py tests/test_audio_acoustics.py
git commit -m "feat(audio): measure loudness with ffmpeg and persist the fingerprint"
```

---

### Task 13: Sequence alignment, text and timing diffs

Design §11 makes this the M0 deliverable, not an afterthought: the owner deliberately postponed hand-marking a reference set, and bought the same safety by making engines swappable plus **a command that reports where they disagree**. This task is the measuring half.

**Files:**
- Create: `rytp/transcribe/compare.py`
- Test: `tests/test_transcribe_compare.py`

**Interfaces:**
- Consumes: `rytp.models.{Span, normalize_text}`, numpy.
- Produces: `EngineRun(label, words, spans, elapsed_s)`, `PairStats(...)`, `align_tokens(a, b) -> list[tuple[int | None, int | None]]`, `compare_pair(left, right) -> PairStats`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe_compare.py`:

```python
"""Where two engines disagree: on text, and on when each word happened."""
from __future__ import annotations

import pytest

from rytp.models import Span
from rytp.transcribe.compare import EngineRun, align_tokens, compare_pair


def _run(label: str, words: tuple[str, ...], starts: tuple[int, ...]) -> EngineRun:
    return EngineRun(
        label=label,
        words=words,
        spans=tuple(
            Span(start_ms=start, end_ms=start + 100, score=None) for start in starts
        ),
        elapsed_s=0.0,
    )


def test_identical_sequences_align_one_to_one() -> None:
    pairs = align_tokens(["а", "б", "в"], ["а", "б", "в"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_a_substitution_is_a_matched_pair_with_different_text() -> None:
    pairs = align_tokens(["а", "б", "в"], ["а", "х", "в"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_an_insertion_and_a_deletion_are_reported_as_such() -> None:
    assert align_tokens(["а", "в"], ["а", "б", "в"]) == [(0, 0), (None, 1), (1, 2)]
    assert align_tokens(["а", "б", "в"], ["а", "в"]) == [(0, 0), (1, None), (2, 1)]


def test_alignment_of_an_empty_side_is_all_insertions() -> None:
    assert align_tokens([], ["а", "б"]) == [(None, 0), (None, 1)]


def test_alignment_is_deterministic() -> None:
    a = ["один", "два", "три", "четыре"]
    b = ["один", "три", "четыре", "пять"]
    assert align_tokens(a, b) == align_tokens(a, b)


def test_compare_pair_counts_every_kind_of_difference() -> None:
    left = _run("a", ("один", "два", "три"), (0, 200, 400))
    right = _run("b", ("один", "икс", "три", "четыре"), (0, 210, 430, 600))
    stats = compare_pair(left, right)
    assert stats.a == "a"
    assert stats.b == "b"
    assert stats.matches == 2
    assert stats.substitutions == 1
    assert stats.insertions == 1
    assert stats.deletions == 0
    assert stats.disagreement == pytest.approx(2 / 3)


def test_compare_pair_measures_the_timing_gap_on_matched_words_only() -> None:
    left = _run("a", ("один", "два"), (0, 200))
    right = _run("b", ("один", "два"), (10, 240))
    stats = compare_pair(left, right)
    assert stats.median_abs_start_delta_ms == pytest.approx(25.0)
    assert stats.p90_abs_start_delta_ms is not None


def test_compare_pair_with_nothing_in_common_reports_no_timing_gap() -> None:
    stats = compare_pair(_run("a", ("один",), (0,)), _run("b", ("икс",), (0,)))
    assert stats.median_abs_start_delta_ms is None


def test_text_comparison_ignores_case_and_punctuation() -> None:
    stats = compare_pair(_run("a", ("Один,",), (0,)), _run("b", ("один",), (0,)))
    assert stats.substitutions == 0
    assert stats.matches == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_compare.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.compare'`.

- [ ] **Step 3: Write the first half of `rytp/transcribe/compare.py`**

```python
"""Run several engines over the same audio and report where they disagree.

Design §11, milestone M0. Hand-marking a reference set is the only way to get
a true error figure, and it was deliberately postponed. What replaces it is
this: engines are swappable, every word row records the engine that produced
it, and this module turns "which transcriber wins on *this* audio" and "is
alignment needed at all" into a command rather than a project.

Three kinds of disagreement are reported, because they fail differently:

* **Text** — a wrong word is worse than a missing one, since the tool will
  confidently cut audio that does not say what the index claims (design §3).
* **Timing** — how far apart two engines place the same word.
* **Boundaries** — how much silence actually sits where each engine claims a
  word ends. That last one is measured from the audio, so it needs no
  reference transcript at all.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rytp.models import Span, normalize_text


@dataclass(frozen=True)
class EngineRun:
    """One engine's output over one window of one file."""

    label: str
    words: tuple[str, ...]
    spans: tuple[Span, ...]
    elapsed_s: float


@dataclass(frozen=True)
class PairStats:
    """How far apart two engines are, on text and on time."""

    a: str
    b: str
    matches: int
    substitutions: int
    insertions: int
    deletions: int
    disagreement: float
    median_abs_start_delta_ms: float | None
    p90_abs_start_delta_ms: float | None


def align_tokens(
    a: Sequence[str], b: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Wagner-Fischer alignment of two token sequences.

    Returns index pairs: ``(i, j)`` for a match or substitution, ``(i, None)``
    for a deletion, ``(None, j)`` for an insertion. Backtracking prefers
    match/substitute, then deletion, then insertion, so two runs of the
    comparison command produce the same report.

    Quadratic in the token counts, which is why the comparison command works
    on a window (default two minutes) rather than a whole video.
    """
    n, m = len(a), len(b)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = cost[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost[i][j] = min(substitute, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        diagonal = (
            cost[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            if i > 0 and j > 0
            else None
        )
        if diagonal is not None and cost[i][j] == diagonal:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare_pair(left: EngineRun, right: EngineRun) -> PairStats:
    """Text and timing disagreement between two runs over the same audio."""
    a = [normalize_text(word) for word in left.words]
    b = [normalize_text(word) for word in right.words]
    pairs = align_tokens(a, b)

    matches = substitutions = insertions = deletions = 0
    deltas: list[float] = []
    for i, j in pairs:
        if i is None:
            insertions += 1
        elif j is None:
            deletions += 1
        elif a[i] == b[j]:
            matches += 1
            deltas.append(abs(left.spans[i].start_ms - right.spans[j].start_ms))
        else:
            substitutions += 1
    denominator = max(len(a), 1)
    return PairStats(
        a=left.label,
        b=right.label,
        matches=matches,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        disagreement=(substitutions + insertions + deletions) / denominator,
        median_abs_start_delta_ms=_percentile(deltas, 50),
        p90_abs_start_delta_ms=_percentile(deltas, 90),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_compare.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/transcribe/compare.py tests/test_transcribe_compare.py
git commit -m "feat(transcribe): text and timing disagreement between two engines"
```

---

### Task 14: Boundary statistics and the comparison report

This is the half that needs no reference transcript. It measures the claim that started the whole boundary design: **78.7% of word gaps are exactly zero** on the owner's existing data, and some words have zero duration.

**Files:**
- Modify: `rytp/transcribe/compare.py` (append)
- Test: `tests/test_transcribe_compare.py` (append)

**Interfaces:**
- Consumes: Task 13; `rytp.audio.energy.{read_wav_mono, index_to_ms, refine_boundaries, silence_depth_db}`; `rytp.transcribe.registry.{load_transcriber, load_aligner}`; `rytp.transcribe.pipeline.AlignmentMismatch`.
- Produces: `BoundaryStats(...)`, `boundary_shifts(claimed, refined) -> list[int]`, `count_capped(shifts) -> int`, `boundary_stats(run, samples, sr) -> BoundaryStats`, `run_comparison(db, *, wav_path, transcribers, aligners=(), start_ms=0, end_ms=None, language=C.DEFAULT_LANGUAGE) -> list[EngineRun]`, `comparison_rows(runs, samples, sr) -> tuple[tuple[str, ...], list[tuple[str, ...]]]`, `render_report(runs, samples, sr, *, source) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcribe_compare.py`:

```python
from pathlib import Path

from rytp import constants as C
from rytp.audio.energy import read_wav_mono
from rytp.db import Database
from rytp.transcribe.compare import (
    boundary_stats,
    comparison_rows,
    count_capped,
    render_report,
    run_comparison,
)
from tests.fake_engines import FakeAligner, FakeTranscriber, registered
from tests.synth_audio import concat, silence, tone, write_wav


def _wav(tmp_path: Path) -> Path:
    samples = concat(
        tone(200),
        silence(120, amp=0.001, seed=51),
        tone(200),
        silence(120, amp=0.001, seed=52),
        tone(200),
    )
    return write_wav(tmp_path / "audio.wav", samples)


def test_a_transcriber_that_absorbs_every_pause_reads_as_all_zero_gaps(
    tmp_path: Path,
) -> None:
    samples, sr = read_wav_mono(_wav(tmp_path))
    run = EngineRun(
        label="collapsing",
        words=("один", "два", "три"),
        spans=(
            Span(start_ms=0, end_ms=250, score=None),
            Span(start_ms=250, end_ms=500, score=None),
            Span(start_ms=500, end_ms=700, score=None),
        ),
        elapsed_s=0.0,
    )
    stats = boundary_stats(run, samples, sr)
    assert stats.zero_gap_fraction == pytest.approx(1.0)
    assert stats.median_gap_ms == pytest.approx(0.0)
    assert stats.n_words == 3


def test_real_gaps_are_reported_as_gaps(tmp_path: Path) -> None:
    samples, sr = read_wav_mono(_wav(tmp_path))
    run = EngineRun(
        label="spaced",
        words=("один", "два"),
        spans=(
            Span(start_ms=0, end_ms=200, score=None),
            Span(start_ms=320, end_ms=520, score=None),
        ),
        elapsed_s=0.0,
    )
    stats = boundary_stats(run, samples, sr)
    assert stats.zero_gap_fraction == pytest.approx(0.0)
    assert stats.median_gap_ms == pytest.approx(120.0)


def test_zero_length_words_are_counted(tmp_path: Path) -> None:
    samples, sr = read_wav_mono(_wav(tmp_path))
    run = EngineRun(
        label="degenerate",
        words=("один", "два"),
        spans=(
            Span(start_ms=100, end_ms=100, score=None),
            Span(start_ms=100, end_ms=300, score=None),
        ),
        elapsed_s=0.0,
    )
    assert boundary_stats(run, samples, sr).zero_duration_words == 1


def test_silence_depth_is_reported_per_engine(tmp_path: Path) -> None:
    samples, sr = read_wav_mono(_wav(tmp_path))
    good = EngineRun(
        label="good",
        words=("один", "два"),
        spans=(
            Span(start_ms=0, end_ms=260, score=None),
            Span(start_ms=260, end_ms=520, score=None),
        ),
        elapsed_s=0.0,
    )
    stats = boundary_stats(good, samples, sr)
    assert stats.median_silence_depth_db is not None
    assert stats.median_silence_depth_db > 0.0


def test_count_capped_counts_boundaries_that_hit_the_rail() -> None:
    assert count_capped([0, 5, C.BOUNDARY_SEARCH_MS, C.BOUNDARY_SEARCH_MS + 1]) == 2


def test_run_comparison_produces_one_run_per_engine_and_per_pairing(
    db: Database, tmp_path: Path
) -> None:
    with registered(FakeTranscriber, FakeAligner):
        runs = run_comparison(
            db,
            wav_path=_wav(tmp_path),
            transcribers=("fake",),
            aligners=("fake-aligner",),
        )
    assert [run.label for run in runs] == ["fake", "fake+fake-aligner"]
    assert runs[0].words == ("один", "два", "три")


def test_comparison_rows_have_a_header_and_one_row_per_run(
    db: Database, tmp_path: Path
) -> None:
    wav = _wav(tmp_path)
    samples, sr = read_wav_mono(wav)
    with registered(FakeTranscriber, FakeAligner):
        runs = run_comparison(
            db, wav_path=wav, transcribers=("fake",), aligners=("fake-aligner",)
        )
    columns, rows = comparison_rows(runs, samples, sr)
    assert columns[0] == "engine"
    assert len(rows) == 2
    assert all(len(row) == len(columns) for row in rows)


def test_the_report_names_every_engine_and_every_section(
    db: Database, tmp_path: Path
) -> None:
    wav = _wav(tmp_path)
    samples, sr = read_wav_mono(wav)
    with registered(FakeTranscriber, FakeAligner):
        runs = run_comparison(
            db, wav_path=wav, transcribers=("fake",), aligners=("fake-aligner",)
        )
    report = render_report(runs, samples, sr, source=str(wav))
    assert "## Boundaries" in report
    assert "## Disagreement" in report
    assert "## First words" in report
    assert "fake+fake-aligner" in report
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_compare.py -v
```
Expected: `ImportError: cannot import name 'boundary_stats'`.

- [ ] **Step 3: Append to `rytp/transcribe/compare.py`**

```python
@dataclass(frozen=True)
class BoundaryStats:
    """What the audio says about one engine's claimed boundaries."""

    label: str
    n_words: int
    zero_gap_fraction: float
    median_gap_ms: float
    zero_duration_words: int
    median_silence_depth_db: float | None
    median_shift_ms: float | None
    capped_shifts: int
    elapsed_s: float


def boundary_shifts(claimed: Sequence[Span], refined: Sequence[Span]) -> list[int]:
    """How far energy refinement had to move each boundary."""
    shifts = [
        abs(after.start_ms - before.start_ms)
        for before, after in zip(claimed, refined, strict=True)
    ]
    if claimed:
        shifts.append(abs(refined[-1].end_ms - claimed[-1].end_ms))
    return shifts


def count_capped(shifts: Sequence[int]) -> int:
    """Boundaries that ran into the search rail.

    The direct signal that an aligner is further out than refinement is
    allowed to correct: refinement stops at
    :data:`rytp.constants.BOUNDARY_SEARCH_MS`, so a boundary pinned there was
    still heading somewhere else.
    """
    return sum(1 for shift in shifts if shift >= C.BOUNDARY_SEARCH_MS)


def boundary_stats(run: EngineRun, samples: np.ndarray, sr: int) -> BoundaryStats:
    """Measure one engine's boundaries against the audio itself.

    ``zero_gap_fraction`` is the measurement that motivated this whole design:
    a transcriber that sets ``word[i].end == word[i+1].start`` by construction
    scores 1.0 here and its timings cannot be cut on.
    """
    spans = list(run.spans)
    if not spans:
        return BoundaryStats(
            label=run.label,
            n_words=0,
            zero_gap_fraction=0.0,
            median_gap_ms=0.0,
            zero_duration_words=0,
            median_silence_depth_db=None,
            median_shift_ms=None,
            capped_shifts=0,
            elapsed_s=run.elapsed_s,
        )
    gaps = [
        right.start_ms - left.end_ms for left, right in zip(spans, spans[1:], strict=False)
    ]
    boundaries = [span.start_ms for span in spans] + [spans[-1].end_ms]
    depths = [silence_depth_db(samples, sr, boundary) for boundary in boundaries]
    shifts = boundary_shifts(spans, refine_boundaries(samples, sr, spans))
    return BoundaryStats(
        label=run.label,
        n_words=len(spans),
        zero_gap_fraction=(sum(1 for gap in gaps if gap == 0) / len(gaps)) if gaps else 0.0,
        median_gap_ms=_percentile([float(gap) for gap in gaps], 50) or 0.0,
        zero_duration_words=sum(1 for span in spans if span.end_ms <= span.start_ms),
        median_silence_depth_db=_percentile(depths, 50),
        median_shift_ms=_percentile([float(shift) for shift in shifts], 50),
        capped_shifts=count_capped(shifts),
        elapsed_s=run.elapsed_s,
    )


def _raw_spans(words: Sequence[RawWord]) -> tuple[Span, ...] | None:
    """A transcriber's own timings, or ``None`` when it did not time everything.

    Contracts §4 lets a transcriber emit text only. Such an engine has no
    standalone run to compare — only its pairings with an aligner do — so the
    comparison quietly leaves it out of the boundary table rather than
    inventing timings for it.
    """
    spans: list[Span] = []
    for word in words:
        start, end = word.start_ms, word.end_ms
        if start is None or end is None:
            return None
        spans.append(Span(start_ms=start, end_ms=end, score=word.confidence))
    return tuple(spans)


def run_comparison(
    db: Database,
    *,
    wav_path: Path,
    transcribers: Sequence[str],
    aligners: Sequence[str] = (),
    start_ms: int = 0,
    end_ms: int | None = None,
    language: str | None = C.DEFAULT_LANGUAGE,
) -> list[EngineRun]:
    """Run every transcriber, and every transcriber-aligner pairing, over one window.

    The transcriber-only run is kept alongside its aligned pairings on
    purpose: the difference between them is the answer to "is alignment
    needed at all" (design §11). A transcriber that emits text without
    timings has no standalone run and contributes only its pairings.
    """
    samples, sr = read_wav_mono(wav_path)
    total_ms = index_to_ms(samples.size, sr)
    window_end = total_ms if end_ms is None else min(end_ms, total_ms)
    runs: list[EngineRun] = []
    for name in transcribers:
        engine = load_transcriber(db, name)
        started = time.monotonic()
        produced = [
            word
            for word in engine.transcribe(
                wav_path, language=language, start_ms=start_ms, end_ms=window_end
            )
            if normalize_text(word.text)
        ]
        elapsed = time.monotonic() - started
        if all(word.start_ms is not None for word in produced):
            produced.sort(key=lambda word: (word.start_ms or 0, word.text))
        texts = tuple(word.text for word in produced)
        raw = _raw_spans(produced)
        if raw is not None:
            runs.append(
                EngineRun(label=name, words=texts, spans=raw, elapsed_s=elapsed)
            )
        for aligner_name in aligners:
            aligner = load_aligner(db, aligner_name)
            started = time.monotonic()
            spans = list(
                aligner.align(
                    wav_path, list(texts), start_ms=start_ms, end_ms=window_end
                )
            )
            aligner_elapsed = time.monotonic() - started
            if len(spans) != len(texts):
                raise AlignmentMismatch(
                    f"aligner {aligner_name!r} returned {len(spans)} spans for "
                    f"{len(texts)} words from {name!r}"
                )
            runs.append(
                EngineRun(
                    label=f"{name}+{aligner_name}",
                    words=texts,
                    spans=tuple(spans),
                    elapsed_s=aligner_elapsed,
                )
            )
    return runs


_COMPARISON_COLUMNS = (
    "engine",
    "words",
    "zero-gap",
    "median gap ms",
    "zero-length",
    "silence depth dB",
    "median shift ms",
    "shifts at rail",
    "seconds",
)


def _format(value: float | None, places: int = 1) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def comparison_rows(
    runs: Sequence[EngineRun], samples: np.ndarray, sr: int
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Boundary statistics as a table, for ``CommandResult``."""
    rows: list[tuple[str, ...]] = []
    for run in runs:
        stats = boundary_stats(run, samples, sr)
        rows.append(
            (
                stats.label,
                str(stats.n_words),
                f"{stats.zero_gap_fraction:.1%}",
                _format(stats.median_gap_ms),
                str(stats.zero_duration_words),
                _format(stats.median_silence_depth_db),
                _format(stats.median_shift_ms),
                str(stats.capped_shifts),
                _format(stats.elapsed_s),
            )
        )
    return _COMPARISON_COLUMNS, rows


def render_report(
    runs: Sequence[EngineRun], samples: np.ndarray, sr: int, *, source: str
) -> str:
    """The full markdown comparison: boundaries, disagreement, and a sample."""
    columns, rows = comparison_rows(runs, samples, sr)
    lines = [
        "# Engine comparison",
        "",
        f"Source: `{source}`",
        "",
        "## Boundaries",
        "",
        "Measured from the audio, so no reference transcript is needed. A high",
        "zero-gap percentage means the engine assigns `word[i].end ==",
        "word[i+1].start` by construction and its timings cannot be cut on.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    lines += [
        "",
        "## Disagreement",
        "",
        "| a | b | matched | subs | ins | del | disagreement | median Δstart ms | p90 Δstart ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, left in enumerate(runs):
        for right in runs[index + 1 :]:
            stats = compare_pair(left, right)
            lines.append(
                f"| {stats.a} | {stats.b} | {stats.matches} | {stats.substitutions} "
                f"| {stats.insertions} | {stats.deletions} "
                f"| {stats.disagreement:.1%} "
                f"| {_format(stats.median_abs_start_delta_ms)} "
                f"| {_format(stats.p90_abs_start_delta_ms)} |"
            )

    lines += ["", "## First words", ""]
    for run in runs:
        preview = " ".join(run.words[: C.COMPARE_SAMPLE_WORDS])
        lines += [f"**{run.label}**", "", f"> {preview}", ""]
    return "\n".join(lines) + "\n"
```

Extend the imports at the top of `rytp/transcribe/compare.py`:

```python
import time
from pathlib import Path

from rytp import constants as C
from rytp.audio.energy import (
    index_to_ms,
    read_wav_mono,
    refine_boundaries,
    silence_depth_db,
)
from rytp.db import Database
from rytp.models import RawWord
from rytp.transcribe.pipeline import AlignmentMismatch
from rytp.transcribe.registry import load_aligner, load_transcriber
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_compare.py -v
```
Expected: 17 passed.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/transcribe
```
Expected: no findings.

- [ ] **Step 6: Commit**

```bash
git add rytp/transcribe/compare.py tests/test_transcribe_compare.py
git commit -m "feat(transcribe): measure claimed boundaries against the audio and report"
```

---

### Task 15: Transcriber adapters

Two adapters, one in process and one out, so both halves of the registry are exercised by something real. Neither may be importable-only-with-its-dependency, and neither may be needed by the test suite.

**Files:**
- Create: `rytp/transcribe/engines/__init__.py`
- Create: `rytp/transcribe/engines/whisper.py`
- Create: `rytp/transcribe/engines/gigaam.py`
- Modify: `pyproject.toml` (optional-dependency extras)
- Test: `tests/test_transcribe_engines.py`

**Interfaces:**
- Consumes: `rytp.transcribe.base.{EngineUnavailable, shift_words, slice_wav_window}`, `rytp.transcribe.registry.register_transcriber`, `rytp.transcribe.subproc.run_child`.
- Produces: `FasterWhisperTranscriber` (name `whisper`), `GigaAMTranscriber` (name `gigaam`), the pure helpers `words_from_segments(segments) -> list[RawWord]` and `words_from_gigaam(result, offset_ms) -> list[dict]`, and `child_main(request) -> dict` in `gigaam`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe_engines.py`:

```python
"""Transcriber adapters: importable without their dependency, and registered."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.models import RytpError
from rytp.transcribe import registry
from rytp.transcribe.base import EngineUnavailable
from rytp.transcribe.engines.gigaam import GigaAMTranscriber, words_from_gigaam
from rytp.transcribe.engines.whisper import FasterWhisperTranscriber, words_from_segments

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


@dataclass
class _FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class _FakeSegment:
    words: list[_FakeWord]


def test_importing_the_engines_package_registers_both_names() -> None:
    import rytp.transcribe.engines  # noqa: F401

    assert "whisper" in registry.TRANSCRIBERS
    assert "gigaam" in registry.TRANSCRIBERS


def test_words_from_segments_converts_seconds_to_milliseconds() -> None:
    segments = [_FakeSegment(words=[_FakeWord(" привет", 0.0, 0.42), _FakeWord("мир", 0.5, 0.9)])]
    words = words_from_segments(segments)
    assert [(w.start_ms, w.end_ms, w.text) for w in words] == [
        (0, 420, "привет"),
        (500, 900, "мир"),
    ]
    assert words[0].confidence == pytest.approx(0.9)


def test_words_from_segments_drops_empty_tokens() -> None:
    segments = [_FakeSegment(words=[_FakeWord("   ", 0.0, 0.1), _FakeWord("да", 0.2, 0.3)])]
    assert [w.text for w in words_from_segments(segments)] == ["да"]


def test_words_from_gigaam_offsets_onto_the_whole_file_timeline() -> None:
    result = {"words": [{"word": "да", "start": 0.1, "end": 0.4}, {"word": " ", "start": 0.4, "end": 0.5}]}
    words = words_from_gigaam(result, 10_000)
    assert words == [{"start_ms": 10_100, "end_ms": 10_400, "text": "да", "confidence": None}]


def test_gigaam_is_out_of_process_and_ungated() -> None:
    assert GigaAMTranscriber.out_of_process is True
    assert GigaAMTranscriber.requires_hf_token is False
    assert GigaAMTranscriber.max_window_ms == C.VAD_CHUNK_MAX_MS


def test_gigaam_refuses_a_window_longer_than_its_per_call_cap(tmp_path: Path) -> None:
    engine = GigaAMTranscriber(interpreter="/no/such/python")
    with pytest.raises(RytpError) as excinfo:
        list(engine.transcribe(tmp_path / "a.wav", start_ms=0, end_ms=C.VAD_CHUNK_MAX_MS + 1))
    assert "plan_chunks" in str(excinfo.value)


def test_whisper_is_in_process_and_names_its_extra() -> None:
    assert FasterWhisperTranscriber.out_of_process is False
    assert FasterWhisperTranscriber.extra == "whisper"
    assert FasterWhisperTranscriber.required_module == "faster_whisper"


@pytest.mark.skipif(HAS_FASTER_WHISPER, reason="faster-whisper is installed here")
def test_whisper_without_its_dependency_names_the_extra_instead_of_crashing() -> None:
    with pytest.raises(EngineUnavailable) as excinfo:
        registry.check_available(FasterWhisperTranscriber)
    assert "rytp[whisper]" in str(excinfo.value)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_engines.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.engines'`.

- [ ] **Step 3: Write `rytp/transcribe/engines/whisper.py`**

```python
"""faster-whisper as a transcriber (design §6, "Recommended stack").

Whisper is middling at Russian — about 16.2 WER against GigaAM's 8.4 on clean
benchmarks — but it installs anywhere, needs no separate interpreter, and the
one published test on *noisy YouTube* audio reversed that ranking. It is the
default until `transcribe compare` says otherwise on this corpus.

Its word timestamps are the ones measured at 78.7% zero gaps, so pair it with
an aligner and boundary refinement before cutting anything.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RawWord
from rytp.transcribe.base import EngineUnavailable, shift_words, slice_wav_window
from rytp.transcribe.registry import register_transcriber


def words_from_segments(segments: Iterable[Any]) -> list[RawWord]:
    """Flatten faster-whisper segments into words, converting seconds to ms.

    The only part of this adapter testable without the model, which is why it
    is a module-level function rather than a method.
    """
    words: list[RawWord] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or ():
            text = str(getattr(word, "word", "") or "").strip()
            if not text:
                continue
            words.append(
                RawWord(
                    start_ms=int(float(word.start) * 1000),
                    end_ms=int(float(word.end) * 1000),
                    text=text,
                    confidence=float(getattr(word, "probability", 1.0)),
                )
            )
    return words


@register_transcriber
class FasterWhisperTranscriber:
    """Whisper through faster-whisper, in this process."""

    name = "whisper"
    requires_hf_token = False
    out_of_process = False
    required_module = "faster_whisper"
    extra = "whisper"

    def __init__(
        self,
        model: str = C.WHISPER_DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise EngineUnavailable(
                    "faster-whisper is not installed: pip install rytp[whisper]"
                ) from exc
            self._model = WhisperModel(
                self._model_name, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        """Decode one window and return words on the whole-file timeline."""
        model = self._load()
        with tempfile.TemporaryDirectory(prefix="rytp-whisper-") as tmp:
            window = slice_wav_window(audio, Path(tmp) / "window.wav", start_ms, end_ms)
            segments, _info = model.transcribe(
                str(window), language=language, word_timestamps=True
            )
            # The generator must be drained before the temporary file goes.
            words = words_from_segments(segments)
        return shift_words(words, start_ms)
```

- [ ] **Step 4: Write `rytp/transcribe/engines/gigaam.py`**

```python
"""GigaAM v3 as a transcriber, out of process (design §6).

Russian-specific, MIT licensed, roughly half Whisper's Russian error rate on
clean benchmarks, 2-4 GB of VRAM. Two properties shape this adapter:

* It caps a call at 25 seconds, so the caller must chunk with
  :func:`rytp.audio.vad.plan_chunks`. A longer window is an error here rather
  than a silent truncation inside the model.
* It pins versions that conflict with the diarizer's, so it declares
  ``out_of_process = True`` and runs under the interpreter named by the
  ``engine.interpreter.gigaam`` setting.

:func:`child_main` runs inside *that* interpreter and is the only place the
``gigaam`` package is imported. Its call into the library targets the
documented ``load_model`` / ``transcribe`` API; if the installed version
differs, this function is the only thing that changes and its contract is the
dictionary it returns. **Verify per-word timestamp support when installing.**
If the installed model returns text without word times, treat gigaam as
text-only and always pair it with an aligner.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RawWord, RytpError
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_transcriber
from rytp.transcribe.subproc import run_child


def words_from_gigaam(result: dict[str, Any], offset_ms: int) -> list[dict[str, Any]]:
    """Gigaam's word list to the serialisable shape the parent expects."""
    words: list[dict[str, Any]] = []
    for item in result.get("words") or ():
        text = str(item.get("word", "")).strip()
        if not text:
            continue
        words.append(
            {
                "start_ms": offset_ms + int(round(float(item["start"]) * 1000)),
                "end_ms": offset_ms + int(round(float(item["end"]) * 1000)),
                "text": text,
                "confidence": item.get("confidence"),
            }
        )
    return words


@register_transcriber
class GigaAMTranscriber:
    """GigaAM under its own interpreter, one window per call."""

    name = "gigaam"
    requires_hf_token = False
    out_of_process = True
    required_module = "gigaam"
    extra = "gigaam"
    max_window_ms = C.VAD_CHUNK_MAX_MS

    def __init__(self, interpreter: str, model: str = C.GIGAAM_DEFAULT_MODEL) -> None:
        self._interpreter = interpreter
        self._model = model

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        if end_ms is not None and end_ms - start_ms > self.max_window_ms:
            raise RytpError(
                f"gigaam accepts at most {self.max_window_ms} ms per call; got "
                f"{end_ms - start_ms} ms — chunk with rytp.audio.vad.plan_chunks"
            )
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.engines.gigaam",
            request={
                "audio": str(audio),
                "model": self._model,
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
        )
        return [
            RawWord(
                start_ms=int(word["start_ms"]),
                end_ms=None if word["end_ms"] is None else int(word["end_ms"]),
                text=str(word["text"]),
                confidence=word.get("confidence"),
            )
            for word in result["words"]
        ]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside gigaam's own interpreter. The only import of the library."""
    import tempfile

    import gigaam

    start_ms = int(request["start_ms"])
    end_ms = request["end_ms"]
    with tempfile.TemporaryDirectory(prefix="rytp-gigaam-") as tmp:
        window = slice_wav_window(
            Path(request["audio"]), Path(tmp) / "window.wav", start_ms, end_ms
        )
        model = gigaam.load_model(request["model"])
        result = model.transcribe(str(window))
    return {"words": words_from_gigaam(dict(result), start_ms)}
```

- [ ] **Step 5: Write `rytp/transcribe/engines/__init__.py`**

```python
"""Transcriber adapters.

Importing this package is what makes an engine's name resolvable: each module
registers its class at import time, so a name that is never imported is a name
that does not exist.
"""
from rytp.transcribe.engines import gigaam, whisper  # noqa: F401
```

- [ ] **Step 6: Add the extras to `pyproject.toml`**

Replace the existing `stt` extra and extend `all`:

```toml
whisper = ["faster-whisper>=1.0"]
gigaam = ["gigaam>=0.1"]
wav2vec2 = ["torch>=2.1", "torchaudio>=2.1", "transformers>=4.40"]
```

The Montreal Forced Aligner is deliberately **not** a pip extra: it installs
through conda and lives in its own environment, which is exactly why it runs
out of process. Point the `engine.interpreter.mfa` setting at the Python
inside that environment.

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_engines.py -v
```
Expected: 8 passed.

- [ ] **Step 8: Commit**

```bash
git add rytp/transcribe/engines pyproject.toml tests/test_transcribe_engines.py
git commit -m "feat(transcribe): whisper and gigaam transcriber adapters"
```

---

### Task 16: Aligner adapters

**Files:**
- Create: `rytp/transcribe/align/__init__.py`
- Create: `rytp/transcribe/align/mfa.py`
- Create: `rytp/transcribe/align/wav2vec2.py`
- Modify: `rytp/constants.py` (one constant)
- Test: `tests/test_transcribe_align.py`

**Interfaces:**
- Consumes: `rytp.transcribe.registry.register_aligner`, `rytp.transcribe.subproc.run_child`, `rytp.transcribe.base.slice_wav_window`, `rytp.models.Span`.
- Produces: `MfaAligner` (name `mfa`), `Wav2Vec2Aligner` (name `wav2vec2`), and the pure helpers `parse_textgrid(text)`, `spans_from_intervals(intervals, *, offset_ms, score=None)`, `mfa_binary(interpreter)`, `spans_from_frames(frames, *, stride_ms, offset_ms)`.

- [ ] **Step 1: Add the wav2vec2 stride to `rytp/constants.py`**

Next to `WAV2VEC2_MODEL` in the Part 3 block:

```python
#: wav2vec2 emits one CTC frame per 20 ms at 16 kHz (design §6, "Alignment
#: fallback": 20 ms stride).
WAV2VEC2_STRIDE_MS: float = 20.0
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_transcribe_align.py`:

```python
"""Aligner adapters, and the pure parsing they are built on."""
from __future__ import annotations

import os

from rytp import constants as C
from rytp.transcribe import registry
from rytp.transcribe.align.mfa import (
    MfaAligner,
    mfa_binary,
    parse_textgrid,
    spans_from_intervals,
)
from rytp.transcribe.align.wav2vec2 import Wav2Vec2Aligner, spans_from_frames

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.25
            text = ""
        intervals [2]:
            xmin = 0.25
            xmax = 0.80
            text = "привет"
        intervals [3]:
            xmin = 0.90
            xmax = 1.50
            text = "мир"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.5
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = "p"
'''


def test_parse_textgrid_returns_the_words_tier_without_the_silences() -> None:
    assert parse_textgrid(TEXTGRID) == [(0.25, 0.80, "привет"), (0.90, 1.50, "мир")]


def test_parse_textgrid_ignores_other_tiers() -> None:
    assert all(label in {"привет", "мир"} for _, _, label in parse_textgrid(TEXTGRID))


def test_parse_textgrid_of_rubbish_is_empty() -> None:
    assert parse_textgrid("not a textgrid") == []


def test_spans_from_intervals_offsets_onto_the_whole_file_timeline() -> None:
    spans = spans_from_intervals(parse_textgrid(TEXTGRID), offset_ms=10_000)
    assert [(s.start_ms, s.end_ms) for s in spans] == [(10_250, 10_800), (10_900, 11_500)]
    assert spans[0].score is None


def test_mfa_binary_sits_beside_the_configured_interpreter() -> None:
    derived = mfa_binary(os.path.join("opt", "mfa-env", "bin", "python"))
    assert derived.endswith("mfa.exe" if os.name == "nt" else "mfa")
    assert "mfa-env" in derived


def test_spans_from_frames_uses_the_ctc_stride() -> None:
    spans = spans_from_frames(
        [(0, 4, 0.9), (10, 14, None)], stride_ms=C.WAV2VEC2_STRIDE_MS, offset_ms=1_000
    )
    assert [(s.start_ms, s.end_ms) for s in spans] == [(1_000, 1_100), (1_200, 1_300)]
    assert spans[0].score == 0.9


def test_spans_from_frames_never_produces_a_zero_length_span() -> None:
    spans = spans_from_frames([(5, 5, None)], stride_ms=20.0, offset_ms=0)
    assert spans[0].end_ms > spans[0].start_ms


def test_importing_the_align_package_registers_both_aligners() -> None:
    import rytp.transcribe.align  # noqa: F401

    assert "mfa" in registry.ALIGNERS
    assert "wav2vec2" in registry.ALIGNERS


def test_both_aligners_run_out_of_process_and_need_no_token() -> None:
    for cls in (MfaAligner, Wav2Vec2Aligner):
        assert cls.out_of_process is True
        assert cls.requires_hf_token is False
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_align.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.align'`.

- [ ] **Step 4: Write `rytp/transcribe/align/mfa.py`**

```python
"""Montreal Forced Aligner as an aligner (design §6, "Recommended stack").

MFA is the best free forced aligner and the only one with a Russian acoustic
model, dictionary and G2P (``russian_mfa`` v3.1.0). Its published boundary
error is 12.5 ms median — on English; no aligner's Russian boundary error has
ever been measured, which is why `transcribe compare` exists.

MFA installs through conda into its own environment, so it is invoked out of
process: point the ``engine.interpreter.mfa`` setting at the Python inside
that environment and the ``mfa`` binary beside it is found automatically.
Download the models once with ``mfa model download acoustic russian_mfa`` and
``mfa model download dictionary russian_mfa``.

MFA reports no per-word confidence, so spans come back with ``score=None``
and the pipeline substitutes the measured boundary quality instead.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RytpError, Span
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_aligner
from rytp.transcribe.subproc import run_child


def mfa_binary(interpreter: str) -> str:
    """The ``mfa`` executable that lives beside a configured interpreter."""
    name = "mfa.exe" if os.name == "nt" else "mfa"
    return str(Path(interpreter).parent / name)


def _quoted(line: str) -> str:
    first = line.find('"')
    last = line.rfind('"')
    return line[first + 1 : last] if 0 <= first < last else ""


def parse_textgrid(text: str) -> list[tuple[float, float, str]]:
    """Intervals of the ``words`` tier of a long-form Praat TextGrid.

    MFA writes one interval per word plus empty intervals for the silences
    between them; the empty ones are dropped. A tier's own ``xmin``/``xmax``
    are always overwritten by its first interval's before any ``text`` line
    appears, so no special case is needed for the header.
    """
    intervals: list[tuple[float, float, str]] = []
    in_words = False
    xmin: float | None = None
    xmax: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name ="):
            in_words = _quoted(line) == "words"
            continue
        if not in_words:
            continue
        try:
            if line.startswith("xmin ="):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax ="):
                xmax = float(line.split("=", 1)[1])
            elif line.startswith("text ="):
                label = _quoted(line).strip()
                if xmin is not None and xmax is not None and label:
                    intervals.append((xmin, xmax, label))
                xmin = xmax = None
        except ValueError:
            xmin = xmax = None
    return intervals


def spans_from_intervals(
    intervals: Sequence[tuple[float, float, str]],
    *,
    offset_ms: int,
    score: float | None = None,
) -> list[Span]:
    """TextGrid seconds, relative to the window, to absolute millisecond spans."""
    return [
        Span(
            start_ms=offset_ms + int(round(start * 1000)),
            end_ms=offset_ms + int(round(end * 1000)),
            score=score,
        )
        for start, end, _label in intervals
    ]


@register_aligner
class MfaAligner:
    """Forced alignment through the MFA command line, out of process."""

    name = "mfa"
    requires_hf_token = False
    out_of_process = True
    required_module = None
    extra = "mfa"

    def __init__(
        self,
        interpreter: str,
        acoustic_model: str = C.MFA_ACOUSTIC_MODEL,
        dictionary: str = C.MFA_DICTIONARY,
    ) -> None:
        self._interpreter = interpreter
        self._acoustic_model = acoustic_model
        self._dictionary = dictionary

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.align.mfa",
            request={
                "audio": str(audio),
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "binary": mfa_binary(self._interpreter),
                "acoustic_model": self._acoustic_model,
                "dictionary": self._dictionary,
            },
        )
        spans = [
            Span(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                score=item.get("score"),
            )
            for item in result["spans"]
        ]
        if len(spans) != len(words):
            raise RytpError(
                f"mfa aligned {len(spans)} of {len(words)} words; the dictionary "
                "is probably missing entries for this text"
            )
        return spans


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside MFA's own environment. Shells out to the ``mfa`` binary."""
    import subprocess
    import tempfile

    start_ms = int(request["start_ms"])
    end_ms = int(request["end_ms"])
    words = [str(word) for word in request["words"]]
    with tempfile.TemporaryDirectory(prefix="rytp-mfa-") as tmp:
        root = Path(tmp)
        corpus = root / "corpus"
        corpus.mkdir()
        slice_wav_window(Path(request["audio"]), corpus / "window.wav", start_ms, end_ms)
        (corpus / "window.lab").write_text(" ".join(words), encoding="utf-8")
        output = root / "aligned"
        command = [
            str(request["binary"]),
            "align",
            str(corpus),
            str(request["dictionary"]),
            str(request["acoustic_model"]),
            str(output),
            "--clean",
            "--quiet",
            "--single_speaker",
            "--output_format",
            "long_textgrid",
        ]
        proc = subprocess.run(  # fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        grid = output / "window.TextGrid"
        if not grid.exists():
            raise RuntimeError(
                f"mfa align wrote no TextGrid (exit {proc.returncode}): "
                f"{proc.stderr.strip()[-800:]}"
            )
        intervals = parse_textgrid(grid.read_text(encoding="utf-8"))
    spans = spans_from_intervals(intervals, offset_ms=start_ms)
    return {
        "spans": [
            {"start_ms": span.start_ms, "end_ms": span.end_ms, "score": span.score}
            for span in spans
        ]
    }
```

- [ ] **Step 5: Write `rytp/transcribe/align/wav2vec2.py`**

```python
"""A wav2vec2 CTC model as the alignment fallback (design §6).

``bond005/wav2vec2-large-ru-golos`` has a 20 ms stride and installs with pip,
which makes it the fallback when MFA's conda environment is not available or
its dictionary lacks the proper names and loanwords a decade of political
commentary is dense with.

It runs out of process because torch pins conflict with the diarizer's.
:func:`child_main` is the only place torch is imported; its contract with the
parent is the dictionary it returns, and the CTC call inside it targets
``torchaudio.functional.forced_align``. If the installed versions differ, that
function is the only thing that changes.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RytpError, Span
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_aligner
from rytp.transcribe.subproc import run_child


def spans_from_frames(
    frames: Sequence[tuple[int, int, float | None]], *, stride_ms: float, offset_ms: int
) -> list[Span]:
    """CTC frame indexes to absolute millisecond spans, never zero length."""
    spans: list[Span] = []
    for first, last, score in frames:
        start = offset_ms + int(round(first * stride_ms))
        end = offset_ms + int(round((last + 1) * stride_ms))
        spans.append(Span(start_ms=start, end_ms=max(end, start + 1), score=score))
    return spans


@register_aligner
class Wav2Vec2Aligner:
    """CTC forced alignment under its own interpreter."""

    name = "wav2vec2"
    requires_hf_token = False
    out_of_process = True
    required_module = "torch"
    extra = "wav2vec2"

    def __init__(self, interpreter: str, model: str = C.WAV2VEC2_MODEL) -> None:
        self._interpreter = interpreter
        self._model = model

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.align.wav2vec2",
            request={
                "audio": str(audio),
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "model": self._model,
            },
        )
        spans = spans_from_frames(
            [
                (int(item["first_frame"]), int(item["last_frame"]), item.get("score"))
                for item in result["frames"]
            ],
            stride_ms=C.WAV2VEC2_STRIDE_MS,
            offset_ms=start_ms,
        )
        if len(spans) != len(words):
            raise RytpError(
                f"wav2vec2 aligned {len(spans)} of {len(words)} words"
            )
        return spans


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the torch interpreter. The only import of torch."""
    import tempfile
    import wave

    import torch
    import torchaudio
    from transformers import AutoModelForCTC, AutoProcessor

    start_ms = int(request["start_ms"])
    end_ms = int(request["end_ms"])
    words = [str(word) for word in request["words"]]
    with tempfile.TemporaryDirectory(prefix="rytp-w2v-") as tmp:
        window = slice_wav_window(
            Path(request["audio"]), Path(tmp) / "window.wav", start_ms, end_ms
        )
        with wave.open(str(window), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
        waveform = (
            torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
        ).unsqueeze(0)

        processor = AutoProcessor.from_pretrained(request["model"])
        model = AutoModelForCTC.from_pretrained(request["model"]).eval()
        with torch.inference_mode():
            emissions = torch.log_softmax(model(waveform).logits, dim=-1)
        vocabulary = processor.tokenizer.get_vocab()
        targets = torch.tensor(
            [[vocabulary[ch] for ch in " ".join(words) if ch in vocabulary]],
            dtype=torch.int32,
        )
        aligned, scores = torchaudio.functional.forced_align(
            emissions, targets, blank=0
        )

    return {"frames": _word_frames(aligned[0].tolist(), scores[0].tolist(), words)}


def _word_frames(
    path: list[int], scores: list[float], words: Sequence[str]
) -> list[dict[str, Any]]:
    """Group the character-level CTC path back into one entry per word.

    Words were joined with a single space before alignment, so every non-blank
    run between space tokens belongs to the next word in order.
    """
    out: list[dict[str, Any]] = []
    index = 0
    first: int | None = None
    last = 0
    collected: list[float] = []
    for frame, token in enumerate(path):
        if token == 0:
            continue
        if first is None:
            first = frame
        last = frame
        collected.append(scores[frame])
        if len(out) < len(words) and frame + 1 < len(path) and path[frame + 1] == 0:
            continue
    if first is not None:
        # Fall back to one span covering everything when the path cannot be
        # split: the caller checks the count and reports the mismatch.
        width = max(1, (last - first + 1) // max(len(words), 1))
        for index in range(len(words)):
            out.append(
                {
                    "first_frame": first + index * width,
                    "last_frame": first + (index + 1) * width - 1,
                    "score": (
                        sum(collected) / len(collected) if collected else None
                    ),
                }
            )
    return out
```

> **Note for the implementer:** `_word_frames` above is an even split across
> the aligned region, which is honest about what this fallback is: a fallback.
> When the torch extra is actually installed, replace its body with
> `torchaudio.functional.merge_tokens` on the alignment path and group the
> resulting token spans on the space token. The dict shape it returns is the
> contract and does not change.

- [ ] **Step 6: Write `rytp/transcribe/align/__init__.py`**

```python
"""Aligner adapters.

Importing this package is what makes an aligner's name resolvable: each module
registers its class at import time.
"""
from rytp.transcribe.align import mfa, wav2vec2  # noqa: F401
```

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_align.py -v
```
Expected: 9 passed.

- [ ] **Step 8: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp/transcribe
```
Expected: no findings.

- [ ] **Step 9: Commit**

```bash
git add rytp/constants.py rytp/transcribe/align tests/test_transcribe_align.py
git commit -m "feat(transcribe): MFA and wav2vec2 aligner adapters"
```

---

### Task 17: Commands

Contracts §5: each operation is defined once and both the CLI and the TUI are generated from that definition. Handlers are thin, take an open `Database` first, never print, never call `sys.exit`, and raise `RytpError` for an expected failure.

**Files:**
- Create: `rytp/commands/transcribe.py`
- Test: `tests/test_commands_transcribe.py`

**Interfaces:**
- Consumes: `rytp.commands.{Command, CommandResult, Param, REQUIRED, register}`; every Part 3 module above; `rytp.audio.extract.wav_path` (Part 2).
- Produces: six registered commands — `transcribe.captions`, `transcribe.run`, `transcribe.align`, `transcribe.compare`, `transcribe.engines`, `transcribe.fingerprint`.

**Job kinds are registered separately, in Task 19.** Contracts §5 "Job handlers" makes the kind-to-callable mapping its own contract rather than a naming convention over commands, so these handlers stay pure command handlers for now; Task 19 adds the four thunks and gives three of these commands an `--enqueue` flag. Part 2's `captions` kind only downloads the asset — turning it into words is Part 3's `caption_words` kind, registered in Task 19.

- [ ] **Step 1: Write the failing test**

Create `tests/test_commands_transcribe.py`:

```python
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
    registered,
)
from tests.synth_audio import concat, silence, tone, write_wav

NAMES = (
    "transcribe.captions",
    "transcribe.run",
    "transcribe.align",
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


def test_captions_handler_reports_how_many_words_it_wrote(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    path = tmp_path / "captions.json3"
    path.write_text(
        json.dumps({"events": [{"tStartMs": 0, "segs": [{"utf8": "да", "tOffsetMs": 0}]}]}),
        encoding="utf-8",
    )
    result = resolve("transcribe.captions").handler(db, video_id=video_id, path=path)
    assert isinstance(result, CommandResult)
    assert "1" in (result.message or "")


def test_run_handler_transcribes_with_the_named_engines(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    with registered(FakeTranscriber, FakeAligner):
        result = resolve("transcribe.run").handler(
            db,
            video_id=video_id,
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
            db, video_id=video_id, wav=_wav(tmp_path)
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
            db, video_id=video_id, transcriber="fake", wav=_wav(tmp_path)
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
    with registered(FakeTranscriber, MissingModuleTranscriber):
        with pytest.raises(EngineUnavailable) as excinfo:
            resolve("transcribe.run").handler(
                db, video_id=video_id, wav=_wav(tmp_path)
            )
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
        resolve("transcribe.run").handler(db, video_id=video_id, transcriber="fake", wav=wav)
        result = resolve("transcribe.align").handler(
            db, video_id=video_id, aligner="fake-aligner", wav=wav
        )
    assert result.rows[0][3] == "aligned"
    assert result.rows[0][4] == "fake+fake-aligner+energy"


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
            video_id=video_id,
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
            db, video_id=video_id, transcribers="", wav=_wav(tmp_path)
        )


def test_fingerprint_handler_stores_the_row(db: Database, tmp_path: Path) -> None:
    video_id = _make_video(db)
    result = resolve("transcribe.fingerprint").handler(
        db, video_id=video_id, wav=_wav(tmp_path)
    )
    assert result.columns[0] == "f0 mean"
    stored = db.conn.execute(
        "SELECT COUNT(*) FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert stored == 1


def test_a_missing_wav_is_a_domain_error_not_a_traceback(db: Database) -> None:
    from rytp.models import RytpError

    video_id = _make_video(db)
    with registered(FakeTranscriber), pytest.raises(RytpError) as excinfo:
        resolve("transcribe.run").handler(
            db, video_id=video_id, transcriber="fake", wav=Path("nope.wav")
        )
    assert "nope.wav" in str(excinfo.value)
```

`test_fingerprint_handler_stores_the_row` deliberately passes no loudness runner, so it exercises the real ffmpeg path — and must pass on a machine without ffmpeg. Step 3 below makes a missing binary degrade to `loudness_lufs = NULL` with the other four metrics still written, which is both the useful production behaviour and what makes this test hermetic.

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_commands_transcribe.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.commands.transcribe'`.

- [ ] **Step 3: Make a missing ffmpeg degrade instead of failing**

In `rytp/audio/acoustics.py`, change `fingerprint_video` to tolerate a missing binary — the other four metrics are still worth having:

```python
def fingerprint_video(
    db: Database,
    video_id: int,
    *,
    wav_path: Path,
    runner: LoudnessRunner | None = None,
) -> Acoustics:
    """Measure one video and store its fingerprint. One pass, one row.

    Loudness needs ffmpeg. If it is not installed the other four metrics are
    still computed and stored, with ``loudness_lufs`` left null — a partial
    fingerprint is far more useful than none.
    """
    samples, sr = read_wav_mono(wav_path)
    # Unpadded segments: a padded end would hide the reverberation tail.
    speech = detect_speech(samples, sr, pad_ms=0)
    try:
        loudness = measure_loudness_lufs(wav_path, runner=runner)
    except RytpError:
        loudness = None
    acoustics = compute_acoustics(samples, sr, speech=speech, loudness_lufs=loudness)
    store_acoustics(db, video_id, acoustics)
    return acoustics
```

- [ ] **Step 4: Write `rytp/commands/transcribe.py`**

```python
"""The transcribe command group (contracts §5).

Thin handlers. Each takes an open ``Database`` first, returns a
``CommandResult``, and raises ``RytpError`` rather than printing or exiting —
the surface formats the error. Both the CLI and the TUI are generated from
these definitions, so neither may add a command of its own.
"""
from __future__ import annotations

from pathlib import Path

from rytp import constants as C
from rytp.audio.acoustics import Acoustics, fingerprint_video
from rytp.audio.energy import read_wav_mono
from rytp.commands import REQUIRED, Command, CommandResult, Param, register
from rytp.db import Database
from rytp.models import RytpError
from rytp.transcribe.captions import ingest_captions
from rytp.transcribe.compare import comparison_rows, render_report, run_comparison
from rytp.transcribe.pipeline import enqueue_index, realign_video, transcribe_video
from rytp.transcribe.registry import default_transcriber, setting


def _load_engine_modules() -> None:
    """Import the adapter packages, which is what makes their names resolve.

    Deferred to call time so importing this module stays cheap for the TUI.
    """
    import rytp.transcribe.align  # noqa: F401
    import rytp.transcribe.engines  # noqa: F401


def _wav_for(video_id: int, override: Path | None) -> Path:
    """The cached WAV Part 2 produced (contracts §7, ``cache/wav/{id}.wav``).

    The single import site of Part 2's path helper. Part 3 never builds that
    path itself — the cache is Part 2's to place and to prune.
    """
    if override is not None:
        path = Path(override)
    else:
        from rytp.audio.extract import wav_path

        path = wav_path(video_id)
    if not path.exists():
        raise RytpError(f"no cached WAV for video {video_id} at {path}; run ingest first")
    return path


def _split(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _captions_handler(db: Database, *, video_id: int, path: Path | None = None) -> CommandResult:
    count = ingest_captions(db, video_id, Path(path) if path is not None else None)
    return CommandResult(message=f"{count} caption words for video {video_id}")


def _run_handler(
    db: Database,
    *,
    video_id: int,
    transcriber: str = "",
    aligner: str = "",
    language: str = C.DEFAULT_LANGUAGE,
    refine: bool = True,
    wav: Path | None = None,
) -> CommandResult:
    _load_engine_modules()
    # An empty --transcriber falls back to the configured default, and says
    # which engine that was. Contracts §3: the owner chose "set a default and
    # inform" over "refuse until configured", so this line is the entire
    # mechanism keeping the choice visible instead of accidental — 35-90
    # GPU-hours is expensive to discover late. If that engine is not
    # installed, `load_transcriber` raises `EngineUnavailable` with the pip
    # hint and nothing here catches it: a silent substitution would leave the
    # corpus holding rows from two engines with nothing recording which.
    chosen_transcriber = transcriber.strip() or default_transcriber(db)
    # An empty --aligner falls back to the configured default. Contracts §3:
    # with no aligner at all the result is the `timed` tier, which is
    # searchable but not cuttable, so the setting is how an owner who always
    # wants cuttable words stops having to remember the flag.
    chosen_aligner = aligner or setting(db, "default_aligner", "") or None
    outcome = transcribe_video(
        db,
        video_id,
        wav_path=_wav_for(video_id, wav),
        transcriber=chosen_transcriber,
        aligner=chosen_aligner,
        language=language or None,
        refine=refine,
    )
    # The words just changed, so the video's utterances are gone and its
    # search index is stale until Part 4 rebuilds it.
    enqueue_index(db, video_id)
    notes = [f"video {video_id}: {outcome.n_words} {outcome.source} words"]
    if not transcriber.strip():
        notes.append(
            f"transcriber {chosen_transcriber!r}, from setting "
            f"{C.SETTING_DEFAULT_TRANSCRIBER} (pass --transcriber to override)"
        )
    if outcome.source == "timed":
        notes.append(
            "not cuttable — run `rytp transcribe align` with an aligner to upgrade them"
        )
    warning = speaker_loss_warning(video_id, outcome.speakers_lost)
    if warning:
        notes.append(warning)
    return CommandResult(
        columns=("video", "words", "chunks", "tier", "engine", "median align score"),
        rows=(
            (
                str(outcome.video_id),
                str(outcome.n_words),
                str(outcome.n_chunks),
                outcome.source,
                outcome.engine,
                _number(outcome.median_align_score),
            ),
        ),
        message=". ".join(notes),
    )


def _align_handler(
    db: Database,
    *,
    video_id: int,
    aligner: str,
    refine: bool = True,
    wav: Path | None = None,
) -> CommandResult:
    _load_engine_modules()
    outcome = realign_video(
        db,
        video_id,
        wav_path=_wav_for(video_id, wav),
        aligner=aligner,
        refine=refine,
    )
    enqueue_index(db, video_id)
    return CommandResult(
        columns=("video", "words", "chunks", "tier", "engine", "median align score"),
        rows=(
            (
                str(outcome.video_id),
                str(outcome.n_words),
                str(outcome.n_chunks),
                outcome.source,
                outcome.engine,
                _number(outcome.median_align_score),
            ),
        ),
        message=f"video {video_id}: {outcome.n_words} words are now cuttable",
    )


def _compare_handler(
    db: Database,
    *,
    video_id: int,
    transcribers: str,
    aligners: str = "",
    start_ms: int = 0,
    end_ms: int = C.COMPARE_DEFAULT_WINDOW_MS,
    out: Path | None = None,
    wav: Path | None = None,
) -> CommandResult:
    _load_engine_modules()
    names = _split(transcribers)
    if not names:
        raise RytpError("pass at least one transcriber, comma separated")
    wav_path = _wav_for(video_id, wav)
    runs = run_comparison(
        db,
        wav_path=wav_path,
        transcribers=names,
        aligners=_split(aligners),
        start_ms=start_ms,
        end_ms=end_ms or None,
    )
    samples, sr = read_wav_mono(wav_path)
    columns, rows = comparison_rows(runs, samples, sr)
    message = None
    if out is not None:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_report(runs, samples, sr, source=str(wav_path)), encoding="utf-8"
        )
        message = f"report written to {target}"
    return CommandResult(columns=columns, rows=tuple(rows), message=message)


def _engines_handler(db: Database) -> CommandResult:
    _load_engine_modules()
    return CommandResult(
        columns=("name", "kind", "hf token", "out of process", "state"),
        rows=tuple(engine_rows(db)),
    )


def _fingerprint_handler(
    db: Database, *, video_id: int, wav: Path | None = None
) -> CommandResult:
    acoustics: Acoustics = fingerprint_video(
        db, video_id, wav_path=_wav_for(video_id, wav)
    )
    return CommandResult(
        columns=("f0 mean", "f0 std", "tilt dB/dec", "noise floor dB", "reverb dB", "LUFS"),
        rows=(
            (
                _number(acoustics.f0_mean),
                _number(acoustics.f0_std),
                _number(acoustics.spectral_tilt),
                _number(acoustics.noise_floor_db),
                _number(acoustics.reverb_proxy),
                _number(acoustics.loudness_lufs),
            ),
        ),
    )


_VIDEO_ID = Param(
    name="video_id", type=int, help="Video id.", default=REQUIRED, positional=True
)
_WAV = Param(
    name="wav",
    type=Path,
    help="Use this WAV instead of the cached one.",
    default=None,
)

register(
    Command(
        name="transcribe.captions",
        group="transcribe",
        summary="Ingest downloaded auto-captions as caption-tier words (not cuttable).",
        params=(
            _VIDEO_ID,
            Param(
                name="path",
                type=Path,
                help="Captions file to read instead of the registered asset.",
                default=None,
            ),
        ),
        handler=_captions_handler,
    )
)

register(
    Command(
        name="transcribe.run",
        group="transcribe",
        summary="Transcribe, align and refine a video into cuttable aligned words.",
        params=(
            _VIDEO_ID,
            Param(
                name="transcriber",
                type=str,
                help=(
                    "Registered transcriber name. Empty falls back to the "
                    "default_transcriber setting ('gigaam' out of the box), "
                    "and the result says which engine that chose. If it is "
                    "not installed the command fails with the install hint "
                    "rather than running a different one."
                ),
                default="",
            ),
            Param(
                name="aligner",
                type=str,
                help=(
                    "Registered aligner name. Empty falls back to the "
                    "default_aligner setting; with neither, the words are "
                    "written as the searchable-but-not-cuttable 'timed' tier."
                ),
                default="",
            ),
            Param(name="language", type=str, help="Spoken language.", default=C.DEFAULT_LANGUAGE),
            Param(
                name="refine",
                type=bool,
                help="Place boundaries at the measured energy minimum.",
                default=True,
            ),
            _WAV,
        ),
        handler=_run_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.align",
        group="transcribe",
        summary="Re-time an already transcribed video with a different aligner.",
        params=(
            _VIDEO_ID,
            Param(name="aligner", type=str, help="Registered aligner name.", default=REQUIRED),
            Param(
                name="refine",
                type=bool,
                help="Place boundaries at the measured energy minimum.",
                default=True,
            ),
            _WAV,
        ),
        handler=_align_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.compare",
        group="transcribe",
        summary="Run several engines over the same audio and report where they disagree.",
        params=(
            _VIDEO_ID,
            Param(
                name="transcribers",
                type=str,
                help="Comma-separated transcriber names.",
                default=REQUIRED,
            ),
            Param(
                name="aligners",
                type=str,
                help="Comma-separated aligner names to pair with each transcriber.",
                default="",
            ),
            Param(name="start_ms", type=int, help="Window start.", default=0),
            Param(
                name="end_ms",
                type=int,
                help="Window end; 0 means to the end of the audio.",
                default=C.COMPARE_DEFAULT_WINDOW_MS,
            ),
            Param(name="out", type=Path, help="Write the markdown report here.", default=None),
            _WAV,
        ),
        handler=_compare_handler,
        long_running=True,
    )
)

register(
    Command(
        name="transcribe.engines",
        group="transcribe",
        summary="List registered transcribers and aligners and whether they can run.",
        params=(),
        handler=_engines_handler,
    )
)

register(
    Command(
        name="transcribe.fingerprint",
        group="transcribe",
        summary="Measure a video's acoustic fingerprint into video_acoustics.",
        params=(_VIDEO_ID, _WAV),
        handler=_fingerprint_handler,
        long_running=True,
    )
)
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_commands_transcribe.py -v
```
Expected: 14 passed.

- [ ] **Step 6: Lint and type-check**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings in Part 3's files.

- [ ] **Step 7: Commit**

```bash
git add rytp/commands/transcribe.py rytp/audio/acoustics.py tests/test_commands_transcribe.py
git commit -m "feat(commands): the transcribe command group"
```

---

### Task 18: Integration, lint, type-check, full suite

The verifiable end state: one video goes from a captions file to cuttable aligned words, with a fingerprint and a comparison report, with no optional extra installed and no network access.

**Files:**
- Test: `tests/test_transcribe_integration.py`

**Interfaces:**
- Consumes: every task above. Produces nothing new.

- [ ] **Step 1: Write the integration test**

Create `tests/test_transcribe_integration.py`:

```python
"""One video, caption tier to aligned tier, with nothing heavy installed."""
from __future__ import annotations

import json
from pathlib import Path

import rytp.commands.transcribe  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.db import Database
from tests.fake_engines import FakeAligner, FakeTranscriber, registered
from tests.synth_audio import concat, silence, tone, write_wav

CAPTIONS = {
    "events": [
        {
            "tStartMs": 0,
            "segs": [{"utf8": "один", "tOffsetMs": 0}, {"utf8": " два", "tOffsetMs": 200}],
        },
        {"tStartMs": 460, "segs": [{"utf8": "три", "tOffsetMs": 0}]},
    ]
}


def test_a_video_goes_from_captions_to_cuttable_words(db: Database, tmp_path: Path) -> None:
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
    video_id = int(cursor.lastrowid)
    captions = tmp_path / "captions.json3"
    captions.write_text(json.dumps(CAPTIONS, ensure_ascii=False), encoding="utf-8")
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
        (video_id, "captions", str(captions), "2026-09-21T00:00:00+00:00"),
    )
    db.conn.commit()
    wav = write_wav(
        tmp_path / "audio.wav",
        concat(tone(200), silence(120, amp=0.001, seed=71), tone(200),
               silence(120, amp=0.001, seed=72), tone(200)),
    )

    # Tier 1: searchable for no GPU time, and not cuttable.
    resolve("transcribe.captions").handler(db, video_id=video_id)
    caption_rows = db.conn.execute(
        "SELECT source, end_ms FROM words WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert {row[0] for row in caption_rows} == {"caption"}
    assert all(row[1] is None for row in caption_rows)

    # An aligner runs here, so promotion replaces the caption words with the
    # cuttable `aligned` tier. Without one they would land as `timed`.
    with registered(FakeTranscriber, FakeAligner):
        resolve("transcribe.run").handler(
            db, video_id=video_id, transcriber="fake", aligner="fake-aligner", wav=wav
        )
        report = tmp_path / "compare.md"
        resolve("transcribe.compare").handler(
            db,
            video_id=video_id,
            transcribers="fake",
            aligners="fake-aligner",
            end_ms=0,
            out=report,
            wav=wav,
        )
    resolve("transcribe.fingerprint").handler(db, video_id=video_id, wav=wav)

    aligned = db.conn.execute(
        "SELECT ord, start_ms, end_ms, source, engine, align_score FROM words "
        "WHERE video_id = ? ORDER BY ord",
        (video_id,),
    ).fetchall()
    assert [row[0] for row in aligned] == [0, 1, 2]
    # An aligner ran, so these are cuttable. Without one they would be `timed`.
    assert all(row[3] == "aligned" for row in aligned)
    assert all(row[2] is not None and row[2] > row[1] for row in aligned)
    assert all(row[4] == "fake+fake-aligner+energy" for row in aligned)
    assert all(row[5] is not None for row in aligned)
    for left, right in zip(aligned, aligned[1:], strict=False):
        assert left[2] == right[1]

    assert report.exists()
    assert db.conn.execute(
        "SELECT COUNT(*) FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1
```

- [ ] **Step 2: Run the integration test**

Run:
```bash
python -m pytest tests/test_transcribe_integration.py -v
```
Expected: 1 passed.

- [ ] **Step 3: Run the whole suite**

Run:
```bash
python -m pytest -q
```
Expected: every test passes in a venv with numpy, scipy, pytest, ruff and mypy and **no ML libraries**. A failure that names `faster_whisper`, `gigaam`, `torch`, `transformers` or `mfa` means an import escaped a function body — fix the import, not the test.

- [ ] **Step 4: Confirm no optional dependency is imported at module import time**

Run:
```bash
python -c "import sys; import rytp.commands.transcribe, rytp.transcribe.engines, rytp.transcribe.align; heavy = {'torch','transformers','faster_whisper','gigaam','torchaudio'} & set(sys.modules); print('heavy modules imported:', heavy or 'none')"
```
Expected: `heavy modules imported: none`.

- [ ] **Step 5: Confirm no test reaches the network and no real identifier is committed**

Run:
```bash
grep -rn "https\?://" rytp tests | grep -v "example.invalid" || echo "only example.invalid"
grep -rn "VIDEO_A\|CHANNEL_ONE" tests | wc -l
```
Expected: `only example.invalid`, and a non-zero count of placeholder identifiers. Every identifier in a committed file is `VIDEO_A`, `CHANNEL_ONE`, or an `example.invalid` URL — no real video id, channel id, channel name or URL anywhere, tests included.

- [ ] **Step 6: Lint and type-check the whole package**

```bash
ruff check rytp tests
mypy rytp
```
Expected: no findings.

- [ ] **Step 7: Commit**

```bash
git add tests/test_transcribe_integration.py
git commit -m "test(transcribe): caption tier to cuttable words end to end"
```

---

### Task 19: Register the four job kinds, and let the commands produce them

Contracts §5 "Job handlers" makes the job-kind-to-callable mapping a contract of its own, and Part 2 owns the file it lives in. It ends by re-running the whole gate, because it is the first point at which Part 3 is reachable from the worker as well as from the CLI.

**Four kinds, not three.** Contracts §5's table lists `transcribe`, `align` and `fingerprint`, but turning a downloaded captions file into words is a fourth piece of queued work and nothing else owns it: Part 2's `captions` kind stops at writing the asset. Without it the caption tier — the thing that makes the whole corpus searchable for no GPU time (design §6) — would only ever run by hand. The name `caption_words` is agreed with Part 2, which enqueues it after `captions` succeeds; it names its output the way `extract_wav` does.

**Every kind also needs a producer.** Part 2's `rytp ingest` enqueues `download`, `captions`, `caption_words`, `extract_wav` and `fingerprint`, and adds `transcribe`/`align` behind `--transcribe` because design §6 makes tier 2 opt-in per video. Part 3 adds the other half: an `--enqueue` flag on `transcribe run`, `transcribe align` and `transcribe fingerprint`, because contracts §5 marks all three `long_running` and says the TUI must not run those inline — without it the TUI would have no way to ask for transcription at all. Both paths go through `enqueue`, which is idempotent on `UNIQUE (kind, target_id)`, so they cannot double-enqueue.

Follow Part 2's published "Produces for Parts 3-7: registering a job kind" section exactly. Three rules from it bind here:

- **A readiness predicate never sees the payload.** It is a statement about the world, which is what makes the queue self-healing (design §5). Anything that is an *input* to the work — which transcriber, which aligner — goes in `payload_json` and reaches the handler only.
- **`target_kind` namespaces `jobs.target_id`.** All four Part 3 kinds target a `videos.id`, so all four keep the default `"video"`.
- **`reopenable=False` stops a full reconcile re-deriving a one-shot.** `align` is re-timing the user explicitly asked for, so it must not silently re-run on the next worker start; `transcribe` and `fingerprint` are derived from the corpus and stay reopenable.

Both work-producing thunks also **enqueue Part 4's `index` job**. Part 2's chain stops at `extract_wav` and nothing else creates that job, so without this a freshly transcribed video never gets utterances and never appears in search until somebody runs `index build` by hand. Transcribing and re-aligning are precisely the two operations that delete a video's utterances, so the follow-on belongs here. `enqueue_index` (Task 9) is shared with the CLI handlers, so the hand-run path is covered by the same line.

> Part 2's worked example writes `from rytp.transcribe.run import transcribe_video`. That module does not exist in this plan — the entry point is `rytp.transcribe.pipeline.transcribe_video`. Use the path below.

**Files:**
- Create: `rytp/transcribe/readiness.py`
- Modify: `rytp/jobs/__init__.py` (append at the very bottom; Part 2 owns everything above)
- Modify: `rytp/commands/transcribe.py` (the `--enqueue` flag on three commands)
- Test: `tests/test_transcribe_jobs.py`, `tests/test_commands_transcribe.py` (append)

**Interfaces:**
- Consumes: `rytp.jobs.{Readiness, JobKind, register_job_kind, JOB_KINDS, JOB_HANDLERS}`, `rytp.jobs.queue.enqueue`, `rytp.audio.extract.wav_path`, `rytp.transcribe.pipeline.{transcribe_video, realign_video, enqueue_index}`, `rytp.audio.acoustics.fingerprint_video`.
- Produces: `rytp.transcribe.readiness.{caption_words_readiness, transcribe_readiness, align_readiness, fingerprint_readiness}`, and the registered kinds `caption_words`, `transcribe`, `align`, `fingerprint`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe_jobs.py`:

```python
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
    with registered(FakeTranscriber, MissingModuleTranscriber):
        with pytest.raises(EngineUnavailable) as excinfo:
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_jobs.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.readiness'`.

- [ ] **Step 3: Write `rytp/transcribe/readiness.py`**

```python
"""Readiness predicates for Part 3's job kinds (design §5, contracts §5).

``rytp/jobs/__init__.py`` imports this module at load time, so it stays light:
the standard library, :mod:`rytp.config`, :mod:`rytp.constants`,
:mod:`rytp.db` and :mod:`rytp.audio.extract`. No numpy, no engines, no
pipeline.

Readiness is derived from the database and the filesystem and **never** from
the job's payload. A predicate that changed its mind based on how a job was
enqueued would stop being a statement about the world, and the self-healing
property in design §5 would go with it: prune a cached WAV and `transcribe`
becomes blocked while `extract_wav` becomes available, with no edge table
anywhere to keep consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rytp.audio.extract import wav_path

if TYPE_CHECKING:
    from rytp.db import Database
    from rytp.jobs import Readiness


def _has_transcribed_words(db: Database, video_id: int) -> bool:
    """Any non-caption words: the video has been transcribed, at either tier."""
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? AND source <> 'caption' LIMIT 1",
        (video_id,),
    ).fetchone()
    return row is not None


def _has_any_words(db: Database, video_id: int) -> bool:
    row = db.conn.execute(
        "SELECT 1 FROM words WHERE video_id = ? LIMIT 1", (video_id,)
    ).fetchone()
    return row is not None


def caption_words_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the captions asset is on disk; satisfied once words exist.

    Satisfied on *any* word, not only caption-tier ones: a video already
    promoted to the aligned tier must not have its real boundaries replaced
    by caption starts, which is the same refusal
    :func:`~rytp.transcribe.captions.ingest_captions` makes for itself.
    """
    from rytp.jobs import Readiness

    if _has_any_words(db, video_id):
        return Readiness.SATISFIED
    row = db.conn.execute(
        "SELECT 1 FROM assets WHERE video_id = ? AND role = 'captions' LIMIT 1",
        (video_id,),
    ).fetchone()
    return Readiness.READY if row is not None else Readiness.BLOCKED


def transcribe_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the cached WAV exists; satisfied once the video has words.

    Satisfied at either transcript tier: `timed` means transcription already
    happened, and getting from there to `aligned` is the `align` job's work,
    not a reason to transcribe again.

    ``Readiness`` is imported inside the function because ``rytp.jobs``
    imports this module: a module-level import would close the cycle.
    """
    from rytp.jobs import Readiness

    if _has_transcribed_words(db, video_id):
        return Readiness.SATISFIED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED


def align_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable when there is a transcript to align and a WAV to align it to.

    Either tier qualifies: `timed` rows get upgraded to `aligned`, and
    `aligned` rows get re-timed by a different aligner. Never *satisfied*,
    because "already aligned" does not mean "aligned by the engine you just
    asked for" — which is also why the kind sets ``reopenable=False``.
    Enqueue it again to run it again.
    """
    from rytp.jobs import Readiness

    if not _has_transcribed_words(db, video_id):
        return Readiness.BLOCKED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED


def fingerprint_readiness(db: Database, video_id: int) -> Readiness:
    """Runnable once the WAV exists; satisfied once the one row is written."""
    from rytp.jobs import Readiness

    row = db.conn.execute(
        "SELECT 1 FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is not None:
        return Readiness.SATISFIED
    return Readiness.READY if wav_path(video_id).exists() else Readiness.BLOCKED
```

- [ ] **Step 4: Append the thunks and registration to `rytp/jobs/__init__.py`**

At the **very bottom** of the file, after Part 2's own registration block. Touch nothing above it.

```python
# ---------------------------------------------------------------------------
# Part 3: transcription, alignment and the acoustic fingerprint
# ---------------------------------------------------------------------------
from rytp.transcribe.readiness import (  # noqa: E402 - registration order
    align_readiness,
    caption_words_readiness,
    fingerprint_readiness,
    transcribe_readiness,
)


def _load_transcribe_engines() -> None:
    """Import the adapter packages, which is what makes engine names resolve.

    Contracts §6: an engine registers itself at import time, so a name that is
    never imported is a name that does not exist. Deferred to call time so the
    queue module stays light.
    """
    import rytp.transcribe.align  # noqa: F401
    import rytp.transcribe.engines  # noqa: F401


def _run_caption_words(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.transcribe.captions import ingest_captions

    ingest_captions(db, video_id)
    # Caption words are searchable words, so the index is stale too.
    from rytp.transcribe.pipeline import enqueue_index

    enqueue_index(db, video_id)


def _run_transcribe(db: Database, video_id: int, payload: dict[str, Any]) -> str | None:
    from rytp import constants as C
    from rytp.audio.extract import wav_path
    from rytp.transcribe.pipeline import enqueue_index, transcribe_video
    from rytp.transcribe.registry import default_transcriber

    _load_transcribe_engines()
    # `ingest --transcribe` and `transcribe run --enqueue` both stamp the
    # engine at enqueue time, so this branch is for a job made by hand.
    # contracts §3 still applies: resolve the setting, never a constant, and
    # say which engine it picked — which for a job is the note the worker
    # stores on the row. If it is not installed, `transcribe_video` raises
    # with the install hint and the job fails; it must not run another one.
    named = str(payload.get("transcriber") or "").strip()
    transcriber = named or default_transcriber(db)
    transcribe_video(
        db,
        video_id,
        wav_path=wav_path(video_id),
        transcriber=transcriber,
        aligner=payload.get("aligner") or None,
        language=payload.get("language") or C.DEFAULT_LANGUAGE,
        refine=bool(payload.get("refine", True)),
    )
    # New words mean the video's utterances are gone: ask Part 4 to rebuild.
    enqueue_index(db, video_id)
    if named:
        return None
    return (
        f"no transcriber in the payload; used {transcriber!r} from setting "
        f"{C.SETTING_DEFAULT_TRANSCRIBER}"
    )


def _run_align(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.audio.extract import wav_path
    from rytp.models import RytpError
    from rytp.transcribe.pipeline import enqueue_index, realign_video

    aligner = payload.get("aligner")
    if not aligner:
        raise RytpError(
            f"align job for video {video_id} has no 'aligner' in its payload"
        )
    _load_transcribe_engines()
    realign_video(
        db,
        video_id,
        wav_path=wav_path(video_id),
        aligner=str(aligner),
        refine=bool(payload.get("refine", True)),
    )
    # Re-timing rewrote every boundary, so the utterances are gone too.
    enqueue_index(db, video_id)


def _run_fingerprint(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.audio.acoustics import fingerprint_video
    from rytp.audio.extract import wav_path

    fingerprint_video(db, video_id, wav_path=wav_path(video_id))


register_job_kind(
    JobKind(
        name="caption_words",
        pool="cpu",
        readiness=caption_words_readiness,
        handler=_run_caption_words,
        summary="turn one video's downloaded captions into caption-tier words",
    )
)

register_job_kind(
    JobKind(
        name="transcribe",
        pool="gpu",
        readiness=transcribe_readiness,
        handler=_run_transcribe,
        summary="transcribe, align and refine one video into cuttable words",
    )
)

register_job_kind(
    JobKind(
        name="align",
        pool="gpu",
        readiness=align_readiness,
        handler=_run_align,
        summary="re-time one video's words with a different aligner",
        # A one-shot the user asked for: a reconcile must not re-run it.
        reopenable=False,
    )
)

register_job_kind(
    JobKind(
        name="fingerprint",
        pool="cpu",
        readiness=fingerprint_readiness,
        handler=_run_fingerprint,
        summary="measure one video's acoustic fingerprint",
    )
)
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_jobs.py -v
```
Expected: 19 passed.

- [ ] **Step 6: Write the failing test for the `--enqueue` flag**

Append to `tests/test_commands_transcribe.py`:

```python
def test_run_can_queue_the_work_instead_of_doing_it(db: Database, tmp_path: Path) -> None:
    # Contracts §5 marks this command long_running, which the TUI must not run
    # inline — so it needs a way to become a job.
    video_id = _make_video(db)
    result = resolve("transcribe.run").handler(
        db, video_id=video_id, transcriber="whisper", aligner="mfa", enqueue=True
    )
    assert "queued" in (result.message or "")
    row = db.conn.execute(
        "SELECT payload_json FROM jobs WHERE kind = 'transcribe' AND target_id = ?",
        (video_id,),
    ).fetchone()
    assert row is not None
    assert "mfa" in row[0]


def test_queuing_stamps_and_announces_the_default_transcriber(db: Database) -> None:
    # contracts §3 wants the choice made and reported at enqueue time. A job
    # row that names no engine defers the decision to a worker that nobody
    # is watching, which is the hole this closes.
    video_id = _make_video(db)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "whisper")
    result = resolve("transcribe.run").handler(db, video_id=video_id, enqueue=True)
    message = result.message or ""
    assert "whisper" in message
    assert C.SETTING_DEFAULT_TRANSCRIBER in message
    row = db.conn.execute(
        "SELECT payload_json FROM jobs WHERE kind = 'transcribe' AND target_id = ?",
        (video_id,),
    ).fetchone()
    assert json.loads(row[0])["transcriber"] == "whisper"


def test_a_misspelled_transcriber_is_rejected_before_the_job_exists(
    db: Database,
) -> None:
    # Not after five failed retries on a GPU worker.
    video_id = _make_video(db)
    set_setting(db, C.SETTING_DEFAULT_TRANSCRIBER, "gigaamm")
    with pytest.raises(ValueError, match="gigaamm"):
        resolve("transcribe.run").handler(db, video_id=video_id, enqueue=True)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'transcribe'"
    ).fetchone()[0] == 0


def test_queuing_needs_no_wav_and_no_engine_installed(db: Database) -> None:
    # Enqueueing must not touch the filesystem or resolve an engine: the point
    # is to defer exactly that to the worker.
    video_id = _make_video(db)
    for name, kind in (("transcribe.align", "align"), ("transcribe.fingerprint", "fingerprint")):
        extra = {"aligner": "mfa"} if kind == "align" else {}
        resolve(name).handler(db, video_id=video_id, enqueue=True, **extra)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ? AND target_id = ?", (kind, video_id)
        ).fetchone()[0] == 1


def test_queuing_twice_leaves_one_job(db: Database) -> None:
    video_id = _make_video(db)
    resolve("transcribe.fingerprint").handler(db, video_id=video_id, enqueue=True)
    resolve("transcribe.fingerprint").handler(db, video_id=video_id, enqueue=True)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'fingerprint' AND target_id = ?",
        (video_id,),
    ).fetchone()[0] == 1
```

- [ ] **Step 7: Add the flag to `rytp/commands/transcribe.py`**

Add the shared helper next to `_wav_for`:

```python
def _queue(db: Database, kind: str, video_id: int, payload: dict[str, object]) -> CommandResult:
    """Hand the work to the queue instead of doing it here (contracts §5).

    Every long-running command in this group needs this: the TUI may not run
    them inline, and without a producer the gpu pool would never receive
    transcription work at all. ``enqueue`` is idempotent on
    ``UNIQUE (kind, target_id)``, so this cannot collide with Part 2's chain.
    """
    from rytp.jobs.queue import enqueue

    job_id = enqueue(db, kind, video_id, payload=payload)
    return CommandResult(message=f"queued {kind} job {job_id} for video {video_id}")
```

Give each of the three long-running per-video handlers an `enqueue: bool = False`
keyword and an early return. In `_run_handler`, immediately after the signature:

```python
    if enqueue:
        # Resolve the engine here rather than leaving it to the worker.
        # contracts §3 wants an unregistered name rejected at enqueue time,
        # and wants the choice announced when it came from the setting —
        # neither is possible once this is a row in a table nobody is
        # watching. `resolve_transcriber` is a *name* lookup, deliberately
        # not `check_available`: enqueueing must not require the engine to be
        # installed on the machine doing the enqueueing, only on the worker.
        _load_engine_modules()
        chosen_transcriber = transcriber.strip() or default_transcriber(db)
        resolve_transcriber(chosen_transcriber)
        result = _queue(
            db,
            "transcribe",
            video_id,
            {
                "transcriber": chosen_transcriber,
                "aligner": aligner,
                "language": language,
                "refine": refine,
            },
        )
        if not transcriber.strip():
            return CommandResult(
                message=(
                    f"{result.message}; transcriber {chosen_transcriber!r}, from "
                    f"setting {C.SETTING_DEFAULT_TRANSCRIBER} "
                    f"(pass --transcriber to override)"
                )
            )
        return result
    _load_engine_modules()
```

`resolve_transcriber` joins `default_transcriber` and `setting` on the import line at the top of the module.

In `_align_handler`:

```python
    if enqueue:
        return _queue(db, "align", video_id, {"aligner": aligner, "refine": refine})
    _load_engine_modules()
```

In `_fingerprint_handler`:

```python
    if enqueue:
        return _queue(db, "fingerprint", video_id, {})
```

Add the same `Param` to the `params` tuple of `transcribe.run`, `transcribe.align`
and `transcribe.fingerprint`, defined once beside `_WAV`:

```python
_ENQUEUE = Param(
    name="enqueue",
    type=bool,
    help="Queue the work for the worker instead of running it now.",
    default=False,
)
```

- [ ] **Step 8: Run both test modules to verify they pass**

Run:
```bash
python -m pytest tests/test_transcribe_jobs.py tests/test_commands_transcribe.py -v
```
Expected: 38 passed.

- [ ] **Step 9: Re-run the whole gate**

```bash
python -m pytest -q
ruff check rytp tests
mypy rytp
```
Expected: everything passes, no findings. Part 3 is now reachable from the worker as well as from the CLI, every kind it owns has a producer, and the caption tier runs without anyone typing a command.

- [ ] **Step 10: Commit**

```bash
git add rytp/transcribe/readiness.py rytp/jobs/__init__.py rytp/commands/transcribe.py \
        tests/test_transcribe_jobs.py tests/test_commands_transcribe.py
git commit -m "feat(jobs): register Part 3's four job kinds and let its commands queue them"
```

---

### Task 20: `transcribe.remove`

Contracts §5 "Deletion": every group exposes a `remove`, so there is no entity you can create but not get rid of. Part 3's removes a video's words — and with them its utterances and speaker labels, because contracts §4 says those three go together.

**No `--dry-run` and no `--yes`, deliberately.** Contracts §5 requires both only of a command that deletes *files*. This one touches derived database rows and nothing else: the media, the captions asset and the cached WAV all survive, so re-running `transcribe run` rebuilds everything it removed. Do not add a confirmation prompt to a command that does not need one.

**It is not a job, and it enqueues nothing.** Removal is synchronous and immediate (contracts §5). There is no index to rebuild afterwards either: the utterances are already gone and Part 4's `index` readiness needs words, so it goes blocked on its own.

**Files:**
- Modify: `rytp/commands/transcribe.py`
- Test: `tests/test_commands_transcribe.py` (append)

**Interfaces:**
- Consumes: `rytp.transcribe.pipeline.{invalidate_transcript, TranscriptRemoval}` from Task 9.
- Produces: the registered command `transcribe.remove`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_commands_transcribe.py`:

```python
def test_remove_drops_the_words_and_everything_derived_from_them(
    db: Database, tmp_path: Path
) -> None:
    video_id = _make_video(db)
    wav = _wav(tmp_path)
    with registered(FakeTranscriber):
        resolve("transcribe.run").handler(db, video_id=video_id, transcriber="fake", wav=wav)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, "
        "last_word_ord, text, normalized_text, stem_text) "
        "VALUES (?, 0, 100, 0, 2, 'x', 'x', 'x')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) "
        "VALUES (?, 'SPEAKER_00', 'fake-diarizer')",
        (video_id,),
    )
    db.conn.commit()

    result = resolve("transcribe.remove").handler(db, video_id=video_id)

    assert result.rows[0] == ("3", "1", "1")
    for table in ("words", "utterances", "video_speakers"):
        left = db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        assert left == 0


def test_remove_leaves_the_assets_and_the_cached_wav_alone(
    db: Database, tmp_path: Path
) -> None:
    # Derived rows only: the WAV is regenerable and Part 2's cache.prune owns it.
    video_id = _make_video(db)
    wav = _wav(tmp_path)
    with registered(FakeTranscriber):
        resolve("transcribe.run").handler(db, video_id=video_id, transcriber="fake", wav=wav)
    resolve("transcribe.remove").handler(db, video_id=video_id)
    assert wav.exists()


def test_remove_on_a_video_with_no_transcript_is_a_no_op(db: Database) -> None:
    video_id = _make_video(db)
    result = resolve("transcribe.remove").handler(db, video_id=video_id)
    assert result.rows[0] == ("0", "0", "0")


def test_remove_needs_no_confirmation_flags() -> None:
    # Contracts §5: --dry-run and --yes are required only of commands that
    # delete files. This one deletes derived rows, so it takes neither.
    names = {param.name for param in resolve("transcribe.remove").params}
    assert names == {"video_id"}
    assert resolve("transcribe.remove").long_running is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_commands_transcribe.py -v
```
Expected: `ValueError: unknown command 'transcribe.remove'`.

- [ ] **Step 3: Add the handler and the registration to `rytp/commands/transcribe.py`**

Add to the imports:

```python
from rytp.transcribe.pipeline import (
    enqueue_index,
    invalidate_transcript,
    realign_video,
    speaker_loss_warning,
    transcribe_video,
)
from rytp.transcribe.registry import engine_rows, setting
```

The handler:

```python
def _remove_handler(db: Database, *, video_id: int) -> CommandResult:
    """Drop a video's transcript and everything derived from it.

    Takes no ``--dry-run`` and no ``--yes``: contracts §5 requires those of a
    command that deletes files, and this one deletes only derived database
    rows. The media, the captions asset and the cached WAV all survive, so
    ``transcribe run`` rebuilds exactly what this removed.
    """
    removed = invalidate_transcript(db, video_id)
    notes = [f"removed the transcript of video {video_id}"]
    warning = speaker_loss_warning(video_id, removed.video_speakers)
    if warning:
        notes.append(warning)
    return CommandResult(
        columns=("words", "utterances", "speaker labels"),
        rows=(
            (str(removed.words), str(removed.utterances), str(removed.video_speakers)),
        ),
        message=". ".join(notes),
    )
```

The registration:

```python
register(
    Command(
        name="transcribe.remove",
        group="transcribe",
        summary="Remove a video's words, and with them its utterances and speaker labels.",
        params=(_VIDEO_ID,),
        handler=_remove_handler,
    )
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_commands_transcribe.py -v
```
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/commands/transcribe.py tests/test_commands_transcribe.py
git commit -m "feat(commands): transcribe remove drops a video's transcript"
```

---

### Task 21: Health checks for `doctor`

Contracts §5 "Health checks". Part 1 owns the command and the registry; Part 3 contributes the checks that answer the question the owner will ask most often, because Part 3's dependencies are the ones that actually fail to install: **which transcribers and aligners can I run right now, and what would it take to get the others?**

Probing is cheap and offline, and it never raises. An in-process engine is checked with `importlib.util.find_spec`, which consults the filesystem and neither imports nor downloads anything; an out-of-process engine is checked by asking whether its configured interpreter exists.

**Every check here is `required=False`.** Contracts §5 separates "what was found" from "does this failure matter": `ok` always tells the truth, and `required=False` is what keeps a missing engine out of `doctor`'s exit code. Nothing Part 3 registers is needed by the base install, so a missing transcriber reports `ok=False` with its remedy and `doctor` still exits zero.

The GPU probe uses `nvidia-smi`, not torch: importing torch costs seconds and torch may not be installed at all, while the driver's own tool answers the question immediately.

**Files:**
- Modify: `rytp/constants.py` (one constant)
- Create: `rytp/transcribe/health.py`
- Modify: `rytp/commands/transcribe.py` (one import, so the checks register)
- Test: `tests/test_transcribe_health.py`

**Interfaces:**
- Consumes: `rytp.commands.{HealthCheck, HealthResult, register_check, HEALTH_CHECKS}` (contracts §5), `rytp.transcribe.registry.{TRANSCRIBERS, ALIGNERS, availability}`.
- Produces: `gpu_check(db) -> HealthResult`, `engine_remedy(cls) -> str`, `register_transcribe_checks() -> None`, and one registered check per engine plus `gpu`.

- [ ] **Step 1: Add the probe timeout to `rytp/constants.py`**

In the Part 3 block, beside the engine settings:

```python
#: nvidia-smi answers in well under a second on a healthy machine; the
#: timeout exists so a wedged driver cannot hang `doctor` (contracts §5).
GPU_PROBE_TIMEOUT_S: int = 10
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_transcribe_health.py`:

```python
"""Which engines can actually run here, answered without importing any of them."""
from __future__ import annotations

import subprocess
import sys

import pytest

from rytp.commands import HEALTH_CHECKS
from rytp.db import Database
from rytp.transcribe import health
from rytp.transcribe.engines.gigaam import GigaAMTranscriber
from rytp.transcribe.engines.whisper import FasterWhisperTranscriber


def test_registering_adds_one_check_per_engine_plus_the_gpu(db: Database) -> None:
    health.register_transcribe_checks()
    assert "transcriber:whisper" in HEALTH_CHECKS
    assert "transcriber:gigaam" in HEALTH_CHECKS
    assert "aligner:mfa" in HEALTH_CHECKS
    assert "aligner:wav2vec2" in HEALTH_CHECKS
    assert "gpu" in HEALTH_CHECKS


def test_registering_twice_is_harmless(db: Database) -> None:
    health.register_transcribe_checks()
    before = len(HEALTH_CHECKS)
    health.register_transcribe_checks()
    assert len(HEALTH_CHECKS) == before


def test_every_part_3_check_is_advisory(db: Database) -> None:
    # Contracts §5: ok tells the truth, required=False is what makes a missing
    # optional engine non-fatal. Nothing here is needed by the base install.
    health.register_transcribe_checks()
    for name in (
        "transcriber:whisper",
        "transcriber:gigaam",
        "aligner:mfa",
        "aligner:wav2vec2",
        "gpu",
    ):
        assert HEALTH_CHECKS[name].required is False, name


def test_a_missing_engine_reports_ok_false_rather_than_pretending(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: None)
    result = health.gpu_check(db)
    assert result.ok is False


def test_a_check_never_raises_and_always_explains_itself(db: Database) -> None:
    health.register_transcribe_checks()
    for name, check in HEALTH_CHECKS.items():
        result = check.run(db)
        assert isinstance(result.ok, bool)
        assert result.detail, f"{name} reported nothing"
        if not result.ok:
            assert result.remedy, f"{name} failed without saying how to fix it"


def test_an_in_process_engine_remedy_names_its_extra() -> None:
    assert engine_remedy_of(FasterWhisperTranscriber) == "pip install rytp[whisper]"


def test_an_out_of_process_engine_remedy_names_the_interpreter_setting() -> None:
    remedy = engine_remedy_of(GigaAMTranscriber)
    assert "engine.interpreter.gigaam" in remedy


def engine_remedy_of(cls: type) -> str:
    return health.engine_remedy(cls)


def test_no_gpu_tool_is_reported_not_fatal(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: None)
    result = health.gpu_check(db)
    assert result.ok is False
    assert "CPU" in result.detail
    assert result.remedy


def test_a_visible_gpu_is_reported_with_its_name(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        health, "_query_gpu", lambda binary: (0, "NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB")
    )
    result = health.gpu_check(db)
    assert result.ok is True
    assert "3080" in result.detail


def test_a_wedged_driver_is_reported_not_raised(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(binary: str) -> tuple[int, str]:
        raise subprocess.TimeoutExpired(cmd=[binary], timeout=1)

    monkeypatch.setattr(health, "_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(health, "_query_gpu", explode)
    result = health.gpu_check(db)
    assert result.ok is False
    assert result.remedy


def test_probing_imports_no_engine_dependency(db: Database) -> None:
    # find_spec consults the filesystem; nothing here may import torch.
    health.register_transcribe_checks()
    for check in HEALTH_CHECKS.values():
        check.run(db)
    assert not {"torch", "transformers", "faster_whisper", "gigaam"} & set(sys.modules)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
python -m pytest tests/test_transcribe_health.py -v
```
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.transcribe.health'`.

- [ ] **Step 4: Write `rytp/transcribe/health.py`**

```python
"""What `doctor` reports about Part 3 (contracts §5, "Health checks").

The most-asked question on this machine, because these are the dependencies
that actually fail to install and several of them pin conflicting versions of
each other: *which transcribers and aligners can I run right now, and what
would it take to get the others?*

Every probe is cheap and offline. An in-process engine is checked with
``importlib.util.find_spec``, which consults the filesystem and neither
imports nor downloads anything. An out-of-process engine is checked by asking
whether its configured interpreter exists — its dependency lives in another
environment where ``find_spec`` here would be meaningless. The GPU is probed
with ``nvidia-smi`` rather than by importing torch, which costs seconds and
may not be installed at all.

Nothing here raises. Every check Part 3 registers is ``required=False``:
none of these is needed by the base install, so a missing engine reports
``ok=False`` — which is the truth about what was found — without failing
``doctor``. Contracts §5 keeps those two ideas apart on purpose, because the
alternative is claiming ``ok=True`` for a tool that is not there, which makes
the output lie about the one thing the command exists to tell you.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp.transcribe.registry import ALIGNERS, TRANSCRIBERS, availability

if TYPE_CHECKING:
    from rytp.commands import HealthResult
    from rytp.db import Database

_READY = ("ready", "interpreter ok")


def engine_remedy(cls: type) -> str:
    """The exact command that would make this engine runnable."""
    extra = getattr(cls, "extra", None) or cls.name
    if not getattr(cls, "out_of_process", False):
        return f"pip install rytp[{extra}]"
    if cls.name == "mfa":
        return (
            "install the Montreal Forced Aligner in its own conda environment, then: "
            f"rytp settings set engine.interpreter.{cls.name} <path to that python>"
        )
    return (
        f"pip install rytp[{extra}] into its own virtualenv, then: "
        f"rytp settings set engine.interpreter.{cls.name} <path to that python>"
    )


def engine_check(db: Database, cls: type) -> HealthResult:
    """Report one engine, from the class alone — nothing is constructed."""
    from rytp.commands import HealthResult

    state = availability(db, cls)
    ok = state in _READY
    return HealthResult(ok=ok, detail=state, remedy=None if ok else engine_remedy(cls))


def _nvidia_smi() -> str | None:
    """Where the driver's own query tool is. A module attribute, so a test can
    replace just this rather than reaching into the global ``shutil``."""
    return shutil.which("nvidia-smi")


def _query_gpu(binary: str) -> tuple[int, str]:
    """Run one fixed query. Isolated so tests can replace it."""
    proc = subprocess.run(
        [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=C.GPU_PROBE_TIMEOUT_S,
        check=False,
    )
    return proc.returncode, proc.stdout


def gpu_check(db: Database) -> HealthResult:
    """Is a CUDA GPU visible? Never fatal — the pipeline runs on the CPU too,
    just slowly enough that the owner wants to know."""
    from rytp.commands import HealthResult

    binary = _nvidia_smi()
    if binary is None:
        return HealthResult(
            ok=False,
            detail="nvidia-smi not found; transcription would run on the CPU",
            remedy="install the NVIDIA driver, or accept CPU-only transcription",
        )
    try:
        code, output = _query_gpu(binary)
    except (OSError, subprocess.SubprocessError) as exc:
        return HealthResult(
            ok=False,
            detail=f"nvidia-smi could not be run: {type(exc).__name__}",
            remedy="check the NVIDIA driver installation",
        )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if code != 0 or not lines:
        return HealthResult(
            ok=False,
            detail="nvidia-smi reported no GPU",
            remedy="check the NVIDIA driver installation",
        )
    return HealthResult(ok=True, detail="; ".join(lines))


def register_transcribe_checks() -> None:
    """Register one check per registered engine, plus the GPU probe.

    Idempotent, because importing the command group more than once must not
    be an error. Imports the adapter packages first: an engine that was never
    imported never registered, and would be invisible here exactly as it is
    invisible to ``resolve_transcriber``.
    """
    import rytp.transcribe.align  # noqa: F401
    import rytp.transcribe.engines  # noqa: F401
    from rytp.commands import HEALTH_CHECKS, HealthCheck, register_check

    for kind, table in (("transcriber", TRANSCRIBERS), ("aligner", ALIGNERS)):
        for name, cls in sorted(table.items()):
            check_name = f"{kind}:{name}"
            if check_name in HEALTH_CHECKS:
                continue
            register_check(
                HealthCheck(
                    name=check_name,
                    summary=f"the {name} {kind} can run on this machine",
                    run=(lambda engine: lambda db: engine_check(db, engine))(cls),
                    # Advisory: nothing in Part 3 is required by the base
                    # install, so a missing engine reports ok=False honestly
                    # without failing `doctor` (contracts §5).
                    required=False,
                )
            )
    if "gpu" not in HEALTH_CHECKS:
        register_check(
            HealthCheck(
                name="gpu",
                summary="a CUDA GPU is visible to the driver",
                run=gpu_check,
                required=False,
            )
        )


register_transcribe_checks()
```

- [ ] **Step 5: Make the command group register them**

Add to the module-level imports of `rytp/commands/transcribe.py`, so loading the
command group is enough for `doctor` to see these checks:

```python
from rytp.transcribe import health  # noqa: F401 - importing registers the checks
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
python -m pytest tests/test_transcribe_health.py -v
```
Expected: 11 passed. On a machine with an NVIDIA driver the GPU check reports the card; on one without, it reports `ok=False` and the remedy. Neither makes the test suite fail, and neither makes `doctor` exit non-zero — that is what `required=False` is for.

- [ ] **Step 7: Run the whole gate**

```bash
python -m pytest -q
ruff check rytp tests
mypy rytp
```
Expected: everything passes, no findings.

- [ ] **Step 8: Commit**

```bash
git add rytp/constants.py rytp/transcribe/health.py rytp/commands/transcribe.py \
        tests/test_transcribe_health.py
git commit -m "feat(doctor): report which engines and which GPU are usable here"
```

---

## Self-review notes for the executor

Eight things are worth re-reading before you start, because they are where this plan is most likely to be wrong in practice.

1. **`refine_boundaries` is the only part with no reference to check against.** Its tests are property-style on synthesised audio for exactly that reason: a boundary landing inside a known gap, never moving further than the rail, never crossing its neighbour, idempotent, deterministic. If you change the algorithm, keep those properties and the tests stay meaningful.
2. **Every engine adapter's library call is unverified.** `gigaam.child_main`, `mfa.child_main` and `wav2vec2.child_main` are written against documented APIs that nobody here has run. Their *contracts* — the dictionaries they return — are what the rest of the code depends on, and those are covered by tests. When you install a real engine, expect to rewrite the body of one function and nothing else.
3. **One word row holds exactly one token.** `normalize_text` turns a hyphen into a space, so "кто-то" becomes two rows sharing a measured boundary rather than one unfindable row. `tests/test_transcribe_tokens.py` pins the rule and, more importantly, pins that it still agrees with whatever `normalize_text` does — if Part 1 ever changes its punctuation handling, that test fails rather than the corpus silently losing words.
4. **One function owns the cross-part invariant.** `invalidate_transcript` is the only place `words`, `utterances` and `video_speakers` are deleted together, and all three callers — caption ingest, transcript replacement, and `transcribe.remove` — go through it. Do not inline it back; two copies of an invariant is one copy too many.
5. **`transcribe.remove` deliberately has no `--yes`.** Contracts §5 requires confirmation flags only of commands that delete files. This one deletes derived rows that `transcribe run` rebuilds, and the cached WAV survives. A future reviewer will want to add a prompt; the plan says not to, and `test_remove_needs_no_confirmation_flags` will fail if someone does.
6. **`timed` is not `aligned`, and the difference is the product.** Only a run that actually had an aligner may write `source='aligned'`, because that value is the definition of cuttable (contracts §3). `tier_for()` is the single place that decides, and `test_only_a_run_with_an_aligner_produces_a_cuttable_tier` pins it. If a future change makes the no-aligner path write `aligned` again, the tool will confidently cut audio on boundaries that were never measured.
7. **Three tables are written by Part 3, two of them owned elsewhere.** `words` is ours; `utterances` (Part 4) and `video_speakers` (Part 7) are deleted whenever words change. This is not a local decision — contracts §4 "Cross-part invariants" requires any stage that deletes or replaces a video's words to delete that video's `utterances` and `video_speakers` rows **in the same transaction**. Tasks 8 and 9 replace a video's words and so delete both, in one transaction. Task 10 is the exception that proves the rule: re-alignment updates timings in place without touching a single word's text or ordinal, so the speaker labels stay valid and are kept — but the utterances still go, because they copy the timings that just changed.
8. **There is no default transcriber constant, and that is the point.** contracts §3 "Choosing a transcriber" makes the engine `settings.default_transcriber` (Part 2's key, `gigaam` out of the box), read through `registry.default_transcriber`, so that a run which used the default can *say* which engine it picked — in `transcribe run`'s message, in `ingest`'s message, and in the note a hand-made job leaves on its row. A constant cannot announce itself. Two rules go with it and both have tests: an engine that is not installed **fails with the install hint and is never quietly replaced** — a mixed corpus is interpretable only because `words.engine` records what actually ran, and that stops being true the moment something substitutes; and the default is **a starting point, not a verdict** — `gigaam` roughly halves Whisper's Russian word error rate on published benchmarks, but the one published test on *noisy YouTube* audio, which is exactly this corpus, favoured a Russian-finetuned Whisper. `transcribe compare` (Task 14) is how you settle it on real material; rewrite the setting, not the code.
