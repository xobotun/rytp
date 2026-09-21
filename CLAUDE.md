# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is, and how much of the docs to believe

The only human-authored parts of the documentation are the **Goal** and **Hardware** sections of `README.md`. The rest of `README.md`, all of `DESIGN.md`, and most module docstrings are AI-generated. They describe an aspirational system and routinely overstate what works — `README.md` claims "All 21 subcommands are wired in v1" and quotes a test count, `cli.py`'s docstring says the same. **Treat every completeness claim as unverified.** Read the code, or run the thing, before relying on any of it.

The actual goal, in the owner's words: a command-line utility that maps videos to *speaker + word* and supports full-text search, with the index running **both ways**:

- **Forward — `video → transcript`:** every word with its timestamp and the speaker who said it.
- **Backward — `speaker + word/sentence → {video + timestamps}`:** given a word or phrase (optionally constrained to a speaker), find every place it was said.

Downstream of the backward index, clips get spliced back together into a new video. In the schema, the forward direction is `words` ordered by `words(video_id, start_ms)`; the backward direction is the `words_fts` FTS5 shadow plus `words(speaker_id)`. Those two access paths are the point of the project — changes that make either one harder to serve are going the wrong way.

**Current state: barely working, most features incomplete.** Expect stubs, half-wired paths, and behavior that contradicts the docs. One gap worth knowing up front, because it sits directly on the stated goal: **the mine stage has no speaker dimension at all.** `rytp/mine.py` never references `speaker_id` and `rytp mine` takes no speaker flag, so the backward index currently answers "word → {video + timestamps}" but not "*speaker* + word → …".

The pipeline is meant to run on the owner's Windows machine (32 GiB RAM, RTX 3080 Laptop 16 GiB, i7-11800H) — a local GPU is available for STT and diarization, so local models are the intended path, not cloud APIs.

Feel free to overwrite any files prior and including commit `44fc214c`.

## A rewrite is planned and specified — read it before touching `rytp/`

Everything described under "Architecture" and "Gotchas" below is the **old**
implementation. It is being replaced wholesale. Three documents supersede
`DESIGN.md` and are the real source of truth:

| Document | What it is |
|---|---|
| `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` | The approved design. Start here. |
| `docs/superpowers/specs/2026-09-21-rytp-contracts.md` | **Binding** shared interfaces — schema DDL, core types, command registry, engine protocols, job handlers, cross-part invariants. Never vary from it; raise a change instead. |
| `docs/superpowers/plans/2026-09-21-part*.md` | Seven implementation plans, 98 tasks, 652 TDD steps. |
| `docs/superpowers/2026-09-21-review-findings.md` | **Read before implementing.** Open defects from an independent review — two that break day one, one that would silently produce bad output. None are fixed. |

The plans are sequenced: 1 foundation, 2 acquisition, 3 transcription,
4 index and search, 5 assembly, 6 render, 7 speakers. Parts 1 and 4 were
verified by extracting every code block and running it; the rest were not.
A backup mirror of the plans lives at `~/.cache/rytp-plans/` because this
repo is on an SMB share that has dropped mid-session.

Four things the redesign changes that will trip you up if you skim:

- **Words are split on punctuation.** `кто-то` is two rows. A stored word row
  holds exactly one token, and tokens equal `normalize_text(text).split()` —
  because a search query runs through the same function.
- **Captions are a first-class transcript tier**, searchable but never
  cuttable, so the whole corpus is searchable without GPU time.
- **Audio and video are separate assets**, never merged, so a rendition can be
  upgraded later without invalidating a transcript.
- **Whisper's word timestamps cannot be cut on.** 78.7% of gaps in the real
  data are exactly zero. Boundaries come from forced alignment, then snap to a
  measured energy minimum.

Nothing below this section has been rewritten yet, so treat it as a
description of the code as it stands, not as guidance for new work.

## Setup

No virtualenv is checked in and no dependencies are installed (`pytest`, `ruff`, `mypy` are not on `PATH`). Requires Python >= 3.11 and `ffmpeg` on `PATH`.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"        # pytest, pytest-cov, ruff, mypy
pip install -e ".[all]"        # yt-dlp, faster-whisper, pyannote-audio, torch
```

Everything heavy is an optional extra (`yt-dlp`, `stt`, `pyannote`, `all`); the base install runs the CRUD subcommands and the test suite. Commands that need a missing extra print a one-line install hint instead of a traceback.

## Commands

```bash
pytest -q                                  # full suite
pytest tests/test_mine.py -q               # one file
pytest tests/test_mine.py::test_name -q    # one test
pytest -k "splice and not loudnorm" -q     # by expression
ruff check rytp tests                      # lint (line-length 100, py311 target)
mypy rytp                                  # type check
python -m rytp <subcommand> --help         # run the CLI from a source checkout
```

`python -m rytp` works without installing; the `rytp` console script appears after `pip install`. `rytp/__main__.py` exists so a bare `python -m rytp` prints help and exits 0 (Typer alone would exit 2).

Two env vars matter: `RYTP_DATA` relocates the whole data tree (default `./data`), and `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) is required only by the pyannote diarizer.

## Architecture

`DESIGN.md` is worth skimming as a map, because nearly every module docstring cites it by section (`DESIGN §7`) — but see the caveat above, and note the concrete drift: it lists a `transcribe/combined_base.py` that doesn't exist, and names the mapper `rytp speakers <video-id>` when the real command is `rytp speakers map <video-id>`.

**Stages communicate only through SQLite.** No stage hands data to another in memory; each reads and writes `data/rytp.db` plus the `data/` tree. This is what makes the queue, pause/resume, and resumable transcription possible. The pipeline is `channel sync → queue/download → transcribe (extract + STT + diarize) → speakers map → mine → loudnorm → splice`, one CLI subcommand per stage.

**`data/audio/{video_id}.wav` (16 kHz mono int16) is the single source of audio truth.** Extracted exactly once per video; STT, spectral features, loudnorm, and `splice --mode concat` all read it. `splice --mode stream-copy` is the only stage that re-reads the original media under `data/media/`. When a video was registered with a separate audio file (`downloaded_audio_path`, set by `+`-format yt-dlp downloads and by `videos add --audio`), extraction reads that file instead of decoding the video.

**Speaker identity is two-level, and denormalized.** The diarizer emits raw per-video labels (`SPEAKER_00`); `videos_speaker_map` maps those to rows in the global `speakers` roster, and that mapping is a human step (`rytp speakers map`). `map_diarizer_to_speaker` writes the mapping *and* back-fills `words.speaker_id` for every matching row (`speakers.py:238`), so both the join and the direct column are valid — but `words.speaker_id` is `NULL` until someone maps the video, and re-running `transcribe` wipes it along with the rest of the words. `transcripts.py` reads speakers via the join; `speakers.py`'s pause stats read the column.

**Engine registry** (`rytp/engines.py`). STT engines, diarizers, and combined STT+diarize engines are `Protocol`s resolved by name through three dicts. Each concrete engine calls `engines.register_stt(cls)` / `register_diarizer(cls)` at the bottom of its own module, and the subpackage `__init__.py` (`rytp/diarize/__init__.py`, `rytp/transcribe/__init__.py`) imports it — **that import is what makes the name resolvable**, so a new engine needs all three pieces. The registry stores *classes, not instances*, because the HF_TOKEN gate inspects the `requires_hf_token` class attribute before anything is constructed (constructing a pyannote pipeline without a token fails with an unhelpful error, and nobody wants to download 2 GB of checkpoints to learn their token is missing).

**Migrations** are an ordered `(version, sql)` list in `rytp/db.py`, applied by `Database.migrate()`. Add a new tuple; never edit an existing one. The `schema_version` table is bootstrapped outside the list (chicken-and-egg with the runner), and migration 17 is special-cased in `migrate()` because SQLite has no `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. The `words_fts` FTS5 shadow is kept in sync by triggers — write to `words` and the index follows.

**Constants.** Every tunable lives in `rytp/constants.py` with a comment explaining the value and citing its DESIGN section. New magic numbers go there, not inline.

**CLI error handling.** `rytp/cli.py` funnels business exceptions through `_handle_business_error` — one line to stderr, exit 1, no traceback — and validates enum-ish flags with Typer callbacks (`_cohesion_callback`, `_splice_mode_callback`, …) so bad input never reaches the stage code. Follow both patterns for new subcommands.

## Gotchas

- **Importing `rytp.config` creates directories.** Its module-level `paths` singleton calls `.ensure()` at import time, so importing it — which `cli.py`, `transcribe/run.py`, and `tests/conftest.py` all do — mkdirs `data/{media,audio,output/normalized,transcripts}` relative to the current working directory.
- **`--stt` / `--diarizer` / `--combined` exist twice.** On the top-level app (`cli.py:156`) they only drive the mutual-exclusion check and the HF_TOKEN pre-launch gate; on `transcribe` itself (`cli.py:820`) they are the values actually used for the run. Placing them before the subcommand gates but doesn't configure; placing them after configures but doesn't gate.
- **Re-running `transcribe` deletes and replaces** all `words` rows for that video (`transcribe/run.py:180`). `transcribe_runs` keeps the audit trail, but only the latest run's words survive.
- **Non-ASCII in help text is deliberate.** `cli.py` and `__main__.py` reconfigure stdout/stderr to UTF-8 at import because docstrings use `§` and `—`; this is a Windows-origin project and those characters were mojibake before the fix. Don't ASCII-fy them.
- The diarizer always runs on **full audio**, never on the STT chunks — speaker labels are context-dependent and chunking makes them inconsistent. Only STT chunks (25 min windows, 5 min overlap, above a 30 min threshold).

## Tests

`tests/conftest.py` overrides pytest's built-in `tmp_path` so scratch dirs land in `./.pytest_tmp/` (gitignored) rather than the system temp dir — a workaround for a Windows sandbox `PermissionError`. The `data_dir` fixture monkeypatches the `config.paths` singleton and `db` gives an opened, migrated `Database`; use them instead of touching the real `data/` tree.

Every external tool is mocked or skipped: yt-dlp behind a fake `YtDlpRunner`, faster-whisper and pyannote behind registered fake engines, ffmpeg and FTS5 behind `pytest.mark.skipif`. `tests/test_real_files_integration.py` additionally needs the bundled sample audio/video pair at the repo root and skips cleanly without them. CLI tests drive the app through `typer.testing.CliRunner`.

Because the suite mocks every heavy dependency, **a green test run says almost nothing about whether the real pipeline works.** Verify end-to-end behavior by actually running `python -m rytp` against a real file.
