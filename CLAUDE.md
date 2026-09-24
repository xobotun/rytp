# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A command-line utility that maps a Russian-language video archive to *speaker + word* and supports full-text search, with the index running **both ways**:

- **Forward — `video → transcript`:** every word with its timestamp and the speaker who said it.
- **Backward — `speaker + word/sentence → {video + timestamps}`:** given a word or phrase, optionally constrained to a speaker, find every place it was said.

Downstream of the backward index, clips are cut and spliced into a new uploadable video with a reference list of every source and timestamp. Those two access paths are the point of the project — changes that make either one harder to serve are going the wrong way.

The pipeline is meant to run on the owner's Windows machine (32 GiB RAM, RTX 3080 Laptop 16 GiB, i7-11800H), so local models are the intended path, not cloud APIs.

## State: the rewrite is implemented

The pre-rewrite tree described by `README.md` and `DESIGN.md` is **gone**. Both of those files still describe it and are stale — do not trust them. Eight plans were implemented across parts 1–8 on 2026-09-24:

| Part | Owns |
|---|---|
| 1 Foundation | `config`, `constants`, `models`, `db/`, `commands/` registry, `cli`, `tui` skeleton, catalog commands |
| 2 Acquisition | `acquire/`, `jobs/`, `audio/extract` |
| 3 Transcription | `transcribe/`, `audio/{vad,energy,acoustics}` |
| 4 Index & search | `index/`, `tui/screens/{search,transcript}` |
| 5 Assembly | `assemble/` |
| 6 Render | `render/` |
| 7 Speakers | `diarize/`, `tui/screens/speakers` |
| 8 Integration | TUI shell, `tui/{navigation,enqueue,jobs_view,cutlist_view}`, the four `tests/test_consistency_*.py` suites, the M1 walk |

**2163 tests, `ruff` and `mypy` clean.** 55 registered commands.

### Documents, in authority order

| Document | What it is |
|---|---|
| `docs/superpowers/specs/2026-09-21-rytp-contracts.md` | **Binding** shared interfaces — schema DDL, core types, command registry, engine protocols, job handlers, cross-part invariants. Never vary from it; raise a change instead. |
| `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` | The approved design. Intent, glossary, milestones. |
| `docs/superpowers/plans/2026-09-2*-part*.md` | The eight implementation plans. Historical now, but they explain *why* a module is shaped the way it is. Parts 1–7 are dated `09-21` and part 8 `09-22`, so a `2026-09-21-part*` glob silently misses it. |
| `docs/superpowers/2026-09-21-review-findings.md` | Review findings, and a section recording decisions **deliberately left alone** so they are not re-litigated. |
| `docs/superpowers/2026-09-24-execution-order.md` | The dependency graph between the parts. |

## What has never been run

The suite mocks every model and every binary it can, so **a green run says nothing about these**:

- **No transcriber, aligner, diarizer or embedder has ever executed.** GigaAM, Whisper, MFA, wav2vec2, pyannote and ReDimNet adapters are written from documented APIs and tested through fakes in `tests/fake_engines.py`. Expect adapter-level surprises on first real use.
- **Nothing has ever been downloaded.** yt-dlp always sits behind `FakeYtDlpRunner`; no test may touch the network.
- **Part 7's speaker-matching thresholds are unvalidated guesses** (`SPEAKER_MATCH_HI/LO`, `SPEAKER_MATCH_HI_CROSS_ERA`, `ACOUSTIC_FEATURE_SCALES`). They need a real corpus.

ffmpeg *is* exercised for real where it is on `PATH` (`tests/test_render_real_ffmpeg.py`), and a real `output.mp4` has been produced by hand.

## Setup

Requires Python >= 3.11 and `ffmpeg` on `PATH`.

```bash
scripts/bootstrap.sh            # or bootstrap.ps1 on Windows
```

That creates `.venv` and installs `.[dev]`. Optional extras are `yt-dlp`, `stt`, `pyannote`, `redimnet`, `all`. Commands needing a missing extra print a one-line install hint instead of a traceback.

## Commands

```bash
python -m pytest                       # full suite, ~30s
python -m pytest tests/test_index_search.py -q
python -m pytest -k "utterance and not caption" -q
ruff check rytp tests                  # line-length 100, py311 target
mypy rytp
python -m rytp <subcommand> --help
```

Two env vars matter: `RYTP_DATA` relocates the data tree (default `./data`), and `HF_TOKEN` / `HUGGINGFACE_TOKEN` is needed only by the pyannote diarizer. `RYTP_TEST_TMP` relocates pytest's scratch root.

## Architecture

**Stages communicate only through SQLite.** Each reads and writes `data/rytp.db` plus the `data/` tree; nothing hands data to anything else in memory. That is what makes the job queue, pause/resume and self-healing possible.

**One command registry generates both surfaces.** `rytp/commands/__init__.py` holds `COMMANDS`; `rytp/cli.py` builds the Typer app from it and `rytp/tui/palette.py` builds the palette from it. A command is defined once. Registration happens as a side effect of importing the command module, and the sibling-import block at the bottom of `commands/__init__.py` is what performs it.

**Readiness, not a dependency graph.** A job kind declares a `readiness(db, target_id) -> Readiness` predicate that is a statement about the world — `READY`, `BLOCKED` or `SATISFIED` — derived from database and filesystem state. **A predicate never sees the payload**, deliberately: one that changed its answer based on how a job was enqueued would stop being a statement about the world, and `reconcile`'s self-healing would go with it. Anything that is an *input* to the work belongs in `payload_json`. Handlers may return a short note, stored in `jobs.note` and shown by `jobs.list`.

**Three transcript tiers, one of them cuttable.** `words.source` is `caption | timed | aligned`. `caption` is downloaded auto-captions — searchable, no end times, never cuttable. `timed` is a transcriber's own word timestamps — good text, unusable boundaries. **Only `aligned` may be cut**, and it comes from forced alignment plus a snap to a measured energy minimum and zero crossing.

**Two indexes for two jobs.** FTS5 over contiguous `utterances` serves human search (exact column, stem column fallback); a word-ordinal pointer walk serves assembly. `rytp/index/search.py` owns the first, `rytp/assemble/match.py` the second.

**Assets are never merged.** Audio, video renditions, captions and container are separate `assets` rows, so a rendition can be upgraded later without invalidating a transcript.

**Speaker identity is two-level.** `video_speakers` holds per-video diarizer labels (`SPEAKER_00`); `speakers` is the global roster; a nullable FK links them, and linking is a human step (`rytp speakers map`, or `F3` in the TUI). Re-transcribing deletes a video's `utterances` *and* `video_speakers` in the same transaction that replaces its words — re-mapping is the accepted price.

**Engines are registered classes, not instances.** `rytp/transcribe/registry.py` and `rytp/diarize/` resolve names through dicts that each adapter module fills at import — *that import is what makes a name resolvable*, so a new engine needs the module, the `register_*` call, and the package `__init__` importing it. Classes rather than instances because the HF-token gate inspects a class attribute before anything is constructed.

**Heavy dependencies are lazy, some out-of-process.** Nothing heavy is imported at module load. `rytp/transcribe/subproc.py` runs an engine under a foreign interpreter for libraries whose dependency pins conflict, which is why `rytp/transcribe/base.py` must stay standard-library-only.

**Migrations** are an ordered `(version, sql)` list in `rytp/db/schema.py`. Append; never edit or renumber an existing tuple.

**Constants.** Every tunable lives in `rytp/constants.py` with a comment citing its design section, grouped by the part that added it. The file is append-only.

## Gotchas

- **`rytp/config.py` creates no directories.** `paths()` is a *function* resolved per call from `RYTP_DATA`; there is no module-level singleton. Call `config.ensure_dir(p.parent)` immediately before writing.
- **Words are split on punctuation.** `кто-то` is two rows. A stored row holds exactly one token, and the tokens stored for a piece of text are exactly `normalize_text(text).split()` — because a query runs through the same function. `normalize_text` also folds `ё` to `е`.
- **Stemming is not idempotent.** An utterance's `normalized_text` and `stem_text` are built by joining the per-word columns, never by re-running the functions over the joined string.
- **FTS5 uses `unicode61`, never `porter`** — porter is English-only and does nothing to Russian. Stemming is Python-side via `rytp.models.stem_text`, which keeps a thread-local stemmer because the pools stem concurrently.
- **mypy's `python_version` is `3.12`** only because the installed numpy's stubs will not parse under a 3.11 target. The runtime floor is still 3.11, and `tests/test_python_311_syntax.py` enforces it directly. Do not "fix" the setting.
- **Non-ASCII in help text is deliberate.** `§` and `—` are intentional; stdout/stderr are reconfigured to UTF-8 at import because this is a Windows-origin project.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file**, tests and fixtures included. Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. Russian test strings are wanted.
- The diarizer always runs on **full audio**, never on transcription chunks — speaker labels are context-dependent and chunking makes them inconsistent.

## Tests

`tests/conftest.py` roots `tmp_path` at `$RYTP_TEST_TMP` when set, falling back to a workspace-local directory — a workaround for a Windows sandbox `PermissionError`. `data_dir` sets `RYTP_DATA`; `db` gives an opened, migrated `Database`. `tests/fakes.py` and `tests/fake_engines.py` hold the shared factories.

**The four `tests/test_consistency_*.py` suites are the cross-part net**: they assert that every command reaches both surfaces, that a flag has one spelling and one meaning, that every job kind has a handler, a predicate and a producer, and that the schema matches what each part declared it consumes. They exist because the recurring failure mode in this project was *naming a responsibility without naming a signature*. Keep them strict.

`tests/test_m1_end_to_end.py` is the milestone walk: catalog, ingest, transcribe, align, index, search, assemble a sentence **from two different source videos**, render a file with a source list — all through registered commands, with only the binaries faked.
