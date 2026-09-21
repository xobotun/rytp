# rytp Part 4 — Index, Search and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The backward half of the index — type a word or a phrase, get every place in the archive it was said, with video, timestamp, speaker, surrounding sentence and whether the fragment is cuttable — plus playback, audio export, a regenerable markdown transcript per video that an agent can read and point back into the database from, and the two TUI screens design §10 promises: reading a transcript, and searching with a key that plays the highlighted hit.

**Architecture:** Contiguous runs of `words` are grouped into `utterances`, split on speaker change where speakers are known and on a silence gap (with word and duration caps) otherwise. `utterances_fts` shadows two columns — `normalized_text` and `stem_text` — so a multi-word query is a **phrase match inside one row** instead of the old implicit AND across one-word rows. Search tries the exact column first and the Python-stemmed column second, reporting which tier produced the hits. When a tier's FTS phrase finds nothing, a bounded ordinal walk over `words` catches occurrences that a split put across two utterances, so an utterance boundary is never a reason to miss a phrase. Playback and export shell out to `ffplay`/`ffmpeg` through a replaceable module-level seam, so no test ever spawns a player. The two TUI screens are shells: every row they draw and every action they take goes through a registered command handler, the same one the CLI calls, so the surfaces cannot drift and the still-settling speaker resolver is touched in exactly one place.

**Tech Stack:** Python 3.11+ (`from __future__ import annotations` everywhere), stdlib `sqlite3` with FTS5 (`unicode61 remove_diacritics 0`, **never `porter`**), `snowballstemmer` (Russian, a base dependency), `subprocess` with list arguments for ffmpeg/ffplay, pytest, ruff + mypy.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (design §4 "Utterances", §5 "job queue", §7 "Search") and `docs/superpowers/specs/2026-09-21-rytp-contracts.md` (binding — §1 constraints, §2 layout, §3 schema, §4 types and cross-part invariants, §5 registry and job handlers, §7 filesystem, §8 conventions).

## Why this part exists

The implementation being replaced gets the backward index wrong in two ways, and both are verified against the owner's real data rather than inferred:

1. **`words_fts` shadowed the `words` table, one word per row.** A two-term FTS query is an implicit AND *within a single row*, so it can never match: `'добрый*'` returned 1 hit and `'добрый* вечер*'` returned 0, with those two words adjacent in the corpus. Utterances fix this structurally — the indexed row is now a sentence-sized run, so a phrase query has somewhere to match.
2. **The tokenizer was `porter`,** which is English-only and therefore performed no stemming at all on a Russian corpus: searching `ощущения*` missed the corpus's `ощущение`. The tokenizer is now `unicode61` and stemming happens in Python at write time into `words.stem` and `utterances.stem_text`.

Both failures have a named regression test in this plan (Task 4, Step 1). If either test is deleted, the part has lost its point.

## Global Constraints

- Python >= 3.11, 3.11 syntax. `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. `pathlib` everywhere; `subprocess` only with list arguments. No `:` in a filename this code writes.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` runs on `rytp/`.
- Heavy dependencies (transcribers, aligners, diarizers) are optional extras, imported lazily inside functions, never at module import time. The **base** dependencies, importable at module level, are `typer`, `rich`, `textual`, `numpy`, `scipy`, `snowballstemmer` (contracts §1).
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. Russian test strings are wanted — the stemming and folding tests need them.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- Milliseconds, integers, everywhere. Database timestamps are `datetime.now(UTC).isoformat()` strings.
- Domain errors subclass `RytpError`. The CLI prints one line to stderr and exits 1 — never a traceback for an expected failure. Handlers never print and never call `sys.exit`.
- Every stage takes an already-open `Database`; nothing opens its own connection. Multi-statement writes use `db.transaction()`.
- **The FTS tokenizer is `unicode61 remove_diacritics 0`, never `porter`** (contracts §3).
- **Cuttable is defined as `source = 'aligned'`** and nothing else (contracts §3).
- **One stored `words` row holds exactly one token** (contracts §4), guaranteed by Part 3's `split_token`.
- **Two speaker flags, two identifier spaces** (contracts §5). `--speaker` is a named person from the `speakers` roster and never accepts a raw diarizer label; `--video-local-speaker` is a `video_speakers.local_label` and **requires `--video-id` alongside it**. One shared resolver in `rytp/commands/__init__.py` serves both; **no part rolls its own**.
- **Supplying every `NOT NULL` column is the inserting part's job** (contracts §3) — `video_speakers.engine` included, in fixtures as much as in production.
- **Cross-part invariant (contracts §4):** whenever a video's words are deleted or replaced, that video's `utterances` go with them in the same transaction. Part 3 already does this. Re-deriving them is this part's job, so `index_video` must be idempotent and cheap.
- Tests: no network, ever, and **no test may spawn a media player**. Anything needing ffmpeg is marked `@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason=...)`.
- **Test invocation (pytest is not on `PATH`):**
  `python -m pytest <args>`
- One commit per task, conventional-commit prefix, present tense.

## What this part consumes from earlier plans

Read these signatures; do not re-derive them.

| From | Name | Shape |
|---|---|---|
| Part 1 | `rytp.db.Database` | `.conn` (`row_factory = sqlite3.Row`, FK on, `isolation_level=None`), `.migrate()`, `.transaction()`, `.close()` |
| Part 1 | `rytp.models` | `RytpError`, `NotFoundError`, `InvalidInputError`, `Fragment`, `normalize_text(text) -> str` (folds ё→е), `stem_text(normalized) -> str` (snowballstemmer Russian, one stem per token, **already implemented**), `utc_now_iso() -> str` |
| Part 1 | `rytp.config` | `paths() -> Paths` with `.transcript(video_id)`, `.cache_wav(video_id)`, `.root`; `ensure_dir(path) -> Path` |
| Part 1 | `rytp.commands` | `REQUIRED`, `Param`, `CommandResult`, `Command`, `register`, `resolve`, `COMMANDS`, and the one shared speaker resolver contracts §5 requires — this plan calls it `speaker_scope(db, *, speaker, video_local_speaker, video_id) -> frozenset[int] \| None` |
| Part 1 | schema | `words`, `utterances`, `utterances_fts` + the `utterances_ai` / `_ad` / `_au` triggers, `videos`, `speakers`, `video_speakers` — migrations 6, 7, 8. **Never write your own FTS DDL or triggers; Part 1 owns them and a second copy would conflict.** |
| Part 1 | `rytp.constants` | `NULL_CELL`, `MS_PER_SECOND`, `DEFAULT_LIST_LIMIT`, `MAX_LIST_LIMIT` |
| Part 2 | `rytp.jobs` | `Readiness` (`READY` / `BLOCKED` / `SATISFIED`), `JobKind(name, pool, readiness, handler, summary, target_kind="video", reopenable=True)`, `JOB_KINDS`, `JOB_HANDLERS`, `register_job_kind`, `resolve_job_kind` |
| Part 2 | `rytp.jobs.queue` | `enqueue(db, kind, target_id, *, priority=0, payload=None, now=None) -> int` |
| Part 2 | `rytp.db.queries` | `asset_for(db, video_id, role) -> sqlite3.Row \| None` |
| Part 2 | `rytp.audio.extract` | `wav_path(video_id) -> Path`; the `_ffmpeg_binary()` / `_run_ffmpeg(cmd)` seam pattern this part copies |
| Part 2 | `rytp.constants` | `AUDIO_SAMPLE_RATE_HZ = 16_000`, `AUDIO_CHANNELS = 1`, `FFMPEG_ERROR_TAIL_CHARS` |
| Part 2 | `tests/fakes.py` | `make_video(db, **overrides) -> int` |
| Part 3 | `words` rows | columns `video_id, ord, start_ms, end_ms, text, normalized_text, stem, confidence, align_score, source, engine, video_speaker_id`; `end_ms IS NULL` **exactly** for `source = 'caption'` |
| Part 3 | `rytp.transcribe.captions` | `ingest_captions(db, video_id, path=None) -> int` — used by Task 11's end-to-end run because it needs no binaries |
| Part 3 | `rytp.transcribe.base` | `split_token(text) -> list[tuple[str, str]]` — the one-token-per-row rule; test fixtures split the same way |

**Two facts about `words` rows that shape the code below.**

First, **one stored row holds exactly one token** — contracts §4: *"A stored word row holds exactly one token, and the tokens stored for a piece of text are exactly `normalize_text(text).split()`."* Part 3's `split_token` enforces it, so `кто-то` becomes two rows with the interior boundary measured rather than guessed. Everything here can therefore treat a row and a token as the same thing: the span locator is a plain subsequence search over rows, and a word ordinal and a token position are interchangeable. Test helpers must split the same way Part 3 does, or they will seed rows the real pipeline can never produce.

Second, **stemming is not idempotent** (`сказали` → `сказа`, and `сказа` stems again to something else). So an utterance's `normalized_text` and `stem_text` must be built by **joining the per-word columns**, never by re-running `normalize_text` / `stem_text` over the joined string — doing the latter desynchronises `utterances.stem_text` from `words.stem` and the stem tier silently stops matching. Joining is also what keeps the two columns token-for-token parallel, so a phrase position in one means the same word as the matching position in the other.

## File Structure

**Created by this plan**

| File | Responsibility |
|---|---|
| `rytp/index/__init__.py` | Empty package marker. Imports nothing — `rytp/jobs/__init__.py` imports a sibling at load time. |
| `rytp/index/utterances.py` | Grouping `words` into `utterances`: the pure `build_utterances`, the idempotent `index_video`, and the `index` job's readiness predicate. |
| `rytp/index/search.py` | The backward lookup: query tokenisation, the two FTS tiers, the cross-boundary ordinal walk, anchors. |
| `rytp/index/export.py` | Getting things back out: markdown transcripts, audio clips, and playback — all three are "render an index row into something outside the database". |
| `rytp/commands/search.py` | The six registered commands (`index.build`, `index.drop`, `search.words`, `search.play`, `search.export`, `transcript.build`) and Part 4's two `doctor` health checks. Registration is a side effect of import, and this module is already in the registry's import block. |
| `tests/test_index_utterances.py` | Task 2. |
| `tests/test_index_job.py` | Task 3. |
| `tests/test_index_search.py` | Tasks 4 and 5, including both named regressions. |
| `tests/test_index_export.py` | Tasks 6 and 7. |
| `tests/test_commands_search.py` | Task 8. |
| `rytp/tui/screens/transcript.py` | Task 9. A reader for the blocks `transcript.show` returns — a thin shell, no logic of its own. |
| `rytp/tui/screens/search.py` | Task 10. Query box, hit list, and a key that plays the highlighted hit. A thin shell over `SearchSession`. |
| `tests/test_tui_transcript.py` | Task 9. |
| `tests/test_tui_search.py` | Task 10. |
| `tests/test_index_end_to_end.py` | Task 11. |
| `scripts/manual_check_index.py` | Task 11's by-hand run. A developer tool outside the package — the one file this plan puts outside `rytp/` and `tests/`. |

**Modified by this plan** (all additive; touch nothing above the section you append)

- `rytp/constants.py` — one new section appended at the very end.
- `rytp/jobs/__init__.py` — one handler thunk and one `register_job_kind` call appended to the existing registration block (Part 2's published pattern, plan part 2 lines 61-110).
- `rytp/commands/__init__.py` — one import line added to the existing sibling-import block.
- `rytp/tui/app.py` — two bindings and two actions appended to Part 1's app (Tasks 9 and 10), matching the shape Part 7 used for `f3`.

**Deliberately not created.** There is no `rytp/index/stem.py`, and Part 4 writes no stemmer. `normalize_text` and `stem_text` are both Part 1's, in `rytp/models.py` (contracts §4), and they have to be: Part 3 calls `stem_text` in production when it writes `words.stem`, so a stemmer under the index package would invert the dependency and a stub would deadlock the build order. Task 1 consumes it and pins the two properties Part 4 relies on.

---

## Tasks

### Task 1: Pin the cross-part assumptions Part 4 stands on

Part 4 is the part with the most neighbours: it reads rows Part 3 writes, registers a job kind in Part 2's file, filters through a resolver Part 1 owns, and is re-fired by Part 7. Every one of those is an assumption, and an independent review of an earlier draft of this plan found three of them already stale — a stemmer that had moved, a speaker flag that meant two different things, and a `NOT NULL` column a fixture omitted. None of those failed loudly; they failed as a search that quietly returned the wrong thing.

So the first task builds nothing. It **reads** the four things Part 4 leans on, and leaves behind one small test module that fails loudly in one place if any of them moves again — instead of failing mysteriously in four call sites later.

**`stem_text` is Part 1's, fully implemented.** Do not write it, do not stub it, do not re-export it. Part 3 calls it in production when it writes `words.stem`, so it could never have been Part 4's to supply — that ordering is what makes it Part 1's. Part 4 only *depends* on two of its properties: one stem per token, and the same token order.

**Files:**
- Create: `tests/test_index_assumptions.py`
- Modify: nothing

**Interfaces:**
- Consumes: `rytp.models.{normalize_text, stem_text}`, `rytp.db.Database`, `rytp.commands`, `rytp.transcribe.base.split_token`.
- Produces: no production code. Records, as executable assertions, the four cross-part facts Tasks 2-11 are written against.

- [ ] **Step 1: Read the four things, and write down what you actually find**

```bash
# from the repo root
# 1. The stemmer: it exists and is Part 1's.
python -c "from rytp.models import stem_text; print(stem_text('ощущения'), stem_text('добрый вечер'))"
# 2. The shared speaker resolver, contracts §5 "Speaker filters".
python -c "import rytp.commands as c; print([n for n in dir(c) if 'speaker' in n.lower()])"
# 3. Part 3's one-token-per-row splitter.
python -c "from rytp.transcribe.base import split_token; print(split_token('Кто-то'))"
# 4. Columns that are NOT NULL and therefore every fixture's job to supply.
python -c "
from rytp.config import ensure_dir, paths
from rytp.db import Database
ensure_dir(paths().root); db = Database(paths().db); db.migrate()
for table in ('words', 'utterances', 'video_speakers'):
    cols = db.conn.execute(f'PRAGMA table_info({table})').fetchall()
    print(table, [c[1] for c in cols if c[3] and c[1] != 'id'])
"
```

**Step 2 writes the resolver's real spelling into the test.** This plan calls it `speaker_scope(db, *, speaker, video_local_speaker, video_id) -> frozenset[int] | None` — the set of `video_speakers.id` values a search may match, or `None` for no filter. If Part 1 spelled it differently, **change the call sites in Tasks 4, 5, 8 and 10 to match what exists; do not add your own resolver** (contracts §5: "One shared resolver in `rytp/commands/__init__.py` serves every command that filters by speaker; no part may roll its own").

- [ ] **Step 2: Write the assumption tests**

Create `tests/test_index_assumptions.py`:

```python
"""What Part 4 assumes about its neighbours, as executable assertions.

Every fact here belongs to another part. This module exists so that when
one of them moves, exactly one test fails and names the thing that moved
— rather than a search quietly returning the wrong rows.
"""

from __future__ import annotations

import threading

import pytest

from rytp.db import Database
from rytp.models import normalize_text, stem_text


# --- Part 1: the stemmer (contracts §4) ------------------------------


def test_stem_text_is_part_ones_and_already_works() -> None:
    """Part 4 does not implement this. Part 3 calls it when it writes
    words.stem, which is what makes it Part 1's."""
    assert stem_text("ощущения") == stem_text("ощущение")


def test_stem_text_is_one_stem_per_token_in_the_same_order() -> None:
    """The only two properties Part 4 depends on.

    `utterances.stem_text` is a join of `words.stem`, and a phrase
    position in the stem FTS column has to mean the same word as the
    matching position in the exact column. A stemmer that dropped or
    reordered a token would break the stem tier silently.
    """
    assert stem_text("добрый вечер дорогие друзья").split() == [
        "добр",
        "вечер",
        "дорог",
        "друз",
    ]
    assert stem_text("") == ""
    assert stem_text("   ") == ""


def test_stem_text_sees_only_normalized_input() -> None:
    """normalize_text folds ё→е (contracts §4), so the stemmer never has
    to know about the two spellings."""
    assert stem_text(normalize_text("ещё")) == stem_text(normalize_text("еще"))


def test_stem_text_is_safe_to_call_from_several_threads() -> None:
    """Part 2's worker runs three pools as threads in one process, and the
    index job stems on the cpu pool while transcription stems on the gpu
    pool."""
    words = ["ощущения", "сказали", "добрым", "вечера", "дорогие", "мысли"]
    expected = [stem_text(word) for word in words]
    results: list[list[str]] = []
    lock = threading.Lock()

    def worker() -> None:
        got = [stem_text(word) for _ in range(200) for word in words]
        with lock:
            results.append(got)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [expected * 200] * 8


# --- Part 3: one row, one token (contracts §4) -----------------------


def test_a_stored_word_row_holds_exactly_one_token() -> None:
    """Contracts §4: "the tokens stored for a piece of text are exactly
    `normalize_text(text).split()`". Part 4's span locator is a plain
    subsequence search over rows because of this."""
    from rytp.transcribe.base import split_token

    assert [normalized for _, normalized in split_token("Кто-то")] == ["кто", "то"]
    for surface, normalized in split_token("Кто-то ещё"):
        assert len(normalized.split()) == 1, (surface, normalized)
    assert split_token("—") == []


# --- Part 1: the schema Part 4 writes into (contracts §3) ------------


def test_the_fts_tokenizer_is_unicode61_and_never_porter(db: Database) -> None:
    """Porter is English-only; on a Russian corpus it stems nothing at
    all, which is the second of the two measured failures this part
    exists to fix."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "unicode61" in sql
    assert "porter" not in sql


def test_the_fts_shadow_has_both_columns(db: Database) -> None:
    """Exact first, stems as the fallback (design §7) needs two columns."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "normalized_text" in sql
    assert "stem_text" in sql


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("video_speakers", "engine"),
        ("words", "engine"),
        ("words", "stem"),
        ("utterances", "stem_text"),
    ],
)
def test_these_columns_are_not_null_so_every_fixture_must_supply_them(
    db: Database, table: str, column: str
) -> None:
    """Contracts §3 makes supplying a NOT NULL column the inserting part's
    job. A fixture that omits one fails at insert time, which is fine —
    but it fails in whichever test happens to run first, so it is worth
    naming the columns here."""
    info = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    not_null = {row[1] for row in info if row[3]}
    assert column in not_null


# --- Part 1: the shared speaker resolver (contracts §5) --------------


def test_the_registries_part_four_writes_into_exist() -> None:
    """Contracts §5: job kinds, commands and health checks all register
    into `rytp.commands` / `rytp.jobs`; Part 4 adds to three of them."""
    import rytp.commands as commands

    assert hasattr(commands, "register_check")
    assert hasattr(commands, "HEALTH_CHECKS")


def test_the_shared_speaker_resolver_exists() -> None:
    """Contracts §5: one resolver in rytp/commands, no part rolls its own.

    If this fails because Part 1 named it differently, change Part 4's
    call sites to the real name — do not add a second resolver.
    """
    import rytp.commands as commands

    assert hasattr(commands, "speaker_scope"), sorted(
        name for name in dir(commands) if "speaker" in name.lower()
    )
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/test_index_assumptions.py -v`
Expected: PASS, 13 passed (the NOT NULL check is parametrized over four columns).

A failure here is information, not an obstacle: it means a neighbouring part moved. Fix this module and the call sites in Tasks 2-11 to match what is really there, and say so in the commit message — do not work around it locally.

- [ ] **Step 4: Lint**

Run: `ruff check tests/test_index_assumptions.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
# from the repo root
git add tests/test_index_assumptions.py
git commit -m "test: pin the cross-part facts the index is written against"
```

---

### Task 2: Building utterances from words

Design §4: *"Contiguous runs of words, split on speaker change where speakers are known and on a silence threshold otherwise — so caption-tier and undiarized videos still get sensible units."* This is the row the FTS index shadows, so getting the unit right is what makes phrase search work.

Four split rules, and each earns its place:

| Rule | Why |
|---|---|
| speaker change | Design §4. `NULL` counts as a value, so an undiarized video is one group and a diarized one splits per turn. |
| silence >= `UTTERANCE_SILENCE_GAP_MS` | The stated fallback for undiarized audio. |
| `UTTERANCE_MAX_WORDS` | **Load-bearing for caption tier.** Caption words inside one segment share a start time, so their gaps are zero; without a word cap a captioned hour would be one 10,000-word row. |
| `UTTERANCE_MAX_DURATION_MS` | Stops a slow, heavily-paused stretch under the word cap from becoming a two-minute row. |

Caption rows have `end_ms IS NULL` (contracts §3), so the builder computes an **implied end**: the next word's start, capped at `CAPTION_WORD_FALLBACK_MS` after this word's start, and the fallback outright for the last word. The cap is what lets the silence rule fire on caption tier at all — without it the implied end would always equal the next start and every gap would be zero.

**`index_video` deletes and rebuilds; it never inserts or upserts.** Three callers depend on
that, and all three hand it a video whose utterances are already wrong rather than missing:
Part 3 replacing a transcript, Part 7 deleting a video's utterances after diarizing it so they
come back split per turn and carrying speakers, and the `index` job re-firing on a reconcile.
So it must be safe to run repeatedly, leave no orphaned rows, and leave **no stale FTS entry** —
a surviving `utterances_fts` row for text that is no longer in `utterances` would silently
poison search with a hit nobody can explain. Part 1's `utterances_ad` trigger removes the FTS
row on delete, which makes this free; Step 1 asserts it anyway rather than trusting it.

**Files:**
- Create: `rytp/index/__init__.py`
- Create: `rytp/index/utterances.py`
- Modify: `rytp/constants.py` (append to the section Task 1 added)
- Test: `tests/test_index_utterances.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.NotFoundError`, `rytp.constants.{UTTERANCE_SILENCE_GAP_MS, UTTERANCE_MAX_WORDS, UTTERANCE_MAX_DURATION_MS, CAPTION_WORD_FALLBACK_MS}`, `rytp.jobs.Readiness` (imported inside the function body — see Step 5).
- Produces:
  - `rytp.index.utterances.IndexedWord` frozen dataclass — `ord, start_ms, end_ms, text, normalized_text, stem, source, video_speaker_id`
  - `rytp.index.utterances.UtteranceDraft` frozen dataclass — `video_speaker_id, start_ms, end_ms, first_word_ord, last_word_ord, text, normalized_text, stem_text`
  - `implied_end_ms(words: Sequence[IndexedWord], position: int) -> int`
  - `build_utterances(words: Sequence[IndexedWord]) -> list[UtteranceDraft]` — pure, no database
  - `words_for(db: Database, video_id: int) -> list[IndexedWord]`
  - `index_video(db: Database, video_id: int) -> int` — idempotent; returns the utterance count
  - `drop_utterances(db: Database, video_id: int) -> int` — deletes them; returns how many went
  - `index_readiness(db: Database, video_id: int) -> Readiness`
  - `videos_needing_index(db: Database) -> list[int]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_index_utterances.py`:

```python
"""Grouping words into the rows the FTS index shadows (design §4, §7)."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.index.utterances import (
    IndexedWord,
    build_utterances,
    drop_utterances,
    implied_end_ms,
    index_video,
    words_for,
)
from rytp.models import NotFoundError, normalize_text, stem_text

NOW = "2026-09-21T00:00:00+00:00"


def make_video(db: Database, title: str = "Sample") -> int:
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, created_at)"
        " VALUES ('ytdlp', 'video', ?, 'https://example.invalid/w/VIDEO_A', ?, ?)",
        (f"VIDEO_{title}", title, NOW),
    )
    return int(cursor.lastrowid)


def make_speaker(db: Database, video_id: int, local_label: str) -> int:
    """A diarizer label for one video.

    `engine` is NOT NULL (contracts §3) and supplying it is the inserting
    part's job — a fixture that omits it fails at insert time, in
    whichever test happens to run first.
    """
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine) VALUES (?, ?, 'fake')",
        (video_id, local_label),
    )
    return int(cursor.lastrowid)


def seed_words(
    db: Database,
    video_id: int,
    spec: list[tuple[str, int, int | None]],
    *,
    source: str = "aligned",
    speaker_ids: list[int | None] | None = None,
) -> None:
    """Insert word rows from (text, start_ms, end_ms) triples."""
    speakers = speaker_ids or [None] * len(spec)
    for ordinal, ((text, start, end), speaker) in enumerate(zip(spec, speakers, strict=True)):
        normalized = normalize_text(text)
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine, video_speaker_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'fake', ?)",
            (video_id, ordinal, start, end, text, normalized, stem_text(normalized),
             source, speaker),
        )


def word(
    ordinal: int,
    text: str,
    start: int,
    end: int | None,
    *,
    speaker: int | None = None,
    source: str = "aligned",
) -> IndexedWord:
    normalized = normalize_text(text)
    return IndexedWord(
        ord=ordinal,
        start_ms=start,
        end_ms=end,
        text=text,
        normalized_text=normalized,
        stem=stem_text(normalized),
        source=source,
        video_speaker_id=speaker,
    )


def test_a_tight_run_with_one_speaker_is_one_utterance() -> None:
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300, 700),
        word(2, "друзья", 750, 1200),
    ]
    drafts = build_utterances(words)
    assert len(drafts) == 1
    assert drafts[0].first_word_ord == 0
    assert drafts[0].last_word_ord == 2
    assert drafts[0].start_ms == 0
    assert drafts[0].end_ms == 1200
    assert drafts[0].text == "Добрый вечер друзья"
    assert drafts[0].normalized_text == "добрый вечер друзья"


def test_a_speaker_change_starts_a_new_utterance() -> None:
    words = [
        word(0, "Добрый", 0, 300, speaker=1),
        word(1, "вечер", 300, 700, speaker=1),
        word(2, "Здравствуйте", 720, 1200, speaker=2),
    ]
    drafts = build_utterances(words)
    assert [(d.first_word_ord, d.last_word_ord) for d in drafts] == [(0, 1), (2, 2)]
    assert [d.video_speaker_id for d in drafts] == [1, 2]


def test_unlabelled_words_are_one_group_because_null_equals_null() -> None:
    """An undiarized video must not split on every word."""
    words = [word(i, "слово", i * 300, i * 300 + 250) for i in range(5)]
    assert len(build_utterances(words)) == 1


def test_a_silence_at_or_over_the_threshold_splits() -> None:
    gap = C.UTTERANCE_SILENCE_GAP_MS
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300, 700),
        word(2, "Сегодня", 700 + gap, 1000 + gap),
    ]
    assert [(d.first_word_ord, d.last_word_ord) for d in build_utterances(words)] == [
        (0, 1),
        (2, 2),
    ]


def test_a_silence_under_the_threshold_does_not_split() -> None:
    gap = C.UTTERANCE_SILENCE_GAP_MS - 1
    words = [
        word(0, "Добрый", 0, 300),
        word(1, "вечер", 300 + gap, 700 + gap),
    ]
    assert len(build_utterances(words)) == 1


def test_the_word_cap_splits_an_unbroken_monologue() -> None:
    count = C.UTTERANCE_MAX_WORDS * 2 + 3
    words = [word(i, "слово", i * 100, i * 100 + 100) for i in range(count)]
    drafts = build_utterances(words)
    assert len(drafts) == 3
    assert [d.last_word_ord - d.first_word_ord + 1 for d in drafts] == [
        C.UTTERANCE_MAX_WORDS,
        C.UTTERANCE_MAX_WORDS,
        3,
    ]


def test_the_duration_cap_splits_a_slow_run_under_the_word_cap() -> None:
    """Short words, long gaps: every gap stays under the silence threshold
    and the run stays under the word cap, so only the duration cap can
    split this."""
    step = C.UTTERANCE_SILENCE_GAP_MS - 50
    words = [word(i, "слово", i * step, i * step + 40) for i in range(35)]
    drafts = build_utterances(words)
    assert len(drafts) > 1
    first_length = drafts[0].last_word_ord - drafts[0].first_word_ord + 1
    assert first_length < C.UTTERANCE_MAX_WORDS
    for draft in drafts:
        assert draft.end_ms - draft.start_ms <= C.UTTERANCE_MAX_DURATION_MS


def test_a_caption_word_with_no_end_time_borrows_the_next_start() -> None:
    words = [
        word(0, "Добрый", 0, None, source="caption"),
        word(1, "вечер", 200, None, source="caption"),
    ]
    assert implied_end_ms(words, 0) == 200


def test_a_caption_words_implied_end_is_capped_so_silence_still_splits() -> None:
    """Without the cap the implied end would be the next start and no
    caption-tier gap could ever reach the silence threshold."""
    far = C.CAPTION_WORD_FALLBACK_MS + C.UTTERANCE_SILENCE_GAP_MS + 1000
    words = [
        word(0, "Добрый", 0, None, source="caption"),
        word(1, "вечер", far, None, source="caption"),
    ]
    assert implied_end_ms(words, 0) == C.CAPTION_WORD_FALLBACK_MS
    assert len(build_utterances(words)) == 2


def test_the_last_caption_word_gets_the_fallback_duration() -> None:
    words = [word(0, "Добрый", 1000, None, source="caption")]
    drafts = build_utterances(words)
    assert drafts[0].end_ms == 1000 + C.CAPTION_WORD_FALLBACK_MS


def test_caption_words_sharing_one_start_do_not_split_and_do_not_go_backwards() -> None:
    words = [
        word(0, "Добрый", 500, None, source="caption"),
        word(1, "вечер", 500, None, source="caption"),
        word(2, "друзья", 500, None, source="caption"),
    ]
    drafts = build_utterances(words)
    assert len(drafts) == 1
    assert drafts[0].end_ms >= drafts[0].start_ms


def test_the_utterance_columns_are_a_join_of_the_word_columns() -> None:
    """Join the per-word columns; never re-run the text functions.

    Stemming is not idempotent — `сказали` stems to `сказа`, which stems
    again — so re-deriving `stem_text` from the joined string would drift
    away from `words.stem`. Joining also keeps the two columns
    token-for-token parallel, which is what lets a phrase position in the
    stem column mean the same word as in the exact column.
    """
    words = [word(0, "Кто", 0, 300), word(1, "то", 300, 600), word(2, "сказали", 600, 1100)]
    draft = build_utterances(words)[0]
    assert draft.text == "Кто то сказали"
    assert draft.normalized_text == "кто то сказали"
    assert draft.stem_text == "кто то сказа"
    assert len(draft.normalized_text.split()) == len(draft.stem_text.split())
    assert draft.stem_text != stem_text(draft.stem_text)


def test_no_words_means_no_utterances() -> None:
    assert build_utterances([]) == []


def test_index_video_writes_rows_and_returns_the_count(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    assert index_video(db, video_id) == 1
    row = db.conn.execute(
        "SELECT * FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert row["normalized_text"] == "добрый вечер"
    assert row["first_word_ord"] == 0
    assert row["last_word_ord"] == 1


def test_index_video_is_idempotent(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    first = index_video(db, video_id)
    second = index_video(db, video_id)
    assert first == second == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_index_video_only_touches_its_own_video(db: Database) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    seed_words(db, first, [("Добрый", 0, 300)])
    seed_words(db, second, [("Здравствуйте", 0, 400)])
    index_video(db, first)
    index_video(db, second)
    index_video(db, first)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 2


def test_reindexing_after_the_words_change_replaces_the_rows(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    seed_words(db, video_id, [("Сегодня", 0, 500)])
    index_video(db, video_id)
    rows = db.conn.execute(
        "SELECT normalized_text FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchall()
    assert [r["normalized_text"] for r in rows] == ["сегодня"]


def test_index_video_populates_the_fts_shadow(db: Database) -> None:
    """Part 1's triggers do this; the test proves the write path reaches them."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    hits = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchall()
    assert len(hits) == 1


def test_reindexing_removes_the_old_text_from_the_fts_shadow(db: Database) -> None:
    """A stale FTS row would be a hit pointing at text that no longer exists.

    Part 1's `utterances_ad` trigger deletes it; this asserts the write
    path actually reaches the trigger rather than trusting that it does.
    """
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    db.conn.execute("DELETE FROM words WHERE video_id = ?", (video_id,))
    seed_words(db, video_id, [("Сегодня", 0, 400), ("поговорим", 400, 900)])
    index_video(db, video_id)
    stale = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchall()
    fresh = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сегодня поговорим"',),
    ).fetchall()
    assert stale == []
    assert len(fresh) == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 1


def test_reindexing_leaves_the_fts_shadow_consistent(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    index_video(db, video_id)
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_drop_utterances_removes_them_and_reports_how_many(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    assert drop_utterances(db, video_id) == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 0


def test_dropping_clears_the_fts_shadow_too(db: Database) -> None:
    """The failure mode most worth pinning: a surviving `utterances_fts`
    row is a search hit for text that is no longer anywhere."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    drop_utterances(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_dropping_leaves_the_words_alone(db: Database) -> None:
    """Derived data only. Removing words is `transcribe.remove` (Part 3)."""
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700)])
    index_video(db, video_id)
    drop_utterances(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2


def test_dropping_twice_is_not_an_error(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300)])
    index_video(db, video_id)
    assert drop_utterances(db, video_id) == 1
    assert drop_utterances(db, video_id) == 0


def test_dropping_only_touches_its_own_video(db: Database) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    seed_words(db, first, [("Добрый", 0, 300)])
    seed_words(db, second, [("Здравствуйте", 0, 400)])
    index_video(db, first)
    index_video(db, second)
    drop_utterances(db, first)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 1


def test_dropping_refuses_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        drop_utterances(db, 404)


def test_index_video_refuses_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError, match="404"):
        index_video(db, 404)


def test_words_for_returns_them_in_ordinal_order(db: Database) -> None:
    video_id = make_video(db)
    seed_words(db, video_id, [("Добрый", 0, 300), ("вечер", 300, 700), ("друзья", 700, 1100)])
    got = words_for(db, video_id)
    assert [w.ord for w in got] == [0, 1, 2]
    assert [w.text for w in got] == ["Добрый", "вечер", "друзья"]


def test_a_diarized_video_splits_per_turn(db: Database) -> None:
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    seed_words(
        db,
        video_id,
        [("Добрый", 0, 300), ("вечер", 300, 700), ("Здравствуйте", 720, 1300)],
        speaker_ids=[host, host, guest],
    )
    assert index_video(db, video_id) == 2
    rows = db.conn.execute(
        "SELECT video_speaker_id FROM utterances WHERE video_id = ? ORDER BY start_ms",
        (video_id,),
    ).fetchall()
    assert [r["video_speaker_id"] for r in rows] == [host, guest]
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_index_utterances.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.index'`.

- [ ] **Step 3: Append the constants**

At the end of the Part 4 section of `rytp/constants.py`:

```python
#: A pause at or above this splits one utterance from the next when the
#: speaker is unknown (design §4: "split ... on a silence threshold
#: otherwise"). Conversational sentence boundaries sit around 0.5-1.0 s;
#: anything shorter is a within-sentence hesitation and must not split,
#: because a split is what forces phrase search onto the slower
#: cross-boundary walk.
UTTERANCE_SILENCE_GAP_MS: Final = 700

#: Hard cap on words per utterance. Load-bearing for caption tier, whose
#: words inside one caption segment all share a start time and therefore
#: have zero gaps: without this a captioned hour is a single row.
#: Forty words is roughly fifteen seconds of ordinary speech.
UTTERANCE_MAX_WORDS: Final = 40

#: Hard cap on utterance duration, for a slow heavily-paused stretch that
#: stays under the word cap.
UTTERANCE_MAX_DURATION_MS: Final = 20_000

#: How long a caption-tier word is assumed to last when the next word is
#: further away than this, or when it is the last word of the video.
#: Captions carry starts and no ends (design §6), and capping the implied
#: end here is what lets UTTERANCE_SILENCE_GAP_MS fire on caption tier at
#: all — an uncapped implied end always equals the next start, making
#: every caption gap zero.
CAPTION_WORD_FALLBACK_MS: Final = 400
```

- [ ] **Step 4: Create the package marker**

`rytp/index/__init__.py`:

```python
"""The backward index: words to utterances, utterances to search hits.

Design §7. This package holds the half of the product the archive exists
for — given a word or a phrase, every place it was said. Nothing here is
imported for its side effects, and the package marker stays empty on
purpose: ``rytp/jobs/__init__.py`` imports
``rytp.index.utterances.index_readiness`` at module load time to register
the ``index`` job kind, so anything heavy added here would end up on the
import path of every command.

There is no ``stem.py``: contracts §4 puts ``stem_text`` in
``rytp/models.py``, because ``words.stem`` is written by the
transcription stage and a stemmer here would invert the dependency.
"""

from __future__ import annotations
```

- [ ] **Step 5: Write `rytp/index/utterances.py`**

```python
"""Grouping words into utterances — the rows the FTS index shadows.

Design §4: "Contiguous runs of words, split on speaker change where
speakers are known and on a silence threshold otherwise — so caption-tier
and undiarized videos still get sensible units."

Why this exists at all: the implementation being replaced shadowed the
`words` table with FTS, one word per row, so a two-term query became an
implicit AND inside a single row and could never match. Verified against
the owner's corpus: `добрый*` returned one hit and `добрый* вечер*`
returned none, with the two words adjacent in the text. Indexing a
sentence-sized run instead is the structural fix.

Contracts §4 makes `utterances` derived data: whenever a video's words
are replaced they are deleted in the same transaction, and re-deriving
them is this module's job. `index_video` is therefore written to be
idempotent and cheap rather than incremental.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rytp import constants as C
from rytp.db import Database
from rytp.models import NotFoundError

if TYPE_CHECKING:  # pragma: no cover - import cycle, see index_readiness
    from rytp.jobs import Readiness

__all__ = [
    "IndexedWord",
    "UtteranceDraft",
    "build_utterances",
    "drop_utterances",
    "implied_end_ms",
    "index_readiness",
    "index_video",
    "videos_needing_index",
    "words_for",
]


@dataclass(frozen=True)
class IndexedWord:
    """One `words` row, as the builder needs to see it.

    `end_ms` is None exactly for caption-sourced rows (contracts §3).
    `normalized_text` is exactly one token and `stem` is its stem, also
    one token: contracts §4 makes a row and a token the same thing, and
    Part 3's `split_token` is what guarantees it.
    """

    ord: int
    start_ms: int
    end_ms: int | None
    text: str
    normalized_text: str
    stem: str
    source: str
    video_speaker_id: int | None


@dataclass(frozen=True)
class UtteranceDraft:
    """One `utterances` row before it has an id."""

    video_speaker_id: int | None
    start_ms: int
    end_ms: int
    first_word_ord: int
    last_word_ord: int
    text: str
    normalized_text: str
    stem_text: str


_SELECT_WORDS = (
    "SELECT ord, start_ms, end_ms, text, normalized_text, stem, source, video_speaker_id"
    " FROM words WHERE video_id = ? ORDER BY ord"
)

_INSERT_UTTERANCE = (
    "INSERT INTO utterances (video_id, video_speaker_id, start_ms, end_ms,"
    " first_word_ord, last_word_ord, text, normalized_text, stem_text)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def implied_end_ms(words: Sequence[IndexedWord], position: int) -> int:
    """When the word at `position` stops, inventing an end for caption rows.

    Aligned rows have a real `end_ms`. Caption rows have none (design §6:
    captions carry per-word starts on a 40 ms grid and no ends), so the
    end is the next word's start, capped at `CAPTION_WORD_FALLBACK_MS`
    after this word's start. The cap is the whole reason the silence rule
    works on caption tier: without it the implied end always equals the
    next start and every gap is zero.
    """
    word = words[position]
    if word.end_ms is not None:
        return word.end_ms
    fallback = word.start_ms + C.CAPTION_WORD_FALLBACK_MS
    if position + 1 >= len(words):
        return fallback
    # max() guards the caption case where several words share one start.
    return max(word.start_ms, min(words[position + 1].start_ms, fallback))


def _starts_new_utterance(
    words: Sequence[IndexedWord], run: Sequence[int], position: int
) -> bool:
    """Does the word at `position` begin a new utterance?"""
    previous = words[run[-1]]
    word = words[position]
    if word.video_speaker_id != previous.video_speaker_id:
        return True
    if len(run) >= C.UTTERANCE_MAX_WORDS:
        return True
    if word.start_ms - implied_end_ms(words, run[-1]) >= C.UTTERANCE_SILENCE_GAP_MS:
        return True
    span = implied_end_ms(words, position) - words[run[0]].start_ms
    return span > C.UTTERANCE_MAX_DURATION_MS


def _draft(words: Sequence[IndexedWord], run: Sequence[int]) -> UtteranceDraft:
    members = [words[position] for position in run]
    first = members[0]
    last = members[-1]
    return UtteranceDraft(
        video_speaker_id=first.video_speaker_id,
        start_ms=first.start_ms,
        end_ms=max(implied_end_ms(words, run[-1]), first.start_ms),
        first_word_ord=first.ord,
        last_word_ord=last.ord,
        # Joined from the per-word columns, never re-derived from the
        # joined string. Stemming is not idempotent (сказали → сказа,
        # which stems again), so re-running it over the joined text would
        # desynchronise utterances.stem_text from words.stem and the stem
        # tier would quietly stop matching.
        text=" ".join(member.text for member in members),
        normalized_text=" ".join(member.normalized_text for member in members),
        stem_text=" ".join(member.stem for member in members),
    )


def build_utterances(words: Sequence[IndexedWord]) -> list[UtteranceDraft]:
    """Group a video's words into utterances. Pure: no database, no clock."""
    drafts: list[UtteranceDraft] = []
    run: list[int] = []
    for position in range(len(words)):
        if run and _starts_new_utterance(words, run, position):
            drafts.append(_draft(words, run))
            run = []
        run.append(position)
    if run:
        drafts.append(_draft(words, run))
    return drafts


def words_for(db: Database, video_id: int) -> list[IndexedWord]:
    """Every word of one video, in ordinal order."""
    return [
        IndexedWord(
            ord=int(row["ord"]),
            start_ms=int(row["start_ms"]),
            end_ms=None if row["end_ms"] is None else int(row["end_ms"]),
            text=str(row["text"]),
            normalized_text=str(row["normalized_text"]),
            stem=str(row["stem"]),
            source=str(row["source"]),
            video_speaker_id=(
                None if row["video_speaker_id"] is None else int(row["video_speaker_id"])
            ),
        )
        for row in db.conn.execute(_SELECT_WORDS, (video_id,))
    ]


def index_video(db: Database, video_id: int) -> int:
    """Rebuild one video's utterances from its words. Returns how many.

    Delete-and-rewrite rather than incremental, because contracts §4 makes
    utterances disposable derived data and a whole video is a few thousand
    rows. Part 1's triggers keep `utterances_fts` in step.
    """
    if db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is None:
        raise NotFoundError(f"no video with id {video_id}")
    drafts = build_utterances(words_for(db, video_id))
    rows = [
        (
            video_id,
            draft.video_speaker_id,
            draft.start_ms,
            draft.end_ms,
            draft.first_word_ord,
            draft.last_word_ord,
            draft.text,
            draft.normalized_text,
            draft.stem_text,
        )
        for draft in drafts
    ]
    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
        db.conn.executemany(_INSERT_UTTERANCE, rows)
    return len(rows)


def drop_utterances(db: Database, video_id: int) -> int:
    """Delete one video's utterances. Returns how many went.

    Contracts §5 gives every group a `remove`, and this is Part 4's. It
    needs no `--dry-run` and no `--yes`: utterances are derived data with
    no file behind them, and `index build` puts them straight back.

    The words are untouched — deleting them is `transcribe.remove`, which
    is Part 3's and takes the utterances with it (contracts §4).
    """
    if db.conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone() is None:
        raise NotFoundError(f"no video with id {video_id}")
    with db.transaction():
        count = int(
            db.conn.execute(
                "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        )
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    return count


def index_readiness(db: Database, video_id: int) -> Readiness:
    """Design §5: answered from database state alone, never from a payload.

    Three ways to need re-indexing, and the third is the one that is easy
    to miss: Part 7 fills `words.video_speaker_id` **in place** rather than
    replacing words, so the contracts §4 delete-the-utterances invariant
    never fires and a video indexed before diarization keeps utterances
    that were never split on speaker.
    """
    # Imported in the body deliberately. `rytp/jobs/__init__.py` imports
    # this module at load time to register the `index` kind, so a
    # module-level `from rytp.jobs import Readiness` would be a real cycle
    # for anyone who imports `rytp.index.utterances` first.
    from rytp.jobs import Readiness

    have = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(ord) AS lo, MAX(ord) AS hi,"
        " COUNT(video_speaker_id) AS labelled FROM words WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if not have["n"]:
        return Readiness.BLOCKED
    built = db.conn.execute(
        "SELECT COUNT(*) AS n, MIN(first_word_ord) AS lo, MAX(last_word_ord) AS hi,"
        " COUNT(video_speaker_id) AS labelled FROM utterances WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if not built["n"]:
        return Readiness.READY
    if (built["lo"], built["hi"]) != (have["lo"], have["hi"]):
        return Readiness.READY
    if have["labelled"] and not built["labelled"]:
        return Readiness.READY
    return Readiness.SATISFIED


def videos_needing_index(db: Database) -> list[int]:
    """Every video whose utterances are missing or stale, in id order."""
    from rytp.jobs import Readiness

    ids = [
        int(row["video_id"])
        for row in db.conn.execute("SELECT DISTINCT video_id FROM words ORDER BY video_id")
    ]
    return [i for i in ids if index_readiness(db, i) is Readiness.READY]
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_utterances.py -v`
Expected: PASS, 29 passed.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check rytp/index tests/test_index_utterances.py`
Run: `mypy rytp/index`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
# from the repo root
git add rytp/index/__init__.py rytp/index/utterances.py rytp/constants.py \
        tests/test_index_utterances.py
git commit -m "feat(index): group words into utterances, and drop them again"
```

---

### Task 3: Registering the `index` job kind

Contracts §5 makes the job-kind-to-callable mapping a contract, and Part 2 owns the file it lives in. Part 2's plan publishes the registration recipe for Parts 3-7 (its "Produces for Parts 3-7: registering a job kind" section): a light predicate in your own package, a handler thunk at the bottom of `rytp/jobs/__init__.py` importing your stage lazily, and an entry in the registration block. `register_job_kind` writes both `JOB_KINDS` and the contracted flat `JOB_HANDLERS: dict[str, Callable[[Database, int, dict], None]]`, so the two cannot drift.

`index` is `pool="cpu"` (design §5's table), `target_kind="video"` (the default) and `reopenable=True` (the default) — it is derived from the corpus rather than a one-shot the user asked for, so a reconcile that finds the utterances gone **should** re-fire it.

**The predicate's most important answer is READY for "has words, has no utterances".** That is not an edge case, it is the hand-off from two other parts. Part 3 deletes a video's utterances when it replaces its words (the contracts §4 invariant), and Part 7 deletes them after diarizing and enqueues this job, because design §7's speaker filter searches *utterances*: until they are rebuilt they carry no speaker and their boundaries were drawn by the silence rule instead of the speaker rule, so a speaker-filtered search returns nothing. A predicate that only fired for newly-transcribed videos would leave both stages stranded. The handler receives the payload and ignores it: the video id is the whole input, and design §5 keeps every *input* to the work in `payload_json` precisely so a readiness predicate never needs one.

**The import cycle is real and this task is where it bites.** `rytp/jobs/__init__.py` imports `rytp.index.utterances` at load time. `rytp.jobs.readiness` gets away with a module-level `from rytp.jobs import Readiness` only because Python imports a parent package before its submodule; `rytp.index.utterances` is not a submodule of `rytp.jobs` and gets no such protection. Task 2 already put that import inside the function body. Step 1 proves it.

**Files:**
- Modify: `rytp/jobs/__init__.py` (append one thunk and one registration to the existing block at the bottom; touch nothing above it)
- Test: `tests/test_index_job.py`

**Interfaces:**
- Consumes: `rytp.jobs.{JobKind, JOB_KINDS, JOB_HANDLERS, Readiness, register_job_kind, resolve_job_kind}`, `rytp.index.utterances.{index_readiness, index_video}`.
- Produces: the registered job kind `index` and the thunk `rytp.jobs._run_index(db, video_id, payload) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_index_job.py`:

```python
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
```

Both probes run `sys.executable` with the parent's environment, so they import the package the same way the test session does; neither touches the data tree, so no `child_env` helper is needed.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_index_job.py -v`
Expected: most tests fail with `ValueError: unknown job kind 'index'; available: captions, download, extract_wav`.

- [ ] **Step 3: Add the thunk and the registration**

In `rytp/jobs/__init__.py`, with the other `_run_*` thunks (above the registration block):

```python
def _run_index(db: Database, video_id: int, payload: dict[str, Any]) -> None:
    from rytp.index.utterances import index_video

    del payload  # the video id is the whole input; nothing is configurable
    index_video(db, video_id)
```

Extend the existing bottom-of-file import block — the one that already pulls in `rytp.jobs.readiness` — with the Part 4 predicate:

```python
from rytp.index.utterances import index_readiness  # noqa: E402
```

and append to the registration block:

```python
register_job_kind(
    JobKind(
        name="index",
        pool="cpu",
        readiness=index_readiness,
        handler=_run_index,
        summary="rebuild one video's utterances and its search index",
    )
)
```

`target_kind` and `reopenable` are left at their defaults on purpose: `index` targets a video, and a reconcile that finds the utterances gone should put the job back.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_job.py -v`
Expected: PASS, 15 passed.

- [ ] **Step 5: Confirm Part 2's suite still passes**

Adding a kind changes `JOB_KINDS` and `kinds_for_pool("cpu")`, which Part 2 asserts on.

Run: `python -m pytest tests/test_jobs_readiness.py tests/test_jobs_queue.py tests/test_jobs_worker.py tests/test_commands_ingest.py -q`
Expected: PASS. If a Part 2 test asserts an exact set of kinds rather than a superset, widen that assertion to `>=` rather than removing the registration — the new kind is correct and the over-tight assertion is not.

- [ ] **Step 6: Prove the import path stayed light**

Run: `python -c "import sys, rytp.jobs; assert 'yt_dlp' not in sys.modules; assert 'rytp.index.search' not in sys.modules; print('clean')"`
Expected: `clean`.

- [ ] **Step 7: Commit**

```bash
# from the repo root
git add rytp/jobs/__init__.py tests/test_index_job.py
git commit -m "feat(index): register the index job kind with a self-healing readiness predicate"
```

---

### Task 4: Phrase search over the two FTS columns

Design §7: *"Human search ... goes through the utterance FTS index. Exact phrase match first; if that returns nothing, retry against the stemmed column so Russian word endings stop mattering. Results show the video, timestamp, speaker, surrounding sentence, and whether the hit is cuttable. Filters: by speaker and cuttable only."*

Three decisions worth stating before the code:

- **A query is always a phrase, never a bag of words.** Tokens are normalized and wrapped in one FTS `"..."` phrase, scoped to a column with `column : "..."`. That is what makes `добрый вечер` mean *adjacent*, and it also means a token that happens to spell `OR` or `NEAR` is literal text rather than an operator. No `*` prefix wildcards: the old code used them to paper over a tokenizer that did no stemming, and the stem column replaces them.
- **The tier is reported, because an inflection is not what the user typed.** `SearchResult.tier` is `MatchTier.EXACT`, `MatchTier.STEM`, or `None` when nothing matched.
- **Speaker filtering takes resolved ids, not a label.** Contracts §5 gives the two identifier spaces — a roster person and a raw diarizer label — one shared resolver in `rytp/commands/__init__.py`, and this module never sees a string. `search` takes `speaker_ids: frozenset[int] | None` of `video_speakers.id`; `None` is no filter, and an empty set is a filter nothing satisfies. Keeping resolution out of here is what stops `--speaker` meaning one thing in `search words` and another in `assemble`.
- **FTS finds the utterance; a scan over its words finds the exact span inside it.** The hit's timestamps have to be the phrase's, not the sentence's, or playback and export would replay the whole paragraph. Contracts §4 puts exactly one token in a row, so that scan is a plain subsequence search over rows and a token position *is* a word ordinal.

**Files:**
- Create: `rytp/index/search.py`
- Modify: `rytp/constants.py` (append to the Part 4 section)
- Test: `tests/test_index_search.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.{InvalidInputError, NotFoundError, normalize_text, stem_text}`, `rytp.constants.{SEARCH_DEFAULT_LIMIT, SEARCH_MAX_LIMIT, CUTTABLE_SOURCE, CAPTION_WORD_FALLBACK_MS}`.
- Produces:
  - `MatchTier` (`EXACT = "exact"`, `STEM = "stem"`)
  - `SearchHit` frozen dataclass — `video_id, video_title, first_word_ord, last_word_ord, start_ms, end_ms, speaker, text, cuttable, tier, crosses_utterances`, plus an `anchor` property
  - `SearchResult` frozen dataclass — `tier: MatchTier | None`, `tokens: tuple[str, ...]`, `hits: tuple[SearchHit, ...]`
  - `AnchorSpan` frozen dataclass — `video_id, first_word_ord, last_word_ord, start_ms, end_ms, text, cuttable`
  - `anchor_for(video_id, first_word_ord, last_word_ord) -> str` — `"v12:340-341"`
  - `parse_anchor(anchor: str) -> tuple[int, int, int]`
  - `anchor_filename(anchor: str, suffix: str = ".wav") -> str` — Windows-safe
  - `query_tokens(query: str) -> tuple[str, ...]`, `tokens_for_tier(tokens, tier) -> tuple[str, ...]`, `fts_phrase(column, tokens) -> str`
  - `span_for_anchor(db, anchor: str) -> AnchorSpan`
  - `search(db, query, *, speaker_ids: frozenset[int] | None = None, cuttable_only=False, video_id=0, limit=C.SEARCH_DEFAULT_LIMIT) -> SearchResult` — takes **resolved** `video_speakers.id` values, never a label

- [ ] **Step 1: Write the failing tests**

Create `tests/test_index_search.py`. The first two tests are the reason this part exists; treat them as load-bearing.

```python
"""The backward index: phrase to places it was said (design §7)."""

from __future__ import annotations

import re

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.index.search import (
    MatchTier,
    anchor_filename,
    anchor_for,
    fts_phrase,
    parse_anchor,
    query_tokens,
    search,
    span_for_anchor,
)
from rytp.index.utterances import index_video
from rytp.models import InvalidInputError, NotFoundError, normalize_text, stem_text

from tests.test_index_utterances import make_speaker, make_video

NOW = "2026-09-21T00:00:00+00:00"


def stored_tokens(raw: str) -> list[tuple[str, str]]:
    """(surface, normalized) pairs, the way Part 3's `split_token` stores them.

    Contracts §4: a row holds exactly one token, and the tokens stored for
    a piece of text are exactly `normalize_text(text).split()`. So a
    hyphenated word is two rows. Seeding any other shape would test rows
    the real pipeline cannot produce.
    """
    pieces = re.split(r"[^\w]+", raw, flags=re.UNICODE)
    return [(piece, normalize_text(piece)) for piece in pieces if normalize_text(piece)]


def add_words(
    db: Database,
    video_id: int,
    text: str,
    *,
    start_ms: int = 0,
    first_ord: int = 0,
    speaker_id: int | None = None,
    source: str = "aligned",
    step_ms: int = 300,
    word_ms: int = 250,
) -> tuple[int, int]:
    """Append one words row per stored token. Returns (next_ord, next_start)."""
    ordinal = first_ord
    clock = start_ms
    for token, normalized in stored_tokens(text):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine, video_speaker_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'fake', ?)",
            (
                video_id,
                ordinal,
                clock,
                None if source == "caption" else clock + word_ms,
                token,
                normalized,
                stem_text(normalized),
                source,
                speaker_id,
            ),
        )
        ordinal += 1
        clock += step_ms
    return ordinal, clock


def corpus(db: Database, text: str, *, source: str = "aligned", title: str = "Sample") -> int:
    """One video holding `text`, indexed."""
    video_id = make_video(db, title)
    add_words(db, video_id, text, source=source)
    index_video(db, video_id)
    return video_id


def name_speaker(db: Database, video_speaker_id: int, label: str) -> int:
    """Attach a global roster entry to a per-video diarizer label."""
    speaker_id = int(
        db.conn.execute(
            "INSERT INTO speakers (label, created_at) VALUES (?, ?)", (label, NOW)
        ).lastrowid
    )
    db.conn.execute(
        "UPDATE video_speakers SET speaker_id = ? WHERE id = ?",
        (speaker_id, video_speaker_id),
    )
    return speaker_id


# --- the two regressions this part exists to fix ---------------------


def test_a_two_word_phrase_whose_words_are_adjacent_returns_a_hit(db: Database) -> None:
    """The exact failure measured on the owner's corpus.

    The replaced implementation shadowed `words` with FTS, one word per
    row, so a two-term query was an implicit AND inside a single row:
    `добрый*` returned 1 hit and `добрый* вечер*` returned 0 with the two
    words adjacent in the text. If this test ever goes back to zero, the
    index has regressed to the old shape.
    """
    corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "добрый").hits) == 1
    result = search(db, "добрый вечер")
    assert len(result.hits) == 1
    assert result.tier is MatchTier.EXACT


def test_a_russian_inflection_is_found_through_the_stem_tier(db: Database) -> None:
    """The second measured failure: the old tokenizer was `porter`, which
    is English-only, so `ощущения*` never reached the corpus's `ощущение`."""
    corpus(db, "Сегодня было странное ощущение")
    exact = search(db, "ощущение")
    assert exact.tier is MatchTier.EXACT
    stemmed = search(db, "ощущения")
    assert stemmed.tier is MatchTier.STEM
    assert len(stemmed.hits) == 1
    assert stemmed.hits[0].text == "Сегодня было странное ощущение"


# --- phrase semantics ------------------------------------------------


def test_words_that_are_present_but_not_adjacent_do_not_match(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    assert search(db, "добрый друзья").hits == ()


def test_a_phrase_absent_from_the_corpus_returns_nothing_and_no_tier(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    result = search(db, "совершенно другое")
    assert result.hits == ()
    assert result.tier is None


def test_the_stem_tier_is_only_consulted_when_the_exact_tier_is_empty(db: Database) -> None:
    """An exact hit must never be reported as an inflection."""
    corpus(db, "Они сказали слово")
    result = search(db, "сказали слово")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1


def test_yo_folding_works_in_both_directions(db: Database) -> None:
    """normalize_text folds ё→е (contracts §4), so neither spelling loses."""
    corpus(db, "И ещё раз", title="WithYo")
    assert len(search(db, "еще раз").hits) == 1
    assert len(search(db, "ещё раз").hits) == 1


def test_yo_folding_also_works_when_the_corpus_has_the_bare_e(db: Database) -> None:
    corpus(db, "И еще раз", title="WithoutYo")
    assert len(search(db, "ещё раз").hits) == 1


def test_a_hyphen_and_a_yo_in_the_same_query(db: Database) -> None:
    """The owner's case, and the one that falls between the other two.

    `кто-то` exercises contracts §4's one-row-per-token rule and `ещё`
    exercises the ё→е fold, and both normalizations have to agree on the
    query side and the corpus side at once. Four of the five spellings
    below differ from what is stored; all five must find the same span.
    """
    video_id = corpus(db, "Кто-то ещё об этом спрашивал")
    result = search(db, "кто-то ещё")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    assert result.hits[0].video_id == video_id
    assert (result.hits[0].first_word_ord, result.hits[0].last_word_ord) == (0, 2)
    for spelling in ("Кто-то ещё", "кто-то еще", "кто то ещё", "кто то еще"):
        assert len(search(db, spelling).hits) == 1, spelling


def test_fts_operators_inside_a_query_are_literal_text(db: Database) -> None:
    """Tokens go inside one quoted phrase, so OR/NEAR/AND are not operators."""
    corpus(db, "Может быть или нет")
    assert search(db, "может OR нет").hits == ()
    assert len(search(db, "или нет").hits) == 1


def test_punctuation_and_quotes_in_a_query_cannot_break_the_fts_expression(
    db: Database,
) -> None:
    corpus(db, "Может быть или нет")
    assert len(search(db, '"может" быть!').hits) == 1
    assert len(search(db, "может - быть").hits) == 1


def test_a_single_token_query_matches(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "друзья").hits) == 1


def test_an_empty_or_punctuation_only_query_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError):
        search(db, "")
    with pytest.raises(InvalidInputError):
        search(db, " ... ")


# --- what a hit carries ----------------------------------------------


def test_a_hit_locates_the_phrase_rather_than_the_whole_sentence(db: Database) -> None:
    """Playback and export use these timings; the sentence would be wrong."""
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    hit = search(db, "дорогие друзья").hits[0]
    assert hit.video_id == video_id
    assert (hit.first_word_ord, hit.last_word_ord) == (2, 3)
    assert hit.start_ms == 600
    assert hit.end_ms == 1150
    assert hit.text == "Добрый вечер дорогие друзья"


def test_a_hyphenated_word_is_two_rows_and_a_hit_spans_both(db: Database) -> None:
    """Contracts §4: `кто-то` is stored as two rows, not one row of two
    tokens, so the span covers both ordinals and its boundary is measured."""
    video_id = corpus(db, "Кто-то сказал")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 3
    hit = search(db, "кто то").hits[0]
    assert (hit.first_word_ord, hit.last_word_ord) == (0, 1)


def test_a_hit_carries_the_video_title_and_the_speaker(db: Database) -> None:
    video_id = make_video(db, "Evening")
    local = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, local, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    hit = search(db, "добрый вечер").hits[0]
    assert hit.video_title == "Evening"
    assert hit.speaker == "Ведущий"


def test_an_unmapped_diarizer_label_shows_as_the_raw_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert search(db, "добрый вечер").hits[0].speaker == "SPEAKER_00"


def test_an_undiarized_hit_has_no_speaker(db: Database) -> None:
    corpus(db, "Добрый вечер")
    assert search(db, "добрый вечер").hits[0].speaker is None


def test_the_anchor_round_trips(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    hit = search(db, "добрый вечер").hits[0]
    assert hit.anchor == f"v{video_id}:0-1"
    assert parse_anchor(hit.anchor) == (video_id, 0, 1)


# --- filters ---------------------------------------------------------


def test_cuttable_is_true_only_for_aligned_words(db: Database) -> None:
    """Contracts §3: cuttable is `source = 'aligned'` and nothing else."""
    corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    hits = search(db, "добрый вечер").hits
    assert sorted(hit.cuttable for hit in hits) == [False, True]


def test_cuttable_only_drops_the_caption_tier_hit(db: Database) -> None:
    aligned = corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    hits = search(db, "добрый вечер", cuttable_only=True).hits
    assert [hit.video_id for hit in hits] == [aligned]


def test_the_speaker_filter_restricts_to_the_resolved_ids(db: Database) -> None:
    """This layer takes `video_speakers.id` values, never a label: the two
    identifier spaces are resolved once, in `rytp.commands` (contracts §5)."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    name_speaker(db, host, "Ведущий")
    ordinal, clock = add_words(db, video_id, "Добрый вечер", speaker_id=host)
    add_words(
        db, video_id, "Добрый вечер", first_ord=ordinal, start_ms=clock, speaker_id=guest
    )
    index_video(db, video_id)
    assert len(search(db, "добрый вечер").hits) == 2
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host})).hits) == 1
    assert search(db, "добрый вечер", speaker_ids=frozenset({host})).hits[0].speaker == (
        "Ведущий"
    )
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({guest})).hits) == 1
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host, guest})).hits) == 2


def test_an_empty_speaker_set_matches_nothing_rather_than_everything(
    db: Database,
) -> None:
    """A resolver that found no labels for a real person must not widen
    the search — that is the silent-wrong-answer failure mode."""
    corpus(db, "Добрый вечер")
    assert search(db, "добрый вечер", speaker_ids=frozenset()).hits == ()
    assert search(db, "добрый вечер", speaker_ids=None).hits != ()


def test_the_video_filter_restricts_to_one_video(db: Database) -> None:
    first = corpus(db, "Добрый вечер", title="First")
    corpus(db, "Добрый вечер", title="Second")
    hits = search(db, "добрый вечер", video_id=first).hits
    assert [hit.video_id for hit in hits] == [first]


def test_the_limit_is_honoured_and_capped(db: Database) -> None:
    for index in range(5):
        corpus(db, "Добрый вечер", title=f"V{index}")
    assert len(search(db, "добрый вечер", limit=2).hits) == 2
    assert len(search(db, "добрый вечер", limit=C.SEARCH_MAX_LIMIT * 10).hits) == 5
    assert len(search(db, "добрый вечер", limit=0).hits) == 1


# --- helpers ---------------------------------------------------------


def test_query_tokens_normalizes_and_splits() -> None:
    assert query_tokens("  Добрый, ВЕЧЕР! ") == ("добрый", "вечер")


def test_fts_phrase_scopes_to_a_column_and_quotes_the_whole_run() -> None:
    assert fts_phrase("normalized_text", ("добрый", "вечер")) == (
        'normalized_text : "добрый вечер"'
    )


def test_anchor_helpers_are_windows_safe() -> None:
    anchor = anchor_for(12, 340, 341)
    assert anchor == "v12:340-341"
    assert ":" not in anchor_filename(anchor)
    assert anchor_filename(anchor) == "v12_340-341.wav"


def test_parse_anchor_rejects_rubbish() -> None:
    for bad in ("", "12:0-1", "v12:0", "vx:0-1", "v12:3-1"):
        with pytest.raises(InvalidInputError):
            parse_anchor(bad)


def test_span_for_anchor_returns_the_words_own_timings(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    span = span_for_anchor(db, anchor_for(video_id, 2, 3))
    assert (span.start_ms, span.end_ms) == (600, 1150)
    assert span.text == "дорогие друзья"
    assert span.cuttable is True


def test_span_for_anchor_raises_when_the_range_holds_no_words(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    with pytest.raises(NotFoundError):
        span_for_anchor(db, anchor_for(video_id, 900, 901))
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_index_search.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.index.search'`.

- [ ] **Step 3: Append the constants**

At the end of the Part 4 section of `rytp/constants.py`:

```python
#: words.source that may be cut. Contracts §3: "Cuttable is defined as
#: source = 'aligned'" and nothing else. Caption-tier words are
#: searchable and never assembled from (design §8).
CUTTABLE_SOURCE: Final = "aligned"

#: Rows a search returns when the caller does not say. Design §7 is a
#: human-facing lookup: a screenful, ranked, not a data dump.
SEARCH_DEFAULT_LIMIT: Final = 20

#: Ceiling on the same, so a typo in --limit cannot pull the corpus into
#: memory.
SEARCH_MAX_LIMIT: Final = 500
```

- [ ] **Step 4: Write `rytp/index/search.py`**

```python
"""Backward lookup: a word or a phrase, to every place it was said.

Design §7. Two tiers over the same two-column FTS shadow — the exact
`normalized_text` column first, the Python-stemmed `stem_text` column
second — and the result says which one answered, because an inflection
is not what the user typed.

The replaced implementation got this wrong twice, and both failures are
measured rather than theoretical. It shadowed the `words` table one word
per row, which makes a two-term query an implicit AND inside a single row
that can never match (`добрый*` → 1 hit, `добрый* вечер*` → 0, with the
words adjacent in the corpus). And it used the `porter` tokenizer, which
is English-only and therefore did no Russian stemming at all
(`ощущения*` missed `ощущение`). Utterances fix the first; the `stem`
columns and `unicode61` fix the second.

A query is always one phrase. Tokens are normalized and wrapped in a
single FTS `"..."`, so adjacency is enforced and a token that spells
`OR` or `NEAR` is text rather than an operator. There are no prefix
wildcards: the old code needed them because nothing stemmed.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, normalize_text, stem_text

__all__ = [
    "AnchorSpan",
    "MatchTier",
    "SearchHit",
    "SearchResult",
    "anchor_filename",
    "anchor_for",
    "fts_phrase",
    "parse_anchor",
    "query_tokens",
    "search",
    "span_for_anchor",
    "tokens_for_tier",
]


class MatchTier(str, Enum):
    """Which column answered. Reported, so a user knows an inflection."""

    EXACT = "exact"
    STEM = "stem"


#: (tier, utterances_fts column, words column). Ordered: exact first.
#: These column names are interpolated into SQL, so they must stay
#: literals from this tuple and never come from a caller.
_TIERS: Final = (
    (MatchTier.EXACT, "normalized_text", "normalized_text"),
    (MatchTier.STEM, "stem_text", "stem"),
)

_ANCHOR_RE: Final = re.compile(r"^v(\d+):(\d+)-(\d+)$")


@dataclass(frozen=True)
class SearchHit:
    """One place a phrase was said."""

    video_id: int
    video_title: str
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    speaker: str | None
    text: str
    cuttable: bool
    tier: MatchTier
    crosses_utterances: bool

    @property
    def anchor(self) -> str:
        """Stable handle for this span: video id plus word-ordinal range."""
        return anchor_for(self.video_id, self.first_word_ord, self.last_word_ord)


@dataclass(frozen=True)
class SearchResult:
    """Hits plus which tier produced them. `tier` is None when none did."""

    tier: MatchTier | None
    tokens: tuple[str, ...]
    hits: tuple[SearchHit, ...]


@dataclass(frozen=True)
class AnchorSpan:
    """What an anchor points at, resolved against the words table."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    cuttable: bool


# -- anchors ----------------------------------------------------------


def anchor_for(video_id: int, first_word_ord: int, last_word_ord: int) -> str:
    """The stable handle a search prints and a transcript block carries."""
    return f"v{video_id}:{first_word_ord}-{last_word_ord}"


def parse_anchor(anchor: str) -> tuple[int, int, int]:
    """(video_id, first_word_ord, last_word_ord) from `v12:340-341`."""
    match = _ANCHOR_RE.match(anchor.strip())
    if match is None:
        raise InvalidInputError(
            f"{anchor!r} is not an anchor; expected v<video>:<first>-<last>, e.g. v12:340-341"
        )
    video_id, first, last = (int(match[1]), int(match[2]), int(match[3]))
    if last < first:
        raise InvalidInputError(f"{anchor!r} ends before it starts")
    return video_id, first, last


def anchor_filename(anchor: str, suffix: str = ".wav") -> str:
    """An anchor as a filename. Windows has no ':' in a path component."""
    return anchor.replace(":", "_") + suffix


# -- queries ----------------------------------------------------------


def query_tokens(query: str) -> tuple[str, ...]:
    """Normalize a user query into the tokens the index stores."""
    tokens = tuple(normalize_text(query).split())
    if not tokens:
        raise InvalidInputError(f"nothing searchable in {query!r}")
    return tokens


def tokens_for_tier(tokens: Sequence[str], tier: MatchTier) -> tuple[str, ...]:
    """The same query, spelled for one tier's column."""
    if tier is MatchTier.EXACT:
        return tuple(tokens)
    return tuple(stem_text(" ".join(tokens)).split())


def fts_phrase(column: str, tokens: Sequence[str]) -> str:
    """One column-scoped FTS5 phrase query.

    Tokens have been through `normalize_text`, so they hold only word
    characters and cannot carry a quote or an operator; stripping `"` is
    belt and braces for a caller that hands over raw text.
    """
    phrase = " ".join(token.replace('"', "") for token in tokens)
    return f'{column} : "{phrase}"'


# -- locating a phrase inside a range of words ------------------------

_SELECT_RANGE: Final = (
    "SELECT ord, start_ms, end_ms, text, normalized_text, stem, source, video_speaker_id"
    " FROM words WHERE video_id = ? AND ord BETWEEN ? AND ? ORDER BY ord"
)


@dataclass(frozen=True)
class _WordRun:
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    cuttable: bool
    video_speaker_id: int | None


def _end_of(row: sqlite3.Row) -> int:
    """A word's end, inventing one for caption rows (contracts §3)."""
    if row["end_ms"] is not None:
        return int(row["end_ms"])
    return int(row["start_ms"]) + C.CAPTION_WORD_FALLBACK_MS


def _run_of(rows: Sequence[sqlite3.Row]) -> _WordRun:
    first, last = rows[0], rows[-1]
    return _WordRun(
        first_word_ord=int(first["ord"]),
        last_word_ord=int(last["ord"]),
        start_ms=int(first["start_ms"]),
        end_ms=max(_end_of(last), int(first["start_ms"])),
        text=" ".join(str(row["text"]) for row in rows),
        cuttable=all(str(row["source"]) == C.CUTTABLE_SOURCE for row in rows),
        video_speaker_id=(
            None if first["video_speaker_id"] is None else int(first["video_speaker_id"])
        ),
    )


def _locate(
    rows: Sequence[sqlite3.Row], tokens: Sequence[str], word_column: str
) -> _WordRun | None:
    """The word rows spanning the first occurrence of `tokens`.

    A plain subsequence search, because contracts §4 stores exactly one
    token per row: `кто-то` is two rows with a measured boundary between
    them, not one row holding two tokens.
    """
    wanted = list(tokens)
    width = len(wanted)
    if width == 0 or width > len(rows):
        return None
    for start in range(len(rows) - width + 1):
        window = rows[start : start + width]
        if [str(row[word_column]) for row in window] == wanted:
            return _run_of(window)
    return None


def _words_in(db: Database, video_id: int, lo: int, hi: int) -> list[sqlite3.Row]:
    return list(db.conn.execute(_SELECT_RANGE, (video_id, lo, hi)))


# -- the FTS tier -----------------------------------------------------

_CANDIDATE_COLUMNS: Final = (
    "u.video_id AS video_id, u.first_word_ord AS first_word_ord,"
    " u.last_word_ord AS last_word_ord, u.text AS text, v.title AS video_title,"
    " COALESCE(s.label, vs.local_label) AS speaker"
)

#: An utterance is cuttable only if every word in it is aligned.
_HAS_UNCUTTABLE_WORD: Final = (
    "EXISTS (SELECT 1 FROM words w WHERE w.video_id = u.video_id"
    " AND w.ord BETWEEN u.first_word_ord AND u.last_word_ord"
    f" AND w.source <> '{C.CUTTABLE_SOURCE}')"
)


def _fts_candidates(
    db: Database,
    expr: str,
    *,
    speaker_ids: frozenset[int] | None,
    cuttable_only: bool,
    video_id: int,
    limit: int,
) -> list[sqlite3.Row]:
    """Utterances matching one FTS expression, filtered in SQL.

    The filters are applied before LIMIT on purpose: filtering afterwards
    would silently return fewer rows than asked for.
    """
    sql = [
        f"SELECT {_CANDIDATE_COLUMNS}, bm25(utterances_fts) AS score",
        "FROM utterances_fts",
        "JOIN utterances u ON u.id = utterances_fts.rowid",
        "JOIN videos v ON v.id = u.video_id",
        "LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id",
        "LEFT JOIN speakers s ON s.id = vs.speaker_id",
        "WHERE utterances_fts MATCH ?",
    ]
    params: list[Any] = [expr]
    if video_id:
        sql.append("AND u.video_id = ?")
        params.append(video_id)
    if speaker_ids:
        # Resolved ids, never a label: contracts §5 keeps the two speaker
        # identifier spaces apart and resolves both in rytp.commands.
        # `search` has already returned early for an empty set, so the
        # IN list below is never empty.
        placeholders = ", ".join("?" * len(speaker_ids))
        sql.append(f"AND u.video_speaker_id IN ({placeholders})")
        params.extend(sorted(speaker_ids))
    if cuttable_only:
        sql.append(f"AND NOT {_HAS_UNCUTTABLE_WORD}")
    sql.append("ORDER BY score, u.video_id, u.start_ms LIMIT ?")
    params.append(limit)
    return list(db.conn.execute("\n".join(sql), params))


def _hit_from_candidate(
    db: Database,
    row: sqlite3.Row,
    tokens: Sequence[str],
    word_column: str,
    tier: MatchTier,
) -> SearchHit:
    rows = _words_in(
        db, int(row["video_id"]), int(row["first_word_ord"]), int(row["last_word_ord"])
    )
    # FTS matched, so the utterance holds the phrase. If the row scan
    # disagrees, the cause is a character unicode61 tokenizes on that
    # `normalize_text` keeps — `_` is the only one — so the row is one
    # token and the index saw two. Falling back to the whole utterance is
    # honest; dropping the hit would not be.
    run = _locate(rows, tokens, word_column) or _run_of(rows)
    return SearchHit(
        video_id=int(row["video_id"]),
        video_title=str(row["video_title"]),
        first_word_ord=run.first_word_ord,
        last_word_ord=run.last_word_ord,
        start_ms=run.start_ms,
        end_ms=run.end_ms,
        speaker=None if row["speaker"] is None else str(row["speaker"]),
        text=str(row["text"]),
        cuttable=run.cuttable,
        tier=tier,
        crosses_utterances=False,
    )


# -- the public lookup ------------------------------------------------


def search(
    db: Database,
    query: str,
    *,
    speaker_ids: frozenset[int] | None = None,
    cuttable_only: bool = False,
    video_id: int = 0,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
) -> SearchResult:
    """Every place a phrase was said. Exact column first, stems second.

    `speaker_ids` is a set of `video_speakers.id`, already resolved — this
    layer never sees a label. Contracts §5 gives the two speaker
    identifier spaces one shared resolver in `rytp.commands`, and keeping
    the resolution there is what stops `--speaker` meaning two different
    things in two commands. `None` means no speaker filter; an **empty**
    set means a filter that nothing can satisfy, which is not the same
    thing and must return no hits rather than everything.

    `video_id=0` means every video — the commands layer passes a scalar
    and 0 is not a row id.
    """
    base = query_tokens(query)
    if speaker_ids is not None and not speaker_ids:
        return SearchResult(tier=None, tokens=base, hits=())
    capped = max(1, min(int(limit), C.SEARCH_MAX_LIMIT))
    for tier, fts_column, word_column in _TIERS:
        tokens = tokens_for_tier(base, tier)
        hits = [
            _hit_from_candidate(db, row, tokens, word_column, tier)
            for row in _fts_candidates(
                db,
                fts_phrase(fts_column, tokens),
                speaker_ids=speaker_ids,
                cuttable_only=cuttable_only,
                video_id=video_id,
                limit=capped,
            )
        ]
        if hits:
            return SearchResult(tier=tier, tokens=tokens, hits=tuple(hits))
    return SearchResult(tier=None, tokens=base, hits=())


def span_for_anchor(db: Database, anchor: str) -> AnchorSpan:
    """Resolve `v12:340-341` against the words table."""
    video_id, first, last = parse_anchor(anchor)
    rows = _words_in(db, video_id, first, last)
    if not rows:
        raise NotFoundError(f"anchor {anchor} matches no words")
    run = _run_of(rows)
    return AnchorSpan(
        video_id=video_id,
        first_word_ord=run.first_word_ord,
        last_word_ord=run.last_word_ord,
        start_ms=run.start_ms,
        end_ms=run.end_ms,
        text=run.text,
        cuttable=run.cuttable,
    )
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_search.py -v`
Expected: PASS, 30 passed.

- [ ] **Step 6: Lint and type-check**

Run: `ruff check rytp/index tests/test_index_search.py`
Run: `mypy rytp/index`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
# from the repo root
git add rytp/index/search.py rytp/constants.py tests/test_index_search.py
git commit -m "feat(index): phrase search over utterances, exact column then stems"
```

---

### Task 5: Phrase search across utterance boundaries

An utterance boundary is a decision this code made, not a fact about the recording. If the silence threshold or the word cap happens to land between `добрый` and `вечер`, an FTS phrase query over utterances misses a place the phrase really was said — the same class of failure as the old one-word-per-row index, wearing different clothes. This task closes it.

**The rule, stated once, seen from two sides.** Utterance splitting and phrase bridging are the same rule: a split on *silence* or on a *cap* is an artifact and gets bridged; a split on a **speaker change** is real and never does. Two people each saying half of a phrase is not a place the phrase was said — it is a false positive, and a false positive is worse than a miss here, because the assembler would confidently cut audio that does not say what the index claims (design §3: "precision beats recall for assembly").

**The walk still requires an indexed video.** It bridges utterance *splits*; it does not stand in for the index. If no utterance covers a matched run — `index.drop` ran, or `index build` never did — the walk skips it, so an unindexed video is invisible to search whether the query is one token or five. Without that rule `index.drop` would silently leave the video findable with a degraded, context-free hit.

**Why the walk is a fallback and not the primary path.** Anchoring on `words.<column> = token` is an index lookup, but a common Russian function word matches hundreds of thousands of rows at full corpus. Running it only when that tier's FTS phrase came back empty means it runs for rare phrases, which is exactly when it is cheap. It also means no deduplication is needed: for a given tier, either the FTS hits or the walked hits are returned, never both.

**Strict tier order:** exact-FTS → exact-walk → stem-FTS → stem-walk. Reversing the middle two would report an inflection as the answer while an exact bridged occurrence existed, which is precisely the misreport the tier label exists to prevent.

**Known limitation, deliberately not solved.** If one video holds the phrase inside a single utterance and another holds it bridged, only the first is returned, because the exact-FTS tier was non-empty and the walk never ran. Step 1 has a test that pins this behaviour so nobody has to rediscover it.

**Reuse note for Part 5.** `walk_matches` is the same access path design §8 specifies for assembly matching — "find every occurrence of the first word, then walk forward comparing ordinals". Part 5 should build on this rather than write a second one.

**Files:**
- Modify: `rytp/index/search.py` (add the walk; add four lines to `search`)
- Modify: `rytp/constants.py` (append to the Part 4 section)
- Test: `tests/test_index_search.py` (append)

**Interfaces:**
- Consumes: Task 4's `_words_in`, `_run_of`, `_WordRun`, `MatchTier`, `SearchHit`, `_TIERS`; `rytp.constants.{SEARCH_WALK_ANCHOR_LIMIT, SEARCH_RARITY_PROBE_LIMIT}`.
- Produces: `rytp.index.search.rarest_token_index(db, tokens, word_column) -> int` and `walk_matches(db, tokens, word_column, tier, *, speaker_ids, cuttable_only, video_id, limit) -> list[SearchHit]`. `SearchHit.crosses_utterances` becomes meaningful.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index_search.py`, and add `rarest_token_index` to the `rytp.index.search` import list at the top of the file:

```python
# --- phrases that cross an utterance boundary ------------------------


def split_corpus(db: Database, title: str = "Split") -> int:
    """A video where `добрый` and `вечер` land in different utterances.

    The split is a silence longer than the threshold, which is an artifact
    of how this code chose to cut the transcript, not a fact about the
    recording — so the phrase must still be findable.
    """
    video_id = make_video(db, title)
    ordinal, clock = add_words(db, video_id, "Добрый")
    add_words(
        db,
        video_id,
        "вечер друзья",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    return video_id


def test_the_split_really_did_produce_two_utterances(db: Database) -> None:
    """Guard for the tests below: if this is one row they prove nothing."""
    video_id = split_corpus(db)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2


def test_a_phrase_split_by_a_silence_is_still_found(db: Database) -> None:
    video_id = split_corpus(db)
    result = search(db, "добрый вечер")
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.video_id == video_id
    assert (hit.first_word_ord, hit.last_word_ord) == (0, 1)
    assert hit.crosses_utterances is True
    assert "Добрый" in hit.text
    assert "вечер" in hit.text


def test_a_phrase_split_by_a_speaker_change_is_not_a_hit(db: Database) -> None:
    """Two people each saying half of it is not a place it was said."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    guest = make_speaker(db, video_id, "SPEAKER_01")
    ordinal, clock = add_words(db, video_id, "Добрый", speaker_id=host)
    add_words(db, video_id, "вечер", first_ord=ordinal, start_ms=clock, speaker_id=guest)
    index_video(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2
    assert search(db, "добрый вечер").hits == ()


def test_the_walk_never_bridges_two_videos(db: Database) -> None:
    corpus(db, "Добрый", title="Ends")
    corpus(db, "вечер", title="Begins")
    assert search(db, "добрый вечер").hits == ()


def test_a_hit_inside_one_utterance_does_not_cross(db: Database) -> None:
    corpus(db, "Добрый вечер")
    hit = search(db, "добрый вечер").hits[0]
    assert hit.crosses_utterances is False


def test_the_walk_does_not_run_when_the_fts_tier_answered(db: Database) -> None:
    """Known limitation, pinned: an in-utterance hit suppresses the walk, so
    a bridged occurrence elsewhere is not reported alongside it."""
    inline = corpus(db, "Добрый вечер", title="Inline")
    split_corpus(db, title="Split")
    hits = search(db, "добрый вечер").hits
    assert [hit.video_id for hit in hits] == [inline]


def test_a_bridged_phrase_is_reachable_through_the_stem_tier(db: Database) -> None:
    video_id = make_video(db, "SplitInflected")
    ordinal, clock = add_words(db, video_id, "странное")
    add_words(
        db,
        video_id,
        "ощущение",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    result = search(db, "странные ощущения")
    assert result.tier is MatchTier.STEM
    assert len(result.hits) == 1
    assert result.hits[0].video_id == video_id
    assert result.hits[0].crosses_utterances is True


def test_a_bridged_hit_honours_the_cuttable_filter(db: Database) -> None:
    video_id = make_video(db, "SplitCaptions")
    ordinal, clock = add_words(db, video_id, "Добрый", source="caption")
    # A caption word's implied end is capped at CAPTION_WORD_FALLBACK_MS
    # after its start, so the silence has to clear both to split.
    add_words(
        db,
        video_id,
        "вечер",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS + C.CAPTION_WORD_FALLBACK_MS,
        source="caption",
    )
    index_video(db, video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances WHERE video_id = ?", (video_id,)
    ).fetchone()[0] == 2
    assert len(search(db, "добрый вечер").hits) == 1
    assert search(db, "добрый вечер").hits[0].cuttable is False
    assert search(db, "добрый вечер", cuttable_only=True).hits == ()


def test_a_bridged_hit_honours_the_speaker_filter(db: Database) -> None:
    video_id = make_video(db, "SplitOneSpeaker")
    host = make_speaker(db, video_id, "SPEAKER_00")
    other = make_speaker(db, video_id, "SPEAKER_01")
    ordinal, clock = add_words(db, video_id, "Добрый", speaker_id=host)
    add_words(
        db,
        video_id,
        "вечер",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
        speaker_id=host,
    )
    index_video(db, video_id)
    assert len(search(db, "добрый вечер", speaker_ids=frozenset({host})).hits) == 1
    assert search(db, "добрый вечер", speaker_ids=frozenset({other})).hits == ()


def test_a_bridged_hit_honours_the_video_and_limit_filters(db: Database) -> None:
    first = split_corpus(db, title="SplitOne")
    split_corpus(db, title="SplitTwo")
    assert len(search(db, "добрый вечер").hits) == 2
    assert [h.video_id for h in search(db, "добрый вечер", video_id=first).hits] == [first]
    assert len(search(db, "добрый вечер", limit=1).hits) == 1


def test_the_walk_does_not_answer_for_an_unindexed_video(db: Database) -> None:
    """The walk bridges splits; it is not a substitute for the index.

    Without this, `index.drop` would leave a video findable through the
    walk alone — a hit with no surrounding sentence — and whether a
    result had context would depend on whether anyone had indexed it.
    """
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    assert len(search(db, "добрый вечер").hits) == 1
    db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    assert search(db, "добрый вечер").hits == ()
    assert search(db, "друзья").hits == ()
    index_video(db, video_id)
    assert len(search(db, "добрый вечер").hits) == 1


def test_the_walk_anchors_on_the_rarest_token(db: Database) -> None:
    """A common function word would make the walk scan the whole corpus."""
    for index in range(5):
        corpus(db, "и что то ещё", title=f"Common{index}")
    corpus(db, "и редкое слово", title="Rare")
    assert rarest_token_index(db, ("и", "редкое"), "normalized_text") == 1
    assert rarest_token_index(db, ("редкое", "и"), "normalized_text") == 0


def test_rarest_token_index_stops_at_a_token_that_is_absent(db: Database) -> None:
    corpus(db, "Добрый вечер")
    assert rarest_token_index(db, ("добрый", "отсутствует"), "normalized_text") == 1
```

- [ ] **Step 2: Run the new tests and watch them fail**

Run: `python -m pytest tests/test_index_search.py -v`
Expected: a collection error on `rarest_token_index`. Temporarily drop that name from the import line and re-run to see the behaviour failure that matters — `test_a_phrase_split_by_a_silence_is_still_found` fails on `assert 0 == 1`, which is the boundary miss. Restore the import before Step 3.

- [ ] **Step 3: Append the constants**

At the end of the Part 4 section of `rytp/constants.py`:

```python
#: How many anchor rows the cross-boundary walk will look at before it
#: gives up. The walk only runs when a tier's FTS phrase found nothing,
#: which means the phrase is rare, so this is a guard against a
#: pathological anchor rather than a routine limit (design §7).
SEARCH_WALK_ANCHOR_LIMIT: Final = 2_000

#: The walk anchors on the query's rarest token, and counting occurrences
#: stops here: the ranking only needs to tell "hundreds" from "hundreds
#: of thousands", and at full corpus a common word is ~200k rows.
SEARCH_RARITY_PROBE_LIMIT: Final = 5_000
```

- [ ] **Step 4: Add the walk to `rytp/index/search.py`**

Append after `_hit_from_candidate`, before `search`:

```python
# -- the cross-boundary walk ------------------------------------------


def rarest_token_index(db: Database, tokens: Sequence[str], word_column: str) -> int:
    """Which token to anchor the walk on: the one with fewest occurrences.

    Counting is capped at ``SEARCH_RARITY_PROBE_LIMIT`` by a subquery with
    its own LIMIT, because the ranking only needs to distinguish "rare"
    from "everywhere" and a full count of a Russian function word would
    scan hundreds of thousands of index entries.

    ``word_column`` comes from ``_TIERS`` and is never caller-supplied,
    which is what makes the interpolation below safe.
    """
    best_index = 0
    best_count: int | None = None
    for index, token in enumerate(tokens):
        count = int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM words WHERE {word_column} = ? LIMIT ?)",
                (token, C.SEARCH_RARITY_PROBE_LIMIT),
            ).fetchone()[0]
        )
        if best_count is None or count < best_count:
            best_index, best_count = index, count
        if count == 0:
            break
    return best_index


def _row_run(
    rows: Sequence[sqlite3.Row], tokens: Sequence[str], word_column: str
) -> _WordRun | None:
    """One words row per query token, in order, from one speaker.

    Contracts §4 stores one token per row, so this is the same row-wise
    comparison :func:`_locate` makes; the difference is that the window
    is fixed by the anchor's ordinal rather than searched for, and that a
    partial window is a miss rather than a shorter match.

    The single-speaker requirement is the rule that makes bridging safe.
    A split on silence or on a length cap is an artifact of how this code
    cut the transcript and gets bridged; a split on a speaker change is a
    fact about the recording and never does.
    """
    if len(rows) != len(tokens):
        return None
    if any(
        str(row[word_column]) != token for row, token in zip(rows, tokens, strict=True)
    ):
        return None
    if len({row["video_speaker_id"] for row in rows}) != 1:
        return None
    return _run_of(rows)


_CONTEXT_SQL: Final = (
    "SELECT u.text AS text, COALESCE(s.label, vs.local_label) AS speaker"
    " FROM utterances u"
    " LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id"
    " LEFT JOIN speakers s ON s.id = vs.speaker_id"
    " WHERE u.video_id = ? AND u.last_word_ord >= ? AND u.first_word_ord <= ?"
    " ORDER BY u.first_word_ord"
)


def _context(db: Database, video_id: int, lo: int, hi: int) -> tuple[str, str | None, bool]:
    """The surrounding sentence(s), the speaker, and whether it spans rows."""
    rows = list(db.conn.execute(_CONTEXT_SQL, (video_id, lo, hi)))
    if not rows:
        return "", None, False
    text = " ".join(str(row["text"]) for row in rows)
    speaker = None if rows[0]["speaker"] is None else str(rows[0]["speaker"])
    return text, speaker, len(rows) > 1


def walk_matches(
    db: Database,
    tokens: Sequence[str],
    word_column: str,
    tier: MatchTier,
    *,
    speaker_ids: frozenset[int] | None,
    cuttable_only: bool,
    video_id: int,
    limit: int,
) -> list[SearchHit]:
    """Occurrences an utterance split hid from the FTS phrase query.

    Anchors on the rarest token through ``words(normalized_text)`` /
    ``words(stem)`` — design §8's assembly access path — then walks by
    ordinal, which nothing about utterance boundaries constrains.
    """
    anchor_index = rarest_token_index(db, tokens, word_column)
    sql = [f"SELECT video_id, ord FROM words WHERE {word_column} = ?"]
    params: list[Any] = [tokens[anchor_index]]
    if video_id:
        sql.append("AND video_id = ?")
        params.append(video_id)
    sql.append("ORDER BY video_id, ord LIMIT ?")
    params.append(C.SEARCH_WALK_ANCHOR_LIMIT)

    titles: dict[int, str] = {}
    hits: list[SearchHit] = []
    for anchor in db.conn.execute("\n".join(sql), params):
        found = int(anchor["video_id"])
        lo = int(anchor["ord"]) - anchor_index
        if lo < 0:
            continue
        run = _row_run(_words_in(db, found, lo, lo + len(tokens) - 1), tokens, word_column)
        if run is None:
            continue
        if cuttable_only and not run.cuttable:
            continue
        # `_row_run` already required one speaker across the whole run, so
        # the run's id is the run's speaker.
        if speaker_ids is not None and run.video_speaker_id not in speaker_ids:
            continue
        text, found_speaker, crosses = _context(
            db, found, run.first_word_ord, run.last_word_ord
        )
        if not text:
            # The words exist but no utterance covers them, so this video
            # is not indexed — `index.drop` ran, or `index build` has not.
            # The walk bridges utterance *splits*; it is not a substitute
            # for the index, and answering here would make `index.drop` a
            # lie and make result quality depend on whether a video
            # happened to be indexed.
            continue
        if found not in titles:
            row = db.conn.execute("SELECT title FROM videos WHERE id = ?", (found,)).fetchone()
            titles[found] = "" if row is None else str(row["title"])
        hits.append(
            SearchHit(
                video_id=found,
                video_title=titles[found],
                first_word_ord=run.first_word_ord,
                last_word_ord=run.last_word_ord,
                start_ms=run.start_ms,
                end_ms=run.end_ms,
                speaker=found_speaker,
                text=text or run.text,
                cuttable=run.cuttable,
                tier=tier,
                crosses_utterances=crosses,
            )
        )
        if len(hits) >= limit:
            break
    return hits
```

The speaker filter is applied in Python here rather than in SQL, and that is correct for this path: the walk is already bounded by `SEARCH_WALK_ANCHOR_LIMIT` anchors, and `limit` is applied after filtering, so a filtered-out anchor cannot eat a result slot. It compares `video_speakers.id` values, not labels — the run is single-speaker by construction, and resolution happened once in `rytp.commands`.

- [ ] **Step 5: Wire the walk into `search`**

In `search`, inside the tier loop, between building `hits` and the `if hits:` return:

```python
        if not hits and len(tokens) > 1:
            # An utterance boundary is this code's decision, not a fact
            # about the recording, so a split must not hide a phrase.
            # Fallback-only: for a given tier either the FTS hits or the
            # walked hits are returned, never both, so nothing is deduped.
            hits = walk_matches(
                db,
                tokens,
                word_column,
                tier,
                speaker_ids=speaker_ids,
                cuttable_only=cuttable_only,
                video_id=video_id,
                limit=capped,
            )
        if hits:
            return SearchResult(tier=tier, tokens=tokens, hits=tuple(hits))
```

Add `"rarest_token_index"` and `"walk_matches"` to `__all__`, and add a closing paragraph to the module docstring:

```python
Tier order is strict: the exact phrase over `normalized_text`, then the
same phrase walked across utterance boundaries, then the stemmed phrase
over `stem_text`, then that walked. Anything else would report an
inflection while an exact occurrence existed.
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_search.py -v`
Expected: PASS, 43 passed.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check rytp/index tests/test_index_search.py`
Run: `mypy rytp/index`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
# from the repo root
git add rytp/index/search.py rytp/constants.py tests/test_index_search.py
git commit -m "feat(index): bridge utterance splits so a phrase boundary cannot hide a hit"
```

---

### Task 6: Playing a hit and exporting it as audio

Design §7: *"Playback: a search hit can be played directly from the TUI, and an export command writes hits out as audio files. Both use ffplay/ffmpeg, which are already required — no audio library to bundle."*

**The seam is the design.** Everything that decides *what* to run is a pure function returning `list[str]`, and exactly one function actually runs it. Tests assert on the list; an autouse fixture replaces the runner with a recorder so **no test can spawn a player even by accident**, and the one test that really invokes ffmpeg opts back in with a marker and is skipped when ffmpeg is absent (it is absent on the development Mac, so that test will skip there — that is expected, not a failure).

Two ffmpeg details that are easy to get wrong and are pinned by tests: `-ss` goes **before** `-i` (input-side seek, which is both fast and accurate in current ffmpeg), and the duration flag is `-t` (a length) rather than `-to` (an absolute output timestamp, which means something different once `-ss` has moved the origin).

**Where the audio comes from.** `cache/wav/{video_id}.wav` if it is on disk — it is 16 kHz mono PCM and already the single audio truth for everything downstream — otherwise the `audio` asset, otherwise the `container` asset. Contracts §7 makes the WAV a prunable cache, so falling back rather than demanding it is what keeps playback working after a `cache prune`.

**Files:**
- Create: `rytp/index/export.py`
- Modify: `rytp/constants.py` (append to the Part 4 section)
- Modify: `pyproject.toml` (register one pytest marker)
- Test: `tests/test_index_export.py`

**Interfaces:**
- Consumes: `rytp.config.{paths, ensure_dir}`, `rytp.db.queries.asset_for`, `rytp.models.RytpError`, `rytp.constants.{AUDIO_CHANNELS, AUDIO_SAMPLE_RATE_HZ, MS_PER_SECOND, FFMPEG_ERROR_TAIL_CHARS, CLIP_PAD_MS}`.
- Produces:
  - `MediaToolMissing(RytpError)`, `ClipError(RytpError)`
  - `clip_source(db, video_id) -> Path`
  - `padded(start_ms, end_ms, pad_ms) -> tuple[int, int]`
  - `build_clip_command(ffmpeg, source, out, start_ms, end_ms) -> list[str]`
  - `build_play_command(ffplay, source, start_ms, end_ms) -> list[str]`
  - `export_clip(db, video_id, start_ms, end_ms, out, *, pad_ms=C.CLIP_PAD_MS) -> Path`
  - `play_clip(db, video_id, start_ms, end_ms, *, pad_ms=C.CLIP_PAD_MS) -> None`
  - the three monkeypatchable seams `_ffmpeg_binary()`, `_ffplay_binary()`, `_run(cmd)`

- [ ] **Step 1: Register the pytest marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add (or extend) the marker list — without it pytest warns on every run:

```toml
markers = [
    "real_ffmpeg: invokes the real ffmpeg binary; skipped when it is not installed",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_index_export.py`:

```python
"""Playback and audio export (design §7). No test spawns a player."""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.config import paths
from rytp.db import Database
from rytp.db.queries import insert_asset
from rytp.index import export
from rytp.index.export import (
    ClipError,
    MediaToolMissing,
    build_clip_command,
    build_play_command,
    clip_source,
    export_clip,
    padded,
    play_clip,
)

from tests.test_index_utterances import make_video


@pytest.fixture(autouse=True)
def never_spawn_a_player(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> list[list[str]]:
    """Replace the one function that runs a subprocess with a recorder.

    Autouse and opt-out rather than opt-in: a new test that forgets to
    patch the seam records a command instead of opening a window.
    """
    recorded: list[list[str]] = []
    if request.node.get_closest_marker("real_ffmpeg") is not None:
        return recorded

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        recorded.append(list(cmd))
        if cmd[-1].endswith(".wav"):
            # Stand in for the file ffmpeg would have written, so callers
            # that check for the output still see it.
            Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def write_silence(path: Path, seconds: float = 1.0) -> Path:
    """A real 16 kHz mono PCM WAV, using nothing but the stdlib."""
    frames = int(C.AUDIO_SAMPLE_RATE_HZ * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * frames)
    return path


def cached_wav(video_id: int) -> Path:
    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_silence(path)


# --- command shape ---------------------------------------------------


def test_the_clip_command_seeks_on_the_input_side() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 1000, 1500)
    assert cmd.index("-ss") < cmd.index("-i")


def test_the_clip_command_uses_a_duration_not_an_end_timestamp() -> None:
    """`-to` means an absolute output time, which is not what we want once
    `-ss` has moved the origin."""
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 1000, 1500)
    assert "-to" not in cmd
    assert cmd[cmd.index("-ss") + 1] == "1.000"
    assert cmd[cmd.index("-t") + 1] == "0.500"


def test_the_clip_command_writes_the_project_audio_format() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 0, 1000)
    assert cmd[cmd.index("-ac") + 1] == str(C.AUDIO_CHANNELS)
    assert cmd[cmd.index("-ar") + 1] == str(C.AUDIO_SAMPLE_RATE_HZ)
    assert cmd[-1] == "out.wav"
    assert "-vn" in cmd


def test_the_play_command_is_headless_and_exits_on_its_own() -> None:
    cmd = build_play_command("ffplay", Path("in.wav"), 2000, 2400)
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert cmd[cmd.index("-ss") + 1] == "2.000"
    assert cmd[cmd.index("-t") + 1] == "0.400"


def test_a_zero_length_span_still_asks_for_a_positive_duration() -> None:
    cmd = build_clip_command("ffmpeg", Path("in.wav"), Path("out.wav"), 500, 500)
    assert float(cmd[cmd.index("-t") + 1]) > 0


def test_padding_widens_the_span_but_never_seeks_before_zero() -> None:
    assert padded(1000, 2000, 150) == (850, 2150)
    assert padded(50, 200, 150) == (0, 350)
    assert padded(1000, 2000, 0) == (1000, 2000)


# --- where the audio comes from --------------------------------------


def test_clip_source_prefers_the_cached_wav(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    wav = cached_wav(video_id)
    insert_asset(db, video_id=video_id, role="audio", path=str(data_dir / "other.m4a"))
    assert clip_source(db, video_id) == wav


def test_clip_source_falls_back_to_the_audio_asset(db: Database, data_dir: Path) -> None:
    """Contracts §7 makes the WAV a prunable cache; a prune must not break
    playback."""
    video_id = make_video(db)
    audio = write_silence(data_dir / "audio.wav")
    insert_asset(db, video_id=video_id, role="audio", path=str(audio))
    assert clip_source(db, video_id) == audio


def test_clip_source_falls_back_to_a_container(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    container = write_silence(data_dir / "whole.wav")
    insert_asset(db, video_id=video_id, role="container", path=str(container))
    assert clip_source(db, video_id) == container


def test_clip_source_raises_when_nothing_is_on_disk(db: Database) -> None:
    video_id = make_video(db)
    with pytest.raises(ClipError, match="no audio"):
        clip_source(db, video_id)


# --- running it ------------------------------------------------------


def test_export_clip_runs_ffmpeg_and_returns_the_path(
    db: Database, data_dir: Path, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "clips" / "hit.wav"
    assert export_clip(db, video_id, 1000, 1500, out) == out
    assert len(never_spawn_a_player) == 1
    assert never_spawn_a_player[0][0] == "/usr/bin/ffmpeg"


def test_export_clip_creates_the_output_directory(db: Database, data_dir: Path) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "deep" / "nested" / "hit.wav"
    export_clip(db, video_id, 0, 500, out)
    assert out.parent.is_dir()


def test_export_clip_applies_the_padding(
    db: Database, data_dir: Path, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    export_clip(db, video_id, 1000, 1500, data_dir / "hit.wav", pad_ms=200)
    cmd = never_spawn_a_player[0]
    assert cmd[cmd.index("-ss") + 1] == "0.800"
    assert cmd[cmd.index("-t") + 1] == "0.900"


def test_export_clip_reports_a_missing_binary_in_one_line(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: None)
    with pytest.raises(MediaToolMissing, match="ffmpeg"):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_export_clip_reports_an_ffmpeg_failure(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)

    def failing(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "Invalid data found")

    monkeypatch.setattr(export, "_run", failing)
    with pytest.raises(ClipError, match="Invalid data found"):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_export_clip_maps_a_vanished_binary_to_a_domain_error(
    db: Database, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """which() said yes, exec() said no. One line, not a traceback."""
    video_id = make_video(db)
    cached_wav(video_id)

    def gone(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(export, "_run", gone)
    with pytest.raises(MediaToolMissing):
        export_clip(db, video_id, 0, 500, data_dir / "hit.wav")


def test_play_clip_invokes_ffplay_with_the_span(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    play_clip(db, video_id, 2000, 2400, pad_ms=0)
    assert len(never_spawn_a_player) == 1
    cmd = never_spawn_a_player[0]
    assert cmd[0] == "/usr/bin/ffplay"
    assert cmd[cmd.index("-t") + 1] == "0.400"


def test_play_clip_reports_a_missing_ffplay(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    with pytest.raises(MediaToolMissing, match="ffplay"):
        play_clip(db, video_id, 0, 500)


@pytest.mark.real_ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_real_export_produces_a_clip_of_the_right_length(
    db: Database, data_dir: Path
) -> None:
    video_id = make_video(db)
    cached_wav(video_id)
    out = data_dir / "real.wav"
    export_clip(db, video_id, 200, 400, out, pad_ms=0)
    with wave.open(str(out), "rb") as handle:
        seconds = handle.getnframes() / handle.getframerate()
    assert 0.15 < seconds < 0.25
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `python -m pytest tests/test_index_export.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.index.export'`.

- [ ] **Step 4: Append the constants**

At the end of the Part 4 section of `rytp/constants.py`:

```python
#: Padding added on each side of an exported or played clip, so a word is
#: not clipped by a boundary that is a few milliseconds optimistic
#: (design §7 "Playback"). Zero disables it.
CLIP_PAD_MS: Final = 150
```

- [ ] **Step 5: Write the audio half of `rytp/index/export.py`**

```python
"""Getting index rows back out: audio clips, playback, and transcripts.

Design §7. Two kinds of export, one module, because both answer the same
question — "render a row of the index as something outside the database".

Everything that decides *what* to run is a pure function returning a
`list[str]`; exactly one function runs it. That is what lets the whole
test suite assert on command shape without ever opening a player, and it
is the same seam pattern Part 2 uses for the WAV cache.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import asset_for
from rytp.models import RytpError

__all__ = [
    "ClipError",
    "MediaToolMissing",
    "build_clip_command",
    "build_play_command",
    "clip_source",
    "export_clip",
    "padded",
    "play_clip",
]


class MediaToolMissing(RytpError):
    """ffmpeg or ffplay is not on PATH."""


class ClipError(RytpError):
    """There was nothing to cut from, or ffmpeg refused to cut it."""


def _ffmpeg_binary() -> str | None:
    """Where ffmpeg is. A module attribute so a test can replace just this."""
    return shutil.which("ffmpeg")


def _ffplay_binary() -> str | None:
    """Where ffplay is. Same reason."""
    return shutil.which("ffplay")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """The only subprocess call in this module. Tests replace it."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def clip_source(db: Database, video_id: int) -> Path:
    """The audio to cut from.

    The cached 16 kHz mono WAV first, because it is already the audio
    every other stage works from. Contracts §7 makes that cache prunable,
    so the asset roles are the fallback and `cache prune` cannot break
    playback.
    """
    cached = config.paths().cache_wav(video_id)
    if cached.exists():
        return cached
    for role in ("audio", "container"):
        row = asset_for(db, video_id, role)
        if row is not None and Path(str(row["path"])).exists():
            return Path(str(row["path"]))
    raise ClipError(
        f"video {video_id}: no cached WAV and no audio on disk; "
        "run `rytp ingest` for it first"
    )


def padded(start_ms: int, end_ms: int, pad_ms: int) -> tuple[int, int]:
    """Widen a span by `pad_ms` on each side, never before the recording."""
    return max(0, start_ms - pad_ms), end_ms + pad_ms


def _seconds(ms: int) -> str:
    """Milliseconds as the seconds string ffmpeg takes."""
    return f"{ms / C.MS_PER_SECOND:.3f}"


def _duration(start_ms: int, end_ms: int) -> str:
    """Span length, never zero — ffmpeg would write an empty file."""
    return _seconds(max(end_ms - start_ms, 1))


def build_clip_command(
    ffmpeg: str, source: Path, out: Path, start_ms: int, end_ms: int
) -> list[str]:
    """Cut one span to a WAV.

    `-ss` goes before `-i`: input-side seek, which current ffmpeg does
    both quickly and accurately. The length is `-t`, not `-to`, because
    `-to` is an absolute output timestamp and the origin has moved.
    """
    return [
        ffmpeg,
        "-nostdin",
        "-y",
        "-ss",
        _seconds(start_ms),
        "-t",
        _duration(start_ms, end_ms),
        "-i",
        str(source),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        str(C.AUDIO_CHANNELS),
        "-ar",
        str(C.AUDIO_SAMPLE_RATE_HZ),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(out),
    ]


def build_play_command(ffplay: str, source: Path, start_ms: int, end_ms: int) -> list[str]:
    """Play one span: no window, no banner, quits when the span ends."""
    return [
        ffplay,
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-ss",
        _seconds(start_ms),
        "-t",
        _duration(start_ms, end_ms),
        str(source),
    ]


def _invoke(cmd: list[str], tool: str) -> subprocess.CompletedProcess[str]:
    try:
        return _run(cmd)
    except FileNotFoundError as exc:
        # which() found it and exec() did not: a moved or half-installed
        # binary. One line, not a traceback (contracts §8).
        raise MediaToolMissing(f"{tool} could not be run: {exc}") from exc


def export_clip(
    db: Database,
    video_id: int,
    start_ms: int,
    end_ms: int,
    out: Path,
    *,
    pad_ms: int = C.CLIP_PAD_MS,
) -> Path:
    """Write one span to `out` as a WAV. Returns the path."""
    source = clip_source(db, video_id)
    ffmpeg = _ffmpeg_binary()
    if ffmpeg is None:
        raise MediaToolMissing(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/ "
            "(on Windows, unzip the release and add its bin/ directory to PATH)"
        )
    config.ensure_dir(out.parent)
    lo, hi = padded(start_ms, end_ms, pad_ms)
    result = _invoke(build_clip_command(ffmpeg, source, out, lo, hi), "ffmpeg")
    if result.returncode != 0:
        raise ClipError(
            f"ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )
    return out


def play_clip(
    db: Database,
    video_id: int,
    start_ms: int,
    end_ms: int,
    *,
    pad_ms: int = C.CLIP_PAD_MS,
) -> None:
    """Play one span through ffplay. Blocks until it finishes."""
    source = clip_source(db, video_id)
    ffplay = _ffplay_binary()
    if ffplay is None:
        raise MediaToolMissing(
            "ffplay not found on PATH. It ships with ffmpeg; install that."
        )
    lo, hi = padded(start_ms, end_ms, pad_ms)
    result = _invoke(build_play_command(ffplay, source, lo, hi), "ffplay")
    if result.returncode != 0:
        raise ClipError(
            f"ffplay failed (rc={result.returncode}): "
            f"{result.stderr[-C.FFMPEG_ERROR_TAIL_CHARS:]}"
        )
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_export.py -v`
Expected: PASS, 18 passed — plus 1 skipped where ffmpeg is not installed (it is not, on the development Mac). If ffmpeg *is* installed, 19 passed.

- [ ] **Step 7: Prove no test reached a real player**

Run: `python -m pytest tests/test_index_export.py -q -m "not real_ffmpeg" -p no:randomly`
Expected: PASS, and the run finishes in well under a second — a real `ffplay` would block for the length of the clip.

- [ ] **Step 8: Lint and type-check**

Run: `ruff check rytp/index tests/test_index_export.py`
Run: `mypy rytp/index`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
# from the repo root
git add rytp/index/export.py rytp/constants.py pyproject.toml tests/test_index_export.py
git commit -m "feat(index): play and export a hit through a seam tests cannot spawn"
```

---

### Task 7: Markdown transcripts an agent can point back into the database from

Design §7: *"Markdown transcripts are regenerable output at `data/transcripts/{video_id}.md`, safe to delete. Each block carries a stable anchor — video id plus word ordinal range — so an agent reading the file can point back into the database precisely. This plus existing off-the-shelf SQLite tooling is the whole AI-access story; no custom server."*

**The blocks are the utterances.** Not a second grouping pass with its own rules — the same rows the FTS index holds, so the anchor printed in a transcript and the anchor printed by `search words` are the same string for the same span, by construction rather than by coincidence. That equality is what makes the file useful to an agent: read a block, take its anchor, run `rytp search play` on it or query `words` for the ordinal range.

**Regenerable, never input** (design §3). Rebuilding overwrites; the file is never read back by any stage.

`transcript build` refuses rather than silently indexing: a transcript of a video that was never indexed would be an empty file that looks like an answer, and the fix is one command away.

**Files:**
- Modify: `rytp/index/export.py` (append the transcript half)
- Test: `tests/test_index_export.py` (append)

**Interfaces:**
- Consumes: Task 4's `anchor_for`, `rytp.config.{paths, ensure_dir}`, `rytp.models.{NotFoundError, RytpError, utc_now_iso}`, `rytp.constants.{NULL_CELL, MS_PER_SECOND}`.
- Produces:
  - `timestamp(ms: int) -> str` — `"00:01:02.345"`
  - `TranscriptBlock` frozen dataclass — `anchor, start_ms, end_ms, speaker, text`
  - `transcript_blocks(db, video_id) -> list[TranscriptBlock]`
  - `render_transcript(db, video_id) -> str`
  - `write_transcript(db, video_id) -> Path`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index_export.py`, adding these to its imports:

```python
from rytp.index.export import (
    render_transcript,
    timestamp,
    transcript_blocks,
    write_transcript,
)
from rytp.index.search import span_for_anchor
from rytp.index.utterances import index_video
from rytp.models import NotFoundError, RytpError

from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker
```

and the tests:

```python
# --- markdown transcripts --------------------------------------------


def test_timestamps_are_hours_minutes_seconds_milliseconds() -> None:
    assert timestamp(0) == "00:00:00.000"
    assert timestamp(62_345) == "00:01:02.345"
    assert timestamp(3_723_004) == "01:02:03.004"


def test_there_is_one_block_per_utterance(db: Database) -> None:
    video_id = make_video(db, "Evening")
    ordinal, clock = add_words(db, video_id, "Добрый вечер")
    add_words(
        db,
        video_id,
        "Сегодня поговорим",
        first_ord=ordinal,
        start_ms=clock + C.UTTERANCE_SILENCE_GAP_MS,
    )
    index_video(db, video_id)
    blocks = transcript_blocks(db, video_id)
    assert [block.text for block in blocks] == ["Добрый вечер", "Сегодня поговорим"]


def test_every_block_carries_an_anchor_that_resolves_back_to_words(
    db: Database,
) -> None:
    """The whole AI-access story: read a block, point at the database."""
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    block = transcript_blocks(db, video_id)[0]
    assert block.anchor == f"v{video_id}:0-3"
    span = span_for_anchor(db, block.anchor)
    assert span.text == "Добрый вечер дорогие друзья"


def test_a_block_anchor_is_the_same_string_search_prints(db: Database) -> None:
    from rytp.index.search import search

    video_id = corpus(db, "Добрый вечер")
    assert transcript_blocks(db, video_id)[0].anchor == search(
        db, "добрый вечер"
    ).hits[0].anchor


def test_the_block_speaker_prefers_the_roster_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, local, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert transcript_blocks(db, video_id)[0].speaker == "Ведущий"


def test_an_unmapped_label_shows_the_raw_diarizer_label(db: Database) -> None:
    video_id = make_video(db)
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    index_video(db, video_id)
    assert transcript_blocks(db, video_id)[0].speaker == "SPEAKER_00"


def test_an_undiarized_block_shows_the_null_cell(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    assert transcript_blocks(db, video_id)[0].speaker == C.NULL_CELL


def test_the_rendered_document_has_a_title_a_header_and_the_blocks(
    db: Database,
) -> None:
    video_id = corpus(db, "Добрый вечер", title="Evening")
    text = render_transcript(db, video_id)
    assert text.startswith("# Evening\n")
    assert f"- video: {video_id}" in text
    assert "- source: aligned" in text
    assert f"## [v{video_id}:0-1]" in text
    assert "00:00:00.000" in text
    assert "Добрый вечер" in text


def test_the_header_names_the_tier_so_cuttability_is_visible(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер", source="caption", title="Captioned")
    assert "- source: caption" in render_transcript(db, video_id)


def test_a_mixed_tier_video_says_so(db: Database) -> None:
    video_id = make_video(db, "Mixed")
    ordinal, clock = add_words(db, video_id, "Добрый")
    add_words(db, video_id, "вечер", first_ord=ordinal, start_ms=clock, source="caption")
    index_video(db, video_id)
    assert "- source: mixed" in render_transcript(db, video_id)


def test_the_document_says_it_is_regenerable(db: Database) -> None:
    """Design §3: markdown transcripts are output, never input."""
    video_id = corpus(db, "Добрый вечер")
    assert "regenerable" in render_transcript(db, video_id).lower()


def test_write_transcript_lands_in_the_data_tree(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    path = write_transcript(db, video_id)
    assert path == paths().transcript(video_id)
    assert path.read_text(encoding="utf-8").startswith("# ")


def test_write_transcript_uses_unix_newlines_on_every_platform(db: Database) -> None:
    """The file is committed to nothing and read by agents; CRLF would be
    noise, and this project's other output is \\n."""
    video_id = corpus(db, "Добрый вечер")
    raw = write_transcript(db, video_id).read_bytes()
    assert b"\r\n" not in raw


def test_rebuilding_overwrites_rather_than_appends(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    first = write_transcript(db, video_id).read_text(encoding="utf-8")
    second = write_transcript(db, video_id).read_text(encoding="utf-8")
    assert first.count("## [") == second.count("## [") == 1


def test_deleting_the_file_is_safe(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    path = write_transcript(db, video_id)
    path.unlink()
    assert write_transcript(db, video_id).exists()


def test_a_video_with_no_utterances_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        render_transcript(db, video_id)


def test_an_unknown_video_raises_not_found(db: Database) -> None:
    with pytest.raises(NotFoundError):
        render_transcript(db, 404)
```

- [ ] **Step 2: Run the new tests and watch them fail**

Run: `python -m pytest tests/test_index_export.py -v`
Expected: collection error — `ImportError: cannot import name 'render_transcript'`.

- [ ] **Step 3: Append the transcript half of `rytp/index/export.py`**

Extend the module docstring's first paragraph to name both halves, add the new names to `__all__`, add the imports (`from dataclasses import dataclass`, `from rytp.index.search import anchor_for`, `from rytp.models import NotFoundError, utc_now_iso`), and append:

```python
# -- markdown transcripts ---------------------------------------------


@dataclass(frozen=True)
class TranscriptBlock:
    """One utterance, ready to render. `anchor` is the handle back."""

    anchor: str
    start_ms: int
    end_ms: int
    speaker: str
    text: str


def timestamp(ms: int) -> str:
    """Milliseconds as HH:MM:SS.mmm, which sorts and greps cleanly."""
    seconds, milliseconds = divmod(int(ms), C.MS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


_TRANSCRIPT_SQL = (
    "SELECT u.video_id AS video_id, u.start_ms AS start_ms, u.end_ms AS end_ms,"
    " u.first_word_ord AS first_word_ord, u.last_word_ord AS last_word_ord,"
    " u.text AS text, COALESCE(s.label, vs.local_label) AS speaker"
    " FROM utterances u"
    " LEFT JOIN video_speakers vs ON vs.id = u.video_speaker_id"
    " LEFT JOIN speakers s ON s.id = vs.speaker_id"
    " WHERE u.video_id = ? ORDER BY u.first_word_ord"
)


def transcript_blocks(db: Database, video_id: int) -> list[TranscriptBlock]:
    """One block per utterance. The blocks *are* the index rows.

    Deliberately not a second grouping pass: the anchor printed here and
    the anchor printed by `search words` describe the same span because
    they come from the same row.
    """
    return [
        TranscriptBlock(
            anchor=anchor_for(
                int(row["video_id"]), int(row["first_word_ord"]), int(row["last_word_ord"])
            ),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            speaker=C.NULL_CELL if row["speaker"] is None else str(row["speaker"]),
            text=str(row["text"]),
        )
        for row in db.conn.execute(_TRANSCRIPT_SQL, (video_id,))
    ]


def _video_row(db: Database, video_id: int) -> tuple[str, str, int]:
    """(title, tier, word count) for the header."""
    row = db.conn.execute("SELECT title FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"no video with id {video_id}")
    counts = db.conn.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT source) AS tiers,"
        " MIN(source) AS tier FROM words WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    tier = "none" if not counts["n"] else (
        str(counts["tier"]) if counts["tiers"] == 1 else "mixed"
    )
    return str(row["title"]), tier, int(counts["n"])


def render_transcript(db: Database, video_id: int) -> str:
    """The whole markdown document for one video, as a string."""
    title, tier, word_count = _video_row(db, video_id)
    blocks = transcript_blocks(db, video_id)
    if not blocks:
        raise RytpError(
            f"video {video_id} has no utterances; run `rytp index build "
            f"--video-id {video_id}` first"
        )
    lines = [
        f"# {title}",
        "",
        f"- video: {video_id}",
        f"- source: {tier}",
        f"- words: {word_count}",
        f"- utterances: {len(blocks)}",
        f"- generated: {utc_now_iso()}",
        "",
        "Regenerable output — safe to delete, rebuilt by "
        f"`rytp transcript build {video_id}`. Never read back by any stage.",
        "",
        "Each heading carries an anchor of the form `v<video>:<first>-<last>`: "
        "the video id and the word-ordinal range the block covers. "
        f"`SELECT * FROM words WHERE video_id = {video_id} AND ord BETWEEN "
        "<first> AND <last>` returns exactly those words, and "
        "`rytp search play <anchor>` plays them.",
        "",
    ]
    for block in blocks:
        lines.append(
            f"## [{block.anchor}] {timestamp(block.start_ms)} - "
            f"{timestamp(block.end_ms)} - {block.speaker}"
        )
        lines.append("")
        lines.append(block.text)
        lines.append("")
    return "\n".join(lines)


def write_transcript(db: Database, video_id: int) -> Path:
    """Render and write `data/transcripts/{video_id}.md`. Returns the path."""
    text = render_transcript(db, video_id)
    path = config.paths().transcript(video_id)
    config.ensure_dir(path.parent)
    # newline="\n" explicitly: this runs on Windows and the file should
    # not pick up CRLF depending on where it was generated.
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `python -m pytest tests/test_index_export.py -v`
Expected: PASS, 35 passed, 1 skipped (ffmpeg absent).

- [ ] **Step 5: Read one for real**

A transcript nobody has looked at is a format nobody has checked.

```bash
# from the repo root
python - <<'PY'
import tempfile, os, pathlib
root = pathlib.Path(tempfile.mkdtemp())
os.environ["RYTP_DATA"] = str(root)
from rytp.config import ensure_dir, paths
from rytp.db import Database
from rytp.index.export import render_transcript
from rytp.index.utterances import index_video
from rytp.models import normalize_text, stem_text

ensure_dir(paths().root)
db = Database(paths().db)
db.migrate()
db.conn.execute(
    "INSERT INTO videos (source, kind, external_id, url, title, created_at)"
    " VALUES ('ytdlp','video','VIDEO_A','https://example.invalid/w/VIDEO_A',"
    " 'Sample evening', '2026-09-21T00:00:00+00:00')"
)
for i, token in enumerate("Добрый вечер дорогие друзья".split()):
    n = normalize_text(token)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
        " stem, source, engine) VALUES (1,?,?,?,?,?,?,'aligned','fake')",
        (i, i * 300, i * 300 + 250, token, n, stem_text(n)),
    )
index_video(db, 1)
print(render_transcript(db, 1))
PY
```

Expected: a readable document with one `## [v1:0-3] 00:00:00.000 - 00:00:01.150 - -` block. Confirm by eye that the anchor, the timestamps and the sentence all look right before moving on.

- [ ] **Step 6: Lint and type-check**

Run: `ruff check rytp/index tests/test_index_export.py`
Run: `mypy rytp/index`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
# from the repo root
git add rytp/index/export.py tests/test_index_export.py
git commit -m "feat(index): write markdown transcripts whose anchors point back into the database"
```

---

### Task 8: The commands both surfaces are generated from, and the health checks

Contracts §5: each operation is defined once, handlers take an open `Database` first and their parameters as keyword arguments, they never print and never call `sys.exit`, and they raise `RytpError` for an expected failure. Design §10 generates the CLI and the TUI from the same registry, so registering here is all it takes to appear in both.

Six commands in three groups, plus two health checks:

| Command | Group | Long-running | What it is for |
|---|---|---|---|
| `index.build` | `index` | yes | Build or rebuild utterances. One video, or every video that needs it. |
| `search.words` | `search` | no | The backward lookup. Prints an anchor per hit. |
| `search.play` | `search` | yes | Play one anchor through ffplay. Blocks for the length of the clip. |
| `search.export` | `search` | yes | Write every hit of a query out as a WAV. |
| `index.drop` | `index` | no | Contracts §5's `remove` for this part: delete a video's utterances. |
| `transcript.build` | `transcript` | no | Write `data/transcripts/{video_id}.md`. |

**`index.build` closes an enqueueing gap, and this is the place to say so.** Part 2's `ingest` enqueues `download`, `captions` and `extract_wav` and stops there; Part 3 registers no job kinds at all. Nothing in the pipeline currently creates an `index` job after words land. `index.build --enqueue` is the seam that does it, and `index.build` with no arguments does the same work inline through `videos_needing_index`, which is derived from database state and therefore self-healing.

**`search.play` takes an anchor, not a query.** An anchor is exact and reproducible, and `search words` prints one per row, so the two commands compose: search, copy the anchor, play it. Guessing which hit the user meant from a re-run query would be neither.

**Two speaker flags, and the resolver is Part 1's** (contracts §5). `--speaker` is a named person — matched against `speakers.label` and then its aliases, never against a raw `SPEAKER_00`, and a miss is an error naming the closest roster entries rather than an empty table. `--video-local-speaker` takes a diarizer label and **requires `--video-id`**: a bare `SPEAKER_00` would match the first-detected voice of every diarized video, which is not a person and not an answer, so the unscoped combination is rejected. Both resolve through the single `speaker_scope` in `rytp/commands/__init__.py`; Part 4 adds no resolution logic of its own, which is what keeps `--speaker` meaning the same thing here and in `rytp assemble`.

**Filenames.** Exported clips are named from the anchor with `:` replaced (`v12_340-341.wav`), because Windows has no colon in a path component.

**`index.drop` takes no `--dry-run` and no `--yes`,** and that is the contract rather than an oversight: contracts §5 requires them only of a command that deletes *files*. Utterances are derived database rows with nothing behind them and `index.build` puts them straight back, so a confirmation prompt would be ceremony. What does need proving is that the FTS shadow comes clean with them — a surviving `utterances_fts` row is a search hit for text that no longer exists anywhere, which is the quietest way this part could break. Task 2's `test_dropping_clears_the_fts_shadow_too` pins it; the command test checks the same thing through the surface.

**Two health checks** (contracts §5). Part 4 contributes `ffplay` and FTS5. The second matters more than it looks: the whole part is a thin layer over an FTS5 virtual table, so a Python whose bundled SQLite was built without it makes `rytp search` return nothing at all, with no error — the tests skip rather than fail, and a user would conclude the corpus simply does not contain their phrase. The remedy has to be a fix, not a restatement. The two flags stay apart, which contracts §5 is explicit about: `ok` always tells the truth about what was found, and `required` decides whether that failure fails the command. So missing `ffplay` is `ok=False` — it really is missing — with `required=False` keeping `doctor` green, and the detail saying that export still works and only playback does not. Reporting it as `ok=True` to protect the exit code would make the output lie about the one thing the command exists to tell you. `fts5` is `required=True`: search is unusable without it, and that should fail.

**Files:**
- Create: `rytp/commands/search.py`
- Modify: `rytp/commands/__init__.py` (one line in the existing sibling-import block at the bottom)
- Modify: `rytp/constants.py` (append to the Part 4 section)
- Test: `tests/test_commands_search.py`

**Interfaces:**
- Consumes: `rytp.commands.{Command, CommandResult, Param, register, speaker_scope, HealthCheck, HealthResult, register_check}` — `speaker_scope(db, *, speaker, video_local_speaker, video_id) -> frozenset[int] | None` is Part 1's shared resolver (contracts §5); Task 1 Step 1 confirms its real spelling, `rytp.jobs.queue.enqueue`, `rytp.index.utterances.{index_video, videos_needing_index}`, `rytp.index.search.{MatchTier, anchor_filename, search, span_for_anchor}`, `rytp.index.export.{export_clip, play_clip, timestamp, write_transcript}`, `rytp.config.ensure_dir`, `rytp.models.NotFoundError`.
- Produces: the registered commands `index.build`, `index.drop`, `search.words`, `search.play`, `search.export`, `transcript.build`, and the registered health checks `ffplay` and `fts5`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_commands_search.py`:

```python
"""The Part 4 commands (contracts §5, design §7, §10)."""

from __future__ import annotations

from pathlib import Path

import pytest

import rytp.commands.search  # noqa: F401 - importing is what registers them
from rytp import constants as C
from rytp.commands import COMMANDS, CommandResult, resolve
from rytp.db import Database
from rytp.index import export
from rytp.index.utterances import index_video
from rytp.models import NotFoundError, RytpError

from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker, make_video

NAMES = (
    "index.build",
    "index.drop",
    "search.words",
    "search.play",
    "search.export",
    "transcript.build",
)


@pytest.fixture(autouse=True)
def never_spawn_a_player(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Same guard as the export tests: nothing here reaches a real binary."""
    recorded: list[list[str]] = []

    def fake_run(cmd: list[str]) -> object:
        import subprocess

        recorded.append(list(cmd))
        if cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def cached_wav(video_id: int) -> Path:
    import struct
    import wave

    from rytp.config import paths

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * C.AUDIO_SAMPLE_RATE_HZ)
    return path


# --- registration ----------------------------------------------------


def test_every_part_four_command_is_registered() -> None:
    for name in NAMES:
        assert name in COMMANDS, name
        assert COMMANDS[name].group == name.split(".")[0]
        assert COMMANDS[name].summary


def test_the_blocking_commands_are_flagged_long_running() -> None:
    """Design §10: the TUI must not run these inline."""
    assert resolve("index.build").long_running is True
    assert resolve("search.play").long_running is True
    assert resolve("search.export").long_running is True
    assert resolve("search.words").long_running is False
    assert resolve("transcript.build").long_running is False


def test_every_handler_returns_a_command_result(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    assert isinstance(resolve("search.words").handler(db, query="добрый вечер"), CommandResult)
    assert isinstance(resolve("index.build").handler(db), CommandResult)
    assert isinstance(
        resolve("transcript.build").handler(db, video_id=video_id), CommandResult
    )


# --- index.build -----------------------------------------------------


def test_index_build_indexes_one_video(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    result = resolve("index.build").handler(db, video_id=video_id)
    assert result.rows == ((str(video_id), "1"),)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 1


def test_index_build_with_no_video_indexes_everything_that_needs_it(
    db: Database,
) -> None:
    first = make_video(db, "First")
    second = make_video(db, "Second")
    add_words(db, first, "Добрый вечер")
    add_words(db, second, "Здравствуйте друзья")
    index_video(db, first)
    result = resolve("index.build").handler(db)
    assert result.rows == ((str(second), "1"),)


def test_index_build_says_so_when_there_is_nothing_to_do(db: Database) -> None:
    result = resolve("index.build").handler(db)
    assert "nothing" in (result.message or "")


def test_index_build_can_enqueue_instead_of_working(db: Database) -> None:
    """Part 2's ingest chain stops at extract_wav; this is what creates the
    index jobs once words exist."""
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    result = resolve("index.build").handler(db, enqueue=True)
    assert db.conn.execute(
        "SELECT state FROM jobs WHERE kind = 'index' AND target_id = ?", (video_id,)
    ).fetchone()["state"] == "pending"
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0
    assert "queued" in (result.message or "")


def test_index_build_rejects_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("index.build").handler(db, video_id=404)


# --- search.words ----------------------------------------------------


def test_search_words_prints_an_anchor_per_hit(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    result = resolve("search.words").handler(db, query="дорогие друзья")
    assert result.columns == ("anchor", "video", "time", "speaker", "cut", "text")
    assert result.rows[0][0] == f"v{video_id}:2-3"
    assert result.rows[0][2] == "00:00:00.600"


def test_search_words_names_the_tier_that_answered(db: Database) -> None:
    """The owner wants to know when a result is an inflection."""
    corpus(db, "Сегодня было странное ощущение")
    exact = resolve("search.words").handler(db, query="ощущение")
    stemmed = resolve("search.words").handler(db, query="ощущения")
    assert "exact" in (exact.message or "")
    assert "stem" in (stemmed.message or "")


def test_search_words_flags_a_hit_that_spans_a_boundary(db: Database) -> None:
    from tests.test_index_search import split_corpus

    split_corpus(db)
    result = resolve("search.words").handler(db, query="добрый вечер")
    assert "boundary" in (result.message or "")


def test_search_words_reports_no_hits_without_raising(db: Database) -> None:
    corpus(db, "Добрый вечер")
    result = resolve("search.words").handler(db, query="совершенно другое")
    assert result.rows == ()
    assert "no hits" in (result.message or "")


def test_search_words_shows_cuttability(db: Database) -> None:
    corpus(db, "Добрый вечер", source="caption")
    result = resolve("search.words").handler(db, query="добрый вечер")
    assert result.rows[0][4] == "no"


def test_search_words_passes_the_filters_through(db: Database) -> None:
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    corpus(db, "Добрый вечер", title="Anonymous")
    words = resolve("search.words").handler
    assert len(words(db, query="добрый вечер").rows) == 2
    assert len(words(db, query="добрый вечер", speaker="Ведущий").rows) == 1
    assert len(words(db, query="добрый вечер", limit=1).rows) == 1


def test_speaker_means_a_person_and_never_a_diarizer_label(db: Database) -> None:
    """Contracts §5: `--speaker` matches the roster, never `SPEAKER_00`.

    The same flag had two meanings across two commands before this rule
    existed, on the query the whole product is for.
    """
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="добрый вечер", speaker="SPEAKER_00")


def test_an_unknown_person_is_an_error_not_an_empty_table(db: Database) -> None:
    """A silent empty result reads as "he never said it", which is a lie."""
    corpus(db, "Добрый вечер")
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="добрый вечер", speaker="Никто")


def test_a_local_speaker_label_requires_a_video(db: Database) -> None:
    """A bare SPEAKER_00 would match the first-detected voice of every
    diarized video — not a person, and not an answer (contracts §5)."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    index_video(db, video_id)
    with pytest.raises(RytpError):
        resolve("search.words").handler(
            db, query="добрый вечер", video_local_speaker="SPEAKER_00"
        )
    scoped = resolve("search.words").handler(
        db, query="добрый вечер", video_local_speaker="SPEAKER_00", video_id=video_id
    )
    assert len(scoped.rows) == 1


def test_both_speaker_flags_reach_the_same_shared_resolver(db: Database) -> None:
    """Part 4 adds no resolution logic; it calls rytp.commands.speaker_scope."""
    from rytp import commands as commands_module

    calls: list[dict[str, object]] = []
    real = commands_module.speaker_scope

    def spy(db_, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real(db_, **kwargs)  # type: ignore[arg-type]

    import rytp.commands.search as search_module

    original = search_module.speaker_scope
    search_module.speaker_scope = spy  # type: ignore[assignment]
    try:
        corpus(db, "Добрый вечер")
        resolve("search.words").handler(db, query="добрый вечер")
    finally:
        search_module.speaker_scope = original  # type: ignore[assignment]
    assert calls == [{"speaker": None, "video_local_speaker": None, "video_id": 0}]


def test_search_words_truncates_a_long_sentence(db: Database) -> None:
    corpus(db, "слово " * 30 + "конец")
    result = resolve("search.words").handler(db, query="конец")
    assert len(result.rows[0][5]) <= C.SEARCH_TEXT_TRUNCATE_CHARS + 1


def test_search_words_rejects_an_empty_query(db: Database) -> None:
    with pytest.raises(RytpError):
        resolve("search.words").handler(db, query="   ")


# --- search.play and search.export -----------------------------------


def test_search_play_plays_the_anchor(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    result = resolve("search.play").handler(db, anchor=f"v{video_id}:0-1")
    assert never_spawn_a_player[0][0] == "/usr/bin/ffplay"
    assert "Добрый вечер" in (result.message or "")


def test_search_play_rejects_an_anchor_that_is_not_one(db: Database) -> None:
    with pytest.raises(RytpError):
        resolve("search.play").handler(db, anchor="добрый вечер")


def test_search_export_writes_one_file_per_hit(db: Database, data_dir: Path) -> None:
    first = corpus(db, "Добрый вечер", title="First")
    second = corpus(db, "Добрый вечер", title="Second")
    cached_wav(first)
    cached_wav(second)
    out_dir = data_dir / "clips"
    result = resolve("search.export").handler(
        db, query="добрый вечер", out_dir=out_dir
    )
    assert len(result.rows) == 2
    assert sorted(p.name for p in out_dir.iterdir()) == [
        f"v{first}_0-1.wav",
        f"v{second}_0-1.wav",
    ]


def test_exported_filenames_have_no_colon(db: Database, data_dir: Path) -> None:
    """Windows path components cannot contain ':'."""
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    resolve("search.export").handler(db, query="добрый вечер", out_dir=data_dir / "clips")
    assert all(":" not in p.name for p in (data_dir / "clips").iterdir())


def test_search_export_raises_when_there_is_nothing_to_export(
    db: Database, data_dir: Path
) -> None:
    corpus(db, "Добрый вечер")
    with pytest.raises(NotFoundError):
        resolve("search.export").handler(
            db, query="совершенно другое", out_dir=data_dir / "clips"
        )


# --- index.drop (contracts §5, Deletion) -----------------------------


def test_index_drop_removes_the_utterances_and_says_how_many(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    result = resolve("index.drop").handler(db, video_id=video_id)
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0
    assert "1" in (result.message or "")
    assert "index build" in (result.message or "")


def test_index_drop_makes_the_video_unsearchable_and_build_brings_it_back(
    db: Database,
) -> None:
    video_id = corpus(db, "Добрый вечер")
    resolve("index.drop").handler(db, video_id=video_id)
    assert resolve("search.words").handler(db, query="добрый вечер").rows == ()
    resolve("index.build").handler(db, video_id=video_id)
    assert len(resolve("search.words").handler(db, query="добрый вечер").rows) == 1


def test_index_drop_leaves_no_stale_fts_row(db: Database) -> None:
    """Through the surface, because this is the quietest way to break."""
    video_id = corpus(db, "Добрый вечер")
    resolve("index.drop").handler(db, video_id=video_id)
    assert db.conn.execute(
        "SELECT COUNT(*) FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "добрый вечер"',),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_index_drop_needs_no_confirmation_because_it_deletes_no_files() -> None:
    """Contracts §5: `--dry-run` and `--yes` are required only of commands
    that delete files. Utterances are derived rows."""
    names = {param.name for param in resolve("index.drop").params}
    assert names == {"video_id"}


def test_index_drop_rejects_an_unknown_video(db: Database) -> None:
    with pytest.raises(NotFoundError):
        resolve("index.drop").handler(db, video_id=404)


# --- health checks (contracts §5) ------------------------------------


def test_both_part_four_checks_are_registered() -> None:
    from rytp.commands import HEALTH_CHECKS

    assert {"ffplay", "fts5"} <= set(HEALTH_CHECKS)
    for name in ("ffplay", "fts5"):
        assert HEALTH_CHECKS[name].summary


def test_ffplay_is_advisory_and_fts5_is_required() -> None:
    """Contracts §5: `required`, not `ok`, decides whether doctor fails.

    Search is inert without FTS5, so that one should take the exit code
    down. Export works without ffplay, so that one should not.
    """
    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["ffplay"].required is False
    assert HEALTH_CHECKS["fts5"].required is True


def test_the_fts5_check_passes_on_a_working_sqlite(db: Database) -> None:
    from rytp.commands import HEALTH_CHECKS

    result = HEALTH_CHECKS["fts5"].run(db)
    assert result.ok is True
    assert "FTS5" in result.detail


def test_the_ffplay_check_reports_a_missing_binary_truthfully(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never fatal: export still works, only playback does not."""
    from rytp.commands import HEALTH_CHECKS

    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    result = HEALTH_CHECKS["ffplay"].run(db)
    # ok tells the truth about what was found; required=False on the
    # registration is what makes it non-fatal (contracts §5).
    assert result.ok is False
    assert HEALTH_CHECKS["ffplay"].required is False
    assert "export" in result.detail
    assert result.remedy and "ffmpeg" in result.remedy


def test_the_ffplay_check_passes_when_it_is_there(db: Database) -> None:
    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["ffplay"].run(db).ok is True


def test_a_check_never_raises(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contracts §5: a check never raises and never blocks."""
    from rytp.commands import HEALTH_CHECKS

    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    for name in ("ffplay", "fts5"):
        result = HEALTH_CHECKS[name].run(db)
        assert isinstance(result.ok, bool)
        assert result.detail


# --- transcript.build ------------------------------------------------


def test_transcript_build_writes_the_file_and_names_it(db: Database) -> None:
    from rytp.config import paths

    video_id = corpus(db, "Добрый вечер")
    result = resolve("transcript.build").handler(db, video_id=video_id)
    path = paths().transcript(video_id)
    assert path.exists()
    assert str(path) in (result.message or "")


def test_transcript_build_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        resolve("transcript.build").handler(db, video_id=video_id)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `python -m pytest tests/test_commands_search.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.commands.search'`.

- [ ] **Step 3: Append the constant**

At the end of the Part 4 section of `rytp/constants.py`:

```python
#: How much of a hit's surrounding sentence a result table shows before
#: it is cut with an ellipsis. Wider than TITLE_TRUNCATE_CHARS because
#: the sentence is the point of the row, but still narrow enough that a
#: screenful of hits stays a table (design §7).
SEARCH_TEXT_TRUNCATE_CHARS: Final = 80
```

- [ ] **Step 4: Write `rytp/commands/search.py`**

```python
"""Index, search, playback and transcript commands (contracts §5).

Design §10: "Each operation is defined once — name, arguments, result
shape — and both the CLI and the TUI are generated from that
definition." Handlers here are thin: they translate scalars into calls on
`rytp.index` and shape the answer into a `CommandResult`. They never
print and never exit; an expected failure is a `RytpError` and the
surface decides what it looks like.
"""

from __future__ import annotations

from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.commands import (
    Command,
    CommandResult,
    HealthCheck,
    HealthResult,
    Param,
    register,
    register_check,
    speaker_scope,
)
from rytp.db import Database
from rytp.index import export
from rytp.index.export import export_clip, play_clip, timestamp, write_transcript
from rytp.index.search import MatchTier, anchor_filename, search, span_for_anchor
from rytp.index.utterances import drop_utterances, index_video, videos_needing_index
from rytp.jobs import queue
from rytp.models import NotFoundError

_SEARCH_COLUMNS = ("anchor", "video", "time", "speaker", "cut", "text")


def _shorten(text: str) -> str:
    """One line of a result table, not a paragraph."""
    if len(text) <= C.SEARCH_TEXT_TRUNCATE_CHARS:
        return text
    return text[: C.SEARCH_TEXT_TRUNCATE_CHARS - 1].rstrip() + "…"


def index_build(db: Database, *, video_id: int = 0, enqueue: bool = False) -> CommandResult:
    """Rebuild utterances. One video, or every video that needs it.

    With no video, the targets come from `videos_needing_index`, which is
    derived from database state (design §5) — so this is also the repair
    command after a transcript is replaced or a video is diarized.
    """
    if video_id:
        targets = [video_id]
    else:
        targets = videos_needing_index(db)
    if not targets:
        return CommandResult(message="nothing to index")
    if enqueue:
        for target in targets:
            queue.enqueue(db, "index", target)
        return CommandResult(message=f"queued {len(targets)} index job(s)")
    rows = tuple((str(target), str(index_video(db, target))) for target in targets)
    return CommandResult(
        columns=("video", "utterances"),
        rows=rows,
        message=f"indexed {len(rows)} video(s)",
    )


def search_words(
    db: Database,
    *,
    query: str,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    cuttable: bool = False,
    video_id: int = 0,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
) -> CommandResult:
    """Where a word or phrase was said."""
    # Contracts §5: one shared resolver, two identifier spaces. It raises
    # when a person cannot be found (naming near matches) and when
    # --video-local-speaker arrives without a video, so this handler does
    # no speaker logic at all.
    speaker_ids = speaker_scope(
        db, speaker=speaker, video_local_speaker=video_local_speaker, video_id=video_id
    )
    result = search(
        db,
        query,
        speaker_ids=speaker_ids,
        cuttable_only=cuttable,
        video_id=video_id,
        limit=limit,
    )
    if not result.hits:
        return CommandResult(message=f"no hits for {query!r}")
    rows = tuple(
        (
            hit.anchor,
            hit.video_title,
            timestamp(hit.start_ms),
            hit.speaker or C.NULL_CELL,
            "yes" if hit.cuttable else "no",
            _shorten(hit.text),
        )
        for hit in result.hits
    )
    # Which tier answered is part of the answer: a stem hit is an
    # inflection of what was typed, not what was typed (design §7).
    tier = (
        "exact match"
        if result.tier is MatchTier.EXACT
        else "stem match, so these are inflected forms"
    )
    crossed = sum(1 for hit in result.hits if hit.crosses_utterances)
    note = f"; {crossed} spanning an utterance boundary" if crossed else ""
    return CommandResult(
        columns=_SEARCH_COLUMNS,
        rows=rows,
        message=f"{len(rows)} hit(s), {tier}{note}",
    )


def search_play(db: Database, *, anchor: str, pad_ms: int = C.CLIP_PAD_MS) -> CommandResult:
    """Play one anchor, as printed by `search words`."""
    span = span_for_anchor(db, anchor)
    play_clip(db, span.video_id, span.start_ms, span.end_ms, pad_ms=pad_ms)
    return CommandResult(message=f"played {anchor}: {span.text}")


def search_export(
    db: Database,
    *,
    query: str,
    out_dir: Path,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    cuttable: bool = False,
    video_id: int = 0,
    limit: int = C.SEARCH_DEFAULT_LIMIT,
    pad_ms: int = C.CLIP_PAD_MS,
) -> CommandResult:
    """Write every hit of a query out as a WAV, one file per hit."""
    speaker_ids = speaker_scope(
        db, speaker=speaker, video_local_speaker=video_local_speaker, video_id=video_id
    )
    result = search(
        db,
        query,
        speaker_ids=speaker_ids,
        cuttable_only=cuttable,
        video_id=video_id,
        limit=limit,
    )
    if not result.hits:
        raise NotFoundError(f"no hits for {query!r}; nothing to export")
    config.ensure_dir(out_dir)
    rows = []
    for hit in result.hits:
        out = out_dir / anchor_filename(hit.anchor)
        export_clip(db, hit.video_id, hit.start_ms, hit.end_ms, out, pad_ms=pad_ms)
        rows.append((hit.anchor, str(out)))
    return CommandResult(
        columns=("anchor", "file"),
        rows=tuple(rows),
        message=f"wrote {len(rows)} clip(s) to {out_dir}",
    )


def index_drop(db: Database, *, video_id: int) -> CommandResult:
    """Delete one video's utterances (contracts §5, Deletion)."""
    dropped = drop_utterances(db, video_id)
    return CommandResult(
        message=(
            f"dropped {dropped} utterance(s) from video {video_id}; "
            f"rebuild with `rytp index build --video-id {video_id}`"
        )
    )


def transcript_build(db: Database, *, video_id: int) -> CommandResult:
    """Write `data/transcripts/{video_id}.md`."""
    return CommandResult(message=f"wrote {write_transcript(db, video_id)}")


def _check_ffplay(db: Database) -> HealthResult:
    """Playback needs it; export does not.

    Returns `ok=False` when it is missing, because it is; `required=False`
    on the registration is what keeps that non-fatal (contracts §5).
    """
    del db
    found = export._ffplay_binary()
    if found:
        return HealthResult(ok=True, detail=f"ffplay at {found}")
    return HealthResult(
        ok=False,
        detail="ffplay not found on PATH; `rytp search play` will not work "
        "(`rytp search export` still will)",
        remedy="install ffmpeg — ffplay ships with it: "
        "https://ffmpeg.org/download.html",
    )


def _check_fts5(db: Database) -> HealthResult:
    """The one Part 4 cannot survive without.

    Every search goes through an FTS5 virtual table. A Python whose
    bundled SQLite was compiled without FTS5 does not error — it just
    never matches, so the corpus looks empty and the user concludes the
    words are not in it.
    """
    enabled = db.conn.execute(
        "SELECT 1 FROM pragma_compile_options WHERE compile_options = 'ENABLE_FTS5'"
    ).fetchone()
    if enabled:
        return HealthResult(ok=True, detail="sqlite3 has FTS5 compiled in")
    return HealthResult(
        ok=False,
        detail="this Python's sqlite3 was built without FTS5, so `rytp search` "
        "cannot match anything and `rytp index build` cannot create its index",
        remedy="use a Python built against a full SQLite — the python.org "
        "installers and pyenv both are — or `pip install pysqlite3-binary`. "
        "Confirm with: python -c \"import sqlite3; "
        "sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)')\"",
    )


register_check(
    HealthCheck(
        name="ffplay",
        summary="ffplay, for playing a search hit.",
        run=_check_ffplay,
        # Advisory: `search export` works without it, only `search play`
        # does not. The result still reports the truth (contracts §5).
        required=False,
    )
)

register_check(
    HealthCheck(
        name="fts5",
        summary="FTS5 in the running Python's SQLite — every search needs it.",
        run=_check_fts5,
        # required=True by default, and correctly so: without FTS5 this
        # whole part is inert, and `doctor` should say so with its exit
        # code rather than only in its output.
        required=True,
    )
)


register(
    Command(
        name="index.build",
        group="index",
        summary="Rebuild utterances and the search index.",
        params=(
            Param(
                "video_id",
                int,
                "Video to index. 0 indexes every video whose utterances are missing or stale.",
                default=0,
            ),
            Param(
                "enqueue",
                bool,
                "Queue index jobs for the worker instead of doing the work now.",
                default=False,
            ),
        ),
        handler=index_build,
        long_running=True,
    )
)

register(
    Command(
        name="search.words",
        group="search",
        summary="Find every place a word or phrase was said.",
        params=(
            Param("query", str, "Word or phrase to look for.", positional=True),
            Param("speaker", str, "Only hits by this person: a roster label or alias.",
                  default=None, short="-s"),
            Param("video_local_speaker", str,
                  "Only hits by this raw diarizer label, e.g. SPEAKER_00."
                  " Requires --video-id.", default=None),
            Param("cuttable", bool, "Only hits that can be cut (aligned words).",
                  default=False),
            Param("video_id", int, "Only hits in this video. 0 searches every video.",
                  default=0),
            Param("limit", int, "Maximum hits.", default=C.SEARCH_DEFAULT_LIMIT, short="-n"),
        ),
        handler=search_words,
    )
)

register(
    Command(
        name="search.play",
        group="search",
        summary="Play one search hit, named by the anchor `search words` printed.",
        params=(
            Param("anchor", str, "Anchor of the span, e.g. v12:340-341.", positional=True),
            Param("pad_ms", int, "Milliseconds of padding on each side.",
                  default=C.CLIP_PAD_MS),
        ),
        handler=search_play,
        long_running=True,
    )
)

register(
    Command(
        name="search.export",
        group="search",
        summary="Write every hit of a query out as a WAV file.",
        params=(
            Param("query", str, "Word or phrase to look for.", positional=True),
            Param("out_dir", Path, "Directory to write the clips into.", positional=True),
            Param("speaker", str, "Only hits by this person: a roster label or alias.",
                  default=None, short="-s"),
            Param("video_local_speaker", str,
                  "Only hits by this raw diarizer label, e.g. SPEAKER_00."
                  " Requires --video-id.", default=None),
            Param("cuttable", bool, "Only hits that can be cut (aligned words).",
                  default=False),
            Param("video_id", int, "Only hits in this video. 0 searches every video.",
                  default=0),
            Param("limit", int, "Maximum hits.", default=C.SEARCH_DEFAULT_LIMIT, short="-n"),
            Param("pad_ms", int, "Milliseconds of padding on each side.",
                  default=C.CLIP_PAD_MS),
        ),
        handler=search_export,
        long_running=True,
    )
)

register(
    Command(
        name="index.drop",
        group="index",
        summary="Delete a video's utterances. Rebuilt by `index build`.",
        params=(
            Param("video_id", int, "Video whose utterances to drop.", positional=True),
        ),
        handler=index_drop,
    )
)

register(
    Command(
        name="transcript.build",
        group="transcript",
        summary="Write the regenerable markdown transcript for one video.",
        params=(
            Param("video_id", int, "Video to write a transcript for.", positional=True),
        ),
        handler=transcript_build,
    )
)
```


- [ ] **Step 5: Register the module**

In `rytp/commands/__init__.py`, in the existing sibling-import block at the very bottom, add one line. Importing a command module is what registers its commands:

```python
from rytp.commands import search as _search  # noqa: E402,F401
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `python -m pytest tests/test_commands_search.py -v`
Expected: PASS, 38 passed.

- [ ] **Step 7: Check both surfaces picked them up**

Part 1's `tests/test_surfaces.py` asserts every registered command is reachable from the CLI and the TUI palette. It should now cover five more without any change to it.

Run: `python -m pytest tests/test_surfaces.py tests/test_cli.py -q`
Expected: PASS. A failure here means a `Param` shape the generated CLI cannot render — fix the `Param`, not the surface.

- [ ] **Step 8: Lint and type-check**

Run: `ruff check rytp/commands tests/test_commands_search.py`
Run: `mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
# from the repo root
git add rytp/commands/search.py rytp/commands/__init__.py rytp/constants.py \
        tests/test_commands_search.py
git commit -m "feat: add the index, drop, search, play, export and transcript commands"
```

---

### Task 9: A TUI screen that actually reads a transcript

Design §10 promises the TUI covers "browsing videos, reading transcripts, searching, playing hits". Today the palette can run `transcript.build` and be told a file was written, which is not reading it.

Two pieces, and the split is Part 7's, which worked: **all the logic is headless and unit-tested, the screen is a thin shell.** The headless piece here is a new command, not a helper — `transcript.show` returns the blocks as a `CommandResult` table instead of writing a file. That earns its place twice over: `rytp transcript show 12` is genuinely useful on the command line (read a transcript without leaving one on disk), and it means the screen and the CLI render the same rows from the same handler, which is the rule for everything in this task. The screen calls `resolve("transcript.show")`; it does not call `transcript_blocks` behind the registry's back.

The video picker is the one exception, and it follows Part 7's precedent: `SpeakerVideosScreen` calls `store.diarized_videos(db)` directly because choosing what to look at is navigation, not an operation on data.

**Files:**
- Create: `rytp/tui/screens/transcript.py`
- Modify: `rytp/commands/search.py` (one handler and one registration)
- Modify: `rytp/index/utterances.py` (one listing helper)
- Modify: `rytp/tui/app.py` (one binding, one action)
- Test: `tests/test_tui_transcript.py`

**Interfaces:**
- Consumes: `rytp.index.export.{transcript_blocks, timestamp}`, `rytp.commands.{Command, CommandResult, Param, register, resolve}`, Part 1's `RytpApp` with its `_db` attribute and `BINDINGS`.
- Produces:
  - `rytp.index.utterances.indexed_videos(db) -> list[tuple[int, str, int]]` — `(video_id, title, utterance_count)` for every video with utterances, in id order
  - the registered command `transcript.show`
  - `rytp.tui.screens.transcript.{TranscriptVideosScreen, TranscriptScreen}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tui_transcript.py`:

```python
"""Reading a transcript in the TUI (design §10).

The screens are shells: every assertion that is not about widgets is an
assertion about the command they call.
"""

from __future__ import annotations

import asyncio

import pytest

import rytp.commands.search  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.db import Database
from rytp.index.utterances import indexed_videos
from rytp.models import RytpError
from rytp.tui.app import RytpApp
from rytp.tui.screens.transcript import TranscriptScreen, TranscriptVideosScreen

from tests.test_index_search import add_words, corpus
from tests.test_index_utterances import make_speaker, make_video


# --- headless: the listing ------------------------------------------


def test_indexed_videos_lists_only_what_has_utterances(db: Database) -> None:
    indexed = corpus(db, "Добрый вечер", title="Indexed")
    bare = make_video(db, "Bare")
    add_words(db, bare, "Здравствуйте друзья")
    assert [row[0] for row in indexed_videos(db)] == [indexed]


def test_indexed_videos_carries_the_title_and_a_count(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер", title="Evening")
    assert indexed_videos(db) == [(video_id, "Evening", 1)]


def test_indexed_videos_is_empty_before_anything_is_indexed(db: Database) -> None:
    make_video(db, "Nothing")
    assert indexed_videos(db) == []


# --- headless: the command the screen renders ------------------------


def test_transcript_show_returns_the_blocks_as_a_table(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")
    result = resolve("transcript.show").handler(db, video_id=video_id)
    assert result.columns == ("anchor", "start", "end", "speaker", "text")
    assert result.rows[0][0] == f"v{video_id}:0-3"
    assert result.rows[0][1] == "00:00:00.000"
    assert result.rows[0][4] == "Добрый вечер дорогие друзья"


def test_transcript_show_writes_no_file(db: Database) -> None:
    """The difference from `transcript.build`, and the reason both exist."""
    from rytp.config import paths

    video_id = corpus(db, "Добрый вечер")
    resolve("transcript.show").handler(db, video_id=video_id)
    assert not paths().transcript(video_id).exists()


def test_transcript_show_names_the_speaker(db: Database) -> None:
    video_id = make_video(db, "Evening")
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    from rytp.index.utterances import index_video

    index_video(db, video_id)
    assert resolve("transcript.show").handler(db, video_id=video_id).rows[0][3] == (
        "SPEAKER_00"
    )


def test_transcript_show_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        resolve("transcript.show").handler(db, video_id=video_id)


def test_transcript_show_is_not_long_running() -> None:
    """It reads rows. The TUI may run it inline; that is the point."""
    assert resolve("transcript.show").long_running is False


# --- the shells ------------------------------------------------------


def test_the_reader_screen_shows_every_block(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptScreen(db, video_id)
            await app.push_screen(screen)
            table = screen.query_one("#transcript-blocks")
            assert table.row_count == 1
            assert "Evening" in str(screen.query_one("#transcript-status").renderable)

    asyncio.run(scenario())


def test_the_reader_screen_reports_an_unindexed_video_instead_of_crashing(
    db: Database,
) -> None:
    """A TUI that raises on a normal mistake is worse than one that says so."""
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptScreen(db, video_id)
            await app.push_screen(screen)
            assert "index" in str(screen.query_one("#transcript-status").renderable)

    asyncio.run(scenario())


def test_the_picker_lists_indexed_videos(db: Database) -> None:
    corpus(db, "Добрый вечер", title="Evening")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptVideosScreen(db)
            await app.push_screen(screen)
            assert screen.query_one("#transcript-videos").row_count == 1

    asyncio.run(scenario())


def test_the_picker_says_so_when_nothing_is_indexed(db: Database) -> None:
    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptVideosScreen(db)
            await app.push_screen(screen)
            assert "index build" in str(
                screen.query_one("#transcript-videos-status").renderable
            )

    asyncio.run(scenario())


def test_the_app_binds_a_key_to_the_reader() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f6" in keys
```

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_tui_transcript.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.screens.transcript'`.

- [ ] **Step 3: Add the listing helper**

In `rytp/index/utterances.py`, after `videos_needing_index`, and add `"indexed_videos"` to `__all__`:

```python
def indexed_videos(db: Database) -> list[tuple[int, str, int]]:
    """(video_id, title, utterance count) for every indexed video.

    Navigation, not an operation on data, which is why this is a plain
    query rather than a command — the same call Part 7's speaker picker
    makes against `diarized_videos`.
    """
    return [
        (int(row["video_id"]), str(row["title"]), int(row["n"]))
        for row in db.conn.execute(
            "SELECT u.video_id AS video_id, v.title AS title, COUNT(*) AS n"
            " FROM utterances u JOIN videos v ON v.id = u.video_id"
            " GROUP BY u.video_id, v.title ORDER BY u.video_id"
        )
    ]
```

- [ ] **Step 4: Add `transcript.show`**

In `rytp/commands/search.py`, beside `transcript_build`:

```python
def transcript_show(db: Database, *, video_id: int) -> CommandResult:
    """The transcript as rows, without writing a file.

    `transcript.build` produces the durable markdown at
    `data/transcripts/{video_id}.md`; this is the same content as a table,
    so the TUI can show a transcript and the CLI can read one without
    leaving a file behind. Both render from these rows, which is what
    keeps the two surfaces saying the same thing (design §10).
    """
    blocks = transcript_blocks(db, video_id)
    if not blocks:
        raise RytpError(
            f"video {video_id} has no utterances; run `rytp index build "
            f"--video-id {video_id}` first"
        )
    return CommandResult(
        columns=("anchor", "start", "end", "speaker", "text"),
        rows=tuple(
            (
                block.anchor,
                timestamp(block.start_ms),
                timestamp(block.end_ms),
                block.speaker,
                block.text,
            )
            for block in blocks
        ),
        message=f"{len(blocks)} block(s) in video {video_id}",
    )
```

Import `transcript_blocks` and `RytpError`, and register it beside `transcript.build`:

```python
register(
    Command(
        name="transcript.show",
        group="transcript",
        summary="Read a video's transcript as a table, without writing a file.",
        params=(
            Param("video_id", int, "Video to read.", positional=True),
        ),
        handler=transcript_show,
    )
)
```

Note it raises the same `RytpError` `render_transcript` does, with the same wording: one message for "not indexed yet", whichever way the user arrived.

- [ ] **Step 5: Write `rytp/tui/screens/transcript.py`**

```python
"""Reading a transcript in the TUI (design §10).

A shell. Every row it draws comes from `resolve("transcript.show")` — the
same handler `rytp transcript show` calls — so the two surfaces cannot
drift, which is the whole point of generating both from one registry.
The only direct query is the video picker's listing: choosing what to
look at is navigation, not an operation on data, and Part 7's speaker
picker does the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp.commands import resolve
from rytp.index.utterances import indexed_videos
from rytp.models import RytpError

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["TranscriptScreen", "TranscriptVideosScreen"]


def _fill(table: DataTable, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Replace a table's contents. Columns are re-added: they can change."""
    table.clear(columns=True)
    table.add_columns(*columns)
    for row in rows:
        table.add_row(*row)


class TranscriptScreen(Screen[None]):
    """One video's blocks, scrollable, with anchors and timestamps."""

    DEFAULT_CSS = """
    #transcript-blocks { height: 1fr; }
    #transcript-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database, video_id: int) -> None:
        super().__init__()
        self._db = db
        self._video_id = video_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="transcript-blocks")
        yield Static("", id="transcript-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transcript-blocks", DataTable)
        table.cursor_type = "row"
        status = self.query_one("#transcript-status", Static)
        try:
            result = resolve("transcript.show").handler(self._db, video_id=self._video_id)
        except RytpError as exc:
            # A TUI that raises on a normal mistake — an unindexed video —
            # is worse than one that says what to do about it.
            _fill(table, ("anchor",), [])
            status.update(str(exc))
            return
        _fill(table, result.columns, [list(row) for row in result.rows])
        title = self._title()
        status.update(f"{title} — {result.message}" if title else (result.message or ""))

    def _title(self) -> str:
        row = self._db.conn.execute(
            "SELECT title FROM videos WHERE id = ?", (self._video_id,)
        ).fetchone()
        return "" if row is None else str(row["title"])


class TranscriptVideosScreen(Screen[None]):
    """Which transcript to read, so nobody has to remember an id."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._video_ids: list[int] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="transcript-videos")
        yield Static("", id="transcript-videos-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#transcript-videos", DataTable)
        table.cursor_type = "row"
        rows = indexed_videos(self._db)
        self._video_ids = [video_id for video_id, _, _ in rows]
        _fill(
            table,
            ("id", "title", "blocks"),
            [(str(video_id), title, str(count)) for video_id, title, count in rows],
        )
        self.query_one("#transcript-videos-status", Static).update(
            f"{len(rows)} indexed video{'' if len(rows) == 1 else 's'}"
            if rows
            else "nothing indexed yet — run `rytp index build`"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "transcript-videos" or not self._video_ids:
            return
        self.app.push_screen(TranscriptScreen(self._db, self._video_ids[event.cursor_row]))
```

- [ ] **Step 6: Bind it in `rytp/tui/app.py`**

Two edits to Part 1's file, in the shape Part 7 used for `f3`. In `BINDINGS`:

```python
        Binding("f6", "transcripts", "Transcripts"),
```

and one action, with the import inside the body so the app does not pull the
index package in just to start:

```python
    def action_transcripts(self) -> None:
        """Read a transcript (design §10: "reading transcripts")."""
        from rytp.tui.screens.transcript import TranscriptVideosScreen

        self.push_screen(TranscriptVideosScreen(self._db))
```

`f6` is chosen to leave `f5` free: Part 7's mapper screen binds it, and an
app-level `f5` would shadow it for anyone reading the footer.

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_tui_transcript.py -v`
Expected: PASS, 13 passed.

If `await app.push_screen(screen)` complains that it is not awaitable, drop the
`await` — the Textual version decides, and Part 7 hit the same fork.

- [ ] **Step 8: Lint and type-check**

Run: `ruff check rytp/tui rytp/commands/search.py rytp/index tests/test_tui_transcript.py`
Run: `mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
# from the repo root
git add rytp/tui/screens/transcript.py rytp/tui/app.py rytp/commands/search.py \
        rytp/index/utterances.py tests/test_tui_transcript.py
git commit -m "feat(tui): read a transcript instead of being told a file was written"
```

---

### Task 10: A TUI screen that searches and plays what it finds

The owner's ask, in his words: pick a hit and press a key to hear it. Today that is `rytp search words "…"`, read an anchor off the screen, retype it into `rytp search play`. The two commands already exist and already work; what is missing is the thing that joins them.

**`SearchSession` holds everything and calls nothing of its own.** It runs `resolve("search.words").handler` and `resolve("search.play").handler` — the same handlers the CLI invokes — so the screen has no query logic, no filter logic and no playback logic to get wrong or to let drift. It works entirely off the `CommandResult` those handlers return: the `anchor` column is what `search.play` needs, the `cut` column is already `yes`/`no`, and the tier is already in `result.message`.

That is also the answer to the speaker-resolver question. The resolver's signature is in flux across parts; this screen never touches it. It passes a `speaker` string to `search.words` exactly as the CLI does, so whichever way the resolver is settled, one fix corrects both surfaces and this screen needs no edit.

**Errors are shown, not raised.** An unknown person is a `RytpError` from the resolver — correct for the CLI, fatal for a TUI. `SearchSession.run` catches `RytpError` and puts the message where the user can read it, which is also what makes "unknown speaker" visible rather than looking like "he never said it".

**Nothing spawns a player.** `search.play` goes through `rytp.index.export._run`, the seam Task 6 built, and the tests replace it exactly as Task 6's do.

**Files:**
- Create: `rytp/tui/screens/search.py`
- Modify: `rytp/tui/app.py` (one binding, one action)
- Test: `tests/test_tui_search.py`

**Interfaces:**
- Consumes: `rytp.commands.resolve`, `rytp.constants.{SEARCH_DEFAULT_LIMIT, SEARCH_TEXT_TRUNCATE_CHARS}`, `rytp.models.RytpError`, Part 1's `RytpApp`.
- Produces:
  - `rytp.tui.screens.search.SearchSession` — `run(query)`, `play(row)`, `columns`, `rows`, `status`, `error`, `anchor_at(row)`; headless, no widgets
  - `rytp.tui.screens.search.SearchScreen`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tui_search.py`:

```python
"""Searching and playing in the TUI (design §7, §10).

The session is headless and carries every assertion that is not about a
widget. No test reaches a real player: `search.play` goes through the
same `_run` seam Task 6 built, and it is replaced here the same way.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

import rytp.commands.search  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.db import Database
from rytp.index import export
from rytp.tui.app import RytpApp
from rytp.tui.screens.search import SearchScreen, SearchSession

from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker, make_video


@pytest.fixture(autouse=True)
def never_spawn_a_player(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Task 6's seam, replaced the same way. Autouse, so a new test that
    forgets cannot open a window."""
    recorded: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        recorded.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def cached_wav(video_id: int) -> Path:
    import struct
    import wave

    from rytp.config import paths

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * C.AUDIO_SAMPLE_RATE_HZ)
    return path


# --- headless: the session ------------------------------------------


def test_a_search_fills_the_session_from_the_command(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert session.columns == ("anchor", "video", "time", "speaker", "cut", "text")
    assert session.rows[0][0] == f"v{video_id}:0-1"
    assert session.error is None


def test_the_session_reports_the_tier_because_an_inflection_is_not_what_was_typed(
    db: Database,
) -> None:
    corpus(db, "Сегодня было странное ощущение")
    session = SearchSession(db)
    session.run("ощущения")
    assert "stem match" in session.status


def test_the_session_shows_cuttability_per_row(db: Database) -> None:
    """A caption-tier hit cannot be cut, and the screen has to say so."""
    corpus(db, "Добрый вечер", source="caption")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert session.rows[0][4] == "no"


def test_no_hits_is_a_status_line_not_an_error(db: Database) -> None:
    corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("совершенно другое")
    assert session.rows == ()
    assert "no hits" in session.status
    assert session.error is None


def test_a_bad_query_is_shown_rather_than_raised(db: Database) -> None:
    """A TUI that dies on an empty search box is not a TUI."""
    corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("   ")
    assert session.error is not None
    assert session.rows == ()


def test_an_unknown_speaker_is_shown_rather_than_raised(db: Database) -> None:
    """The resolver raises — right for the CLI, fatal for a screen. Showing
    it is also what keeps "unknown person" from reading as "never said it"."""
    corpus(db, "Добрый вечер")
    session = SearchSession(db, speaker="Никто")
    session.run("добрый вечер")
    assert session.error is not None


def test_the_session_passes_the_filters_to_the_command(db: Database) -> None:
    """It hands `search.words` a speaker string exactly as the CLI does, so
    the shared resolver is settled in one place, not two."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    from rytp.index.utterances import index_video

    index_video(db, video_id)
    corpus(db, "Добрый вечер", title="Anonymous")

    assert len(SearchSession(db).also("добрый вечер")) == 2
    assert len(SearchSession(db, speaker="Ведущий").also("добрый вечер")) == 1


def test_cuttable_only_filters_through_the_same_command(db: Database) -> None:
    aligned = corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    assert len(SearchSession(db).also("добрый вечер")) == 2
    rows = SearchSession(db, cuttable=True).also("добрый вечер")
    assert [row[0] for row in rows] == [f"v{aligned}:0-1"]


def test_play_routes_the_highlighted_row_through_the_play_command(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    cached_wav(video_id)
    session = SearchSession(db)
    session.run("дорогие друзья")
    message = session.play(0)
    assert never_spawn_a_player[0][0] == "/usr/bin/ffplay"
    assert f"v{video_id}:2-3" in message


def test_play_on_an_empty_result_says_so_rather_than_indexing_into_nothing(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    session = SearchSession(db)
    session.run("совершенно другое")
    session.play(0)
    assert never_spawn_a_player == []
    assert session.error is not None


def test_play_reports_a_missing_ffplay_instead_of_raising(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    session = SearchSession(db)
    session.run("добрый вечер")
    session.play(0)
    assert session.error is not None
    assert "ffplay" in session.error


def test_anchor_at_is_the_column_search_play_takes(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert session.anchor_at(0) == f"v{video_id}:0-1"
    assert session.anchor_at(99) is None


# --- the shell -------------------------------------------------------


def test_the_screen_fills_its_table_from_a_query(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("добрый вечер")
            assert screen.query_one("#search-hits").row_count == 1

    asyncio.run(scenario())


def test_the_screen_shows_the_tier_in_its_status_line(db: Database) -> None:
    corpus(db, "Сегодня было странное ощущение")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("ощущения")
            assert "stem match" in str(screen.query_one("#search-status").renderable)

    asyncio.run(scenario())


def test_the_screen_plays_the_highlighted_hit(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("добрый вечер")
            screen.action_play()
            assert never_spawn_a_player and never_spawn_a_player[0][0] == "/usr/bin/ffplay"

    asyncio.run(scenario())


def test_the_screen_survives_a_bad_query(db: Database) -> None:
    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("   ")
            assert screen.query_one("#search-hits").row_count == 0
            assert str(screen.query_one("#search-status").renderable)

    asyncio.run(scenario())


def test_the_screen_binds_play_and_the_filters(db: Database) -> None:
    keys = {binding.key for binding in SearchScreen.BINDINGS}
    assert {"ctrl+p", "ctrl+t", "escape"} <= keys


def test_the_app_binds_a_key_to_search() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f4" in keys
```

`SearchSession.also(query)` is a one-line convenience used only by tests: it
runs the query and returns `self.rows`. Define it on the session.

- [ ] **Step 2: Run them and watch them fail**

Run: `python -m pytest tests/test_tui_search.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.screens.search'`.

- [ ] **Step 3: Write `rytp/tui/screens/search.py`**

```python
"""Searching and playing a hit in the TUI (design §7, §10).

Design §10 wants the two surfaces generated from one definition, so this
screen runs the registered handlers rather than reaching past them:
`search.words` for the query and `search.play` for playback, the same two
`rytp search words` and `rytp search play` invoke. The screen exists to
remove the copy-and-paste between them.

Everything that is not a widget lives in `SearchSession`, which is why
the shell below is short enough to read in one go.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from rytp import constants as C
from rytp.commands import CommandResult, resolve
from rytp.models import RytpError

if TYPE_CHECKING:
    from rytp.db import Database

__all__ = ["SearchScreen", "SearchSession"]

#: Column index of the anchor in a `search.words` result. The anchor is
#: what `search.play` takes, which is what lets the two compose.
_ANCHOR = 0


@dataclass
class SearchSession:
    """Query, filters, and the last result. No widgets, no SQL.

    Every database access goes through a registered command handler, so
    the screen cannot drift from the CLI and the speaker resolver — whose
    signature is still settling across parts — is touched in exactly one
    place, inside `search.words`.
    """

    db: Database
    speaker: str | None = None
    cuttable: bool = False
    limit: int = C.SEARCH_DEFAULT_LIMIT
    query: str = ""
    result: CommandResult = field(default_factory=CommandResult)
    error: str | None = None

    # -- running ---------------------------------------------------------

    def run(self, query: str) -> None:
        """Search, keeping any failure as text rather than an exception."""
        self.query = query
        self.error = None
        try:
            self.result = resolve("search.words").handler(
                self.db,
                query=query,
                speaker=self.speaker,
                cuttable=self.cuttable,
                limit=self.limit,
            )
        except RytpError as exc:
            # An empty box, or a person who is not on the roster. Both are
            # ordinary mistakes; a screen that raises on them is unusable,
            # and showing the resolver's message is also what stops
            # "unknown speaker" reading as "he never said it".
            self.result = CommandResult()
            self.error = str(exc)

    def also(self, query: str) -> tuple[tuple[str, ...], ...]:
        """Run and hand back the rows. A convenience for tests."""
        self.run(query)
        return self.rows

    def play(self, row: int) -> str:
        """Play one row through `search.play`. Returns what it reported."""
        self.error = None
        anchor = self.anchor_at(row)
        if anchor is None:
            self.error = "nothing to play"
            return self.error
        try:
            played = resolve("search.play").handler(self.db, anchor=anchor)
        except RytpError as exc:
            # No ffplay, or no audio for that video. Worth saying, not
            # worth losing the result table over.
            self.error = str(exc)
            return self.error
        return played.message or f"played {anchor}"

    def toggle_cuttable(self) -> None:
        """Flip the cuttable-only filter and re-run the current query."""
        self.cuttable = not self.cuttable
        if self.query:
            self.run(self.query)

    # -- reading ---------------------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return self.result.columns

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        return self.result.rows

    @property
    def status(self) -> str:
        """One line: the error if there is one, else what the command said."""
        if self.error:
            return self.error
        filters = " · cuttable only" if self.cuttable else ""
        if self.speaker:
            filters += f" · speaker {self.speaker}"
        return (self.result.message or "") + filters

    def anchor_at(self, row: int) -> str | None:
        if not (0 <= row < len(self.rows)):
            return None
        return self.rows[row][_ANCHOR]


class SearchScreen(Screen[None]):
    """A query box, the hits, and one key that plays the highlighted one."""

    DEFAULT_CSS = """
    #search-hits { height: 1fr; }
    #search-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("ctrl+p", "play", "Play hit"),
        Binding("ctrl+t", "toggle_cuttable", "Cuttable only"),
    ]

    def __init__(self, db: Database, speaker: str | None = None) -> None:
        super().__init__()
        self.session = SearchSession(db, speaker=speaker)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="word or phrase, then Enter", id="search-query")
        yield DataTable(id="search-hits")
        yield Static("", id="search-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-hits", DataTable).cursor_type = "row"
        self.query_one("#search-query", Input).focus()
        self._draw()

    # -- actions ---------------------------------------------------------

    def search(self, query: str) -> None:
        """Run a query and redraw. Called by Enter and by the tests."""
        self.session.run(query)
        self._draw()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-query":
            self.search(event.value)

    def action_play(self) -> None:
        """Play whichever hit the cursor is on (design §7: "Playback")."""
        row = self.query_one("#search-hits", DataTable).cursor_row
        message = self.session.play(row)
        self.query_one("#search-status", Static).update(message)

    def action_toggle_cuttable(self) -> None:
        self.session.toggle_cuttable()
        self._draw()

    # -- drawing ---------------------------------------------------------

    def _draw(self) -> None:
        table = self.query_one("#search-hits", DataTable)
        table.clear(columns=True)
        if self.session.columns:
            table.add_columns(*self.session.columns)
            for row in self.session.rows:
                table.add_row(*row)
        self.query_one("#search-status", Static).update(self.session.status)
```

- [ ] **Step 4: Bind it in `rytp/tui/app.py`**

In `BINDINGS`, beside the `f6` added in Task 9:

```python
        Binding("f4", "search", "Search"),
```

and the action:

```python
    def action_search(self) -> None:
        """Search the corpus and play a hit (design §7, §10)."""
        from rytp.tui.screens.search import SearchScreen

        self.push_screen(SearchScreen(self._db))
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_tui_search.py -v`
Expected: PASS, 18 passed.

- [ ] **Step 6: Prove no player was spawned**

Run: `python -m pytest tests/test_tui_search.py -q`
Expected: PASS, and the run finishes in well under a second — a real `ffplay`
would block for the length of each clip.

- [ ] **Step 7: Check the palette still covers every command**

Both screens are reachable by key, but the commands behind them must also stay
in the palette, which is Part 1's parity test.

Run: `python -m pytest tests/test_surfaces.py tests/test_tui.py -q`
Expected: PASS. `transcript.show` appears in the palette like any other command.

- [ ] **Step 8: Lint and type-check**

Run: `ruff check rytp/tui tests/test_tui_search.py`
Run: `mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
# from the repo root
git add rytp/tui/screens/search.py rytp/tui/app.py tests/test_tui_search.py
git commit -m "feat(tui): search the corpus and play a hit without retyping an anchor"
```

---

### Task 11: End to end, at scale, and the gate

`CLAUDE.md` is blunt about this project: *"a green test run says almost nothing about whether the real pipeline works. Verify end-to-end behaviour by actually running `python -m rytp` against a real file."* This task does both — an automated walk from caption words to a written transcript through the generated CLI, a scale check that the index still answers when it holds more than four rows, and then a real command-line run.

The automated run uses **caption tier** deliberately: Part 3's `ingest_captions` parses a json3 file with no model and no binary, so the whole chain from words to transcript is exercised on a machine with neither. It calls `ingest_captions` directly rather than through `transcribe captions`, because the parameter spelling of Part 3's command is Part 3's to choose and this test should not break when it changes.

The argv in this test (`--video-id`, `--cuttable`, `--speaker`) is the one place Part 4 assumes how Part 1's generated CLI spells a flag. **If it spells one differently, change the argv here, not the `Param`** — the registry definition is the contract and the surface is generated from it.

**Files:**
- Create: `tests/test_index_end_to_end.py`
- Create: `scripts/manual_check_index.py`
- Modify: whatever lint or mypy flags

**Interfaces:**
- Consumes: `rytp.cli.build_app`, `rytp.transcribe.captions.ingest_captions`, every Part 4 module.
- Produces: nothing new. This task adds confidence, not surface.

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_index_end_to_end.py`:

```python
"""Words in, searchable index and a readable transcript out.

Caption tier on purpose: Part 3's json3 ingest needs no model and no
binary, so this runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import constants as C
from rytp.cli import build_app
from rytp.config import paths
from rytp.db import Database
from rytp.index.search import MatchTier, search
from rytp.index.utterances import index_video
from rytp.models import normalize_text, stem_text
from rytp.transcribe.captions import ingest_captions

from tests.test_index_utterances import make_video

LINES = (
    (0, "Добрый вечер дорогие друзья"),
    (4_000, "Сегодня у нас странное ощущение"),
    (9_000, "Ну что же поговорим об этом"),
)


def write_json3(path: Path) -> Path:
    """A minimal json3 caption track: one event per line, one seg per word."""
    events = [
        {
            "tStartMs": base,
            "segs": [
                {"utf8": token, "tOffsetMs": index * 400}
                for index, token in enumerate(line.split())
            ],
        }
        for base, line in LINES
    ]
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return path


@pytest.fixture()
def captioned(db: Database, data_dir: Path) -> int:
    video_id = make_video(db, "Evening")
    ingest_captions(db, video_id, write_json3(data_dir / "captions.json3"))
    return video_id


def test_captions_become_a_searchable_index_and_a_transcript(
    captioned: int, db: Database
) -> None:
    runner = CliRunner()
    app = build_app()

    built = runner.invoke(app, ["index", "build", "--video-id", str(captioned)])
    assert built.exit_code == 0, built.output
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] >= 3

    found = runner.invoke(app, ["search", "words", "добрый вечер"])
    assert found.exit_code == 0, found.output
    assert f"v{captioned}:0-1" in found.output
    assert "exact match" in found.output

    inflected = runner.invoke(app, ["search", "words", "ощущения"])
    assert inflected.exit_code == 0, inflected.output
    assert "stem match" in inflected.output

    # Caption words are searchable and never cuttable (contracts §3).
    uncuttable = runner.invoke(app, ["search", "words", "добрый вечер", "--cuttable"])
    assert uncuttable.exit_code == 0, uncuttable.output
    assert "no hits" in uncuttable.output

    missing = runner.invoke(app, ["search", "words", "совершенно другое"])
    assert missing.exit_code == 0, missing.output
    assert "no hits" in missing.output

    written = runner.invoke(app, ["transcript", "build", str(captioned)])
    assert written.exit_code == 0, written.output
    text = paths().transcript(captioned).read_text(encoding="utf-8")
    assert "# Evening" in text
    assert f"## [v{captioned}:" in text
    assert "- source: caption" in text


def test_the_cli_reports_a_bad_anchor_as_one_line_not_a_traceback(
    captioned: int,
) -> None:
    result = CliRunner().invoke(build_app(), ["search", "play", "not-an-anchor"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_reindexing_after_the_words_change_keeps_search_honest(
    captioned: int, db: Database, data_dir: Path
) -> None:
    """Contracts §4: utterances die with the words; Part 4 re-derives them."""
    index_video(db, captioned)
    assert len(search(db, "добрый вечер").hits) == 1
    with db.transaction():
        db.conn.execute("DELETE FROM utterances WHERE video_id = ?", (captioned,))
        db.conn.execute("DELETE FROM words WHERE video_id = ?", (captioned,))
    assert search(db, "добрый вечер").hits == ()
    ingest_captions(db, captioned, data_dir / "captions.json3")
    index_video(db, captioned)
    assert len(search(db, "добрый вечер").hits) == 1


def bulk_words(db: Database, video_id: int, lines: list[str]) -> None:
    """Insert one words row per token of every line, in one transaction."""
    rows = []
    ordinal = 0
    clock = 0
    for line in lines:
        for token in line.split():
            normalized = normalize_text(token)
            rows.append(
                (
                    video_id,
                    ordinal,
                    clock,
                    clock + 250,
                    token,
                    normalized,
                    stem_text(normalized),
                    "aligned",
                    "fake",
                )
            )
            ordinal += 1
            clock += 300
    with db.transaction():
        db.conn.executemany(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text,"
            " normalized_text, stem, source, engine)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_the_index_still_answers_at_a_realistic_size(db: Database) -> None:
    """Four rows prove nothing about an FTS index. Build a few thousand
    words across several videos and find one phrase in the middle."""
    needle = "совершенно неповторимая фраза"
    for video_index in range(5):
        video_id = make_video(db, f"Bulk{video_index}")
        lines = [
            needle if (video_index == 3 and chunk == 100) else "и что то ещё"
            for chunk in range(200)
        ]
        bulk_words(db, video_id, lines)
        index_video(db, video_id)

    result = search(db, needle)
    assert result.tier is MatchTier.EXACT
    assert len(result.hits) == 1
    assert result.hits[0].video_title == "Bulk3"
    # And the stem tier reaches it through an inflection.
    assert search(db, "совершенно неповторимые фразы").tier is MatchTier.STEM


def test_a_query_of_only_common_words_stays_bounded(db: Database) -> None:
    """The walk anchors on the rarest token and stops at the anchor limit,
    so a query of nothing but function words cannot walk the corpus."""
    for video_index in range(3):
        video_id = make_video(db, f"Common{video_index}")
        bulk_words(db, video_id, ["и " * 100] * 3)
        index_video(db, video_id)
    result = search(db, "и и", limit=5)
    assert len(result.hits) <= 5
    assert C.SEARCH_WALK_ANCHOR_LIMIT > 0
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_index_end_to_end.py -v`
Expected: PASS, 5 passed. If `test_the_index_still_answers_at_a_realistic_size` takes more than a couple of seconds, chase it before moving on — the FTS candidate query should not be scanning.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS, with the ffmpeg-marked test skipped. Nothing from Parts 1-3 may regress; if a Part 2 job test now sees a fourth kind, widen its assertion to a superset rather than unregistering `index`.

- [ ] **Step 4: Lint and type-check everything**

Run: `ruff check rytp tests`
Run: `mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 5: Run it by hand, the way the owner will**

One driver script, because it has to work on the owner's Windows machine as well
as a developer's: it sets `RYTP_DATA` in its own environment and shells out to
`python -m rytp` with list arguments (contracts §1).

Create `scripts/manual_check_index.py` — it is a developer tool, not part of the
package, and it is the only file this plan puts outside `rytp/` and `tests/`:

```python
"""Walk captions -> index -> search -> transcript through the real CLI.

Needs no model and no binary: caption ingest is pure Python. Run it from
the repo root with the virtualenv activated:

    python scripts/manual_check_index.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / ".rytp-manual"

LINES = (
    (0, "Добрый вечер дорогие друзья"),
    (4_000, "Сегодня у нас странное ощущение"),
)


def seed() -> None:
    from rytp.config import ensure_dir, paths
    from rytp.db import Database
    from rytp.transcribe.captions import ingest_captions

    ensure_dir(paths().root)
    db = Database(paths().db)
    db.migrate()
    db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, created_at)"
        " VALUES ('ytdlp', 'video', 'VIDEO_A', 'https://example.invalid/w/VIDEO_A',"
        " 'Sample evening', '2026-09-21T00:00:00+00:00')"
    )
    events = [
        {
            "tStartMs": base,
            "segs": [
                {"utf8": token, "tOffsetMs": index * 400}
                for index, token in enumerate(line.split())
            ],
        }
        for base, line in LINES
    ]
    path = paths().root / "captions.json3"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    print("words:", ingest_captions(db, 1, path))
    db.close()


def run(*args: str) -> None:
    print(f"\n$ rytp {' '.join(args)}")
    result = subprocess.run([sys.executable, "-m", "rytp", *args], check=False)
    if result.returncode != 0:
        print(f"  (exit {result.returncode})")


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    os.environ["RYTP_DATA"] = str(DATA)
    seed()
    run("index", "build")
    run("search", "words", "добрый вечер")
    run("search", "words", "ощущения")
    run("search", "words", "добрый вечер", "--cuttable")
    run("transcript", "build", "1")
    print("\n--- transcripts/1.md ---")
    print((DATA / "transcripts" / "1.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `python scripts/manual_check_index.py`

Check each by eye, not just the exit code:

- `index build` reports one video indexed.
- `search words "добрый вечер"` prints a table with an `anchor` column holding `v1:0-1` and a message ending `exact match`. **This is the query that returned zero in the old implementation** — it is the whole reason this part exists.
- `search words "ощущения"` finds the corpus's `ощущение` and says `stem match`.
- `search words "добрый вечер" --cuttable` prints `no hits`: caption words are searchable and never cuttable.
- `transcript build 1` names the file, and the printed document has `## [v1:0-3]`-style headings with `HH:MM:SS.mmm` timestamps and a line saying it is regenerable.

Add `.rytp-manual/` to `.gitignore` if it is not already covered.

- [ ] **Step 6: Commit**

```bash
# from the repo root
git add tests/test_index_end_to_end.py scripts/manual_check_index.py
git commit -m "test: walk captions through index, search and transcript end to end"
```

---

## Self-review

Run this after the last task, with the design and the contracts open.

- [ ] **Spec coverage.** Design §7, sentence by sentence: utterance FTS index (Task 2); exact phrase first, stem column as fallback (Task 4); results carry video, timestamp, speaker, surrounding sentence and cuttability (Task 4); filters by speaker and cuttable only (Task 4); playback and export through ffplay/ffmpeg (Task 6); markdown transcripts at `data/transcripts/{video_id}.md` with stable anchors (Task 7). Design §4 "Utterances": contiguous runs, speaker split, silence fallback, two FTS columns (Task 2). Contracts §3: `unicode61` never `porter` (Part 1's migration, re-asserted in Task 2), cuttable is `source = 'aligned'` (Task 4). Contracts §4: `stem_text` in `rytp/models.py` (Task 1), ё→е inherited from `normalize_text` and tested in both directions (Task 4), utterances re-derived after words change (Tasks 2, 3, 9). Contracts §5: `index` in `JOB_HANDLERS` with a payload-free readiness predicate (Task 3), commands in the registry (Task 8).
- [ ] **Both named regressions have a test that cannot be weakened.** `test_a_two_word_phrase_whose_words_are_adjacent_returns_a_hit` and `test_a_russian_inflection_is_found_through_the_stem_tier`, Task 4 Step 1.
- [ ] **Cross-boundary coverage is explicit.** Task 5 covers a silence split (found), a speaker split (correctly not found), a cross-video pair (not found), a bridged hit through the stem tier, the three filters on a bridged hit, and the known limitation.
- [ ] **No test spawns a player.** `grep -rn "subprocess" tests/test_index_*.py tests/test_commands_search.py tests/test_tui_search.py` should show only the deliberate import probes in `test_index_job.py` and the `CompletedProcess` fakes. `rytp/index/export.py` is the only Part 4 module importing `subprocess`; the TUI screens reach it only through `search.play`.
- [ ] **Neither screen bypasses the registry.** `grep -n "conn.execute" rytp/tui/screens/search.py` returns nothing, and in `rytp/tui/screens/transcript.py` only the title lookup and `indexed_videos` — navigation, the same allowance Part 7's speaker picker takes. Every row drawn and every action taken goes through `resolve(...)`, which is what keeps the CLI and the TUI saying the same thing (design §10) and what means the speaker resolver is settled in one place.
- [ ] **Screen logic is headless and tested as such.** `SearchSession` has no widget in it; twelve of Task 10's tests never start an app. The shells are smoke-tested with `asyncio.run(app.run_test())`, core Textual, no plugin — Part 7's pattern.
- [ ] **No real identifiers.** `grep -rniE "youtube|youtu\.be|watch\?v=" rytp/index rytp/commands/search.py tests/test_index_*.py tests/test_commands_search.py` returns nothing. Every URL is `example.invalid`; every external id is `VIDEO_*`.
- [ ] **Windows.** No `:` in any filename this code writes (`anchor_filename`); every subprocess call is a list; every path is a `Path`; transcripts are written with `newline="\n"`.
- [ ] **Constants.** Everything tunable this plan introduced is in the Part 4 section of `rytp/constants.py` with a comment naming its design section: `STEMMER_LANGUAGE`, `UTTERANCE_SILENCE_GAP_MS`, `UTTERANCE_MAX_WORDS`, `UTTERANCE_MAX_DURATION_MS`, `CAPTION_WORD_FALLBACK_MS`, `CUTTABLE_SOURCE`, `SEARCH_DEFAULT_LIMIT`, `SEARCH_MAX_LIMIT`, `SEARCH_WALK_ANCHOR_LIMIT`, `SEARCH_RARITY_PROBE_LIMIT`, `CLIP_PAD_MS`, `SEARCH_TEXT_TRUNCATE_CHARS`.
- [ ] **Names agree across tasks.** Task 2: `IndexedWord`, `UtteranceDraft`, `implied_end_ms`, `build_utterances`, `words_for`, `index_video`, `index_readiness`, `videos_needing_index`. Tasks 4-5: `MatchTier`, `SearchHit`, `SearchResult`, `AnchorSpan`, `anchor_for`, `parse_anchor`, `anchor_filename`, `query_tokens`, `tokens_for_tier`, `fts_phrase`, `span_for_anchor`, `search`, `rarest_token_index`, `walk_matches`. Tasks 6-7: `MediaToolMissing`, `ClipError`, `clip_source`, `padded`, `build_clip_command`, `build_play_command`, `export_clip`, `play_clip`, `timestamp`, `TranscriptBlock`, `transcript_blocks`, `render_transcript`, `write_transcript`. Task 8 imports exactly these spellings and no others.
- [ ] **`index.drop` leaves nothing behind.** Utterances gone, `utterances_fts` gone with them (trigger), `words` untouched, `integrity-check` clean, and the video is no longer searchable by any route — including the cross-boundary walk, which skips a run no utterance covers. Four tests in Task 2, three in Task 8.
- [ ] **Two health checks, honest results, correct exit code.** `ffplay` and `fts5` are in `HEALTH_CHECKS`; both return a `HealthResult` rather than raising; `ok` reports what was actually found in each case; `ffplay` is `required=False` and `fts5` is `required=True` (contracts §5); and the FTS5 remedy is a fix a user can run rather than a restatement of the problem.
- [ ] **Two speaker identifier spaces, one resolver.** `grep -n "local_label" rytp/index/ rytp/commands/search.py` finds it only in the *display* COALESCE, never in a filter. `rytp/index/search.py` takes `frozenset[int]` and no speaker string; `rytp/commands/search.py` calls `speaker_scope` and defines no resolution of its own (contracts §5). `--speaker` rejects `SPEAKER_00`; `--video-local-speaker` without `--video-id` is an error. All four pinned by tests in Task 8.
- [ ] **Every `NOT NULL` column is supplied by its inserting fixture** (contracts §3) — `video_speakers.engine` most easily missed. Task 1's parametrized test names them; `make_speaker` supplies `engine`.
- [ ] **Portable run steps** (contracts §1). no run step in this plan names an absolute interpreter, a home directory or a system temp path. Every step is `python -m pytest`, `ruff`, `mypy` or `python scripts/...`.
- [ ] **One row, one token.** Contracts §4 and Part 3's `split_token` are what let `_locate` and `_row_run` compare rows to tokens directly. `grep -n "split()" rytp/index/search.py` should find it only in `query_tokens` and `tokens_for_tier`, never over a `words` column. The test helper `stored_tokens` seeds rows the same way, and `test_a_hyphen_and_a_yo_in_the_same_query` pins the interaction with the ё fold.
- [ ] **SQL interpolation.** The only interpolated identifiers are `word_column` / the FTS column, and both come from the module-level `_TIERS` tuple, plus `C.CUTTABLE_SOURCE` in a constant string. Nothing a caller supplies is ever interpolated; everything else is a bound parameter.

## Notes for the parts that come after

- **Part 5 (assembly)** should build its longest-run matcher on `walk_matches` / `rarest_token_index` in `rytp/index/search.py` rather than writing a second ordinal walk. That function is already design §8's access path — anchor on a word through `words(normalized_text)`, then walk forward comparing ordinals — and it already enforces the single-speaker rule that keeps a match honest. What Part 5 adds is longest-run-wins, scoring and `Fragment`, not a new traversal.
- **Part 7 (diarization)** deletes a video's utterances and enqueues its `index` job after assigning speakers, which is exactly right: design §7's speaker filter searches utterances, so they have to be rebuilt to carry the speakers and to be re-split on speaker change instead of on silence. Task 3's `test_the_state_part_seven_leaves_behind_is_ready` pins that hand-off from this side. `index_readiness` also covers the weaker case where speakers are filled in place and the utterances are left alone, so a reconcile re-fires even if that enqueue is ever missed.
- **Part 5 and Part 7** both had a `resolve_speaker` of their own when this plan was reviewed. Contracts §5 now puts one in `rytp/commands/__init__.py` and forbids the rest; Part 4 calls that one and nothing else. If Part 1's spelling differs from `speaker_scope`, Task 1 Step 1 catches it in one place.
- **Whoever adds more TUI screens** should keep the `SearchSession` split: logic in a plain dataclass calling registered handlers, the `Screen` a shell over it. It is what let Task 10's behaviour be tested without a terminal, and it is why the unsettled speaker-resolver signature does not reach the TUI at all.
- **Whoever owns the ingest chain** should decide whether `ingest` gains `index` (and `transcribe`) as a follow-on. Today Part 2's chain stops at `extract_wav`, Part 3 registers no job kinds, and `index.build --enqueue` (Task 8) is the only thing that creates an `index` job.
