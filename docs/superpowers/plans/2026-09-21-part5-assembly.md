# rytp Part 5 — Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Type a Russian sentence and get back a cut list — an ordered, hand-editable TOML file at `cutlists/{name}.toml` naming the real corpus fragments, with millisecond in and out points and ranked alternatives, that together say that sentence.

**Architecture:** Three layers under `rytp/assemble/`. `match.py` walks the target text against the corpus: it finds every occurrence of a token through `words(normalized_text)`, extends each occurrence forward through `words(video_id, ord)` one ordinal at a time, and then solves a shortest-path problem over (target position × last source video) to choose a segmentation and a source for every slot at once. `score.py` supplies the costs that path minimises — one `consistency` knob interpolates between a "fewest seams" and a "most consistent sound" weight profile, with acoustic distance read from `video_acoustics`. `cutlist.py` turns the result into the durable artifact and reads it back after a human has edited it. `rytp/commands/assemble.py` registers three commands; nothing here enqueues a job and nothing here renders.

**Tech Stack:** Python 3.11+, stdlib only — `sqlite3` through the shared `Database`, `tomllib` for reading TOML, a small hand-written canonical emitter for writing it, `hashlib.blake2b` for seeded jitter, `difflib` for "did you mean" on hand-edit typos. No new third-party dependency. `snowballstemmer` (already a base dependency, reached only through `rytp.models.stem_text`) supplies the stem tier of substitution ranking.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (design — §8 Assembly is this part, §4 and §7 give the data it reads) and `docs/superpowers/specs/2026-09-21-rytp-contracts.md` (binding shared interfaces — §2 package layout, §3 schema, §4 types, §5 registry, §7 filesystem, §8 conventions).

## Global Constraints

- Python >= 3.11, 3.11 syntax. `from __future__ import annotations` in **every** module.
- SQLite only, reached only through an already-open `rytp.db.Database` passed in by the caller. Nothing in this part opens a connection, and nothing in this part writes to the database — assembly is a pure read.
- Runs on Windows. `pathlib` everywhere, no POSIX-only calls, no shell pipelines, no symlinks. Every file this part writes is opened with `encoding="utf-8"` and `newline="\n"` so a cut list is byte-identical on both platforms.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` runs on `rytp/`.
- No heavy dependency, no optional extra, no network. Assembly is pure logic over the database and **no test in this part may touch the network, ffmpeg, or an audio file.** Corpus fixtures are word rows.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. Russian test strings are wanted — they are what this part is for.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers. Part 5 appends one section to that file and never reorders what is already there.
- Milliseconds, integers, everywhere. Timestamps written into a cut list are `rytp.models.utc_now_iso()` strings (contracts §8).
- Domain errors subclass `RytpError` (`rytp/models.py`). The CLI prints one line to stderr and exits 1 — never a traceback for an expected failure. Handlers never print and never call `sys.exit`.
- Commands are registered in `rytp/commands/__init__.py`'s import block and are generated onto both surfaces from that one definition (contracts §5). Part 5 defines no CLI code and no TUI code.
- **Test invocation (pytest is not on `PATH`):**
  `python -m pytest <args>`
- One commit per task, conventional-commit prefix, present tense.

### Hard dependencies on other parts

| Needed | Owner | Used for |
|---|---|---|
| `rytp.db.Database`, fixtures `db` / `data_dir` / `tmp_path` | Part 1 | every test and every function here |
| `rytp.models.{RytpError, NotFoundError, InvalidInputError, Fragment, normalize_text, stem_text, utc_now_iso}` | Part 1 | types, tokenisation, stem-tier substitutions |
| `rytp.config.paths().cutlist(name)`, `rytp.config.ensure_dir` | Part 1 | where a cut list lives |
| `rytp.commands.{Command, CommandResult, Param, register}` | Part 1 | the three commands |
| `rytp.commands.resolve_speaker_filter` — the one shared speaker resolver (contracts §5) | Part 1 | `--speaker`; Part 5 must not roll its own |
| `words` rows with `ord`, real `start_ms`/`end_ms`, `align_score`, `source='aligned'` | Part 3 | everything cuttable |
| `video_acoustics` rows | Part 3 | the consistency half of the knob |
| `video_speakers` / `speakers` | Parts 1 (schema) and 7 (rows) | the `--speaker` filter; null until a video is diarized |

**`stem_text` lives in `rytp/models.py`** and Part 1 implements it there in full, over `snowballstemmer` (a base dependency, contracts §1). Import it from `rytp.models`; **never import `rytp.index` from this part** — assembly is not search, and depending on the index package would make the creative core wait on it.

### Two assumptions about the data, stated up front

1. **A target token is exactly one `words` row, both sides.** This is a contract invariant (contracts §4): "A stored word row holds exactly one token, and the tokens stored for a piece of text are exactly `normalize_text(text).split()`." `normalize_text` turns every non-`\w` character into a space, so "кто-то" is two tokens — and Part 3's `split_token()` splits at write time to match, giving the interior boundary a measured time from `refine_boundaries` rather than an interpolated one. Assembly tokenises the target through the same function, so the two sides agree by construction: a hyphenated source word is two ordinally adjacent rows, fully searchable and fully assemblable, and a run may extend straight through it. Task 2 pins that with a test. Nothing in this part compensates for punctuation, and nothing needs to.
2. **`align_score` is nullable.** MFA-class aligners report no per-word confidence and Part 3 stores `None`. Every read of `align_score` — in SQL `ORDER BY` as well as in Python — substitutes `ASSEMBLE_DEFAULT_ALIGN_SCORE`, or NULL rows sort to one end and get silently dropped by a candidate cap.

---

## File Structure

| File | Responsibility |
|---|---|
| `rytp/constants.py` | **Modify.** Append the "Assembly" section: caps, cost weights, the two knob endpoint profiles, acoustic scales, substitution limits. |
| `rytp/assemble/__init__.py` | `assemble_target(...) -> CutList` — the one call that ties matching, scoring and the artifact together, plus the package's public re-exports. |
| `rytp/assemble/match.py` | Corpus walk and coverage. Occurrence lookup, level-synchronous forward extension, the (position × last-video) DP, substitution search, bounded edit distance. Produces a `Plan`. |
| `rytp/assemble/score.py` | Costs only, no SQL beyond one acoustics read. Weight profiles and the knob, acoustic distance, fragment cost, transition cost, seeded jitter, the deterministic comparison key. |
| `rytp/assemble/cutlist.py` | The durable artifact. `CutList`/`Slot`/`Alternative`/`Substitution`/`CutlistParams`, `from_plan`, the canonical TOML emitter, and a tolerant loader that says exactly what a human broke. |
| `rytp/commands/assemble.py` | `assemble.plan`, `assemble.show`, `assemble.suggest`, `assemble.remove`. |
| `rytp/commands/__init__.py` | **Modify.** One line in the import block at the bottom. |
| `tests/assembly_corpus.py` | Corpus fixtures as word rows: `add_video`, `add_words`, `add_acoustics`, `add_speaker`. Not named `tests/corpus.py` — Part 4 is being written in parallel and that name is contested. |
| `tests/test_assemble_match.py` | Occurrence lookup, extension, the DP, substitutions. |
| `tests/test_assemble_score.py` | The knob, acoustic distance, costs, jitter. |
| `tests/test_assemble_cutlist.py` | Emitter, round-trip, and every way a hand-edit can be wrong. |
| `tests/test_commands_assemble.py` | The three commands, as handlers and through the CLI. |
| `tests/test_assemble_determinism.py` | Byte-identical output, seed behaviour, and the query budget. |
| `tests/test_commands_assemble.py` | Also covers `assemble.remove` — see Task 11. |

**Layering, one direction only:** `score.py` → (nothing in this part). `match.py` → `score.py`. `cutlist.py` → `match.py` (for the `Plan` types it converts). `__init__.py` → all three. `commands/assemble.py` → `__init__.py` and `cutlist.py`. Contracts §2 lists `assemble/ match.py score.py cutlist.py`; `__init__.py` is the package marker Python requires, and it holds the orchestrator the same way `rytp/db/__init__.py` holds `Database`.

---

## Why a DP and not greedy-longest

Design §8 describes the move as "find the longest run of words starting there […] Take it, jump forward, repeat." Before replacing that with something more expensive, it is worth being exact about what greedy actually gets wrong, because the obvious objection to it is false.

**The obvious objection is false.** "A shorter first fragment can enable a much longer second, so greedy uses more fragments" does not happen here. The family of coverable target intervals is closed under sub-intervals: if tokens `[a, b)` occur contiguously in some video, then so does every `[c, d)` with `a <= c < d <= b`, because a contiguous run's sub-run is a contiguous run. With that closure, the standard stays-ahead argument applies — if greedy's `i`-th breakpoint is at least the optimum's `i`-th breakpoint, then greedy's next reach is at least the optimum's next, since the optimum's next interval, trimmed to start where greedy stands, is still coverable. **Greedy-longest provably minimises the number of fragments.** A plan that justified the DP on fragment count would be justifying it on a bug that is not there.

**What greedy does get wrong is everything else the design asks for.** Fragment count is one of three terms in §8's objective; greedy optimises that one and is blind to the other two.

*Source consistency.* Target `мы все понимаем что это`, corpus:

- video A says `… мы все понимаем что …` and never says `это`
- video B says `… мы все понимаем …` and, elsewhere, `… что это …`

Greedy takes the longest run at position 0 — four words from A — and then has to fetch `это` from somewhere else: two fragments, **two sources, one audible seam between different rooms and microphones.** Taking B's three-word run instead gives `мы все понимаем` + `что это`, both from B: the same two fragments, **one source, no cross-video seam at all.** Greedy cannot see this, because it commits at position 0 knowing only lengths. This is design §8's "preferring few distinct source videos is a strong and cheap proxy", and greedy has no way to express it.

*Cut quality.* Two runs of equal length at the same position, one whose first and last words carry `align_score` 0.95 and one whose boundary words sit at 0.2. Greedy has no preference between them — they are the same length. Design §4 put `align_score` in the schema precisely "so the assembler can skip badly-anchored instances", and §3 rules that "precision beats recall for assembly". A cost function has the preference; a length comparison does not.

*The knob itself.* At the consistency end of the dial the right answer frequently has **more** fragments than greedy's — three cuts inside one video beat two cuts across two videos when the two videos do not sound alike. An algorithm that minimises fragment count by construction cannot ever return that answer, so the knob would have nothing to turn.

**So: a shortest path over target positions.** Nodes are positions, an edge from `i` to `j` exists when tokens `[i, j)` occur contiguously in some video, and the edge weight is that fragment's cost. Consistency is not a property of one edge, so the last source is carried in the state: a node is `(position, last_video)`, and an edge gains a transition term that is zero when the source does not change and positive — a switch charge plus scaled acoustic distance — when it does.

Adjacent-change count is the Markovian relaxation of "number of distinct videos": `A B A` scores two changes but uses two videos. That is a feature. Cutting back to a source you already left is a second audible seam, and a measure that priced `A B A` the same as `A A B` would be wrong about the thing the owner can actually hear. It also upper-bounds `distinct − 1` and coincides with it whenever a video's fragments end up adjacent, which is what minimising it produces.

Cost, with `N` target tokens, `L = ASSEMBLE_MAX_RUN_WORDS` and `V` candidate videos: the naive edge relaxation is `O(N · L · V²)`, which at `V = 200` is tens of millions of operations for one sentence. It collapses to `O(N · (V² + L · V))` by noticing that the transition term depends only on `(previous video, next video)` and not on the edge's length — so each position first computes one *arrival cost* per candidate video, `min over previous sources of (cost there + transition)`, and every edge into that video then reuses it. `N` is a sentence and `L` is 40, so this is roughly a million operations at the worst realistic width.

**Greedy is still in there.** Set the knob to 0 with uniform alignment and the acoustic weight vanishes, the switch weight is small, and the seam weight dominates every other term — which is exactly the fragment-count objective greedy solves. The DP is a strict generalisation, not a different algorithm.

---

## The knob, in one paragraph

`consistency` is a float in `[0, 1]`, default `ASSEMBLE_DEFAULT_CONSISTENCY = 0.25` — design §8 says "defaulting toward fewer seams". It selects a point on a straight line between two named weight profiles: `ASSEMBLE_WEIGHTS_FEWEST_SEAMS` (a fragment is expensive, switching source is nearly free, acoustics are ignored) and `ASSEMBLE_WEIGHTS_MOST_CONSISTENT` (a fragment is cheap, switching source is expensive, and acoustic distance between the two sources is priced). Each endpoint is a three-number `Weights` record with a comment saying what it is for, so tuning the knob later — design §11, M4 — is editing two constants, not re-deriving an algorithm. The seam weight never reaches zero at `consistency = 1`: if it did, splitting one long run into single words inside one video would be free, and the cut list would be shredded for no gain.

---

## Tasks

### Task 1: Constants and the corpus test fixture

Everything downstream reads these numbers, and every test builds its corpus with these helpers, so both land first. The fixture module is the more important half: design §8 is pure logic over `words` rows, so the whole part can be tested without a single byte of audio, and the quality of the tests is bounded by how easy it is to write a corpus.

**Files:**
- Modify: `rytp/constants.py` (append one section at the end)
- Create: `tests/assembly_corpus.py`
- Test: `tests/test_assembly_corpus.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.{normalize_text, stem_text, utc_now_iso}`, the `db` fixture from Part 1.
- Produces:
  - `rytp.constants` — `ASSEMBLE_DEFAULT_CONSISTENCY`, `ASSEMBLE_WEIGHTS_FEWEST_SEAMS`, `ASSEMBLE_WEIGHTS_MOST_CONSISTENT`, `ASSEMBLE_EDGE_ALIGN_WEIGHT`, `ASSEMBLE_MEAN_ALIGN_WEIGHT`, `ASSEMBLE_DEFAULT_ALIGN_SCORE`, `ASSEMBLE_MIN_ALIGN_SCORE`, `ASSEMBLE_GAP_COST`, `ASSEMBLE_MAX_RUN_WORDS`, `ASSEMBLE_MAX_INTERNAL_GAP_MS`, `ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN`, `ASSEMBLE_MAX_SEEDS_PER_VIDEO`, `ASSEMBLE_SQL_BATCH`, `ASSEMBLE_MAX_ALTERNATIVES`, `ASSEMBLE_COST_DECIMALS`, `ASSEMBLE_SEED_JITTER`, `ASSEMBLE_ACOUSTIC_SCALES`, `ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE`, `ASSEMBLE_ACOUSTIC_DISTANCE_CAP`, `ASSEMBLE_SUBSTITUTION_MAX_DISTANCE`, `ASSEMBLE_SUBSTITUTION_LEN_WINDOW`, `ASSEMBLE_SUBSTITUTION_LIMIT`, `ASSEMBLE_SUBSTITUTION_VOCAB_LIMIT`, `ASSEMBLE_DEFAULT_PAD_MS`, `ASSEMBLE_MAX_PAD_MS`, `ASSEMBLE_MAX_TARGET_WORDS`, `ASSEMBLE_NAME_MAX_CHARS`, `ASSEMBLE_FALLBACK_NAME`, `CUTLIST_SCHEMA_VERSION`, `CUTLIST_SUFFIX`.
  - `tests/assembly_corpus.py` — `add_video(db, *, external_id, title=…, duration_ms=None, source="youtube", kind="video") -> int`, `add_speaker(db, label) -> int`, `add_video_speaker(db, video_id, local_label, *, speaker_id=None, engine="fake") -> int`, `add_words(db, video_id, text, *, start_ms=0, word_ms=WORD_MS, gap_ms=GAP_MS, source="aligned", align_score=0.9, engine="fake", video_speaker_id=None) -> int`, `add_acoustics(db, video_id, **fields) -> None`, `word_rows(db, video_id) -> list[sqlite3.Row]`, constants `WORD_MS`, `GAP_MS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_assembly_corpus.py`:

```python
"""The corpus fixture builder. Word rows, no audio — design §8 is pure logic."""

from __future__ import annotations

import pytest

from rytp.db import Database
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_acoustics,
    add_speaker,
    add_video,
    add_video_speaker,
    add_words,
    word_rows,
)


def test_add_words_lays_out_ords_and_timings(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все понимаем")
    rows = word_rows(db, video_id)
    assert [row["ord"] for row in rows] == [0, 1, 2]
    assert [row["normalized_text"] for row in rows] == ["мы", "все", "понимаем"]
    assert rows[0]["start_ms"] == 0
    assert rows[0]["end_ms"] == WORD_MS
    assert rows[1]["start_ms"] == WORD_MS + GAP_MS
    assert rows[2]["end_ms"] == 3 * WORD_MS + 2 * GAP_MS


def test_add_words_normalizes_and_stems(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "Ещё Раз")
    rows = word_rows(db, video_id)
    assert [row["text"] for row in rows] == ["Ещё", "Раз"]
    assert [row["normalized_text"] for row in rows] == ["еще", "раз"]
    assert all(row["stem"] for row in rows)


def test_a_second_call_continues_the_ordinals(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    next_ord = add_words(db, video_id, "мы все")
    assert next_ord == 2
    add_words(db, video_id, "понимаем", start_ms=10_000)
    rows = word_rows(db, video_id)
    assert [row["ord"] for row in rows] == [0, 1, 2]
    assert rows[2]["start_ms"] == 10_000


def test_caption_words_have_no_end_and_the_check_constraint_holds(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", source="caption")
    rows = word_rows(db, video_id)
    assert [row["source"] for row in rows] == ["caption", "caption"]
    assert rows[0]["end_ms"] is None


def test_video_speakers_rows_carry_the_engine_that_made_them(db: Database) -> None:
    """contracts §3: engine is NOT NULL and this fixture must supply it."""
    video_id = add_video(db, external_id="VIDEO_A")
    local_id = add_video_speaker(db, video_id, "SPEAKER_00", engine="fake-diarizer")
    row = db.conn.execute(
        "SELECT engine FROM video_speakers WHERE id = ?", (local_id,)
    ).fetchone()
    assert row["engine"] == "fake-diarizer"


def test_words_can_carry_a_speaker(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    speaker_id = add_speaker(db, "host")
    local_id = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=speaker_id)
    add_words(db, video_id, "мы все", video_speaker_id=local_id)
    assert {row["video_speaker_id"] for row in word_rows(db, video_id)} == {local_id}


def test_align_score_may_be_null(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", align_score=None)
    assert word_rows(db, video_id)[0]["align_score"] is None


def test_add_acoustics_defaults_every_field_and_takes_overrides(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_acoustics(db, video_id, f0_mean=180.0)
    row = db.conn.execute(
        "SELECT * FROM video_acoustics WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert row["f0_mean"] == 180.0
    assert row["loudness_lufs"] is not None


def test_add_acoustics_rejects_a_field_the_table_does_not_have(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    with pytest.raises(KeyError, match="brightness"):
        add_acoustics(db, video_id, brightness=1.0)


def test_two_videos_get_distinct_ids(db: Database) -> None:
    first = add_video(db, external_id="VIDEO_A")
    second = add_video(db, external_id="VIDEO_B")
    assert first != second
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assembly_corpus.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tests.assembly_corpus'`.

- [ ] **Step 3: Write `tests/assembly_corpus.py`**

```python
"""Build a corpus out of word rows, for the assembly tests.

Design §8 is pure logic over ``words`` and ``video_acoustics``: nothing in
part 5 opens an audio file, so nothing in its tests needs one. A test says
what was said, in which video, by whom, and these helpers turn that into
rows the matcher can walk.

Timings are synthetic but realistic in shape: every word lasts ``WORD_MS``
and every inter-word silence lasts ``GAP_MS``, so a run's span is
predictable arithmetic a test can assert on. Pass ``gap_ms`` explicitly to
simulate a long pause that must stop a run from extending across it.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from rytp.db import Database
from rytp.models import normalize_text, stem_text, utc_now_iso

#: Synthetic duration of one spoken word.
WORD_MS = 300

#: Synthetic silence between two consecutive words.
GAP_MS = 100

#: Every column of ``video_acoustics`` except the key and the timestamp,
#: with a plausible default so a test overrides only what it cares about.
ACOUSTIC_DEFAULTS: dict[str, float] = {
    "f0_mean": 120.0,
    "f0_std": 20.0,
    "spectral_tilt": -1.0,
    "noise_floor_db": -60.0,
    "reverb_proxy": 0.2,
    "loudness_lufs": -23.0,
}


def add_video(
    db: Database,
    *,
    external_id: str,
    title: str = "Sample",
    duration_ms: int | None = None,
    source: str = "youtube",
    kind: str = "video",
) -> int:
    """Catalog one video. ``external_id`` is a placeholder, never a real id."""
    cursor = db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, duration_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            source,
            kind,
            external_id,
            f"https://example.invalid/v/{external_id}",
            title,
            duration_ms,
            utc_now_iso(),
        ),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_speaker(db: Database, label: str) -> int:
    """One row in the global roster."""
    cursor = db.conn.execute(
        "INSERT INTO speakers (label, created_at) VALUES (?, ?)", (label, utc_now_iso())
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_video_speaker(
    db: Database,
    video_id: int,
    local_label: str,
    *,
    speaker_id: int | None = None,
    engine: str = "fake",
) -> int:
    """One diarizer label in one video, optionally mapped to the roster.

    ``engine`` is NOT NULL in contracts §3, and §3 puts supplying every
    NOT NULL column on the part whose fixtures insert the row.
    """
    cursor = db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine) "
        "VALUES (?, ?, ?, ?)",
        (video_id, local_label, speaker_id, engine),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def add_words(
    db: Database,
    video_id: int,
    text: str,
    *,
    start_ms: int = 0,
    word_ms: int = WORD_MS,
    gap_ms: int = GAP_MS,
    source: str = "aligned",
    align_score: float | None = 0.9,
    engine: str = "fake",
    video_speaker_id: int | None = None,
) -> int:
    """Append ``text`` to a video as word rows. Returns the next free ordinal.

    Ordinals continue after whatever the video already has, so a test can
    build one video from several calls with different speakers or gaps.
    Caption rows get no ``end_ms`` and no ``align_score``: contracts §3
    makes that the definition of the caption tier, and the CHECK constraint
    enforces it.
    """
    row = db.conn.execute(
        "SELECT COALESCE(MAX(ord) + 1, 0) AS next FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()
    ordinal = int(row["next"])
    captions = source == "caption"
    cursor_ms = start_ms
    rows: list[tuple[Any, ...]] = []
    for token in text.split():
        normalized = normalize_text(token)
        rows.append(
            (
                video_id,
                ordinal,
                cursor_ms,
                None if captions else cursor_ms + word_ms,
                token,
                normalized,
                stem_text(normalized),
                None,
                None if captions else align_score,
                source,
                engine,
                video_speaker_id,
            )
        )
        ordinal += 1
        cursor_ms += word_ms + gap_ms
    db.conn.executemany(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem, "
        "confidence, align_score, source, engine, video_speaker_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.conn.commit()
    return ordinal


def add_acoustics(db: Database, video_id: int, **fields: float) -> None:
    """One ``video_acoustics`` row; every unnamed column takes its default."""
    values = dict(ACOUSTIC_DEFAULTS)
    for key, value in fields.items():
        if key not in ACOUSTIC_DEFAULTS:
            raise KeyError(f"video_acoustics has no column {key!r}")
        values[key] = value
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    db.conn.execute(
        f"INSERT INTO video_acoustics (video_id, {columns}, computed_at) "
        f"VALUES (?, {placeholders}, ?)",
        (video_id, *values.values(), utc_now_iso()),
    )
    db.conn.commit()


def word_rows(db: Database, video_id: int) -> list[sqlite3.Row]:
    """Every word of one video, in reading order."""
    return list(
        db.conn.execute("SELECT * FROM words WHERE video_id = ? ORDER BY ord", (video_id,))
    )
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assembly_corpus.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Append the assembly section to `rytp/constants.py`**

At the very end of the file, after every existing section:

```python
# ---------------------------------------------------------------------------
# Assembly (design §8)
# ---------------------------------------------------------------------------

#: The one knob, from 0.0 "fewest seams" to 1.0 "most consistent sound".
#: Design §8: "defaulting toward fewer seams".
ASSEMBLE_DEFAULT_CONSISTENCY: Final = 0.25

#: The knob's two endpoints, as (seam, switch, acoustic) cost weights.
#: At "fewest seams" a fragment is expensive and changing source is nearly
#: free, so the optimiser buys the longest runs it can from anywhere. The
#: switch weight is not zero even here: given two equally long coverings,
#: the one that stays in one video is still the better video.
ASSEMBLE_WEIGHTS_FEWEST_SEAMS: Final = (1.0, 0.15, 0.0)

#: At "most consistent sound" a fragment is cheap and changing source is
#: expensive, priced by how differently the two videos sound. The seam
#: weight stays at a quarter rather than dropping to zero: free seams
#: would let a long run inside one video be shredded into single words
#: for no gain, which is not what anyone means by "consistent".
ASSEMBLE_WEIGHTS_MOST_CONSISTENT: Final = (0.25, 2.5, 2.0)

#: Cost of a fragment whose first and last words are badly anchored.
#: Only those two boundaries are actually cut, so they dominate — design
#: §4: "align_score — per-word alignment confidence, so the assembler can
#: skip badly-anchored instances."
ASSEMBLE_EDGE_ALIGN_WEIGHT: Final = 0.6

#: Cost of a fragment whose interior words are badly anchored. Interior
#: boundaries are never cut, but a low score there means the transcript
#: itself is doubtful, and design §3 puts precision above recall.
ASSEMBLE_MEAN_ALIGN_WEIGHT: Final = 0.2

#: Stand-in for a NULL align_score. MFA-class aligners report no per-word
#: confidence (design §6), so NULL means "unknown", not "bad"; 0.5 keeps
#: such a word usable while preferring a measured, well-anchored one.
ASSEMBLE_DEFAULT_ALIGN_SCORE: Final = 0.5

#: Default floor for --min-align-score: accept everything. Raising it is
#: how the owner trades recall for cut quality on a specific run.
ASSEMBLE_MIN_ALIGN_SCORE: Final = 0.0

#: Cost of leaving one target word uncovered. Large enough that a gap is
#: only ever chosen when the word is genuinely absent, never as a cheaper
#: alternative to an awkward fragment — design §8: "never apply one
#: silently".
ASSEMBLE_GAP_COST: Final = 1000.0

#: Longest run the matcher will build. A fragment longer than this is a
#: quotation, not an assembly, and extending past it wastes queries.
ASSEMBLE_MAX_RUN_WORDS: Final = 40

#: A run stops rather than spanning a silence longer than this. The cut
#: would otherwise carry a pause the target sentence does not have, and
#: design §9 gives pause length to the renderer, not to the matcher.
ASSEMBLE_MAX_INTERNAL_GAP_MS: Final = 800

#: Rows fetched for one target token before extension begins. A Russian
#: function word occurs hundreds of thousands of times in a full corpus
#: (design §4: ~14M word rows); this caps the read, and the ORDER BY
#: makes which rows survive deterministic.
ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN: Final = 2000

#: Occurrences per video carried forward into extension. One video cannot
#: crowd out the rest of the corpus, so the cap preserves source diversity
#: — which is exactly what the consistency knob needs to choose between.
ASSEMBLE_MAX_SEEDS_PER_VIDEO: Final = 4

#: Rows per parameterised IN clause. SQLite's oldest supported limit on
#: host parameters is 999 and each pair costs two, so 400 pairs is safe
#: everywhere without being chatty.
ASSEMBLE_SQL_BATCH: Final = 400

#: Ranked alternatives kept per slot, so design §8's "a choice can be
#: swapped without re-running" costs one hand-edit and no recompute.
ASSEMBLE_MAX_ALTERNATIVES: Final = 3

#: Decimal places costs are rounded to before comparison. Determinism
#: (design §8) needs near-equal candidates to actually tie so the
#: tie-break decides; without rounding, float noise decides instead.
ASSEMBLE_COST_DECIMALS: Final = 6

#: How far a non-zero seed may move a candidate's cost. Design §8: a seed
#: "shakes up choices among near-equal candidates" — small enough that it
#: reorders ties, never overturns a real preference.
ASSEMBLE_SEED_JITTER: Final = 0.05

#: Per-field "one unit of audible difference" for video_acoustics
#: (design §4). A 40 Hz difference in mean pitch, 8 dB of noise floor or
#: 4 LU of loudness each count as one unit; the distance is the mean over
#: the fields both videos actually have.
ASSEMBLE_ACOUSTIC_SCALES: Final = {
    "f0_mean": 40.0,
    "f0_std": 20.0,
    "spectral_tilt": 0.5,
    "noise_floor_db": 8.0,
    "reverb_proxy": 0.3,
    "loudness_lufs": 4.0,
}

#: Distance used when either video has no acoustics row — undiarized and
#: un-fingerprinted videos are the normal case early on. "Middling", so
#: an unmeasured source is neither preferred nor excluded.
ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE: Final = 0.5

#: Acoustic distance is clamped here. Beyond one unit per field the two
#: recordings are simply incomparable and further difference changes no
#: decision.
ASSEMBLE_ACOUSTIC_DISTANCE_CAP: Final = 1.0

#: Largest edit distance offered as a substitution. Design §8 calls edit
#: distance "a workable proxy for sound in Russian"; past two edits it
#: stops being one.
ASSEMBLE_SUBSTITUTION_MAX_DISTANCE: Final = 2

#: Only vocabulary within this many characters of the missing word is
#: considered, since a longer or shorter form cannot be within
#: ASSEMBLE_SUBSTITUTION_MAX_DISTANCE anyway.
ASSEMBLE_SUBSTITUTION_LEN_WINDOW: Final = 2

#: Substitutions offered per missing word.
ASSEMBLE_SUBSTITUTION_LIMIT: Final = 5

#: Distinct normalized forms pulled into memory for the edit-distance
#: pass. The length window usually keeps this far smaller; the cap is
#: what stops a one-letter target word from reading the whole vocabulary.
ASSEMBLE_SUBSTITUTION_VOCAB_LIMIT: Final = 50_000

#: Default tail padding added after each fragment's last word (design §8
#: "add padding after a chosen word"). Zero: cut exactly on the measured
#: boundary unless asked otherwise.
ASSEMBLE_DEFAULT_PAD_MS: Final = 0

#: Upper bound on --pad-ms. Past two seconds the padding is a fragment of
#: its own and should be cut as one.
ASSEMBLE_MAX_PAD_MS: Final = 2000

#: Longest target accepted. The DP is quadratic in the target length and
#: a sentence is tens of words; a paste of a whole transcript is a
#: mistake worth naming.
ASSEMBLE_MAX_TARGET_WORDS: Final = 200

#: Characters kept when deriving a cut list name from the target text.
ASSEMBLE_NAME_MAX_CHARS: Final = 40

#: Name used when the target slugifies to nothing at all.
ASSEMBLE_FALLBACK_NAME: Final = "cutlist"

#: Version stamped into every cut list. Bumped only when the on-disk
#: shape changes in a way an old reader would misread; part 6 refuses a
#: version it does not know.
CUTLIST_SCHEMA_VERSION: Final = 1

#: Extension of a cut list file under cutlists/ (contracts §7).
CUTLIST_SUFFIX: Final = ".toml"
```

- [ ] **Step 6: Check the constants import and lint both files**

Run: `python -c "from rytp import constants as C; print(C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT, C.CUTLIST_SCHEMA_VERSION)"`
Expected: `(0.25, 2.5, 2.0) 1`

Run: `python -m ruff check rytp/constants.py tests/assembly_corpus.py tests/test_assembly_corpus.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add rytp/constants.py tests/assembly_corpus.py tests/test_assembly_corpus.py
git commit -m "feat: add assembly constants and the word-row corpus fixture"
```

---

### Task 2: Candidate occurrences and forward extension

The pointer walk itself. Design §7: "Assembly matching — exact, no fuzziness, driven by `words(normalized_text)` and `words(video_id, ord)`. Find every occurrence of the first word, then walk forward comparing ordinals. No n-gram table is needed."

Extension is **level-synchronous**: every surviving candidate advances one ordinal in one batched query, rather than each candidate being walked to exhaustion on its own. That turns the corpus read from "one query per occurrence per step" into "one query per step", which is what makes the walk affordable when the first token is a function word with a hundred thousand occurrences.

Runs are recorded at **every** prefix length, not only the maximal one, because the DP in Task 4 needs the short edges too.

**Files:**
- Create: `rytp/assemble/__init__.py` (empty package marker for now; Task 9 fills it), `rytp/assemble/match.py`
- Test: `tests/test_assemble_match.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.normalize_text`, `rytp.constants`, `tests/assembly_corpus.py`.
- Produces:
  - `WordRow(video_id, ord, start_ms, end_ms, text, normalized_text, align_score, video_speaker_id)` — frozen dataclass; `align_score` is never `None` here, NULL is already coalesced.
  - `CandidateRun(video_id, first_word_ord, last_word_ord, start_ms, end_ms, text, n_words, first_align, last_align, mean_align, video_speaker_id)` — frozen.
  - `MatchFilters(exclude_video_ids=frozenset(), video_speaker_ids=None, min_align_score=C.ASSEMBLE_MIN_ALIGN_SCORE, max_internal_gap_ms=C.ASSEMBLE_MAX_INTERNAL_GAP_MS, max_run_words=C.ASSEMBLE_MAX_RUN_WORDS, max_occurrences=C.ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN, max_seeds_per_video=C.ASSEMBLE_MAX_SEEDS_PER_VIDEO)` — frozen.
  - `RunTable = dict[tuple[int, int], dict[int, CandidateRun]]` — keyed `(target position, run length in words)`, value maps `video_id` to the best run of that shape in that video.
  - `tokenize(target: str) -> tuple[str, ...]`
  - `find_occurrences(db, token: str, filters: MatchFilters) -> list[WordRow]`
  - `build_run_table(db, tokens: Sequence[str], filters: MatchFilters) -> RunTable`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_match.py`:

```python
"""The pointer walk: occurrences of the first word, then forward by ordinal."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.assemble.match import (
    MatchFilters,
    build_run_table,
    find_occurrences,
    tokenize,
)
from rytp.db import Database
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_acoustics,
    add_video,
    add_video_speaker,
    add_words,
)

TARGET = "мы все понимаем что это неизбежно"


@pytest.fixture()
def corpus(db: Database) -> tuple[int, int]:
    """Two overlapping videos. Neither says the whole target; together they do."""
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем что это")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "все понимаем что это неизбежно")
    return first, second


def test_tokenize_normalizes_and_splits() -> None:
    assert tokenize("  Мы  ВСЁ, понимаем! ") == ("мы", "все", "понимаем")


def test_tokenize_splits_a_hyphenated_word_into_two_tokens() -> None:
    assert tokenize("кто-то") == ("кто", "то")


def test_a_hyphenated_word_is_assemblable(db: Database) -> None:
    """Contracts §4: a stored row holds exactly one token, on both sides.

    Part 3's split_token() writes "кто-то" as two ordinally adjacent rows
    with a measured interior boundary, and the target normalizes to the
    same two tokens — so a run walks straight through the hyphen.
    """
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "кто-то еще")
    assert [row.normalized_text for row in find_occurrences(db, "кто", MatchFilters())] == ["кто"]
    table = build_run_table(db, tokenize("кто-то еще"), MatchFilters())
    run = table[(0, 3)][video_id]
    assert run.n_words == 3
    assert run.first_word_ord == 0
    assert run.last_word_ord == 2


def test_find_occurrences_returns_every_video_that_says_the_word(
    db: Database, corpus: tuple[int, int]
) -> None:
    first, second = corpus
    found = find_occurrences(db, "все", MatchFilters())
    assert {row.video_id for row in found} == {first, second}
    assert {row.ord for row in found} == {0, 1}


def test_find_occurrences_coalesces_a_null_align_score(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=None)
    assert find_occurrences(db, "неизбежно", MatchFilters())[0].align_score == pytest.approx(
        C.ASSEMBLE_DEFAULT_ALIGN_SCORE
    )


def test_caption_words_are_never_eligible(db: Database) -> None:
    """Design §8: 'Caption-tier words are searchable but never assembled from.'"""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", source="caption")
    assert find_occurrences(db, "неизбежно", MatchFilters()) == []


def test_excluded_videos_are_dropped(db: Database, corpus: tuple[int, int]) -> None:
    first, second = corpus
    found = find_occurrences(db, "все", MatchFilters(exclude_video_ids=frozenset({first})))
    assert {row.video_id for row in found} == {second}


def test_min_align_score_drops_badly_anchored_words(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно", align_score=0.3)
    assert find_occurrences(db, "неизбежно", MatchFilters(min_align_score=0.5)) == []
    assert find_occurrences(db, "неизбежно", MatchFilters(min_align_score=0.2)) != []


def test_occurrences_are_capped_per_video_to_keep_sources_diverse(db: Database) -> None:
    crowded = add_video(db, external_id="VIDEO_A")
    for _ in range(10):
        add_words(db, crowded, "так так так так")
    quiet = add_video(db, external_id="VIDEO_B")
    add_words(db, quiet, "так")
    found = find_occurrences(db, "так", MatchFilters(max_seeds_per_video=2))
    assert len([row for row in found if row.video_id == crowded]) == 2
    assert quiet in {row.video_id for row in found}


def test_occurrences_are_ordered_deterministically(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "так так так")
    db.conn.execute("UPDATE words SET align_score = 0.99 WHERE ord = 2")
    db.conn.commit()
    found = find_occurrences(db, "так", MatchFilters())
    assert [row.ord for row in found] == [2, 0, 1]
    assert find_occurrences(db, "так", MatchFilters()) == found


def test_run_table_records_every_prefix_length(db: Database, corpus: tuple[int, int]) -> None:
    first, _second = corpus
    table = build_run_table(db, tokenize(TARGET), MatchFilters())
    for length in range(1, 6):
        assert first in table[(0, length)], length
    assert table[(0, 5)][first].text == "мы все понимаем что это"
    assert (0, 6) not in table  # video A never says "неизбежно"


def test_run_table_finds_the_long_run_starting_at_position_one(
    db: Database, corpus: tuple[int, int]
) -> None:
    _first, second = corpus
    table = build_run_table(db, tokenize(TARGET), MatchFilters())
    run = table[(1, 5)][second]
    assert run.text == "все понимаем что это неизбежно"
    assert run.first_word_ord == 0
    assert run.last_word_ord == 4
    assert run.n_words == 5


def test_a_run_carries_the_span_of_its_words(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    run = table[(0, 2)][video_id]
    assert run.start_ms == 0
    assert run.end_ms == 2 * WORD_MS + GAP_MS


def test_a_run_does_not_span_a_long_silence(db: Database) -> None:
    """Design §9 gives pause length to the renderer; a fragment must not
    smuggle in a pause the target sentence does not have."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы")
    add_words(db, video_id, "все", start_ms=WORD_MS + 5_000)
    table = build_run_table(db, ("мы", "все"), MatchFilters(max_internal_gap_ms=800))
    assert (0, 2) not in table
    assert video_id in table[(0, 1)]
    relaxed = build_run_table(db, ("мы", "все"), MatchFilters(max_internal_gap_ms=6_000))
    assert video_id in relaxed[(0, 2)]


def test_a_run_does_not_span_a_speaker_change(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host = add_video_speaker(db, video_id, "SPEAKER_00")
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы", video_speaker_id=host)
    add_words(db, video_id, "все", start_ms=WORD_MS + GAP_MS, video_speaker_id=guest)
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    assert (0, 2) not in table
    assert video_id in table[(0, 1)]


def test_an_unknown_word_produces_no_entry_at_its_position(
    db: Database, corpus: tuple[int, int]
) -> None:
    tokens = tokenize("мы все зеленеем")
    table = build_run_table(db, tokens, MatchFilters())
    assert (2, 1) not in table
    assert (0, 3) not in table


def test_max_run_words_caps_the_longest_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    text = " ".join(["так"] * 10)
    add_words(db, video_id, text)
    table = build_run_table(db, tokenize(text), MatchFilters(max_run_words=3))
    assert (0, 3) in table
    assert (0, 4) not in table


def test_the_best_run_per_video_wins_on_alignment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все", align_score=0.4)
    add_words(db, video_id, "мы все", start_ms=100_000, align_score=0.95)
    table = build_run_table(db, ("мы", "все"), MatchFilters())
    assert table[(0, 2)][video_id].first_word_ord == 2


def test_a_speaker_filter_restricts_the_walk(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host = add_video_speaker(db, video_id, "SPEAKER_00")
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы все", video_speaker_id=host)
    add_words(db, video_id, "мы все", start_ms=100_000, video_speaker_id=guest)
    filters = MatchFilters(video_speaker_ids=frozenset({guest}))
    table = build_run_table(db, ("мы", "все"), filters)
    assert table[(0, 2)][video_id].first_word_ord == 2
```

`add_acoustics` is imported here but first used in Task 4; leave the import in place rather than adding and removing it.

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.assemble'`.

- [ ] **Step 3: Create the package marker**

Create `rytp/assemble/__init__.py`:

```python
"""Assembly: target sentence in, cut list out (design §8).

Task 9 turns this module into the orchestrator. Until then it is only the
package marker, and it must stay importable without pulling in anything
heavy — assembly is pure logic over SQLite.
"""

from __future__ import annotations
```

- [ ] **Step 4: Write `rytp/assemble/match.py`**

```python
"""Walk a target sentence against the corpus (design §7 "Assembly matching").

The corpus read is two SQL shapes and nothing else. ``words(normalized_text)``
finds where a token was said; ``words(video_id, ord)`` steps forward from
there. No n-gram table, no FTS, no fuzziness: an assembly match is exact or
it is not a match.

Extension is level-synchronous. Every candidate that survived the previous
token advances by one ordinal in one batched query, so the number of
queries is bounded by the length of the target rather than by how many
times its first word was ever said. A run is recorded at every prefix
length, because the coverage search below needs the short edges as well as
the long ones.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rytp import constants as C
from rytp.db import Database
from rytp.models import normalize_text

__all__ = [
    "CandidateRun",
    "MatchFilters",
    "RunTable",
    "WordRow",
    "build_run_table",
    "find_occurrences",
    "tokenize",
]


@dataclass(frozen=True)
class WordRow:
    """One cuttable word. ``align_score`` is already coalesced, never None."""

    video_id: int
    ord: int
    start_ms: int
    end_ms: int
    text: str
    normalized_text: str
    align_score: float
    video_speaker_id: int | None


@dataclass(frozen=True)
class CandidateRun:
    """A contiguous run of cuttable words in one video, matching the target."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    n_words: int
    first_align: float
    last_align: float
    mean_align: float
    video_speaker_id: int | None


@dataclass(frozen=True)
class MatchFilters:
    """Everything that narrows what may be cut (design §8 "Controls")."""

    exclude_video_ids: frozenset[int] = frozenset()
    video_speaker_ids: frozenset[int] | None = None
    min_align_score: float = C.ASSEMBLE_MIN_ALIGN_SCORE
    max_internal_gap_ms: int = C.ASSEMBLE_MAX_INTERNAL_GAP_MS
    max_run_words: int = C.ASSEMBLE_MAX_RUN_WORDS
    max_occurrences: int = C.ASSEMBLE_MAX_OCCURRENCES_PER_TOKEN
    max_seeds_per_video: int = C.ASSEMBLE_MAX_SEEDS_PER_VIDEO


#: (target position, run length in words) -> video_id -> the best run there.
RunTable = dict[tuple[int, int], dict[int, CandidateRun]]

_SELECT = (
    "SELECT video_id, ord, start_ms, end_ms, text, normalized_text, "
    "COALESCE(align_score, :default_align) AS align_score, video_speaker_id "
)


def tokenize(target: str) -> tuple[str, ...]:
    """The target as the tokens the corpus stores, in order.

    Uses the same normalizer the ``normalized_text`` column was written
    with, so a match is a string equality and never a guess.
    """
    return tuple(normalize_text(target).split())


def _int_list(values: Iterable[int]) -> str:
    """Inline a set of integers into SQL.

    Safe because every element is coerced with ``int()`` first, and
    necessary because the exclusion list has no fixed arity and would
    otherwise need a parameter per element on every query.
    """
    return ", ".join(str(int(value)) for value in values)


def _eligibility_sql(filters: MatchFilters) -> str:
    """The shared WHERE tail: cuttable, in scope, well enough anchored.

    Contracts §3: "Cuttable is defined as ``source = 'aligned'`` and
    nothing else may be cut." ``end_ms IS NOT NULL`` is implied by the
    table's CHECK constraint but stated anyway, because a cut without an
    end is not a cut.
    """
    clauses = [
        "source = 'aligned'",
        "end_ms IS NOT NULL",
        "COALESCE(align_score, :default_align) >= :min_align",
    ]
    if filters.exclude_video_ids:
        clauses.append(f"video_id NOT IN ({_int_list(filters.exclude_video_ids)})")
    if filters.video_speaker_ids is not None:
        clauses.append(f"video_speaker_id IN ({_int_list(filters.video_speaker_ids)})")
    return " AND ".join(clauses)


def _row(record: sqlite3.Row) -> WordRow:
    """One result row, with every value given its real type once."""
    speaker = record["video_speaker_id"]
    return WordRow(
        video_id=int(record["video_id"]),
        ord=int(record["ord"]),
        start_ms=int(record["start_ms"]),
        end_ms=int(record["end_ms"]),
        text=str(record["text"]),
        normalized_text=str(record["normalized_text"]),
        align_score=float(record["align_score"]),
        video_speaker_id=None if speaker is None else int(speaker),
    )


def find_occurrences(db: Database, token: str, filters: MatchFilters) -> list[WordRow]:
    """Every eligible place ``token`` was said, best-anchored first.

    Two caps apply, and the per-video one is applied *inside* the query.
    Applying it afterwards would be useless: a single talkative video
    would fill the overall limit before any other source was seen, and
    the consistency knob would then have nothing to choose between.
    """
    sql = (
        f"{_SELECT}FROM (SELECT *, ROW_NUMBER() OVER ("
        "  PARTITION BY video_id"
        "  ORDER BY COALESCE(align_score, :default_align) DESC, ord"
        ") AS seed_rank FROM words"
        f" WHERE normalized_text = :token AND {_eligibility_sql(filters)}"
        ") WHERE seed_rank <= :max_seeds"
        " ORDER BY align_score DESC, video_id, ord"
        " LIMIT :max_occurrences"
    )
    cursor = db.conn.execute(
        sql,
        {
            "token": token,
            "default_align": C.ASSEMBLE_DEFAULT_ALIGN_SCORE,
            "min_align": filters.min_align_score,
            "max_seeds": filters.max_seeds_per_video,
            "max_occurrences": filters.max_occurrences,
        },
    )
    return [_row(record) for record in cursor]


def _fetch_next(
    db: Database, wanted: Sequence[tuple[int, int]], filters: MatchFilters
) -> dict[tuple[int, int], WordRow]:
    """One batched read of the ``(video_id, ord)`` pairs the frontier needs.

    SQLite has supported row values in ``IN (VALUES …)`` since 3.15, and
    3.11 bundles far newer; the composite ``words(video_id, ord)`` index
    serves it directly. Batched because the oldest host-parameter limit
    is 999 and each pair spends two.
    """
    found: dict[tuple[int, int], WordRow] = {}
    tail = _eligibility_sql(filters)
    for start in range(0, len(wanted), C.ASSEMBLE_SQL_BATCH):
        batch = wanted[start : start + C.ASSEMBLE_SQL_BATCH]
        pairs = ", ".join(f"(:v{index}, :o{index})" for index in range(len(batch)))
        params: dict[str, object] = {
            "default_align": C.ASSEMBLE_DEFAULT_ALIGN_SCORE,
            "min_align": filters.min_align_score,
        }
        for index, (video_id, ordinal) in enumerate(batch):
            params[f"v{index}"] = video_id
            params[f"o{index}"] = ordinal
        sql = f"{_SELECT}FROM words WHERE (video_id, ord) IN (VALUES {pairs}) AND {tail}"
        for record in db.conn.execute(sql, params):
            row = _row(record)
            found[(row.video_id, row.ord)] = row
    return found


def _run_from(words: Sequence[WordRow]) -> CandidateRun:
    """Freeze a list of consecutive words into the run they form."""
    scores = [word.align_score for word in words]
    return CandidateRun(
        video_id=words[0].video_id,
        first_word_ord=words[0].ord,
        last_word_ord=words[-1].ord,
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        text=" ".join(word.text for word in words),
        n_words=len(words),
        first_align=words[0].align_score,
        last_align=words[-1].align_score,
        mean_align=sum(scores) / len(scores),
        video_speaker_id=words[0].video_speaker_id,
    )


def _better(candidate: CandidateRun, incumbent: CandidateRun) -> bool:
    """Which of two runs of the same shape in the same video to keep.

    Two runs of equal length in the same video differ in cost only
    through their alignment terms (see :mod:`rytp.assemble.score`), so
    ranking by alignment here is ranking by cost — without this module
    needing to know the knob. The ordinal breaks the last tie so the
    answer never depends on row order.
    """
    key_new = (
        -candidate.mean_align,
        -(candidate.first_align + candidate.last_align),
        candidate.first_word_ord,
    )
    key_old = (
        -incumbent.mean_align,
        -(incumbent.first_align + incumbent.last_align),
        incumbent.first_word_ord,
    )
    return key_new < key_old


def _record(table: RunTable, position: int, run: CandidateRun) -> None:
    bucket = table.setdefault((position, run.n_words), {})
    incumbent = bucket.get(run.video_id)
    if incumbent is None or _better(run, incumbent):
        bucket[run.video_id] = run


def _extends(previous: WordRow, nxt: WordRow, filters: MatchFilters) -> bool:
    """May a run step from ``previous`` to ``nxt``?

    Two reasons to stop that ordinal adjacency alone does not catch: a
    silence long enough that the cut would carry a pause the target does
    not have, and a change of voice mid-fragment, which is wrong however
    the run is filtered.
    """
    if nxt.start_ms - previous.end_ms > filters.max_internal_gap_ms:
        return False
    if (
        previous.video_speaker_id is not None
        and nxt.video_speaker_id is not None
        and previous.video_speaker_id != nxt.video_speaker_id
    ):
        return False
    return True


def build_run_table(db: Database, tokens: Sequence[str], filters: MatchFilters) -> RunTable:
    """Every contiguous corpus run that matches the target, at every prefix.

    Occurrence lists are cached per token, so a target that repeats a word
    reads the corpus for it once.
    """
    table: RunTable = {}
    max_words = max(1, filters.max_run_words)
    seeds: dict[str, list[WordRow]] = {}
    for position, token in enumerate(tokens):
        if token not in seeds:
            seeds[token] = find_occurrences(db, token, filters)
        alive: list[list[WordRow]] = []
        for occurrence in seeds[token]:
            alive.append([occurrence])
            _record(table, position, _run_from([occurrence]))
        step = 1
        while alive and step < max_words and position + step < len(tokens):
            expected = tokens[position + step]
            wanted = [(run[-1].video_id, run[-1].ord + 1) for run in alive]
            found = _fetch_next(db, wanted, filters)
            grown: list[list[WordRow]] = []
            for run in alive:
                nxt = found.get((run[-1].video_id, run[-1].ord + 1))
                if nxt is None or nxt.normalized_text != expected:
                    continue
                if not _extends(run[-1], nxt, filters):
                    continue
                longer = [*run, nxt]
                _record(table, position, _run_from(longer))
                grown.append(longer)
            alive = grown
            step += 1
    return table
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: every test in the file passes.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_match.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/assemble tests/test_assemble_match.py
git commit -m "feat: add the corpus pointer walk and candidate run table"
```

---

### Task 3: Acoustics, the consistency knob, and fragment cost

Every number the coverage search minimises, and nothing else. `score.py` is a leaf: it must not import `match.py`, so what it needs from a run is expressed as a read-only `Protocol` that `CandidateRun` satisfies structurally.

Design §8: "Scoring balances run length against acoustic consistency, exposed as **one knob** from 'fewest seams' to 'most consistent sound', defaulting toward fewer seams. […] Consistency is judged from `video_acoustics`, and preferring few distinct source videos is a strong and cheap proxy." The knob is therefore not a formula with a free parameter — it is a straight line between two named weight profiles, and the "few distinct sources" proxy is the `switch` weight, charged every time the cut list changes video.

**Files:**
- Create: `rytp/assemble/score.py`
- Test: `tests/test_assemble_score.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models.InvalidInputError`, `rytp.constants`.
- Produces:
  - `Weights(seam: float, switch: float, acoustic: float)` — frozen.
  - `RunLike` — read-only Protocol with `n_words`, `first_align`, `last_align`, `mean_align`.
  - `AcousticsMap = dict[int, dict[str, float]]`
  - `weights_for(consistency: float) -> Weights`
  - `load_acoustics(db, video_ids: Iterable[int]) -> AcousticsMap`
  - `acoustic_distance(first: Mapping[str, float] | None, second: Mapping[str, float] | None) -> float`
  - `fragment_cost(run: RunLike, weights: Weights) -> float`
  - `transition_cost(previous_video: int | None, video: int, weights: Weights, acoustics: AcousticsMap) -> float`
  - `jitter(seed: int, video_id: int, first_word_ord: int, position: int) -> float`
  - `rounded(cost: float) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_score.py`:

```python
"""One knob, two endpoint profiles, and the costs the DP minimises."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rytp import constants as C
from rytp.assemble.score import (
    Weights,
    acoustic_distance,
    fragment_cost,
    jitter,
    load_acoustics,
    rounded,
    transition_cost,
    weights_for,
)
from rytp.db import Database
from rytp.models import InvalidInputError
from tests.assembly_corpus import add_acoustics, add_video


@dataclass(frozen=True)
class FakeRun:
    """Structurally a CandidateRun, without importing the matcher."""

    n_words: int = 3
    first_align: float = 1.0
    last_align: float = 1.0
    mean_align: float = 1.0


def test_the_knob_ends_are_the_two_named_profiles() -> None:
    assert weights_for(0.0) == Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    assert weights_for(1.0) == Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)


def test_the_knob_is_a_straight_line_between_them() -> None:
    middle = weights_for(0.5)
    low = Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    high = Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)
    assert middle.seam == pytest.approx((low.seam + high.seam) / 2)
    assert middle.switch == pytest.approx((low.switch + high.switch) / 2)
    assert middle.acoustic == pytest.approx((low.acoustic + high.acoustic) / 2)


def test_turning_the_knob_up_makes_seams_cheaper_and_switches_dearer() -> None:
    assert weights_for(1.0).seam < weights_for(0.0).seam
    assert weights_for(1.0).switch > weights_for(0.0).switch
    assert weights_for(1.0).acoustic > weights_for(0.0).acoustic


def test_the_seam_weight_never_reaches_zero() -> None:
    """Free seams would shred a long run into single words for no gain."""
    assert weights_for(1.0).seam > 0.0


def test_the_knob_rejects_a_value_off_the_dial() -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        weights_for(1.5)
    with pytest.raises(InvalidInputError, match="consistency"):
        weights_for(-0.1)


def test_a_perfectly_aligned_fragment_costs_exactly_one_seam() -> None:
    weights = weights_for(0.0)
    assert fragment_cost(FakeRun(), weights) == pytest.approx(weights.seam)


def test_a_badly_anchored_fragment_costs_more() -> None:
    weights = weights_for(0.0)
    good = fragment_cost(FakeRun(first_align=0.95, last_align=0.95, mean_align=0.95), weights)
    bad = fragment_cost(FakeRun(first_align=0.2, last_align=0.2, mean_align=0.2), weights)
    assert bad > good


def test_bad_edges_cost_more_than_a_bad_interior() -> None:
    """Only the first and last boundary are actually cut."""
    weights = weights_for(0.0)
    bad_edges = fragment_cost(FakeRun(first_align=0.2, last_align=0.2, mean_align=0.8), weights)
    bad_middle = fragment_cost(FakeRun(first_align=0.9, last_align=0.9, mean_align=0.4), weights)
    assert bad_edges > bad_middle


def test_staying_in_one_video_is_free() -> None:
    weights = weights_for(1.0)
    assert transition_cost(None, 1, weights, {}) == 0.0
    assert transition_cost(1, 1, weights, {}) == 0.0


def test_changing_video_costs_more_when_the_sources_sound_different() -> None:
    weights = weights_for(1.0)
    acoustics = {
        1: {"f0_mean": 120.0, "loudness_lufs": -23.0},
        2: {"f0_mean": 122.0, "loudness_lufs": -23.5},
        3: {"f0_mean": 230.0, "loudness_lufs": -14.0},
    }
    near = transition_cost(1, 2, weights, acoustics)
    far = transition_cost(1, 3, weights, acoustics)
    assert 0 < near < far


def test_at_fewest_seams_acoustics_do_not_enter_the_price() -> None:
    weights = weights_for(0.0)
    acoustics = {1: {"f0_mean": 120.0}, 2: {"f0_mean": 400.0}}
    assert transition_cost(1, 2, weights, acoustics) == pytest.approx(weights.switch)


def test_identical_acoustics_are_zero_apart() -> None:
    profile = {"f0_mean": 120.0, "noise_floor_db": -60.0}
    assert acoustic_distance(profile, dict(profile)) == 0.0


def test_an_unmeasured_video_is_middling_rather_than_excluded() -> None:
    assert acoustic_distance(None, {"f0_mean": 120.0}) == C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    assert acoustic_distance({}, {}) == C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE


def test_distance_is_symmetric_and_capped_per_field() -> None:
    near = {"f0_mean": 120.0, "loudness_lufs": -23.0}
    far = {"f0_mean": 9_000.0, "loudness_lufs": -23.0}
    assert acoustic_distance(near, far) == acoustic_distance(far, near)
    # one absurd field is clamped, so the shared field still counts for half
    assert acoustic_distance(near, far) == pytest.approx(C.ASSEMBLE_ACOUSTIC_DISTANCE_CAP / 2)


def test_only_fields_both_videos_have_are_compared() -> None:
    assert acoustic_distance({"f0_mean": 120.0, "reverb_proxy": 0.9}, {"f0_mean": 120.0}) == 0.0


def test_load_acoustics_reads_rows_and_skips_the_missing(db: Database) -> None:
    measured = add_video(db, external_id="VIDEO_A")
    unmeasured = add_video(db, external_id="VIDEO_B")
    add_acoustics(db, measured, f0_mean=180.0)
    loaded = load_acoustics(db, [measured, unmeasured])
    assert loaded[measured]["f0_mean"] == 180.0
    assert unmeasured not in loaded


def test_load_acoustics_skips_a_null_column(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_acoustics(db, video_id)
    db.conn.execute(
        "UPDATE video_acoustics SET reverb_proxy = NULL WHERE video_id = ?", (video_id,)
    )
    db.conn.commit()
    assert "reverb_proxy" not in load_acoustics(db, [video_id])[video_id]


def test_load_acoustics_on_no_videos_is_empty(db: Database) -> None:
    assert load_acoustics(db, []) == {}


def test_seed_zero_means_no_jitter_at_all() -> None:
    assert jitter(0, 1, 0, 0) == 0.0


def test_jitter_is_stable_across_processes_and_bounded() -> None:
    """blake2b, not hash(): Python salts str hashing per process."""
    value = jitter(7, 3, 120, 2)
    assert value == jitter(7, 3, 120, 2)
    assert 0.0 <= value <= C.ASSEMBLE_SEED_JITTER
    assert jitter(7, 3, 120, 2) == pytest.approx(0.03881913275923114)


def test_jitter_separates_candidates_and_seeds() -> None:
    assert jitter(7, 3, 120, 2) != jitter(7, 4, 120, 2)
    assert jitter(7, 3, 120, 2) != jitter(8, 3, 120, 2)


def test_rounded_makes_float_noise_tie() -> None:
    assert rounded(1.0) == rounded(1.0 + 1e-12)
    assert rounded(1.0) != rounded(1.1)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_score.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.assemble.score'`.

- [ ] **Step 3: Write `rytp/assemble/score.py`**

```python
"""What a cut list costs (design §8 "Scoring").

One knob, ``consistency``, runs from 0.0 "fewest seams" to 1.0 "most
consistent sound". It does not parameterise a formula; it picks a point on
a straight line between two named weight profiles in
:mod:`rytp.constants`, so retuning later — design §11, M4 — is editing two
triples rather than re-deriving anything.

Three costs, all in the same units, all minimised together:

``seam``
    Paid once per fragment. This is the "fewer and longer fragments" force.
``switch``
    Paid whenever the cut list changes source video. This is design §8's
    "preferring few distinct source videos is a strong and cheap proxy",
    priced as an audible event rather than counted as a set size.
``acoustic``
    Multiplies how differently the two videos actually sound, measured
    from ``video_acoustics``. Zero at the "fewest seams" end: a short mix
    tolerates many sources.

This module is a leaf. It must not import :mod:`rytp.assemble.match` —
what scoring needs from a run is the read-only :class:`RunLike` protocol
below, which ``CandidateRun`` satisfies structurally.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError

__all__ = [
    "AcousticsMap",
    "RunLike",
    "Weights",
    "acoustic_distance",
    "fragment_cost",
    "jitter",
    "load_acoustics",
    "rounded",
    "transition_cost",
    "weights_for",
]

#: video_id -> the acoustic fields that video actually has measured.
AcousticsMap = dict[int, dict[str, float]]


class RunLike(Protocol):
    """The part of a candidate run that scoring looks at."""

    @property
    def n_words(self) -> int: ...

    @property
    def first_align(self) -> float: ...

    @property
    def last_align(self) -> float: ...

    @property
    def mean_align(self) -> float: ...


@dataclass(frozen=True)
class Weights:
    """One point on the knob."""

    seam: float
    switch: float
    acoustic: float


def _lerp(low: float, high: float, position: float) -> float:
    return low + (high - low) * position


def weights_for(consistency: float) -> Weights:
    """The weight profile at ``consistency`` on the 0..1 dial."""
    if not 0.0 <= consistency <= 1.0:
        raise InvalidInputError(
            f"consistency must be between 0.0 (fewest seams) and 1.0 "
            f"(most consistent sound); got {consistency}"
        )
    low = Weights(*C.ASSEMBLE_WEIGHTS_FEWEST_SEAMS)
    high = Weights(*C.ASSEMBLE_WEIGHTS_MOST_CONSISTENT)
    return Weights(
        seam=_lerp(low.seam, high.seam, consistency),
        switch=_lerp(low.switch, high.switch, consistency),
        acoustic=_lerp(low.acoustic, high.acoustic, consistency),
    )


def load_acoustics(db: Database, video_ids: Iterable[int]) -> AcousticsMap:
    """Read ``video_acoustics`` for the videos in play.

    A video with no row, or a row whose column is NULL, simply does not
    appear — an unmeasured source is unknown, not bad, and
    :func:`acoustic_distance` treats it that way.
    """
    wanted = sorted({int(video_id) for video_id in video_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(video_id) for video_id in wanted)
    columns = ", ".join(C.ASSEMBLE_ACOUSTIC_SCALES)
    loaded: AcousticsMap = {}
    for row in db.conn.execute(
        f"SELECT video_id, {columns} FROM video_acoustics WHERE video_id IN ({inline})"
    ):
        profile = {
            name: float(row[name])
            for name in C.ASSEMBLE_ACOUSTIC_SCALES
            if row[name] is not None
        }
        if profile:
            loaded[int(row["video_id"])] = profile
    return loaded


def acoustic_distance(
    first: Mapping[str, float] | None, second: Mapping[str, float] | None
) -> float:
    """How differently two videos sound, on a 0..1 scale.

    Each field is scaled by the difference that counts as one unit
    (``ASSEMBLE_ACOUSTIC_SCALES``) and clamped *before* averaging, so one
    wild field cannot swamp the rest — a bad pitch estimate should not
    make two otherwise identical recordings look incompatible. Only
    fields both videos have are compared.
    """
    if not first or not second:
        return C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    shared = [name for name in C.ASSEMBLE_ACOUSTIC_SCALES if name in first and name in second]
    if not shared:
        return C.ASSEMBLE_UNKNOWN_ACOUSTIC_DISTANCE
    total = 0.0
    for name in shared:
        raw = abs(first[name] - second[name]) / C.ASSEMBLE_ACOUSTIC_SCALES[name]
        total += min(raw, C.ASSEMBLE_ACOUSTIC_DISTANCE_CAP)
    return total / len(shared)


def fragment_cost(run: RunLike, weights: Weights) -> float:
    """What taking this run as one fragment costs.

    One seam, plus what its alignment is worth. The two align terms are
    separate on purpose: the first and last words are the boundaries that
    actually get cut, while the interior only tells us how much to trust
    that the fragment says what the index claims (design §3, "Precision
    beats recall for assembly").
    """
    edges = (2.0 - run.first_align - run.last_align) / 2.0
    interior = 1.0 - run.mean_align
    return (
        weights.seam
        + C.ASSEMBLE_EDGE_ALIGN_WEIGHT * edges
        + C.ASSEMBLE_MEAN_ALIGN_WEIGHT * interior
    )


def transition_cost(
    previous_video: int | None,
    video: int,
    weights: Weights,
    acoustics: AcousticsMap,
) -> float:
    """What following ``previous_video`` with ``video`` costs.

    Zero for the first fragment and for staying put. This is the term
    that makes "few distinct sources" an objective rather than a wish.
    """
    if previous_video is None or previous_video == video:
        return 0.0
    distance = acoustic_distance(acoustics.get(previous_video), acoustics.get(video))
    return weights.switch + weights.acoustic * distance


def jitter(seed: int, video_id: int, first_word_ord: int, position: int) -> float:
    """A tiny, reproducible cost nudge keyed by the seed and the candidate.

    Design §8: same input, same output; "a user-supplied seed shakes up
    choices among near-equal candidates". Seed 0 disables it entirely.
    ``blake2b`` rather than ``hash()`` because Python salts string hashing
    per process, which would make the same seed give different answers in
    different runs — the exact thing this is meant to prevent.
    """
    if seed == 0:
        return 0.0
    payload = f"{seed}:{video_id}:{first_word_ord}:{position}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / float(1 << 64)
    return C.ASSEMBLE_SEED_JITTER * fraction


def rounded(cost: float) -> float:
    """Costs, rounded so near-equal candidates genuinely tie.

    Without this, float noise silently decides between two candidates the
    scoring considers equivalent, and the explicit tie-break — or the
    seed — never gets a say.
    """
    return round(cost, C.ASSEMBLE_COST_DECIMALS)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_score.py -v`
Expected: every test in the file passes. The exact jitter constant was computed against this formula while writing the plan; if it disagrees, the formula was mistyped — fix the code, not the number.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_score.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/assemble/score.py tests/test_assemble_score.py
git commit -m "feat: add the consistency knob and assembly cost model"
```

---

### Task 4: The coverage DP

The creative core. Everything before this found what *could* be cut; this decides what *is*.

Read "Why a DP and not greedy-longest" above before starting — in particular, the arrival-cost trick in `plan_coverage` is not an optimisation to skip. Without it the inner loop is `O(N · L · V²)` and a sentence against a wide corpus takes tens of millions of operations.

**Files:**
- Modify: `rytp/assemble/match.py` (append)
- Test: `tests/test_assemble_match.py` (append)

**Interfaces:**
- Consumes: Task 2's `CandidateRun` / `RunTable`; `rytp.assemble.score.{Weights, AcousticsMap, fragment_cost, transition_cost, jitter, rounded}`; `rytp.models.InvalidInputError`.
- Produces:
  - `SLOT_FRAGMENT = "fragment"`, `SLOT_GAP = "gap"`
  - `ScoredRun(run: CandidateRun, cost: float)` — frozen.
  - `PlanSlot(kind, target_first, target_last, text, cost, run=None, alternatives=())` — frozen; `target_last` is **inclusive**; `run` is `None` exactly when `kind == SLOT_GAP`.
  - `Plan(target, tokens, slots, total_cost)` — frozen, with properties `fragments`, `gaps`, `source_video_ids`, `n_sources`.
  - `plan_coverage(target: str, tokens: Sequence[str], table: RunTable, *, weights: Weights, acoustics: AcousticsMap, seed: int = 0, max_alternatives: int = C.ASSEMBLE_MAX_ALTERNATIVES) -> Plan`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_assemble_match.py`:

```python
from rytp.assemble.match import SLOT_FRAGMENT, SLOT_GAP, Plan, plan_coverage
from rytp.assemble.score import load_acoustics, weights_for
from rytp.models import InvalidInputError


def make_plan(
    db: Database,
    target: str,
    *,
    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY,
    seed: int = 0,
    filters: MatchFilters | None = None,
) -> Plan:
    """Tokenize, walk, score, cover — what Task 9 will do for real."""
    tokens = tokenize(target)
    table = build_run_table(db, tokens, filters or MatchFilters())
    videos = {run.video_id for bucket in table.values() for run in bucket.values()}
    return plan_coverage(
        target,
        tokens,
        table,
        weights=weights_for(consistency),
        acoustics=load_acoustics(db, videos),
        seed=seed,
    )


def test_one_video_that_says_it_all_gives_one_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все понимаем")
    plan = make_plan(db, "мы все понимаем")
    assert len(plan.slots) == 1
    slot = plan.slots[0]
    assert slot.kind == SLOT_FRAGMENT
    assert slot.run is not None
    assert slot.run.video_id == video_id
    assert (slot.target_first, slot.target_last) == (0, 2)
    assert plan.n_sources == 1


def test_the_greedy_counterexample_from_the_plan(db: Database) -> None:
    """Greedy takes A's longer run and pays a source switch; the DP takes
    B's shorter run and covers the whole target from one video."""
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем что")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "мы все понимаем")
    add_words(db, second, "что это", start_ms=60_000)

    tokens = tokenize("мы все понимаем что это")
    table = build_run_table(db, tokens, MatchFilters())
    assert first in table[(0, 4)]  # greedy's four-word run exists
    assert (0, 5) not in table  # and nobody covers the whole target

    plan = make_plan(db, "мы все понимаем что это")
    assert len(plan.fragments) == 2
    assert plan.n_sources == 1
    assert plan.source_video_ids == (second,)


def test_the_knob_trades_fragments_for_sources(db: Database) -> None:
    """Three cuts in one video beat two cuts across two that sound nothing alike."""
    lone = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, lone, "мы все")
    add_words(db, lone, "понимаем это", start_ms=60_000)
    add_words(db, lone, "неизбежно", start_ms=120_000)
    wide = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, wide, "мы все понимаем это")
    tail = add_video(db, external_id="VIDEO_C", title="C")
    add_words(db, tail, "неизбежно")
    add_acoustics(db, lone, f0_mean=120.0)
    add_acoustics(db, wide, f0_mean=110.0, loudness_lufs=-23.0)
    add_acoustics(db, tail, f0_mean=260.0, loudness_lufs=-9.0)

    target = "мы все понимаем это неизбежно"
    seams = make_plan(db, target, consistency=0.0)
    assert len(seams.fragments) == 2
    assert seams.n_sources == 2

    consistent = make_plan(db, target, consistency=1.0)
    assert len(consistent.fragments) == 3
    assert consistent.n_sources == 1
    assert consistent.source_video_ids == (lone,)


def test_a_badly_anchored_run_loses_to_an_equal_length_clean_one(db: Database) -> None:
    sloppy = add_video(db, external_id="VIDEO_A")
    add_words(db, sloppy, "мы все", align_score=0.25)
    clean = add_video(db, external_id="VIDEO_B")
    add_words(db, clean, "мы все", align_score=0.95)
    plan = make_plan(db, "мы все", consistency=0.0)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.video_id == clean


def test_a_missing_word_becomes_a_gap_and_the_rest_is_still_cut(db: Database) -> None:
    """Design §8: 'report the gap plainly and render everything else'."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все зеленеем")
    assert [slot.kind for slot in plan.slots] == [SLOT_FRAGMENT, SLOT_GAP]
    gap = plan.gaps[0]
    assert gap.text == "зеленеем"
    assert (gap.target_first, gap.target_last) == (2, 2)
    assert gap.run is None
    assert len(plan.fragments) == 1


def test_every_word_missing_gives_a_gap_per_word_and_no_crash(db: Database) -> None:
    add_video(db, external_id="VIDEO_A")
    plan = make_plan(db, "мы все")
    assert [slot.kind for slot in plan.slots] == [SLOT_GAP, SLOT_GAP]
    assert [slot.text for slot in plan.slots] == ["мы", "все"]
    assert plan.n_sources == 0


def test_a_gap_is_never_preferred_to_a_real_fragment(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы", align_score=0.01)
    plan = make_plan(db, "мы", consistency=1.0)
    assert plan.slots[0].kind == SLOT_FRAGMENT
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.video_id == video_id


def test_slot_costs_add_up_to_the_total(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    add_words(db, video_id, "это", start_ms=60_000)
    plan = make_plan(db, "мы все это зеленеем")
    assert plan.total_cost == pytest.approx(sum(slot.cost for slot in plan.slots))


def test_alternatives_rank_the_other_sources_and_exclude_the_chosen_one(
    db: Database,
) -> None:
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все", align_score=0.95)
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все", align_score=0.80)
    third = add_video(db, external_id="VIDEO_C")
    add_words(db, third, "мы все", align_score=0.60)
    plan = make_plan(db, "мы все", consistency=0.0)
    chosen = plan.slots[0]
    assert chosen.run is not None
    assert chosen.run.video_id == first
    assert [scored.run.video_id for scored in chosen.alternatives] == [second, third]
    assert chosen.alternatives[0].cost < chosen.alternatives[1].cost


def test_alternatives_are_capped(db: Database) -> None:
    for index in range(6):
        video_id = add_video(db, external_id=f"VIDEO_{index}")
        add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все")
    assert len(plan.slots[0].alternatives) == C.ASSEMBLE_MAX_ALTERNATIVES


def test_the_same_input_gives_the_same_plan(db: Database) -> None:
    """Design §8: 'Determinism. Same input gives the same output.'"""
    for index in range(4):
        video_id = add_video(db, external_id=f"VIDEO_{index}")
        add_words(db, video_id, "мы все понимаем")
    assert make_plan(db, "мы все понимаем") == make_plan(db, "мы все понимаем")


def test_a_seed_shakes_up_a_tie(db: Database) -> None:
    """Design §8: 'A user-supplied seed shakes up choices among near-equal
    candidates.' Two identical sources: seed 0 always picks the same one,
    and some seeds pick the other."""
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все")
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все")
    unseeded = {
        make_plan(db, "мы все").slots[0].run.video_id  # type: ignore[union-attr]
        for _ in range(3)
    }
    assert unseeded == {first}
    seeded = {
        make_plan(db, "мы все", seed=seed).slots[0].run.video_id  # type: ignore[union-attr]
        for seed in range(1, 25)
    }
    assert seeded == {first, second}


def test_a_seed_cannot_overturn_a_real_preference(db: Database) -> None:
    good = add_video(db, external_id="VIDEO_A")
    add_words(db, good, "мы все", align_score=0.95)
    bad = add_video(db, external_id="VIDEO_B")
    add_words(db, bad, "мы все", align_score=0.20)
    chosen = {
        make_plan(db, "мы все", seed=seed).slots[0].run.video_id  # type: ignore[union-attr]
        for seed in range(1, 25)
    }
    assert chosen == {good}
    assert bad not in chosen


def test_an_empty_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        plan_coverage("   ", (), {}, weights=weights_for(0.0), acoustics={})
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: collection error — `ImportError: cannot import name 'plan_coverage'`.

- [ ] **Step 3: Extend the imports at the top of `rytp/assemble/match.py`**

```python
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from rytp import constants as C
from rytp.assemble.score import (
    AcousticsMap,
    Weights,
    fragment_cost,
    jitter,
    rounded,
    transition_cost,
)
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, normalize_text
```

and extend `__all__` with `"Plan"`, `"PlanSlot"`, `"SLOT_FRAGMENT"`, `"SLOT_GAP"`, `"ScoredRun"`, `"plan_coverage"`.

- [ ] **Step 4: Append the coverage search to `rytp/assemble/match.py`**

```python
#: Slot kinds. A cut list is one ordered list of these, fragments and
#: holes together, because the renderer has to walk the timeline in order.
SLOT_FRAGMENT: Final = "fragment"
SLOT_GAP: Final = "gap"

#: DP state for "no fragment emitted yet". An int, not None, so the state
#: keys stay sortable and iteration order stays deterministic.
_NO_SOURCE: Final = -1


@dataclass(frozen=True)
class ScoredRun:
    """A candidate the search did not take, and what it would have cost."""

    run: CandidateRun
    cost: float


@dataclass(frozen=True)
class PlanSlot:
    """One position in the output: a fragment, or a hole where a word was."""

    kind: str
    target_first: int
    target_last: int  # inclusive
    text: str
    cost: float
    run: CandidateRun | None = None
    alternatives: tuple[ScoredRun, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A whole target, covered."""

    target: str
    tokens: tuple[str, ...]
    slots: tuple[PlanSlot, ...]
    total_cost: float

    @property
    def fragments(self) -> tuple[PlanSlot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_FRAGMENT)

    @property
    def gaps(self) -> tuple[PlanSlot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_GAP)

    @property
    def source_video_ids(self) -> tuple[int, ...]:
        """Distinct source videos, in the order they are first used."""
        seen: list[int] = []
        for slot in self.fragments:
            if slot.run is not None and slot.run.video_id not in seen:
                seen.append(slot.run.video_id)
        return tuple(seen)

    @property
    def n_sources(self) -> int:
        return len(self.source_video_ids)


@dataclass(frozen=True)
class _Step:
    """How the search arrived at one DP node."""

    cost: float
    prev_position: int
    prev_source: int
    run: CandidateRun | None


def _step_key(step: _Step) -> tuple[float, int, int, int, int]:
    """A total order over arrivals at one node, so ties never depend on luck.

    Cost is rounded first: two candidates the cost model considers
    equivalent must actually tie, or float noise decides instead of the
    tie-break — and design §8 requires the same input to give the same
    output.
    """
    run = step.run
    return (
        rounded(step.cost),
        0 if run is not None else 1,
        run.video_id if run is not None else -1,
        run.first_word_ord if run is not None else -1,
        step.prev_position,
    )


def _relax(node: dict[int, _Step], source: int, step: _Step) -> None:
    incumbent = node.get(source)
    if incumbent is None or _step_key(step) < _step_key(incumbent):
        node[source] = step


def _arrival_costs(
    node: Sequence[tuple[int, _Step]],
    candidates: Iterable[int],
    weights: Weights,
    acoustics: AcousticsMap,
) -> dict[int, tuple[float, int]]:
    """For each candidate source, the cheapest way to arrive in it.

    The transition term depends on the pair of videos and not on how long
    the fragment is, so it is computed once per (previous source, next
    source) pair rather than once per edge. That is what turns the inner
    loop from O(N·L·V²) into O(N·(V² + L·V)).
    """
    arrivals: dict[int, tuple[float, int]] = {}
    for video_id in sorted(candidates):
        best_cost = math.inf
        best_source = _NO_SOURCE
        for source, step in node:
            previous = None if source == _NO_SOURCE else source
            cost = step.cost + transition_cost(previous, video_id, weights, acoustics)
            if (rounded(cost), source) < (rounded(best_cost), best_source):
                best_cost, best_source = cost, source
        arrivals[video_id] = (best_cost, best_source)
    return arrivals


def _alternatives(
    table: RunTable,
    position: int,
    length: int,
    chosen: _Step,
    weights: Weights,
    acoustics: AcousticsMap,
    seed: int,
    limit: int,
) -> tuple[ScoredRun, ...]:
    """The runs of the same shape the search did not take, ranked.

    Design §8: "Every fragment keeps its ranked alternatives so a choice
    can be swapped without re-running." They are priced against the
    fragment that actually precedes this one, so swapping one in is a
    like-for-like comparison rather than a context-free score.
    """
    previous = None if chosen.prev_source == _NO_SOURCE else chosen.prev_source
    taken = chosen.run.video_id if chosen.run is not None else None
    scored = [
        ScoredRun(
            run=run,
            cost=(
                fragment_cost(run, weights)
                + transition_cost(previous, video_id, weights, acoustics)
                + jitter(seed, video_id, run.first_word_ord, position)
            ),
        )
        for video_id, run in table.get((position, length), {}).items()
        if video_id != taken
    ]
    scored.sort(key=lambda item: (rounded(item.cost), item.run.video_id, item.run.first_word_ord))
    return tuple(scored[:limit])


def plan_coverage(
    target: str,
    tokens: Sequence[str],
    table: RunTable,
    *,
    weights: Weights,
    acoustics: AcousticsMap,
    seed: int = 0,
    max_alternatives: int = C.ASSEMBLE_MAX_ALTERNATIVES,
) -> Plan:
    """Choose a segmentation and a source for every slot, together.

    A shortest path over ``(target position, last source video)``. The gap
    edge — step over one token for ``ASSEMBLE_GAP_COST`` — always exists,
    so a target the corpus cannot say still produces a plan: design §8
    says "report the gap plainly and render everything else", not "fail".
    """
    if not tokens:
        raise InvalidInputError("the target is empty: there is nothing to assemble")
    count = len(tokens)

    lengths: dict[int, list[int]] = {}
    for position, length in table:
        lengths.setdefault(position, []).append(length)
    for available in lengths.values():
        available.sort()

    best: list[dict[int, _Step]] = [{} for _ in range(count + 1)]
    best[0][_NO_SOURCE] = _Step(cost=0.0, prev_position=-1, prev_source=_NO_SOURCE, run=None)

    for position in range(count):
        node = sorted(best[position].items())
        if not node:
            continue
        for source, step in node:
            _relax(
                best[position + 1],
                source,
                _Step(
                    cost=step.cost + C.ASSEMBLE_GAP_COST,
                    prev_position=position,
                    prev_source=source,
                    run=None,
                ),
            )
        available = lengths.get(position, [])
        candidates = {video_id for length in available for video_id in table[(position, length)]}
        arrivals = _arrival_costs(node, candidates, weights, acoustics)
        for length in available:
            for video_id, run in sorted(table[(position, length)].items()):
                arrival, source = arrivals[video_id]
                _relax(
                    best[position + length],
                    video_id,
                    _Step(
                        cost=(
                            arrival
                            + fragment_cost(run, weights)
                            + jitter(seed, video_id, run.first_word_ord, position)
                        ),
                        prev_position=position,
                        prev_source=source,
                        run=run,
                    ),
                )

    final_source, final_step = min(
        best[count].items(), key=lambda item: (_step_key(item[1]), item[0])
    )

    walked: list[_Step] = []
    position, source = count, final_source
    while position > 0:
        step = best[position][source]
        walked.append(step)
        position, source = step.prev_position, step.prev_source
    walked.reverse()

    slots: list[PlanSlot] = []
    for step in walked:
        start = step.prev_position
        end = start + (1 if step.run is None else step.run.n_words)
        edge_cost = step.cost - best[start][step.prev_source].cost
        if step.run is None:
            slots.append(
                PlanSlot(
                    kind=SLOT_GAP,
                    target_first=start,
                    target_last=end - 1,
                    text=tokens[start],
                    cost=edge_cost,
                )
            )
        else:
            slots.append(
                PlanSlot(
                    kind=SLOT_FRAGMENT,
                    target_first=start,
                    target_last=end - 1,
                    text=step.run.text,
                    cost=edge_cost,
                    run=step.run,
                    alternatives=_alternatives(
                        table,
                        start,
                        step.run.n_words,
                        step,
                        weights,
                        acoustics,
                        seed,
                        max_alternatives,
                    ),
                )
            )

    return Plan(
        target=target,
        tokens=tuple(tokens),
        slots=tuple(slots),
        total_cost=final_step.cost,
    )
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: every test in the file passes.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_match.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/assemble/match.py tests/test_assemble_match.py
git commit -m "feat: add the coverage search over target positions and sources"
```

---

### Task 5: Padding, clamped to what is actually there

Design §8's last control: "add padding after a chosen word". A cut that stops exactly on a measured word boundary can clip the release of a final consonant; a few tens of milliseconds of tail usually sounds better. The padding is applied here, at plan time, and **baked into `end_ms`**, so the cut list stays a plain list of in and out points that a human can read and a renderer can obey without knowing this control exists.

Clamping matters as much as the padding. Running past the start of the next word in the source video would smuggle a word into the fragment that the target never asked for — a correctness bug, not a taste one.

**Files:**
- Modify: `rytp/assemble/match.py` (append)
- Test: `tests/test_assemble_match.py` (append)

**Interfaces:**
- Consumes: Task 4's `Plan`, `PlanSlot`, `ScoredRun`; `rytp.db.Database`.
- Produces: `pad_fragments(db: Database, plan: Plan, pad_ms: int) -> Plan`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_assemble_match.py`:

```python
from rytp.assemble.match import pad_fragments


def test_zero_padding_changes_nothing(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = make_plan(db, "мы все")
    assert pad_fragments(db, plan, 0) == plan


def test_padding_extends_the_tail(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=600_000)
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 50)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS + 50


def test_padding_stops_at_the_next_word(db: Database) -> None:
    """Otherwise the fragment quietly says one more word than the target."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все это")
    plan = pad_fragments(db, make_plan(db, "мы все"), 5_000)
    run = plan.slots[0].run
    assert run is not None
    next_start = db.conn.execute(
        "SELECT start_ms FROM words WHERE video_id = ? AND ord = 2", (video_id,)
    ).fetchone()["start_ms"]
    assert run.end_ms == next_start


def test_padding_stops_at_the_end_of_the_video(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=1_000)
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 5_000)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 1_000


def test_padding_is_unclamped_when_nothing_bounds_it(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все"), 250)
    assert plan.slots[0].run is not None
    assert plan.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS + 250


def test_padding_never_shortens_a_fragment(db: Database) -> None:
    """A clamp below the measured end must not drag the cut backwards.

    Possible whenever a stored duration is shorter than the last word's
    end — a rounded probe, or a re-downloaded rendition.
    """
    video_id = add_video(db, external_id="VIDEO_A", duration_ms=500)
    add_words(db, video_id, "мы все")
    unpadded = make_plan(db, "мы все")
    padded = pad_fragments(db, unpadded, 400)
    assert unpadded.slots[0].run is not None
    assert padded.slots[0].run is not None
    assert unpadded.slots[0].run.end_ms == 2 * WORD_MS + GAP_MS
    assert padded.slots[0].run.end_ms == unpadded.slots[0].run.end_ms


def test_alternatives_are_padded_too(db: Database) -> None:
    """A swapped-in alternative must be usable without re-running."""
    first = add_video(db, external_id="VIDEO_A")
    add_words(db, first, "мы все", align_score=0.95)
    second = add_video(db, external_id="VIDEO_B")
    add_words(db, second, "мы все", align_score=0.80)
    plan = pad_fragments(db, make_plan(db, "мы все", consistency=0.0), 50)
    assert plan.slots[0].alternatives[0].run.end_ms == 2 * WORD_MS + GAP_MS + 50


def test_padding_leaves_gaps_alone(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plan = pad_fragments(db, make_plan(db, "мы все зеленеем"), 50)
    assert plan.gaps[0].run is None
    assert plan.gaps[0].text == "зеленеем"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_match.py -k pad -v`
Expected: collection error — `ImportError: cannot import name 'pad_fragments'`.

- [ ] **Step 3: Append padding to `rytp/assemble/match.py`**

Change the dataclasses import to `from dataclasses import dataclass, replace`, add `"pad_fragments"` to `__all__`, and append:

```python
def _tail_limits(db: Database, runs: Sequence[CandidateRun]) -> dict[tuple[int, int], int]:
    """How far past its last word each run may reach, keyed by (video, ord).

    Two bounds, whichever is nearer: the start of the next word in that
    video, and the video's own duration. A video with neither — no next
    word and no known duration — is unbounded here; ffmpeg will stop at
    the end of the file regardless, and guessing a limit would be worse
    than letting the renderer clamp it.
    """
    if not runs:
        return {}
    wanted = sorted({(run.video_id, run.last_word_ord + 1) for run in runs})
    limits: dict[tuple[int, int], int] = {}
    for start in range(0, len(wanted), C.ASSEMBLE_SQL_BATCH):
        batch = wanted[start : start + C.ASSEMBLE_SQL_BATCH]
        pairs = ", ".join(f"(:v{index}, :o{index})" for index in range(len(batch)))
        params: dict[str, object] = {}
        for index, (video_id, ordinal) in enumerate(batch):
            params[f"v{index}"] = video_id
            params[f"o{index}"] = ordinal
        for record in db.conn.execute(
            f"SELECT video_id, ord, start_ms FROM words "
            f"WHERE (video_id, ord) IN (VALUES {pairs})",
            params,
        ):
            limits[(int(record["video_id"]), int(record["ord"]))] = int(record["start_ms"])
    return limits


def _durations(db: Database, video_ids: Iterable[int]) -> dict[int, int]:
    """Known durations, so a fragment at the very end cannot overrun the file."""
    wanted = sorted({int(video_id) for video_id in video_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(video_id) for video_id in wanted)
    return {
        int(row["id"]): int(row["duration_ms"])
        for row in db.conn.execute(f"SELECT id, duration_ms FROM videos WHERE id IN ({inline})")
        if row["duration_ms"] is not None
    }


def pad_fragments(db: Database, plan: Plan, pad_ms: int) -> Plan:
    """Extend every fragment's tail by ``pad_ms``, clamped to what is there.

    Design §8 lists padding among the controls. It is applied here rather
    than at render time so that the cut list stays exactly what it claims
    to be — an in point and an out point — and a hand-edited timing is
    never silently re-derived.
    """
    if pad_ms <= 0:
        return plan

    runs = [slot.run for slot in plan.slots if slot.run is not None]
    runs.extend(scored.run for slot in plan.slots for scored in slot.alternatives)
    limits = _tail_limits(db, runs)
    durations = _durations(db, (run.video_id for run in runs))

    def padded(run: CandidateRun) -> CandidateRun:
        ceiling = limits.get((run.video_id, run.last_word_ord + 1))
        duration = durations.get(run.video_id)
        if duration is not None:
            ceiling = duration if ceiling is None else min(ceiling, duration)
        end_ms = run.end_ms + pad_ms
        if ceiling is not None:
            end_ms = min(end_ms, ceiling)
        # A clamp below the measured end must not drag the cut backwards.
        return replace(run, end_ms=max(end_ms, run.end_ms))

    slots = tuple(
        replace(
            slot,
            run=None if slot.run is None else padded(slot.run),
            alternatives=tuple(
                replace(scored, run=padded(scored.run)) for scored in slot.alternatives
            ),
        )
        for slot in plan.slots
    )
    return replace(plan, slots=slots)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_match.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/assemble/match.py tests/test_assemble_match.py
git commit -m "feat: add clamped tail padding for assembled fragments"
```

---

### Task 6: Substitutions for a word the corpus never says

Design §8: "Offer ranked substitutions — same stem first, then orthographically close by edit distance, which is a workable proxy for sound in Russian — but never apply one silently. True phonetic matching is not in scope."

Two tiers, in that order, and never applied — they are written into the cut list beside the gap, each carrying a concrete cuttable occurrence so that filling the hole is a copy-paste rather than another run.

Edit distance is computed in Python over a *vocabulary slice*, not over word rows: the candidate set is the distinct `normalized_text` values within `ASSEMBLE_SUBSTITUTION_LEN_WINDOW` characters of the missing word, which is tens of thousands of short strings at full corpus, not 14 million rows. The distance function aborts as soon as the whole DP row exceeds the budget, so most candidates cost a few character comparisons.

**Files:**
- Modify: `rytp/assemble/match.py` (append)
- Test: `tests/test_assemble_match.py` (append)

**Interfaces:**
- Consumes: Task 2's `MatchFilters`, `_eligibility_sql`, `find_occurrences`, `_run_from`; `rytp.models.stem_text`.
- Produces:
  - `edit_distance_at_most(first: str, second: str, max_distance: int) -> int | None`
  - `edit_distance(first: str, second: str) -> int`
  - `SubstitutionHit(text: str, reason: str, distance: int, occurrences: int, run: CandidateRun)` — frozen; `reason` is `"stem"` or `"edit"`.
  - `suggest_substitutions(db, token: str, filters: MatchFilters, *, limit: int = C.ASSEMBLE_SUBSTITUTION_LIMIT) -> tuple[SubstitutionHit, ...]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_assemble_match.py`:

```python
from rytp.assemble.match import SubstitutionHit, edit_distance_at_most, suggest_substitutions


def test_edit_distance_counts_substitutions_and_insertions() -> None:
    assert edit_distance_at_most("дела", "дела", 2) == 0
    assert edit_distance_at_most("дела", "тела", 2) == 1
    assert edit_distance_at_most("дела", "делами", 2) == 2


def test_edit_distance_gives_up_past_the_budget() -> None:
    assert edit_distance_at_most("дела", "неизбежно", 2) is None
    assert edit_distance_at_most("дела", "делами", 1) is None


def test_edit_distance_shortcuts_on_length_alone() -> None:
    assert edit_distance_at_most("а", "аааааааа", 2) is None


@pytest.fixture()
def substitution_corpus(db: Database) -> int:
    """A corpus that says everything near "дела" except "дела"."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело тела неизбежно")
    add_words(db, video_id, "дело делами", start_ms=60_000)
    return video_id


def test_same_stem_is_offered_before_merely_similar_spelling(
    db: Database, substitution_corpus: int
) -> None:
    hits = suggest_substitutions(db, "дела", MatchFilters())
    assert [hit.text for hit in hits] == ["дело", "делами", "тела"]
    assert [hit.reason for hit in hits] == ["stem", "stem", "edit"]


def test_a_substitution_carries_a_cuttable_occurrence(
    db: Database, substitution_corpus: int
) -> None:
    hit = suggest_substitutions(db, "дела", MatchFilters())[0]
    assert isinstance(hit, SubstitutionHit)
    assert hit.run.video_id == substitution_corpus
    assert hit.run.n_words == 1
    assert hit.run.end_ms > hit.run.start_ms
    assert hit.occurrences == 2


def test_nothing_remotely_close_is_offered(db: Database, substitution_corpus: int) -> None:
    offered = {hit.text for hit in suggest_substitutions(db, "дела", MatchFilters())}
    assert offered
    assert "неизбежно" not in offered


def test_the_word_itself_is_never_offered(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дела дело")
    assert "дела" not in {hit.text for hit in suggest_substitutions(db, "дела", MatchFilters())}


def test_caption_words_are_never_offered_as_substitutions(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело", source="caption")
    assert suggest_substitutions(db, "дела", MatchFilters()) == ()


def test_an_excluded_video_offers_nothing(db: Database, substitution_corpus: int) -> None:
    filters = MatchFilters(exclude_video_ids=frozenset({substitution_corpus}))
    assert suggest_substitutions(db, "дела", filters) == ()


def test_substitutions_are_limited(db: Database, substitution_corpus: int) -> None:
    assert len(suggest_substitutions(db, "дела", MatchFilters(), limit=1)) == 1


def test_substitutions_are_deterministic(db: Database, substitution_corpus: int) -> None:
    assert suggest_substitutions(db, "дела", MatchFilters()) == suggest_substitutions(
        db, "дела", MatchFilters()
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_match.py -k substitut -v`
Expected: collection error — `ImportError: cannot import name 'suggest_substitutions'`.

- [ ] **Step 3: Append substitutions to `rytp/assemble/match.py`**

Add `stem_text` to the `rytp.models` import, extend `__all__` with `"SubstitutionHit"`, `"edit_distance"`, `"edit_distance_at_most"`, `"suggest_substitutions"`, and append:

```python
@dataclass(frozen=True)
class SubstitutionHit:
    """A word the corpus does say, offered in place of one it does not."""

    text: str
    reason: str  # "stem" | "edit"
    distance: int
    occurrences: int
    run: CandidateRun


def edit_distance_at_most(first: str, second: str, max_distance: int) -> int | None:
    """Levenshtein distance, or ``None`` once it is certainly past the budget.

    Written out rather than pulled in: the standard library has no edit
    distance (``difflib`` measures something else), and a dependency for
    twenty lines is not worth it. The early abort matters — it is what
    makes scanning a whole vocabulary slice cheap, because most candidates
    are eliminated after a row or two.
    """
    if abs(len(first) - len(second)) > max_distance:
        return None
    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, start=1):
        current = [index]
        row_min = index
        for position, right in enumerate(second, start=1):
            value = min(
                previous[position] + 1,
                current[position - 1] + 1,
                previous[position - 1] + (0 if left == right else 1),
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= max_distance else None


def edit_distance(first: str, second: str) -> int:
    """Full Levenshtein distance — the budgeted form with an unreachable budget.

    Used only to report how far a same-stem suggestion is: the stem tier
    is chosen by the stem, not by distance, so the number is information
    rather than a filter.
    """
    distance = edit_distance_at_most(first, second, len(first) + len(second))
    return len(first) + len(second) if distance is None else distance


def _vocabulary(
    db: Database, token: str, filters: MatchFilters, *, same_stem: bool
) -> list[tuple[str, int]]:
    """Distinct cuttable spellings, with how often each is said.

    ``same_stem`` picks the tier: the stem index for tier one, a length
    window for tier two. Both exclude the token itself and any row whose
    normalized text holds a space — a hyphenated source word lands in one
    row as two words, and offering it as a substitution for one word
    would be wrong.
    """
    tail = _eligibility_sql(filters)
    params: dict[str, object] = {
        "token": token,
        "default_align": C.ASSEMBLE_DEFAULT_ALIGN_SCORE,
        "min_align": filters.min_align_score,
        "limit": C.ASSEMBLE_SUBSTITUTION_VOCAB_LIMIT,
    }
    if same_stem:
        params["stem"] = stem_text(token)
        where = "stem = :stem"
    else:
        window = C.ASSEMBLE_SUBSTITUTION_LEN_WINDOW
        params["low"] = max(1, len(token) - window)
        params["high"] = len(token) + window
        where = "LENGTH(normalized_text) BETWEEN :low AND :high"
    rows = db.conn.execute(
        f"SELECT normalized_text, COUNT(*) AS n FROM words "
        f"WHERE {where} AND normalized_text <> :token "
        f"AND normalized_text NOT LIKE '% %' AND {tail} "
        f"GROUP BY normalized_text ORDER BY n DESC, normalized_text LIMIT :limit",
        params,
    )
    return [(str(row["normalized_text"]), int(row["n"])) for row in rows]


def suggest_substitutions(
    db: Database,
    token: str,
    filters: MatchFilters,
    *,
    limit: int = C.ASSEMBLE_SUBSTITUTION_LIMIT,
) -> tuple[SubstitutionHit, ...]:
    """Ranked stand-ins for a word the corpus never says.

    Design §8: same stem first, then orthographically close by edit
    distance. Nothing here is ever applied — the caller writes these
    beside the gap and the owner decides.
    """
    ranked: list[tuple[tuple[int, int, str], str, str, int, int]] = []
    seen: set[str] = set()

    for form, occurrences in _vocabulary(db, token, filters, same_stem=True):
        seen.add(form)
        ranked.append(
            ((0, -occurrences, form), form, "stem", edit_distance(token, form), occurrences)
        )

    for form, occurrences in _vocabulary(db, token, filters, same_stem=False):
        if form in seen:
            continue
        distance = edit_distance_at_most(token, form, C.ASSEMBLE_SUBSTITUTION_MAX_DISTANCE)
        if distance is None:
            continue
        ranked.append(((1, distance, form), form, "edit", distance, occurrences))

    ranked.sort(key=lambda item: item[0])

    hits: list[SubstitutionHit] = []
    for _key, form, reason, distance, occurrences in ranked:
        if len(hits) == limit:
            break
        found = find_occurrences(db, form, filters)
        if not found:
            continue
        hits.append(
            SubstitutionHit(
                text=form,
                reason=reason,
                distance=distance,
                occurrences=occurrences,
                run=_run_from([found[0]]),
            )
        )
    return tuple(hits)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_match.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_match.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/assemble/match.py tests/test_assemble_match.py
git commit -m "feat: rank stem and edit-distance substitutions for missing words"
```

---

### Task 7: The cut list file and its writer

**This task defines a contract with Part 6.** Design §8: "The cut list is a plain editable file — fragments, source ids, in/out points in milliseconds. It is the durable representation: hand-edit it and re-render. Timings are editable by hand, which is deliberately preferred over clever heuristics."

Three decisions, all made for the benefit of the human who edits the file:

**One ordered `[[slot]]` array, not separate fragment and gap lists.** A cut list is a timeline, and a missing word is a position on it. With two arrays the renderer would have to interleave them from index fields, and a hand-edit that moved a fragment would silently desynchronise those indices. With one array, **document order is the timeline** — there is no index field at all, and reordering, deleting or inserting a slot means exactly what it looks like.

**Optional keys are omitted, never written as a null.** TOML has no null. A key that is absent means "not known" or "you decide"; that is how `gap_before_ms` lets a human override the renderer's pause without assembly having an opinion about pauses.

**A hand-written emitter, not `tomli-w`.** `tomllib` reads TOML in the standard library but writes nothing, and the obvious dependency cannot emit comments. This file opens with a comment telling its reader that the timings are authoritative and how to fill a gap — worth more than the sixty lines the emitter costs. The emitter also gives byte-stable canonical output, which is what Task 11's determinism test checks.

**Files:**
- Create: `rytp/assemble/cutlist.py`
- Test: `tests/test_assemble_cutlist.py`

**Interfaces:**
- Consumes: `rytp.assemble.match.{Plan, PlanSlot, SubstitutionHit, SLOT_FRAGMENT, SLOT_GAP}`, `rytp.models.{Fragment, InvalidInputError, normalize_text}`, `rytp.config.{paths, ensure_dir}`, `rytp.constants`.
- Produces:
  - `Alternative(video_id, first_word_ord, last_word_ord, start_ms, end_ms, text, cost)` — frozen.
  - `Substitution(text, reason, distance, occurrences, video_id, first_word_ord, last_word_ord, start_ms, end_ms)` — frozen.
  - `Slot(kind, target_first, target_last, text, video_id=None, first_word_ord=None, last_word_ord=None, start_ms=None, end_ms=None, align_score=None, cost=None, video_speaker_id=None, speaker_label=None, gap_before_ms=None, alternatives=(), substitutions=())` — frozen, with `as_fragment() -> Fragment`.
  - `CutlistParams(consistency, seed, pad_ms, speaker, exclude, min_align_score)` — frozen.
  - `CutList(schema_version, name, target, created_at, params, slots)` — frozen, with properties `fragments`, `gaps`, `duration_ms`.
  - `from_plan(plan, *, name, params, created_at, substitutions, speaker_labels) -> CutList`
  - `dumps_cutlist(cutlist: CutList) -> str`
  - `write_cutlist(cutlist: CutList, path: Path) -> Path`
  - `cutlist_name(target: str) -> str`, `validate_name(name: str) -> str`, `cutlist_path(name: str) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_cutlist.py`:

```python
"""The durable artifact: one ordered slot list, written so a human can edit it."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistParams,
    Slot,
    Substitution,
    cutlist_name,
    cutlist_path,
    dumps_cutlist,
    validate_name,
    write_cutlist,
)
from rytp.models import Fragment, InvalidInputError

CREATED = "2026-09-21T09:00:00+00:00"

PARAMS = CutlistParams(
    consistency=0.25, seed=0, pad_ms=0, speaker="", exclude=(), min_align_score=0.0
)


def sample() -> CutList:
    """A cut list with one fragment, one alternative, one gap, one substitution."""
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="demo",
        target="мы всё исправим",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=1,
                text="мы всё",
                video_id=3,
                first_word_ord=1204,
                last_word_ord=1205,
                start_ms=612_340,
                end_ms=613_100,
                align_score=0.81,
                cost=1.42,
                video_speaker_id=11,
                speaker_label="host",
                alternatives=(
                    Alternative(
                        video_id=7,
                        first_word_ord=88,
                        last_word_ord=89,
                        start_ms=10_500,
                        end_ms=11_220,
                        text="мы всё",
                        cost=1.77,
                    ),
                ),
            ),
            Slot(
                kind="gap",
                target_first=2,
                target_last=2,
                text="исправим",
                substitutions=(
                    Substitution(
                        text="исправит",
                        reason="edit",
                        distance=1,
                        occurrences=3,
                        video_id=9,
                        first_word_ord=502,
                        last_word_ord=502,
                        start_ms=220_100,
                        end_ms=220_780,
                    ),
                ),
            ),
        ),
    )


def test_the_file_opens_with_a_comment_for_the_person_editing_it() -> None:
    text = dumps_cutlist(sample())
    assert text.startswith("#")
    assert "hand-edit" in text.lower()


def test_it_parses_as_toml_and_keeps_the_top_level_fields() -> None:
    parsed = tomllib.loads(dumps_cutlist(sample()))
    assert parsed["schema_version"] == C.CUTLIST_SCHEMA_VERSION
    assert parsed["name"] == "demo"
    assert parsed["target"] == "мы всё исправим"
    assert parsed["created_at"] == CREATED
    assert parsed["params"]["consistency"] == 0.25
    assert parsed["params"]["exclude"] == []


def test_slots_are_one_ordered_array_with_a_kind() -> None:
    slots = tomllib.loads(dumps_cutlist(sample()))["slot"]
    assert [slot["kind"] for slot in slots] == ["fragment", "gap"]
    assert "index" not in slots[0]


def test_a_fragment_slot_carries_every_field_the_renderer_needs() -> None:
    fragment = tomllib.loads(dumps_cutlist(sample()))["slot"][0]
    assert fragment["video_id"] == 3
    assert fragment["first_word_ord"] == 1204
    assert fragment["last_word_ord"] == 1205
    assert fragment["start_ms"] == 612_340
    assert fragment["end_ms"] == 613_100
    assert fragment["text"] == "мы всё"
    assert fragment["speaker_label"] == "host"
    assert fragment["alternative"][0]["video_id"] == 7


def test_a_gap_slot_carries_the_word_and_its_substitutions() -> None:
    gap = tomllib.loads(dumps_cutlist(sample()))["slot"][1]
    assert gap["text"] == "исправим"
    assert gap["target_first"] == gap["target_last"] == 2
    assert gap["substitution"][0]["text"] == "исправит"
    assert gap["substitution"][0]["reason"] == "edit"


def test_absent_optional_keys_are_omitted_rather_than_nulled() -> None:
    gap = tomllib.loads(dumps_cutlist(sample()))["slot"][1]
    for absent in ("video_id", "start_ms", "end_ms", "align_score", "gap_before_ms"):
        assert absent not in gap


def test_gap_before_ms_is_written_when_a_human_set_it() -> None:
    cutlist = sample()
    edited = CutList(
        schema_version=cutlist.schema_version,
        name=cutlist.name,
        target=cutlist.target,
        created_at=cutlist.created_at,
        params=cutlist.params,
        slots=(
            Slot(**{**vars(cutlist.slots[0]), "gap_before_ms": 320}),
            cutlist.slots[1],
        ),
    )
    assert tomllib.loads(dumps_cutlist(edited))["slot"][0]["gap_before_ms"] == 320


def test_quotes_and_backslashes_survive_the_round_trip() -> None:
    cutlist = sample()
    awkward = Slot(
        kind="fragment",
        target_first=0,
        target_last=0,
        text='он сказал "нет" \\ и ушёл',
        video_id=1,
        first_word_ord=0,
        last_word_ord=0,
        start_ms=0,
        end_ms=100,
    )
    edited = CutList(
        schema_version=cutlist.schema_version,
        name=cutlist.name,
        target=cutlist.target,
        created_at=cutlist.created_at,
        params=cutlist.params,
        slots=(awkward,),
    )
    parsed = tomllib.loads(dumps_cutlist(edited))
    assert parsed["slot"][0]["text"] == 'он сказал "нет" \\ и ушёл'


def test_the_same_cut_list_serializes_byte_for_byte_identically() -> None:
    assert dumps_cutlist(sample()) == dumps_cutlist(sample())


def test_write_cutlist_creates_the_directory_and_writes_utf8_with_lf(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cutlists" / "demo.toml"
    written = write_cutlist(sample(), target)
    assert written == target
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert "мы всё".encode() in raw


def test_properties_split_the_slots_and_measure_the_output() -> None:
    cutlist = sample()
    assert len(cutlist.fragments) == 1
    assert len(cutlist.gaps) == 1
    assert cutlist.duration_ms == 613_100 - 612_340


def test_a_fragment_slot_converts_to_the_contract_fragment_type() -> None:
    fragment = sample().fragments[0].as_fragment()
    assert isinstance(fragment, Fragment)
    assert fragment.video_id == 3
    assert fragment.start_ms == 612_340


def test_a_gap_slot_refuses_to_be_a_fragment() -> None:
    with pytest.raises(InvalidInputError, match="gap"):
        sample().gaps[0].as_fragment()


def test_cutlist_name_slugifies_the_target() -> None:
    assert cutlist_name("Мы всё исправим!") == "мы-все-исправим"


def test_cutlist_name_truncates_and_never_ends_in_a_dash() -> None:
    name = cutlist_name("слово " * 40)
    assert len(name) <= C.ASSEMBLE_NAME_MAX_CHARS
    assert not name.endswith("-")


def test_cutlist_name_falls_back_when_the_target_slugifies_to_nothing() -> None:
    assert cutlist_name("!!! ???") == C.ASSEMBLE_FALLBACK_NAME


def test_validate_name_rejects_anything_that_could_escape_the_directory() -> None:
    for bad in ("", "   ", "..", "a/b", "a\\b", "/abs", "C:name"):
        with pytest.raises(InvalidInputError):
            validate_name(bad)
    assert validate_name(" demo ") == "demo"


def test_cutlist_path_lands_under_the_data_tree(data_dir: Path) -> None:
    assert cutlist_path("demo") == data_dir / C.CUTLISTS_DIRNAME / "demo.toml"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_cutlist.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.assemble.cutlist'`.

- [ ] **Step 3: Write `rytp/assemble/cutlist.py`**

```python
"""The cut list: the durable thing assembly produces (design §8).

    "The cut list is a plain editable file — fragments, source ids,
    in/out points in milliseconds. It is the durable representation:
    hand-edit it and re-render."

The file is one ordered ``[[slot]]`` array. Document order **is** the
output timeline, and a slot is either a fragment (something to cut) or a
gap (a word the corpus never says). There is deliberately no index field:
a number that duplicates position is a number a hand-edit can falsify.

Optional keys are omitted rather than written as an empty value, because
TOML has no null and "absent" carries meaning here — an absent
``gap_before_ms`` means the renderer decides the pause, a present one
means a human already did.

Reading is ``tomllib`` from the standard library. Writing is the small
canonical emitter at the bottom of this module rather than ``tomli-w``:
the header comment is the file's instructions to its own reader and no
TOML writer emits comments, and a canonical emitter makes the output
byte-stable, which is half of design §8's determinism requirement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from rytp import constants as C
from rytp.assemble.match import SLOT_FRAGMENT, SLOT_GAP, Plan, SubstitutionHit
from rytp.config import ensure_dir, paths
from rytp.models import Fragment, InvalidInputError, normalize_text

__all__ = [
    "Alternative",
    "CutList",
    "CutlistParams",
    "Slot",
    "Substitution",
    "cutlist_name",
    "cutlist_path",
    "dumps_cutlist",
    "from_plan",
    "validate_name",
    "write_cutlist",
]

_HEADER = (
    "# rytp cut list — hand-edit this file and re-render; the timings below\n"
    "# are authoritative and are never recomputed.\n"
    "# Slots run in document order: that order is the output timeline.\n"
    "# To fill a gap, copy a substitution's fields into its slot and change\n"
    "# kind = \"gap\" to kind = \"fragment\".\n"
    "# To swap a fragment, copy one of its alternatives over its own fields.\n"
    "# gap_before_ms is optional: add it to force a pause, leave it out to\n"
    "# let the renderer decide.\n"
)


@dataclass(frozen=True)
class Alternative:
    """A source this slot could have used instead."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str
    cost: float


@dataclass(frozen=True)
class Substitution:
    """A word the corpus does say, offered for one it does not.

    Carries full fragment fields on purpose: filling a gap is then a
    copy-paste, not another run of the assembler.
    """

    text: str
    reason: str  # "stem" | "edit"
    distance: int
    occurrences: int
    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Slot:
    """One position on the output timeline."""

    kind: str  # SLOT_FRAGMENT | SLOT_GAP
    target_first: int
    target_last: int  # inclusive
    text: str
    video_id: int | None = None
    first_word_ord: int | None = None
    last_word_ord: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    align_score: float | None = None
    cost: float | None = None
    video_speaker_id: int | None = None
    speaker_label: str | None = None
    gap_before_ms: int | None = None
    alternatives: tuple[Alternative, ...] = ()
    substitutions: tuple[Substitution, ...] = ()

    def as_fragment(self) -> Fragment:
        """This slot as the contracts §4 ``Fragment``.

        No asserts: ``-O`` strips those, and this is the boundary where a
        hand-edited file turns into something the renderer will cut.
        """
        if self.kind != SLOT_FRAGMENT:
            raise InvalidInputError(
                f"slot at target word {self.target_first} is a gap ({self.text!r}); "
                "a gap has nothing to cut"
            )
        if (
            self.video_id is None
            or self.first_word_ord is None
            or self.last_word_ord is None
            or self.start_ms is None
            or self.end_ms is None
        ):
            raise InvalidInputError(
                f"fragment slot at target word {self.target_first} is missing a "
                "required field; it needs video_id, first_word_ord, last_word_ord, "
                "start_ms and end_ms"
            )
        return Fragment(
            video_id=self.video_id,
            first_word_ord=self.first_word_ord,
            last_word_ord=self.last_word_ord,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            text=self.text,
        )


@dataclass(frozen=True)
class CutlistParams:
    """The knob positions this cut list was produced with, for provenance."""

    consistency: float
    seed: int
    pad_ms: int
    speaker: str
    exclude: tuple[int, ...]
    min_align_score: float


@dataclass(frozen=True)
class CutList:
    """A whole cut list, in memory."""

    schema_version: int
    name: str
    target: str
    created_at: str
    params: CutlistParams
    slots: tuple[Slot, ...]

    @property
    def fragments(self) -> tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_FRAGMENT)

    @property
    def gaps(self) -> tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.kind == SLOT_GAP)

    @property
    def duration_ms(self) -> int:
        """Total cut length, pauses excluded — the renderer adds those."""
        return sum(
            (slot.end_ms or 0) - (slot.start_ms or 0)
            for slot in self.fragments
            if slot.start_ms is not None and slot.end_ms is not None
        )


# -- naming -----------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^\w-]+", re.UNICODE)
_SLUG_RUNS = re.compile(r"-{2,}")


def cutlist_name(target: str) -> str:
    """A filename-safe, deterministic name derived from the target text."""
    slug = _SLUG_STRIP.sub("-", normalize_text(target))
    slug = _SLUG_RUNS.sub("-", slug).strip("-")
    slug = slug[: C.ASSEMBLE_NAME_MAX_CHARS].strip("-")
    return slug or C.ASSEMBLE_FALLBACK_NAME


def validate_name(name: str) -> str:
    """Reject anything that would write outside ``cutlists/``.

    The name reaches this from the command line, so a separator or a
    ``..`` in it is a path traversal, not a naming preference. Windows
    also treats ``C:name`` as a drive-relative path, hence the colon.
    """
    cleaned = name.strip()
    if not cleaned:
        raise InvalidInputError("a cut list needs a name")
    if cleaned in {".", ".."} or any(bad in cleaned for bad in ("/", "\\", ":")):
        raise InvalidInputError(
            f"bad cut list name {name!r}: no path separators, no '..', no drive letters"
        )
    return cleaned


def cutlist_path(name: str) -> Path:
    """Where a cut list of this name lives (contracts §7)."""
    return paths().cutlist(validate_name(name))


# -- building from a plan ---------------------------------------------


def from_plan(
    plan: Plan,
    *,
    name: str,
    params: CutlistParams,
    created_at: str,
    substitutions: Mapping[int, Sequence[SubstitutionHit]],
    speaker_labels: Mapping[int, str],
) -> CutList:
    """Turn a coverage plan into the artifact.

    ``substitutions`` is keyed by the gap's first target word;
    ``speaker_labels`` maps ``video_speakers.id`` to a roster label, and
    is simply empty for an undiarized corpus.
    """
    slots: list[Slot] = []
    for slot in plan.slots:
        run = slot.run
        if slot.kind == SLOT_GAP or run is None:
            slots.append(
                Slot(
                    kind=SLOT_GAP,
                    target_first=slot.target_first,
                    target_last=slot.target_last,
                    text=slot.text,
                    substitutions=tuple(
                        Substitution(
                            text=hit.text,
                            reason=hit.reason,
                            distance=hit.distance,
                            occurrences=hit.occurrences,
                            video_id=hit.run.video_id,
                            first_word_ord=hit.run.first_word_ord,
                            last_word_ord=hit.run.last_word_ord,
                            start_ms=hit.run.start_ms,
                            end_ms=hit.run.end_ms,
                        )
                        for hit in substitutions.get(slot.target_first, ())
                    ),
                )
            )
            continue
        slots.append(
            Slot(
                kind=SLOT_FRAGMENT,
                target_first=slot.target_first,
                target_last=slot.target_last,
                text=run.text,
                video_id=run.video_id,
                first_word_ord=run.first_word_ord,
                last_word_ord=run.last_word_ord,
                start_ms=run.start_ms,
                end_ms=run.end_ms,
                align_score=run.mean_align,
                cost=slot.cost,
                video_speaker_id=run.video_speaker_id,
                speaker_label=(
                    None
                    if run.video_speaker_id is None
                    else speaker_labels.get(run.video_speaker_id)
                ),
                alternatives=tuple(
                    Alternative(
                        video_id=scored.run.video_id,
                        first_word_ord=scored.run.first_word_ord,
                        last_word_ord=scored.run.last_word_ord,
                        start_ms=scored.run.start_ms,
                        end_ms=scored.run.end_ms,
                        text=scored.run.text,
                        cost=scored.cost,
                    )
                    for scored in slot.alternatives
                ),
            )
        )
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name=name,
        target=plan.target,
        created_at=created_at,
        params=params,
        slots=tuple(slots),
    )


# -- the emitter ------------------------------------------------------

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_string(value: str) -> str:
    """A TOML basic string. Cyrillic passes through; controls are escaped."""
    out: list[str] = ['"']
    for char in value:
        if char in _ESCAPES:
            out.append(_ESCAPES[char])
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _toml_value(value: object) -> str:
    """One scalar or one integer array, canonically."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, C.ASSEMBLE_COST_DECIMALS))
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"a cut list cannot hold {type(value).__name__}")


def _emit(lines: list[str], header: str, record: object, skip: Sequence[str] = ()) -> None:
    """One TOML table from a dataclass, omitting None and the named fields."""
    lines.append(header)
    for field in fields(record):  # type: ignore[arg-type]
        if field.name in skip:
            continue
        value = getattr(record, field.name)
        if value is None:
            continue
        lines.append(f"{field.name} = {_toml_value(value)}")
    lines.append("")


def dumps_cutlist(cutlist: CutList) -> str:
    """The whole cut list as canonical TOML text."""
    lines: list[str] = [_HEADER.rstrip("\n"), ""]
    lines.append(f"schema_version = {cutlist.schema_version}")
    lines.append(f"name = {_toml_string(cutlist.name)}")
    lines.append(f"target = {_toml_string(cutlist.target)}")
    lines.append(f"created_at = {_toml_string(cutlist.created_at)}")
    lines.append("")
    _emit(lines, "[params]", cutlist.params)
    for slot in cutlist.slots:
        _emit(lines, "[[slot]]", slot, skip=("alternatives", "substitutions"))
        for alternative in slot.alternatives:
            _emit(lines, "[[slot.alternative]]", alternative)
        for substitution in slot.substitutions:
            _emit(lines, "[[slot.substitution]]", substitution)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_cutlist(cutlist: CutList, path: Path) -> Path:
    """Write the cut list, creating its directory. Returns the path written."""
    ensure_dir(path.parent)
    path.write_text(dumps_cutlist(cutlist), encoding="utf-8", newline="\n")
    return path
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_cutlist.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_cutlist.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/assemble/cutlist.py tests/test_assemble_cutlist.py
git commit -m "feat: add the cut list format and its canonical writer"
```

---

### Task 8: Reading a cut list a human has edited

The other half of the Part 6 contract, and the half that decides whether the file is genuinely editable. Design §8 makes hand-editing the intended workflow — "hand-edit it and re-render" — so the loader is written for a file that has been changed by a person, not one this program just wrote.

Two principles decide every tolerance question here:

**Structure is required; provenance is not.** A fragment with no `video_id` or no `end_ms` is not a cut and must fail. A fragment whose `first_word_ord` is gone after someone retyped a timing is still perfectly cuttable, so the loader fills those with `-1` and moves on. Likewise `name`, `target` and `created_at` are informational: they default rather than fail.

**Every refusal names the slot and the field.** "invalid cut list" is useless to someone holding a text editor. `demo.toml: slot 3: end_ms (612000) must be greater than start_ms (613100)` is actionable, and an unknown key gets a "did you mean" from `difflib` because the realistic mistake is a typo, not an invention.

The loader does **not** touch the database. Whether `video_id = 42` still exists is a question for the caller — `assemble.show` reports it as a warning, and Part 6 refuses only the fragment it genuinely cannot read.

**Files:**
- Modify: `rytp/assemble/cutlist.py` (append)
- Test: `tests/test_assemble_cutlist.py` (append)

**Interfaces:**
- Consumes: Task 7's dataclasses; `rytp.models.RytpError`.
- Produces: `CutlistError(RytpError)`, `load_cutlist(path: Path) -> CutList`, `read_cutlist(name: str) -> CutList`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_assemble_cutlist.py`:

```python
from rytp.assemble.cutlist import CutlistError, load_cutlist, read_cutlist

MINIMAL = """
schema_version = 1
name = "demo"
target = "мы все"

[[slot]]
kind = "fragment"
target_first = 0
target_last = 1
text = "мы все"
video_id = 3
start_ms = 1000
end_ms = 2000
"""


def write(tmp_path: Path, text: str, name: str = "demo.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_a_written_cut_list_loads_back_identically(tmp_path: Path) -> None:
    path = write_cutlist(sample(), tmp_path / "demo.toml")
    assert load_cutlist(path) == sample()


def test_a_minimal_hand_written_file_loads(tmp_path: Path) -> None:
    cutlist = load_cutlist(write(tmp_path, MINIMAL))
    assert cutlist.name == "demo"
    assert len(cutlist.fragments) == 1
    assert cutlist.fragments[0].start_ms == 1000


def test_missing_provenance_defaults_rather_than_failing(tmp_path: Path) -> None:
    """Someone retyping a timing should not have to keep the ordinals."""
    cutlist = load_cutlist(write(tmp_path, MINIMAL))
    assert cutlist.fragments[0].first_word_ord == -1
    assert cutlist.fragments[0].last_word_ord == -1
    assert cutlist.fragments[0].as_fragment().video_id == 3


def test_missing_informational_fields_default(tmp_path: Path) -> None:
    text = 'schema_version = 1\n\n[[slot]]\nkind = "gap"\ntext = "х"\n'
    cutlist = load_cutlist(write(tmp_path, text, name="notes.toml"))
    assert cutlist.name == "notes"
    assert cutlist.target == ""
    assert cutlist.created_at == ""
    assert cutlist.params.consistency == C.ASSEMBLE_DEFAULT_CONSISTENCY


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="nope.toml"):
        load_cutlist(tmp_path / "nope.toml")


def test_broken_toml_reports_the_parser_message(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, 'schema_version = 1\nname = "unclosed\n'))
    assert "demo.toml" in str(excinfo.value)
    assert "line" in str(excinfo.value).lower()


def test_a_future_schema_version_is_refused_by_number(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="schema_version"):
        load_cutlist(write(tmp_path, MINIMAL.replace("schema_version = 1", "schema_version = 99")))


def test_a_missing_schema_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="schema_version"):
        load_cutlist(write(tmp_path, MINIMAL.replace("schema_version = 1\n", "")))


def test_a_file_with_no_slots_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="no slots"):
        load_cutlist(write(tmp_path, "schema_version = 1\n"))


def test_an_unknown_slot_kind_lists_the_two_that_exist(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="fragment"):
        load_cutlist(write(tmp_path, MINIMAL.replace('kind = "fragment"', 'kind = "clip"')))


def test_a_fragment_without_a_video_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("video_id = 3\n", "")))
    assert "slot 1" in str(excinfo.value)
    assert "video_id" in str(excinfo.value)


def test_a_fragment_without_an_end_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="end_ms"):
        load_cutlist(write(tmp_path, MINIMAL.replace("end_ms = 2000\n", "")))


def test_an_end_before_the_start_is_refused_with_both_numbers(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("end_ms = 2000", "end_ms = 500")))
    assert "500" in str(excinfo.value)
    assert "1000" in str(excinfo.value)


def test_a_negative_timing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="start_ms"):
        load_cutlist(write(tmp_path, MINIMAL.replace("start_ms = 1000", "start_ms = -5")))


def test_a_wrong_type_names_the_field_and_what_was_expected(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("start_ms = 1000", 'start_ms = "1000"')))
    assert "start_ms" in str(excinfo.value)
    assert "integer" in str(excinfo.value)


def test_an_integer_is_accepted_where_a_float_belongs(tmp_path: Path) -> None:
    """TOML distinguishes 1 from 1.0; a person editing by hand does not."""
    cutlist = load_cutlist(write(tmp_path, MINIMAL + "align_score = 1\n"))
    assert cutlist.fragments[0].align_score == 1.0


def test_a_typo_in_a_key_suggests_the_real_one(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("video_id = 3", "video-id = 3\nvideo_id = 3")))
    assert "video-id" in str(excinfo.value)
    assert "video_id" in str(excinfo.value)


def test_a_hand_added_gap_before_survives_the_round_trip(tmp_path: Path) -> None:
    cutlist = load_cutlist(write(tmp_path, MINIMAL + "gap_before_ms = 250\n"))
    assert cutlist.fragments[0].gap_before_ms == 250
    again = load_cutlist(write_cutlist(cutlist, tmp_path / "again.toml"))
    assert again.fragments[0].gap_before_ms == 250


def test_alternatives_and_substitutions_load_back(tmp_path: Path) -> None:
    path = write_cutlist(sample(), tmp_path / "demo.toml")
    cutlist = load_cutlist(path)
    assert cutlist.fragments[0].alternatives[0].video_id == 7
    assert cutlist.gaps[0].substitutions[0].text == "исправит"


def test_a_broken_alternative_names_its_parent_slot(tmp_path: Path) -> None:
    text = MINIMAL + '\n[[slot.alternative]]\nvideo_id = 7\nstart_ms = 0\ntext = "x"\n'
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, text))
    assert "slot 1" in str(excinfo.value)
    assert "alternative 1" in str(excinfo.value)


def test_slot_order_in_the_file_is_the_timeline(tmp_path: Path) -> None:
    text = (
        'schema_version = 1\n'
        '[[slot]]\nkind = "gap"\ntext = "первое"\n'
        '[[slot]]\nkind = "fragment"\ntext = "второе"\nvideo_id = 1\n'
        'start_ms = 0\nend_ms = 10\n'
    )
    assert [slot.text for slot in load_cutlist(write(tmp_path, text)).slots] == [
        "первое",
        "второе",
    ]


def test_read_cutlist_resolves_a_name_under_the_data_tree(data_dir: Path) -> None:
    write_cutlist(sample(), cutlist_path("demo"))
    assert read_cutlist("demo").name == "demo"


def test_read_cutlist_rejects_a_traversing_name(data_dir: Path) -> None:
    with pytest.raises(InvalidInputError):
        read_cutlist("../escape")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_cutlist.py -v`
Expected: collection error — `ImportError: cannot import name 'CutlistError'`.

- [ ] **Step 3: Append the loader to `rytp/assemble/cutlist.py`**

Add `import difflib` and `import tomllib` to the imports, widen the models import to `from rytp.models import Fragment, InvalidInputError, RytpError, normalize_text`, add `from typing import Final, cast`, extend `__all__` with `"CutlistError"`, `"load_cutlist"`, `"read_cutlist"`, and append:

```python
class CutlistError(RytpError):
    """A cut list file that cannot be read as one.

    Always names the file and, where it applies, the slot and the field,
    because the person who will fix it is looking at a text editor.
    """


#: Target position of a slot a hand-edit inserted without saying which
#: word of the target it covers. Informational only.
_UNPLACED: Final = -1

#: Provenance a hand-edit may drop: the cut still works without it.
_ORD_DEFAULT: Final = -1


def _keys_of(record: type) -> tuple[str, ...]:
    """Field names of a dataclass, as they appear in the file."""
    return tuple(field.name for field in fields(record))  # type: ignore[arg-type]


_SLOT_KEYS: Final = tuple(
    name for name in _keys_of(Slot) if name not in {"alternatives", "substitutions"}
) + ("alternative", "substitution")


def _reject(where: str, detail: str) -> CutlistError:
    return CutlistError(f"{where}: {detail}")


def _check_keys(where: str, table: Mapping[str, object], allowed: Sequence[str]) -> None:
    """Refuse an unknown key, guessing what was meant.

    The realistic failure is a typo or a hyphen where an underscore
    belongs, so a bare "unknown key" would be a worse message than the
    file deserves.
    """
    for key in table:
        if key in allowed:
            continue
        close = difflib.get_close_matches(key, allowed, n=1)
        hint = f"; did you mean {close[0]!r}?" if close else f"; known keys: {', '.join(allowed)}"
        raise _reject(where, f"unknown key {key!r}{hint}")


_TYPE_NAMES: Final = {int: "integer", float: "number", str: "string"}


def _raw(
    where: str, table: Mapping[str, object], key: str, kind: type, *, required: bool
) -> object | None:
    """One field, type-checked once, so the readers below need no casts of their own."""
    if key not in table:
        if required:
            raise _reject(where, f"missing required key {key!r}")
        return None
    value = table[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)  # TOML separates 1 from 1.0; a person editing does not
    if isinstance(value, bool) or not isinstance(value, kind):
        raise _reject(where, f"{key} must be a {_TYPE_NAMES[kind]}, got {value!r}")
    return value


def _int_or(where: str, table: Mapping[str, object], key: str, default: int) -> int:
    """An integer field with a fallback. Never ``or``: a stored 0 is a real 0."""
    value = _raw(where, table, key, int, required=False)
    return default if value is None else cast(int, value)


def _opt_int(where: str, table: Mapping[str, object], key: str) -> int | None:
    value = _raw(where, table, key, int, required=False)
    return None if value is None else cast(int, value)


def _req_int(where: str, table: Mapping[str, object], key: str) -> int:
    return cast(int, _raw(where, table, key, int, required=True))


def _float_or(where: str, table: Mapping[str, object], key: str, default: float) -> float:
    value = _raw(where, table, key, float, required=False)
    return default if value is None else cast(float, value)


def _opt_float(where: str, table: Mapping[str, object], key: str) -> float | None:
    value = _raw(where, table, key, float, required=False)
    return None if value is None else cast(float, value)


def _str_or(where: str, table: Mapping[str, object], key: str, default: str = "") -> str:
    value = _raw(where, table, key, str, required=False)
    return default if value is None else cast(str, value)


def _opt_str(where: str, table: Mapping[str, object], key: str) -> str | None:
    value = _raw(where, table, key, str, required=False)
    return None if value is None else cast(str, value)


def _req_str(where: str, table: Mapping[str, object], key: str) -> str:
    return cast(str, _raw(where, table, key, str, required=True))


def _non_negative(where: str, key: str, value: int) -> int:
    if value < 0:
        raise _reject(where, f"{key} must be zero or more, got {value}")
    return value


def _span(where: str, table: Mapping[str, object]) -> tuple[int, int]:
    """The in and out points, validated as a real span.

    Shared by fragments, alternatives and substitutions: all three are
    something to cut, and all three fail the same way.
    """
    start = _non_negative(where, "start_ms", _req_int(where, table, "start_ms"))
    end = _non_negative(where, "end_ms", _req_int(where, table, "end_ms"))
    if end <= start:
        raise _reject(where, f"end_ms ({end}) must be greater than start_ms ({start})")
    return start, end


def _alternative(where: str, table: Mapping[str, object]) -> Alternative:
    _check_keys(where, table, _keys_of(Alternative))
    start, end = _span(where, table)
    return Alternative(
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
        text=_str_or(where, table, "text"),
        cost=_float_or(where, table, "cost", 0.0),
    )


def _substitution(where: str, table: Mapping[str, object]) -> Substitution:
    _check_keys(where, table, _keys_of(Substitution))
    start, end = _span(where, table)
    return Substitution(
        text=_req_str(where, table, "text"),
        reason=_str_or(where, table, "reason", "edit"),
        distance=_int_or(where, table, "distance", 0),
        occurrences=_int_or(where, table, "occurrences", 0),
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
    )


def _nested(where: str, table: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    raw = table.get(key, [])
    if not isinstance(raw, list):
        raise _reject(where, f"{key} must be written as [[slot.{key}]] tables")
    return [item for item in raw if isinstance(item, dict)]


def _slot(where: str, table: Mapping[str, object]) -> Slot:
    _check_keys(where, table, _SLOT_KEYS)
    kind = _req_str(where, table, "kind")
    if kind not in {SLOT_FRAGMENT, SLOT_GAP}:
        raise _reject(where, f"kind must be {SLOT_FRAGMENT!r} or {SLOT_GAP!r}, got {kind!r}")

    first = _int_or(where, table, "target_first", _UNPLACED)
    last = _int_or(where, table, "target_last", _UNPLACED)
    alternatives = tuple(
        _alternative(f"{where}: alternative {index}", item)
        for index, item in enumerate(_nested(where, table, "alternative"), start=1)
    )
    substitutions = tuple(
        _substitution(f"{where}: substitution {index}", item)
        for index, item in enumerate(_nested(where, table, "substitution"), start=1)
    )

    if kind == SLOT_GAP:
        return Slot(
            kind=SLOT_GAP,
            target_first=first,
            target_last=last,
            text=_str_or(where, table, "text"),
            alternatives=alternatives,
            substitutions=substitutions,
        )

    start, end = _span(where, table)
    gap_before = _opt_int(where, table, "gap_before_ms")
    return Slot(
        kind=SLOT_FRAGMENT,
        target_first=first,
        target_last=last,
        text=_str_or(where, table, "text"),
        video_id=_req_int(where, table, "video_id"),
        first_word_ord=_int_or(where, table, "first_word_ord", _ORD_DEFAULT),
        last_word_ord=_int_or(where, table, "last_word_ord", _ORD_DEFAULT),
        start_ms=start,
        end_ms=end,
        align_score=_opt_float(where, table, "align_score"),
        cost=_opt_float(where, table, "cost"),
        video_speaker_id=_opt_int(where, table, "video_speaker_id"),
        speaker_label=_opt_str(where, table, "speaker_label"),
        gap_before_ms=(
            None if gap_before is None else _non_negative(where, "gap_before_ms", gap_before)
        ),
        alternatives=alternatives,
        substitutions=substitutions,
    )


def _params(where: str, table: Mapping[str, object]) -> CutlistParams:
    _check_keys(where, table, _keys_of(CutlistParams))
    exclude = table.get("exclude", [])
    if not isinstance(exclude, list) or any(not isinstance(item, int) for item in exclude):
        raise _reject(where, "exclude must be a list of video ids, for example [3, 7]")
    return CutlistParams(
        consistency=_float_or(where, table, "consistency", C.ASSEMBLE_DEFAULT_CONSISTENCY),
        seed=_int_or(where, table, "seed", 0),
        pad_ms=_int_or(where, table, "pad_ms", C.ASSEMBLE_DEFAULT_PAD_MS),
        speaker=_str_or(where, table, "speaker"),
        exclude=tuple(int(item) for item in exclude),
        min_align_score=_float_or(where, table, "min_align_score", C.ASSEMBLE_MIN_ALIGN_SCORE),
    )


def load_cutlist(path: Path) -> CutList:
    """Read a cut list, tolerating a hand-edit and naming what it cannot read.

    Structure is required and provenance is not: a fragment with no
    ``video_id`` or no ``end_ms`` is not a cut and fails, while a missing
    ``first_word_ord`` is filled in and forgotten. Nothing here reads the
    database — whether a ``video_id`` still exists is the caller's
    question.
    """
    where = path.name
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CutlistError(f"cannot read the cut list at {path}: {exc}") from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CutlistError(f"{where}: not UTF-8 text: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CutlistError(f"{where}: not valid TOML: {exc}") from exc

    version = _req_int(where, document, "schema_version")
    if version != C.CUTLIST_SCHEMA_VERSION:
        raise _reject(
            where,
            f"schema_version is {version}; this build reads "
            f"{C.CUTLIST_SCHEMA_VERSION}",
        )

    raw_slots = document.get("slot", [])
    if not isinstance(raw_slots, list) or not raw_slots:
        raise _reject(where, "no slots; a cut list needs at least one [[slot]] table")

    params_table = document.get("params", {})
    if not isinstance(params_table, dict):
        raise _reject(where, "params must be a [params] table")

    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name=_str_or(where, document, "name", path.stem),
        target=_str_or(where, document, "target"),
        created_at=_str_or(where, document, "created_at"),
        params=_params(f"{where}: [params]", params_table),
        slots=tuple(
            _slot(f"{where}: slot {index}", table)
            for index, table in enumerate(raw_slots, start=1)
            if isinstance(table, dict)
        ),
    )


def read_cutlist(name: str) -> CutList:
    """Load the cut list of this name from under the data tree."""
    return load_cutlist(cutlist_path(name))
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_cutlist.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_cutlist.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`. If mypy reports an *unused* `type: ignore` (it does when `warn_unused_ignores` is on), the annotation was already precise enough — delete that comment rather than widening the type.

- [ ] **Step 6: Commit**

```bash
git add rytp/assemble/cutlist.py tests/test_assemble_cutlist.py
git commit -m "feat: read a hand-edited cut list and say what is wrong with it"
```

---

### Task 9: `assemble_target` — the whole move in one call

One function that a command, a test, or a future TUI screen can call: target text in, `CutList` out. It validates the controls, resolves the speaker, walks, scores, covers, pads, looks up substitutions for the gaps, and builds the artifact.

`created_at` is a parameter with a default rather than a call to the clock inside, because otherwise "same input, same output" (design §8) would be untestable and every re-run would produce a different file.

**Files:**
- Modify: `rytp/assemble/__init__.py` (replace the marker)
- Test: `tests/test_assemble_target.py`

**Interfaces:**
- Consumes: everything in Tasks 2–8; `rytp.models.{InvalidInputError, NotFoundError, utc_now_iso}`.
- Produces:
  - `AssembleControls(consistency, seed, pad_ms, speaker, exclude, min_align_score)` — frozen, with `validated() -> AssembleControls`.
  - `assemble_target(db, target, *, name=None, controls: AssembleControls | None = None, created_at=None) -> CutList` — `None` means the defaults. A dataclass call cannot be a default argument: ruff's `B008` forbids it.
  - `speaker_labels(db, video_speaker_ids) -> dict[int, str]`
  - Re-exports: `CutList`, `CutlistError`, `CutlistParams`, `Slot`, `Alternative`, `Substitution`, `load_cutlist`, `read_cutlist`, `write_cutlist`, `dumps_cutlist`, `cutlist_name`, `cutlist_path`, `validate_name`, `suggest_substitutions`, `tokenize`, `MatchFilters`, `Plan`, `SLOT_FRAGMENT`, `SLOT_GAP`.

**Speaker resolution is Part 1's, not ours.** Contracts §5 "Speaker filters": "One shared resolver in `rytp/commands/__init__.py` serves every command that filters by speaker; no part may roll its own." Part 5 consumes it as
`resolve_speaker_filter(db, *, speaker: str = "", video_local_speaker: str = "", video: str = "") -> SpeakerScope | None`,
where `SpeakerScope` exposes `video_speaker_ids: frozenset[int]` and a human-readable `description`, `None` means no speaker filter was asked for, and an unresolvable `--speaker` raises naming the closest roster entries. **If Part 1 lands a different name or shape, adopt Part 1's verbatim and change nothing else here** — the one thing that must not happen is a second resolver.

**Part 5 offers `--speaker` only, not `--video-local-speaker`.** A raw diarizer label is scoped to one video, and contracts §5 requires `--video` alongside it; for assembly, "only this video" is already `--exclude` for everything else, and inspecting one video's raw labels is a `rytp speakers` concern. Offering the flag would add a `--video` parameter to `assemble.plan` that means something different from every other use of the word here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_target.py`:

```python
"""Target text in, cut list out — the call both surfaces go through."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.assemble import AssembleControls, assemble_target, speaker_labels
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, RytpError
from tests.assembly_corpus import (
    GAP_MS,
    WORD_MS,
    add_speaker,
    add_video,
    add_video_speaker,
    add_words,
)

CREATED = "2026-09-21T09:00:00+00:00"


@pytest.fixture()
def two_videos(db: Database) -> tuple[int, int]:
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "это неизбежно")
    return first, second


def test_it_assembles_across_two_videos(db: Database, two_videos: tuple[int, int]) -> None:
    first, second = two_videos
    cutlist = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    assert [slot.video_id for slot in cutlist.fragments] == [first, second]
    assert cutlist.gaps == ()
    assert cutlist.target == "мы все понимаем это неизбежно"
    assert cutlist.created_at == CREATED
    assert cutlist.schema_version == C.CUTLIST_SCHEMA_VERSION


def test_the_name_defaults_to_a_slug_of_the_target(
    db: Database, two_videos: tuple[int, int]
) -> None:
    cutlist = assemble_target(db, "Мы всё понимаем!", created_at=CREATED)
    assert cutlist.name == "мы-все-понимаем"
    named = assemble_target(db, "Мы всё понимаем!", name="кино", created_at=CREATED)
    assert named.name == "кино"


def test_the_controls_are_recorded_in_the_file(
    db: Database, two_videos: tuple[int, int]
) -> None:
    controls = AssembleControls(
        consistency=0.8, seed=5, pad_ms=40, speaker="", exclude=(99,), min_align_score=0.1
    )
    params = assemble_target(db, "мы все", controls=controls, created_at=CREATED).params
    assert params.consistency == 0.8
    assert params.seed == 5
    assert params.pad_ms == 40
    assert params.exclude == (99,)
    assert params.min_align_score == 0.1


def test_padding_reaches_the_written_timings(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы все")
    plain = assemble_target(db, "мы все", created_at=CREATED)
    padded = assemble_target(
        db, "мы все", controls=AssembleControls(pad_ms=60), created_at=CREATED
    )
    assert plain.fragments[0].end_ms == 2 * WORD_MS + GAP_MS
    assert padded.fragments[0].end_ms == 2 * WORD_MS + GAP_MS + 60


def test_a_missing_word_becomes_a_gap_with_substitutions(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы дело")
    cutlist = assemble_target(db, "мы дела", created_at=CREATED)
    assert len(cutlist.fragments) == 1
    gap = cutlist.gaps[0]
    assert gap.text == "дела"
    assert gap.substitutions[0].text == "дело"
    assert gap.substitutions[0].video_id == video_id
    assert gap.substitutions[0].end_ms > gap.substitutions[0].start_ms


def test_excluded_videos_are_not_used(db: Database, two_videos: tuple[int, int]) -> None:
    first, _second = two_videos
    cutlist = assemble_target(
        db, "мы все понимаем", controls=AssembleControls(exclude=(first,)), created_at=CREATED
    )
    assert cutlist.fragments == ()
    assert len(cutlist.gaps) == 3


def test_a_speaker_filter_restricts_the_sources_and_labels_them(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    host_row = add_speaker(db, "host")
    host = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=host_row)
    guest = add_video_speaker(db, video_id, "SPEAKER_01")
    add_words(db, video_id, "мы все", video_speaker_id=guest)
    add_words(db, video_id, "мы все", start_ms=100_000, video_speaker_id=host)
    cutlist = assemble_target(
        db, "мы все", controls=AssembleControls(speaker="host"), created_at=CREATED
    )
    assert cutlist.fragments[0].start_ms == 100_000
    assert cutlist.fragments[0].video_speaker_id == host
    assert cutlist.fragments[0].speaker_label == "host"


def test_an_unknown_speaker_is_refused_by_the_shared_resolver(db: Database) -> None:
    """contracts §5: resolution failure is an error, never a silent empty result."""
    add_speaker(db, "host")
    with pytest.raises(RytpError):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="guest"))


def test_a_raw_diarizer_label_is_not_a_speaker(db: Database) -> None:
    """contracts §5: --speaker never accepts a raw diarizer label."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_video_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "мы все")
    with pytest.raises(RytpError):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="SPEAKER_00"))


def test_a_speaker_nobody_has_been_mapped_to_says_so(db: Database) -> None:
    add_speaker(db, "host")
    with pytest.raises(NotFoundError, match="not mapped"):
        assemble_target(db, "мы все", controls=AssembleControls(speaker="host"))


def test_an_empty_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        assemble_target(db, "   !!!   ")


def test_an_absurdly_long_target_is_rejected(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="words"):
        assemble_target(db, "слово " * (C.ASSEMBLE_MAX_TARGET_WORDS + 1))


def test_the_controls_validate_themselves() -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        AssembleControls(consistency=2.0).validated()
    with pytest.raises(InvalidInputError, match="pad"):
        AssembleControls(pad_ms=-1).validated()
    with pytest.raises(InvalidInputError, match="pad"):
        AssembleControls(pad_ms=C.ASSEMBLE_MAX_PAD_MS + 1).validated()
    with pytest.raises(InvalidInputError, match="align"):
        AssembleControls(min_align_score=1.5).validated()
    assert AssembleControls().validated() == AssembleControls()


def test_speaker_labels_maps_only_the_mapped_ones(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    roster = add_speaker(db, "host")
    mapped = add_video_speaker(db, video_id, "SPEAKER_00", speaker_id=roster)
    unmapped = add_video_speaker(db, video_id, "SPEAKER_01")
    assert speaker_labels(db, [mapped, unmapped]) == {mapped: "host"}
    assert speaker_labels(db, []) == {}


def test_the_same_call_twice_gives_an_equal_cut_list(
    db: Database, two_videos: tuple[int, int]
) -> None:
    first = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    second = assemble_target(db, "мы все понимаем это неизбежно", created_at=CREATED)
    assert first == second
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_assemble_target.py -v`
Expected: collection error — `ImportError: cannot import name 'assemble_target' from 'rytp.assemble'`.

- [ ] **Step 3: Replace `rytp/assemble/__init__.py`**

```python
"""Assembly: a target sentence in, a cut list out (design §8).

    "You type a sentence. The tool finds real fragments where those words
    were actually said, cuts them out of the source videos, and glues
    them into a single file" — design §1.

This module is the seam between the three below it. :mod:`match` finds
what the corpus can say and chooses a covering, :mod:`score` prices the
choices, :mod:`cutlist` makes the result durable. Nothing here writes to
the database and nothing here enqueues a job: assembly is a read, and
rendering is part 6's.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistError,
    CutlistParams,
    Slot,
    Substitution,
    cutlist_name,
    cutlist_path,
    dumps_cutlist,
    from_plan,
    load_cutlist,
    read_cutlist,
    validate_name,
    write_cutlist,
)
from rytp.assemble.match import (
    SLOT_FRAGMENT,
    SLOT_GAP,
    MatchFilters,
    Plan,
    SubstitutionHit,
    build_run_table,
    pad_fragments,
    plan_coverage,
    suggest_substitutions,
    tokenize,
)
from rytp.assemble.score import load_acoustics, weights_for
from rytp.commands import resolve_speaker_filter
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError, utc_now_iso

__all__ = [
    "SLOT_FRAGMENT",
    "SLOT_GAP",
    "Alternative",
    "AssembleControls",
    "CutList",
    "CutlistError",
    "CutlistParams",
    "MatchFilters",
    "Plan",
    "Slot",
    "Substitution",
    "assemble_target",
    "cutlist_name",
    "cutlist_path",
    "dumps_cutlist",
    "load_cutlist",
    "read_cutlist",
    "speaker_labels",
    "suggest_substitutions",
    "tokenize",
    "validate_name",
    "write_cutlist",
]


@dataclass(frozen=True)
class AssembleControls:
    """Design §8's controls, in one value both surfaces can pass around."""

    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY
    seed: int = 0
    pad_ms: int = C.ASSEMBLE_DEFAULT_PAD_MS
    speaker: str = ""
    exclude: tuple[int, ...] = ()
    min_align_score: float = C.ASSEMBLE_MIN_ALIGN_SCORE

    def validated(self) -> AssembleControls:
        """Reject an out-of-range control before any work is done."""
        weights_for(self.consistency)  # raises InvalidInputError, names the range
        if not 0 <= self.pad_ms <= C.ASSEMBLE_MAX_PAD_MS:
            raise InvalidInputError(
                f"pad must be between 0 and {C.ASSEMBLE_MAX_PAD_MS} ms, got {self.pad_ms}"
            )
        if not 0.0 <= self.min_align_score <= 1.0:
            raise InvalidInputError(
                f"min align score must be between 0.0 and 1.0, got {self.min_align_score}"
            )
        return replace(self, exclude=tuple(sorted({int(item) for item in self.exclude})))


def speaker_labels(db: Database, video_speaker_ids: Iterable[int]) -> dict[int, str]:
    """Per-video label row id -> roster label, for the ones that have one."""
    wanted = sorted({int(item) for item in video_speaker_ids})
    if not wanted:
        return {}
    inline = ", ".join(str(item) for item in wanted)
    return {
        int(row["id"]): str(row["label"])
        for row in db.conn.execute(
            "SELECT video_speakers.id AS id, speakers.label AS label "
            "FROM video_speakers JOIN speakers ON speakers.id = video_speakers.speaker_id "
            f"WHERE video_speakers.id IN ({inline})"
        )
    }


def _filters(db: Database, controls: AssembleControls) -> MatchFilters:
    """Turn the controls into what the matcher understands."""
    labels: frozenset[int] | None = None
    if controls.speaker:
        # contracts §5: one shared resolver, and it matches speakers.label
        # then speakers.aliases_json. A raw SPEAKER_00 is not accepted here
        # and resolution failure raises, naming the closest roster entries.
        scope = resolve_speaker_filter(db, speaker=controls.speaker)
        labels = None if scope is None else scope.video_speaker_ids
        if not labels:
            raise NotFoundError(
                f"speaker {controls.speaker!r} is not mapped to any video yet, so "
                "nothing can be cut for them; map a diarized video to them first"
            )
    return MatchFilters(
        exclude_video_ids=frozenset(controls.exclude),
        video_speaker_ids=labels,
        min_align_score=controls.min_align_score,
    )


def assemble_target(
    db: Database,
    target: str,
    *,
    name: str | None = None,
    controls: AssembleControls | None = None,
    created_at: str | None = None,
) -> CutList:
    """Find real fragments that say ``target``, and return the cut list.

    ``created_at`` is a parameter rather than a call to the clock so that
    design §8's "same input gives the same output" is something a test
    can check and a re-run can reproduce.
    """
    controls = (controls or AssembleControls()).validated()
    tokens = tokenize(target)
    if not tokens:
        raise InvalidInputError(f"the target {target!r} is empty once normalized")
    if len(tokens) > C.ASSEMBLE_MAX_TARGET_WORDS:
        raise InvalidInputError(
            f"the target is {len(tokens)} words; the limit is "
            f"{C.ASSEMBLE_MAX_TARGET_WORDS} words — assemble a sentence at a time"
        )

    filters = _filters(db, controls)
    table = build_run_table(db, tokens, filters)
    videos = {run.video_id for bucket in table.values() for run in bucket.values()}
    plan = plan_coverage(
        target,
        tokens,
        table,
        weights=weights_for(controls.consistency),
        acoustics=load_acoustics(db, videos),
        seed=controls.seed,
    )
    plan = pad_fragments(db, plan, controls.pad_ms)

    substitutions: dict[int, Sequence[SubstitutionHit]] = {
        slot.target_first: suggest_substitutions(db, slot.text, filters)
        for slot in plan.gaps
    }
    used_speakers = {
        slot.run.video_speaker_id
        for slot in plan.fragments
        if slot.run is not None and slot.run.video_speaker_id is not None
    }

    return from_plan(
        plan,
        name=validate_name(name or cutlist_name(target)),
        params=CutlistParams(
            consistency=controls.consistency,
            seed=controls.seed,
            pad_ms=controls.pad_ms,
            speaker=controls.speaker,
            exclude=controls.exclude,
            min_align_score=controls.min_align_score,
        ),
        created_at=created_at or utc_now_iso(),
        substitutions=substitutions,
        speaker_labels=speaker_labels(db, used_speakers),
    )
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_assemble_target.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Run everything written so far**

Run: `python -m pytest tests/test_assemble_match.py tests/test_assemble_score.py tests/test_assemble_cutlist.py tests/test_assemble_target.py tests/test_assembly_corpus.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/assemble tests/test_assemble_target.py && python -m mypy rytp/assemble`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/assemble/__init__.py tests/test_assemble_target.py
git commit -m "feat: add assemble_target, the one call from text to cut list"
```

---

### Task 10: Commands

Contracts §5: each operation is defined once and both surfaces are generated from that definition. Handlers take an open `Database` first, never print, never call `sys.exit`, and raise `RytpError` for an expected failure.

**None of the three is `long_running`.** Contracts §5's job table has no `assemble` kind and design §5 never queues assembly — marking one long-running would make the TUI refuse to run it inline with no worker to run it instead. Keeping planning interactive is what the candidate caps in Task 1 are for.

**Files:**
- Create: `rytp/commands/assemble.py`
- Modify: `rytp/commands/__init__.py` (one line in the import block at the bottom)
- Test: `tests/test_commands_assemble.py`

**Interfaces:**
- Consumes: `rytp.commands.{Command, CommandResult, Param, register}`, `rytp.assemble.*`, `rytp.config.paths`.
- Produces: handlers `assemble_plan`, `assemble_show`, `assemble_suggest`; helpers `parse_ids(text) -> tuple[int, ...]`, `format_ms(value: int) -> str`; registered commands `assemble.plan`, `assemble.show`, `assemble.suggest`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_commands_assemble.py`:

```python
"""The assemble command group: registered once, used by both surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import rytp.commands.assemble as assemble_commands  # noqa: F401 - registers the commands
from rytp import constants as C
from rytp.assemble import cutlist_path, read_cutlist
from rytp.cli import build_app
from rytp.commands import COMMANDS
from rytp.commands.assemble import (
    assemble_plan,
    assemble_show,
    assemble_suggest,
    format_ms,
    parse_ids,
)
from rytp.db import Database
from rytp.models import InvalidInputError, RytpError
from tests.assembly_corpus import add_video, add_words

runner = CliRunner()

NAMES = ("assemble.plan", "assemble.show", "assemble.suggest")


@pytest.fixture()
def corpus(db: Database) -> tuple[int, int]:
    first = add_video(db, external_id="VIDEO_A", title="A")
    add_words(db, first, "мы все понимаем")
    second = add_video(db, external_id="VIDEO_B", title="B")
    add_words(db, second, "это неизбежно")
    return first, second


def test_every_command_is_registered_in_the_assemble_group() -> None:
    for name in NAMES:
        assert name in COMMANDS
        assert COMMANDS[name].group == "assemble"
        assert COMMANDS[name].summary
        assert COMMANDS[name].long_running is False


def test_parse_ids_reads_a_comma_separated_flag() -> None:
    assert parse_ids("") == ()
    assert parse_ids(" 3, 7 ,3 ") == (3, 7)
    with pytest.raises(InvalidInputError, match="seven"):
        parse_ids("3,seven")


def test_format_ms_reads_like_a_timestamp() -> None:
    assert format_ms(0) == "0:00.000"
    assert format_ms(612_340) == "10:12.340"
    assert format_ms(3_723_004) == "1:02:03.004"


def test_plan_writes_the_file_and_reports_the_slots(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    result = assemble_plan(db, target="мы все понимаем это неизбежно", name="демо")
    assert cutlist_path("демо").exists()
    assert result.columns == ("#", "kind", "target", "source", "in", "out", "text")
    assert [row[1] for row in result.rows] == ["fragment", "fragment"]
    assert "демо" in (result.message or "")
    assert read_cutlist("демо").target == "мы все понимаем это неизбежно"


def test_plan_names_the_file_after_the_target_when_not_told(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="Мы всё понимаем")
    assert cutlist_path("мы-все-понимаем").exists()


def test_plan_refuses_to_overwrite_without_force(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    with pytest.raises(RytpError, match="force"):
        assemble_plan(db, target="мы все понимаем", name="демо")
    assemble_plan(db, target="мы все понимаем", name="демо", force=True)


def test_plan_reports_gaps_in_the_table_and_the_message(
    db: Database, data_dir: Path
) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "мы дело")
    result = assemble_plan(db, target="мы дела", name="демо")
    assert [row[1] for row in result.rows] == ["fragment", "gap"]
    assert "1 word not found" in (result.message or "")


def test_plan_passes_the_controls_through(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(
        db,
        target="мы все понимаем",
        name="демо",
        consistency=0.9,
        seed=4,
        pad=30,
        exclude="99",
        min_align=0.2,
    )
    params = read_cutlist("демо").params
    assert (params.consistency, params.seed, params.pad_ms) == (0.9, 4, 30)
    assert params.exclude == (99,)
    assert params.min_align_score == 0.2


def test_plan_rejects_a_knob_off_the_dial(db: Database, data_dir: Path) -> None:
    with pytest.raises(InvalidInputError, match="consistency"):
        assemble_plan(db, target="мы все", name="демо", consistency=3.0)


def test_show_reads_a_cut_list_back(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем это неизбежно", name="демо")
    result = assemble_show(db, name="демо")
    assert [row[1] for row in result.rows] == ["fragment", "fragment"]
    assert "2 fragments" in (result.message or "")


def test_show_says_a_dangling_video_makes_the_cut_list_unrenderable(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """A hand-edit can leave a dangling id. Show it, and say render will refuse."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    path = cutlist_path("демо")
    path.write_text(
        path.read_text(encoding="utf-8").replace("video_id = 1", "video_id = 4242"),
        encoding="utf-8",
        newline="\n",
    )
    result = assemble_show(db, name="демо")
    assert "4242" in (result.message or "")
    assert "NOT RENDERABLE" in (result.message or "")
    assert result.rows  # still shown: this is the file you are about to fix


def test_show_reports_a_broken_file_on_one_line(db: Database, data_dir: Path) -> None:
    path = cutlist_path("сломано")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('schema_version = 1\nname = "oops\n', encoding="utf-8", newline="\n")
    with pytest.raises(RytpError, match="сломано.toml"):
        assemble_show(db, name="сломано")


def test_suggest_ranks_stand_ins_for_a_word(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "дело тела")
    result = assemble_suggest(db, word="дела")
    assert result.columns == ("word", "why", "distance", "occurrences", "source", "in", "out")
    assert [row[0] for row in result.rows] == ["дело", "тела"]
    assert [row[1] for row in result.rows] == ["stem", "edit"]


def test_suggest_says_so_when_nothing_is_close(db: Database) -> None:
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, "неизбежно")
    result = assemble_suggest(db, word="дела")
    assert result.rows == ()
    assert "nothing" in (result.message or "").lower()


def test_the_cli_runs_the_whole_group(data_dir: Path) -> None:
    app = build_app()
    add = runner.invoke(app, ["videos", "add", "--help"])
    assert add.exit_code == 0
    planned = runner.invoke(
        app, ["assemble", "plan", "мы все", "--name", "демо"], env={"COLUMNS": "200"}
    )
    assert planned.exit_code == 0, planned.output
    shown = runner.invoke(app, ["assemble", "show", "демо"], env={"COLUMNS": "200"})
    assert shown.exit_code == 0, shown.output


def test_the_cli_reports_a_bad_knob_on_one_line_and_exits_one(data_dir: Path) -> None:
    result = runner.invoke(
        build_app(), ["assemble", "plan", "мы все", "--consistency", "5"]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_the_default_consistency_is_the_one_the_design_asks_for() -> None:
    params = {param.name: param for param in COMMANDS["assemble.plan"].params}
    assert params["consistency"].default == C.ASSEMBLE_DEFAULT_CONSISTENCY
    assert params["target"].positional is True
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_commands_assemble.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.commands.assemble'`.

- [ ] **Step 3: Write `rytp/commands/assemble.py`**

```python
"""Assembly commands: plan a cut list, read one back, suggest stand-ins.

Design §8 is reached from here and from the TUI, through the one registry
entry each (contracts §5). None of these is ``long_running``: there is no
``assemble`` job kind in contracts §5 and design §5 never queues
assembly, so a long-running mark would leave the TUI with a command it
refuses to run and no worker to run it.
"""

from __future__ import annotations

from rytp import constants as C
from rytp.assemble import (
    SLOT_FRAGMENT,
    AssembleControls,
    CutList,
    MatchFilters,
    assemble_target,
    cutlist_path,
    read_cutlist,
    suggest_substitutions,
    write_cutlist,
)
from rytp.commands import Command, CommandResult, Param, register, resolve_speaker_filter
from rytp.db import Database
from rytp.models import InvalidInputError, RytpError

__all__ = ["assemble_plan", "assemble_show", "assemble_suggest", "format_ms", "parse_ids"]

_SLOT_COLUMNS = ("#", "kind", "target", "source", "in", "out", "text")


def parse_ids(text: str) -> tuple[int, ...]:
    """A comma-separated list of video ids (contracts §5 multi-value flags)."""
    ids: list[int] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            raise InvalidInputError(
                f"{piece!r} is not a video id; give a comma-separated list like '3,7'"
            ) from None
        if value not in ids:
            ids.append(value)
    return tuple(ids)


def format_ms(value: int) -> str:
    """Milliseconds as a timestamp a person can find in a player."""
    seconds, milliseconds = divmod(max(0, value), C.MS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def _slot_rows(cutlist: CutList) -> tuple[tuple[str, ...], ...]:
    """The cut list as a table, gaps included and in timeline order."""
    rows: list[tuple[str, ...]] = []
    for index, slot in enumerate(cutlist.slots, start=1):
        span = (
            f"{slot.target_first}"
            if slot.target_first == slot.target_last
            else f"{slot.target_first}-{slot.target_last}"
        )
        if slot.kind == SLOT_FRAGMENT and slot.start_ms is not None and slot.end_ms is not None:
            source = str(slot.video_id)
            start, end = format_ms(slot.start_ms), format_ms(slot.end_ms)
        else:
            source, start, end = C.NULL_CELL, C.NULL_CELL, C.NULL_CELL
        rows.append((str(index), slot.kind, span, source, start, end, slot.text))
    return tuple(rows)


def _summary(cutlist: CutList) -> str:
    """One line: how many pieces, from how many sources, how long, what is missing."""
    sources = {slot.video_id for slot in cutlist.fragments if slot.video_id is not None}
    parts = [
        f"{len(cutlist.fragments)} fragment{'' if len(cutlist.fragments) == 1 else 's'}",
        f"{len(sources)} source{'' if len(sources) == 1 else 's'}",
        format_ms(cutlist.duration_ms),
    ]
    if cutlist.gaps:
        missing = ", ".join(slot.text for slot in cutlist.gaps)
        parts.append(
            f"{len(cutlist.gaps)} word{'' if len(cutlist.gaps) == 1 else 's'} "
            f"not found: {missing}"
        )
    return "; ".join(parts)


def assemble_plan(
    db: Database,
    *,
    target: str,
    name: str = "",
    consistency: float = C.ASSEMBLE_DEFAULT_CONSISTENCY,
    seed: int = 0,
    pad: int = C.ASSEMBLE_DEFAULT_PAD_MS,
    speaker: str = "",
    exclude: str = "",
    min_align: float = C.ASSEMBLE_MIN_ALIGN_SCORE,
    force: bool = False,
) -> CommandResult:
    """Work out which real fragments say ``target``, and write the cut list."""
    controls = AssembleControls(
        consistency=consistency,
        seed=seed,
        pad_ms=pad,
        speaker=speaker,
        exclude=parse_ids(exclude),
        min_align_score=min_align,
    )
    cutlist = assemble_target(db, target, name=name or None, controls=controls)
    path = cutlist_path(cutlist.name)
    if path.exists() and not force:
        raise RytpError(f"{path} already exists; pass --force to replace it")
    write_cutlist(cutlist, path)
    return CommandResult(
        columns=_SLOT_COLUMNS,
        rows=_slot_rows(cutlist),
        message=f"wrote {path} — {_summary(cutlist)}",
    )


def assemble_show(db: Database, *, name: str) -> CommandResult:
    """Read a cut list back, validate it, and say what is in it."""
    cutlist = read_cutlist(name)
    wanted = sorted({slot.video_id for slot in cutlist.fragments if slot.video_id is not None})
    known: set[int] = set()
    if wanted:
        inline = ", ".join(str(video_id) for video_id in wanted)
        known = {
            int(row["id"])
            for row in db.conn.execute(f"SELECT id FROM videos WHERE id IN ({inline})")
        }
    message = _summary(cutlist)
    missing = [video_id for video_id in wanted if video_id not in known]
    if missing:
        # Showing a broken cut list is the point of this command — it is
        # the file you are about to repair. Rendering one is not: part 6
        # aborts, because a silently shortened video is worse than a
        # refusal. Say so here rather than letting render be the surprise.
        listed = ", ".join(str(video_id) for video_id in missing)
        message += f"; NOT RENDERABLE — not in the catalog: {listed}"
    return CommandResult(columns=_SLOT_COLUMNS, rows=_slot_rows(cutlist), message=message)


def assemble_suggest(
    db: Database,
    *,
    word: str,
    limit: int = C.ASSEMBLE_SUBSTITUTION_LIMIT,
    speaker: str = "",
    exclude: str = "",
) -> CommandResult:
    """Ranked stand-ins for a word: same stem first, then closest spelling."""
    scope = resolve_speaker_filter(db, speaker=speaker) if speaker else None
    filters = MatchFilters(
        exclude_video_ids=frozenset(parse_ids(exclude)),
        video_speaker_ids=None if scope is None else scope.video_speaker_ids,
    )
    hits = suggest_substitutions(db, word.strip().lower(), filters, limit=limit)
    return CommandResult(
        columns=("word", "why", "distance", "occurrences", "source", "in", "out"),
        rows=tuple(
            (
                hit.text,
                hit.reason,
                str(hit.distance),
                str(hit.occurrences),
                str(hit.run.video_id),
                format_ms(hit.run.start_ms),
                format_ms(hit.run.end_ms),
            )
            for hit in hits
        ),
        message=(
            f"{len(hits)} suggestion{'' if len(hits) == 1 else 's'} for {word!r}"
            if hits
            else f"nothing close to {word!r} is cuttable in the corpus"
        ),
    )


register(
    Command(
        name="assemble.plan",
        group="assemble",
        summary="Find real fragments that say a sentence and write a cut list.",
        params=(
            Param("target", str, "The sentence to assemble.", positional=True),
            Param("name", str, "Cut list name. Defaults to a slug of the target.", default=""),
            Param(
                "consistency",
                float,
                "0.0 fewest seams .. 1.0 most consistent sound.",
                default=C.ASSEMBLE_DEFAULT_CONSISTENCY,
                short="-c",
            ),
            Param("seed", int, "Shake up choices among near-equal candidates.", default=0),
            Param(
                "pad",
                int,
                "Milliseconds of tail added after each fragment.",
                default=C.ASSEMBLE_DEFAULT_PAD_MS,
            ),
            Param(
                "speaker",
                str,
                "Only cut words said by this roster person (label or alias).",
                default="",
            ),
            Param("exclude", str, "Video ids never to cut from, comma-separated.", default=""),
            Param(
                "min_align",
                float,
                "Skip words whose alignment score is below this.",
                default=C.ASSEMBLE_MIN_ALIGN_SCORE,
            ),
            Param("force", bool, "Replace an existing cut list of this name.", default=False),
        ),
        handler=assemble_plan,
    )
)

register(
    Command(
        name="assemble.show",
        group="assemble",
        summary="Read a cut list back and show its fragments and gaps.",
        params=(Param("name", str, "Cut list name.", positional=True),),
        handler=assemble_show,
    )
)

register(
    Command(
        name="assemble.suggest",
        group="assemble",
        summary="Ranked stand-ins for a word the corpus does not say.",
        params=(
            Param("word", str, "The missing word.", positional=True),
            Param(
                "limit",
                int,
                "How many suggestions.",
                default=C.ASSEMBLE_SUBSTITUTION_LIMIT,
                short="-n",
            ),
            Param(
                "speaker",
                str,
                "Only consider words said by this roster person (label or alias).",
                default="",
            ),
            Param("exclude", str, "Video ids to ignore, comma-separated.", default=""),
        ),
        handler=assemble_suggest,
    )
)
```

**Multi-word parameter names.** `Param.name` is the handler's keyword argument, so it must be a valid Python identifier: `min_align`, never `min-align`. Whether the CLI renders that as `--min-align` or `--min_align` is Part 1's `build_app` decision. **Read `build_app` before writing this line** and match what it already does for multi-word parameters elsewhere; do not add a translation layer here, and do not rename the parameter to avoid the question. The test below asserts only that the parameter exists and carries the right default, so it passes either way.

- [ ] **Step 4: Register the module**

Append to the import block at the very bottom of `rytp/commands/__init__.py`, after the existing lines:

```python
from rytp.commands import assemble as _assemble  # noqa: E402,F401
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_commands_assemble.py -v`
Expected: every test in the file passes.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp tests && python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/commands/assemble.py rytp/commands/__init__.py tests/test_commands_assemble.py
git commit -m "feat: add the assemble command group"
```

---

### Task 11: `assemble.remove`

Contracts §5 "Deletion": every group exposes a `<group>.remove`, and `assemble.remove` owns "A cut list file." Two of the three binding rules apply directly.

**It deletes a file, so `--dry-run` and `--yes` are mandatory.** Not ceremony: a cut list is where the owner's hand-editing lives. Design §8 chose hand-editable timings *over* clever heuristics — "Timings are editable by hand, which is deliberately preferred" — so a cut list he has been adjusting for an hour is unreproducible work, and a mistyped name that silently destroys it is a real loss. Without `--yes` the command refuses and says what it would have done; `--dry-run` prints the path and its size and touches nothing.

**Removal is not a job.** Synchronous, immediate, no queue.

The third rule — cascade — has nothing to cascade here. This command writes **no** database rows at all. That matters because a cut list can already have been rendered: `renders.cutlist_name` references it by name (contracts §3), and `render.remove` and the `renders` table are Part 6's. So this command *looks*, *reports*, and leaves them alone.

**Settled with Part 6 by message, not assumed.** `assemble.remove` never touches `renders` or anything under `output/`; it names the renders that referenced this cut list in its message, with a `rytp render remove <id>` hint; it **warns rather than refuses**; and the `renders` row keeps its now-dangling `cutlist_name`. Part 6's three reasons for warn-rather-than-refuse, recorded here so neither plan re-argues it:

1. `renders.cutlist_name` is TEXT, not a foreign key — contracts §3 made it a name precisely so a render's history outlives its source. Refusing would turn an audit record into a lock.
2. Renders accumulate, one row per attempt. "Refuse while referenced" would mean deleting every render before you could delete the cut list, so tidying up would get *harder* the more the tool is used. Wrong gradient.
3. It is the house pattern in contracts §5 already: `channel.remove` orphans its videos, `speakers.remove` nulls `video_speakers.speaker_id`. Warn-and-orphan is the norm; refuse-while-referenced would be the odd one out.

Nothing breaks on Part 6's side either, and this was confirmed rather than assumed: **Part 6 never re-reads a cut list after a successful render.** It reads one exactly twice, both before any encoding — `rytp/commands/render.py::_load_request` for a foreground run and `rytp/render/run.py::run_render_job` when the worker picks up a queued one. The report is written at render time from the plan and never regenerated, and `render.list` / `render.remove` read only the `renders` table. So a `renders` row whose cut list is gone stays valid, and the single remaining consequence is that Part 6's readiness predicate returns BLOCKED when `paths().cutlist(name)` is missing, so a *queued* render of a deleted cut list parks instead of failing. The message says so, because that is the one consequence a user cannot see from the filesystem. Part 6 mirrors the warning from its end: `render.remove --dry-run` says "whose cut list is already gone" when the file is missing, so deleting in either order tells the user the same thing.

This is deliberately *not* the dangling-`video_id` rule from Task 10. There the render's **input** is corrupt, so Part 6 aborts. Here the render already happened and its output is untouched; only the provenance is gone.

**Files:**
- Modify: `rytp/commands/assemble.py` (append)
- Test: `tests/test_commands_assemble.py` (append)

**Interfaces:**
- Consumes: `rytp.assemble.cutlist_path`; `rytp.commands.{Command, CommandResult, Param, register}`; `rytp.models.{NotFoundError, RytpError}`.
- Produces: `orphaned_renders(db, name) -> tuple[tuple[str, ...], ...]`; handler `assemble_remove(db, *, name, dry_run=False, yes=False)`; registered command `assemble.remove`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_commands_assemble.py`:

```python
from rytp.assemble import cutlist_name
from rytp.commands.assemble import assemble_remove, orphaned_renders
from rytp.models import NotFoundError, utc_now_iso


def add_render(db: Database, cutlist: str, *, state: str = "rendered") -> int:
    """A renders row, as part 6 would write it (contracts §3)."""
    cursor = db.conn.execute(
        "INSERT INTO renders (cutlist_name, output_path, canvas_mode, state, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (cutlist, f"output/9/output.mp4", "pillarbox", state, utc_now_iso()),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def test_remove_is_registered_and_is_not_long_running() -> None:
    assert "assemble.remove" in COMMANDS
    assert COMMANDS["assemble.remove"].group == "assemble"
    assert COMMANDS["assemble.remove"].long_running is False
    params = {param.name for param in COMMANDS["assemble.remove"].params}
    assert {"name", "dry_run", "yes"} <= params


def test_remove_refuses_without_yes_and_keeps_the_file(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Contracts §5: without --yes, a command that deletes files refuses."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    with pytest.raises(RytpError, match="--yes"):
        assemble_remove(db, name="демо")
    assert cutlist_path("демо").exists()


def test_dry_run_reports_and_deletes_nothing(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    size = cutlist_path("демо").stat().st_size
    result = assemble_remove(db, name="демо", dry_run=True)
    assert cutlist_path("демо").exists()
    assert "демо.toml" in (result.message or "")
    assert str(size) in (result.message or "")
    assert "would" in (result.message or "").lower()


def test_dry_run_needs_no_yes(db: Database, data_dir: Path, corpus: tuple[int, int]) -> None:
    """A dry run deletes nothing, so demanding --yes for it would be noise."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    assert assemble_remove(db, name="демо", dry_run=True).message


def test_remove_deletes_the_file(db: Database, data_dir: Path, corpus: tuple[int, int]) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    result = assemble_remove(db, name="демо", yes=True)
    assert not cutlist_path("демо").exists()
    assert "removed" in (result.message or "")


def test_removing_a_cut_list_that_is_not_there_says_so(db: Database, data_dir: Path) -> None:
    with pytest.raises(NotFoundError, match="нет-такого"):
        assemble_remove(db, name="нет-такого", yes=True)


def test_remove_rejects_a_traversing_name(db: Database, data_dir: Path) -> None:
    with pytest.raises(InvalidInputError):
        assemble_remove(db, name="../escape", yes=True)


def test_remove_names_the_renders_that_used_this_cut_list(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Contracts §5 says do not orphan silently; part 6 owns those rows."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо")
    result = assemble_remove(db, name="демо", yes=True)
    assert str(render_id) in (result.message or "")
    assert "keep their output" in (result.message or "")
    assert "render remove" in (result.message or "")


def test_remove_warns_but_does_not_refuse_when_renders_exist(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """A rendered file stands on its own; losing the cut list costs
    reproducibility, not the artifact."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    add_render(db, "демо")
    assemble_remove(db, name="демо", yes=True)
    assert not cutlist_path("демо").exists()


def test_remove_leaves_the_renders_table_completely_alone(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """render.remove is part 6's. This command writes no rows at all."""
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо")
    assemble_remove(db, name="демо", yes=True)
    row = db.conn.execute("SELECT cutlist_name FROM renders WHERE id = ?", (render_id,)).fetchone()
    assert row is not None
    assert row["cutlist_name"] == "демо"


def test_a_dry_run_lists_the_renders_too(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    assemble_plan(db, target="мы все понимаем", name="демо")
    render_id = add_render(db, "демо", state="failed")
    result = assemble_remove(db, name="демо", dry_run=True)
    assert [row[0] for row in result.rows] == [str(render_id)]
    assert result.columns == ("render", "state", "output")


def test_orphaned_renders_finds_only_this_cut_lists_renders(db: Database) -> None:
    mine = add_render(db, "демо")
    add_render(db, "другое")
    assert [row[0] for row in orphaned_renders(db, "демо")] == [str(mine)]
    assert orphaned_renders(db, "ничего") == ()


def test_the_cli_runs_remove_end_to_end(data_dir: Path) -> None:
    app = build_app()
    assert runner.invoke(app, ["assemble", "plan", "мы все", "--name", "демо"]).exit_code == 0
    refused = runner.invoke(app, ["assemble", "remove", "демо"])
    assert refused.exit_code == 1
    assert "Traceback" not in refused.output
    assert cutlist_path("демо").exists()
    removed = runner.invoke(app, ["assemble", "remove", "демо", "--yes"])
    assert removed.exit_code == 0, removed.output
    assert not cutlist_path("демо").exists()


def test_cutlist_name_round_trips_through_plan_and_remove(
    db: Database, data_dir: Path, corpus: tuple[int, int]
) -> None:
    """Whatever plan named the file, remove must accept the same name."""
    target = "Мы всё понимаем!"
    assemble_plan(db, target=target)
    assemble_remove(db, name=cutlist_name(target), yes=True)
    assert not cutlist_path(cutlist_name(target)).exists()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_commands_assemble.py -k remove -v`
Expected: collection error — `ImportError: cannot import name 'assemble_remove'`.

- [ ] **Step 3: Append the command to `rytp/commands/assemble.py`**

Add `NotFoundError` to the `rytp.models` import, extend `__all__` with `"assemble_remove"` and `"orphaned_renders"`, and append:

```python
def orphaned_renders(db: Database, name: str) -> tuple[tuple[str, ...], ...]:
    """Renders that named this cut list, as table rows.

    Part 6 owns the ``renders`` table and ``render.remove``; this only
    looks, so that removing a cut list does not orphan them in silence
    (contracts §5, "Deletion").
    """
    return tuple(
        (str(row["id"]), str(row["state"]), str(row["output_path"] or C.NULL_CELL))
        for row in db.conn.execute(
            "SELECT id, state, output_path FROM renders WHERE cutlist_name = ? ORDER BY id",
            (name,),
        )
    )


def assemble_remove(
    db: Database, *, name: str, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    """Delete a cut list file. Nothing else — the renders are part 6's.

    Contracts §5: anything that deletes a file supports ``--dry-run`` and
    requires ``--yes``, and removal is synchronous rather than a job. The
    caution is earned here: design §8 made hand-edited timings the
    intended workflow, so a cut list can hold work that exists nowhere
    else and cannot be regenerated.
    """
    path = cutlist_path(name)  # validates the name; rejects path traversal
    if not path.exists():
        raise NotFoundError(f"no cut list named {name!r} at {path}")
    size = path.stat().st_size
    renders = orphaned_renders(db, path.stem)
    note = ""
    if renders:
        listed = ", ".join(row[0] for row in renders)
        # Warn, never refuse: the rendered file stands on its own, and
        # only its provenance is lost. Part 6 aborts on a dangling
        # video_id because there the *input* is corrupt; this is not that.
        note = (
            f"; {len(renders)} render(s) still reference it by name and keep their "
            f"output ({listed}); a queued render of it parks as blocked — use "
            f"`rytp render remove <id>` to delete an output too"
        )

    if dry_run:
        return CommandResult(
            columns=("render", "state", "output"),
            rows=renders,
            message=f"would remove {path} ({size} bytes){note}",
        )
    if not yes:
        raise RytpError(
            f"{path} ({size} bytes) would be deleted and a cut list can hold "
            f"hand-edited timings that exist nowhere else; pass --yes to confirm, "
            f"or --dry-run to see what would go{note}"
        )
    path.unlink()
    return CommandResult(
        columns=("render", "state", "output"),
        rows=renders,
        message=f"removed {path} ({size} bytes){note}",
    )


register(
    Command(
        name="assemble.remove",
        group="assemble",
        summary="Delete a cut list file. Renders that used it are reported, not touched.",
        params=(
            Param("name", str, "Cut list name.", positional=True),
            Param("dry_run", bool, "Show what would be deleted and stop.", default=False),
            Param("yes", bool, "Confirm the deletion. Required.", default=False),
        ),
        handler=assemble_remove,
    )
)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_commands_assemble.py -v`
Expected: every test in the file passes.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp tests && python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/commands/assemble.py tests/test_commands_assemble.py
git commit -m "feat: add assemble remove with a dry run and a renders warning"
```

---

### Task 12: Determinism, the query budget, and the gate

Two properties that no earlier test can prove on its own, and the gate that says the part is done.

**Determinism is a file-level property.** Earlier tests compare `Plan` objects; design §8 promises the same *output*, and the output is a file. This checks the bytes.

**The query budget is what makes the design's own claim true.** Design §7 says assembly needs "no n-gram table" because the walk is a pointer walk. That is only true if the number of queries is bounded by the length of the target rather than by how often its words were said — otherwise the corpus would need an index this part deliberately does not build. `sqlite3.Connection.set_trace_callback` counts statements without touching the code under test.

**Files:**
- Create: `tests/test_assemble_determinism.py`
- Test: itself, plus the whole suite

**Interfaces:**
- Consumes: everything above. Produces nothing new.

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_determinism.py`:

```python
"""Same input, same bytes — and a walk whose cost is the target, not the corpus."""

from __future__ import annotations

from pathlib import Path

from rytp.assemble import AssembleControls, assemble_target, dumps_cutlist, write_cutlist
from rytp.db import Database
from tests.assembly_corpus import add_video, add_words

CREATED = "2026-09-21T09:00:00+00:00"
TARGET = "мы все понимаем что это неизбежно"


def build(db: Database, videos: int) -> None:
    """A corpus of ``videos`` videos that between them say the target."""
    for index in range(videos):
        first = add_video(db, external_id=f"VIDEO_A{index}", title=f"A{index}")
        add_words(db, first, "мы все понимаем что это")
        second = add_video(db, external_id=f"VIDEO_B{index}", title=f"B{index}")
        add_words(db, second, "все понимаем что это неизбежно")


def test_two_runs_write_the_same_bytes(db: Database, tmp_path: Path) -> None:
    build(db, 3)
    first = write_cutlist(
        assemble_target(db, TARGET, name="демо", created_at=CREATED), tmp_path / "one.toml"
    )
    second = write_cutlist(
        assemble_target(db, TARGET, name="демо", created_at=CREATED), tmp_path / "two.toml"
    )
    assert first.read_bytes() == second.read_bytes()


def test_a_seed_changes_the_bytes_and_then_keeps_them(db: Database) -> None:
    build(db, 3)

    def dump(seed: int) -> str:
        return dumps_cutlist(
            assemble_target(
                db, TARGET, name="демо", controls=AssembleControls(seed=seed), created_at=CREATED
            )
        )

    unseeded = dump(0)
    assert dump(0) == unseeded
    seeded = {dump(seed) for seed in range(1, 12)}
    assert len(seeded) > 1
    assert dump(7) == dump(7)


def test_the_knob_changes_the_bytes(db: Database) -> None:
    build(db, 3)

    def dump(consistency: float) -> str:
        return dumps_cutlist(
            assemble_target(
                db,
                TARGET,
                name="демо",
                controls=AssembleControls(consistency=consistency),
                created_at=CREATED,
            )
        )

    assert dump(0.0) != dump(1.0)


def count_statements(path: Path, videos: int) -> int:
    """How many SQL statements one assemble_target costs on this corpus."""
    database = Database(path)
    try:
        database.migrate()
        build(database, videos)
        statements = 0

        def trace(_sql: str) -> None:
            nonlocal statements
            statements += 1

        database.conn.set_trace_callback(trace)
        assemble_target(database, TARGET, name="демо", created_at=CREATED)
        database.conn.set_trace_callback(None)
        return statements
    finally:
        database.close()


def test_the_query_count_follows_the_target_and_not_the_corpus(tmp_path: Path) -> None:
    """Design §7: the walk is a pointer walk, so "no n-gram table is needed"."""
    small = count_statements(tmp_path / "small.db", 3)
    large = count_statements(tmp_path / "large.db", 30)
    assert small == large
    assert small < 40, f"{small} statements for a six-word target is too many"


def test_a_caption_only_corpus_assembles_nothing_at_all(db: Database) -> None:
    """The one rule that must never leak: design §8, captions are not cuttable."""
    video_id = add_video(db, external_id="VIDEO_A")
    add_words(db, video_id, TARGET, source="caption")
    cutlist = assemble_target(db, TARGET, name="демо", created_at=CREATED)
    assert cutlist.fragments == ()
    assert len(cutlist.gaps) == len(TARGET.split())
    assert all(slot.substitutions == () for slot in cutlist.gaps)
```

- [ ] **Step 2: Run it and watch it fail or pass honestly**

Run: `python -m pytest tests/test_assemble_determinism.py -v`
Expected: all pass with no new code — everything it needs already exists. If `test_the_query_count_follows_the_target_and_not_the_corpus` fails with `small != large`, something is querying per occurrence instead of per step; find it before going on, because it is the bug this whole design exists to avoid. If it fails on the `< 40` bound, print `small` and check the arithmetic before relaxing the number: six distinct tokens is six occurrence queries, at most five extension steps per position, one acoustics read, and no padding or substitution work.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest -q`
Expected: every test in the repository passes, Parts 1–4 included.

- [ ] **Step 4: Lint and type-check the whole package**

Run: `python -m ruff check rytp tests`
Expected: `All checks passed!`

Run: `python -m mypy rytp`
Expected: `Success: no issues found`.

- [ ] **Step 5: Run it for real, through the CLI**

The suite mocks nothing here — assembly has nothing to mock — but a green suite still does not prove the commands are wired onto the surface. Drive them by hand against a scratch data tree:

Point `RYTP_DATA` at a scratch directory inside the repository (gitignored, and on the same filesystem on every platform), with the virtualenv activated:

```bash
python - <<'PY'
import pathlib, shutil
root = pathlib.Path(".rytp-smoke")
shutil.rmtree(root, ignore_errors=True)
root.mkdir()
PY
```

Then, with `RYTP_DATA` set to `.rytp-smoke` for the three commands (`set RYTP_DATA=.rytp-smoke` on Windows `cmd`, `$env:RYTP_DATA=".rytp-smoke"` in PowerShell, `RYTP_DATA=.rytp-smoke` prefixed per command on a POSIX shell):

```
python -m rytp assemble plan --help
python -m rytp assemble suggest дела
python -m rytp assemble plan "мы все понимаем"
```

and read `.rytp-smoke/cutlists/мы-все-понимаем.toml`.

Expected: `--help` lists `--consistency`, `--seed`, `--pad`, `--speaker`, `--exclude` and `--force`; `suggest` prints "nothing close to 'дела' is cuttable in the corpus" and exits 0; `plan` writes a cut list whose slots are all gaps, because the scratch corpus is empty, and exits 0. That last one is the honest behaviour design §8 asks for — report the gap plainly, do not fail.

- [ ] **Step 6: Commit**

```bash
git add tests/test_assemble_determinism.py
git commit -m "test: pin assembly determinism and the pointer-walk query budget"
```

---

## What Part 6 can rely on

The cut list is the contract between Part 5 and Part 6, so it is restated here in one place.

**Location.** `cutlists/{name}.toml` under `RYTP_DATA` (contracts §7), reachable as `rytp.config.paths().cutlist(name)` or `rytp.assemble.cutlist_path(name)`. Render addresses a cut list **by name**.

**API.** `load_cutlist(path) -> CutList` and `read_cutlist(name) -> CutList` both raise `CutlistError(RytpError)` — one line, no traceback — for anything unreadable. `write_cutlist(cutlist, path) -> Path`. `Slot.as_fragment() -> Fragment` gives the contracts §4 type.

**Shape.** Top level: `schema_version` (must be 1), `name`, `target`, `created_at`, a `[params]` table, and one ordered `[[slot]]` array. **Document order is the output timeline.** There is no index field. Each slot has `kind = "fragment"` or `kind = "gap"`, `target_first` / `target_last` (inclusive indices into the target's tokens) and `text`. A fragment adds `video_id`, `first_word_ord`, `last_word_ord`, `start_ms`, `end_ms`, and optionally `align_score`, `cost`, `video_speaker_id`, `speaker_label`, `gap_before_ms`. Nested `[[slot.alternative]]` tables give other sources for that slot; nested `[[slot.substitution]]` tables give stand-ins for a gap, each carrying full fragment fields so a gap can be filled by copy-paste. Optional keys are **omitted**, never written empty.

**Division of labour.**

| | Part 5 | Part 6 |
|---|---|---|
| `start_ms` / `end_ms` | authoritative, padding already applied | obey exactly, never recompute |
| `gap_before_ms` | written only if a human added it | absent means you choose the pause (design §9) |
| `pad_ms` in `[params]` | provenance only | ignore |
| Speaker | `video_speaker_id` + `speaker_label`, both nullable | use for per-speaker pause statistics; fall back to per-video |
| `render` job kind | never enqueued here | yours (contracts §5) |
| Dangling `video_id` | `assemble.show` displays the slots and says the cut list is **not renderable**, naming the ids | **abort the whole render** |

**A cut list naming a video that is no longer in the catalog is corrupt input, and render refuses it outright.** Part 5 said otherwise at first — "render only the fragments you can read" — and Part 6 was right to reject that. A partial render is a file that looks finished, is silently shorter than the sentence the owner typed, and says something he did not write; a refusal is one line of stderr and a cut list he can fix by hand, which design §8 makes the intended workflow anyway. Note that this is the opposite of a **gap**, which is also a hole but a *declared* one: a gap is Part 5 reporting plainly that the corpus cannot say a word (design §8, "report the gap plainly and render everything else"), whereas a dangling id is the file disagreeing with the database about what exists. Declared holes render; undeclared ones do not. `assemble.show` still displays a broken cut list — refusing to *show* the file you need to repair would be perverse — but it says plainly that it cannot be rendered.

---

## Self-review

Run against the spec after finishing the plan, as the writing-plans skill requires.

**Spec coverage — design §8, clause by clause.**

| Design §8 says | Where |
|---|---|
| "Walk the target text left to right […] longest run […] exists contiguously" | Task 2 `build_run_table`; Task 4 chooses among the runs rather than taking the longest blindly, justified in "Why a DP and not greedy-longest" |
| "Fewer and longer fragments means fewer seams" | Task 3 `Weights.seam`; Task 1 `ASSEMBLE_WEIGHTS_FEWEST_SEAMS` |
| "one knob from fewest seams to most consistent sound, defaulting toward fewer seams" | Task 3 `weights_for`; `ASSEMBLE_DEFAULT_CONSISTENCY = 0.25` |
| "Consistency is judged from `video_acoustics`" | Task 3 `load_acoustics`, `acoustic_distance` |
| "preferring few distinct source videos is a strong and cheap proxy" | Task 3 `transition_cost`; Task 4 carries the last source in the DP state |
| "Only cuttable words are eligible" | Task 2 `_eligibility_sql`; Task 11 end-to-end guard |
| "Determinism. Same input gives the same output." | Task 3 `rounded`; Task 4 `_step_key`; Task 11 byte comparison |
| "A user-supplied seed shakes up choices among near-equal candidates." | Task 3 `jitter`; Task 4 tests both directions |
| "Controls: exclude specific videos, restrict to a speaker, set the seed, add padding" | Task 2 `MatchFilters`, Task 5 `pad_fragments`, Task 9 `AssembleControls`, Task 10 flags |
| "report the gap plainly and render everything else" | Task 4 gap edges; Task 10 `_summary` |
| "Offer ranked substitutions — same stem first, then […] edit distance" | Task 6 |
| "never apply one silently" | Task 6 returns them; Task 7 writes them beside the gap; nothing applies them |
| "True phonetic matching is not in scope." | stated in Task 6; no phonetic code anywhere |
| "Every fragment keeps its ranked alternatives so a choice can be swapped" | Task 4 `_alternatives`; Task 7 `[[slot.alternative]]`; Task 5 pads them so a swap is usable |
| "The cut list is a plain editable file […] hand-edit it and re-render" | Tasks 7 and 8 |
| contracts §5 "Deletion": `assemble.remove` removes "A cut list file" | Task 11 — `--dry-run` and `--yes` per contracts, no job, no `renders` writes |
| "Whole words only for now, but timings stay precise enough" | every fragment is a whole-word run; `start_ms`/`end_ms` are per-word boundaries, so sub-word cutting later needs no schema change |

Contracts coverage: §1 global constraints are the Global Constraints section; §2 puts `match.py`, `score.py`, `cutlist.py` under `rytp/assemble/` and `assemble.py` under `rytp/commands/`; §3 is read exactly as written, including `source = 'aligned'` as the definition of cuttable; §4's `Fragment` is produced by `Slot.as_fragment` and `RytpError` subclasses everything raised; §5's registry is used for all three commands, its Speaker filters rule is obeyed by consuming Part 1's one shared resolver and offering `--speaker` only (Task 9 says why `--video-local-speaker` is out of scope here), and no job kind is claimed; §7 puts cut lists at `cutlists/{name}.toml`; §8's error, time, database and test conventions hold throughout.

**Placeholder scan.** No "TBD", no "add error handling", no "similar to Task N", no test described without its code. Every step that produces code contains that code. Two steps deliberately defer to what Part 1 built rather than restating it: Task 10's note about how `build_app` renders a multi-word parameter name, and Task 12 Step 3's "the whole suite" — both name exactly what to check.

**Type consistency.** `CandidateRun` is produced only by `_run_from` and consumed by `fragment_cost` (through `RunLike`), `_alternatives`, `pad_fragments` and `from_plan`, with the same field names throughout. `PlanSlot.target_last` and `Slot.target_last` are both inclusive, stated in both dataclasses. `Plan.slots` is `tuple[PlanSlot, ...]`; `CutList.slots` is `tuple[Slot, ...]` — two different types with the same role at two layers, which is why `from_plan` exists and why the file structure table names both. `SubstitutionHit` (in `match.py`, carries a `CandidateRun`) becomes `Substitution` (in `cutlist.py`, flat fields) at exactly one place, `from_plan`. `MatchFilters.video_speaker_ids` is `frozenset[int] | None` everywhere, `None` meaning "any speaker" and never an empty set — `_filters` raises rather than passing an empty one.

**No health check is registered.** Contracts §5's `doctor` section lists what each part probes; Part 5 probes nothing because it has nothing external to probe. Assembly is pure logic over SQLite — no binary, no model, no network, no optional extra — so the only thing that could be wrong is the database, and Part 1 already checks that. Registering an always-green check would be noise in `doctor`'s output.

**Three things deliberately not done.**

- **No `assemble` job kind, no worker integration.** Contracts §5's table does not list one and design §5 does not queue assembly. Adding one would be a design change.
- **No transcript or search dependency.** This part never imports `rytp.index`. Search (Part 4) and assembly answer different questions over the same rows, exactly as design §3 says.
- **No sub-word cutting and no phonetic matching.** Design §8 rules both out, and the whole-word boundaries stored here keep the first one possible later without a schema change.
