# QA fix batch — the first real run's 36 entries, partitioned for concurrent agents

**Source:** `BUGS.md` at the repo root, written while driving rytp by hand on the
owner's Windows machine on 2026-09-24. 36 entries, a prioritised *Fix first*
list, and two design questions.

**Shape of this plan:** nineteen tasks. Each is handed to a separate
implementation agent working **in the same tree**, so the partition is by
**file ownership**, not by bug number. **No two tasks ever edit the same file
concurrently.** Where entries collide on a file they are merged into one task;
where merging would delay a Fix-first item behind unrelated work, the tasks are
sequenced instead and the file's hand-over is recorded in §2a below. Nothing is
left to chance: every shared file has a named first owner and a named successor.

`TODO.md` is out of scope in its entirety.

---

## 0. Rules every task inherits

Read these before starting. They are not repeated per task.

**Platform.** Python >= 3.11, must run on Windows. No POSIX-only calls, no
symlinks, no shell pipelines, `subprocess` with list arguments. **Never assume
`/` is the path separator, in code or in tests** — three tests were fixed for
exactly that on 2026-09-24. Use `pathlib`; compare against `str(Path(...))`,
never a hardcoded `"/a/b"`.

**Anonymity.** No YouTube video ids, channel ids, channel names or URLs in any
file, tests and fixtures included. Use `VIDEO_A`, `CHANNEL_ONE`,
`https://example.invalid/...`. Russian test strings are wanted.

**Contracts.** `docs/superpowers/specs/2026-09-21-rytp-contracts.md` is binding.
Task 1 is the only task that may amend it. If your task needs a deviation that
Task 1 did not write down, **stop and report** — do not vary silently.

**Schema.** `rytp/db/schema.py`'s `MIGRATIONS` is an ordered `(version, sql)`
list. Append only; never edit or renumber an existing tuple. **Task 2b is the
only task that may touch it.**

**Constants.** Every tunable lives in `rytp/constants.py` with a comment citing
its design section. **Task 2a is the only task that may touch it**, and it adds
every constant this batch needs up front. If you find you need one Task 2a did
not add, stop and report rather than editing the file.

**The consistency net stays strict.** `tests/test_consistency_{flags,jobs,
schema,surfaces}.py` are the cross-part net. A new command must reach both the
CLI and the TUI palette. A flag has one spelling and one meaning. Do not relax
an assertion to make your change pass; if the net catches you, the change is
wrong or the net needs a deliberate, stated amendment.

**Tests.** Activate `~/.cache/rytp-dev-venv`, then:

```
RYTP_TEST_TMP=<a local directory outside the repo> python -m pytest
ruff check rytp tests
mypy rytp
```

`RYTP_TEST_TMP` must point outside the repo — the checkout is on an SMB mount
and scratch I/O over it takes the suite from ~75 s to over ten minutes. **Never
pass `-q`**: `pyproject.toml` already sets `-ra -q`, and a second one suppresses
the summary line.

**Test file ownership is real ownership.** Each task lists the test files it
owns. If your change breaks a test file you do not own, **stop and report** —
do not edit it. `tests/conftest.py`, `tests/fakes.py` and
`tests/test_m1_end_to_end.py` are unowned by default; touching them requires
reporting first (Task 6 has a conditional claim on the M1 walk).

**Before you start, confirm your test claim.** The lists below were built by
grepping the test tree for importers of each owned module, but a list written in
advance can be wrong. Run, for every module you own:

```
grep -rl "<module path or a symbol it exports>" tests/
```

If something comes back that no task owns, claim it in your report before
editing. If something comes back that **another** task owns, your change is
reaching into its territory — say so rather than editing across the line. A
purely additive change (adding a `report()` call, adding a class attribute) can
often leave the other task's tests untouched; say which case you are in.

**Nothing is committed and nothing should be.** The working tree already carries
2026-09-24's uncommitted work: the `settings` command group, three Windows test
fixes, and a hand edit to `rytp/transcribe/align/wav2vec2.py`. **Preserve it.**

**Handler contract.** Handlers never print and never call `sys.exit`; they raise
and the surface formats the error (contracts §8). A job handler may return a
short note, stored in `jobs.note` and shown by `jobs.list` — that is the channel
for "this succeeded, and there is something you should know".

---

## 1. Conventions this batch pins, written once here

Three cross-task conventions. They are in the plan text rather than in one
task's file so that tasks which cannot share a file can still agree.

### 1a. Score precedence and scale

`words.align_score` currently holds at least three incommensurable things
(entry 13). From this batch on, every score is written with the scale that
produced it, in a new `words.align_scale` column.

| What ran | `align_score` | `align_scale` |
|---|---|---|
| An aligner that reports a score (wav2vec2) | the aligner's number | `logprob` |
| An aligner that reports none (MFA), refine off | `NULL` | `none` |
| No aligner, `--refine` on (energy boundary scoring) | the 0–1 energy measure | `energy` |
| No aligner, no refine | `NULL` | `NULL` |
| Rows written before this batch | whatever is there | backfilled `unknown` / `energy` |

An aligner that reports a score **and** was refined keeps the aligner's score
and scale: refinement moves boundaries, it does not re-measure alignment. Where
both exist the report may show either, but `align_scale` always names the one
stored.

No scale is ever converted into another. Normalising would be lossy and would
bake in a conversion nobody can justify (entry 13's recorded decision).

### 1b. Device selection and reporting

Entry 34: four of the five engines never place their tensors, so an RTX 3080
idles while an i7 runs inference. Every engine adapter adopts the same shape:

- A `device` parameter, `"auto" | "cuda" | "cpu"`, default `C.ENGINE_DEFAULT_DEVICE`
  (`"auto"`), carried in the out-of-process request dict as `"device"`.
- The child resolves `"auto"` to `"cuda"` when `torch.cuda.is_available()` else
  `"cpu"`, moves both the model and every input tensor to it, and returns the
  **chosen** device in its response dict as `"device"`.
- The parent surfaces the chosen device: in the command result message for a
  foreground run, and in the job note for a queued one. "Silently ran 40×
  slower" is exactly the failure this project keeps producing.
- An engine with no torch (MFA) reports `"device": "n/a"`.

Task 5 applies this to the two aligner adapters; Task 7 applies it to
`gigaam`, `pyannote`, `embed` and makes `whisper` *report* what it already
selects. `rytp/transcribe/align/wav2vec2.py` already has the auto-selection
applied by hand — Task 3 must preserve it and Task 5 completes it.

### 1c. Progress

One seam, three surfaces (entries 3, 10, 22). `rytp/progress.py` (Task 8a) holds
a sink resolved from a `ContextVar`:

```python
def report(stage: str, *, done: int | None = None, total: int | None = None,
           detail: str = "") -> None: ...
```

- **Default sink**: one carriage-return-updated line on `stderr`, **only when
  `sys.stderr.isatty()`**. So a CLI run in a terminal gets progress with no
  change to `rytp/cli.py`, and piped output and the test suite stay clean.
- **Worker sink**: writes `jobs.progress` for the running job, throttled to
  `C.PROGRESS_DB_INTERVAL_MS`. `jobs.list` shows it.
- **TUI sink**: posts a Textual message from the worker thread; the event loop
  is never blocked.

Emitters call `report` and know nothing about sinks. The unit is whatever the
work already counts — for transcription and alignment that is the chunk, which
already exists (`plan_chunks`, and the completion table already prints
`chunks = 38`).

---

## 2. Dependency order and waves

**Task 2 is split in two** so that the constants, which fifteen tasks need and
which depend on no prose, are not gated behind the contracts amendment:

- **T2a constants** — `rytp/constants.py`. No prerequisites.
- **T2b migrations** — `rytp/db/schema.py`. After T1, because the amendment is
  written first.

```
T1 contracts ──► T2b migrations ──┐
                                  ├─► T5 scale ──► T6 assembly ──┐
T2a constants ──┬─────────────────┘                              │
                ├─► T7 engine-seam ──► T8b engine progress       ├─► T14 time
                ├─► T9 cli ──► T8c progress display              │
                ├─► T8a progress ──► T11 tui-home                │
                ├─► T13 cutlist                                  │
                └─► T15 videos-list ──┬─► T16 videos-screen      │
T3 w2v2 ────────────────────────────┐ └──────────────────────────┘
T4 tui-markup ──┬─► T11  ├─► T12 ───┴─► T16
                ├─► T13
                └─► T8c
T10 palette (independent)
```

| Wave | Tasks that can run **concurrently** | Why |
|---|---|---|
| A | **T1, T2a, T3, T4, T10** | No prerequisites, disjoint files. Includes Fix-first #1 (T3) and #4 (T4). |
| B | **T2b, T7, T9, T12, T13, T15** | T2b needs T1. T7/T9/T13/T15 need T2a's constants. T12/T13 need T4. |
| C | **T5, T8a** | T5 needs T2b (the column) and T3 (the adapter files). T8a needs T2a and T2b. |
| D | **T6, T8b, T8c, T11, T16** | T6 needs T5. T8b needs T5+T7+T8a. T8c needs T4+T9+T8a. T11 needs T4+T8a. T16 needs T12+T13+T15. |
| E | **T14** | Needs T6 (the recorded tier) and T15 (`commands/catalog.py`). |

Practical concurrency is five agents in wave A, six in wave B, falling to one in
wave E.

### 2a. Shared files, and the order they change hands

Fourteen files are edited by two tasks. Each is listed here with its first owner
and its successor. **The successor must not start until the first owner has
finished**; that is the whole of the concurrency contract.

| File | First | Then | Why not merged |
|---|---|---|---|
| `rytp/transcribe/align/wav2vec2.py` | T3 | T5 | Merging would delay Fix-first #1 behind the schema chain. |
| `rytp/transcribe/align/mfa.py` | T3 | T5 | Same. |
| `rytp/tui/app.py` | T4 | T11 | Merging would delay the crash fix behind progress work. |
| `rytp/tui/screens/cutlist.py` | T4 | T13 | Same. |
| `rytp/tui/screens/jobs.py` | T4 | T8c | Same. |
| `rytp/tui/screens/search.py` | T4 | T12 | Same. |
| `rytp/transcribe/pipeline.py` | T5 | T8b | Two unrelated concerns; merging makes one agent hold both. |
| `rytp/transcribe/subproc.py` | T7 | T8b | Same. |
| `rytp/commands/ingest.py` | T9 | T8c | Same. |
| `rytp/commands/catalog.py` | T15 | T14 | Same. |
| `rytp/commands/assemble.py` | T6 | T14 | Same. |
| `tests/test_transcribe_align.py` | T3 | T5 | Follows its module. |
| `tests/test_transcribe_subproc.py` | T7 | T8b | Follows its module. |
| `tests/test_commands_ingest.py` | T9 | T8c | Follows its module. |
| `tests/test_commands_assemble.py` | T6 | T14 | Follows its module. |
| `tests/test_catalog_videos.py` | T15 | T14 | Follows its module. |

### Deviation from `BUGS.md`'s *Fix first* ordering, and why

*Fix first* reads: 36, 26, 28, 20, then {3, 10, 22}.

- **36 (T3) and 20 (T4) start in wave A**, first thing, with no prerequisites.
  The ordering is honoured where it costs nothing.
- **26 and 28 (T6) land in wave D, not second.** The concrete reason is in
  `BUGS.md` entry 26 itself: of the three fixes it lists, option 1 — "land the
  scale tag from entry 13, and compare per scale" — is named the principled one,
  and the other two are named as traps. That fix *requires* the contracts
  amendment (T1), the migration (T2) and the writers (T5). Doing 26 sooner means
  doing it by option 2 or 3, which `BUGS.md` argues against. So the batch
  delivers the ordering's intent rather than its sequence.
- **{3, 10, 22} (T8a/T8b/T8c/T11) span waves B–C**, which is later than nothing
  else but earlier than their Fix-first rank suggests, because T8a has only one
  prerequisite.

---

## 3. Contracts changes queued in this batch

Three, all written by Task 1 **before any code**:

1. **`words.align_scale`** (entry 13) — a new column; contracts §3 schema block
   and the "Three transcript tiers" section.
2. **`jobs.progress`** (entries 3, 10) — a new column; contracts §3 and the "Job
   handlers" section, which must say progress is distinct from `note`.
3. **`--allow-timed`** (D1) — an override, not a gate; contracts §3's "Cuttable
   is `source = 'aligned'` and nothing else may be cut" gains an explicit,
   named exception, and contracts §5 gains the flag.

Two smaller amendments ride along:

4. **`GROUP_SUMMARIES`** (entry 6) — contracts §5 fixes the registry's shape and
   has no group-level field. Adding one is a contracts change.
5. **Engine resolution error type** (entry 16) — contracts §6 says
   `resolve_*` raise `ValueError`. Task 1 records that the raised class is
   `UnknownEngineError(RytpError, ValueError)`, which satisfies the existing
   wording *and* the CLI funnel.
6. **An engine notes channel** (entry 36) — `Aligner.align` returns
   `list[Span]` and has nowhere to report a non-fatal finding. Task 1 adds a
   `notes: list[str]` attribute to the three protocols rather than widening the
   return type.

---

## Task 1 — Amend the contracts

**Closes:** nothing on its own. Unblocks entries 13, 26, 27, D1, 6, 16, and the
progress work.

**Owns:**
- `docs/superpowers/specs/2026-09-21-rytp-contracts.md`
- `docs/superpowers/2026-09-25-contracts-amendments.md` (new)

**Must not touch:** any code, any test, `BUGS.md`, `TODO.md`.

**What to write.**

*In the amendment document*, one section per change: what is changing, the entry
or design question that forces it, what breaks if it is not made, and the
alternative that was rejected. This is the justification record; the contracts
file itself stays terse.

*In the contracts file*:

1. **§3 schema** — add `align_scale TEXT` to the `words` DDL block with a comment
   naming the permitted values `energy | logprob | none | unknown` and stating
   that the constraint is enforced in Python, not by a `CHECK`, because SQLite
   cannot add a `CHECK` by `ALTER TABLE`. Add `progress TEXT` to the `jobs` DDL
   block. Note in both places that the live tables gain these by migration 13
   and 14 and that the DDL block is the target state.
2. **§3 "Three transcript tiers"** — replace the bare sentence "Cuttable is
   `source = 'aligned'` and nothing else may be cut" with: `aligned` is the only
   tier cuttable **by default**; `assemble plan --allow-timed` is an explicit,
   per-invocation override that admits `timed` words with no threshold and no
   quality logic, and every fragment records its tier so a degraded cut is
   visible in the render's source list rather than only at plan time. Record why
   a threshold was dropped (a threshold cannot be written against a scale that
   changes per engine) and why the spelling is not `--force` (`assemble plan`
   already has a `--force` meaning "replace an existing cut list").
3. **§3 score precedence** — copy §1a of this plan verbatim as the normative
   table. State that MFA reporting no score stays valid: §6 already makes
   `Span.score` optional.
4. **§5 command registry** — add `GROUP_SUMMARIES: dict[str, str]` beside
   `COMMANDS`, described as: a short phrase saying what a group is *about*,
   read by both surfaces, with the command list generated rather than written
   out. State that a group with no entry is an error the consistency suite
   catches, so a new group cannot ship undescribed.
5. **§5 job handlers** — state that `jobs.progress` is written by the worker
   while a job runs and is advisory and lossy: it may be stale, it is cleared
   when the job leaves `running`, and it never affects job state. It is not
   `jobs.note`, which is a handler's return value and survives completion.
6. **§6 engine protocols** — record `UnknownEngineError(RytpError, ValueError)`
   as the class `resolve_transcriber` / `resolve_aligner` / `resolve_diarizer`
   raise, and add `score_scale: str` and `device: str` to the three protocols as
   class-level declarations, cross-referencing §1a and §1b of this plan.
7. **§6, the notes channel.** `Aligner.align` returns `list[Span]` and has
   nowhere to put a non-fatal finding, so Task 3's "a word received no frames"
   note currently has no path to `jobs.note`. **Do not widen the return type** —
   every caller and every fake would change. Instead declare an instance
   attribute `notes: list[str]` on the three engine protocols: an engine appends
   to it during a call and the pipeline drains it afterwards via
   `getattr(engine, "notes", [])`, clearing it between calls. State that it is
   advisory, never a failure, and that an engine that reports nothing simply
   never defines it. This is the same reasoning that gave job handlers a return
   note: an operation that can only raise or stay silent has nowhere to put
   "this succeeded, and there is something you should know".

**Done looks like:** a reader of the contracts file alone can implement tasks 2,
5, 6, 7, 8a and 9 without reading `BUGS.md`, and the amendment document explains
every change's cause.

**Tests:** none. Task 2b adds the schema-consistency assertions.

---

## Task 2a — Constants foundation

**Closes:** no entry on its own. It is the sole owner of `rytp/constants.py`, so
that fifteen other tasks never collide on it.

**Depends on:** nothing. It is in wave A precisely because the constants gate
six wave-B tasks and depend on no prose.

**Owns:**
- `rytp/constants.py`

**Must not touch:** anything else, `rytp/db/schema.py` included — that is
Task 2b's.

Append the constants table below under a new `# --- 2026-09-25 QA fix batch ---`
heading at the end of the file, and make the one in-place value change. Then
stop: this task adds no behaviour, and nothing reads most of these until its
owning task runs. `ruff` and `mypy` must still be clean.

---

## Task 2b — Schema migrations

**Closes:** no entry on its own. Sole owner of the append-only migration list.

**Depends on:** Task 1 (the amendment is written before the DDL that implements
it), Task 2a (the constants the migration comments cite).

**Owns:**
- `rytp/db/schema.py`
- `tests/test_schema.py`
- `tests/test_consistency_schema.py`

**Must not touch:** anything else. In particular not `rytp/db/queries.py`
(Task 15), not `rytp/models.py` (Task 7), and not `rytp/constants.py` (Task 2a).

**Migrations to append.** Three new tuples, 13 through 15. Never edit 1–12.

- **13 — `words.align_scale`.**
  `ALTER TABLE words ADD COLUMN align_scale TEXT;` then two backfills:
  `UPDATE words SET align_scale = 'energy' WHERE source = 'timed' AND align_score IS NOT NULL;`
  and
  `UPDATE words SET align_scale = 'unknown' WHERE source = 'aligned';`
  Note the second has **no `align_score IS NOT NULL` clause**: MFA has never run
  on this machine, so every pre-batch `aligned` row came from the broken
  wav2vec2 aligner whatever its score column holds, and a null-scored one is no
  more trustworthy than a scored one.
  The `unknown` backfill is deliberate and load-bearing: every `aligned` row in
  the owner's live database was produced by the broken aligner of entry 36 and
  may additionally have been sign-flipped by entry 28's hack, and the sign is
  gone so the affected rows cannot be identified. `unknown` marks them as needing
  re-alignment rather than pretending they are comparable. Do not attempt a
  repair.
- **14 — `jobs.progress`.** `ALTER TABLE jobs ADD COLUMN progress TEXT;`
- **15 — a second partial index for the `timed` tier.**
  `CREATE INDEX words_timed ON words(normalized_text) WHERE source = 'timed';`
  A second index rather than widening `words_alignable`, because dropping and
  recreating an existing index inside a migration is a rewrite of something an
  earlier migration created, and because the assembler's default path must stay
  exactly as narrow as it is today. Task 6 asserts the `--allow-timed` query
  reaches it by name under `EXPLAIN QUERY PLAN`.

**Constants Task 2a appends** (listed here, beside the migrations they serve,
so one table covers the whole foundation), each with a comment citing its design
section:

| Constant | Value | For |
|---|---|---|
| `ALIGN_SCALE_ENERGY` | `"energy"` | entry 13 |
| `ALIGN_SCALE_LOGPROB` | `"logprob"` | entry 13 |
| `ALIGN_SCALE_NONE` | `"none"` | entry 13 |
| `ALIGN_SCALE_UNKNOWN` | `"unknown"` | entry 13 |
| `ALIGN_SCALES` | the four above, as a tuple | validation |
| `ASSEMBLE_MIN_ALIGN_BY_SCALE` | `{"energy": 0.0, "logprob": -5.0, "none": None, "unknown": None}` | entries 26, 28 |
| `ASSEMBLE_EXCLUDE_UNKNOWN_SCALE` | `True` | entry 26 |
| `ENGINE_DEFAULT_DEVICE` | `"auto"` | entry 34 |
| `ENGINE_DEVICES` | `("auto", "cuda", "cpu")` | entry 34 |
| `ENGINE_PROBE_TIMEOUT_S` | `60` | entries 7, 33 |
| `HF_SYMLINK_WARNING_ENV` | `"HF_HUB_DISABLE_SYMLINKS_WARNING"` | entry 11 |
| `PROGRESS_TTY_INTERVAL_MS` | `250` | entries 3, 10 |
| `PROGRESS_DB_INTERVAL_MS` | `2000` | entries 3, 10 |
| `TUI_CUTLIST_SHIFT_NUDGE_MS` | `100` | entry 30 |
| `TRANSCRIPT_DEFAULT_LINE_LENGTH` | `80` | entry 15 |
| `CELL_TICK` | `"✓"` | entry 25 |
| `CELL_CROSS` | `"✗"` | entry 25 |

One **value change in place** for Task 2a, which append-only does not forbid (it governs
ordering and removal, not retuning a number): `TUI_CUTLIST_NUDGE_MS` from `40`
to `10` (entry 30 — 40 ms is a good fraction of a phoneme at conversational
speed). Leave `TUI_CUTLIST_COARSE_NUDGE_MS = 250` alone; it means gaps, not
word edges, and `TUI_CUTLIST_SHIFT_NUDGE_MS` is the new Shift step.

`ASSEMBLE_MIN_ALIGN_BY_SCALE["logprob"] = -5.0` is a first guess in the spirit of
Part 7's unvalidated thresholds — comment it as such. `exp(-5)` is about 0.7 %
probability, which admits essentially everything while still excluding a
catastrophic mismatch; it exists so the filter is meaningful, not so it is tight.

**Done looks like:** `python -m rytp doctor` on a database migrated from version
12 reports version 15; `PRAGMA table_info(words)` shows `align_scale`; a
pre-existing `aligned` row reads back `align_scale = 'unknown'`; a pre-existing
`timed` row with a score reads back `'energy'`.

**Tests:** in `tests/test_schema.py`, a migration test that builds a database at
version 12 with one `timed` scored row and one `aligned` scored row, migrates,
and asserts both backfills. In `tests/test_consistency_schema.py`, extend the
declared-columns assertions to cover `words.align_scale` and `jobs.progress`,
and assert `ALIGN_SCALES` matches the values the contracts comment names.

---

## Task 3 — The wav2vec2 aligner actually aligns

**Closes:** entry 36. **Fix first #1.**

**Depends on:** nothing. Starts immediately.

**Owns:**
- `rytp/transcribe/align/wav2vec2.py`
- `rytp/transcribe/align/mfa.py`
- `tests/test_transcribe_align.py`

**Must not touch:** `rytp/transcribe/base.py`, `registry.py`, `subproc.py`,
`pipeline.py`. The `score_scale` and `device` attributes are Task 5's, applied
to these same two files **after** this task lands.

**Preserve the uncommitted hand edit.** `child_main` already carries
`device = "cuda" if torch.cuda.is_available() else "cpu"` and three `.to(device)`
calls. Keep every one of them.

**The defect.** `_word_frames` is documented as grouping the character-level CTC
path back into one entry per word. Nothing in its loop ever appends to `out` —
the trailing `if … : continue` is a no-op at the end of the body — so `out` is
empty when the loop ends and the block commented as a *fallback* runs every
time, handing each word `(last - first + 1) // len(words)` frames and the mean
score of all frames. Every `aligned` word wav2vec2 ever produced has a
fabricated, evenly-spaced boundary.

**A second defect, upstream of the first, which must be fixed or the grouping
cannot work.** The targets are built as

```python
[vocabulary[ch] for ch in " ".join(words) if ch in vocabulary]
```

wav2vec2 CTC vocabularies spell the word delimiter `|`, not `" "`, so
`" " in vocabulary` is false and **every word boundary is silently filtered
out**. The target sequence is the words concatenated with no separator. Even a
correct grouper would have nothing to split on.

**How to implement it.**

1. **Resolve the delimiter.** Take `processor.tokenizer.word_delimiter_token_id`,
   falling back to the id of `"|"` if that attribute is absent. If neither
   resolves, raise `RytpError` naming the model and saying the vocabulary has no
   word delimiter. Do **not** fall back to equal division: fabricating boundaries
   silently is the bug being fixed.
2. **Build targets deliberately, and count per word.** Encode each word's
   characters against the vocabulary, dropping unknowns, and record each word's
   resulting token count *before* joining them with the delimiter. Check the
   casing of the vocabulary against the words — if the vocabulary is lower-case
   and the words are not, whole words vanish, so case-fold to whatever the
   vocabulary uses. A word whose filtered token count is zero is a hard failure:
   raise `RytpError` naming the word, because alignment cannot place a word it
   cannot spell.
3. **Make the grouping a pure function.** Extract it so it takes plain Python
   values only and imports nothing heavy:

   ```python
   def word_frames(
       path: Sequence[int],            # forced_align's per-frame token ids
       scores: Sequence[float],        # forced_align's per-frame scores
       *,
       targets: Sequence[int],         # the target token ids, in order
       delimiter: int,                 # the word-delimiter token id
       blank: int = 0,
   ) -> tuple[list[dict[str, Any]], list[str]]: ...
   ```

   Return the per-word frame entries **and** a list of notes. `child_main` stays
   the only place torch is imported; this function is unit-testable with lists.
4. **Walk the path with a target cursor.** The path forced_align returns is a
   valid alignment of the target sequence, so the target index per frame is
   recoverable: skip blanks; advance the cursor when the emitted non-blank id
   differs from the previous emitted non-blank id, or when a blank intervened
   between two identical ids. That is the standard CTC token-merge rule and it
   handles a doubled letter correctly. The word index is then the number of
   delimiter targets passed.
5. **Emit per word:** `first_frame` = the first frame of that word's own tokens,
   `last_frame` = the last, `score` = the mean of **that word's own** frame
   scores. Not the mean of everything.
6. **Keep a real, narrow, observable fallback.** Only for a word that received no
   frames at all: give it a share interpolated between its neighbours' bounds,
   and append a note naming the word. `child_main` returns the notes in its
   response dict, and `Wav2Vec2Aligner.align` appends them to `self.notes` — an
   instance attribute this task creates and Task 1 records as the engine notes
   channel. **Do not widen `Aligner.align`'s return type**: contracts §6 fixes it
   as `list[Span]` and every caller and fake would change. Task 5 declares the
   attribute on the protocol in `base.py` and drains it into `jobs.note`; this
   task only has to set it. A silent fallback is what produced this bug.
7. **Strengthen the guard.** `AlignmentMismatchError` never fired because the
   count of spans was always right — it checks quantity, not correctness. Add,
   in `Wav2Vec2Aligner.align` after `spans_from_frames`: spans are
   non-decreasing in `start_ms`, and not every span has the same width when
   there is more than one word and the words differ in length. The second is a
   direct regression assertion against equal division.

**The MFA adapter — audit, expect no behaviour change.** MFA is a separate
adapter and has never been run. It has **no equal-division defect**: it takes one
TextGrid interval per word and the count mismatch already raises with a useful
message. The analogous risks are (a) MFA emitting `sil` / `sp` / `spn` intervals
in the words tier, which would shift the count, and (b) `parse_textgrid`'s
xmin/xmax state machine mis-reading a tier header. Audit both, add tests, and
change behaviour only if a test fails. Record the audit's conclusion in the
task's report either way — `BUGS.md` entry 36 asks for it explicitly.

**Done looks like:** given a synthetic CTC path where three words genuinely
occupy 2, 2 and 10 frames, the aligner returns three spans of 2, 2 and 10 frames
with three different scores. `BUGS.md` records the current output for exactly
that case — equal widths, identical `-0.1` scores — so the test is written from
the bug report.

**Tests** (all in `tests/test_transcribe_align.py`, no torch, no network):
- The 2/2/10 case from `BUGS.md`, asserting unequal widths and distinct scores.
- A doubled letter within one word (`"аа"`), asserting the cursor does not
  double-advance.
- A word separated by several consecutive delimiter frames.
- A missing delimiter id raising `RytpError` naming the model.
- A word that encodes to zero tokens raising `RytpError` naming the word.
- A word with no frames producing the interpolated fallback **and** a note.
- `Wav2Vec2Aligner.align` rejecting a fabricated child response whose spans are
  all the same width.
- MFA: a TextGrid containing `sil` and `sp` intervals; a TextGrid whose interval
  count does not match the word count, asserting the existing error.

---

## Task 4 — The TUI stops crashing on square brackets

**Closes:** entry 20. **Fix first #4.**

**Depends on:** nothing. Starts immediately.

**Owns:**
- `rytp/tui/text.py` (new)
- `rytp/tui/app.py`
- `rytp/tui/screens/cutlist.py`
- `rytp/tui/screens/jobs.py`
- `rytp/tui/screens/speakers.py`
- `rytp/tui/screens/transcript.py`
- `rytp/tui/screens/search.py`
- `tests/test_tui_markup.py` (new)

**Must not touch:** any existing TUI test file (`tests/test_tui_app.py`,
`test_tui_cutlist.py`, `test_tui_jobs.py`, `test_tui_speakers.py`,
`test_tui_transcript.py`, `test_tui_search.py`) — they belong to tasks 11, 13
and 16, or to nobody. Put every new assertion in the new sweep file. Do not
change any screen's behaviour, layout or bindings; this task is one mechanical
concern.

**The defect.** `set_status` hands arbitrary text to `Static.update`, which
parses it as Textual console markup. `usage_line` spells optional parameters
`[title=…]`, and square brackets *are* markup syntax, so any message carrying a
usage line kills the app with
`MarkupError: Expected markup value (found '…] [kind=…] [channel=…]')`. It needs
no unusual input: `parse_arguments(COMMANDS["videos.add"], "a b c d e")` raises
the identical message — five bare words.

**Scope is wider than errors.** `show_result` routes `result.message` through the
same call, so any command *result* containing brackets crashes identically.
There are eleven `Static.update` call sites across the app and five screens.

**How to implement it.**

1. `rytp/tui/text.py` exposes one helper — `set_text(widget, text)` — that
   renders plain text with markup parsing off. Two approaches were verified to
   work by hand: `textual.markup.escape(text)` before `update`, and constructing
   `textual.content.Content(text)` without markup parsing. Prefer `Content`:
   escaping mutates the string the user sees if it ever reaches a non-markup
   renderer, whereas `Content` states the intent. Fall back to `escape` if the
   installed Textual makes `Content` awkward, and say which you chose.
2. Route **every** `Static.update` call in the owned files through it.
3. Check `DataTable` cells for the same exposure. `show_result` adds
   `result.rows` straight into a `DataTable`; if Textual parses markup in cells,
   a cut-list path or a usage line in a cell crashes the same way. Fix it the
   same way if so, and say so if not.

**Done looks like:** with `videos.add` highlighted, typing `a b c d e` into the
arguments field and pressing Enter shows the "too many values" message including
its `[title=…] [kind=…]` usage line, and the app stays alive.

**Tests** (`tests/test_tui_markup.py`):
- Assert against **`textual.content.Content.from_markup`**, not `rich.markup`.
  Rich's parser tolerates the string; Textual 8 ships a stricter one, so a test
  written against `rich.markup` passes while the app still crashes. This is
  recorded in `BUGS.md` and is the single most important line of this task.
- A parametrised sweep over every owned module asserting no bare
  `Static.update(` remains — a source-level scan is acceptable and is what keeps
  a new screen from regressing.
- An app-level test driving the exact `videos.add` / `a b c d e` reproduction.
- A `show_result` test with a bracketed `message` and a bracketed cell.

---

## Task 5 — Record the scale alongside the align score

**Closes:** entry 13. Implements §1a and, for the two aligner adapters, §1b.

**Depends on:** Task 2a (the constants), Task 2b (the column), Task 3 (same two adapter
files).

**Owns:**
- `rytp/transcribe/base.py`
- `rytp/transcribe/pipeline.py`
- `rytp/commands/transcribe.py`
- `rytp/transcribe/align/wav2vec2.py`
- `rytp/transcribe/align/mfa.py`
- `tests/fake_engines.py`
- `tests/test_transcribe_pipeline.py`
- `tests/test_commands_transcribe.py`
- `tests/test_transcribe_align.py` *(inherited from Task 3, which lands first)*
- `tests/test_transcribe_integration.py`

**Must not touch:** `rytp/transcribe/registry.py`, `health.py`, `subproc.py`,
`engines/*` — all Task 7's. `rytp/assemble/*` is Task 6's: this task **writes**
the scale, Task 6 **reads** it.

**The defect.** The same column printed `1.00` for a `whisper+energy` run whose
tier was `timed`, and `-0.96` for a `whisper+wav2vec2+energy` run. Both numbers
are real and neither is comparable to the other: `_score_boundaries` yields a
bounded 0–1 energy measure, `torchaudio.functional.forced_align` yields
log-probabilities, MFA reports nothing. The one column a user would check to
decide "is this good enough to cut?" said yes while the tier column said no.

**How to implement it.**

1. `base.py`: add `score_scale: str` to the `Aligner` protocol, and `device: str`
   and `notes: list[str]` to all three protocols, as class-level declarations.
   **`base.py` must stay standard-library-only** — it is imported under a foreign
   interpreter. String and list class attributes are fine; imports are not.
2. The two adapters: `Wav2Vec2Aligner.score_scale = C.ALIGN_SCALE_LOGPROB`,
   `MfaAligner.score_scale = C.ALIGN_SCALE_NONE`. Add the `device` parameter and
   response field per §1b — wav2vec2 threads it through the request and returns
   the chosen device; MFA reports `"n/a"`.
3. `pipeline.py`: `replace_words` and `realign_video` write `align_scale`
   alongside `align_score`, following §1a's table exactly. Extend
   `_WORD_COLUMNS` and both SQL statements. `TranscriptOutcome` gains the scale
   beside `median_align_score`, and the aligner's chosen device so the command
   can report it.
4. `commands/transcribe.py`: the column headed `median align score` is renamed to
   name what produced it — label it by the scale (`median energy score`,
   `median logprob score`), and **leave it blank for a tier where no aligner
   ran and no refine happened**. Add the chosen device to the result message.
   Both `_run_handler` and `_align_handler` print this table; change both.
5. **Drain the engine notes channel.** After each `align` call, read
   `getattr(engine, "notes", [])`, clear it, and accumulate. Return the
   accumulated notes as the job's note so Task 3's "word N received no frames"
   findings reach `jobs.list` on the bulk path and the command result on the
   foreground one. A handler that can only raise or stay silent has nowhere to
   put a non-fatal finding, which is why `jobs.note` exists.

**Done looks like:** `transcribe run <id> --transcriber whisper --refine` prints
a row whose tier is `timed` and whose score column is headed with the energy
scale, not "align score". `transcribe align <id> --aligner wav2vec2` prints a row
headed with the logprob scale. The two are never printed under one heading.
`SELECT DISTINCT source, align_scale FROM words` returns only pairs §1a permits.

**Tests:**
- Fakes in `tests/fake_engines.py` gain a `score_scale` and a `device`, with at
  least one fake per scale so the pipeline is exercised on all three.
- `test_transcribe_pipeline.py`: each row of §1a's table asserted as a stored
  `(source, align_score, align_scale)` triple, including the both-ran case.
- `test_commands_transcribe.py`: the column heading differs between an
  energy-scored run and a logprob-scored run, and is blank for the neither case.
- A test that `rytp/transcribe/base.py` imports nothing outside the standard
  library and `rytp.models` — if one does not already exist, add it; the
  out-of-process seam depends on it.

---

## Task 6 — Assembly can see a wav2vec2 corpus again

**Closes:** entries 26, 27, 28, and design question D1. **Fix first #2 and #3.**

**Depends on:** Task 2a (the constants), Task 2b (the column and index 15), Task 5 (something
must write the scale before anything can read it).

**Owns:**
- `rytp/assemble/__init__.py`
- `rytp/assemble/match.py`
- `rytp/assemble/cutlist.py`
- `rytp/assemble/score.py`
- `rytp/commands/assemble.py`
- `tests/test_assemble_match.py`
- `tests/test_commands_assemble.py`
- `tests/test_assemble_cutlist.py`
- `tests/test_assemble_determinism.py`
- `tests/test_assemble_target.py`
- `tests/test_assemble_score.py`
- `tests/test_assembly_corpus.py`
- `tests/assembly_corpus.py`
- `tests/test_surfaces.py`
- `tests/test_m1_end_to_end.py` *(conditional: only if `assemble list` or the
  new plan message breaks the walk; report before editing)*

**Must not touch:** `rytp/render/*` — Task 14 renders the tier in the source
list. `rytp/constants.py` — Task 2a supplied `ASSEMBLE_MIN_ALIGN_BY_SCALE`.

**Entry 26, the defect.** The matcher requires
`COALESCE(align_score, :default_align) >= :min_align` with `min_align` defaulting
to `0.0`. wav2vec2 writes log-probabilities, negative by definition, so **every
aligned word fails**. The escape hatch is closed too:
`AssembleControls.validated` rejects `min_align_score` outside `0.0..1.0`, so
`--min-align -5` is invalid input. There is no value of the flag that admits a
negatively-scored word. The failure is not an error but an empty result, which
reads as "the phrase is not in the corpus".

**Entry 28, the live-database damage.** `UPDATE words SET align_score =
-align_score WHERE align_score < 0` was applied to the owner's database to get
past entry 26. In log-probability space closer to zero is better, so negation
reverses the ranking — and the matcher orders by
`COALESCE(align_score, :default_align) DESC`, so it now **prefers the
worst-aligned words**. Affected rows cannot be identified after the fact. **26
and 28 retire together:** fixing 26 properly is what makes the hack unnecessary,
and Task 2b's `unknown` backfill plus a re-run of `transcribe align` is what
removes it. This task attempts no row repair.

**How to implement entries 26 and 28.**

1. Replace the single `min_align` comparison with a **per-scale** one.
   `MatchFilters` carries `min_align_by_scale: Mapping[str, float | None]`
   defaulting to `C.ASSEMBLE_MIN_ALIGN_BY_SCALE`. The eligibility SQL becomes a
   scale-keyed disjunction: a row is eligible when its `align_scale` has a floor
   and its score clears that floor, or when its scale has no floor (`none` —
   MFA writes no score and contracts §6 permits it) and therefore no threshold
   applies.
2. `align_scale = 'unknown'` is **excluded** by default. Those rows are the
   pre-batch wav2vec2 output that entry 36 proved is fabricated.
3. **`--min-align` keeps its present meaning and its present validation.**
   Decided rather than widened: it is the **energy** floor, 0–1, and
   `AssembleControls.validated` still rejects anything outside that range. The
   logprob floor is `C.ASSEMBLE_MIN_ALIGN_BY_SCALE` only, with no flag this
   batch. Giving one flag a range that changes per corpus is the trap `BUGS.md`
   names in entry 26's option 3 — "the same trap, moved". Say which scale it
   applies to in the help text. If a logprob override is later wanted it gets
   its own flag with its own name.
4. Keep the `DESC` ordering, which is now correct within a scale because the
   comparison never crosses scales.
5. **The empty result must never be silent again.** This is the acceptance
   criterion that matters most. When `assemble plan` matches fewer words than the
   target has, its message names why, with counts: how many candidate words were
   excluded for tier, how many for `unknown` scale, how many for falling below
   their scale's floor — and it names the command that fixes the common case
   (`transcribe align`) and the flag that works around it (`--allow-timed`).
   A user must never again read "no fragments" and conclude the corpus does not
   contain the phrase.

**How to implement D1, `--allow-timed`.**

- A boolean parameter on `assemble.plan`, default `False`. Not `--force`: that
  spelling is taken on the same command and two meanings on one name is exactly
  what the flag-vocabulary suite forbids.
- It is an **override, not a gate**: no threshold, no quality logic. `aligned`
  remains the only tier cuttable by default; the flag admits `timed` as well.
- **Not `source IN ('aligned','timed')`.** SQLite uses a partial index only when
  the query's `WHERE` implies the index's predicate, and an `IN` over two values
  implies neither `source = 'aligned'` nor `source = 'timed'`. The planner would
  fall back to `words_normalized` and scan the caption tier — precisely the cost
  the partial index exists to avoid, and Task 2b's `words_timed` would never be
  used. **Write the flag's path as a `UNION ALL` of two per-tier queries**, each
  with its own equality predicate, so each half sits on its own partial index.
  Assert both index names under `EXPLAIN QUERY PLAN`; statement counting cannot
  see a plan that scans captions inside one query, which is why the existing
  budget test works by index name.
- **Each fragment records its tier** in the cut-list TOML, always, not only under
  the flag. Task 14 renders it in the source list. Without the recorded tier a
  degraded cut is visible only at plan time, which D1 explicitly rejects.
  **On read the field is optional and defaults to `aligned`**, so every existing
  cut list, every render fixture and `tests/render_fakes.py` keep loading. A
  cut list written before this batch has no tier field and was, by the rule in
  force when it was written, aligned-only.
- `assemble show` reports the tier per fragment, and the plan's summary says how
  many fragments are `timed`.

**How to implement entry 27.**

- **Add `assemble list`**, not an optional argument on `show`. `list` is the
  established verb — `videos list`, `channel list`, `render list`, `speakers
  list`, `jobs list` — and a `show` that lists when given nothing and displays
  when given something is two meanings on one name. Output is **bare names**,
  exactly what `assemble show` and `render run` accept: not paths, not
  `<name>.toml`. It must be copy-pasteable.
- **Accept what the tool printed.** `assemble plan` prints
  `wrote …/cutlists/<name>.toml` and `assemble show <name>.toml` currently fails
  on `<name>.toml.toml`. Strip a trailing `.toml`, and accept a path as well as a
  bare name, in `cutlist_name` / `validate_name`.
- **Point at the new command from the missing-argument error.** That error
  currently comes from Typer before the handler runs, so give `assemble show`'s
  `name` parameter a default of `""` and raise from the handler with
  ``no cut list named; try `assemble list` ``. That keeps the fix inside this
  task's files — `rytp/cli.py` is Task 9's.
- Adding a command means `tests/test_surfaces.py`'s exact command set gains
  `assemble.list`, and the command must reach the TUI palette too.

**Done looks like:**
- On a corpus of wav2vec2-aligned words with `logprob` scale, `assemble plan`
  returns fragments instead of nothing.
- On a corpus of `unknown`-scale words, `assemble plan` returns nothing **and
  says so**, naming the count and `transcribe align`.
- `assemble plan --allow-timed` on a `timed`-only corpus returns fragments, each
  marked `timed` in the written TOML.
- `assemble list` prints names that `assemble show` accepts unchanged, and
  `assemble show <name>.toml` works.

**Tests:** the per-scale eligibility matrix (four scales × above/below floor);
the ordering within a scale; an `EXPLAIN QUERY PLAN` assertion naming
`words_timed` for the `--allow-timed` path and `words_alignable` for the default
path; the diagnostic message's counts; the tier recorded in a round-tripped
TOML; `assemble list` output accepted by `assemble show`; `.toml` stripping;
the missing-name error naming `assemble list`.

---

## Task 7 — Make the engine boundary tell the truth

**Closes:** entries 5, 7, 11, 16, 33, 34 (for `gigaam`, `pyannote`, `embed`,
`whisper`), 35.

**Depends on:** Task 1 (the error class and the protocol attributes), Task 2a
(the constants). Not on Task 5 — the two aligner adapters are Task 5's files and
this task does not open them.

**Owns:**
- `rytp/transcribe/subproc.py`
- `rytp/transcribe/registry.py`
- `rytp/transcribe/health.py`
- `rytp/transcribe/engines/gigaam.py`
- `rytp/transcribe/engines/whisper.py`
- `rytp/diarize/base.py`
- `rytp/diarize/health.py`
- `rytp/diarize/pyannote.py`
- `rytp/diarize/embed.py`
- `rytp/models.py`
- `docs/setup.md` (new)
- `tests/subproc_engine_stub.py`
- `tests/test_transcribe_subproc.py`
- `tests/test_transcribe_registry.py`
- `tests/test_transcribe_health.py`
- `tests/test_diarize_registry.py`
- `tests/test_diarize_health.py`
- `tests/test_doctor.py`
- `tests/test_health_checks.py`
- `tests/test_transcribe_engines.py`
- `tests/test_diarize_pyannote.py`
- `tests/test_diarize_embed.py`
- `tests/test_diarize_embed_stage.py`

**Must not touch:** `rytp/transcribe/base.py`, `align/*`, `pipeline.py`,
`commands/transcribe.py` — Task 5's. `tests/fake_engines.py` — Task 5's; use
`tests/subproc_engine_stub.py` for anything this task needs to fake.

**Note before starting:** `EngineUnavailable` and `EngineSubprocessError` are
**already** `RytpError` subclasses. Entry 5 is therefore not an exception-type
problem; it is that `run_child` builds the error message by embedding
`error['traceback']` verbatim. The funnel prints one line — of a message that
contains a traceback.

**One probe serves four entries.** Entries 7, 33, 34 and 35 all want the same
thing in the engine's child interpreter: does the module import, is CUDA
available, which device would be selected. Build it once.

1. **`subproc.py` gains a probe.** A `probe(interpreter, module, required_module)`
   that runs the same child harness with a tiny built-in request and returns
   `{"module_ok": bool, "module_error": str, "torch": str|None,
   "cuda_available": bool|None, "device": str}`. Timeout
   `C.ENGINE_PROBE_TIMEOUT_S`, never raises, always answers.
2. **Entry 7 — `check_available` and `availability` stop lying.** Today they skip
   the module check for an out-of-process engine on the reasoning that
   `find_spec` here says nothing about the child's — correct reasoning, wrong
   conclusion. Run the probe instead. Five out of five out-of-process engines
   reported `interpreter ok` and none of them could run: `gigaam`, `mfa`,
   `wav2vec2`, `pyannote`, `redimnet`. The column must mean "this will actually
   run". Cache the result per interpreter for the life of the process;
   `transcribe engines` lists six engines and must not spawn six processes twice.
3. **Entry 33 — report the CUDA fact.** `transcribe engines` and `doctor` report
   whether the engine's interpreter has a CUDA-enabled torch. On Windows
   `pip install torch` takes the CPU-only wheel, so an engine venv created the
   obvious way is CPU-only and nothing says so. `doctor` already found the 3080;
   pair the two facts in one line and name the remedy (`HealthResult.remedy`
   exists for exactly this).
4. **Entry 34 — device selection for the remaining engines.** Apply §1b to
   `gigaam`, `pyannote` and `embed`, which construct models and tensors without
   ever moving them. `whisper` already selects; make it *report*. Note the
   consequence recorded in `BUGS.md`: `transcribe`, `align` and `diarize` are all
   declared `pool=gpu` and three of them do not touch the card, so the pool is
   serialising CPU work for no reason. **Do not change the pools in this task** —
   record the finding in your report; it is a queue design decision.
5. **Entry 5 — the install hint survives the subprocess seam.** When the child's
   error payload is a `ModuleNotFoundError`, map the missing module name to the
   owning engine's `extra` attribute and raise
   ``install the gigaam extra: pip install -e ".[gigaam]"`` — one line, no
   traceback. Keep the traceback available behind a debug path or a log, but out
   of the message the funnel prints. CLAUDE.md states this contract and the code
   currently breaks it.
6. **Entry 16 — an unknown engine name is a one-line error.** Define
   `UnknownEngineError(RytpError, ValueError)` in `rytp/models.py` and raise it
   from `resolve_transcriber`, `resolve_aligner` (both in `registry.py`) and
   `resolve_diarizer` (`diarize/base.py`). Dual inheritance keeps contracts §6's
   "raises `ValueError`" literally true and lets the CLI funnel catch it. The
   message is already exactly right — `unknown aligner 'wav2vec'; available: mfa,
   wav2vec2` — only the delivery is wrong, a ~60-line rich traceback. `BUGS.md`
   asks for an audit of every non-`RytpError` raise on a command path: do the
   audit, fix what is in your owned files, and **list anything outside them in
   your report** rather than reaching for another task's file.
7. **Entry 11 — the Hugging Face symlink warning.** Set
   `C.HF_SYMLINK_WARNING_ENV` in the child environment `run_child` builds, and in
   the process environment before an in-process engine loads. It reached the user
   twice, each time with a paragraph about Windows Developer Mode, and it is
   noise from a dependency rather than something rytp asks the user to act on.
   The underlying condition is real — without symlinks the HF cache stores
   duplicates — so `doctor` reports it once, as an advisory line with the disk
   cost, rather than the library printing it per process.
8. **Entry 35 — `docs/setup.md`.** A real user-facing setup document, because
   following the current instructions produces engines that cannot use the card
   this project was designed around. It must cover: the extras install a CPU-only
   torch on Windows and the GPU build needs
   `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
   with `cuXXX` matching the driver; the separate-interpreter workflow, now
   reachable because `settings set engine.interpreter.<name> <path>` exists; and
   that on Python 3.14 a separate 3.12 interpreter is not optional but the only
   way `gigaam`, `wav2vec2`, `pyannote` and `redimnet` run at all (entries 9, 17).
   Do **not** rewrite `README.md`; it is stale and out of scope.

**Done looks like:** `transcribe engines` against an interpreter with no `torch`
reports a state that says so, not `interpreter ok`. Against an interpreter with a
CPU-only torch it says CUDA is unavailable and names the index-url remedy.
`transcribe run <id>` with the extra missing prints one line and exits 1.
`transcribe align 1 --aligner wav2vec` prints one line and exits 1.

**Tests:** the probe against `tests/subproc_engine_stub.py` for module-present,
module-missing, torch-absent, CUDA-absent and timeout; `availability` returning
a not-runnable state where the probe fails; the `ModuleNotFoundError`-to-hint
mapping including the "no traceback in the message" assertion; each `resolve_*`
raising something caught by both `except RytpError` and `except ValueError`;
`doctor`'s new lines. Every test fakes the child; **no test may touch the
network**.

---

## Task 8a — The progress seam and its non-engine emitters

**Closes:** entry 3, and the queued-job half of entries 3 and 10.

**Depends on:** Task 2a (the interval constants), Task 2b (`jobs.progress`).

**Owns:**
- `rytp/progress.py` (new)
- `rytp/jobs/worker.py`
- `rytp/jobs/queue.py`
- `rytp/acquire/ytdlp.py`
- `rytp/acquire/captions.py`
- `rytp/audio/extract.py`
- `rytp/diarize/pipeline.py`
- `tests/test_progress.py` (new)
- `tests/test_jobs_worker.py`
- `tests/test_jobs_queue.py`

Its changes to `acquire/*`, `audio/extract.py` and `diarize/pipeline.py` are
**purely additive** — a `report()` call in an existing loop, silent when no sink
is installed — so `tests/test_acquire_*.py`, `tests/test_audio_extract.py` and
`tests/test_diarize_pipeline.py` should need no edit. If one does, that means
the call is not silent by default and the sink is wrong; fix the sink, not the
test.

**Must not touch:** `rytp/cli.py` (Task 9) — and it needs no change, which is the
point of the default sink. `rytp/transcribe/*` (Tasks 5 and 8b),
`rytp/tui/*` (Tasks 4 and 11), `rytp/commands/ingest.py` (Task 9).

**The defect.** `ingest` and `fetch` sit silently for minutes, indistinguishable
from a hang. Nothing anywhere reports that it is working. This is not broken, but
it is why a download, a model fetch and a network call are all indistinguishable
from a hang — and it is best solved once at the seam every engine passes through
rather than four times.

**How to implement it.** Build §1c exactly. Specifically:

- The `ContextVar` default is the tty sink, throttled to
  `C.PROGRESS_TTY_INTERVAL_MS`, writing to `stderr` and only when
  `sys.stderr.isatty()`. Piped output and pytest capture see nothing, so no
  existing assertion changes.
- The worker installs a db sink for the duration of a handler call, writing
  `jobs.progress` at most every `C.PROGRESS_DB_INTERVAL_MS`, and clears the
  column when the job leaves `running`.
- **The db write must not join the handler's transaction.** Verify that the
  emitting loops sit outside `db.transaction()` — in `transcribe_video` and
  `realign_video` they do today, and Task 8b must keep it so. A progress write
  inside the handler's transaction either vanishes on rollback or is invisible
  until commit, which defeats the purpose. Contracts §8 forbids opening a second
  connection, so write on the shared `db` and keep the call outside.
- Emitters: yt-dlp's download (it reports bytes; forward them), caption fetch,
  wav extraction, and the diarizer's segment pass.

**Done looks like:** `rytp ingest <id>` in a terminal shows a moving line naming
the stage; the same command with stdout piped to a file produces byte-identical
output to today; `jobs list` shows a progress cell for a running job and an empty
one for a finished job.

**Tests:** the sink contract (default, override, restore, nesting); the tty sink
silent when `isatty()` is false; the throttle; the worker writing and clearing
`jobs.progress`; a progress call raising nothing when no sink is installed.

---

## Task 8b — Progress across the engine seam

**Closes:** entry 10 (the first-run model download and the silent transcription
run).

**Depends on:** Task 8a (the sink), Task 5 (`transcribe/pipeline.py`), Task 7
(`transcribe/subproc.py`).

**Owns:**
- `rytp/transcribe/pipeline.py`
- `rytp/transcribe/subproc.py`
- `tests/test_progress_engines.py` (new)
- `tests/test_transcribe_subproc.py`

**Must not touch:** `rytp/transcribe/align/*`, `engines/*`, `registry.py`,
`base.py`, `commands/transcribe.py`.

**The defect.** `transcribe run --transcriber whisper` printed two warnings then
went quiet for a long time while downloading several GiB of `large-v3` weights.
A user who does not know that is happening will kill it. And it is broader than
the first run: a normal `transcribe run` is silent for its whole duration — the
owner reports being able to tell it is working only from the GPU usage graph. The
aligner does the same once its interpreter is configured. So this is every
engine, and it is the first thing a user meets on each one.

**How to implement it.**

1. **The chunk counter costs nothing.** The pipeline already splits audio into
   chunks before transcribing and the completion table already prints the count.
   Emit `report("transcribe", done=i, total=len(chunks))` in
   `transcribe_video`'s loop and the same in `realign_video`'s. Nothing computes
   `chunk 12/38` today only because nothing asks.
2. **Keep the emit outside `db.transaction()`.** Both loops are outside today.
   Do not move them inside.
3. **The model download happens in the child**, whose stderr `run_child`
   currently buffers with `capture_output=True` and discards as noise. Stream it
   instead: read the child's stderr line by line on a reader thread, forward each
   line to the progress sink as a `detail`, and keep the tail for the error
   message exactly as today. The response still comes from the file, never from
   stdout — that choice is deliberate and stays. A first-run fetch then shows the
   library's own progress lines rather than silence.
4. Keep the parent's behaviour identical when no sink is installed.

**Done looks like:** a `transcribe run` in a terminal shows `chunk 12/38`; a
first run additionally shows the child's download lines; the same run piped to a
file is unchanged; `jobs list` on the queued form shows the chunk counter.

**Tests:** the chunk counter emitted once per chunk with the right total; the
child-stderr forwarding driven through `tests/subproc_engine_stub.py` with a stub
that writes several stderr lines then succeeds; the error path still carrying the
stderr tail; nothing emitted when the sink is absent.

---

## Task 8c — Progress where a user looks for it

**Closes:** the display half of entries 3 and 10.

**Depends on:** Task 8a (the column is written), Task 9 (`commands/ingest.py`),
Task 4 (`tui/screens/jobs.py`).

**Owns:**
- `rytp/commands/ingest.py`
- `rytp/tui/screens/jobs.py`
- `tests/test_tui_jobs.py`
- `tests/test_commands_ingest.py`

**Must not touch:** `rytp/progress.py`, `rytp/jobs/*`.

**What to do.** `jobs list` gains a progress column, blank unless the job is
running. `jobs stats` is unchanged. The TUI queue screen (F7) shows the same
value and refreshes it without blocking — it already polls; reuse that timer
rather than adding a second one. Route every status string through Task 4's
`rytp/tui/text.py` helper; a progress detail can contain anything.

**Done looks like:** with a worker running, `jobs list` shows a moving counter for
the running job and nothing for the others, and F7 shows the same.

**Tests:** the column present and populated for a `running` job with a progress
value, empty for `done`; the TUI screen rendering a bracket-containing progress
detail without crashing.

---

## Task 9 — The CLI says what it means

**Closes:** entries 1, 2, 4, 6, 15.

**Depends on:** Task 1 (`GROUP_SUMMARIES` is a contracts addition).

**Owns:**
- `rytp/cli.py`
- `rytp/commands/__init__.py`
- `rytp/commands/ingest.py` *(until Task 8c takes it; 8c depends on this task)*
- `rytp/commands/search.py`
- `tests/test_cli.py`
- `tests/test_commands_search.py`
- `tests/test_commands_ingest.py` *(until Task 8c takes it; 8c depends on this task)*
- `tests/test_video_reference.py` (new)
- `tests/test_registry.py`

**Must not touch:** `tests/test_catalog_videos.py` (Task 15) — put
`resolve_video_id` tests in the new file. `rytp/commands/catalog.py` (Task 15).

**Entry 1 — a URL is rejected with an unhelpful message.** `fetch-video <url>`
and `ingest <url>` both answer `no video matches '<url>'`. `resolve_video_id`
matches a row id or an `external_id` and nothing else. Registering first works:
`videos add <url>` accepts the URL. Fix both halves:
- **Accept a URL** by also matching `videos.url`. A URL that is registered
  resolves; a URL that is not still fails, but
- **the error says what to do**: `no video matches '<ref>'; register it first
  with: rytp videos add <ref>` when the reference looks like a URL, and a plain
  form otherwise. `resolve_channel` already does exactly this
  (`register it with: rytp channel add <url>`), so follow that shape — one
  spelling of "which video?" and one of how to fix it.

**Entry 2 — the parameter description.** Both commands describe `video` as
"Catalog id or external id of the video", which is accurate and unhelpful.
Reword to say a URL works only once registered and name `videos add`. The
parameter is declared once in `rytp/commands/ingest.py`; change it there.

**Entry 4 — `index build` says "nothing to index" and stops.** Say *why*: no
transcribed videos, or every transcribed video already indexed, with counts, and
name the next step (`transcribe run`). The targets already come from
`videos_needing_index`, so the two cases are distinguishable with one extra
count.

**Entry 6 — `--help` group headings carry no information.** Every group is
introduced as "Commands in the *X* group." Add `GROUP_SUMMARIES` to
`rytp/commands/__init__.py` per Task 1's amendment: a two-or-three-word phrase
saying what the section is *about*. `cli.py` builds the Typer sub-app help as
that phrase followed by a generated comma-separated list of the group's
commands, so the shape of the surface is visible at a glance without drilling
into each group. Write a summary for **every** group, not just the new one — the
current text is equally thin everywhere — and add a consistency assertion that
every group has one.

**Entry 15 — `transcript show --line-length`.** Add a `line_length` parameter
defaulting to `C.TRANSCRIPT_DEFAULT_LINE_LENGTH`, wrapping the transcript to the
chosen width instead of whatever the table decides. Put it on **`transcript
show` only**: `transcript build` writes the durable markdown file, where a
hard wrap bakes a terminal's width into a stored artefact. Say so in the help
text of both so the asymmetry reads as a decision.

**Done looks like:** `rytp fetch-video https://example.invalid/x` on an
unregistered URL prints one line naming `videos add`; on a registered one it
runs. `rytp --help` shows each group described by what it does, with its
commands listed. `rytp index build` on an empty database says why. `rytp
transcript show --video 1 --line-length 60` wraps at 60.

**Tests:** `resolve_video_id` by id, by external id, by URL, and the two error
shapes; the group help containing the phrase and the command list; every group
having a summary; the two `index build` messages; the wrap width honoured and
the default applied.

---

## Task 10 — The TUI argument line stops corrupting Windows paths

**Closes:** entries 21, 24.

**Depends on:** nothing.

**Owns:**
- `rytp/tui/palette.py`
- `tests/test_palette.py`

**Must not touch:** `rytp/tui/app.py` (Tasks 4, 11). Entry 18's placeholder is
Task 11's; this task only guarantees `usage_line` and `PaletteEntry.usage` keep
their current shape so Task 11 can read them.

**Entry 21 — the serious half is Windows paths.** `parse_arguments` uses
`shlex.split` in POSIX mode, where backslash is an escape character, so an
unquoted native path is **silently mangled**:
`D:\Work\rytp\sample_inputs\file.mp4` becomes
`D:Workrytpsample_inputsfile.mp4`. No error. The user then gets "… is not a
file" naming a path they never typed. Quoted, it survives. That is worse than
the spaces case: it is silent, it is wrong rather than refused, and it happens on
the target platform for the most natural thing a user can type.

`BUGS.md` lists three options. **Choose `posix=False`** — `shlex.split(text,
posix=False)` stops treating backslash as an escape, still honours quotes, and
is one argument rather than a hand-rolled splitter. It keeps quoted tokens
working (verified in the report: `"…«имя файла» с пробелами.mp4"` parses to one
token) and it does not need the "trailing positional is the rest of the line"
special case, which would make the grammar position-dependent. Note that
`posix=False` **keeps the quote characters in the token**, so strip a matching
surrounding pair — and strip it in **two** places, because the token is split on
`=` before conversion: on the whole token for a positional, *and* on the
right-hand side after `partition("=")`, or `title="a b"` reaches the handler
with its quotes still attached. Test both explicitly, with `"` and with `'`.

**Entry 24 — accept CLI flag syntax.** The TUI's grammar is `value` and
`name=value`; `--captions` contains no `=` so it is taken as a positional and
`fetch-video` has only one, giving "too many values" — and then, before Task 4,
a crash. A user who has spent the day in the CLI will type the CLI form. Accept
`--name`, `--name=value` and `--no-name` as synonyms for `name=value`. The
parser already normalises `PARAM_ALIASES` by stripping leading dashes, so the
shape is half there; apply the same normalisation to typed input. `--name` on a
boolean means `true`, `--no-name` means `false`, and `--name` on a non-boolean
takes the next token as its value. A `--name` that matches no parameter keeps
today's error, which already prints the usage line.

**Done looks like:** `videos add D:\Work\rytp\sample_inputs\file.mp4` unquoted
reaches the handler with the path intact. `fetch-video 1 --captions` runs the
same command as `fetch-video 1 captions=true`.

**Tests:** the exact `D:\Work\…` string from `BUGS.md` round-tripping; a quoted
Cyrillic filename with spaces; `--flag`, `--no-flag`, `--name=value`,
`--name value`; an unknown `--flag` erroring with the usage line; every existing
`test_palette.py` case still passing, since `posix=False` changes tokenisation
for every command.

---

## Task 11 — The TUI home screen becomes usable

**Closes:** entries 12, 14a, 14b, 14c, 18, 22.

**Depends on:** Task 4 (same file; markup safety lands first), Task 8a (the
progress sink).

**Owns:**
- `rytp/tui/app.py`
- `tests/test_tui_app.py`
- `tests/test_tui_shell.py`

**Must not touch:** `rytp/tui/palette.py` (Task 10), `rytp/tui/text.py` and the
screens (Task 4), `rytp/tui/navigation.py` (Task 16).

**Entry 22 is the structural one; do it first.** `videos.add` is registered
`long_running=False`, so `run_selected` calls its handler inline on Textual's
event loop — and given a URL that handler shells out to yt-dlp for a metadata
request. A network round trip on the UI thread. The flag is not wrong; it
answers a different question. `long_running` classifies whether a command has a
*queued* form and belongs to the worker. `videos.add` is correctly not
queueable — cataloguing is one metadata call — and still blocks for seconds.
`channel.add` is the obvious sibling; it resolves a channel URL.
**Run every handler in a Textual worker**, not just the slow ones, and show that
something is in flight. Install Task 8a's TUI sink for the duration so the
in-flight indicator carries real progress where an emitter exists.

Then the four usability entries, all in the same file:

- **14a — arrows do not reach the list.** `compose` yields `#filter` before
  `#palette`, so initial focus is in the filter where Up/Down move the text
  cursor, and nothing forwards them. Forward Up/Down from the filter to the
  `OptionList` highlight, the way every command palette behaves.
- **14c — Enter in the filter does nothing, invisibly.** `on_input_submitted`
  acts only for `#arguments`. Make Enter in the filter move focus to the
  arguments field when the highlighted command takes parameters, and run the
  command when it takes none. Never silently discard it.
- **14b — two spellings on screen at once.** The palette lists `entry.name`
  (`videos.add`) while `usage_line` renders `videos add`. Show one spelling. Use
  the **spaced** form, which is what both surfaces accept, and label the filter
  as a filter so it does not read as a place to type a command.
- **18 — the arguments placeholder never updates.** `PaletteEntry.usage` already
  carries `videos add <url> [audio=…]`. Set it as the `#arguments` placeholder in
  `on_option_list_option_selected`, which already fires on selection. This is
  also most of the answer to entry 21's "nothing says quoting works": show the
  real shape before the mistake is made.
- **12 — the home screen does not look like it runs anything.** It does: the
  palette runs a short command outright, queues a long-running one that has an
  `enqueue`/`queue` flag, and refuses the eight in `FOREGROUND_ONLY` with a
  reason and a copyable command line. Say so on the screen, in one or two lines:
  type to filter, Enter or Tab to reach the list, select to fill the argument
  field, Enter to run or queue.

**Done looks like:** a user who has never seen the app can, using only the keys
they would try first, filter to `videos add`, reach it with Down, see
`videos add <url> [audio=…]` in the argument field, type a URL, press Enter, and
watch it complete without the interface freezing.

**Tests:** Up/Down from the filter moving the highlight; Enter in the filter
doing the right thing for a parameterised and a parameterless command; the
placeholder changing on selection; one spelling on screen; the handler running
off the event loop (assert the worker is used, not that timing is fast); the
home text present.

---

## Task 12 — The TUI explains `caption`, `timed` and `aligned`

**Closes:** entry 19.

**Depends on:** Task 4 (`screens/search.py` is Task 4's file first).

**Owns:**
- `rytp/tui/screens/help.py`
- `rytp/tui/screens/search.py`
- `tests/test_tui_glossary.py` (new)
- `tests/test_tui_search.py`

**Must not touch:** `rytp/tui/app.py`, `rytp/tui/navigation.py`.

**The gap.** The tier vocabulary is load-bearing and unexplained. The search
screen has a "Cuttable only" filter on Ctrl+T and prints "· cuttable only" in its
status line, with nothing anywhere saying what makes a word cuttable or why some
are not.

**What to write.** A short glossary on F1, which is the obvious home and already
shows every binding:

- `caption` — downloaded subtitles. Searchable, no end times, never cuttable.
- `timed` — the transcriber's own word timestamps. Good text, boundaries not
  trustworthy: measured on real data, 78.7 % of Whisper's word gaps are exactly
  zero.
- `aligned` — forced alignment plus a snap to a measured energy minimum and zero
  crossing. The only tier that can be cut.

Plus a one-line hint wherever a tier or the word "cuttable" appears — the search
screen's filter and status line. This becomes more important once `--allow-timed`
lands (Task 6): with the flag, a user has to understand the difference to make
the choice it offers, so mention the flag by name.

**Done looks like:** F1 shows the three tiers with one sentence each; the search
screen's cuttable filter carries a hint saying what cuttable means.

**Tests:** all three tier names and the word "cuttable" present in the help
screen's rendered text; the search screen's hint present; no bare
`Static.update` reintroduced (Task 4's sweep test covers `search.py`, so do not
break it).

---

## Task 13 — The cut-list screen's keys fit, and trim finely

**Closes:** entries 29, 30.

**Depends on:** Task 4 (same file), Task 2a (the nudge constants).

**Owns:**
- `rytp/tui/screens/cutlist.py`
- `rytp/tui/cutlist_view.py`
- `tests/test_tui_cutlist.py`

**Must not touch:** `rytp/audio/energy.py` — import it, do not change it.
`rytp/constants.py` — Task 2a set `TUI_CUTLIST_NUDGE_MS = 10` and added
`TUI_CUTLIST_SHIFT_NUDGE_MS = 100`.

**Entry 29 must be resolved before entry 30 adds a key.** F8 binds eleven keys
today and the footer runs off the end of the line, recoverable only because
PowerShell keeps scrollback. It is the screen with the most bindings by some
margin, so whatever is chosen should be checked against it rather than against
the lighter screens. Keep the footer to the handful used constantly and move the
full list behind a key — the help screen already exists and
`navigation.ScreenEntry.children` already makes this screen's bindings visible to
it, so a `?` that shows the full set and a trimmed footer is the smallest honest
change. Assert the rendered footer fits a normal terminal width.

**Entry 30 — finer trimming and a snap key.** 40 ms is too coarse: at
conversational speed it is a good fraction of a phoneme, so trimming a word edge
overshoots. The scheme is **plain 10 ms, Shift 100 ms, plus a snap key** —
Shift as *coarser*, because when a fragment is wrong it is usually wrong by a
syllable, not by a millisecond.

1 ms was requested and is declined, with the reason recorded: 1 ms is 16 samples
at 16 kHz and inaudible as a timing change, and reaching it by keyboard takes
~40 presses to cover one of today's steps. What a 1 ms nudge is actually chasing
is a clicking seam, and a click comes from an amplitude discontinuity, not from
timing precision. The project already solves that during alignment — boundaries
snap to the local energy minimum and the nearest zero crossing — so **expose it
as a key**. `rytp/audio/energy.py` already holds the machinery; one press does
what forty 1 ms nudges do, and does it better. If 1 ms is still wanted later, a
Ctrl level can be added, but the snap key is the one that earns its place.

**Done looks like:** the footer fits an 80-column terminal; `[` moves a start by
10 ms and `Shift+[` by 100 ms; the snap key moves the highlighted boundary to the
nearest measured energy minimum and zero crossing and says by how much it moved.

**Tests:** the footer's rendered length under a normal width; each nudge step;
the snap key against a synthesised waveform from `tests/synth_audio.py` with a
known minimum; the snap being a no-op with a reported reason when no better
boundary exists nearby.

---

## Task 14 — One duration format per purpose, and the tier in the source list

**Closes:** entry 31, and the render-side half of D1.

**Depends on:** Task 6 (the cut list must record the tier first), Task 15
(`commands/catalog.py`).

**Owns:**
- `rytp/timefmt.py` (new)
- `rytp/render/report.py`
- `rytp/commands/catalog.py`
- `rytp/commands/assemble.py`
- `tests/test_timefmt.py` (new)
- `tests/test_render_report.py`
- `tests/test_commands_assemble.py` *(inherited from Task 6, which lands first)*
- `tests/test_catalog_videos.py` *(inherited from Task 15, which lands first)*

**Must not touch:** `rytp/assemble/*` (Task 6) — only the command module's
display helpers.

**The defect.** `assemble plan` reports a cut list as `0:02.340`; `render run`
reports the same kind of value as `0:00:02.700`. Same quantity, different shape,
and the only visual difference is an extra colon. There are at least four
formatters: `format_duration` (`H:MM:SS`, catalog), `format_timecode`
(`H:MM:SS.mmm`, report), `format_clock` (`M:SS` or `H:MM:SS`, report), and
whatever `assemble` uses for `M:SS.mmm`.

**Several formatters is fine — that is not the bug.** A paste-ready description
wants `M:SS`; a seek target wants milliseconds. The bug is that two commands in
one pipeline print *the same measurement* differently, so a reader comparing a
plan against its render has to count colons.

**How to implement it.** Collect every formatter into `rytp/timefmt.py`, **named
for its purpose, not its shape**: something like `format_seek` (a target you type
into a player, `H:MM:SS.mmm`), `format_spoken` (a description, `M:SS`),
`format_length` (a duration in a listing, `H:MM:SS`). Re-export or delegate from
the existing call sites so nothing outside the owned files changes. Then make the
plan and the render agree: both a plan's fragment offsets and a render's report
offsets are seek targets, so both use `format_seek`.

Add the consistency test entry 31 asks for, in the spirit of the flag-vocabulary
suite: **one formatter per purpose, and no module defines a private one.** A
source-level scan for a locally-defined `f"{...:02d}:{...:02d}"` pattern outside
`rytp/timefmt.py` is enough and is what stops the fifth formatter appearing.

**Also — D1's render half.** The cut list now records each fragment's tier
(Task 6). Mark it in the render's source list, so a degraded cut is visible in
the output rather than only at plan time. D1 says this matters *more*, not less,
once quality is not being checked at all.

**Not a defect, recorded so it is not chased:** the plan says `2.700` and the
container measures `2.750`. That is frame and AAC-frame padding.

**Done looks like:** the same measurement prints identically in `assemble plan`
and in `render run`'s report; a render whose cut list contains a `timed` fragment
marks it in the source list.

**Tests:** each formatter's boundaries (zero, sub-second, exactly an hour,
negative clamped); the plan and the report producing the same string for the same
millisecond value; the no-private-formatter scan; the tier appearing in a
rendered report for a `timed` fragment and not misreported for an `aligned` one.

---

## Task 15 — `videos list --long`

**Closes:** entry 25.

**Depends on:** Task 2a (the tick/cross constants).

**Owns:**
- `rytp/db/queries.py`
- `rytp/commands/catalog.py` *(until Task 14 takes it; 14 depends on this task)*
- `tests/test_catalog_videos.py`
- `tests/test_queries.py`
- `tests/test_consistency_flags.py`

**Must not touch:** `rytp/jobs/readiness.py` and each part's readiness module —
**call** the predicates, do not reimplement them. A second, disagreeing
definition of "done" is the thing this entry exists to avoid.

**What to build.** Per video: whether it is downloaded, transcribed, timed,
aligned, and which engines produced that.

- **`--long`, not `--short`** (owner's revision). Today's columns stay the
  default and the detailed view is opt-in, so nothing existing changes shape and
  no script parsing the listing breaks. `--long` must still clear the
  flag-vocabulary suite, which asserts one spelling with one meaning across every
  command — check it, and add the pair to `BANNED_SPELLINGS` if `--short` should
  be reserved against.
- **Columns named for the thing, ticks in the cells** (owner's revision). One
  column per asset role and per pipeline stage — `audio`, `captions`, `videos`,
  then `transcribed`, `aligned`, `indexed`, `diarized` — each cell a tick or a
  cross. One vocabulary covers both halves: a tick means the file is present, or
  the process completed. That scans down a column as easily as across a row,
  which is what makes "nothing forgotten" work.
- **Cell renderer, decided:** `0` renders as `C.CELL_CROSS`, `1` as `C.CELL_TICK`,
  `2` and above as the number itself. So a single video rendition reads as a tick
  like every other satisfied thing, and only genuine plurality shows a digit. One
  rule for every column, asset counts and stage flags alike.
- **`audio` is a tick, never a count.** `SINGLETON_ASSET_ROLES` is
  `{audio, captions, container}` and `cache/wav/{video_id}.wav` is the single
  audio timeline every word timestamp refers to. A source with several language
  tracks is not representable, and changing that is a contracts change — it is in
  `TODO.md` and stays there. Only `videos` needs a count.
- **Stage state comes from readiness.** Nine of the ten job kinds target a
  video; render one column per relevant kind from its
  `readiness(db, target_id)` predicate. No new logic.
- **Tier and engine.** `words.source` answers "can I cut this yet" and is the
  column worth showing most. A video can hold more than one tier at once —
  captions ingested, then a transcriber run — so the column shows the **best**
  tier present and says so in the header's help. `words.engine` already records
  the chain (`whisper+energy`), so `DISTINCT engine` answers "what was used" and
  the alignment question at the same time. Show `align_scale` too now that it
  exists: it is what tells the owner which videos still need re-aligning after
  entry 36.
- **Fold it into `q.list_videos` as a join**, one statement, rather than a query
  per video. With ~1.6K videos and a default limit either works, but the join is
  the better shape.
- **`--longer` is not built.** The owner asked for renditions with their
  qualities, the audio track and the engine chain. A third ever-wider table is
  hard to read at 1.6K rows, and per-video detail is a different shape of
  question from listing. Report this as a recommendation for a `videos show <id>`
  command in a later batch; do not add a third flag here.

**Done looks like:** `rytp videos list` prints exactly what it prints today.
`rytp videos list --long` adds the asset and stage columns with ticks, the best
tier, the engine chain and the align scale, in one SQL statement.

**Tests:** the default listing byte-identical to today's; each cell renderer case
(0, 1, 2, 5); a video with two renditions showing `2` under `videos` and a tick
under `audio`; a video with both caption and timed words showing the better tier;
stage columns agreeing with the readiness predicates (assert against the
predicate, not against a duplicated rule); the flag-vocabulary suite passing.

---

## Task 16 — A Videos screen in the TUI

**Closes:** entry 23.

**Depends on:** Task 15 (share its state query — do not write a second one),
Task 12 (the glossary this screen most needs), Task 13 (the footer lesson).

**Owns:**
- `rytp/tui/screens/videos.py` (new)
- `rytp/tui/navigation.py`
- `tests/test_tui_videos.py` (new)
- `tests/test_tui_navigation.py`

**Must not touch:** `rytp/tui/app.py` — a screen is "one row in
`navigation.SCREENS` and no code there", and that property is worth preserving.
`rytp/db/queries.py` and `rytp/commands/catalog.py` (Task 15) — call them.

**Why it is worth building.** This is the missing overview. Every other screen
answers a question about one thing — a transcript, a search, a cut list — and
nothing shows where each video *is* on the path from registered to cuttable.

**Most of what it needs exists.** Stage state comes free from the readiness
predicates, which are statements about the world derived from database and
filesystem state. The actions are registered commands — `ingest`,
`transcribe run --enqueue`, `transcribe align --enqueue`, `speakers enqueue`,
`index build` — and a shortcut must call the same handler the palette calls, not
a parallel path.

**Three decisions, made here.**

1. **Shortcuts enqueue, they do not run.** Consistent with the palette's
   behaviour for long-running commands, and it keeps the UI responsive — which
   entry 22 shows matters.
2. **Show the tier per video.** It is the single most useful column for "is this
   cuttable yet", and it is where entry 19's glossary most needs to appear. Carry
   a one-line hint, and link to F1.
3. **Distinguish "done" from "cannot be redone automatically".** `align` and
   `diarize` are `reopenable=False`, so a satisfied stage will not re-fire under
   `reconcile`. If the screen shows both as a plain tick, re-running a step from
   here will look broken. Use a distinct marker and explain it in the footer or
   the help.

**On F5.** It is unbound at the app level and sits naturally before Search — but
it is **not simply free**. `rytp/tui/navigation.py`'s docstring records that it is
skipped deliberately because the speakers mapper binds `f5` locally
(`toggle_suggestions`), and a global plus a local `f5` would collide. Run the
binding-collision assertions in `tests/test_tui_navigation.py` before committing
to the key. It is likely fine because the mapper's binding is screen-scoped, but
verify rather than assume; if it does collide, take the next free function key
and say which.

**Done looks like:** F5 (or the chosen key) opens a filterable list of videos
with one column per pipeline stage, the best tier, and shortcuts that enqueue the
missing step for the highlighted video. The screen appears in the home strip and
the help screen with no edit to `rytp/tui/app.py`.

**Tests:** the navigation table gaining one row and the app deriving its binding
from it; no key collision; the stage columns matching the readiness predicates;
a shortcut enqueueing rather than running; the `reopenable=False` marker.

---

## 4. Deliberately not in this batch

Every `BUGS.md` entry is either assigned above or listed here.

| Entry | Why not |
|---|---|
| **8** Settings have no CLI surface | **Already done** in the working tree, uncommitted: `rytp/commands/settings.py` gives `list`/`get`/`set`/`unset` on both surfaces. The first task to run should confirm the suite passes with it, not reimplement it. `docs/superpowers/plans/2026-09-24-settings-integration.md` holds the deliberately-deferred surfacing work. |
| **9** `sentencepiece` has no cp314 Windows wheel | Not a rytp defect — an ecosystem gap. The escape hatch it needs is entry 8 (done) plus Task 7's `docs/setup.md`. |
| **17** No aligner can run on the target machine | A record of the environment, not a defect. Unblocked by entry 8 plus Task 7's setup document; nothing to change in the code. |
| **32** The out-of-process engine reloads its model once per chunk | Excluded. Fixing it means either batching a whole video's chunks into one request — which changes the `Aligner.align` signature and is therefore a **contracts §6 change needing its own design pass** — or a persistent child with a lifecycle the one-shot design does not have. It would also collide with tasks 3, 5, 7 and 8b on four files. `BUGS.md` itself notes it must land alongside `TODO.md`'s chunking work. The sharper half of the entry — a long-running job silently changing behaviour when its source is edited mid-run — is a separate observation worth its own entry; it is not addressed here. |
| **D2** Should captions be promotable to cuttable by alignment alone? | Excluded: `BUGS.md` states the question is empirical, not architectural — how verbatim are this archive's captions. Answering it means measuring with `transcribe compare`, which is a study, not a fix. Revisit after Task 3 makes alignment trustworthy enough to measure against. |
| `README.md` is stale (noted inside entry 35) | Task 7 writes `docs/setup.md` instead. Rewriting the README to describe the post-rewrite tree is a documentation project, not a bug fix; `DESIGN.md` is stale in the same way. |

Also out of scope, stated explicitly because they are adjacent: the four items in
`TODO.md` (multiple audio tracks, verifying a render by re-transcribing it, the
CTC short-chunk failure, and the chunking change it needs), and the three "Still
open" items in `docs/superpowers/2026-09-21-review-findings.md` (`transcribe.run
--enqueue` on a video that already has words, `jobs.retry` by id, and lazy
command registration).

---

## 5. Post-batch operator actions

The batch changes code; two things in the owner's live data need a human.

1. **Re-align everything.** Entry 36 means every `aligned` word produced so far
   has a fabricated boundary, and entry 28's negation hack means their scores are
   sign-flipped with no way to identify which. Task 2b marks them
   `align_scale = 'unknown'` and Task 6 excludes them from assembly with a message
   that says so. The fix is `transcribe align` over every previously aligned
   video, after tasks 3, 5 and 6 have landed. Nothing in the batch attempts a row
   repair, and nothing should.
2. **Expect the re-alignment to be slow.** Excluded entry 32 means the engine
   reloads its model once per chunk — 38 times for the first real video. That
   cost is paid once here. If it proves intolerable at archive scale, promote
   entry 32 ahead of the next batch.

Until the re-alignment is done, `assemble plan --allow-timed` is the working
path: it is an override, the tier is recorded on every fragment, and Task 14
marks it in the render's source list.
