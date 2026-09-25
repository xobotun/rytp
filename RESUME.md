# Resume — QA fix batch, paused 2026-09-24 late

Nothing is committed. The plan is
`docs/superpowers/plans/2026-09-25-qa-fix-batch.md` (19 tasks, 5 waves).
`BUGS.md` has 37 entries + a Fix-first list; `TODO.md` holds deferred work.

## Done and verified (waves A, B, C)

T1 contracts · T2a constants · T2b migrations 13-15 · T3 wav2vec2 CTC grouping
(entry 36) · T4 TUI markup safety (entry 20) · T5 align scale + MFA binary
check · T7 engine-seam probe (entries 5,7,11,16,33,34,35) · T8a progress seam ·
T9 CLI messages/group summaries/--line-length · T10 argument grammar ·
T12 tier glossary · T13 cut-list keys · T15 `videos list --long`

Plus, by hand: group help drops the repeated group prefix and colours leaf
names cyan (`rytp/cli.py` `_group_help`), summary escaped because Typer 0.27
has `rich_markup_mode="rich"`. Contracts DDL gained `words_timed`.

## STOPPED MID-EDIT — check these first

Wave D was killed part-way. **Their files may hold partial edits.** Review
the diff for each before resuming, and consider reverting rather than
continuing from an unknown state:

- T6 assembly eligibility (entries 26, 28, D1) — had only been reading, no
  edits reported yet
- T8b engine-seam progress — was mid-rewrite of `run_child` to stream stderr
- T8c progress display — had edits, was about to run tests
- T11 TUI home — had only been reading
- T16 Videos screen — had only been reading. **Its one finding worth keeping:
  F5 is already bound by `SpeakerMapperScreen`, so an app-level F5 would be
  shadowed and fail `test_no_app_level_key_is_shadowed_by_a_screen`. It chose
  F9.**

## Not started

- **Wave E: T14** (one duration format per purpose, entry 31 + D1's render half)
- **New cleanup task** — seven orphans no plan task owns:
  1. `rytp/transcribe/engines/{whisper,gigaam}.py` — missing `score_scale`/
     `notes`, causing **2 live mypy errors right now**
  2. `rytp/diarize/none.py` — `NullDiarizer` blocks adding the three members to
     the `Diarizer` protocol that contracts §6 requires
  3. `rytp/diarize/pipeline.py` — device never surfaced
  4. `rytp/jobs/__init__.py` — `_run_align` discards its outcome, so no note
     channel on the queued path
  5. `rytp/tui/app.py` + `navigation.py` — footer is ~156 cols at 80x24,
     ~93-105 of it app-level, so every screen overflows
  6. `rytp/tui/palette.py` — `GROUP_SUMMARIES` never reaches the TUI surface,
     though contracts §5 says both surfaces read it
  7. `rytp/tui/screens/transcript.py` — see **entry 37**: split transcript rows
     at word boundaries instead of wrapping, which dissolves the DataTable
     `height=1` clipping rather than patching it
- Also pending: add `help.py` to Task 4's AST sweep in `tests/test_tui_markup.py`;
  the plan's Task 10 text still says `posix=False`, which was overridden.

## Final verification still owed

One uncontended `python -m pytest` + `ruff check rytp tests` + `mypy rytp`,
plus a diff review for anyone who edited outside their owned files or
weakened a consistency test to go green. Neither shows up in a green run.

## Operator actions for the owner, once the batch lands

Existing aligned words were backfilled `align_scale='unknown'` and are
**excluded from assembly by design** — their boundaries were fabricated by
entry 36. Re-run `transcribe align` on every aligned video. The negation hack
(`UPDATE words SET align_score = -align_score`) must not be re-applied; it
inverted the quality ordering (entry 28).

## Environment

venv `~/.cache/rytp-dev-venv`; always `RYTP_TEST_TMP=/private/tmp/...` outside
the repo (SMB makes the suite 10x slower otherwise); never a second `-q`;
isolated `MYPY_CACHE_DIR` per agent (concurrent runs corrupted the shared one).

## Working tree at pause

```
 M rytp/commands/ingest.py
 M rytp/transcribe/pipeline.py
 M rytp/transcribe/subproc.py
 M tests/test_cli.py
 M tests/test_commands_ingest.py
 M tests/test_tui_jobs.py
```
