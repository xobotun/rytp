# Part 8 — Integration and Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the seven independently-planned parts into one application: a TUI with a home view, a reconciled binding scheme, help, the two screens design §10 promises and nobody built (cut-list editing, queue and job progress), a way to start long-running work from the TUI without turning it into a second worker, an end-to-end walk of milestone M1, and a cross-part consistency suite that turns every defect the reviews found by hand into a test that fails loudly.

**Architecture:** Nothing here owns a domain. Two kinds of work: **integration** — a data-driven navigation map (`rytp/tui/navigation.py`) from which `RytpApp.BINDINGS`, the home view and the help screen are all generated, so a new screen cannot be added without appearing in all three; and **coherence** — four test modules that read the registries (`COMMANDS`, `JOB_KINDS`, `HEALTH_CHECKS`, the schema) and assert the invariants that hold *between* parts. Both new screens follow Part 7's shape, which worked: all logic in a headless view class with plain tests, the Textual screen a shell of one-line actions.

**Tech Stack:** Python 3.11+, Textual, stdlib `sqlite3` and `tomllib`, pytest. No new dependencies. Nothing in this part imports a heavy optional extra.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (§10 Surfaces, §8 Assembly, §5 the queue, §11 M1) and the binding `docs/superpowers/specs/2026-09-21-rytp-contracts.md` (§2 ownership, §5 registry and job handlers). Review context: `docs/superpowers/2026-09-21-review-findings.md`.

## Global Constraints

Copied from contracts §1 and §8. Every task's requirements implicitly include this section.

- Python >= 3.11. Target 3.11 syntax; `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. `pathlib` everywhere; `subprocess` only with list arguments.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` runs on `rytp/`.
- Heavy dependencies are optional extras, imported lazily inside functions, never at module import time.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`. Russian test strings are wanted.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- Milliseconds, integers, everywhere. Database timestamps are `datetime.now(UTC).isoformat()` strings.
- Domain errors subclass `RytpError`. The CLI prints one line to stderr and exits 1 — never a traceback for an expected failure.
- Every stage takes an already-open `Database`; nothing opens its own connection. Multi-statement writes use `db.transaction()`.
- Tests: **no network, ever; no model download; no media player spawned.** The suite must pass with no optional extra installed.
- **Test invocation:** `python -m pytest <args>` from the repository root with the project virtualenv activated. No absolute interpreter paths, no `/tmp`.
- One commit per task, conventional-commit prefix, present tense.

---

## When this part runs, and what a failing test means here

**Part 8 executes after Parts 1–7 are code, not plans.** That single fact decides
the shape of every task below.

Tasks 1–4 are a test suite with no production code of its own. They are still
TDD in the only sense that matters: each test is written to state an invariant,
run, and watched. But the thing under test already exists, so a red test is a
**bug in an already-written module**, and the fix belongs in that module, in the
same task, named in the commit message — never worked around locally and never
by weakening the assertion.

Some defects die before Part 8 ever runs, because a part's own suite catches
them. Some survive, because no single part could see them. Only the second group
is what Part 8 is for. Both are pinned, so neither can come back:

| Defect | Where it is | Dies at | The fix |
|---|---|---|---|
| `Param(..., short="c")`, `short="n"`, `short="p"`, `short="n"` in `rytp/commands/ingest.py` | Part 2 | Part 2's own import — `register` requires `-c` | add the leading dash |
| Part 7's assertion that ingest enqueues "`download`, `captions` and `extract_wav` and nothing else" | Part 7 | when Part 3 lands `caption_words` and `fingerprint` | assert `"diarize" not in chain` instead of pinning the chain's contents |
| `--video` is `str` in `videos.remove` and `SPEAKER_PARAMS`, `int` in `search.words`, `search.export`, `index.drop`, `transcript.build`, `transcript.show` | Parts 1, 4 | **survives** | one type. `str`, because `SPEAKER_PARAMS` and `resolve_video_id` accept an external id too |
| The same concept spelled `--video` (Parts 1, 4) and `--video-id` (Parts 2, 3, 7) | Parts 1–4, 7 | **survives** | the one job here that is not a one-liner: about twelve registrations rename `video_id` to `video`, each handler renames its keyword to match, and the ones that took an `int` now take a `str` and call `resolve_video_id`. Task 1's handler-signature test catches a half-done rename |
| The same concept spelled `--enqueue` (Parts 3, 4) and `--queue` (Part 6) | Parts 3, 4, 6 | **survives** | rename `render.run`'s `queue` to `enqueue` |
| `-s` means `--search` on `videos.list` and `--speaker` on `search.words` / `search.export` | Parts 1, 4 | **survives** | drop `-s` from `videos.list`; `--speaker` is the one worth a letter |
| Part 3's job thunks are `-> None` and return nothing, so the re-transcribe speaker warning reaches the CLI and vanishes on the worker | Part 3 | **survives** | `_run_transcribe` returns the warning string that `_run_handler` already computes |
| `ingest --transcribe` stamps only `{"aligner": …}`, so the `transcribe` job always falls back to `C.DEFAULT_TRANSCRIBER` (`"whisper"`, an optional extra) | Part 2 | **survives** | add `settings.default_transcriber`, resolved once before the loop, exactly as `default_aligner` is |
| `speaker` defaults to `None` in `SPEAKER_PARAMS` and to `""` in `assemble.plan` / `assemble.suggest` | Parts 1, 5 | **survives** | `None`, so "not asked for" is distinguishable from "asked for nothing" |
| Promoting a video to tier 2 deletes its caption words, which can make `caption_words` READY again, so the next `reconcile` re-adds a caption tier under an aligned transcript | Parts 2, 3 | **survives** | `caption_words_readiness` answers SATISFIED when the video has *any* words, not only caption ones |

Tasks 5–13 are ordinary TDD: write the failing test, watch it fail, implement,
watch it pass, commit.

## What Part 8 consumes

Every symbol below is owned by another part and used verbatim. Part 8 defines no
schema, registers no job kind, and adds no command outside `tui`.

| Symbol | Owner |
|---|---|
| `rytp.commands.{COMMANDS, Command, CommandResult, Param, REQUIRED, PARAM_ALIASES, PARAM_TYPES, SPEAKER_PARAMS, register, resolve, resolve_video_id, cli_path, leaf_name}` | 1 |
| `rytp.commands.{HealthCheck, HealthResult, HEALTH_CHECKS, register_check}` | 1 |
| `rytp.commands.{SpeakerFilter, resolve_speaker_filter}` | 1 |
| `rytp.cli.build_app` | 1 |
| `rytp.config.{paths, ensure_dir}`, `rytp.constants`, `rytp.models.{RytpError, InvalidInputError, NotFoundError, Fragment, normalize_text, stem_text, utc_now_iso}` | 1 |
| `rytp.db.{Database}`, `rytp.db.schema.MIGRATIONS`, `rytp.db.queries.{get_setting, set_setting}` | 1 |
| `rytp.tui.palette.{PaletteEntry, palette_entries, match_entries, usage_line, parse_arguments, cli_invocation}` | 1 |
| `rytp.jobs.{JOB_KINDS, JOB_HANDLERS, JobKind, Readiness, register_job_kind, resolve_job_kind, kinds_for_pool}` | 2 |
| `rytp.jobs.queue.{Job, enqueue, list_jobs, stats, JobStats, retry, get_job, pause, resume, is_paused, reconcile}` | 2 |
| `rytp.jobs.worker.{run_worker, WorkerOptions}` | 2 |
| `rytp.audio.extract.{wav_path, _run_ffmpeg, _ffmpeg_binary}` | 2 |
| `tests.fakes.{make_video, touch, FakeYtDlpRunner, temp_job_kind}` | 2 |
| `tests.fake_engines.{FakeTranscriber, FakeAligner, registered}` | 3 |
| `rytp.transcribe.registry.{TRANSCRIBERS, ALIGNERS}` | 3 |
| `rytp.index.search` (through the `search.words` command only) | 4 |
| `rytp.tui.screens.search.SearchScreen`, `rytp.tui.screens.transcript.TranscriptVideosScreen` | 4 |
| `rytp.assemble.cutlist.{CutList, CutlistParams, Slot, Alternative, Substitution, CutlistError, load_cutlist, read_cutlist, write_cutlist, dumps_cutlist, cutlist_path, validate_name}` | 5 |
| `rytp.render.ffmpeg.{Tools, CompletedRun}` | 6 |
| `rytp.tui.screens.speakers.SpeakerVideosScreen` | 7 |

## Files

**Created by this plan**

| File | Responsibility |
|---|---|
| `rytp/tui/navigation.py` | The screen map: one row per screen, its key, title and factory. Textual-free. The single source `BINDINGS`, the home strip and the help screen are all generated from. |
| `rytp/tui/enqueue.py` | Turning a `long_running` command into queued work instead of a command line, and classifying the ones that genuinely cannot be queued. Textual-free. |
| `rytp/tui/jobs_view.py` | Headless queue view: filters, rows, stats line, retry and cancel. No Textual. |
| `rytp/tui/cutlist_view.py` | Headless cut-list editing: slots, alternatives, swapping, nudging, saving. No Textual. |
| `rytp/tui/screens/help.py` | The help screen. A table of `navigation.help_rows()`. |
| `rytp/tui/screens/jobs.py` | Thin Textual shell over `JobsView`. |
| `rytp/tui/screens/cutlist.py` | `CutlistPickerScreen` and `CutlistScreen` — thin shells over `CutlistView`. |
| `tests/consistency.py` | Not a test: the loader and helpers the four consistency modules share. |
| `tests/test_consistency_surfaces.py` | Commands, both surfaces, handlers, health checks. |
| `tests/test_consistency_flags.py` | The flag vocabulary. |
| `tests/test_consistency_jobs.py` | Job kinds, handlers, predicates, producers. |
| `tests/test_consistency_schema.py` | Schema, fixtures, import hygiene, the declared `Consumes`. |
| `tests/test_tui_navigation.py`, `tests/test_tui_shell.py`, `tests/test_tui_enqueue.py`, `tests/test_tui_jobs.py`, `tests/test_tui_cutlist.py` | One per task. |
| `tests/test_m1_end_to_end.py` | The M1 walk. |

**Modified by this plan**

- `rytp/tui/app.py` — Part 1 created it and Parts 4 and 7 each appended a binding and an action. Contracts §2 gives the *app shell* to Part 8, so this task takes the file over: bindings become derived, the home view gains a navigation strip, and the three hand-written actions stay as one-line wrappers so Parts 4 and 7 keep the entry points they published.
- `rytp/constants.py` — one appended section (Part 8's, at the very end).

**Three modules live directly under `rytp/tui/` rather than in `screens/`**, because
contracts §2 names `tui/ app.py palette.py screens/` and `palette.py` is already
the precedent: the logic is Textual-free so it can be unit-tested without a
terminal, and the thing in `screens/` is the shell. Part 7 proved the split is
worth it — `MappingSession` carries 22 plain tests and `SpeakerMapperScreen` is
six one-line actions.

## The binding scheme, reconciled

Three parts chose keys without seeing each other's. Reconstructed from the code:

| Key | Scope | Who chose it |
|---|---|---|
| `ctrl+q` | app — quit | 1 |
| `f2` | app — focus the palette filter | 1 |
| `f3` | app — speakers | 7 |
| `f4` | app — search | 4 |
| `f6` | app — transcripts | 4 |
| `f5` | **screen-local**, speaker mapper — toggle suggestions | 7 |
| `escape` | screen-local, every screen — back | 4, 7 |
| `enter`, `ctrl+n`, `ctrl+u`, `ctrl+down`, `ctrl+up` | screen-local, speaker mapper | 7 |
| `ctrl+p`, `ctrl+t` | screen-local, search | 4 |

There is **no collision** to resolve, and that is not luck: Part 4 skipped `f5`
on purpose and said so ("`f6` is chosen to leave `f5` free: Part 7's mapper
screen binds it"). Part 8 keeps every key exactly where it is and fills the gaps:

| Key | Scope | Added by Part 8 |
|---|---|---|
| `f1` | app — help | the one key every terminal user tries first |
| `f7` | app — queue and job progress | |
| `f8` | app — cut lists | |
| `f5` | **left unbound at app level, permanently** | it belongs to the mapper |

`f9` and above stay free. The reservation of `f5` is not written down as a list —
a hand-maintained reserved-key list is precisely the kind of thing that goes
stale, and this plan exists because one did. Task 5 derives it: the test imports
every screen class named in the navigation map, reads its `BINDINGS`, and fails
if an app-level key appears in any of them. Bind `f5` at app level and the test
names the mapper.

**The rules, stated once so a later screen can follow them:**

1. **Home is the palette.** Part 1's `RytpApp` root *is* the home view — the
   command palette, an argument line, a result table. Part 8 adds a navigation
   strip above it naming every screen and its key. Making home a separate pushed
   screen would move `#palette` off the app root and break all seven of Part 1's
   app tests for no gain.
2. **Home is the hub.** Every screen is pushed from home and `escape` pops back
   to it. No screen navigates sideways to another.
3. **App-level keys are function keys and `ctrl+q`.** Nothing else, because the
   palette filter `Input` has focus on the home view and must keep receiving
   typed text.
4. **A screen whose default focus is an `Input` binds no bare letter** — Part 7's
   rule, kept. The cut-list screen has no `Input` focused by default, so bare
   letters are legal there and Task 11 uses them; the navigation test scopes the
   no-bare-letter rule with an explicit opt-out rather than applying it blindly.

## Long-running work, and why the TUI does not run it

Design §10: "Long-running work is a CLI worker process… The worker itself stays a
separate process." Part 1 honours that by refusing to run a `long_running`
command inline and printing the equivalent command line instead. That is correct
and stays correct — but it means the TUI cannot start an ingest, which review
finding 3.3 left open as a scope decision.

**The decision: the TUI enqueues and observes. It never executes.** Concretely,
for a `long_running` command whose own parameters include an enqueue flag —
`transcribe.run --enqueue`, `index.build --enqueue`, `render.run --queue` — the
TUI runs the handler *with that flag forced true*. That is not running the work:
the handler writes one row to `jobs` and returns in microseconds. The work is
then done by a worker process the owner starts himself, and `f7` watches it.

For the rest there is no queued form and inventing one is out of scope, so they
keep Part 1's behaviour — the copy-pasteable command line — with a one-line hint
saying why. `speakers.diarize` and `speakers.embed` point at `speakers enqueue`,
which is a registered command and already in the palette.

The classification is a table, and a table goes stale — so Task 7's test derives
its domain from the registry: every `long_running` command must have either an
enqueue flag or an entry in the table, and a new one that has neither fails the
suite by name.

---

## Tasks

### Task 1: The consistency harness, and surface parity

Design §10: "Each operation is defined once — name, arguments, result shape — and
both the CLI and the TUI are generated from that definition. Alignment between
the two then holds by construction rather than by discipline." *By
construction* is a claim, and nobody has ever checked it across all seven parts
at once. This task checks it.

It also fixes the one thing that makes every later consistency test possible: a
registry is only complete once every module that registers into it has been
imported. `rytp/commands/__init__.py` imports its siblings at the bottom and
`rytp/jobs/__init__.py` ends with every part's registration block, so importing
those two modules is enough — but if a part forgets its one-line import, the
registry is quietly short and every test below passes vacuously. `load_registries`
asserts a floor, so a forgotten line fails here instead of nowhere.

The highest-value assertion in the file is the last one: that each handler's
keyword arguments cover the `Param`s declared beside it. A `Param("video")` in
front of a handler that takes `video_id` is a `TypeError` at run time, in one
command, discovered by whoever runs that command. This finds all of them at once.

**Files:**
- Create: `tests/consistency.py`
- Test: `tests/test_consistency_surfaces.py`

**Interfaces:**
- Consumes: `rytp.commands.{COMMANDS, Command, Param, REQUIRED, PARAM_TYPES, HEALTH_CHECKS, HealthResult, cli_path}`, `rytp.cli.build_app`, `rytp.tui.palette.palette_entries`, `rytp.jobs.JOB_KINDS`, `rytp.db.Database`.
- Produces: `tests/consistency.py` — `load_registries() -> tuple[dict[str, Command], dict[str, JobKind]]`, `MIN_COMMANDS`, `MIN_JOB_KINDS`, `iter_params(commands) -> Iterator[tuple[Command, Param]]`, `source_files(root) -> list[Path]`, `cli_command_paths() -> set[tuple[str, ...]]`, `DELETION_OWNERS: dict[str, str]`.

- [ ] **Step 1: Write the shared harness**

Create `tests/consistency.py`. It is a plain module, not a conftest: the four
consistency test modules import it, and nothing else should pay for it.

```python
"""Shared loading and scanning for the cross-part consistency suite.

Part 8 owns no domain. What it owns is the set of invariants that hold
*between* the other seven parts, and every one of them starts from a fully
populated registry. Importing ``rytp.commands`` runs the sibling-import block
at the bottom of that module, which is what registers every command; importing
``rytp.jobs`` runs each part's registration block, which is what registers
every job kind. If either is short, the assertions below would pass on an
empty set, so the floors are checked once, here.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rytp.commands import Command, Param
    from rytp.jobs import JobKind

#: Importing these two modules is what populates both registries.
REGISTRY_MODULES: tuple[str, ...] = ("rytp.commands", "rytp.jobs")

#: Floors, not exact counts: a new command must not have to edit this file.
#: They exist so that a part which forgot its one-line sibling import fails
#: here, loudly, instead of making every test below vacuously true.
MIN_COMMANDS = 50
MIN_JOB_KINDS = 9

#: contracts §5, "Deletion — every entity gets a `remove`". Group -> the
#: command that removes one of its entities. Top-level entries have no group
#: and are keyed by their own name.
DELETION_OWNERS: dict[str, str] = {
    "channel": "channel.remove",
    "videos": "videos.remove",
    "assets": "assets.remove",
    "jobs": "jobs.cancel",
    "transcribe": "transcribe.remove",
    "index": "index.drop",
    "assemble": "assemble.remove",
    "render": "render.remove",
    "speakers": "speakers.remove",
}

#: Removals that delete files, and therefore must offer --dry-run and demand
#: --yes (contracts §5). `channel.remove` orphans videos rather than deleting
#: them, so it is not here.
FILE_DELETING_COMMANDS: frozenset[str] = frozenset(
    {"videos.remove", "assets.remove", "assemble.remove", "render.remove"}
)

#: Removals that only touch derived database rows, and so need neither.
DERIVED_ONLY_REMOVALS: frozenset[str] = frozenset({"index.drop", "transcribe.remove"})


def load_registries() -> tuple[dict[str, Command], dict[str, JobKind]]:
    """Import everything that registers, then hand back both registries."""
    for name in REGISTRY_MODULES:
        importlib.import_module(name)
    from rytp.commands import COMMANDS
    from rytp.jobs import JOB_KINDS

    assert len(COMMANDS) >= MIN_COMMANDS, (
        f"only {len(COMMANDS)} commands registered; a part is probably missing "
        f"its line in the sibling-import block at the bottom of "
        f"rytp/commands/__init__.py. Registered: {sorted(COMMANDS)}"
    )
    assert len(JOB_KINDS) >= MIN_JOB_KINDS, (
        f"only {len(JOB_KINDS)} job kinds registered; a part is probably missing "
        f"its register_job_kind block at the bottom of rytp/jobs/__init__.py. "
        f"Registered: {sorted(JOB_KINDS)}"
    )
    return COMMANDS, JOB_KINDS


def iter_params(commands: dict[str, Command]) -> Iterator[tuple[Command, Param]]:
    """Every (command, parameter) pair, in a stable order."""
    for name in sorted(commands):
        for param in commands[name].params:
            yield commands[name], param


def repo_root() -> Path:
    """The checkout root, found from this file rather than from the cwd."""
    return Path(__file__).resolve().parent.parent


def source_files(package: str = "rytp") -> list[Path]:
    """Every Python source file of one top-level directory, sorted."""
    return sorted(
        path
        for path in (repo_root() / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def read_sources(package: str = "rytp") -> dict[Path, str]:
    """Path -> text, for the regex scans. UTF-8 because the project is Russian."""
    return {path: path.read_text(encoding="utf-8") for path in source_files(package)}


def cli_command_paths() -> set[tuple[str, ...]]:
    """Every argv path the generated Typer app actually answers to.

    Walks the Click command tree rather than parsing ``--help`` text, so a
    command that exists but is hidden still counts and a summary that happens
    to contain a word does not.
    """
    import typer.main

    from rytp.cli import build_app

    root = typer.main.get_command(build_app())
    found: set[tuple[str, ...]] = set()

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        children = getattr(node, "commands", None)
        if not children:
            if prefix:
                found.add(prefix)
            return
        for name, child in children.items():
            walk(child, (*prefix, name))

    walk(root, ())
    return found
```

- [ ] **Step 2: Write the surface-parity tests**

Create `tests/test_consistency_surfaces.py`:

```python
"""Both surfaces, generated from one registry — checked, not assumed.

design §10 claims alignment "holds by construction". Nobody had ever verified
that across all seven parts at once, and the reviews that did it by hand found
a flag spelled two ways and a registry list that had gone stale before a line
of code existed. This module is that reading, turned into assertions.
"""

from __future__ import annotations

import inspect

import pytest

from rytp.commands import HEALTH_CHECKS, REQUIRED, CommandResult, HealthResult, cli_path
from rytp.db import Database
from rytp.tui.palette import palette_entries
from tests.consistency import (
    DELETION_OWNERS,
    cli_command_paths,
    load_registries,
)

COMMANDS, JOB_KINDS = load_registries()
NAMES = sorted(COMMANDS)


def test_both_registries_are_fully_populated() -> None:
    """The floor check in load_registries() ran at import; this names the counts."""
    assert len(COMMANDS) >= 50, sorted(COMMANDS)
    assert len(JOB_KINDS) >= 9, sorted(JOB_KINDS)


@pytest.mark.parametrize("name", NAMES)
def test_every_command_is_reachable_from_the_cli(name: str) -> None:
    """Including `cli_only` ones — the CLI is the surface that launches them."""
    assert cli_path(COMMANDS[name]) in cli_command_paths()


def test_the_cli_exposes_nothing_the_registry_does_not_define() -> None:
    """contracts §5: "Both surfaces are generated from this; neither may define
    a command of its own."."""
    registered = {cli_path(cmd) for cmd in COMMANDS.values()}
    assert cli_command_paths() == registered


@pytest.mark.parametrize("name", NAMES)
def test_every_command_except_cli_only_is_in_the_tui_palette(name: str) -> None:
    entries = {entry.name for entry in palette_entries(COMMANDS)}
    if COMMANDS[name].cli_only:
        assert name not in entries, (
            f"{name} is cli_only: it launches a surface and cannot be invoked "
            "from inside the surface it launches"
        )
    else:
        assert name in entries


def test_only_surface_launchers_are_cli_only() -> None:
    """A `cli_only` command is a surface launcher, not a way to hide work."""
    assert {name for name in NAMES if COMMANDS[name].cli_only} == {"tui"}


@pytest.mark.parametrize("group", sorted(DELETION_OWNERS))
def test_every_group_that_creates_something_can_remove_it(group: str) -> None:
    """contracts §5: "there is no entity you can create but not get rid of"."""
    assert DELETION_OWNERS[group] in COMMANDS


def test_the_deletion_table_covers_every_group_that_has_one() -> None:
    """A new group with entities must appear in the table above, or say why not.

    `queue` and `cache` hold no entities: `queue.pause` is a switch and
    `cache.prune` deletes a regenerable cache, which contracts §7 says is
    prunable by design rather than an entity with an identity.
    """
    groups = {cmd.group for cmd in COMMANDS.values() if cmd.group}
    assert groups - set(DELETION_OWNERS) == {"queue", "cache", "search", "transcript"}


@pytest.mark.parametrize("name", NAMES)
def test_the_handler_accepts_every_parameter_declared_beside_it(name: str) -> None:
    """The defect this catches is a `Param("video")` in front of a handler whose
    keyword is `video_id`: a TypeError at run time, in one command, found by
    whoever runs that command. contracts §5 fixes the calling convention —
    "Handlers take keyword arguments matching `Param.name`" — so it is checkable
    for all of them at once.
    """
    cmd = COMMANDS[name]
    signature = inspect.signature(cmd.handler)
    accepts_anything = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_anything:
        return
    keywords = {
        p.name
        for p in signature.parameters.values()
        if p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    declared = {param.name for param in cmd.params}
    assert declared <= keywords, (
        f"{name}: declared {sorted(declared - keywords)} but the handler "
        f"{cmd.handler.__name__} accepts {sorted(keywords)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_handler_takes_the_database_first(name: str) -> None:
    """contracts §5: handlers "receive an open `Database` as the first positional
    argument"."""
    positional = [
        p
        for p in inspect.signature(COMMANDS[name].handler).parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"{name}: handler takes no positional database argument"
    assert positional[0].name == "db", f"{name}: first argument is {positional[0].name!r}"


@pytest.mark.parametrize("name", NAMES)
def test_every_command_carries_a_summary_both_surfaces_can_show(name: str) -> None:
    summary = COMMANDS[name].summary
    assert summary.strip(), name
    assert summary[0].isupper() or summary[0].isdigit(), f"{name}: {summary!r}"
    assert summary.rstrip().endswith("."), f"{name}: {summary!r}"


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_carries_help_text(name: str) -> None:
    for param in COMMANDS[name].params:
        assert param.help.strip(), f"{name}.{param.name}"


def test_no_command_declares_a_required_parameter_after_an_optional_one() -> None:
    """Part 1's validator already refuses this per command; asserted here across
    the whole registry so a part cannot ship one behind a conditional import."""
    for name in NAMES:
        seen_optional = False
        for param in COMMANDS[name].params:
            if param.default is REQUIRED:
                assert not seen_optional, f"{name}.{param.name} is required but comes late"
            else:
                seen_optional = True


# --- health checks (contracts §5) ------------------------------------


def test_every_health_check_the_contract_names_is_registered() -> None:
    """contracts §5 assigns the checks by part; `doctor` is useless if one is
    missing and nothing says so."""
    names = set(HEALTH_CHECKS)
    for expected in ("ffmpeg", "ffprobe", "fts5", "ffplay"):
        assert expected in names, f"{expected} missing; registered: {sorted(names)}"


def test_only_the_things_that_must_work_are_required() -> None:
    """contracts §5 is explicit about which side of the line each check sits on:
    a missing optional engine reports `ok=False` and is *advisory*, never a
    reason for `doctor` to exit non-zero."""
    required = {name for name, check in HEALTH_CHECKS.items() if check.required}
    for name in ("ffmpeg", "ffprobe", "fts5"):
        assert name in required, f"{name} must be required"
    for name in ("ffplay", "yt-dlp"):
        assert name not in required, f"{name} must be advisory"
    for name, check in HEALTH_CHECKS.items():
        if name.startswith(("transcriber:", "aligner:", "diarizer:")):
            assert not check.required, f"{name} is an engine and must be advisory"


@pytest.mark.parametrize("name", sorted(HEALTH_CHECKS))
def test_no_health_check_raises_on_an_empty_database(db: Database, name: str) -> None:
    """contracts §5: "A check never raises and never blocks"."""
    result = HEALTH_CHECKS[name].run(db)
    assert isinstance(result, HealthResult)
    assert isinstance(result.ok, bool)
    assert result.detail.strip(), f"{name} reported nothing"


#: Commands that produce a table on an empty database, with no arguments
#: beyond their defaults — one per part that lists anything.
LISTABLE = (
    "channel.list",
    "videos.list",
    "jobs.list",
    "jobs.stats",
    "speakers.list",
    "render.list",
    "doctor",
)


@pytest.mark.parametrize("name", LISTABLE)
def test_a_result_is_strings_all_the_way_down(db: Database, name: str) -> None:
    """contracts §5: "Strings only. Formatting a value for display is the
    handler's job, so the CLI and the TUI cannot render the same result
    differently." A handler that leaked an `int` would render as `3` in one
    surface and crash the other's `add_row`.
    """
    cmd = COMMANDS[name]
    from rytp.tui.palette import parse_arguments

    result = cmd.handler(db, **parse_arguments(cmd, ""))
    assert isinstance(result, CommandResult)
    for column in result.columns:
        assert isinstance(column, str), (name, column)
    for row in result.rows:
        assert len(row) == len(result.columns), (name, row)
        for cell in row:
            assert isinstance(cell, str), (name, cell)
```

- [ ] **Step 3: Run it and read every failure**

Run: `python -m pytest tests/test_consistency_surfaces.py -q`

Expected on a correct tree: all pass. Anything red is a bug in the module the
assertion message names. Fix it there, in this task, and say so in the commit —
do not weaken the assertion and do not skip the test.

The two failures most likely to appear, with their fixes:

- `test_the_handler_accepts_every_parameter_declared_beside_it` naming a command
  whose `Param` and keyword disagree. Rename the `Param`, not the handler
  keyword — the flag is the published surface.
- `test_the_deletion_table_covers_every_group_that_has_one` naming a group nobody
  planned a `remove` for. Add the command in the owning part, per the contracts
  §5 table; extend the allowed-without-entities set only for a group that
  genuinely owns no entity.

- [ ] **Step 4: Lint**

Run: `python -m ruff check tests/consistency.py tests/test_consistency_surfaces.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tests/consistency.py tests/test_consistency_surfaces.py
git commit -m "test: assert both surfaces are generated from one registry"
```

---

### Task 2: One flag, one meaning — the vocabulary tests

The defect that survived longest was a spelling: `--video` in five places and
`--video-id` in two, found only when someone grepped. It is not an interesting
bug and that is exactly why it lasted — nothing fails when a flag means one
thing here and another there, until a user types the wrong one.

This task turns the vocabulary into assertions. Every test below names the
finding it would have caught, in its docstring, because a consistency test whose
purpose is forgotten becomes a consistency test somebody deletes.

**Expect this module to be red on its first run.** The defect table at the top of
this plan lists what it should find and the one-line fix for each. Apply the
fixes in the owning modules; this task is not finished until the module is green.

**Files:**
- Test: `tests/test_consistency_flags.py`

**Interfaces:**
- Consumes: `tests/consistency.{load_registries, iter_params, FILE_DELETING_COMMANDS, DERIVED_ONLY_REMOVALS}`, `rytp.commands.{REQUIRED, PARAM_ALIASES, PARAM_TYPES, SPEAKER_PARAMS}`.
- Produces: no production code.

- [ ] **Step 1: Write the tests**

Create `tests/test_consistency_flags.py`:

```python
"""One flag, one meaning, one spelling — across all fifty-odd commands.

Each test names the review finding it exists to prevent. The findings are in
`docs/superpowers/2026-09-21-review-findings.md`; the ones with no test were
found by a person reading seven plans side by side, which is not a repeatable
process.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from rytp.commands import PARAM_ALIASES, PARAM_TYPES, REQUIRED, SPEAKER_PARAMS
from tests.consistency import (
    DERIVED_ONLY_REMOVALS,
    FILE_DELETING_COMMANDS,
    iter_params,
    load_registries,
)

COMMANDS, _JOB_KINDS = load_registries()
NAMES = sorted(COMMANDS)

#: A concept that must have exactly one spelling, mapped from the spellings
#: that are *not* it. Finding 4.2 chose `--video`; `--enqueue` is the spelling
#: three of the four enqueue flags already use.
BANNED_SPELLINGS: dict[str, str] = {
    "video_id": "video",
    "videoid": "video",
    "queue": "enqueue",
    "global_speaker": "speaker",
    "limit_rows": "limit",
    "max_rows": "limit",
}

#: Names whose type is fixed across the whole surface, because a flag that is
#: a string here and an integer there is two flags wearing one name.
PINNED_TYPES: dict[str, type] = {
    "video": str,  # resolve_video_id accepts a row id or an external id
    "speaker": str,
    "video_local_speaker": str,
    "limit": int,
    "priority": int,
    "seed": int,
    "pad_ms": int,
    "dry_run": bool,
    "yes": bool,
    "force": bool,
    "enqueue": bool,
}


def _params_by_name() -> dict[str, list[tuple[str, object]]]:
    grouped: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for cmd, param in iter_params(COMMANDS):
        grouped[param.name].append((cmd.name, param))
    return grouped


PARAMS_BY_NAME = _params_by_name()


@pytest.mark.parametrize("banned", sorted(BANNED_SPELLINGS))
def test_no_concept_is_spelled_two_ways(banned: str) -> None:
    """Finding 4.2: `--video-id` and `--video` were the same thing, and the
    rename was applied to the prose but not to every registration."""
    users = [name for name, _ in PARAMS_BY_NAME.get(banned, [])]
    assert not users, (
        f"{users} declare --{banned.replace('_', '-')}; the agreed spelling is "
        f"--{BANNED_SPELLINGS[banned].replace('_', '-')}"
    )


@pytest.mark.parametrize("param_name", sorted(PINNED_TYPES))
def test_one_parameter_name_has_one_type(param_name: str) -> None:
    """`--video` as a `str` on one command and an `int` on another is two
    flags sharing a name: one accepts `VIDEO_A`, the other refuses it."""
    expected = PINNED_TYPES[param_name]
    wrong = [
        (name, param.type.__name__)
        for name, param in PARAMS_BY_NAME.get(param_name, [])
        if param.type is not expected
    ]
    assert not wrong, f"--{param_name} must be {expected.__name__}; wrong: {wrong}"


def test_one_parameter_name_has_one_type_everywhere_else_too() -> None:
    """The open-ended half of the rule above: any name used by two commands
    with two different types is a finding, whether or not it is pinned.

    Types, not meanings. A *positional* may legitimately mean different things
    in different commands — `videos add <target>` takes a URL or a path and
    `assemble plan <target>` takes a sentence — because a positional is read
    in the context of the verb in front of it and is never typed as a flag. An
    option's name is typed alone, so its meaning must not move; that is what
    `BANNED_SPELLINGS` and the speaker tests above enforce.
    """
    clashes = {
        name: sorted({param.type.__name__ for _, param in entries})
        for name, entries in PARAMS_BY_NAME.items()
        if len({param.type for _, param in entries}) > 1
    }
    assert not clashes, f"one name, several types: {clashes}"


def test_every_parameter_name_is_a_python_identifier() -> None:
    """`Param.name` is the handler's keyword argument, so `min-align` could
    never be passed. Whether the CLI renders it with a dash is Part 1's job."""
    bad = [
        f"{cmd.name}.{param.name}"
        for cmd, param in iter_params(COMMANDS)
        if not param.name.isidentifier() or param.name != param.name.lower()
    ]
    assert not bad, bad


def test_every_parameter_type_is_one_the_surfaces_can_render() -> None:
    bad = [
        f"{cmd.name}.{param.name}: {param.type!r}"
        for cmd, param in iter_params(COMMANDS)
        if param.type not in PARAM_TYPES
    ]
    assert not bad, bad


# --- short flags -----------------------------------------------------


def test_every_short_flag_is_a_dash_and_one_letter() -> None:
    """Part 1's validator demands `-n`; `short="n"` raises at import. Pinned
    here so the shape is stated once for every part rather than discovered by
    whoever adds the next flag."""
    bad = [
        f"{cmd.name}.{param.name}={param.short!r}"
        for cmd, param in iter_params(COMMANDS)
        if param.short is not None and not re.fullmatch(r"-[A-Za-z]", param.short)
    ]
    assert not bad, bad


def test_a_short_flag_is_unique_within_its_command() -> None:
    for name in NAMES:
        shorts = [p.short for p in COMMANDS[name].params if p.short]
        assert len(shorts) == len(set(shorts)), f"{name}: {shorts}"


def test_a_short_letter_means_one_thing_across_the_whole_surface() -> None:
    """Muscle memory does not know which command it is in. `-s` meaning
    `--search` in one place and `--speaker` in another is how a filtered list
    silently becomes an unfiltered one."""
    meanings: dict[str, set[str]] = defaultdict(set)
    for cmd, param in iter_params(COMMANDS):
        if param.short:
            meanings[param.short].add(param.name)
    clashes = {short: sorted(names) for short, names in meanings.items() if len(names) > 1}
    assert not clashes, f"one letter, several meanings: {clashes}"


# --- the speaker flags (contracts §5) --------------------------------

SPEAKER_PARAM_NAMES = {param.name for param in SPEAKER_PARAMS}


def test_the_shared_speaker_flags_mean_the_same_thing_everywhere() -> None:
    """contracts §5 pins two identifier spaces and one resolver. A command may
    legitimately offer only some of the three — assembling from the corpus has
    no use for a per-video diarizer label — so this checks each name it *does*
    offer against the shared declaration rather than demanding all three."""
    shared = {param.name: param for param in SPEAKER_PARAMS}
    for cmd, param in iter_params(COMMANDS):
        if param.name not in shared:
            continue
        reference = shared[param.name]
        assert param.type is reference.type, f"{cmd.name}.{param.name}: type"
        assert param.default is reference.default, (
            f"{cmd.name}.{param.name} defaults to {param.default!r}; the shared "
            f"declaration uses {reference.default!r}. `None` means 'not asked "
            f"for', which is a different answer from 'asked for nothing'."
        )


def test_a_per_video_label_always_comes_with_a_video() -> None:
    """contracts §5: "`--video-local-speaker` **requires a video to be named as
    well**". A bare SPEAKER_00 would match the first-detected voice of every
    diarized video, which is not a person."""
    for name in NAMES:
        params = {p.name for p in COMMANDS[name].params}
        if "video_local_speaker" in params:
            assert "video" in params, f"{name} takes a local label but no --video"


def test_the_speaker_alias_is_declared_once_and_reaches_the_speaker_flag() -> None:
    assert PARAM_ALIASES.get("speaker") == ("--global-speaker",)


# --- conventions -----------------------------------------------------


def test_a_boolean_flag_defaults_to_false_or_says_why_not() -> None:
    """A flag reads as "turn this on". The three that default to True are
    behaviour you turn *off*, and each is named here so a fourth cannot
    appear silently."""
    on_by_default = {
        (cmd.name, param.name)
        for cmd, param in iter_params(COMMANDS)
        if param.type is bool and param.default is True
    }
    assert on_by_default == {
        ("fetch-video", "captions"),
        ("render.run", "loudnorm"),
        ("transcribe.run", "refine"),
        ("transcribe.align", "refine"),
    }, sorted(on_by_default)


@pytest.mark.parametrize("name", sorted(FILE_DELETING_COMMANDS))
def test_anything_that_deletes_a_file_plans_first_and_asks(name: str) -> None:
    """contracts §5: "Anything that deletes a file supports `--dry-run` and
    requires `--yes`"."""
    assert name in COMMANDS
    params = {p.name for p in COMMANDS[name].params}
    assert {"dry_run", "yes"} <= params, f"{name} is missing {sorted({'dry_run', 'yes'} - params)}"


@pytest.mark.parametrize("name", sorted(DERIVED_ONLY_REMOVALS))
def test_a_removal_of_derived_rows_needs_no_ceremony(name: str) -> None:
    """The same contract sentence, read the other way: `index.drop` rebuilds
    from `index build`, so demanding --yes would be noise."""
    params = {p.name for p in COMMANDS[name].params}
    assert "yes" not in params, f"{name} demands --yes for rows it can rebuild"


def test_removal_is_never_a_job() -> None:
    """contracts §5: "Removal is not a job… A half-deleted entity recovered
    from a crashed queue is worse than a slow command"."""
    removals = sorted(FILE_DELETING_COMMANDS | DERIVED_ONLY_REMOVALS)
    for name in removals:
        assert COMMANDS[name].long_running is False, f"{name} is marked long_running"
        params = {p.name for p in COMMANDS[name].params}
        assert "enqueue" not in params, f"{name} offers to queue a deletion"
```

- [ ] **Step 2: Run it and expect red**

Run: `python -m pytest tests/test_consistency_flags.py -q`

Expected, against the tree as Parts 1–7 leave it: several failures. Work through
them in the owning modules using the table at the top of this plan. In
particular, renaming `Param("video_id", …)` to `Param("video", …)` means
renaming the handler keyword too — the previous task's handler-signature test
will catch it if you rename only one of the pair, which is the point of doing
that task first.

- [ ] **Step 3: Run the whole suite, because renaming a flag moves a call site**

Run: `python -m pytest -q`
Expected: green. A test that passed a flag by its old name fails here; update the
call, never the assertion.

- [ ] **Step 4: Lint**

Run: `python -m ruff check tests/test_consistency_flags.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tests/test_consistency_flags.py rytp/commands
git commit -m "test: pin one spelling and one meaning for every flag"
```

---

### Task 3: Every job kind has a handler, a predicate and a producer

The review found "a job kind had a handler but no producer, so the GPU pool would
never have received work" and "a hardcoded list of job kinds went stale before a
line of code existed". Both are the same shape: the queue's registry is assembled
from five files owned by five parts, and nothing looks at the assembled whole.

Five parts append to the bottom of `rytp/jobs/__init__.py`. This module reads
what they produced together.

**Files:**
- Modify: `tests/consistency.py` (append the WAV helper and the producer table)
- Test: `tests/test_consistency_jobs.py`

**Interfaces:**
- Consumes: `rytp.jobs.{JOB_KINDS, JOB_HANDLERS, Readiness}`, `rytp.jobs.queue.{enqueue, get_job}`, `rytp.jobs.worker.{run_worker, WorkerOptions}`, `rytp.constants.{POOLS, INGEST_CHAIN_REMOTE, INGEST_CHAIN_LOCAL, INGEST_CHAIN_TRANSCRIBE}`, `tests.fakes.{make_video, temp_job_kind}`, `tests.fake_engines.{FakeTranscriber, FakeAligner, registered}`.
- Produces: `tests/consistency.{PRODUCERS, write_test_wav, enqueued_kinds_in_source}`.

- [ ] **Step 1: Append the helpers to `tests/consistency.py`**

```python
#: Every registered job kind, mapped to the registered *commands* a person can
#: run to create one. A kind with no producer is work the worker will never
#: receive — the review found exactly that on the gpu pool. The test below
#: asserts this table's keys are the registry's keys, so adding a kind without
#: adding a producer fails by name instead of going unnoticed.
PRODUCERS: dict[str, tuple[str, ...]] = {
    "download": ("ingest", "fetch-video"),
    "captions": ("ingest", "fetch-video"),
    "caption_words": ("ingest",),
    "extract_wav": ("ingest", "fetch-video"),
    "fingerprint": ("ingest", "transcribe.fingerprint"),
    "transcribe": ("ingest", "transcribe.run"),
    "align": ("ingest", "transcribe.align"),
    "index": ("index.build",),
    "diarize": ("speakers.enqueue",),
    "render": ("render.run",),
}


def enqueued_kinds_in_source() -> dict[str, list[str]]:
    """Job-kind string literals handed to ``enqueue`` anywhere under ``rytp/``.

    A regex, deliberately: the alternative is importing and calling everything,
    and a literal is exactly what goes stale. Kinds passed through a variable
    are invisible here and are covered by the ingest-chain test instead.
    """
    import re

    pattern = re.compile(r"""enqueue\(\s*db\s*,\s*["']([a-z_]+)["']""")
    found: dict[str, list[str]] = {}
    for path, text in read_sources("rytp").items():
        for kind in pattern.findall(text):
            found.setdefault(kind, []).append(str(path.name))
    return found


def write_test_wav(path: Path, *, duration_ms: int = 1000, freq_hz: int = 220) -> Path:
    """A real 16 kHz mono int16 WAV, so nothing downstream has to be faked.

    A tone rather than silence: the acoustic fingerprint measures f0 and
    spectral tilt, and those are undefined on a flat zero signal.
    """
    import math
    import wave

    from rytp import config

    sample_rate = 16_000
    frames = int(sample_rate * duration_ms / 1000)
    config.ensure_dir(path.parent)
    samples = bytearray()
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * freq_hz * index / sample_rate))
        samples += int(value).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))
    return path
```

- [ ] **Step 2: Write the tests**

Create `tests/test_consistency_jobs.py`:

```python
"""The queue's registry, read as a whole rather than one part at a time.

Five parts append registrations to the bottom of `rytp/jobs/__init__.py` and
nobody reads the result. Two of the reviews' findings came from exactly that:
a kind whose only producer was never written, and a hardcoded list of kinds
that was stale before there was any code to be stale about.
"""

from __future__ import annotations

import inspect

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.jobs import JOB_HANDLERS, JOB_KINDS, Readiness
from rytp.jobs import queue as Q
from rytp.jobs.worker import WorkerOptions, run_worker
from tests.consistency import PRODUCERS, enqueued_kinds_in_source, load_registries
from tests.fakes import make_video

COMMANDS, _ = load_registries()
KINDS = sorted(JOB_KINDS)

#: contracts §5 "Job handlers" assigns every kind to a part. Kept as a set so
#: a kind that appears without a contract entry is visible.
CONTRACTED_KINDS = {
    "download",
    "captions",
    "extract_wav",
    "caption_words",
    "transcribe",
    "align",
    "fingerprint",
    "index",
    "diarize",
    "render",
}


def test_the_registry_is_exactly_the_contracted_set() -> None:
    """contracts §5 lists the kinds by owner; `caption_words` was added to it
    by agreement between Parts 2 and 3 and belongs here too."""
    assert set(JOB_KINDS) == CONTRACTED_KINDS


def test_the_flat_handler_view_cannot_drift_from_the_registry() -> None:
    """contracts §5 requires `JOB_HANDLERS`; Part 2 writes it only through
    `register_job_kind`, which is what makes drift impossible. Pinned so a
    later part cannot start writing it directly."""
    assert set(JOB_HANDLERS) == set(JOB_KINDS)
    for name, kind in JOB_KINDS.items():
        assert JOB_HANDLERS[name] is kind.handler


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_has_a_handler_and_a_readiness_predicate(kind: str) -> None:
    entry = JOB_KINDS[kind]
    assert callable(entry.handler), kind
    assert callable(entry.readiness), kind
    assert entry.summary.strip(), kind


@pytest.mark.parametrize("kind", KINDS)
def test_a_handler_takes_the_database_the_target_and_the_payload(kind: str) -> None:
    """contracts §5 fixes the shape: `(Database, int, dict) -> str | None`."""
    params = list(inspect.signature(JOB_KINDS[kind].handler).parameters)
    assert len(params) == 3, f"{kind}: {params}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_readiness_predicate_never_sees_the_payload(kind: str) -> None:
    """design §5: readiness is "a statement about the world". A predicate that
    changed its mind based on how a job was enqueued would take the
    self-healing property with it."""
    params = list(inspect.signature(JOB_KINDS[kind].readiness).parameters)
    assert len(params) == 2, f"{kind}: {params}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_predicate_answers_blocked_for_a_target_that_does_not_exist(
    db: Database, kind: str
) -> None:
    """The worker calls every predicate against whatever ids the table holds,
    including one whose row was removed. None of them may raise."""
    assert JOB_KINDS[kind].readiness(db, 999_999) is Readiness.BLOCKED


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_runs_on_a_declared_pool(kind: str) -> None:
    assert JOB_KINDS[kind].pool in C.POOLS, f"{kind}: {JOB_KINDS[kind].pool}"


def test_the_gpu_pool_is_the_expensive_one_and_holds_only_what_belongs_there() -> None:
    """design §5's table. A cheap kind on the gpu pool serialises behind
    transcription for no reason."""
    gpu = {name for name, kind in JOB_KINDS.items() if kind.pool == "gpu"}
    assert gpu == {"transcribe", "align", "diarize"}


def test_target_kind_namespaces_the_only_kind_that_needs_it() -> None:
    """contracts §3: `render` targets a `renders` row, everything else a video.
    The namespace is what stops render 7 re-evaluating video 7's jobs."""
    by_target = {name: kind.target_kind for name, kind in JOB_KINDS.items()}
    assert by_target.pop("render") == "render"
    assert set(by_target.values()) == {"video"}


def test_only_the_one_shots_refuse_to_be_reopened() -> None:
    """`reconcile` turns a done job back into work when its output vanishes.
    That is right for anything derived from the corpus and wrong for work a
    person explicitly asked for once — re-timing, and hours of GPU time."""
    not_reopenable = {name for name, kind in JOB_KINDS.items() if not kind.reopenable}
    assert not_reopenable == {"align", "diarize", "render"}


# --- producers -------------------------------------------------------


def test_the_producer_table_covers_the_registry_exactly() -> None:
    """The staleness guard. A new kind must be classified here, and this is the
    assertion that makes forgetting impossible."""
    assert set(PRODUCERS) == set(JOB_KINDS), (
        f"unclassified kinds: {sorted(set(JOB_KINDS) - set(PRODUCERS))}; "
        f"kinds that no longer exist: {sorted(set(PRODUCERS) - set(JOB_KINDS))}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_every_registered_kind_has_at_least_one_producer(kind: str) -> None:
    """The review's finding, in one line: a kind with a handler and no producer
    is work the worker will never receive."""
    producers = PRODUCERS[kind]
    assert producers, f"nothing can create a {kind} job"
    for command in producers:
        assert command in COMMANDS, f"{kind}'s producer {command!r} is not registered"


def test_every_kind_enqueued_in_the_source_tree_is_registered() -> None:
    """A typo in an `enqueue(db, "fingerpint", …)` is a job nobody ever runs."""
    found = enqueued_kinds_in_source()
    unknown = {kind: files for kind, files in found.items() if kind not in JOB_KINDS}
    assert not unknown, f"enqueued but never registered: {unknown}"


@pytest.mark.parametrize(
    "chain",
    [C.INGEST_CHAIN_REMOTE, C.INGEST_CHAIN_LOCAL, C.INGEST_CHAIN_TRANSCRIBE],
)
def test_every_ingest_chain_names_registered_kinds(chain: tuple[str, ...]) -> None:
    assert set(chain) <= set(JOB_KINDS), sorted(set(chain) - set(JOB_KINDS))


def test_diarization_is_never_reached_by_accident() -> None:
    """design §6: "Diarization is opt-in per video… at full corpus it is the
    single largest GPU cost". This is the robust form of the assertion — it
    survives Part 3 adding two kinds to the chain, which the version that
    pinned the chain's exact contents did not."""
    for chain in (C.INGEST_CHAIN_REMOTE, C.INGEST_CHAIN_LOCAL, C.INGEST_CHAIN_TRANSCRIBE):
        assert "diarize" not in chain, chain


def test_promoting_a_video_does_not_make_its_captions_runnable_again(
    db: Database,
) -> None:
    """design §6: "Promoting a video from tier 1 to tier 2 deletes its caption
    words. One tier per video at a time."

    `reconcile` reopens a done job whose output has gone, which is what makes
    the queue self-healing — and a caption-words predicate that only looks for
    caption-tier rows sees exactly that after a promotion. The result would be
    caption words reappearing underneath an aligned transcript, which contracts
    §3 says cannot coexist. The predicate must read "this video has words",
    not "this video has caption words".
    """
    from rytp.db.queries import insert_asset

    video_id = make_video(db)
    insert_asset(db, video_id=video_id, role="captions", path="captions.json3")
    db.conn.execute(
        "INSERT INTO words"
        " (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        "  source, engine)"
        " VALUES (?, 0, 0, 100, 'раз', 'раз', 'раз', 'aligned', 'fake')",
        (video_id,),
    )
    assert JOB_KINDS["caption_words"].readiness(db, video_id) is Readiness.SATISFIED


def test_transcription_is_opt_in_too() -> None:
    """design §6: tier 2 is "for videos you actually want to cut from", and
    design §13 prices the corpus at 130-200 GPU hours."""
    assert "transcribe" not in C.INGEST_CHAIN_REMOTE
    assert "transcribe" not in C.INGEST_CHAIN_LOCAL
    assert "align" not in C.INGEST_CHAIN_REMOTE


# --- notes survive the worker (review finding 3.2) -------------------


def test_a_handler_note_reaches_the_job_row(db: Database) -> None:
    """contracts §5: "A handler may return a short note… The worker stores it
    in `jobs.note`". This is the mechanism; the next test is the case it was
    added for."""
    from tests.fakes import temp_job_kind

    def noted(db_: Database, target_id: int, payload: dict[str, object]) -> str:
        return "discarded 4 speaker labels"

    with temp_job_kind("t_note", pool="cpu", handler=noted):
        video_id = make_video(db)
        job_id = Q.enqueue(db, "t_note", video_id)
        run_worker(db, WorkerOptions(pools=("cpu",), once=True))
        job = Q.get_job(db, job_id)
    assert job.state == "done"
    row = db.conn.execute("SELECT note FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["note"] == "discarded 4 speaker labels"


def test_re_transcribing_a_mapped_video_leaves_the_warning_on_the_job(
    db: Database, data_dir: object
) -> None:
    """Review finding 3.2, on the path it was reported for.

    Re-transcribing deletes a video's speaker mapping. Part 3 computes the
    warning and shows it to whoever runs `transcribe run` by hand — and the
    worker, which is the bulk path, is the one that will do this across
    hundreds of videos with nobody reading stdout. If this fails, the fix is
    in `rytp/jobs/__init__.py::_run_transcribe`: return the string
    `transcribe_video` already produced instead of dropping it.
    """
    from rytp.audio.extract import wav_path
    from tests.consistency import write_test_wav
    from tests.fake_engines import FakeAligner, FakeTranscriber, registered

    video_id = make_video(db)
    write_test_wav(wav_path(video_id))
    db.conn.execute(
        "INSERT INTO words"
        " (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        "  source, engine)"
        " VALUES (?, 0, 0, 100, 'раз', 'раз', 'раз', 'timed', 'old')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (?, 'SPEAKER_00', 'pyannote')",
        (video_id,),
    )

    with registered(FakeTranscriber, FakeAligner):
        job_id = Q.enqueue(
            db,
            "transcribe",
            video_id,
            payload={"transcriber": "fake", "aligner": "fake-aligner"},
        )
        run_worker(db, WorkerOptions(pools=("gpu",), once=True))

    job = Q.get_job(db, job_id)
    assert job.state == "done", job.last_error
    note = db.conn.execute(
        "SELECT note FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["note"]
    assert note, (
        "re-transcribing discarded a speaker mapping and the worker recorded "
        "nothing; see rytp/jobs/__init__.py::_run_transcribe"
    )
    assert "speaker" in note.lower()
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/test_consistency_jobs.py -q`

Expected red on `test_re_transcribing_a_mapped_video_leaves_the_warning_on_the_job`,
because Part 3's thunks are written `-> None` and return nothing. The fix, in
`rytp/jobs/__init__.py`:

```python
def _run_transcribe(db: Database, video_id: int, payload: dict[str, Any]) -> str | None:
    ...
    note = transcribe_video(...)   # the warning Part 3's CLI handler already shows
    enqueue_index(db, video_id)
    return note
```

`transcribe_video`'s return type decides the exact line; read Part 3's
`rytp/transcribe/pipeline.py` and return whatever it already computed for the
CLI path rather than recomputing it here.

`test_only_the_one_shots_refuse_to_be_reopened` may also be red: Part 6 leaves
`render` reopenable so that deleting `output.mp4` re-runs it, while Part 2's
prose says a render "must not silently re-run on the next worker start". Read
Part 6's Task 8 reasoning, decide, and make the assertion match the decision —
this one is a genuine disagreement between two plans, not a typo.

- [ ] **Step 4: Lint**

Run: `python -m ruff check tests/consistency.py tests/test_consistency_jobs.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tests/consistency.py tests/test_consistency_jobs.py rytp/jobs/__init__.py
git commit -m "test: every job kind has a handler, a predicate and a producer"
```

---

### Task 4: The schema, the fixtures, and what each part swore it consumes

Three more findings, all of the same family: something one part depends on lived
somewhere else, or had a different shape, or was never supplied at all.

- A fixture omitted `video_speakers.engine`, a `NOT NULL` column. Contracts §3
  makes supplying it "the inserting part's job", which means it is nobody's job
  to check.
- Three parts assumed three shapes for the speaker resolver, because the
  responsibility was named and the signature was not.
- `rytp/models.py` imports `snowballstemmer` lazily, which is what keeps Part
  3's `base.py` stdlib-clean under a foreign interpreter — "load-bearing and
  currently unpinned", in the review's own words. This task pins it.

**Files:**
- Test: `tests/test_consistency_schema.py`

**Interfaces:**
- Consumes: `rytp.db.{Database}`, `rytp.db.schema.MIGRATIONS`, `tests.consistency.{repo_root, read_sources, source_files}`.
- Produces: no production code.

- [ ] **Step 1: Write the tests**

Create `tests/test_consistency_schema.py`:

```python
"""The schema, the fixtures that write into it, and every declared dependency.

Contracts §3 makes each part responsible for the NOT NULL columns of the tables
it inserts into, which means no part is responsible for checking that they all
did. Contracts §5 pins signatures precisely because naming a responsibility
without naming a shape is how three parts invented three speaker resolvers.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rytp.db import Database
from rytp.db.schema import MIGRATIONS
from tests.consistency import read_sources, repo_root, source_files

# --- the schema is exactly the contract ------------------------------

CONTRACT_TABLES = {
    "channels",
    "videos",
    "assets",
    "speakers",
    "video_speakers",
    "words",
    "utterances",
    "utterances_fts",
    "video_acoustics",
    "jobs",
    "renders",
    "settings",
    "schema_version",
}

CONTRACT_INDEXES = {
    "videos_channel",
    "videos_source",
    "assets_video_role",
    "assets_one_audio",
    "assets_one_captions",
    "video_speakers_speaker",
    "words_video_ord",
    "words_normalized",
    "words_alignable",
    "words_stem",
    "words_speaker",
    "utterances_video",
    "jobs_claim",
    "renders_cutlist",
}

CONTRACT_TRIGGERS = {"utterances_ai", "utterances_ad", "utterances_au"}


def _names(db: Database, kind: str) -> set[str]:
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
    ).fetchall()
    return {
        str(row["name"])
        for row in rows
        if not str(row["name"]).startswith("sqlite_")
        and not re.fullmatch(r"utterances_fts_\w+", str(row["name"]))
    }


def test_the_tables_are_exactly_the_ones_the_contract_lists(db: Database) -> None:
    """contracts §3: "Part 1 creates every table in this section… Later parts
    consume the schema; none of them add DDL"."""
    assert _names(db, "table") == CONTRACT_TABLES


def test_the_indexes_are_exactly_the_ones_the_contract_lists(db: Database) -> None:
    """`words_alignable` is here because finding 4.1 was an index that did not
    exist: without it the assembler's hot lookup scales with the caption tier."""
    assert _names(db, "index") == CONTRACT_INDEXES


def test_the_fts_triggers_all_exist(db: Database) -> None:
    assert _names(db, "trigger") == CONTRACT_TRIGGERS


def test_migrations_are_contiguous_and_nobody_renumbered(db: Database) -> None:
    versions = [version for version, _sql in MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1)), versions
    assert db.schema_version() == versions[-1]


def test_the_three_transcript_tiers_are_enforced_by_the_database(db: Database) -> None:
    """contracts §3: "Cuttable is `source = 'aligned'` and nothing else may be
    cut". `timed` exists because a transcriber's own word timings are not good
    enough to cut on, and calling them `aligned` is the mistake the tier was
    invented to make impossible."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words'"
    ).fetchone()["sql"]
    for tier in ("caption", "timed", "aligned"):
        assert f"'{tier}'" in sql, tier


def test_a_caption_row_is_the_only_one_allowed_to_have_no_end(db: Database) -> None:
    """contracts §3's CHECK. Part 4's `implied_end_ms` exists because of it."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words'"
    ).fetchone()["sql"]
    assert "source = 'caption' OR end_ms IS NOT NULL" in sql.replace("\n", " ")


def test_the_fts_tokenizer_is_never_the_english_one(db: Database) -> None:
    """Porter is English-only; on a Russian corpus it stems nothing at all."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()["sql"]
    assert "unicode61" in sql
    assert "porter" not in sql


# --- NOT NULL columns and the fixtures that must supply them ---------


def _required_columns(db: Database, table: str) -> set[str]:
    """NOT NULL, no default, not the rowid alias — i.e. every insert's job."""
    return {
        str(row["name"])
        for row in db.conn.execute(f"PRAGMA table_info({table})")
        if row["notnull"] and row["dflt_value"] is None and not row["pk"]
    }


_INSERT_RE = re.compile(
    r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL
)


def _literal_inserts() -> list[tuple[Path, str, set[str]]]:
    """(file, table, columns) for every literal column-list INSERT under tests/."""
    found: list[tuple[Path, str, set[str]]] = []
    for path, text in read_sources("tests").items():
        for table, columns in _INSERT_RE.findall(text):
            names = {
                part.strip().strip('"').strip("'")
                for part in columns.replace("\n", " ").split(",")
                if part.strip()
            }
            found.append((path, table, names))
    return found


def test_every_fixture_supplies_the_not_null_columns_of_the_table_it_writes(
    db: Database,
) -> None:
    """The `video_speakers.engine` finding, generalised.

    An omitted NOT NULL column fails at insert time, in whichever test happens
    to run first, with a message about a constraint rather than about a
    fixture. This names the file and the column instead.
    """
    tables = _names(db, "table")
    missing: list[str] = []
    for path, table, columns in _literal_inserts():
        if table not in tables or table.endswith("_fts"):
            continue
        absent = _required_columns(db, table) - columns
        if absent:
            missing.append(f"{path.name}: INSERT INTO {table} omits {sorted(absent)}")
    assert not missing, missing


def test_the_shared_video_fixture_writes_a_complete_row(db: Database) -> None:
    """`tests/fakes.py::make_video` is used by six modules; if it ever stops
    filling a required column, everything downstream fails at once."""
    from tests.fakes import make_video

    video_id = make_video(db)
    row = db.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    for column in _required_columns(db, "videos"):
        assert row[column] is not None, column


# --- import hygiene --------------------------------------------------


def _child(code: str) -> subprocess.CompletedProcess[str]:
    """Run one line in a fresh interpreter, from a directory with no data tree."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )


def test_models_imports_snowballstemmer_lazily() -> None:
    """Explicitly called out by the review as load-bearing and unpinned.

    `rytp/transcribe/base.py` is imported inside a foreign interpreter — an
    engine's own virtualenv, which has rytp's dependencies but not necessarily
    rytp's extras. It reaches `rytp.models` for `RawWord` and `Span`. A
    module-level `import snowballstemmer` there turns a missing package in
    someone else's environment into an out-of-process engine that cannot start.
    """
    source = (repo_root() / "rytp" / "models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in getattr(node, "names", [])
    }
    module_names = {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "snowballstemmer" not in top_level | module_names, (
        "rytp/models.py imports snowballstemmer at module level; move it inside "
        "stem_text()"
    )
    assert "snowballstemmer" in source, "the stemmer must still be used, just lazily"

    proc = _child(
        "import sys, rytp.models; "
        "print('leaked' if 'snowballstemmer' in sys.modules else 'clean')"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "clean"


def test_importing_the_registries_pulls_in_no_optional_extra() -> None:
    """contracts §1: heavy dependencies are "imported lazily inside functions,
    never at module import time". The registries are what every command and the
    worker import, so this is the import path that matters."""
    proc = _child(
        "import sys, rytp.commands, rytp.jobs, rytp.tui.app; "
        "heavy = {'torch', 'transformers', 'faster_whisper', 'gigaam', "
        "'pyannote', 'yt_dlp', 'torchaudio'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_importing_rytp_creates_no_directories(tmp_path: Path) -> None:
    """The gotcha the old codebase had: a module-level `paths` singleton that
    mkdir'd on import. contracts §7: created "by the command that needs it —
    **not at module import time**"."""
    proc = subprocess.run(
        [sys.executable, "-c", "import rytp, rytp.config, rytp.commands"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(repo_root())},
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_every_base_dependency_is_importable() -> None:
    """contracts §1 names six, and a `.[dev]` install must run every test."""
    for name in ("typer", "rich", "textual", "numpy", "scipy", "snowballstemmer"):
        importlib.import_module(name)


def test_every_module_uses_postponed_annotations() -> None:
    """contracts §1: "`from __future__ import annotations` in **every** module"."""
    missing = [
        str(path.relative_to(repo_root()))
        for path in source_files("rytp")
        if path.name != "__init__.py" or path.read_text(encoding="utf-8").strip()
        if "from __future__ import annotations" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, missing


def test_no_real_identifier_is_committed() -> None:
    """contracts §1, and the owner's standing instruction: no video ids, channel
    ids, channel names or URLs anywhere, tests included."""
    pattern = re.compile(r"youtube\.com|youtu\.be|/@[A-Za-z0-9_]{3,}")
    offenders = [
        str(path)
        for package in ("rytp", "tests")
        for path, text in read_sources(package).items()
        if pattern.search(text)
    ]
    assert not offenders, offenders


# --- what every part declared it consumes ----------------------------

#: (module, attribute, required parameter names or None). Assembled from the
#: `Interfaces / Consumes` blocks of Parts 1-7. contracts §5 fixes the ones
#: with parameter lists; the rest need only exist, because a part that renamed
#: an export would otherwise fail at whatever hour the consumer first runs.
CONSUMED: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("rytp.models", "normalize_text", ("text",)),
    ("rytp.models", "stem_text", ("normalized",)),
    ("rytp.models", "utc_now_iso", ()),
    ("rytp.models", "RawWord", None),
    ("rytp.models", "Span", None),
    ("rytp.models", "DiarSegment", None),
    ("rytp.models", "Fragment", None),
    ("rytp.models", "RytpError", None),
    ("rytp.models", "NotFoundError", None),
    ("rytp.models", "InvalidInputError", None),
    ("rytp.commands", "SpeakerFilter", None),
    (
        "rytp.commands",
        "resolve_speaker_filter",
        ("db", "speaker", "video_local_speaker", "video_id"),
    ),
    ("rytp.commands", "resolve_video_id", ("db", "ref")),
    ("rytp.commands", "register", ("cmd",)),
    ("rytp.commands", "resolve", ("name",)),
    ("rytp.commands", "register_check", ("check",)),
    ("rytp.commands", "HEALTH_CHECKS", None),
    ("rytp.commands", "SPEAKER_PARAMS", None),
    ("rytp.commands", "PARAM_ALIASES", None),
    ("rytp.cli", "build_app", None),
    ("rytp.config", "paths", ()),
    ("rytp.config", "ensure_dir", ("path",)),
    ("rytp.db", "Database", None),
    ("rytp.db.queries", "get_setting", ("db", "key", "default")),
    ("rytp.db.queries", "set_setting", ("db", "key", "value")),
    ("rytp.jobs", "register_job_kind", ("kind",)),
    ("rytp.jobs", "resolve_job_kind", ("name",)),
    ("rytp.jobs", "kinds_for_pool", ("pool",)),
    ("rytp.jobs.queue", "enqueue", ("db", "kind", "target_id", "priority", "payload")),
    ("rytp.jobs.queue", "list_jobs", ("db", "state", "pool", "kind", "limit")),
    ("rytp.jobs.queue", "stats", ("db",)),
    ("rytp.jobs.queue", "retry", ("db", "job_id", "kind", "state")),
    ("rytp.jobs.queue", "get_job", ("db", "job_id")),
    ("rytp.jobs.queue", "is_paused", ("db",)),
    ("rytp.jobs.worker", "run_worker", None),
    ("rytp.audio.extract", "wav_path", ("video_id",)),
    ("rytp.transcribe.registry", "TRANSCRIBERS", None),
    ("rytp.transcribe.registry", "ALIGNERS", None),
    ("rytp.diarize.base", "DIARIZERS", None),
    ("rytp.assemble.cutlist", "load_cutlist", ("path",)),
    ("rytp.assemble.cutlist", "write_cutlist", ("cutlist", "path")),
    ("rytp.assemble.cutlist", "cutlist_path", ("name",)),
    ("rytp.assemble.cutlist", "validate_name", ("name",)),
    ("rytp.assemble.cutlist", "CutList", None),
    ("rytp.assemble.cutlist", "Slot", None),
    ("rytp.assemble.cutlist", "Alternative", None),
    ("rytp.assemble.cutlist", "Substitution", None),
    ("rytp.assemble.cutlist", "CutlistError", None),
    ("rytp.render.ffmpeg", "Tools", None),
    ("rytp.tui.palette", "palette_entries", ("commands",)),
    ("rytp.tui.palette", "parse_arguments", ("cmd", "text")),
    ("rytp.tui.palette", "cli_invocation", ("cmd", "values")),
    ("rytp.tui.screens.search", "SearchScreen", None),
    ("rytp.tui.screens.transcript", "TranscriptVideosScreen", None),
    ("rytp.tui.screens.speakers", "SpeakerVideosScreen", None),
)


@pytest.mark.parametrize(("module", "attribute", "params"), CONSUMED)
def test_a_declared_dependency_exists_with_the_declared_shape(
    module: str, attribute: str, params: tuple[str, ...] | None
) -> None:
    """Every plan's `Interfaces / Consumes` block is a promise from another
    part. This is the one place all of them are checked at once."""
    obj = getattr(importlib.import_module(module), attribute)
    if params is None:
        return
    signature = inspect.signature(obj)
    declared = set(signature.parameters)
    assert set(params) <= declared, (
        f"{module}.{attribute} takes {sorted(declared)}; consumers expect "
        f"{sorted(set(params) - declared)} as well"
    )


def test_the_speaker_filter_keeps_the_two_fields_every_consumer_reads() -> None:
    """contracts §5 rule 3: the expansion to `video_speakers.id` happens inside
    the resolver, so consumers filter on `video_speaker_ids` directly."""
    import dataclasses

    from rytp.commands import SpeakerFilter

    fields = {f.name for f in dataclasses.fields(SpeakerFilter)}
    assert fields == {"video_speaker_ids", "description"}
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_consistency_schema.py -q`

Expected: green if Parts 1–7 landed as planned. The likely reds and their fixes:

- `test_the_indexes_are_exactly_the_ones_the_contract_lists` naming an index a
  part added for itself. Contracts §3 says later parts add no DDL; either the
  index belongs in Part 1's migrations or it should not exist. Decide, move it,
  and update the set here if it is now part of the contract.
- `test_models_imports_snowballstemmer_lazily`. Move the import into
  `stem_text`'s body.
- `test_a_declared_dependency_exists_with_the_declared_shape` naming a renamed
  export. The consumer's expectation is what the plans agreed; rename back.

- [ ] **Step 3: Lint**

Run: `python -m ruff check tests/test_consistency_schema.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add tests/test_consistency_schema.py
git commit -m "test: pin the schema, the fixtures and every declared dependency"
```

---

### Task 5: The navigation map

Three parts each appended a key and an action to `rytp/tui/app.py` without seeing
each other's. It happened to work, because Part 4 read Part 7's plan and left
`f5` alone — which is the good outcome of a process that has no guard rail.

This task adds the guard rail: one table of screens, from which the bindings, the
home view's navigation strip and the help screen are all derived, plus a test
that discovers collisions by **importing the screens and reading their
`BINDINGS`** rather than by consulting a hand-written reserved-key list. A
reserved-key list is the thing that goes stale; the mapper's own code is not.

The module imports no Textual at the top. Screen classes are loaded inside the
functions that need them, which keeps the table itself testable and keeps the
help screen from dragging in the search, transcript, speaker, jobs and cut-list
modules just to start the app.

**Files:**
- Create: `rytp/tui/navigation.py`
- Test: `tests/test_tui_navigation.py`

**Interfaces:**
- Consumes: nothing at module level. Lazily, inside functions: the five screen classes named in `SCREENS`.
- Produces:
  - `ScreenEntry(key, id, title, summary, module, factory, bare_letters=False)` — frozen.
  - `SCREENS: tuple[ScreenEntry, ...]`, `APP_ACTIONS: tuple[tuple[str, str, str], ...]`, `BACK_KEY: str`
  - `screen_by_id(screen_id) -> ScreenEntry`, `screen_by_key(key) -> ScreenEntry | None`
  - `load_screen_class(entry) -> type`
  - `binding_rows() -> tuple[tuple[str, str, str], ...]` — `(key, action, description)`, Textual-free
  - `home_strip() -> str`
  - `help_rows() -> tuple[tuple[str, str, str], ...]` — `(key, where, what)`
  - `app_keys() -> frozenset[str]`, `screen_binding_keys() -> dict[str, frozenset[str]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_navigation.py`:

```python
"""One table of screens, and the collision check that reads the real code.

design §10 wants the TUI "aligned with" the CLI. Alignment between screens is
a separate problem, and it was solved so far by three plan authors reading each
other. This module is that reading, automated.
"""

from __future__ import annotations

import re

import pytest
from textual.screen import Screen

from rytp.tui import navigation as N

ENTRIES = list(N.SCREENS)
IDS = [entry.id for entry in ENTRIES]


def test_every_screen_is_named_once() -> None:
    assert len(IDS) == len(set(IDS))
    assert set(IDS) == {"speakers", "search", "transcripts", "jobs", "cutlists"}


def test_every_screen_has_a_title_and_a_summary() -> None:
    for entry in ENTRIES:
        assert entry.title.strip(), entry.id
        assert entry.summary.strip().endswith("."), entry.id


def test_no_key_is_claimed_twice_at_app_level() -> None:
    keys = [key for key, _action, _label in N.APP_ACTIONS] + [e.key for e in ENTRIES]
    assert len(keys) == len(set(keys)), sorted(keys)


def test_app_level_keys_are_function_keys_or_control_chords() -> None:
    """The palette filter has focus on the home view and must keep receiving
    typed text, so no app-level key may be a bare letter."""
    for key in N.app_keys():
        assert re.fullmatch(r"f\d{1,2}|ctrl\+\w+", key), key


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_named_screen_really_exists(entry: N.ScreenEntry) -> None:
    for cls in N.load_screen_classes(entry):
        assert isinstance(cls, type)
        assert issubclass(cls, Screen), f"{entry.module}: {cls}"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_every_screen_goes_back_the_same_way(entry: N.ScreenEntry) -> None:
    """Rule 2 of the binding scheme: home is the hub, escape returns to it."""
    for cls in N.load_screen_classes(entry):
        keys = {binding.key for binding in cls.BINDINGS}
        assert N.BACK_KEY in keys, f"{cls.__name__} has no {N.BACK_KEY} binding"


def test_the_pushed_screens_are_the_ones_with_the_bindings() -> None:
    """Three of the five entries are pickers. `f5` is on `SpeakerMapperScreen`,
    not on `SpeakerVideosScreen`, and eleven cut-list keys are on
    `CutlistScreen`, not on the picker. A collision test that only read the
    entry screens would find nothing and pass, which is worse than no test."""
    keys = N.screen_binding_keys()
    assert "f5" in keys["speakers"], "the mapper's suggestions key is invisible"
    assert "ctrl+s" in keys["cutlists"], "the editor's save key is invisible"
    assert len(keys["cutlists"]) >= 10


def test_one_key_means_one_thing_wherever_it_is_bound() -> None:
    """The `-s` finding, indoors. A key bound on more than one screen must
    describe the same action, or muscle memory is wrong half the time."""
    meanings: dict[str, set[str]] = {}
    for entry in ENTRIES:
        for cls in N.load_screen_classes(entry):
            for binding in cls.BINDINGS:
                meanings.setdefault(binding.key, set()).add(
                    (binding.description or binding.action).lower()
                )
    clashes = {key: sorted(what) for key, what in meanings.items() if len(what) > 1}
    assert not clashes, f"one key, several meanings: {clashes}"


def test_no_app_level_key_is_shadowed_by_a_screen() -> None:
    """The reason `f5` is not an app-level key, derived rather than declared.

    Part 7's mapper binds `f5` to toggle suggestions. An app-level `f5` would
    still work — screen bindings win — but the footer would show two meanings
    for one key, which is worse than not having the key. A hand-maintained
    reserved list would go stale the moment a screen changed a binding; this
    reads the screens.
    """
    app = N.app_keys() - {N.BACK_KEY}
    clashes = {
        screen_id: sorted(app & keys)
        for screen_id, keys in N.screen_binding_keys().items()
        if app & keys
    }
    assert not clashes, (
        f"app-level keys shadowed by a screen: {clashes}. Move the app-level "
        f"binding; the screen's key is the more specific one."
    )


def test_a_screen_with_a_text_box_binds_no_bare_letter() -> None:
    """Part 7's rule, kept and scoped. The speaker mapper and the search screen
    both focus an `Input` by default; a bare `a` there would eat a keystroke
    the user meant to type."""
    for entry in ENTRIES:
        if entry.bare_letters:
            continue
        keys = N.screen_binding_keys()[entry.id]
        bare = sorted(key for key in keys if re.fullmatch(r"[a-z]", key))
        assert not bare, f"{entry.id} binds bare letters {bare} but has a focused Input"


def test_the_bindings_the_app_uses_come_from_this_table() -> None:
    rows = N.binding_rows()
    assert ("f1", "help", "Help") in rows
    assert ("ctrl+q", "quit", "Quit") in rows
    for entry in ENTRIES:
        assert (entry.key, f"open('{entry.id}')", entry.title) in rows


def test_the_home_strip_names_every_screen_and_its_key() -> None:
    strip = N.home_strip()
    for entry in ENTRIES:
        assert entry.key.upper() in strip, entry.id
        assert entry.title in strip, entry.id
    assert "F1" in strip


def test_help_covers_the_app_bindings_and_every_screen_binding() -> None:
    rows = N.help_rows()
    keys = {(key, where) for key, where, _what in rows}
    for key, _action, _label in N.APP_ACTIONS:
        assert (key, "anywhere") in keys, key
    for entry in ENTRIES:
        assert (entry.key, "anywhere") in keys, entry.key
        for screen_key in N.screen_binding_keys()[entry.id]:
            assert (screen_key, entry.title) in keys, (entry.id, screen_key)


def test_looking_a_screen_up_by_a_name_that_is_not_one_is_an_error() -> None:
    from rytp.models import NotFoundError

    with pytest.raises(NotFoundError, match="nonsense"):
        N.screen_by_id("nonsense")


def test_the_table_imports_no_textual_and_no_screen() -> None:
    """Keeping the map import-light is what lets the help screen list every
    binding without the app importing five workflow packages to start."""
    import subprocess
    import sys

    from tests.consistency import repo_root

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.tui.navigation; "
            "print(sorted(m for m in sys.modules if m.startswith('rytp.tui.screens')))",
        ],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_navigation.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.navigation'`.

- [ ] **Step 3: Write `rytp/tui/navigation.py`**

```python
"""The screen map (design §10).

Parts 1, 4 and 7 each contribute screens and each appended a key and an action
to the app by hand. This table is the one place a screen is declared: the app's
`BINDINGS`, the home view's navigation strip and the help screen are all
generated from it, so a screen cannot exist without being reachable, listed and
documented.

Deliberately free of Textual imports. Screen classes are loaded inside the
functions that need them, which keeps the table unit-testable and keeps
`rytp tui` from importing the search, index, diarize, assemble and queue
packages before it has drawn anything.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Final

from rytp.models import NotFoundError

__all__ = [
    "APP_ACTIONS",
    "BACK_KEY",
    "SCREENS",
    "ScreenEntry",
    "app_keys",
    "binding_rows",
    "help_rows",
    "home_strip",
    "load_screen_class",
    "load_screen_classes",
    "screen_binding_keys",
    "screen_by_id",
    "screen_by_key",
]

#: How every screen returns to the home view. One key, one meaning.
BACK_KEY: Final = "escape"


@dataclass(frozen=True)
class ScreenEntry:
    """One pushable screen, and everything both surfaces need to describe it."""

    key: str
    id: str
    title: str
    summary: str
    module: str
    factory: str
    #: Screens the entry screen pushes in turn, in the same module. Three of
    #: the five are pickers — a video, a video, a cut list — and every
    #: interesting binding lives on the screen they push. Listing them is what
    #: makes the collision test and the help screen see `f5` on the mapper,
    #: `ctrl+p` on the transcript reader and eleven keys on the cut-list
    #: editor; without it both would read the pickers, find nothing, and pass.
    children: tuple[str, ...] = ()
    #: True only for a screen whose default focus is not a text box, and which
    #: may therefore bind bare letters. The mapper and the search screen both
    #: focus an `Input`, so they may not.
    bare_letters: bool = False


#: design §10 lists what the TUI must offer: "browsing videos, reading
#: transcripts, searching, playing hits, editing cut lists, adding to the queue,
#: watching job progress", plus speaker assignment, which it calls the one
#: genuinely interactive task. Browsing videos is the palette itself
#: (`videos list` on the home view); the other five are here.
SCREENS: Final[tuple[ScreenEntry, ...]] = (
    ScreenEntry(
        key="f3",
        id="speakers",
        title="Speakers",
        summary="Name the voices in a diarized video.",
        module="rytp.tui.screens.speakers",
        factory="SpeakerVideosScreen",
        children=("SpeakerMapperScreen",),
    ),
    ScreenEntry(
        key="f4",
        id="search",
        title="Search",
        summary="Find where a word or phrase was said, and play it.",
        module="rytp.tui.screens.search",
        factory="SearchScreen",
    ),
    ScreenEntry(
        key="f6",
        id="transcripts",
        title="Transcripts",
        summary="Read one video's transcript.",
        module="rytp.tui.screens.transcript",
        factory="TranscriptVideosScreen",
        children=("TranscriptScreen",),
    ),
    ScreenEntry(
        key="f7",
        id="jobs",
        title="Queue",
        summary="Watch what is queued, running, done and failed.",
        module="rytp.tui.screens.jobs",
        factory="JobsScreen",
    ),
    ScreenEntry(
        key="f8",
        id="cutlists",
        title="Cut lists",
        summary="Swap a fragment for one of its alternatives, and fix timings.",
        module="rytp.tui.screens.cutlist",
        factory="CutlistPickerScreen",
        children=("CutlistScreen",),
        # No `Input` has focus on either cut-list screen, so bare letters are
        # free and the editing verbs read better as `s`, `a`, `g`.
        bare_letters=True,
    ),
)

#: App-level keys that are not a screen. `f5` is absent on purpose and stays
#: absent: Part 7's mapper binds it. Nothing here says so — the test derives it
#: from the mapper's own BINDINGS, because a comment cannot fail.
APP_ACTIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("f1", "help", "Help"),
    ("f2", "focus_filter", "Filter"),
    ("ctrl+q", "quit", "Quit"),
)


def screen_by_id(screen_id: str) -> ScreenEntry:
    """Look a screen up by id, naming the alternatives when it is missing."""
    for entry in SCREENS:
        if entry.id == screen_id:
            return entry
    known = ", ".join(entry.id for entry in SCREENS)
    raise NotFoundError(f"no screen called {screen_id!r}; the TUI has: {known}")


def screen_by_key(key: str) -> ScreenEntry | None:
    """The screen a key opens, or None if the key is not a screen key."""
    return next((entry for entry in SCREENS if entry.key == key), None)


def load_screen_class(entry: ScreenEntry) -> type:
    """Import one screen's module and return the class the key opens.

    Lazy on purpose: opening the search screen should not cost the cut-list
    parser, and starting the app should cost neither.
    """
    module = importlib.import_module(entry.module)
    return getattr(module, entry.factory)  # type: ignore[no-any-return]


def load_screen_classes(entry: ScreenEntry) -> tuple[type, ...]:
    """The entry screen and every screen it pushes, from the same module.

    Three of the five entries are pickers whose only binding is `escape`;
    everything a user actually presses is on the screen they push. Anything
    that reasons about bindings must look at both, or it reasons about nothing.
    """
    module = importlib.import_module(entry.module)
    names = (entry.factory, *entry.children)
    return tuple(getattr(module, name) for name in names)


def binding_rows() -> tuple[tuple[str, str, str], ...]:
    """Every app-level binding as ``(key, action, description)``.

    Plain tuples rather than Textual `Binding` objects, so this module stays
    import-light; `rytp/tui/app.py` does the one-line conversion.
    """
    rows = list(APP_ACTIONS)
    rows.extend((entry.key, f"open('{entry.id}')", entry.title) for entry in SCREENS)
    return tuple(rows)


def app_keys() -> frozenset[str]:
    """Every key bound at app level."""
    return frozenset(key for key, _action, _label in binding_rows())


def screen_binding_keys() -> dict[str, frozenset[str]]:
    """Screen id -> the keys that screen binds itself.

    Imports every screen, which is why only the tests and the help screen call
    it.
    """
    return {
        entry.id: frozenset(
            binding.key
            for screen in load_screen_classes(entry)
            for binding in screen.BINDINGS
        )
        for entry in SCREENS
    }


def home_strip() -> str:
    """The one line the home view shows above the palette."""
    parts = [f"{key.upper()} {label}" for key, _action, label in APP_ACTIONS[:1]]
    parts.extend(f"{entry.key.upper()} {entry.title}" for entry in SCREENS)
    return "  ·  ".join(parts)


def help_rows() -> tuple[tuple[str, str, str], ...]:
    """Every binding in the application as ``(key, where, what)``.

    Generated, so a screen that adds a key documents itself.
    """
    rows: list[tuple[str, str, str]] = [
        (key, "anywhere", label) for key, _action, label in APP_ACTIONS
    ]
    rows.extend((entry.key, "anywhere", f"Open {entry.title}") for entry in SCREENS)
    for entry in SCREENS:
        for screen in load_screen_classes(entry):
            for binding in screen.BINDINGS:
                rows.append(
                    (binding.key, entry.title, binding.description or binding.action)
                )
    return tuple(rows)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_tui_navigation.py -q`
Expected: the tests naming `jobs` and `cutlists` fail with `ModuleNotFoundError`,
because Tasks 9 and 11 create those screens. Everything else passes.

**Do not stub the two screens to make this green.** Mark exactly the four
parametrized cases and the two aggregate tests that need them, using a module-
level guard at the top of the test file:

```python
def _screens_exist() -> bool:
    try:
        for entry in N.SCREENS:
            N.load_screen_class(entry)
    except ModuleNotFoundError:
        return False
    return True


needs_all_screens = pytest.mark.skipif(
    not _screens_exist(), reason="jobs and cut-list screens arrive in tasks 9 and 11"
)
```

and apply `@needs_all_screens` to `test_every_named_screen_really_exists`,
`test_every_screen_goes_back_the_same_way`,
`test_no_app_level_key_is_shadowed_by_a_screen`,
`test_a_screen_with_a_text_box_binds_no_bare_letter` and
`test_help_covers_the_app_bindings_and_every_screen_binding`. Task 13 asserts
that the skip is gone.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/tui/navigation.py tests/test_tui_navigation.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/tui/navigation.py tests/test_tui_navigation.py
git commit -m "feat(tui): declare every screen in one table the bindings come from"
```

---

### Task 6: The app shell — a home view that says where you can go, and help

Part 1 built a palette, an argument line and a result table, and said workflow
screens belong to later parts. Parts 4 and 7 added three, each by appending a
binding and an action. What is missing is the thing that makes them one
application: something on screen that says the other screens exist.

**Home is the palette, upgraded.** Not a new pushed screen: `#palette` lives on
the app root and all seven of Part 1's app tests drive it there. Moving it would
cost those tests and buy nothing — the palette *is* the natural home, because it
is the list of everything the tool can do.

Three additions. A navigation strip above the palette, generated from the map.
A help screen on `f1` listing every binding in the application, also generated.
And `BINDINGS` itself becomes derived, so the next screen needs no edit here.

**Files:**
- Modify: `rytp/tui/app.py` (contracts §2 gives the app shell to Part 8)
- Create: `rytp/tui/screens/help.py`
- Test: `tests/test_tui_shell.py`

**Interfaces:**
- Consumes: `rytp.tui.navigation.{SCREENS, BACK_KEY, binding_rows, help_rows, home_strip, load_screen_class, screen_by_id}`, Part 1's `RytpApp` internals (`_db`, `_commands`, `refresh_palette`, `set_status`, `run_selected`).
- Produces: `RytpApp.action_open(screen_id: str)`, `RytpApp.action_help()`, `RytpApp.action_speakers/search/transcripts()` (kept as one-line wrappers, because Parts 4 and 7 published them), `rytp.tui.screens.help.HelpScreen(...)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_shell.py`:

```python
"""The shell that makes five screens one application."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from textual.widgets import DataTable, OptionList, Static

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.tui import navigation as N
from rytp.tui.app import RytpApp


def noop(db: Database, **kwargs: Any) -> CommandResult:
    return CommandResult(message="ok")


SAMPLE = {
    "videos.list": Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(Param("limit", int, "Rows.", default=50),),
        handler=noop,
    ),
}


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_the_home_view_says_where_you_can_go(db: Database) -> None:
    """Design §10 lists five things the TUI must offer besides the palette.
    A user who cannot see that they exist does not have them."""

    async def body(app: RytpApp, pilot: Any) -> None:
        strip = app.query_one("#home-nav", Static)
        rendered = str(strip.renderable)
        for entry in N.SCREENS:
            assert entry.title in rendered
            assert entry.key.upper() in rendered

    drive(db, body)


def test_the_palette_is_still_on_the_app_root(db: Database) -> None:
    """Part 1's seven app tests drive `#palette` here. Keeping it is the whole
    reason home is the palette rather than a pushed screen."""

    async def body(app: RytpApp, pilot: Any) -> None:
        assert app.query_one("#palette", OptionList) is not None
        assert app.query_one("#results", DataTable) is not None

    drive(db, body)


def test_every_app_binding_comes_from_the_navigation_map(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        bound = {(b.key, b.action) for b in app.BINDINGS}
        for key, action, _label in N.binding_rows():
            assert (key, action) in bound, (key, action)

    drive(db, body)


@pytest.mark.parametrize("entry", list(N.SCREENS), ids=[e.id for e in N.SCREENS])
def test_each_screen_key_opens_that_screen(db: Database, entry: N.ScreenEntry) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press(entry.key)
        await pilot.pause()
        assert isinstance(app.screen, N.load_screen_class(entry))

    drive(db, body)


def test_escape_comes_back_to_the_home_view(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f7")
        await pilot.pause()
        assert app.screen is not app.screen_stack[0]
        await pilot.press(N.BACK_KEY)
        await pilot.pause()
        assert app.screen is app.screen_stack[0]
        assert app.query_one("#palette", OptionList) is not None

    drive(db, body)


def test_help_lists_every_binding_in_the_application(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f1")
        await pilot.pause()
        table = app.screen.query_one("#help-bindings", DataTable)
        shown = {
            (str(table.get_cell_at((row, 0))), str(table.get_cell_at((row, 1))))
            for row in range(table.row_count)
        }
        for key, where, _what in N.help_rows():
            assert (key, where) in shown, (key, where)

    drive(db, body)


def test_help_closes_the_same_way_every_screen_does(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f1")
        await pilot.pause()
        await pilot.press(N.BACK_KEY)
        await pilot.pause()
        assert app.screen is app.screen_stack[0]

    drive(db, body)


def test_the_actions_parts_four_and_seven_published_still_work(db: Database) -> None:
    """Their plans list `RytpApp.action_speakers()` as produced. Keeping the
    three names is one line each and costs nothing."""

    async def body(app: RytpApp, pilot: Any) -> None:
        for action, screen_id in (
            (app.action_speakers, "speakers"),
            (app.action_search, "search"),
            (app.action_transcripts, "transcripts"),
        ):
            action()
            await pilot.pause()
            assert isinstance(app.screen, N.load_screen_class(N.screen_by_id(screen_id)))
            await pilot.press(N.BACK_KEY)
            await pilot.pause()

    drive(db, body)


def test_opening_a_screen_that_does_not_exist_is_reported_not_raised(
    db: Database,
) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.action_open("nonsense")
        await pilot.pause()
        assert "nonsense" in app.status_text
        assert app.is_running

    drive(db, body)


def test_starting_the_app_imports_no_workflow_package() -> None:
    """`rytp tui` must not pay for the index, assemble, diarize and render
    packages before it draws anything."""
    import subprocess
    import sys

    from tests.consistency import repo_root

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.tui.app; "
            "print(sorted(m for m in sys.modules "
            "if m.startswith(('rytp.index', 'rytp.assemble', 'rytp.render', "
            "'rytp.diarize', 'rytp.transcribe'))))",
        ],
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_shell.py -q`
Expected: failures on `#home-nav` not existing and `action_open` not existing.
The `jobs` and `cutlists` parametrized cases fail with `ModuleNotFoundError` until
Tasks 9 and 11; apply the same `needs_all_screens` guard the previous task
defined, imported from `tests/test_tui_navigation.py`.

- [ ] **Step 3: Write `rytp/tui/screens/help.py`**

```python
"""Every binding in the application, generated from the navigation map.

Nothing here is typed by hand, which is the point: a screen that adds a key
documents itself, and a key that is removed disappears from help in the same
commit.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp.tui.navigation import BACK_KEY, help_rows

__all__ = ["HelpScreen"]


class HelpScreen(Screen[None]):
    """A table of (key, where, what)."""

    DEFAULT_CSS = """
    #help-bindings { height: 1fr; }
    #help-note { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="help-bindings")
        yield Static(
            "Every screen returns here with Escape. Long-running commands are "
            "queued, never run in the TUI — press F7 to watch the queue.",
            id="help-note",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#help-bindings", DataTable)
        table.cursor_type = "row"
        table.add_columns("key", "where", "what")
        for key, where, what in help_rows():
            table.add_row(key, where, what)
```

- [ ] **Step 4: Rework `rytp/tui/app.py`**

Four edits to Part 1's file. Nothing else in it changes — the palette, the
argument line, the result table, the error handling and `run_tui` stay exactly as
they are.

Replace the `BINDINGS` class variable:

```python
    # Derived from the navigation map (Part 8). A new screen is added there,
    # not here, so it cannot exist without a key, a home-strip entry and a
    # help line. `list[BindingType]` because App declares the wider type and
    # list is invariant.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key, action, description)
        for key, action, description in binding_rows()
    ]
```

with these imports added at the top:

```python
from rytp.tui.navigation import BACK_KEY, binding_rows, home_strip, load_screen_class, screen_by_id
```

Add the navigation strip to `compose`, immediately after `yield Header()`:

```python
        yield Static(home_strip(), id="home-nav")
```

and add the actions, replacing the three hand-written ones Parts 4 and 7 added:

```python
    # -- navigation --------------------------------------------------

    def action_open(self, screen_id: str) -> None:
        """Push one of the screens named in the navigation map.

        One action for every screen, parameterised by id, so adding a screen
        is one row in `navigation.SCREENS` and no code here. The screen class
        is imported only now, which is what keeps `rytp tui` from loading the
        index, assemble, diarize and render packages at start-up.
        """
        try:
            entry = screen_by_id(screen_id)
            screen_class = load_screen_class(entry)
        except (NotFoundError, ModuleNotFoundError) as exc:
            self.set_status(str(exc))
            return
        self.push_screen(screen_class(self._db))

    def action_help(self) -> None:
        """Every binding in the application, on the key people try first."""
        from rytp.tui.screens.help import HelpScreen

        self.push_screen(HelpScreen())

    # The three names Parts 4 and 7 published in their Interfaces blocks.
    def action_speakers(self) -> None:
        self.action_open("speakers")

    def action_search(self) -> None:
        self.action_open("search")

    def action_transcripts(self) -> None:
        self.action_open("transcripts")
```

`NotFoundError` comes from `rytp.models`, which `app.py` already imports
`RytpError` from; add it to that import.

If `self.push_screen(screen)` complains that the result is not awaitable, or
that it must be awaited, follow the note Parts 4 and 7 both left: the installed
Textual version decides, and the call is the only version-sensitive line.

- [ ] **Step 5: Run the shell tests and Part 1's**

Run: `python -m pytest tests/test_tui_shell.py tests/test_tui_app.py tests/test_palette.py -q`
Expected: PASS. Part 1's seven app tests must still pass untouched — if one of
them fails, the shell moved something it should not have.

- [ ] **Step 6: Run the other parts' screen tests**

Run: `python -m pytest tests/test_tui_search.py tests/test_tui_transcript.py tests/test_tui_speakers.py -q`
Expected: PASS. Parts 4 and 7 drive their screens directly, so the binding rework
should be invisible to them; if it is not, the wrappers are the fix.

- [ ] **Step 7: Lint and type-check**

Run: `python -m ruff check rytp/tui tests/test_tui_shell.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 8: Commit**

```bash
git add rytp/tui/app.py rytp/tui/screens/help.py tests/test_tui_shell.py
git commit -m "feat(tui): a home view that names every screen, and generated help"
```

---

### Task 7: Starting long work from the TUI, without becoming a worker

Review finding 3.3, left open as a scope decision: "The palette lists those
commands but hands you the command line rather than running them." This task
makes the decision and implements it.

**The TUI enqueues; a separate process works.** Design §5 puts long-running work
in a worker process and design §10 repeats it — "The worker itself stays a
separate process." So the TUI never calls a transcriber. What it can do, and
what closes the gap, is write the job row: `transcribe.run --enqueue`,
`index.build --enqueue` and `render.run --queue` already exist as flags on the
commands themselves, put there by Parts 3, 4 and 6 for exactly this reason.
Running a handler with its enqueue flag forced true is a single `INSERT` and
returns immediately. It is not running the work.

Three spellings exist for one idea, which is itself a finding: `--enqueue` on
four commands, `--queue` on `render.run`, and `speakers.enqueue` as a sibling
command of its own. This module accepts the first two so the feature works
whichever survives, and Task 2's vocabulary test is what argues for `--enqueue`.

**The classification table is the risk.** A hand-written list of "commands that
cannot be queued" is precisely the shape of thing that went stale in the reviews.
So its domain is derived: every `long_running` command must have either an
enqueue flag or a row here, and one that has neither fails the suite by name.

**Files:**
- Create: `rytp/tui/enqueue.py`
- Modify: `rytp/tui/app.py` (`run_selected` consults it)
- Test: `tests/test_tui_enqueue.py`

**Interfaces:**
- Consumes: `rytp.commands.{COMMANDS, Command}`, `rytp.tui.palette.cli_invocation`.
- Produces:
  - `ENQUEUE_FLAGS: tuple[str, ...]`, `FOREGROUND_ONLY: dict[str, str]`
  - `EnqueuePlan(command: str, flag: str, values: dict[str, Any])` — frozen
  - `enqueue_plan(cmd, values) -> EnqueuePlan | None`
  - `foreground_hint(cmd, values) -> str`
  - `unclassified_long_running(commands) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_enqueue.py`:

```python
"""Queue it, do not run it (design §5, §10)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from textual.widgets import Static

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.tui import enqueue as E
from rytp.tui.app import RytpApp
from tests.consistency import load_registries

REGISTERED, _ = load_registries()

calls: list[dict[str, Any]] = []


def queueing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(dict(kwargs))
    if not kwargs.get("enqueue"):  # pragma: no cover - the bug this guards
        raise AssertionError("the TUI ran long work inline")
    return CommandResult(message="queued transcribe job 4 for video 1")


def never(db: Database, **kwargs: Any) -> CommandResult:  # pragma: no cover
    raise AssertionError("this command must never run from the TUI")


QUEUEABLE = Command(
    name="transcribe.run",
    group="transcribe",
    summary="Transcribe one video.",
    params=(
        Param("video", str, "Video id.", positional=True),
        Param("enqueue", bool, "Queue it for the worker.", default=False),
    ),
    handler=queueing,
    long_running=True,
)

FOREGROUND = Command(
    name="worker",
    group="",
    summary="Run the worker pools.",
    params=(Param("once", bool, "Drain and exit.", default=False),),
    handler=never,
    long_running=True,
)

SAMPLE = {"transcribe.run": QUEUEABLE, "worker": FOREGROUND}


def test_a_command_with_an_enqueue_flag_gets_a_plan() -> None:
    plan = E.enqueue_plan(QUEUEABLE, {"video": "VIDEO_A", "enqueue": False})
    assert plan is not None
    assert plan.command == "transcribe.run"
    assert plan.flag == "enqueue"
    assert plan.values["enqueue"] is True
    assert plan.values["video"] == "VIDEO_A"


def test_the_plan_does_not_mutate_what_the_user_typed() -> None:
    typed = {"video": "VIDEO_A", "enqueue": False}
    E.enqueue_plan(QUEUEABLE, typed)
    assert typed["enqueue"] is False


def test_the_older_spelling_is_accepted_too() -> None:
    """`render.run` calls it `--queue`. Accepting both is what lets this work
    before or after the vocabulary is unified."""
    render = Command(
        name="render.run",
        group="render",
        summary="Render a cut list.",
        params=(
            Param("name", str, "Cut list.", positional=True),
            Param("queue", bool, "Enqueue it.", default=False),
        ),
        handler=queueing,
        long_running=True,
    )
    plan = E.enqueue_plan(render, {"name": "demo", "queue": False})
    assert plan is not None and plan.flag == "queue"


def test_a_command_with_no_queued_form_gets_no_plan() -> None:
    assert E.enqueue_plan(FOREGROUND, {"once": True}) is None


def test_a_command_that_is_not_long_running_gets_no_plan() -> None:
    quick = Command(
        name="videos.list",
        group="videos",
        summary="List videos.",
        params=(Param("enqueue", bool, "Nonsense.", default=False),),
        handler=queueing,
    )
    assert E.enqueue_plan(quick, {"enqueue": False}) is None


def test_the_hint_for_a_foreground_command_says_why_and_what_to_type() -> None:
    hint = E.foreground_hint(FOREGROUND, {"once": True})
    assert "rytp worker" in hint
    assert E.FOREGROUND_ONLY["worker"] in hint


def test_every_long_running_command_is_classified() -> None:
    """The staleness guard. A new long-running command must either offer a
    queued form or say here why it cannot have one — the alternative is a
    command the TUI silently cannot start, which is the gap this task closes.
    """
    assert E.unclassified_long_running(REGISTERED) == ()


def test_the_foreground_table_names_no_command_that_does_not_exist() -> None:
    assert set(E.FOREGROUND_ONLY) <= set(REGISTERED)


def test_every_foreground_reason_is_a_sentence() -> None:
    for name, reason in E.FOREGROUND_ONLY.items():
        assert reason.strip() and not reason.endswith("."), name


# --- the wiring ------------------------------------------------------


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_running_a_queueable_command_enqueues_it(db: Database) -> None:
    calls.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("transcribe.run")
        app.run_selected("VIDEO_A")
        await pilot.pause()
        assert calls == [{"video": "VIDEO_A", "enqueue": True}]
        assert "queued" in app.status_text

    drive(db, body)


def test_a_foreground_command_still_hands_over_a_command_line(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("worker")
        app.run_selected("once=true")
        await pilot.pause()
        assert "rytp worker" in app.status_text
        assert E.FOREGROUND_ONLY["worker"] in app.status_text

    drive(db, body)


def test_the_status_line_points_at_the_queue_screen(db: Database) -> None:
    """Enqueuing without a way to watch is worse than not enqueuing."""

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("transcribe.run")
        app.run_selected("VIDEO_A")
        await pilot.pause()
        assert "F7" in app.status_text

    drive(db, body)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_enqueue.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.enqueue'`.

- [ ] **Step 3: Write `rytp/tui/enqueue.py`**

```python
"""Turning a long-running command into queued work (design §5, §10).

design §10: "Long-running work is a CLI worker process… The worker itself stays
a separate process." Part 1 honours that by refusing to run a `long_running`
command inline and printing the command line instead, which is correct and also
means the TUI cannot start an ingest.

The way out is not to relax the rule. Parts 3, 4 and 6 each gave their long
commands a flag that enqueues rather than runs — `transcribe run --enqueue`,
`index build --enqueue`, `render run --queue` — because contracts §5 already
said the TUI must not run them. Calling the handler with that flag forced true
writes one row to `jobs` and returns; the GPU is still touched only by a worker
the owner started himself.

Commands with no queued form keep Part 1's behaviour, with a reason attached.
That table's *domain* is derived from the registry rather than written down, so
a new long-running command with neither a flag nor a reason fails the suite by
name instead of quietly becoming unreachable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from rytp.commands import Command
from rytp.tui.palette import cli_invocation

__all__ = [
    "ENQUEUE_FLAGS",
    "FOREGROUND_ONLY",
    "EnqueuePlan",
    "enqueue_plan",
    "foreground_hint",
    "unclassified_long_running",
]

#: Boolean parameter names that mean "write the job row instead of doing the
#: work". `enqueue` is the spelling four of the five use and the one the
#: vocabulary test argues for; `queue` is `render.run`'s and is accepted so
#: this works either side of that rename.
ENQUEUE_FLAGS: Final[tuple[str, ...]] = ("enqueue", "queue")

#: Long-running commands with no queued form, and why. Each value is a clause
#: that reads after "cannot be queued: ". Keeping the reason beside the name
#: is what stops this becoming a list nobody can audit.
FOREGROUND_ONLY: Final[dict[str, str]] = {
    "worker": "it is the process that drains the queue",
    "channel.sync": "enumerating a channel is one network call, not per-video work",
    "fetch-video": "this is the do-it-now path; `rytp ingest` is its queued twin",
    "search.play": "playback is interactive — press F4 and play the hit there",
    "search.export": "it writes files into the directory you name, now",
    "transcribe.compare": "it is a measurement you read, not a batch",
    "speakers.diarize": "queue it with `speakers enqueue`, which is in the palette",
    "speakers.embed": "queue it with `speakers enqueue`, which embeds as it diarizes",
}


@dataclass(frozen=True)
class EnqueuePlan:
    """How to ask for one long-running command without running it."""

    command: str
    flag: str
    values: dict[str, Any]


def _flag_for(cmd: Command) -> str | None:
    """The name of this command's enqueue flag, if it has one."""
    boolean = {param.name for param in cmd.params if param.type is bool}
    return next((flag for flag in ENQUEUE_FLAGS if flag in boolean), None)


def enqueue_plan(cmd: Command, values: Mapping[str, Any]) -> EnqueuePlan | None:
    """The values to call this command's handler with, to queue it.

    Returns None when there is nothing to queue — either the command is not
    long-running, or it has no queued form and belongs in
    :data:`FOREGROUND_ONLY`.
    """
    if not cmd.long_running:
        return None
    flag = _flag_for(cmd)
    if flag is None:
        return None
    return EnqueuePlan(command=cmd.name, flag=flag, values={**values, flag: True})


def foreground_hint(cmd: Command, values: Mapping[str, Any]) -> str:
    """What to say about a long command that cannot be queued."""
    reason = FOREGROUND_ONLY.get(cmd.name, "it runs in the foreground")
    return f"{reason}; run it in a terminal: {cli_invocation(cmd, values)}"


def unclassified_long_running(commands: Mapping[str, Command]) -> tuple[str, ...]:
    """Long-running commands that neither queue nor explain why they cannot.

    The whole point of this function is to fail a test. A command that reaches
    this list is one the TUI lists and cannot start, which is the gap review
    finding 3.3 recorded.
    """
    return tuple(
        sorted(
            name
            for name, cmd in commands.items()
            if cmd.long_running
            and _flag_for(cmd) is None
            and name not in FOREGROUND_ONLY
        )
    )
```

- [ ] **Step 4: Wire it into `rytp/tui/app.py`**

Replace the `if cmd.long_running:` branch of `run_selected` — Part 1's version
printed a command line for every long command — with:

```python
        if cmd.long_running:
            plan = enqueue_plan(cmd, values)
            if plan is None:
                self.set_status(foreground_hint(cmd, values))
                return
            # Not running the work: the handler writes one row to `jobs` and
            # returns. design §5 keeps the worker a separate process.
            try:
                result = cmd.handler(self._db, **plan.values)
            except RytpError as exc:
                self.set_status(str(exc))
                return
            self.show_result(result)
            self.set_status(f"{result.message or 'queued'} — F7 to watch the queue")
            return
```

and add the import:

```python
from rytp.tui.enqueue import enqueue_plan, foreground_hint
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_tui_enqueue.py tests/test_tui_app.py -q`
Expected: PASS. Part 1's `test_a_long_running_command_is_not_run_inline` uses
`channel.sync`, which is in `FOREGROUND_ONLY`, so it still sees a command line.

`test_every_long_running_command_is_classified` may be red. If it names a
command, decide which side it belongs on: give it an `enqueue` flag in its
owning part if there is a job kind that does its work, or add a row to
`FOREGROUND_ONLY` with a clause saying why there is not.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/tui tests/test_tui_enqueue.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/tui/enqueue.py rytp/tui/app.py tests/test_tui_enqueue.py
git commit -m "feat(tui): queue long-running work instead of handing over a command line"
```

---

### Task 8: The queue view, headless

Design §10 lists "watching job progress" as something the TUI must offer, and
Part 2 built everything it needs: `jobs list` already renders id, kind, target,
state, pool, attempts, backoff, error and — since review finding 3.2 — the
handler's `note`. `jobs stats` already counts throttling so it "shows up in
`rytp jobs stats` rather than as a mystery stall".

So this view is deliberately a **thin adapter over the registered commands**, not
a second implementation. It calls `jobs.list`, `jobs.stats`, `jobs.retry`,
`jobs.cancel`, `queue.pause` and `queue.resume` through the registry, which means
the two surfaces cannot disagree about what a job looks like — which is design
§10's whole claim. What the view adds is what a table cannot do from a command
line: cycling a filter with one key, refreshing on a timer, and putting the note
of the highlighted job where it can be read in full.

Its filter cycles are **derived from the commands' own `choices`**, so a new job
state or a fourth pool appears here without an edit. A hardcoded list of job
states is the same defect as the hardcoded list of job kinds the review found.

**Files:**
- Modify: `rytp/constants.py` (append Part 8's section at the very end)
- Create: `rytp/tui/jobs_view.py`
- Test: `tests/test_tui_jobs.py` (the headless half)

**Interfaces:**
- Consumes: `rytp.commands.{resolve, CommandResult}`, `rytp.db.Database`, `rytp.jobs.queue.{get_job, is_paused}`, `rytp.constants.{TUI_QUEUE_REFRESH_S, TUI_QUEUE_ROW_LIMIT}`.
- Produces:
  - `JobsView(db, *, limit=C.TUI_QUEUE_ROW_LIMIT)` with attributes `state`, `pool`, `columns`, `rows`, `status`
  - `refresh() -> None`, `cycle_state() -> str`, `cycle_pool() -> str`
  - `job_id_at(index) -> int | None`, `note_at(index) -> str`
  - `retry_at(index) -> str`, `cancel_at(index) -> str`, `retry_all_failed() -> str`
  - `toggle_pause() -> str`

- [ ] **Step 1: Append the constants**

At the very end of `rytp/constants.py`:

```python
# ---------------------------------------------------------------------------
# Part 8 — the TUI shell and its two screens (design §10)
# ---------------------------------------------------------------------------
#: How often the queue screen re-reads the jobs table, in seconds. The worker
#: is a separate process (design §5), so the only way to see progress is to
#: look again. Two seconds is short enough to feel live on a download that
#: takes a minute and long enough that a full-corpus queue is not re-rendered
#: continuously.
TUI_QUEUE_REFRESH_S: float = 2.0

#: Rows the queue screen asks for. design §11 M2 is ~1,600 videos and the
#: chain is five kinds per video, so the table is never scrolled to the end;
#: what matters is what is running and what failed, and the filters get you
#: there faster than scrolling would.
TUI_QUEUE_ROW_LIMIT: int = 200

#: Milliseconds one keystroke moves a cut-list boundary. design §8 prefers
#: hand-edited timings "over clever heuristics", and 40 ms is one caption
#: grid step and about one video frame at 25 fps — small enough to be a
#: nudge, large enough to hear.
TUI_CUTLIST_NUDGE_MS: int = 40

#: A coarse nudge, for when the boundary is plainly in the wrong place.
TUI_CUTLIST_COARSE_NUDGE_MS: int = 250
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_tui_jobs.py`:

```python
"""The queue view: one adapter over the commands both surfaces already share."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.jobs import queue as Q
from rytp.tui.jobs_view import JobsView
from tests.fakes import make_video


@pytest.fixture()
def queued(db: Database) -> Database:
    """One job in each interesting state, plus one carrying a note."""
    first = make_video(db, external_id="VIDEO_A", title="Первое видео")
    second = make_video(db, external_id="VIDEO_B", title="Второе видео")
    Q.enqueue(db, "download", first)
    failed = Q.enqueue(db, "captions", first)
    Q.fail(db, failed, error="HTTP Error 403")
    done = Q.enqueue(db, "download", second)
    Q.finish(db, done, note="взяты субтитры другого языка")
    return db


def test_the_columns_are_the_ones_the_cli_prints(queued: Database) -> None:
    """design §10: one definition, two surfaces. The view calls `jobs.list`
    rather than writing its own SELECT, so they cannot drift."""
    view = JobsView(queued)
    assert view.columns == (
        "id", "kind", "target", "state", "pool", "attempts", "not_before",
        "error", "note",
    )
    assert len(view.rows) == 3


def test_every_cell_is_a_string(queued: Database) -> None:
    view = JobsView(queued)
    assert all(isinstance(cell, str) for row in view.rows for cell in row)


def test_the_status_line_carries_the_stats_the_design_asks_for(queued: Database) -> None:
    """design §5: "Throttling shows up in `rytp jobs stats` rather than as a
    mystery stall"."""
    status = JobsView(queued).status
    for token in ("pending", "failed", "throttled", "notes", "paused"):
        assert token in status.lower(), status


def test_cycling_the_state_filter_walks_the_command_s_own_choices(
    queued: Database,
) -> None:
    """Derived, not listed: a new job state appears here without an edit, which
    is the fix for the same defect that made a hardcoded list of job kinds go
    stale before there was any code."""
    view = JobsView(queued)
    assert view.state == ""
    seen = [view.cycle_state() for _ in range(3)]
    assert seen[0] != ""
    assert all(value in view.state_choices for value in seen)


def test_filtering_by_state_narrows_the_rows(queued: Database) -> None:
    view = JobsView(queued)
    for _ in range(len(view.state_choices)):
        if view.state == "failed":
            break
        view.cycle_state()
    assert view.state == "failed", view.state_choices
    assert [row[1] for row in view.rows] == ["captions"]
    assert "failed" in view.status


def test_filtering_by_pool_narrows_the_rows(queued: Database) -> None:
    view = JobsView(queued)
    for _ in range(len(view.pool_choices)):
        if view.pool == "network":
            break
        view.cycle_pool()
    assert view.pool == "network", view.pool_choices
    assert {row[4] for row in view.rows} == {"network"}


def test_the_note_of_the_highlighted_job_is_readable_in_full(queued: Database) -> None:
    """The reason `jobs.note` exists: on the bulk path nobody reads stdout, so
    a warning that only fits in a truncated column is a warning nobody sees."""
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "done")
    assert view.note_at(index) == "взяты субтитры другого языка"


def test_a_job_with_no_note_reads_as_empty(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "pending")
    assert view.note_at(index) == ""


def test_retrying_the_highlighted_job_revives_it(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "failed")
    job_id = view.job_id_at(index)
    assert job_id is not None
    message = view.retry_at(index)
    assert "retried" in message
    assert Q.get_job(queued, job_id).state == "pending"


def test_cancelling_the_highlighted_job_drops_it(queued: Database) -> None:
    view = JobsView(queued)
    index = next(i for i, row in enumerate(view.rows) if row[3] == "pending")
    job_id = view.job_id_at(index)
    assert job_id is not None
    view.cancel_at(index)
    assert Q.get_job(queued, job_id).state == "cancelled"


def test_retrying_everything_failed_takes_one_key(queued: Database) -> None:
    view = JobsView(queued)
    assert "retried" in view.retry_all_failed()
    assert not [row for row in view.rows if row[3] == "failed"]


def test_pausing_and_resuming_round_trips(queued: Database) -> None:
    view = JobsView(queued)
    assert "paused" in view.toggle_pause().lower()
    assert Q.is_paused(queued) is True
    assert "resumed" in view.toggle_pause().lower()
    assert Q.is_paused(queued) is False


def test_an_index_off_the_end_of_the_table_is_not_an_error(queued: Database) -> None:
    """A cursor on an empty table is a real state, not an exception."""
    view = JobsView(queued)
    assert view.job_id_at(99) is None
    assert view.note_at(99) == ""
    assert "no job" in view.retry_at(99).lower()


def test_an_empty_queue_says_so_rather_than_showing_nothing(db: Database) -> None:
    view = JobsView(db)
    assert view.rows == ()
    assert "nothing queued" in view.status.lower()
```

- [ ] **Step 3: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_jobs.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.jobs_view'`.

- [ ] **Step 4: Write `rytp/tui/jobs_view.py`**

```python
"""Watching the queue (design §10, "watching job progress").

An adapter, not an implementation. Part 2's `jobs list`, `jobs stats`,
`jobs retry`, `jobs cancel` and `queue pause`/`resume` already do all of this
and are already registered, so this view calls them through the registry.
design §10's claim is that both surfaces come from one definition; a screen
that wrote its own SELECT would quietly make that false.

What it adds is what a command line cannot: filters you cycle with one key,
a refresh that keeps up with a worker in another process, and the highlighted
job's `note` shown in full rather than in a column.
"""

from __future__ import annotations

from typing import Final

from rytp import constants as C
from rytp.commands import CommandResult, resolve
from rytp.db import Database
from rytp.models import RytpError

__all__ = ["JobsView"]

#: Column positions in `jobs.list`'s result. Named rather than counted so a
#: reader can see what the view depends on; the test above pins the header.
_ID: Final = 0
_STATE: Final = 3
_NOTE: Final = 8


def _choices(command: str, param: str) -> tuple[str, ...]:
    """A parameter's declared choices, used as this view's filter cycle.

    Derived rather than restated: a new job state or a fourth pool shows up
    here with no edit. A hardcoded list of states would be the same defect as
    the hardcoded list of job kinds that went stale before any code existed.
    """
    for declared in resolve(command).params:
        if declared.name == param:
            return declared.choices or ("",)
    raise KeyError(f"{command} has no parameter {param!r}")


class JobsView:
    """What is queued, running, done and failed — and why."""

    def __init__(self, db: Database, *, limit: int = C.TUI_QUEUE_ROW_LIMIT) -> None:
        self._db = db
        self._limit = limit
        self.state: str = ""
        self.pool: str = ""
        self.state_choices: tuple[str, ...] = _choices("jobs.list", "state")
        self.pool_choices: tuple[str, ...] = _choices("jobs.list", "pool")
        self.columns: tuple[str, ...] = ()
        self.rows: tuple[tuple[str, ...], ...] = ()
        self.status: str = ""
        self.refresh()

    # -- reading -----------------------------------------------------

    def refresh(self) -> None:
        """Re-read the queue. Called on a timer and after every action."""
        listing: CommandResult = resolve("jobs.list").handler(
            self._db,
            state=self.state,
            pool=self.pool,
            kind="",
            limit=self._limit,
        )
        self.columns = listing.columns
        self.rows = listing.rows
        self.status = self._status_line()

    def _status_line(self) -> str:
        stats = resolve("jobs.stats").handler(self._db)
        flat = {name: value for name, value in stats.rows}
        parts = [
            f"{name} {value}"
            for name, value in flat.items()
            if value not in ("", "0") or name in ("paused", "throttled")
        ]
        where = []
        if self.state:
            where.append(f"state={self.state}")
        if self.pool:
            where.append(f"pool={self.pool}")
        shown = f"{len(self.rows)} shown"
        if where:
            shown += " (" + ", ".join(where) + ")"
        if not self.rows and not self.state and not self.pool:
            shown = "nothing queued"
        # `with notes` is spelled `notes` here so one short line carries it.
        return shown + " · " + " · ".join(parts).replace("with notes", "notes")

    # -- the highlighted row -----------------------------------------

    def _row(self, index: int) -> tuple[str, ...] | None:
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def job_id_at(self, index: int) -> int | None:
        row = self._row(index)
        return int(row[_ID]) if row else None

    def note_at(self, index: int) -> str:
        """The full note, which the column can only show the start of."""
        job_id = self.job_id_at(index)
        if job_id is None:
            return ""
        found = self._db.conn.execute(
            "SELECT note FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return str(found["note"]) if found and found["note"] else ""

    # -- filters -----------------------------------------------------

    def cycle_state(self) -> str:
        self.state = self._next(self.state_choices, self.state)
        self.refresh()
        return self.state

    def cycle_pool(self) -> str:
        self.pool = self._next(self.pool_choices, self.pool)
        self.refresh()
        return self.pool

    @staticmethod
    def _next(choices: tuple[str, ...], current: str) -> str:
        position = choices.index(current) if current in choices else -1
        return choices[(position + 1) % len(choices)]

    # -- acting ------------------------------------------------------

    def retry_at(self, index: int) -> str:
        job_id = self.job_id_at(index)
        if job_id is None:
            return "no job selected"
        return self._run("jobs.retry", job_id=job_id, kind="", state="")

    def retry_all_failed(self) -> str:
        return self._run("jobs.retry", job_id=0, kind="", state="failed")

    def cancel_at(self, index: int) -> str:
        job_id = self.job_id_at(index)
        if job_id is None:
            return "no job selected"
        return self._run(
            "jobs.cancel", job_id=job_id, kind="", state="", target_id=0, dry_run=False
        )

    def toggle_pause(self) -> str:
        from rytp.jobs.queue import is_paused

        name = "queue.resume" if is_paused(self._db) else "queue.pause"
        return self._run(name)

    def _run(self, command: str, **values: object) -> str:
        """Run a registered command and refresh. Errors become a status line."""
        try:
            result = resolve(command).handler(self._db, **values)
        except RytpError as exc:
            return str(exc)
        self.refresh()
        return result.message or f"{command} done"
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_tui_jobs.py -q`
Expected: PASS.

If `jobs.retry` refuses `state=""`, read Part 2's handler: it defaults to
`"failed"` and passes `state` through to `Q.retry`. Retrying one job by id must
not be restricted by state — the id *is* the selection — so pass whatever value
that handler treats as "any". If there is none, that is a finding: record it and
pass `state="failed"`, which is the only state a person retries from a table.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/constants.py rytp/tui/jobs_view.py tests/test_tui_jobs.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/constants.py rytp/tui/jobs_view.py tests/test_tui_jobs.py
git commit -m "feat(tui): a headless queue view over the registered job commands"
```

---

### Task 9: The queue screen

Part 7's shape: the screen holds no state, reads `self.view` and writes
keystrokes into it. Six actions, each one line, plus a timer.

The timer is the only thing here that is not in the mapper: the worker is another
process, so nothing tells the TUI that a job finished. `set_interval` at
`C.TUI_QUEUE_REFRESH_S` is the whole of "watching job progress".

**Files:**
- Create: `rytp/tui/screens/jobs.py`
- Test: `tests/test_tui_jobs.py` (append the screen half)

**Interfaces:**
- Consumes: `rytp.tui.jobs_view.JobsView`, `rytp.tui.navigation.BACK_KEY`, `rytp.constants.TUI_QUEUE_REFRESH_S`.
- Produces: `JobsScreen(db)` with `.view` and the actions `refresh`, `cycle_state`, `cycle_pool`, `retry`, `retry_all`, `cancel`, `toggle_pause`.

- [ ] **Step 1: Append the failing test**

To `tests/test_tui_jobs.py`:

```python
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from textual.widgets import DataTable, Static

from rytp.tui.screens.jobs import JobsScreen


def drive_screen(
    db: Database, body: Callable[[JobsScreen, Any], Awaitable[None]]
) -> None:
    """Mount the screen on a bare app, headlessly."""
    from textual.app import App, ComposeResult

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(JobsScreen(db))

    async def main() -> None:
        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, JobsScreen)
            await body(app.screen, pilot)

    asyncio.run(main())


def test_the_screen_shows_the_queue(queued: Database) -> None:
    def check(screen: JobsScreen, pilot: Any) -> Awaitable[None]:
        async def body() -> None:
            table = screen.query_one("#jobs-table", DataTable)
            assert table.row_count == 3
            assert [str(c.label) for c in table.columns.values()] == list(
                screen.view.columns
            )
            assert str(screen.query_one("#jobs-status", Static).renderable)

        return body()

    drive_screen(queued, lambda screen, pilot: check(screen, pilot))


def test_one_key_cycles_the_state_filter(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        before = screen.view.state
        screen.action_cycle_state()
        await pilot.pause()
        assert screen.view.state != before
        assert screen.query_one("#jobs-table", DataTable).row_count == len(
            screen.view.rows
        )

    drive_screen(queued, body)


def test_the_note_of_the_highlighted_job_is_shown(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        table = screen.query_one("#jobs-table", DataTable)
        table.move_cursor(
            row=next(i for i, r in enumerate(screen.view.rows) if r[3] == "done")
        )
        await pilot.pause()
        screen.action_refresh()
        await pilot.pause()
        assert "субтитры" in str(screen.query_one("#jobs-note", Static).renderable)

    drive_screen(queued, body)


def test_retrying_from_the_screen_changes_the_row(queued: Database) -> None:
    async def body(screen: JobsScreen, pilot: Any) -> None:
        table = screen.query_one("#jobs-table", DataTable)
        table.move_cursor(
            row=next(i for i, r in enumerate(screen.view.rows) if r[3] == "failed")
        )
        await pilot.pause()
        screen.action_retry()
        await pilot.pause()
        assert not [row for row in screen.view.rows if row[3] == "failed"]

    drive_screen(queued, body)


def test_the_screen_refreshes_on_a_timer(queued: Database) -> None:
    """The worker is another process; nothing tells the TUI a job finished."""
    async def body(screen: JobsScreen, pilot: Any) -> None:
        assert screen._timer is not None

    drive_screen(queued, body)


def test_the_screen_holds_no_state_of_its_own(queued: Database) -> None:
    """Part 7's rule, which is why nine tests were enough for its mapper.

    Compared against a bare `Screen`, not against nothing: Textual's own
    `__init__` sets dozens of attributes and mounting sets more, so the only
    meaningful question is what *this* class adds on top.
    """
    from textual.screen import Screen

    added = set(vars(JobsScreen(queued))) - set(vars(Screen()))
    assert added == {"view", "_timer"}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_jobs.py -q`
Expected: `ModuleNotFoundError: No module named 'rytp.tui.screens.jobs'`.

- [ ] **Step 3: Write `rytp/tui/screens/jobs.py`**

```python
"""The queue, watched (design §10).

A shell. Every decision — which rows, which filters, what the status line says,
what retrying means — is `JobsView`'s, and every action below is one line.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp import constants as C
from rytp.db import Database
from rytp.tui.jobs_view import JobsView
from rytp.tui.navigation import BACK_KEY

__all__ = ["JobsScreen"]


class JobsScreen(Screen[None]):
    """What is queued, running, done and failed."""

    DEFAULT_CSS = """
    #jobs-table  { height: 1fr; }
    #jobs-note   { height: auto; padding: 0 1; }
    #jobs-status { height: auto; padding: 0 1; }
    """

    # No bare letters: consistent with every other screen, and the footer
    # stays readable next to the mapper's.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("ctrl+r", "refresh", "Refresh"),
        # Not ctrl+s: the cut-list editor saves with it, and one key that
        # means "save" on one screen and "filter" on another is the `-s`
        # finding happening again inside Part 8.
        Binding("ctrl+e", "cycle_state", "State filter"),
        Binding("ctrl+o", "cycle_pool", "Pool filter"),
        Binding("ctrl+y", "retry", "Retry job"),
        Binding("ctrl+g", "retry_all", "Retry all failed"),
        Binding("ctrl+x", "cancel", "Cancel job"),
        Binding("ctrl+b", "toggle_pause", "Pause/resume"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self.view = JobsView(db)
        self._timer: object | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="jobs-table")
        yield Static("", id="jobs-note")
        yield Static("", id="jobs-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#jobs-table", DataTable).cursor_type = "row"
        self._draw()
        # The worker is a separate process (design §5), so progress only
        # arrives by looking again.
        self._timer = self.set_interval(C.TUI_QUEUE_REFRESH_S, self.action_refresh)

    # -- actions -----------------------------------------------------

    def action_refresh(self) -> None:
        self.view.refresh()
        self._draw()

    def action_cycle_state(self) -> None:
        self.view.cycle_state()
        self._draw()

    def action_cycle_pool(self) -> None:
        self.view.cycle_pool()
        self._draw()

    def action_retry(self) -> None:
        self._announce(self.view.retry_at(self._cursor()))

    def action_retry_all(self) -> None:
        self._announce(self.view.retry_all_failed())

    def action_cancel(self) -> None:
        self._announce(self.view.cancel_at(self._cursor()))

    def action_toggle_pause(self) -> None:
        self._announce(self.view.toggle_pause())

    # -- drawing -----------------------------------------------------

    def _cursor(self) -> int:
        return int(self.query_one("#jobs-table", DataTable).cursor_row or 0)

    def _announce(self, message: str) -> None:
        self._draw()
        self.query_one("#jobs-status", Static).update(message)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Only the note pane: redrawing the table here would move its cursor
        # and raise this message again. Without it the note would change only
        # on the two-second timer, which is a long time to stare at a row.
        if event.data_table.id == "jobs-table":
            self.query_one("#jobs-note", Static).update(
                self.view.note_at(int(event.cursor_row))
            )

    def _draw(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        cursor = table.cursor_row or 0
        table.clear(columns=True)
        if self.view.columns:
            table.add_columns(*self.view.columns)
            for row in self.view.rows:
                table.add_row(*row)
        if self.view.rows:
            table.move_cursor(row=min(cursor, len(self.view.rows) - 1))
        self.query_one("#jobs-note", Static).update(self.view.note_at(cursor))
        self.query_one("#jobs-status", Static).update(self.view.status)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_tui_jobs.py tests/test_tui_navigation.py -q`
Expected: PASS, and the `jobs` entry in the navigation map now loads. Remove the
`needs_all_screens` skip from any case that only waited on this screen.

If `set_interval` is not available on the installed Textual version's `Screen`,
call it on `self.app` instead; that is the only version-sensitive line here.

- [ ] **Step 5: Prove nothing ran a job**

Run: `python -m pytest tests/test_tui_jobs.py -q`
Expected: PASS, in well under a second. The screen enqueues, cancels and retries;
it never claims a job, and a worker loop would take visibly longer.

- [ ] **Step 6: Lint and type-check**

Run: `python -m ruff check rytp/tui tests/test_tui_jobs.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add rytp/tui/screens/jobs.py tests/test_tui_jobs.py
git commit -m "feat(tui): watch the queue, with the handler's note in full"
```

---

### Task 10: The cut-list view, headless

Design §8: "Every fragment keeps its ranked alternatives so a choice can be
swapped without re-running", and "The cut list is a plain editable file… hand-edit
it and re-render. Timings are editable by hand, which is deliberately preferred
over clever heuristics." Part 5 built both halves — an emitter that writes the
alternatives and a loader written for a file a person has changed — and left the
person in a text editor, counting milliseconds.

This view is the other end of that. Four verbs, and nothing else:

- **Swap.** Promote one of a fragment's ranked alternatives, or adopt one of a
  gap's ranked substitutions.
- **Nudge.** Move one boundary by a frame or by a quarter second.
- **Gap.** Set or clear the pause the renderer inserts before a slot.
- **Undo.** A stack of whole cut lists, because every verb above is a small
  edit to an immutable structure and keeping the previous one costs nothing.

Two things it deliberately does not do. It never touches the database — Part 5's
loader does not either, and "does video 42 still exist" is a question for
`assemble show`. And it never re-plans: re-running the assembler is
`assemble plan`, and a screen that silently re-planned would throw away exactly
the hand edits it exists to preserve.

**One honest loss, pinned by a test.** `Alternative` carries seven fields and a
chosen fragment carries eleven: promoting an alternative cannot know its
`align_score`, `video_speaker_id` or `speaker_label`, because the assembler never
wrote them for the runner-up. The view sets all three to `None` rather than
copying the displaced fragment's, which would be a lie about a different video.
The visible consequence is that Part 6's per-speaker pause statistics fall back
to per-video for a swapped fragment, which is design §9's own documented
fallback.

**Files:**
- Create: `rytp/tui/cutlist_view.py`
- Test: `tests/test_tui_cutlist.py` (the headless half)

**Interfaces:**
- Consumes: `rytp.assemble.cutlist.{CutList, Slot, Alternative, Substitution, CutlistError, load_cutlist, write_cutlist, cutlist_path, validate_name}`, `rytp.config.paths`, `rytp.constants.{CUTLISTS_DIRNAME, TUI_CUTLIST_NUDGE_MS, TUI_CUTLIST_COARSE_NUDGE_MS, NULL_CELL}`, `rytp.models.InvalidInputError`.
- Produces:
  - `CutlistSummary(name, path, target, slots, gaps, sources, duration_ms, error)` — frozen
  - `available_cutlists() -> tuple[CutlistSummary, ...]`
  - `CutlistView(path)` with `name`, `path`, `cutlist`, `dirty`, `columns`, `rows`, `status`
  - `slot_at(index) -> Slot | None`, `option_columns(index) -> tuple[str, ...]`, `option_rows(index) -> tuple[tuple[str, ...], ...]`
  - `swap(slot_index, option_index) -> str`, `nudge(slot_index, *, edge, delta_ms) -> str`
  - `set_gap_before(slot_index, gap_ms) -> str`, `clear_gap(slot_index) -> str`
  - `undo() -> str`, `save() -> Path`, `reload() -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_cutlist.py`:

```python
"""Editing a cut list without leaving the tool (design §8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistParams,
    Slot,
    Substitution,
    load_cutlist,
    write_cutlist,
)
from rytp.tui.cutlist_view import CutlistView, available_cutlists

CREATED = "2026-09-21T09:00:00+00:00"
PARAMS = CutlistParams(
    consistency=0.25, seed=0, pad_ms=0, speaker="", exclude=(), min_align_score=0.0
)


def sample() -> CutList:
    """One fragment with an alternative, one gap with a substitution."""
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
                speaker_label="Ведущий",
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


@pytest.fixture()
def written(data_dir: object, tmp_path: Path) -> Path:
    from rytp.assemble.cutlist import cutlist_path

    return write_cutlist(sample(), cutlist_path("demo"))


def test_the_picker_lists_what_is_on_disk(written: Path) -> None:
    """The mapper has `SpeakerVideosScreen` for the same reason: nobody should
    have to remember a name to edit the thing they just planned."""
    listed = available_cutlists()
    assert [item.name for item in listed] == ["demo"]
    assert listed[0].target == "мы всё исправим"
    assert listed[0].gaps == 1
    assert listed[0].sources == 1
    assert listed[0].error is None


def test_a_broken_file_is_listed_with_its_complaint_not_hidden(
    written: Path, data_dir: object
) -> None:
    """A cut list the owner mistyped is exactly the one he needs to find."""
    written.with_name("broken.toml").write_text("this is not toml [", encoding="utf-8")
    broken = next(item for item in available_cutlists() if item.name == "broken")
    assert broken.error is not None
    assert broken.slots == 0


def test_the_table_is_the_timeline_in_document_order(written: Path) -> None:
    """design §8: one ordered slot array, "document order is the timeline"."""
    view = CutlistView(written)
    assert view.columns == (
        "#", "kind", "text", "source", "in", "out", "gap", "alts",
    )
    assert [row[1] for row in view.rows] == ["fragment", "gap"]
    assert view.rows[0][3] == "v3"
    assert view.rows[1][3] == C.NULL_CELL


def test_the_options_of_a_fragment_are_its_ranked_alternatives(written: Path) -> None:
    view = CutlistView(written)
    assert view.option_columns(0) == ("#", "text", "source", "in", "out", "cost")
    assert view.option_rows(0)[0][2] == "v7"


def test_the_options_of_a_gap_are_its_ranked_substitutions(written: Path) -> None:
    """design §8: "Offer ranked substitutions… but never apply one silently"."""
    view = CutlistView(written)
    assert view.option_columns(1) == ("#", "text", "reason", "source", "in", "out")
    assert view.option_rows(1)[0][1] == "исправит"
    assert view.option_rows(1)[0][2] == "edit"


def test_swapping_promotes_the_alternative_and_keeps_the_old_choice(
    written: Path,
) -> None:
    """Reversible by construction: the displaced fragment becomes an
    alternative in the slot it was displaced from."""
    view = CutlistView(written)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (7, 10_500, 11_220)
    assert [alt.video_id for alt in slot.alternatives] == [3]
    assert view.dirty is True


def test_swapping_twice_returns_to_where_it_started(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (3, 612_340, 613_100)


def test_a_promoted_alternative_admits_it_does_not_know_the_speaker(
    written: Path,
) -> None:
    """The one lossy edge, stated rather than papered over. `Alternative` has
    no `align_score`, `video_speaker_id` or `speaker_label`, so keeping the
    displaced fragment's would claim that a clip from video 7 was said by the
    person identified in video 3."""
    view = CutlistView(written)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.align_score is None
    assert slot.video_speaker_id is None
    assert slot.speaker_label is None
    assert "speaker" in view.status.lower()


def test_adopting_a_substitution_turns_a_gap_into_a_fragment(written: Path) -> None:
    view = CutlistView(written)
    view.swap(1, 0)
    slot = view.slot_at(1)
    assert slot is not None
    assert slot.kind == "fragment"
    assert slot.text == "исправит"
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (9, 220_100, 220_780)
    assert slot.substitutions, "the ranked list survives so the choice can change"


def test_nudging_moves_one_boundary_by_one_frame(written: Path) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="start", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 612_340 - C.TUI_CUTLIST_NUDGE_MS
    assert slot.end_ms == 613_100


def test_a_nudge_may_not_turn_a_fragment_inside_out(written: Path) -> None:
    view = CutlistView(written)
    message = view.nudge(0, edge="start", delta_ms=10_000)
    assert "shorter than" in message or "would not leave" in message
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 612_340


def test_a_nudge_may_not_move_a_boundary_before_the_start_of_the_video(
    written: Path,
) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="start", delta_ms=-999_999)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 0


def test_a_gap_slot_has_no_boundary_to_nudge(written: Path) -> None:
    view = CutlistView(written)
    assert "gap" in view.nudge(1, edge="start", delta_ms=40).lower()


def test_setting_a_pause_writes_the_key_the_renderer_reads(written: Path) -> None:
    """design §9's gap is "Configurable… Disableable if it doesn't sound
    right", and contracts §7's cut list carries `gap_before_ms` for exactly
    that."""
    view = CutlistView(written)
    view.set_gap_before(1, 320)
    slot = view.slot_at(1)
    assert slot is not None and slot.gap_before_ms == 320
    view.clear_gap(1)
    slot = view.slot_at(1)
    assert slot is not None and slot.gap_before_ms is None


def test_a_negative_pause_is_refused(written: Path) -> None:
    view = CutlistView(written)
    assert "negative" in view.set_gap_before(1, -5).lower()


def test_undo_walks_back_through_every_kind_of_edit(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    view.nudge(0, edge="end", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
    view.set_gap_before(1, 200)
    view.undo()
    view.undo()
    view.undo()
    assert view.cutlist == load_cutlist(written)
    assert view.dirty is False
    assert "nothing to undo" in view.undo().lower()


def test_saving_round_trips_through_the_file_part_five_wrote(written: Path) -> None:
    """The file is the durable representation (design §8), so what the screen
    saves must be what Part 6 can read."""
    view = CutlistView(written)
    view.swap(0, 0)
    view.set_gap_before(1, 250)
    path = view.save()
    assert path == written
    assert view.dirty is False
    reloaded = load_cutlist(path)
    assert reloaded.slots[0].video_id == 7
    assert reloaded.slots[1].gap_before_ms == 250
    assert CutlistView(path).cutlist == reloaded


def test_saving_keeps_the_name_and_the_target(written: Path) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="end", delta_ms=40)
    reloaded = load_cutlist(view.save())
    assert reloaded.name == "demo"
    assert reloaded.target == "мы всё исправим"


def test_reloading_discards_unsaved_edits_and_says_so(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    message = view.reload()
    assert "discarded" in message.lower()
    assert view.dirty is False
    assert view.cutlist == load_cutlist(written)


def test_the_status_line_says_what_will_be_rendered(written: Path) -> None:
    status = CutlistView(written).status
    assert "1 fragment" in status
    assert "1 gap" in status
    assert "unsaved" not in status.lower()


def test_an_index_off_the_end_is_reported_not_raised(written: Path) -> None:
    view = CutlistView(written)
    assert view.slot_at(99) is None
    assert view.option_rows(99) == ()
    assert "no slot" in view.swap(99, 0).lower()


def test_swapping_to_an_option_that_is_not_there_is_reported(written: Path) -> None:
    view = CutlistView(written)
    assert "no alternative" in view.swap(0, 9).lower()


def test_a_fragment_with_no_alternatives_says_so(written: Path, tmp_path: Path) -> None:
    """The realistic case: the assembler found one occurrence and nothing else."""
    bare = CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="bare",
        target="раз",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=0,
                text="раз",
                video_id=1,
                first_word_ord=0,
                last_word_ord=0,
                start_ms=0,
                end_ms=500,
            ),
        ),
    )
    path = write_cutlist(bare, tmp_path / "bare.toml")
    view = CutlistView(path)
    assert view.option_rows(0) == ()
    assert "no alternative" in view.swap(0, 0).lower()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_cutlist.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui.cutlist_view'`.

- [ ] **Step 3: Write `rytp/tui/cutlist_view.py`**

```python
"""Editing a cut list (design §8, design §10).

design §8 makes the cut list "a plain editable file… hand-edit it and re-render",
and the owner asked for exactly that: adjusting which fragment is used and when
it starts. Part 5 wrote the emitter and a loader built for a file a person has
changed. What was missing is a place to make the change that is not another
window with a text editor in it.

Four verbs — swap, nudge, gap, undo — and a save. No database and no re-planning:
the loader does not look at the database either, and a screen that quietly
re-ran the assembler would discard the hand edits it exists to keep.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistError,
    Slot,
    load_cutlist,
    write_cutlist,
)

__all__ = ["CutlistSummary", "CutlistView", "available_cutlists"]


@dataclass(frozen=True)
class CutlistSummary:
    """One line of the picker."""

    name: str
    path: Path
    target: str
    slots: int
    gaps: int
    sources: int
    duration_ms: int
    error: str | None = None


def available_cutlists() -> tuple[CutlistSummary, ...]:
    """Every cut list on disk, broken ones included.

    A file the owner mistyped is the one he most needs to find, so a load
    failure becomes a row with its complaint rather than a missing row.
    """
    directory = config.paths().root / C.CUTLISTS_DIRNAME
    if not directory.is_dir():
        return ()
    found: list[CutlistSummary] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            cutlist = load_cutlist(path)
        # `ValueError` because `tomllib.TOMLDecodeError` subclasses it, and a
        # file that is not TOML at all may never reach Part 5's own error type.
        # A picker that raises on one bad file shows none of the good ones.
        except (CutlistError, ValueError, OSError) as exc:
            found.append(
                CutlistSummary(
                    name=path.stem,
                    path=path,
                    target="",
                    slots=0,
                    gaps=0,
                    sources=0,
                    duration_ms=0,
                    error=str(exc),
                )
            )
            continue
        found.append(
            CutlistSummary(
                name=cutlist.name or path.stem,
                path=path,
                target=cutlist.target,
                slots=len(cutlist.slots),
                gaps=len(cutlist.gaps),
                sources=len({slot.video_id for slot in cutlist.fragments}),
                duration_ms=cutlist.duration_ms,
            )
        )
    return tuple(found)


def _ms(value: int | None) -> str:
    if value is None:
        return C.NULL_CELL
    seconds, milliseconds = divmod(int(value), 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"


class CutlistView:
    """The slots of one cut list, and the four edits worth making by hand."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cutlist: CutList = load_cutlist(path)
        self.name = self.cutlist.name or path.stem
        self.dirty = False
        self._undo: list[CutList] = []
        self.status = self._status()

    # -- reading -----------------------------------------------------

    @property
    def columns(self) -> tuple[str, ...]:
        return ("#", "kind", "text", "source", "in", "out", "gap", "alts")

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (
                str(index),
                slot.kind,
                slot.text,
                f"v{slot.video_id}" if slot.video_id is not None else C.NULL_CELL,
                _ms(slot.start_ms),
                _ms(slot.end_ms),
                _ms(slot.gap_before_ms),
                str(len(self._options(slot)[1])),
            )
            for index, slot in enumerate(self.cutlist.slots)
        )

    def slot_at(self, index: int) -> Slot | None:
        slots = self.cutlist.slots
        return slots[index] if 0 <= index < len(slots) else None

    @staticmethod
    def _options(slot: Slot) -> tuple[str, Sequence[object]]:
        """What this slot can be swapped to, and which kind of thing that is.

        Not simply "alternatives for a fragment": adopting a substitution
        turns a gap into a fragment, and its ranked substitutions are exactly
        what a second thought needs. So the ranked list that *has* entries
        wins, and a fragment that was always a fragment shows alternatives.
        """
        if slot.kind == "fragment" and slot.alternatives:
            return "alternative", slot.alternatives
        if slot.substitutions:
            return "substitution", slot.substitutions
        return "alternative", slot.alternatives

    def option_columns(self, index: int) -> tuple[str, ...]:
        slot = self.slot_at(index)
        if slot is None:
            return ()
        if self._options(slot)[0] == "alternative":
            return ("#", "text", "source", "in", "out", "cost")
        return ("#", "text", "reason", "source", "in", "out")

    def option_rows(self, index: int) -> tuple[tuple[str, ...], ...]:
        slot = self.slot_at(index)
        if slot is None:
            return ()
        if self._options(slot)[0] == "alternative":
            return tuple(
                (
                    str(position),
                    alt.text,
                    f"v{alt.video_id}",
                    _ms(alt.start_ms),
                    _ms(alt.end_ms),
                    f"{alt.cost:.2f}",
                )
                for position, alt in enumerate(slot.alternatives)
            )
        return tuple(
            (
                str(position),
                sub.text,
                sub.reason,
                f"v{sub.video_id}",
                _ms(sub.start_ms),
                _ms(sub.end_ms),
            )
            for position, sub in enumerate(slot.substitutions)
        )

    # -- editing -----------------------------------------------------

    def _replace_slot(self, index: int, slot: Slot) -> None:
        self._undo.append(self.cutlist)
        slots = list(self.cutlist.slots)
        slots[index] = slot
        self.cutlist = dataclasses.replace(self.cutlist, slots=tuple(slots))
        self.dirty = True

    def swap(self, slot_index: int, option_index: int) -> str:
        """Promote a ranked alternative, or adopt a ranked substitution."""
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        what, options = self._options(slot)
        if not (0 <= option_index < len(options)):
            return f"slot {slot_index} has no {what} {option_index}"
        if what == "alternative":
            return self._promote(slot_index, slot, option_index)
        return self._adopt(slot_index, slot, option_index)

    def _promote(self, index: int, slot: Slot, option_index: int) -> str:
        chosen = slot.alternatives[option_index]
        # `is not None`, not `or`: ordinals are 0-based, so `ord == 0` is the
        # first word of a video and not a missing value. Part 5's loader uses
        # -1 for "whoever retyped a timing dropped the provenance".
        def _or(value: int | None, absent: int) -> int:
            return absent if value is None else value

        displaced = Alternative(
            video_id=_or(slot.video_id, -1),
            first_word_ord=_or(slot.first_word_ord, -1),
            last_word_ord=_or(slot.last_word_ord, -1),
            start_ms=_or(slot.start_ms, 0),
            end_ms=_or(slot.end_ms, 0),
            text=slot.text,
            cost=slot.cost if slot.cost is not None else 0.0,
        )
        alternatives = list(slot.alternatives)
        alternatives[option_index] = displaced
        self._replace_slot(
            index,
            dataclasses.replace(
                slot,
                video_id=chosen.video_id,
                first_word_ord=chosen.first_word_ord,
                last_word_ord=chosen.last_word_ord,
                start_ms=chosen.start_ms,
                end_ms=chosen.end_ms,
                text=chosen.text,
                cost=chosen.cost,
                # An Alternative carries none of these, and copying the
                # displaced fragment's would claim a clip from another video
                # was said by a person identified in this one.
                align_score=None,
                video_speaker_id=None,
                speaker_label=None,
                alternatives=tuple(alternatives),
            ),
        )
        self.status = self._status(
            f"slot {index} now comes from v{chosen.video_id}; its speaker and "
            f"alignment score are unknown, so the render will use this video's "
            f"pause statistics rather than a speaker's"
        )
        return self.status

    def _adopt(self, index: int, slot: Slot, option_index: int) -> str:
        chosen = slot.substitutions[option_index]
        self._replace_slot(
            index,
            dataclasses.replace(
                slot,
                kind="fragment",
                text=chosen.text,
                video_id=chosen.video_id,
                first_word_ord=chosen.first_word_ord,
                last_word_ord=chosen.last_word_ord,
                start_ms=chosen.start_ms,
                end_ms=chosen.end_ms,
                align_score=None,
                video_speaker_id=None,
                speaker_label=None,
            ),
        )
        self.status = self._status(
            f'slot {index} now says "{chosen.text}" instead of "{slot.text}"'
        )
        return self.status

    def nudge(self, slot_index: int, *, edge: str, delta_ms: int) -> str:
        """Move one boundary. design §8 prefers this to a clever heuristic."""
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        if slot.kind != "fragment" or slot.start_ms is None or slot.end_ms is None:
            return f"slot {slot_index} is a gap: there is no boundary to move"
        start, end = slot.start_ms, slot.end_ms
        if edge == "start":
            start = max(0, start + delta_ms)
        else:
            end = max(0, end + delta_ms)
        if end - start < C.TUI_CUTLIST_NUDGE_MS:
            return (
                f"that would leave slot {slot_index} shorter than "
                f"{C.TUI_CUTLIST_NUDGE_MS} ms"
            )
        self._replace_slot(
            slot_index, dataclasses.replace(slot, start_ms=start, end_ms=end)
        )
        self.status = self._status(
            f"slot {slot_index} is now {_ms(start)} to {_ms(end)}"
        )
        return self.status

    def set_gap_before(self, slot_index: int, gap_ms: int) -> str:
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        if gap_ms < 0:
            return "a pause cannot be negative"
        self._replace_slot(slot_index, dataclasses.replace(slot, gap_before_ms=gap_ms))
        self.status = self._status(f"slot {slot_index} is preceded by {gap_ms} ms")
        return self.status

    def clear_gap(self, slot_index: int) -> str:
        slot = self.slot_at(slot_index)
        if slot is None:
            return f"no slot {slot_index}"
        self._replace_slot(slot_index, dataclasses.replace(slot, gap_before_ms=None))
        self.status = self._status(f"slot {slot_index} uses the measured pause again")
        return self.status

    def undo(self) -> str:
        if not self._undo:
            return "nothing to undo"
        self.cutlist = self._undo.pop()
        self.dirty = bool(self._undo)
        self.status = self._status("undone")
        return self.status

    # -- the file ----------------------------------------------------

    def save(self) -> Path:
        """Write it back, through Part 5's emitter, so Part 6 can read it."""
        config.ensure_dir(self.path.parent)
        written = write_cutlist(self.cutlist, self.path)
        self._undo.clear()
        self.dirty = False
        self.status = self._status(f"saved to {written}")
        return written

    def reload(self) -> str:
        self.cutlist = load_cutlist(self.path)
        self._undo.clear()
        self.dirty = False
        self.status = self._status("discarded the unsaved edits")
        return self.status

    def _status(self, note: str = "") -> str:
        fragments = len(self.cutlist.fragments)
        gaps = len(self.cutlist.gaps)
        sources = len({slot.video_id for slot in self.cutlist.fragments})
        parts = [
            f"{fragments} fragment{'' if fragments == 1 else 's'}",
            f"{gaps} gap{'' if gaps == 1 else 's'}",
            f"{sources} source{'' if sources == 1 else 's'}",
            _ms(self.cutlist.duration_ms),
        ]
        if self.dirty:
            parts.append("unsaved")
        if note:
            parts.append(note)
        return " · ".join(parts)
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `python -m pytest tests/test_tui_cutlist.py -q`
Expected: PASS.

If `Slot` turns out not to be a dataclass, `dataclasses.replace` will raise —
read `rytp/assemble/cutlist.py` and construct the replacement explicitly instead.
If `CutList.duration_ms` sums only fragments (it should), the status line is
already right.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/tui/cutlist_view.py tests/test_tui_cutlist.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/tui/cutlist_view.py tests/test_tui_cutlist.py
git commit -m "feat(tui): headless cut-list editing — swap, nudge, gap, undo"
```

---

### Task 11: The cut-list screens

Two screens, for the reason Part 7's mapper needed two: the editor needs a cut
list, and the owner should not have to remember a name. `CutlistPickerScreen`
lists what is on disk; `CutlistScreen` edits one.

The editor is the one screen in the application with no focused text box, so
bare letters are legal here and they read better than chords for verbs used
dozens of times in a row: `s` swap, `a` adopt is the same key, `[` `]` nudge,
`g` gap, `u` undo, `ctrl+s` save. The navigation map marks it `bare_letters=True`
and the Task 5 test allows exactly that.

Every action is one line over `CutlistView`. The screen holds the two cursors and
nothing else.

**Files:**
- Create: `rytp/tui/screens/cutlist.py`
- Test: `tests/test_tui_cutlist.py` (append the screen half)

**Interfaces:**
- Consumes: `rytp.tui.cutlist_view.{CutlistView, available_cutlists}`, `rytp.tui.navigation.BACK_KEY`, `rytp.constants.{TUI_CUTLIST_NUDGE_MS, TUI_CUTLIST_COARSE_NUDGE_MS}`, `rytp.db.Database`.
- Produces: `CutlistPickerScreen(db)`, `CutlistScreen(db, path)` with `.view` and the actions `swap`, `nudge_start_back`, `nudge_start_on`, `nudge_end_back`, `nudge_end_on`, `set_gap`, `clear_gap`, `undo`, `save`, `reload`.

- [ ] **Step 1: Append the failing test**

To `tests/test_tui_cutlist.py`:

```python
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from rytp.db import Database
from rytp.tui.screens.cutlist import CutlistPickerScreen, CutlistScreen


def drive(screen_factory: Callable[[], Any], body: Callable[[Any, Any], Awaitable[None]]) -> None:
    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(screen_factory())

    async def main() -> None:
        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            await body(app.screen, pilot)

    asyncio.run(main())


def test_the_picker_shows_what_is_on_disk(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        table = screen.query_one("#cutlist-picker", DataTable)
        assert table.row_count == 1
        assert "demo" in str(table.get_cell_at((0, 0)))

    drive(lambda: CutlistPickerScreen(db), body)


def test_the_picker_says_so_when_there_is_nothing_to_edit(db: Database, data_dir: object) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        text = str(screen.query_one("#cutlist-picker-status", Static).renderable)
        assert "assemble plan" in text

    drive(lambda: CutlistPickerScreen(db), body)


def test_the_editor_shows_the_slots_and_the_options_of_the_first(
    db: Database, written: Path
) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        slots = screen.query_one("#cutlist-slots", DataTable)
        options = screen.query_one("#cutlist-options", DataTable)
        assert slots.row_count == 2
        assert options.row_count == 1

    drive(lambda: CutlistScreen(db, written), body)


def test_moving_to_the_gap_shows_its_substitutions(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        screen.query_one("#cutlist-slots", DataTable).move_cursor(row=1)
        await pilot.pause()
        options = screen.query_one("#cutlist-options", DataTable)
        assert [str(c.label) for c in options.columns.values()][2] == "reason"

    drive(lambda: CutlistScreen(db, written), body)


def test_one_key_swaps_the_highlighted_option_in(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.video_id == 7

    drive(lambda: CutlistScreen(db, written), body)


def test_the_bracket_keys_move_the_boundaries(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("[")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.start_ms < 612_340

    drive(lambda: CutlistScreen(db, written), body)


def test_undo_is_one_key(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.press("u")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.video_id == 3

    drive(lambda: CutlistScreen(db, written), body)


def test_saving_writes_the_file(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert screen.view.dirty is False
        assert load_cutlist(written).slots[0].video_id == 7

    drive(lambda: CutlistScreen(db, written), body)


def test_the_screen_holds_only_its_view(db: Database, written: Path) -> None:
    """Every decision is `CutlistView`'s; the screen keeps two cursors, and
    Textual keeps those in the `DataTable`s. Measured against a bare `Screen`,
    because Textual's own `__init__` sets dozens of attributes."""
    from textual.screen import Screen

    added = set(vars(CutlistScreen(db, written))) - set(vars(Screen()))
    assert added == {"view", "_db"}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_cutlist.py -q`
Expected: `ModuleNotFoundError: No module named 'rytp.tui.screens.cutlist'`.

- [ ] **Step 3: Write `rytp/tui/screens/cutlist.py`**

```python
"""Editing a cut list without another window (design §8, §10).

Two screens, like the mapper: one to choose a cut list, one to edit it. Both
are shells — every decision belongs to `CutlistView`, and each action below is
a single call into it.

This is the only screen with no focused `Input`, so its verbs are bare letters.
They are used dozens of times in a row on one cut list and a chord for each
would be a worse tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from rytp import constants as C
from rytp.db import Database
from rytp.tui.cutlist_view import CutlistView, available_cutlists
from rytp.tui.navigation import BACK_KEY

__all__ = ["CutlistPickerScreen", "CutlistScreen"]


def _fill(
    table: DataTable, columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
) -> None:
    table.clear(columns=True)
    if columns:
        table.add_columns(*columns)
        for row in rows:
            table.add_row(*row)


class CutlistPickerScreen(Screen[None]):
    """Which cut list. Nobody should have to remember a name."""

    DEFAULT_CSS = """
    #cutlist-picker { height: 1fr; }
    #cutlist-picker-status { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
    ]

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db
        self._paths: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="cutlist-picker")
        yield Static("", id="cutlist-picker-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#cutlist-picker", DataTable)
        table.cursor_type = "row"
        found = available_cutlists()
        self._paths = [item.path for item in found]
        _fill(
            table,
            ("name", "target", "slots", "gaps", "sources", "problem"),
            tuple(
                (
                    item.name,
                    item.target,
                    str(item.slots),
                    str(item.gaps),
                    str(item.sources),
                    item.error or "",
                )
                for item in found
            ),
        )
        self.query_one("#cutlist-picker-status", Static).update(
            f"{len(found)} cut list{'' if len(found) == 1 else 's'}"
            if found
            else "no cut lists yet — plan one with `rytp assemble plan \"…\"`"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "cutlist-picker" and self._paths:
            self.app.push_screen(CutlistScreen(self._db, self._paths[event.cursor_row]))


class CutlistScreen(Screen[None]):
    """The timeline, the highlighted slot's options, and five verbs."""

    DEFAULT_CSS = """
    #cutlist-slots   { height: 2fr; }
    #cutlist-options { height: 1fr; }
    #cutlist-status  { height: auto; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(BACK_KEY, "app.pop_screen", "Back"),
        Binding("s", "swap", "Swap in option"),
        Binding("[", "nudge_start_back", "Start earlier"),
        Binding("]", "nudge_start_on", "Start later"),
        Binding("ctrl+left", "nudge_end_back", "End earlier"),
        Binding("ctrl+right", "nudge_end_on", "End later"),
        Binding("g", "set_gap", "Pause before"),
        Binding("c", "clear_gap", "Measured pause"),
        Binding("u", "undo", "Undo"),
        Binding("r", "reload", "Discard edits"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, db: Database, path: Path) -> None:
        super().__init__()
        self._db = db
        self.view = CutlistView(path)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="cutlist-slots")
        yield DataTable(id="cutlist-options")
        yield Static("", id="cutlist-status")
        yield Footer()

    def on_mount(self) -> None:
        for table_id in ("#cutlist-slots", "#cutlist-options"):
            self.query_one(table_id, DataTable).cursor_type = "row"
        self.action_redraw()

    # -- actions -----------------------------------------------------

    def action_swap(self) -> None:
        self._after(self.view.swap(self._slot(), self._option()))

    def action_nudge_start_back(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="start", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_start_on(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="start", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_end_back(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="end", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_nudge_end_on(self) -> None:
        self._after(
            self.view.nudge(self._slot(), edge="end", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
        )

    def action_set_gap(self) -> None:
        self._after(self.view.set_gap_before(self._slot(), C.TUI_CUTLIST_COARSE_NUDGE_MS))

    def action_clear_gap(self) -> None:
        self._after(self.view.clear_gap(self._slot()))

    def action_undo(self) -> None:
        self._after(self.view.undo())

    def action_reload(self) -> None:
        self._after(self.view.reload())

    def action_save(self) -> None:
        self.view.save()
        self._after(self.view.status)

    # -- drawing -----------------------------------------------------

    def action_redraw(self) -> None:
        """Redraw everything. Called after a mutation, never from a handler
        that a redraw itself can trigger."""
        index = self._slot()
        table = self.query_one("#cutlist-slots", DataTable)
        _fill(table, self.view.columns, self.view.rows)
        if self.view.rows:
            table.move_cursor(row=min(index, len(self.view.rows) - 1))
        self._draw_options()
        self.query_one("#cutlist-status", Static).update(self.view.status)

    def _draw_options(self) -> None:
        index = self._slot()
        _fill(
            self.query_one("#cutlist-options", DataTable),
            self.view.option_columns(index),
            self.view.option_rows(index),
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Only the options pane, and only from the slot table: redrawing the
        # slot table here would move its cursor, raise this message again, and
        # loop.
        if event.data_table.id == "cutlist-slots":
            self._draw_options()

    def _slot(self) -> int:
        return int(self.query_one("#cutlist-slots", DataTable).cursor_row or 0)

    def _option(self) -> int:
        return int(self.query_one("#cutlist-options", DataTable).cursor_row or 0)

    def _after(self, message: str) -> None:
        self.action_redraw()
        self.query_one("#cutlist-status", Static).update(message)

    # `_draw_slots` refills the slot table, which moves its cursor, which
    # raises RowHighlighted. If that handler refilled the slot table again the
    # screen would loop forever, so highlighting only redraws the options.
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_tui_cutlist.py tests/test_tui_navigation.py tests/test_tui_shell.py -q`
Expected: PASS, with every `needs_all_screens` skip now gone — both missing
screens exist, so the navigation map's collision test finally covers all five.

Check the footer by eye once: `python -m rytp tui`, then `F8`. The bare letters
must not fight the picker, which has no `Input`.

- [ ] **Step 5: Lint and type-check**

Run: `python -m ruff check rytp/tui tests/test_tui_cutlist.py`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add rytp/tui/screens/cutlist.py tests/test_tui_cutlist.py
git commit -m "feat(tui): pick a cut list and edit its fragments and timings"
```

---

### Task 12: The M1 walk, end to end

Design §11, M1: *"Add a handful of URLs, download audio and video, transcribe and
align, search, assemble a short sentence drawing on **multiple source videos**,
render a real file with a source list. Multiple sources is the point: cross-video
seams, volume matching and the consistency knob are the risky parts, and a
single-source milestone would prove almost nothing."*

Seven parts have seven green suites and have never been run in the same process.
This is the first test that says they compose.

**What is faked, and where.** Four seams, each one the owning part published for
exactly this:

| Real thing | Seam | Why it is the right one |
|---|---|---|
| yt-dlp metadata | `rytp.commands.catalog._probe_video` | Part 1's own tests monkeypatch it |
| yt-dlp downloads | `RealYtDlpRunner` in `rytp.acquire` and `rytp.acquire.captions` | both do `runner = runner or RealYtDlpRunner()` |
| ffmpeg, decoding to WAV | `rytp.audio.extract.{_ffmpeg_binary, _run_ffmpeg}` | Part 2 calls them "the two module-level seams tests monkeypatch" |
| ffmpeg and ffprobe, rendering | `rytp.render.ffmpeg.Tools.resolve` | `Tools.faked` exists so "the whole orchestration suite runs on a machine with no ffmpeg installed" |

Nothing opens a socket, downloads a model, or spawns a player. The WAV the fake
ffmpeg writes is a **real** 16 kHz mono tone, not silence, because the acoustic
fingerprint measures f0 and spectral tilt and both are undefined on a flat zero
signal.

**Two transcripts, deliberately different.** A corpus where every video says the
same words cannot demonstrate a cross-video seam, which is the thing M1 says is
the point. The scripted transcriber keys off the video id in the WAV's filename.

**What it asserts** is what M1 asks for, not that every job reached `done`. The
caption tier and the acoustic fingerprint are Part 2's and Part 3's to test; a
failure there must not be reported as "the pipeline does not compose".

**Files:**
- Test: `tests/test_m1_end_to_end.py`

**Interfaces:**
- Consumes: `rytp.commands.resolve`, `rytp.db.Database`, `rytp.db.queries.set_setting`, `rytp.jobs.queue`, `rytp.models.RawWord`, `tests.consistency.write_test_wav`, `tests.fakes.FakeYtDlpRunner`, `tests.fake_engines.{FakeAligner, registered}`.
- Produces: no production code.

- [ ] **Step 1: Write the test**

Create `tests/test_m1_end_to_end.py`:

```python
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
from rytp.jobs import queue as Q
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
    assert rendered.message
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
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_m1_end_to_end.py -q`

This is the first time the seven parts have run together, so read every failure
as information about a seam rather than about this test. The likely ones:

- A flag name in one of the argument lines above that no longer exists. Every
  value is positional or a settled flag precisely so this should not happen; if
  it does, the argument line is what changes, never the registration.
- A `REQUIRED` parameter with no value in an argument line: `parse_arguments`
  refuses it by name, which tells you exactly what to add.
- `caption_words` or `fingerprint` failing. Neither is asserted here on purpose —
  they are Part 2's and Part 3's to test — but if one fails, note it and hand it
  to that part rather than absorbing it.
- The fake render runner's output-path guess. If Part 6 builds a command whose
  last argument is not the output, read `concat_command` and match it.

- [ ] **Step 3: Run it against real ffmpeg, once, by hand**

Design §11 says M1 ends in "a real file". The faked walk proves composition; a
real encode proves the arguments. This is not part of the suite:

```bash
python -m rytp doctor
python -m pytest tests/test_m1_end_to_end.py -q
```

then, on a machine with ffmpeg, register a local media file you already have and
walk the same commands by hand. Never put a real video id, channel name or URL in
a committed file.

- [ ] **Step 4: Lint**

Run: `python -m ruff check tests/test_m1_end_to_end.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tests/test_m1_end_to_end.py
git commit -m "test: walk M1 end to end with every external tool faked"
```

---

### Task 13: The gate

Everything at once, on a tree with no optional extras installed. This is the task
that says Part 8 is done, and the only one whose steps are all `run`.

**Files:**
- Modify: none, unless a check fails.

**Interfaces:**
- Consumes: everything.
- Produces: a green tree.

- [ ] **Step 1: The whole suite**

Run: `python -m pytest -q`
Expected: PASS, no failures, no errors.

- [ ] **Step 2: No skip is hiding an unwritten screen**

The navigation tests carry a `needs_all_screens` guard while Tasks 9 and 11 are
outstanding. Both have landed, so there must be nothing left to guard.

Run: `python -m pytest tests/test_tui_navigation.py tests/test_tui_shell.py -q -rs`
Expected: PASS with **no** `SKIPPED` lines. Delete the `needs_all_screens` marker
and its helper; a guard that can never fire is a guard that will one day hide
something.

- [ ] **Step 3: The suite passes with nothing optional installed**

Contracts §1: a `.[dev]` install must run every test.

```bash
python -m pip list --format=freeze > /dev/null
python -c "import importlib.util as u; print([n for n in ('torch','faster_whisper','gigaam','pyannote.audio','yt_dlp') if u.find_spec(n)])"
python -m pytest -q
```
Expected: the second command prints `[]` in a clean `.[dev]` environment, and the
suite still passes. If an extra is installed, the suite must pass anyway — no
test may depend on one being absent.

- [ ] **Step 4: Lint and type-check the whole tree**

Run: `python -m ruff check rytp tests`
Run: `python -m mypy rytp`
Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 5: The consistency suite on its own, and read it**

Run: `python -m pytest tests/test_consistency_surfaces.py tests/test_consistency_flags.py tests/test_consistency_jobs.py tests/test_consistency_schema.py -v`
Expected: PASS. Read the test names as a list: that list is what Part 8 promises
stays true, and it is the thing to extend when the next cross-part defect is
found by hand.

- [ ] **Step 6: Open the application and walk it**

Nothing above proves the TUI is usable, only that it works.

```bash
python -m rytp tui
```
Press `F1` and read the help. Press each of `F3`, `F4`, `F6`, `F7`, `F8` and
`Escape` back from each. Type `transcribe.run` in the filter, press Enter on it,
type a video id, and confirm the status line says the job was queued and names
`F7`. Press `F7` and see it. Quit with `Ctrl+Q`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: green gate for part 8 — suite, lint, types and the walk"
```

---

## Which test would have caught which finding

The reviews found every one of these by a person reading seven plans side by
side. That is not a repeatable process, and this is the column that makes it one.

| Finding | Test | Task |
|---|---|---|
| 1.1 — three parts invented three speaker resolvers | `test_a_declared_dependency_exists_with_the_declared_shape` (the `resolve_speaker_filter` row), `test_the_speaker_filter_keeps_the_two_fields_every_consumer_reads`, `test_the_shared_speaker_flags_mean_the_same_thing_everywhere` | 4, 2 |
| 1.2 — `align` jobs with no payload | `test_every_registered_kind_has_at_least_one_producer`, `test_every_ingest_chain_names_registered_kinds` | 3 |
| 2.1 — unaligned words marked cuttable | `test_the_three_transcript_tiers_are_enforced_by_the_database`, and the M1 walk's `tiers == {"aligned"}` | 4, 12 |
| 3.1 — a hardcoded `VIDEO_JOB_KINDS` went stale | `test_the_registry_is_exactly_the_contracted_set`, `test_the_producer_table_covers_the_registry_exactly`, `test_cycling_the_state_filter_walks_the_command_s_own_choices` | 3, 8 |
| 3.2 — a warning that reached the CLI and vanished on the worker | `test_re_transcribing_a_mapped_video_leaves_the_warning_on_the_job`, `test_a_handler_note_reaches_the_job_row`, `test_the_note_of_the_highlighted_job_is_readable_in_full` | 3, 8 |
| 3.3 — TUI scope: cut lists, queueing, job progress unclaimed | Tasks 7, 8, 9, 10 and 11 in their entirety; `test_every_long_running_command_is_classified` is the one that stops it recurring | 7–11 |
| 4.1 — the assembler's hot query could not exclude the caption tier | `test_the_indexes_are_exactly_the_ones_the_contract_lists` (the `words_alignable` row) | 4 |
| 4.2 — `--video-id` renamed in the prose, not in the registrations | `test_no_concept_is_spelled_two_ways`, `test_one_parameter_name_has_one_type` | 2 |
| "worth a test that has none" — lazy `snowballstemmer` | `test_models_imports_snowballstemmer_lazily` | 4 |
| A job kind with a handler and no producer | `test_every_registered_kind_has_at_least_one_producer`, `test_the_gpu_pool_is_the_expensive_one_and_holds_only_what_belongs_there` | 3 |
| A `NOT NULL` column a fixture omitted | `test_every_fixture_supplies_the_not_null_columns_of_the_table_it_writes` | 4 |

And the ones no review found, which these tests found while this plan was being
written — each is in the defect table at the top with its one-line fix:
`--enqueue` against `--queue`; `-s` meaning two things; four short flags missing
their leading dash; `ingest --transcribe` naming no transcriber; `speaker`
defaulting to `""` in one group and `None` everywhere else; Part 7 asserting the
exact contents of an ingest chain that Part 3 was always going to extend; and
`caption_words` becoming runnable again after a promotion, which `reconcile`
would act on (`test_promoting_a_video_does_not_make_its_captions_runnable_again`).

## End state a reviewer can check

1. `python -m pytest -q` is green with no optional extra installed, and
   `python -m pytest -rs` prints no skips from Part 8's modules.
2. `python -m ruff check rytp tests` and `python -m mypy rytp` are clean.
3. `python -m rytp tui` opens on a home view whose top line reads
   `F1 Help · F3 Speakers · F4 Search · F6 Transcripts · F7 Queue · F8 Cut lists`.
   Each key opens that screen; `Escape` returns; `F1` lists every binding in the
   application, screen-local ones included, generated rather than typed.
4. Selecting `transcribe.run` in the palette and pressing Enter **queues** a job
   and says so, naming `F7`. `F7` shows it, with the handler's note in full.
   Nothing long-running ever runs in the TUI process.
5. `F8` lists the cut lists on disk, opens one, and swapping an alternative,
   nudging a boundary and setting a pause all work, are undoable, and survive a
   save and reload through Part 5's own emitter and loader.
6. `tests/test_m1_end_to_end.py` walks design §11's M1 — two URLs, download,
   transcribe, align, index, search, a sentence assembled from **two** source
   videos, a render with a report naming both — with no network, no model and no
   binary.
7. Four consistency modules, roughly seventy assertions, each naming the
   cross-part invariant it holds. Running them after any change to a command, a
   flag, a job kind or the schema is how the next defect of this family gets
   found by a machine instead of by a person.

## Self-review

**Spec coverage.** Design §10's list of what the TUI must offer — "browsing
videos, reading transcripts, searching, playing hits, editing cut lists, adding
to the queue, watching job progress" plus speaker assignment — maps to: browsing
and playing (Parts 1 and 4, reached from home), transcripts and search (Part 4,
`F6`/`F4`), speakers (Part 7, `F3`), **editing cut lists** (Tasks 10–11, `F8`),
**adding to the queue** (Task 7), **watching job progress** (Tasks 8–9, `F7`).
Design §10's "each operation is defined once and both surfaces are generated from
that definition" is Task 1. Design §11's M1 is Task 12. Contracts §2's sentence
that "Part 8's job is to turn that reading into tests" is Tasks 1–4.

**Placeholders.** None: every step carries the code it asks for, every test
module is complete, and every "if this fails" names the module to fix rather than
saying "handle the error".

**Type consistency.** `JobsView` and `CutlistView` are constructed and used with
the same signatures in their headless tasks and their screen tasks.
`navigation.binding_rows()` returns `(key, action, description)` in Task 5 and is
consumed as exactly that in Task 6. `EnqueuePlan.values` is a `dict[str, Any]` in
Task 7 and is splatted into a handler in the same task. `CutlistSummary.path` is
what `CutlistPickerScreen` pushes into `CutlistScreen(db, path)`.

**Three things this plan deliberately does not do.**

- It does not add a field to `Command` for "which job kind this queues". That
  would be a contracts §5 change, and `rytp/tui/enqueue.py` gets the same result
  from parameters the commands already have.
- It does not amend any sibling plan. Every defect it found is recorded with its
  one-line fix and pinned by a test that fails until the fix lands, which is a
  finding an implementer can act on rather than an edit that can itself drift.
- It does not make the TUI run long work, however convenient that would be.
  Design §5 and §10 both say the worker is a separate process, and a TUI that
  held the GPU for forty minutes would be a worse tool than one that cannot.
