# rytp Part 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working `rytp` CLI that lazily resolves its data tree, creates and migrates the whole SQLite schema, catalogs channels and videos (URLs and local paths, no downloading), lists them with filters, removes them again — rows, files and queued jobs — reports on its own health through a registry later parts extend, and a Textual TUI that runs the same commands because both surfaces are generated from one command registry.

**Architecture:** One `COMMANDS` registry holds every operation as data (name, params, handler, result shape). `rytp/cli.py` turns that dict into a Typer app by synthesising a callback per command; `rytp/tui/palette.py` turns the same dict into a command palette. Handlers are plain functions taking an open `Database` plus keyword arguments and returning a `CommandResult` — they never print and never exit. Paths are resolved on every call from `RYTP_DATA` and directories are created only by the command that writes into them.

**Tech Stack:** Python 3.11+ (`from __future__ import annotations` everywhere), stdlib `sqlite3` with FTS5, Typer + Click for the CLI, Textual for the TUI, pytest for tests, ruff + mypy for the gate. No network, no heavy ML dependencies in this part.

**Spec:** `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md` (design) and `docs/superpowers/specs/2026-09-21-rytp-contracts.md` (binding shared interfaces — §3 schema, §4 types, §5 registry, §7 filesystem, §8 conventions).

## Global Constraints

- Python >= 3.11, 3.11 syntax. `from __future__ import annotations` in **every** module.
- SQLite only. No server databases, no Docker, no cloud APIs.
- Runs on Windows. No POSIX-only calls, no shell pipelines, no symlinks. `pathlib` everywhere; `subprocess` only with list arguments.
- Line length 100. `ruff` rules `E,F,I,B,UP,N,SIM,RUF`. `mypy` runs on `rytp/`.
- Heavy dependencies are optional extras, imported lazily inside functions, never at module import time.
- **No YouTube video ids, channel ids, channel names or URLs in any committed file, including tests and fixtures.** Use `VIDEO_A`, `CHANNEL_ONE`, `https://example.invalid/...`.
- Every tunable value lives in `rytp/constants.py` with a comment naming its design section. No inline magic numbers.
- Milliseconds, integers, everywhere. Database timestamps are `datetime.now(UTC).isoformat()` strings.
- Domain errors subclass `RytpError`. The CLI prints one line to stderr and exits 1 — never a traceback for an expected failure.
- Every stage takes an already-open `Database`; nothing opens its own connection. Multi-statement writes use `db.transaction()`.
- Tests: no network, ever. Every external binary is faked or the test is skipped.
- **Test invocation (pytest is not on `PATH`):**
  `python -m pytest <args>`
- One commit per task, conventional-commit prefix, present tense.

---

## File Structure

| File | Responsibility |
|---|---|
| `rytp/__init__.py` | Package marker. Imports nothing — importing `rytp` must have no side effects. |
| `rytp/constants.py` | Every tunable, each with a comment naming its design section. |
| `rytp/config.py` | `Paths` (pure path arithmetic), `paths()` (resolves `RYTP_DATA` on each call), `ensure_dir`. Creates nothing at import. |
| `rytp/models.py` | Contracts §4 dataclasses, `ChannelEntry`, `RytpError` + subclasses, `normalize_text` and `stem_text`. |
| `rytp/db/schema.py` | `MIGRATIONS: list[tuple[int, str]]` — the complete contracts §3 schema. |
| `rytp/db/__init__.py` | `Database`: connection setup, migration runner, `transaction()`. |
| `rytp/db/queries.py` | Parameterised SQL for channels, videos and settings. |
| `rytp/commands/__init__.py` | `Param`, `CommandResult`, `Command`, `COMMANDS`, `register`, `resolve`, the shared speaker resolver, the health-check registry, `doctor`, the `tui` launcher; imports command modules at the bottom. |
| `rytp/commands/catalog.py` | `channel.add`, `channel.list`, `channel.remove`, `channel.sync`, `videos.add`, `videos.list`, `videos.remove`. |
| `rytp/cli.py` | Typer app generated from `COMMANDS`; root callback; result and error rendering. |
| `rytp/__main__.py` | `python -m rytp` entry point; UTF-8 stdio. |
| `rytp/tui/palette.py` | `palette_entries()` — the same registry rendered for the TUI. |
| `rytp/tui/app.py` | Textual app: filterable palette + result table. |
| `tests/conftest.py` | `tmp_path` (honours `RYTP_TEST_TMP`), `data_dir`, `db`. |

---

## Tasks

### Task 1: A virtualenv anyone can create, on either platform

Nothing else in this plan can run until `python -m pytest` works, and the project ships on Windows (contracts §1). So the first deliverable is a bootstrap script per platform and a `pyproject.toml` whose base dependencies match contracts §1 — `typer`, `rich`, `textual`, `numpy`, `scipy`, `snowballstemmer`, all importable at module level, because audio analysis and stemming are core to the product rather than optional add-ons and a `.[dev]` install must be able to collect every test in the repository.

`scripts/` sits outside the contracts §2 package tree deliberately: these are developer entry points, not importable modules, and putting them inside `rytp/` would ship them in the wheel.

ffmpeg, ffprobe and yt-dlp are system binaries the owner installs himself. The script covers Python dependencies only.

**Files:**
- Create: `scripts/bootstrap.sh`, `scripts/bootstrap.ps1`
- Modify: `pyproject.toml`, `.gitignore`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a `.venv` at the repository root with `rytp` installed editable plus its `dev` extra, so every later step's `python -m pytest` works with no path juggling.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
"""How the project installs, asserted rather than assumed.

Contracts §1 fixes the base dependency set and requires that a plain
`.[dev]` install can collect the whole suite. These tests read
`pyproject.toml` instead of the installed environment, so they fail on
the declaration rather than on whichever machine happens to run them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Contracts §1: importable at module level, needed for the suite to collect.
BASE_DEPENDENCIES = {"typer", "rich", "textual", "numpy", "scipy", "snowballstemmer"}

# Contracts §1: transcribers, aligners and diarizers stay optional and are
# imported lazily inside functions.
HEAVY_DEPENDENCIES = {"yt-dlp", "faster-whisper", "pyannote-audio", "torch"}


def pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def names(requirements: list[str]) -> set[str]:
    """Distribution names, without version specifiers or extras."""
    return {re.split(r"[<>=!~;\[ ]", item, maxsplit=1)[0].strip().lower() for item in requirements}


def test_base_dependencies_match_contracts_section_1() -> None:
    assert names(pyproject()["project"]["dependencies"]) == BASE_DEPENDENCIES


def test_heavy_dependencies_are_not_base_dependencies() -> None:
    assert names(pyproject()["project"]["dependencies"]) & HEAVY_DEPENDENCIES == set()


def test_every_heavy_dependency_is_reachable_through_an_extra() -> None:
    extras = pyproject()["project"]["optional-dependencies"]
    offered: set[str] = set()
    for requirements in extras.values():
        offered |= names(requirements)
    assert offered >= HEAVY_DEPENDENCIES


def test_the_dev_extra_carries_the_toolchain() -> None:
    dev = names(pyproject()["project"]["optional-dependencies"]["dev"])
    assert {"pytest", "ruff", "mypy"} <= dev


def test_the_console_script_points_at_the_cli_entry_point() -> None:
    assert pyproject()["project"]["scripts"]["rytp"] == "rytp.cli:main"


def test_a_bootstrap_script_exists_for_each_platform() -> None:
    for name in ("bootstrap.sh", "bootstrap.ps1"):
        script = REPO_ROOT / "scripts" / name
        assert script.is_file(), f"missing {script}"
        body = script.read_text(encoding="utf-8")
        assert ".venv" in body
        assert '.[dev]' in body or '".[dev]"' in body


def test_the_shell_bootstrap_stops_on_the_first_error() -> None:
    body = (REPO_ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "set -eu" in body


def test_the_venv_is_not_committed() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".venv/" in ignored
    assert ".pytest_tmp/" in ignored
    assert ".smoke/" in ignored
```

- [ ] **Step 2: Run it and watch it fail**

You do not have a virtualenv yet, so bootstrap by hand this once — from the repository root, `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -e ".[dev]"` on macOS or Linux, `py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -e ".[dev]"` on Windows. Every later step assumes that virtualenv is activated.

Run: `python -m pytest tests/test_packaging.py -v`
Expected: failures — `scripts/bootstrap.sh` does not exist, and the dependency sets do not match.

- [ ] **Step 3: Rewrite `pyproject.toml`**

The whole file, so the dependency tiers, the toolchain configuration and the console script all land together. `numpy` and `scipy` move out of an extra and into the base set: part 3's `rytp/audio/` imports numpy at module level throughout, so with them behind an extra a `.[dev]` install could not even collect its tests.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rytp"
version = "0.2.0"
description = "Index a video archive by speaker and word; search it; cut and splice the results."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "rytp" }]
# Contracts §1 base dependencies: importable at module level, and required
# for the whole suite to collect. Audio analysis and stemming are core to
# the product, not optional add-ons.
dependencies = [
    "typer>=0.27",
    "rich>=13",
    "textual>=8",
    "numpy>=1.26",
    "scipy>=1.11",
    "snowballstemmer>=2.2",
]

[project.optional-dependencies]
# Contracts §1: transcribers, aligners and diarizers only. Everything here
# is imported lazily inside a function, never at module import time, so the
# base install still runs every command that does not need a model.
yt-dlp = ["yt-dlp>=2024.5"]
stt = ["faster-whisper>=1.0"]
diarize = ["pyannote-audio>=3.1", "torch>=2.1"]
all = [
    "yt-dlp>=2024.5",
    "faster-whisper>=1.0",
    "pyannote-audio>=3.1",
    "torch>=2.1",
]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
rytp = "rytp.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["rytp*"]
exclude = ["tests*", "data*", "scripts*"]

[tool.setuptools.package-data]
rytp = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM", "RUF"]
# The rule *selection* is fixed by contracts §1. These four ignores are
# not exceptions to a rule so much as to a heuristic:
#   N812  — `from rytp import constants as C` is the house alias, used in
#           every module. The rule only objects to the capital letter.
#   RUF001/2/3 — this is a Russian-language project. Every Cyrillic string
#           literal, docstring and test fixture trips the ambiguous-unicode
#           check (`е` looks like `e`), and CLAUDE.md is explicit that the
#           non-ASCII is deliberate.
ignore = ["N812", "RUF001", "RUF002", "RUF003"]

[tool.mypy]
python_version = "3.11"
strict_optional = true
warn_unused_ignores = true
warn_redundant_casts = true

# `rytp/commands/catalog.py` imports these lazily inside a try/except, so a
# missing yt-dlp extra becomes a one-line hint and a missing job registry
# becomes an empty tuple, instead of a traceback. Plan part 2 supplies both
# modules; until then their absence is not an error.
[[tool.mypy.overrides]]
module = ["rytp.acquire.*", "rytp.jobs", "rytp.jobs.*"]
ignore_missing_imports = true

# snowballstemmer ships neither stubs nor a py.typed marker.
[[tool.mypy.overrides]]
module = ["snowballstemmer.*"]
ignore_missing_imports = true
```

- [ ] **Step 4: Write `scripts/bootstrap.sh`**

```sh
#!/bin/sh
# Create .venv at the repository root and install rytp with its development
# dependencies. ffmpeg, ffprobe and yt-dlp are system binaries you install
# yourself; this script covers Python dependencies only.
set -eu

cd "$(dirname "$0")/.."

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"

echo "Done. Activate it with:  . .venv/bin/activate"
echo "Then run the suite with: python -m pytest"
```

- [ ] **Step 5: Write `scripts/bootstrap.ps1`**

```powershell
# Create .venv at the repository root and install rytp with its development
# dependencies. ffmpeg, ffprobe and yt-dlp are system binaries you install
# yourself; this script covers Python dependencies only.
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host "Done. Activate it with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the suite with: python -m pytest"
```

- [ ] **Step 6: Make sure the virtualenv and the scratch root stay out of git**

Add to `.gitignore` if either line is missing:

```gitignore
.venv/
.pytest_tmp/
.smoke/
```

- [ ] **Step 7: Run the test and watch it pass**

Run: `python -m pytest tests/test_packaging.py -v`
Expected: 8 passed.

- [ ] **Step 8: Prove the script works from nothing**

Delete the hand-made virtualenv and let the script rebuild it, so the thing the next person runs is the thing that was tested.

```sh
rm -rf .venv && sh scripts/bootstrap.sh && . .venv/bin/activate && python -m pytest tests/test_packaging.py -q
```

On Windows: `Remove-Item -Recurse -Force .venv; .\scripts\bootstrap.ps1; .\.venv\Scripts\Activate.ps1; python -m pytest tests/test_packaging.py -q`

Expected: the script finishes without error and the tests pass.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore scripts/bootstrap.sh scripts/bootstrap.ps1 tests/test_packaging.py
git commit -m "build: add a cross-platform bootstrap and fix the dependency tiers"
```

---

### Task 2: Reset the tree and the test harness

The existing `rytp/` package is the AI-generated implementation being replaced. It imports a `rytp.db` module that Task 5 turns into a package, so leaving it in place would break `ruff`, `mypy` and the suite for the rest of the plan. Delete it now. The old code stays readable at `git show 44fc214c:rytp/<path>` — design §12 says to keep the *patterns* (SQLite between every stage, the register-a-class engine registry, one-line CLI errors, `constants.py` as the only home for tunables, the migration runner), not the files.

**Files:**
- Delete: every tracked file under `rytp/` except `rytp/__init__.py` and `rytp/py.typed`; every tracked file under `tests/` except `tests/__init__.py`
- Modify: `rytp/__init__.py`
- Test: `tests/conftest.py`, `tests/test_harness.py`

**Interfaces:**
- Consumes: Task 1's virtualenv and `pyproject.toml`.
- Produces: `tests/conftest.py::_tmp_root() -> Path`; fixtures `tmp_path` (scratch dir under `RYTP_TEST_TMP`, default `<repo>/.pytest_tmp`) and `data_dir` (same dir, with `RYTP_DATA` pointed at it). The `db` fixture arrives in Task 5.

- [ ] **Step 1: Delete the superseded modules**

```bash
git rm -q rytp/__main__.py rytp/channels.py rytp/cli.py rytp/config.py rytp/constants.py \
  rytp/db.py rytp/engines.py rytp/loudnorm.py rytp/mine.py rytp/models.py rytp/speakers.py \
  rytp/spectrogram.py rytp/splice.py rytp/transcripts.py
git rm -q rytp/diarize/__init__.py rytp/diarize/base.py rytp/diarize/energy.py \
  rytp/diarize/none.py rytp/diarize/pyannote.py
git rm -q rytp/download/__init__.py rytp/download/queue.py rytp/download/ytdlp.py
git rm -q rytp/transcribe/__init__.py rytp/transcribe/base.py rytp/transcribe/chunking.py \
  rytp/transcribe/extract.py rytp/transcribe/faster_whisper.py rytp/transcribe/run.py
git rm -q rytp/tui/__init__.py rytp/tui/app.py rytp/tui/screens/__init__.py \
  rytp/tui/screens/speakers.py rytp/tui/screens/videos.py
```

- [ ] **Step 2: Delete the superseded tests**

```bash
git rm -q tests/conftest.py tests/test_channels.py tests/test_chunking.py tests/test_cli.py \
  tests/test_cli_download_format.py tests/test_config.py tests/test_db.py tests/test_diarize.py \
  tests/test_download.py tests/test_extract.py tests/test_faster_whisper.py \
  tests/test_local_video_pair.py tests/test_loudnorm.py tests/test_merge_audio_video.py \
  tests/test_mine.py tests/test_models.py tests/test_queue.py \
  tests/test_real_files_integration.py tests/test_speakers.py tests/test_spectrogram.py \
  tests/test_splice.py tests/test_transcribe_base.py tests/test_transcribe_run.py \
  tests/test_transcripts.py tests/test_ytdlp.py
find . -name '__pycache__' -type d -prune -exec rm -rf {} +
```

- [ ] **Step 3: Reduce `rytp/__init__.py` to a side-effect-free marker**

Importing `rytp` must do nothing but define the version. Everything else is imported explicitly.

```python
"""rytp — a searchable, cuttable index over a personal video archive.

Design: `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md`.
Binding interfaces: `docs/superpowers/specs/2026-09-21-rytp-contracts.md`.

This module deliberately imports nothing. Importing `rytp` must not read
the environment, touch the filesystem, or pull in an optional dependency.
"""

from __future__ import annotations

__version__ = "0.2.0"
```

- [ ] **Step 4: Write the failing harness test**

Create `tests/test_harness.py`:

```python
"""The test harness itself: where scratch files land, and where the data tree points."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import _tmp_root


def test_tmp_root_honours_rytp_test_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repo is usually on a slow SMB mount; RYTP_TEST_TMP moves test I/O off it."""
    monkeypatch.setenv("RYTP_TEST_TMP", os.path.join(os.sep, "somewhere", "fast"))
    assert _tmp_root() == Path(os.path.join(os.sep, "somewhere", "fast"))


def test_tmp_root_falls_back_to_the_workspace_local_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the override, behaviour is unchanged: <repo>/.pytest_tmp."""
    monkeypatch.delenv("RYTP_TEST_TMP", raising=False)
    root = _tmp_root()
    assert root.name == ".pytest_tmp"
    assert root.parent == Path(__file__).resolve().parent.parent


def test_tmp_path_is_a_fresh_directory_under_that_root(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
    assert not any(tmp_path.iterdir())
    assert tmp_path.parent == _tmp_root()


def test_data_dir_points_rytp_data_at_the_scratch_dir(data_dir: Path) -> None:
    assert os.environ["RYTP_DATA"] == str(data_dir)
```

- [ ] **Step 5: Run it and watch it fail**

Run: `python -m pytest tests/test_harness.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tests.conftest'` (the file was deleted in Step 2).

- [ ] **Step 6: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures.

Two environment variables shape the harness:

* ``RYTP_TEST_TMP`` — where scratch directories go. It defaults to
  ``<repo>/.pytest_tmp`` (gitignored), which exists because
  ``tempfile.mkdtemp`` followed by ``pathlib.Path.mkdir`` raises
  ``PermissionError`` in the owner's Windows sandbox. The override exists
  because the repository is normally an SMB mount and test I/O against it
  is slow: point it at a local disk.
* ``RYTP_DATA`` — where ``rytp.config`` roots the data tree. It is
  resolved on every call rather than at import, so the ``data_dir``
  fixture simply sets it; there is no module-level singleton to patch.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


def _tmp_root() -> Path:
    """Root directory for per-test scratch dirs."""
    override = os.environ.get("RYTP_TEST_TMP")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / ".pytest_tmp"


@pytest.fixture()
def tmp_path() -> Iterator[Path]:  # type: ignore[override]
    """Per-test scratch dir. Overrides pytest's built-in ``tmp_path``.

    ``os.makedirs`` rather than ``Path.mkdir`` is deliberate: the sandbox
    this project is developed in rejects the ``pathlib`` call path.
    """
    root = _tmp_root()
    os.makedirs(str(root), exist_ok=True)
    directory = root / f"t-{uuid.uuid4().hex[:8]}"
    os.makedirs(str(directory))
    try:
        yield directory
    finally:
        shutil.rmtree(str(directory), ignore_errors=True)


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the rytp data tree at a fresh scratch dir for this test."""
    monkeypatch.setenv("RYTP_DATA", str(tmp_path))
    yield tmp_path
```

- [ ] **Step 7: Run the test and watch it pass**

Run: `python -m pytest tests/test_harness.py -v`
Expected: 4 passed.

- [ ] **Step 8: Confirm the whole suite is green and nothing stale is collected**

Run: `python -m pytest -q`
Expected: 12 passed (4 here plus Task 1's 8), no errors, no import failures.

- [ ] **Step 9: Commit**

Never `git add -A` in this repository: the working tree has untracked scratch files, two of which are named after a real video id and must not be committed.

```bash
git add rytp/__init__.py tests/conftest.py tests/test_harness.py
git add -u rytp tests
git commit -m "chore: reset the package tree and make the test tmp root configurable"
```

---

### Task 3: Lazy paths and constants

Fixes a real bug. The old `rytp/config.py` built a module-level `Paths` singleton and called `.ensure()` on it while being imported, so importing `rytp.config` — which the CLI, the TUI and `conftest.py` all did — created a `data/` tree in whatever directory the process started in. Paths are now resolved per call and created only by the code that writes into them (contracts §7: "Created on demand by the command that needs it — **not at module import time**").

**Files:**
- Create: `rytp/constants.py`, `rytp/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `rytp.constants` — `DATA_ROOT_ENV_VAR`, `DEFAULT_DATA_DIRNAME`, `DB_FILENAME`, `MEDIA_DIRNAME`, `CACHE_DIRNAME`, `OUTPUT_DIRNAME`, `TRANSCRIPTS_DIRNAME`, `CUTLISTS_DIRNAME`, `MS_PER_SECOND`, `SQLITE_JOURNAL_MODE`, `SQLITE_BUSY_TIMEOUT_MS`, `CHANNEL_TABS`, `CHANNEL_TAB_KINDS`, `VIDEO_SOURCES`, `VIDEO_KINDS`, `REMOTE_SOURCE`, `LOCAL_SOURCE`, `DEFAULT_VIDEO_KIND`, `DEFAULT_LIST_LIMIT`, `MAX_LIST_LIMIT`, `TITLE_TRUNCATE_CHARS`, `NULL_CELL`, `HF_TOKEN_ENV_VARS`.
  - `rytp.config` — `Paths` (frozen dataclass: `root`, `.db`, `.media_dir(video_id)`, `.cache_wav(video_id)`, `.output_dir(render_id)`, `.transcript(video_id)`, `.cutlist(name)`), `data_root() -> Path`, `paths() -> Paths`, `ensure_dir(path: Path) -> Path`, `hf_token() -> str | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Paths are arithmetic; only `ensure_dir` touches the disk."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C

REPO_ROOT = Path(__file__).resolve().parent.parent


def child_env() -> dict[str, str]:
    """Environment for a subprocess import check: no RYTP_DATA, rytp importable."""
    env = dict(os.environ)
    env.pop("RYTP_DATA", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_importing_rytp_creates_no_directories(tmp_path: Path) -> None:
    """The old config.py mkdir'd a data tree at import time. It must not.

    This has to run in a subprocess with its own cwd: an in-process check
    from the repository root would pass against a `data/` left behind by
    the old behaviour.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import rytp, rytp.config, rytp.constants"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_data_root_defaults_to_data_under_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(C.DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    # `.resolve()`: Path.cwd() follows symlinks, and on macOS /tmp is one.
    assert config.data_root() == tmp_path.resolve() / "data"


def test_rytp_data_relocates_the_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(C.DATA_ROOT_ENV_VAR, str(tmp_path / "elsewhere"))
    assert config.data_root() == tmp_path / "elsewhere"
    assert config.paths().root == tmp_path / "elsewhere"


def test_rytp_data_is_re_read_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No singleton: changing the variable changes the next answer."""
    monkeypatch.setenv(C.DATA_ROOT_ENV_VAR, str(tmp_path / "one"))
    assert config.paths().root == tmp_path / "one"
    monkeypatch.setenv(C.DATA_ROOT_ENV_VAR, str(tmp_path / "two"))
    assert config.paths().root == tmp_path / "two"


def test_layout_matches_contracts_section_7(tmp_path: Path) -> None:
    p = config.Paths.from_root(tmp_path)
    assert p.db == tmp_path / "rytp.db"
    assert p.media_dir(7) == tmp_path / "media" / "7"
    assert p.cache_wav(7) == tmp_path / "cache" / "wav" / "7.wav"
    assert p.output_dir("r1") == tmp_path / "output" / "r1"
    assert p.transcript(7) == tmp_path / "transcripts" / "7.md"
    assert p.cutlist("monologue") == tmp_path / "cutlists" / "monologue.toml"


def test_building_paths_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    p = config.Paths.from_root(root)
    for candidate in (p.db, p.media_dir(1), p.cache_wav(1), p.output_dir("r"), p.transcript(1)):
        assert not candidate.exists()
    assert not root.exists()


def test_ensure_dir_creates_parents_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert config.ensure_dir(target) == target
    assert target.is_dir()
    assert config.ensure_dir(target) == target


def test_hf_token_prefers_hf_token_then_huggingface_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert config.hf_token() is None
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "second")
    assert config.hf_token() == "second"
    monkeypatch.setenv("HF_TOKEN", "first")
    assert config.hf_token() == "first"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.config'`.

- [ ] **Step 3: Write `rytp/constants.py`**

```python
"""Every tunable value in rytp, each with a comment naming its design section.

No other module may hold a magic number (contracts §1). Sections are
appended, never reordered — later plan parts add their own sections at
the end of this file.

Design: `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md`.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Filesystem layout (design §7 "Search"/"transcripts", contracts §7)
# ---------------------------------------------------------------------------

#: Environment variable that relocates the whole data tree. Contracts §7:
#: "Rooted at RYTP_DATA, default ./data".
DATA_ROOT_ENV_VAR: Final = "RYTP_DATA"

#: Directory used when RYTP_DATA is unset, relative to the working
#: directory (contracts §7).
DEFAULT_DATA_DIRNAME: Final = "data"

#: The one SQLite file. Design §3: "The database is the only source of truth."
DB_FILENAME: Final = "rytp.db"

#: Downloaded assets, one directory per video: media/{video_id}/… (contracts §7).
MEDIA_DIRNAME: Final = "media"

#: Regenerable, prunable working files: cache/wav/{video_id}.wav. Design §4
#: demoted the canonical WAV from a database column to a cache entry.
CACHE_DIRNAME: Final = "cache"

#: Render outputs: output/{render_id}/output.mp4 and report.md (design §9).
OUTPUT_DIRNAME: Final = "output"

#: Regenerable markdown transcripts: transcripts/{video_id}.md. Design §7
#: calls these "regenerable output, never input".
TRANSCRIPTS_DIRNAME: Final = "transcripts"

#: Hand-editable cut lists: cutlists/{name}.toml. Design §8 makes the cut
#: list "the durable representation".
CUTLISTS_DIRNAME: Final = "cutlists"

# ---------------------------------------------------------------------------
# Time (contracts §8)
# ---------------------------------------------------------------------------

#: Contracts §8: "Milliseconds, integers, everywhere." Anything that
#: arrives in seconds — yt-dlp durations, ffmpeg timings — is converted
#: through this, and nothing writes a bare 1000.
MS_PER_SECOND: Final = 1000

# ---------------------------------------------------------------------------
# SQLite connection (design §3, design §5, contracts §8)
# ---------------------------------------------------------------------------

#: Write-ahead logging, so an interactive command can read while the
#: worker writes. Design §5 runs three concurrency pools at once.
SQLITE_JOURNAL_MODE: Final = "WAL"

#: How long a connection waits for a write lock before giving up with
#: "database is locked". Five seconds comfortably covers a worker's
#: longest single write (a video's worth of word rows, design §5).
SQLITE_BUSY_TIMEOUT_MS: Final = 5_000

# ---------------------------------------------------------------------------
# Catalog (design §5 "Acquisition order", design §13 "Catalog size")
# ---------------------------------------------------------------------------

#: The channel listings that must each be enumerated separately. Design
#: §13: "Cataloguing must cover the streams and shorts listings as well
#: as the main one" — live streams are listed apart from videos, and
#: missing them undercounts the corpus. Walked in this order; on an id
#: that appears in two listings, the later listing decides its kind.
CHANNEL_TABS: Final = ("videos", "streams", "shorts")

#: videos.kind implied by the listing an entry came from (design §13).
CHANNEL_TAB_KINDS: Final = {
    "videos": "video",
    "streams": "livestream",
    "shorts": "short",
}

#: Allowed videos.source values. Mirrors the CHECK constraint in
#: contracts §3 so a surface can reject bad input before SQLite does.
VIDEO_SOURCES: Final = ("youtube", "ytdlp", "local")

#: Allowed videos.kind values. Mirrors contracts §3.
VIDEO_KINDS: Final = ("video", "short", "livestream", "other")

#: What anything reached through yt-dlp is recorded as. Design §5 treats
#: acquisition as one yt-dlp path regardless of site.
REMOTE_SOURCE: Final = "ytdlp"

#: What a registered local file is recorded as. Design §5: "Local files
#: register as assets and never get download jobs."
LOCAL_SOURCE: Final = "local"

#: kind assigned when the listing or probe says nothing better.
DEFAULT_VIDEO_KIND: Final = "video"

# ---------------------------------------------------------------------------
# Deletion (contracts §5 "Deletion")
# ---------------------------------------------------------------------------

#: The `JobKind.target_kind` value meaning "this job's target_id is a
#: videos.id". Removal derives the kinds to cancel from part 2's
#: registry rather than listing them (contracts §5): a hardcoded list
#: went stale before any code existed, because part 3 registers
#: `caption_words` on top of the eight that were obvious.
VIDEO_TARGET_KIND: Final = "video"

#: Tables that lose their rows by `ON DELETE CASCADE` when a video goes
#: (contracts §3). Counted before a removal so the report can say what
#: is about to disappear; never deleted by hand.
VIDEO_CASCADE_TABLES: Final = (
    "words",
    "utterances",
    "video_speakers",
    "assets",
    "video_acoustics",
)

# ---------------------------------------------------------------------------
# Health checks (contracts §5 "Health checks")
# ---------------------------------------------------------------------------

#: Oldest interpreter the project supports (contracts §1).
MIN_PYTHON_VERSION: Final = (3, 11)

#: Filename `doctor` writes and deletes to prove the data tree is
#: writable. Distinctive so a leftover is obviously ours.
WRITE_PROBE_FILENAME: Final = ".rytp-write-probe"

#: Status column values in the `doctor` report. Three, not two: a
#: `required=False` check that reports `ok=False` is advisory — a real
#: finding about a real absence that must not fail the command
#: (contracts §5).
CHECK_OK: Final = "ok"
CHECK_FAILED: Final = "FAILED"
CHECK_ADVISORY: Final = "advisory"

# ---------------------------------------------------------------------------
# Surfaces (design §10)
# ---------------------------------------------------------------------------

#: Default row cap on every `list` command, shared by the CLI and the TUI
#: because design §10 requires the two to show the same thing. A 1,600
#: video corpus would otherwise flood a terminal.
DEFAULT_LIST_LIMIT: Final = 50

#: Hard ceiling a caller may ask for, so `--limit 1000000` cannot page
#: the whole corpus into memory.
MAX_LIST_LIMIT: Final = 5_000

#: Title truncation in a rendered result table, in characters (design §10).
TITLE_TRUNCATE_CHARS: Final = 60

#: Printed in place of a NULL cell so columns stay aligned.
NULL_CELL: Final = "-"

# ---------------------------------------------------------------------------
# Engines (design §6)
# ---------------------------------------------------------------------------

#: Environment variables consulted for a Hugging Face token, in order.
#: Only the pyannote diarizer (design §6) needs one.
HF_TOKEN_ENV_VARS: Final = ("HF_TOKEN", "HUGGINGFACE_TOKEN")
```

- [ ] **Step 4: Write `rytp/config.py`**

```python
"""Where rytp keeps its files, and how it finds the environment.

Nothing here touches the filesystem at import time. The previous
implementation kept a module-level ``paths`` singleton whose constructor
called ``.ensure()``, so merely importing this module — which the CLI,
the TUI and the test suite all do — created a ``data/`` tree in whatever
directory the process started in. Contracts §7: paths are "created on
demand by the command that needs it — not at module import time".

``Paths`` is pure arithmetic. Call :func:`ensure_dir` immediately before
writing, on the directory you are about to write into.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rytp import constants as C


@dataclass(frozen=True, slots=True)
class Paths:
    """The contracts §7 layout, rooted at ``root``. Creates nothing."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str) -> Paths:
        return cls(root=Path(root))

    @property
    def db(self) -> Path:
        """The single SQLite file."""
        return self.root / C.DB_FILENAME

    def media_dir(self, video_id: int) -> Path:
        """Directory holding one video's downloaded assets."""
        return self.root / C.MEDIA_DIRNAME / str(video_id)

    def cache_wav(self, video_id: int) -> Path:
        """Regenerable 16 kHz mono WAV for one video."""
        return self.root / C.CACHE_DIRNAME / "wav" / f"{video_id}.wav"

    def output_dir(self, render_id: str) -> Path:
        """Directory holding one render's output and report."""
        return self.root / C.OUTPUT_DIRNAME / render_id

    def transcript(self, video_id: int) -> Path:
        """Regenerable markdown transcript for one video."""
        return self.root / C.TRANSCRIPTS_DIRNAME / f"{video_id}.md"

    def cutlist(self, name: str) -> Path:
        """A hand-editable cut list."""
        return self.root / C.CUTLISTS_DIRNAME / f"{name}.toml"


def data_root() -> Path:
    """Root of the data tree, re-read from the environment on every call."""
    override = os.environ.get(C.DATA_ROOT_ENV_VAR)
    if override:
        return Path(override)
    return Path.cwd() / C.DEFAULT_DATA_DIRNAME


def paths() -> Paths:
    """The current layout. Cheap; call it rather than caching it."""
    return Paths.from_root(data_root())


def ensure_dir(path: Path) -> Path:
    """Create ``path`` and its parents if absent, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_token() -> str | None:
    """The Hugging Face token, or ``None``. Only pyannote needs one."""
    for name in C.HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add rytp/constants.py rytp/config.py tests/test_config.py
git commit -m "feat: resolve data paths lazily instead of mkdir-ing at import"
```

---

### Task 4: Core types, errors, and the two text functions

Contracts §4 verbatim, plus the error base class every surface catches (contracts §8), plus `ChannelEntry` — the shape part 2's `rytp/acquire/ytdlp.py` returns from `enumerate_channel` and `probe_video`, defined here so part 1 can catalog against a fake and part 2 has nothing to invent.

Both text functions are **fully implemented here** (contracts §4), `stem_text` included. It is not in `rytp/index/`: `words.stem` is written by part 3 as words are created, so a stemmer living under the index package would invert the dependency, and there is no `rytp/index/stem.py`. Nor is it a stub — stubbing it and deferring the body to part 4 would create a build-order cycle, since part 3 calls it in production while part 4 would supply the body. It is a thin wrapper over `snowballstemmer`'s Russian stemmer, which Task 1 made a base dependency, so there is nothing to defer.

Two properties of the pair matter downstream and are pinned by tests here. Punctuation normalizes to a **space**, so `кто-то` becomes two tokens — the invariant every later part relies on is that the tokens stored for a piece of text are exactly `normalize_text(text).split()`. And stemming is **not idempotent** (`сказали` → `сказа` → `сказ`), which is why an utterance's `stem_text` is built by joining per-word `words.stem` values rather than by re-running the function over joined text.

**Files:**
- Create: `rytp/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RytpError`, `NotFoundError`, `InvalidInputError`, `RawWord`, `Span`, `DiarSegment`, `Fragment`, `ChannelEntry`, `normalize_text(text: str) -> str`, `stem_text(normalized: str) -> str`, `utc_now_iso() -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
"""Contracts §4 types, the error base class, and the text normalizer."""

from __future__ import annotations

import dataclasses
import re

import pytest

from rytp.models import (
    ChannelEntry,
    DiarSegment,
    Fragment,
    InvalidInputError,
    NotFoundError,
    RawWord,
    RytpError,
    Span,
    normalize_text,
    stem_text,
    utc_now_iso,
)


def test_every_core_type_is_a_frozen_dataclass() -> None:
    for cls in (RawWord, Span, DiarSegment, Fragment, ChannelEntry):
        assert dataclasses.is_dataclass(cls), cls
        assert cls.__dataclass_params__.frozen, cls


def test_core_type_fields_match_contracts_section_4() -> None:
    def names(cls: type) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    assert names(RawWord) == ["text", "start_ms", "end_ms", "confidence"]
    assert names(Span) == ["start_ms", "end_ms", "score"]
    assert names(DiarSegment) == ["start_ms", "end_ms", "local_label"]
    assert names(Fragment) == [
        "video_id",
        "first_word_ord",
        "last_word_ord",
        "start_ms",
        "end_ms",
        "text",
    ]
    assert names(ChannelEntry) == [
        "external_id",
        "title",
        "url",
        "duration_ms",
        "kind",
        "published_at",
    ]


def test_a_transcriber_may_emit_text_with_no_timings() -> None:
    """Contracts §4: every RawWord field but the text is optional."""
    word = RawWord(text="привет")
    assert word.start_ms is None
    assert word.end_ms is None
    assert word.confidence is None
    assert RawWord("привет", 0, 100, 0.9) == RawWord(
        text="привет", start_ms=0, end_ms=100, confidence=0.9
    )


def test_stem_text_is_importable_from_models() -> None:
    """Part 3 and part 4 both import it from here; nothing imports rytp.index."""
    assert stem_text.__module__ == "rytp.models"


def test_domain_errors_share_one_base() -> None:
    assert issubclass(NotFoundError, RytpError)
    assert issubclass(InvalidInputError, RytpError)
    assert issubclass(RytpError, Exception)


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize_text("  Привет   МИР \n") == "привет мир"


def test_normalize_drops_punctuation_and_leaves_a_word_boundary() -> None:
    assert normalize_text("Привет, мир!") == "привет мир"
    assert normalize_text("кто-то") == "кто то"


def test_normalize_folds_yo_to_ye() -> None:
    """Captions and ASR disagree about ё constantly; search must not care.

    The FTS tokenizer is `unicode61 remove_diacritics 0` precisely so it
    does NOT fold Cyrillic letters (it would wreck й), so the ё/е fold
    has to happen here, at write time.
    """
    assert normalize_text("ЁЖ и ещё") == "еж и еще"


def test_normalize_keeps_short_i_intact() -> None:
    """й must survive: it is a letter, not an и with a diacritic."""
    assert normalize_text("Мой") == "мой"


def test_normalize_is_idempotent() -> None:
    once = normalize_text("Привет, мир!")
    assert normalize_text(once) == once


def test_normalize_handles_empty_and_punctuation_only_input() -> None:
    assert normalize_text("") == ""
    assert normalize_text("  ...  ") == ""


def test_utc_now_iso_is_an_iso_utc_timestamp() -> None:
    stamp = utc_now_iso()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$", stamp), stamp


def test_stem_text_strips_russian_endings() -> None:
    """Contracts §4: real from the start — part 3 calls it to fill words.stem."""
    assert stem_text("сказали слово") == "сказа слов"
    assert stem_text("говорил") == "говор"
    assert stem_text("говорить") == "говор"


def test_stem_text_leaves_an_unstemmable_word_alone() -> None:
    assert stem_text("привет мир") == "привет мир"


def test_stem_text_handles_empty_input() -> None:
    assert stem_text("") == ""
    assert stem_text("   ") == ""


def test_stem_text_is_token_for_token() -> None:
    """One token in, one token out — utterances are joined, never re-stemmed."""
    normalized = normalize_text("Сказали кто-то слово")
    assert len(stem_text(normalized).split()) == len(normalized.split())


def test_stemming_is_not_idempotent() -> None:
    """Contracts §4 warns about this: never re-stem an already-stemmed string."""
    once = stem_text("сказали")
    assert stem_text(once) != once


def test_the_yo_fold_reaches_the_stem() -> None:
    """ещё and еще must land on the same stem, or search splits in two."""
    assert stem_text(normalize_text("ещё")) == stem_text(normalize_text("еще"))


def test_part_one_never_imports_the_index_package() -> None:
    """`rytp/index/` is part 4's. Checked in a subprocess so nothing else can mask it."""
    import subprocess
    import sys

    from tests.test_config import child_env

    probe = (
        "import sys, rytp, rytp.config, rytp.constants, rytp.models;"
        "assert not [n for n in sys.modules if n.startswith('rytp.index')], sys.modules.keys()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], env=child_env(), capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_frozen_types_reject_mutation() -> None:
    word = RawWord(start_ms=0, end_ms=100, text="да", confidence=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        word.text = "нет"  # type: ignore[misc]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_models.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.models'`.

- [ ] **Step 3: Write `rytp/models.py`**

```python
"""The types every stage shares, and the error every surface catches.

Contracts §4 fixes the dataclasses below; do not add or rename fields
without changing the contracts document first. ``ChannelEntry`` is the
one addition: it is what ``rytp/acquire/ytdlp.py`` (plan part 2) returns
from ``enumerate_channel`` and ``probe_video``, and it lives here so the
catalog commands can be written and tested before that module exists.

Both text functions are implemented here, not stubbed. Contracts §4 keeps
them in this module on purpose: ``words.stem`` is written by part 3 as
words are created, so putting the stemmer under ``rytp/index/`` would
invert the dependency. There is no ``rytp/index/stem.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

__all__ = [
    "ChannelEntry",
    "DiarSegment",
    "Fragment",
    "InvalidInputError",
    "NotFoundError",
    "RawWord",
    "RytpError",
    "Span",
    "normalize_text",
    "stem_text",
    "utc_now_iso",
]


class RytpError(Exception):
    """An expected failure with a message a user can act on.

    Contracts §8: the surface catches this, prints ``str(exc)`` as one
    line on stderr and exits 1. Never a traceback. Anything that is not
    a ``RytpError`` is a bug and is allowed to propagate.
    """


class NotFoundError(RytpError):
    """A referenced row, command or file does not exist."""


class InvalidInputError(RytpError):
    """Arguments were well-formed but wrong (bad enum value, empty id)."""


@dataclass(frozen=True)
class RawWord:
    """What a transcriber emits, before alignment.

    Both timings are optional: a transcriber may emit text only, leaving
    every boundary to the aligner. Timings, when present, are absolute
    against the source audio, never relative to a chunk.
    """

    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Span:
    """A refined time range for one word."""

    start_ms: int
    end_ms: int
    score: float | None


@dataclass(frozen=True)
class DiarSegment:
    """One diarizer segment, labelled per video, not globally."""

    start_ms: int
    end_ms: int
    local_label: str


@dataclass(frozen=True)
class Fragment:
    """One contiguous run taken from one video. The unit of a cut list."""

    video_id: int
    first_word_ord: int
    last_word_ord: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class ChannelEntry:
    """One video as a channel listing or a metadata probe describes it.

    ``kind`` is one of ``constants.VIDEO_KINDS``; a channel listing
    derives it from the tab it came from (design §13). There is no
    ``source`` field: everything reached through yt-dlp is recorded as
    ``constants.REMOTE_SOURCE``.
    """

    external_id: str
    title: str
    url: str
    duration_ms: int | None
    kind: str
    published_at: str | None


# ``\w`` under re.UNICODE keeps Cyrillic letters and digits; everything
# else becomes a space so "кто-то" splits into two searchable tokens.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)

# ё/Ё → е/Е. Russian sources spell these interchangeably and a search for
# "еще" must find "ещё". The FTS5 tokenizer cannot do this for us: it is
# configured `remove_diacritics 0` (contracts §3) because the alternative
# also folds й into и, which is a different letter.
_YO_FOLD = str.maketrans({"ё": "е", "Ё": "Е"})


def normalize_text(text: str) -> str:
    """Canonical form used by ``normalized_text`` columns and the FTS index.

    NFC-normalize, fold ё to е, lowercase, replace punctuation with a
    space, collapse whitespace, strip. Idempotent.
    """
    text = unicodedata.normalize("NFC", text).translate(_YO_FOLD).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _russian_stemmer() -> Any:
    """The snowball stemmer, built once. Constructing it is not free."""
    import snowballstemmer

    return snowballstemmer.stemmer("russian")


def stem_text(normalized: str) -> str:
    """Reduce every token of an already-normalized string to its Russian stem.

    Token for token: ``stem_text(s).split()`` is as long as ``s.split()``,
    which is what lets an utterance's ``stem_text`` be assembled by
    joining per-word ``words.stem`` values.

    **Not idempotent.** ``сказали`` stems to ``сказа``, which stems again
    to ``сказ``. Never re-stem an already-stemmed string (contracts §4).

    Pass the output of :func:`normalize_text`, not raw text: the ё fold
    and the punctuation split have to happen first or the two columns
    stop agreeing.
    """
    tokens = normalized.split()
    if not tokens:
        return ""
    return " ".join(_russian_stemmer().stemWords(tokens))


def utc_now_iso() -> str:
    """Now, as the ISO-8601 UTC string every timestamp column stores (contracts §8)."""
    return datetime.now(UTC).isoformat()
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/models.py tests/test_models.py
git commit -m "feat: add core types, RytpError and the text normalizer"
```

---

### Task 5: Database and the migration runner

The runner and its connection setup, proved against the first two migrations. Task 6 appends the other nine. `rytp/db.py` became the package `rytp/db/` in Task 2's deletion, so there is no module/package clash to worry about.

**Files:**
- Create: `rytp/db/__init__.py`, `rytp/db/schema.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `rytp.constants` (`SQLITE_JOURNAL_MODE`, `SQLITE_BUSY_TIMEOUT_MS`), `rytp.models.RytpError`.
- Produces:
  - `rytp.db.schema.MIGRATIONS: list[tuple[int, str]]`, `rytp.db.schema.LATEST_VERSION: int`.
  - `rytp.db.Database(path: Path)` with `.path`, `.conn` (`sqlite3.Connection`, `row_factory = sqlite3.Row`, `isolation_level=None`), `.migrate() -> int`, `.schema_version() -> int`, `.transaction()` context manager, `.close()`, `__enter__`/`__exit__`.
  - pytest fixture `db` — an opened, migrated `Database` on the `data_dir` tree.

- [ ] **Step 1: Write the failing test**

Create `tests/test_db.py`:

```python
"""The connection setup and the migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rytp.config import ensure_dir, paths
from rytp.db import Database, schema


def test_migration_versions_are_contiguous_and_ascending() -> None:
    """Guards the append-only rule: a duplicate or edited tuple shows up here."""
    versions = [version for version, _sql in schema.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert versions[-1] == schema.LATEST_VERSION


def test_migrate_reports_the_version_it_reached(data_dir: Path) -> None:
    database = Database(paths().db)
    try:
        assert database.schema_version() == 0
        assert database.migrate() == schema.LATEST_VERSION
        assert database.schema_version() == schema.LATEST_VERSION
    finally:
        database.close()


def test_migrate_is_idempotent(db: Database) -> None:
    assert db.migrate() == schema.LATEST_VERSION
    assert db.migrate() == schema.LATEST_VERSION


def test_channels_and_videos_exist_after_migrating(db: Database) -> None:
    names = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"schema_version", "channels", "videos"} <= names


def test_connection_pragmas(db: Database) -> None:
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_rows_are_indexable_by_column_name(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO channels (url, title) VALUES (?, ?)",
        ("https://example.invalid/c/CHANNEL_ONE", "Channel One"),
    )
    row = db.conn.execute("SELECT id, title FROM channels").fetchone()
    assert row["title"] == "Channel One"


def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO videos (source, kind, channel_id, external_id, title, created_at)"
            " VALUES ('ytdlp', 'video', 999, 'VIDEO_A', 'A', '2026-01-01T00:00:00+00:00')"
        )


def test_transaction_commits_on_success(db: Database) -> None:
    with db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/two', 'two')"
        )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 2


def test_transaction_rolls_back_on_error(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'dup')"
        )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_nested_transactions_join_the_outer_one(db: Database) -> None:
    """SQLite has no nested transactions; the inner block must not BEGIN again."""
    with db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        with db.transaction():
            db.conn.execute(
                "INSERT INTO channels (url, title)"
                " VALUES ('https://example.invalid/c/two', 'two')"
            )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 2


def test_an_inner_failure_rolls_back_the_whole_outer_transaction(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        db.conn.execute(
            "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'one')"
        )
        with db.transaction():
            db.conn.execute(
                "INSERT INTO channels (url, title)"
                " VALUES ('https://example.invalid/c/one', 'dup')"
            )
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_a_partly_migrated_database_is_brought_forward(data_dir: Path) -> None:
    """Simulates an older install: stop at version 1, reopen, finish."""
    ensure_dir(paths().root)
    first = Database(paths().db)
    try:
        first.migrate_to(1)
        assert first.schema_version() == 1
    finally:
        first.close()

    second = Database(paths().db)
    try:
        assert second.migrate() == schema.LATEST_VERSION
    finally:
        second.close()


def test_database_is_a_context_manager(data_dir: Path) -> None:
    ensure_dir(paths().root)
    with Database(paths().db) as database:
        database.migrate()
    with pytest.raises(sqlite3.ProgrammingError):
        database.conn.execute("SELECT 1")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_db.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.db'`.

- [ ] **Step 3: Write `rytp/db/schema.py` with the first two migrations**

```python
"""The rytp schema, as an ordered list of migrations (contracts §3).

Three rules, none of them negotiable:

* **One migration per table.** Small diffs review better than one
  500-line CREATE block, and a later version can add a column without
  disturbing what shipped.
* **Never edit a tuple that has run.** A migration that has executed on
  the owner's machine is history. Append a new one instead.
* **`schema_version` is not a migration.** The runner needs somewhere to
  record its progress before it can run anything, so it bootstraps that
  table itself.

The DDL text is copied verbatim from contracts §3. The only liberty
taken is ordering: `speakers` is created before `video_speakers` so the
foreign key is not a forward reference.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE channels (
            id              INTEGER PRIMARY KEY,
            url             TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL,
            last_synced_at  TEXT
        );
        """,
    ),
    (
        2,
        """
        CREATE TABLE videos (
            id            INTEGER PRIMARY KEY,
            source        TEXT NOT NULL CHECK (source IN ('youtube','ytdlp','local')),
            kind          TEXT NOT NULL CHECK (kind IN ('video','short','livestream','other')),
            channel_id    INTEGER REFERENCES channels(id),
            external_id   TEXT,
            url           TEXT,
            title         TEXT NOT NULL,
            duration_ms   INTEGER,
            published_at  TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL,
            UNIQUE (source, external_id)
        );
        CREATE INDEX videos_channel ON videos(channel_id);
        CREATE INDEX videos_source  ON videos(source);
        """,
    ),
]

#: The version a fully migrated database reports.
LATEST_VERSION: int = MIGRATIONS[-1][0]
```

- [ ] **Step 4: Write `rytp/db/__init__.py`**

```python
"""The one SQLite connection every stage is handed.

Design §3: "Every stage reads and writes SQLite; nothing is handed
between stages in memory." Contracts §8: "Every stage takes an open
``Database``; nothing opens its own connection." Only the surfaces
(``rytp/cli.py``, ``rytp/tui/app.py``) and the worker construct one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from rytp import constants as C
from rytp.db.schema import LATEST_VERSION, MIGRATIONS

__all__ = ["LATEST_VERSION", "MIGRATIONS", "Database"]


class Database:
    """An open SQLite connection plus the migration runner.

    ``isolation_level=None`` turns off :mod:`sqlite3`'s implicit
    transaction handling, so a bare ``db.conn.execute(...)`` autocommits
    and :meth:`transaction` controls its own BEGIN/COMMIT. Multi-statement
    writes must use :meth:`transaction`.

    ``check_same_thread=False`` lets the worker hand a connection between
    threads; callers are responsible for not sharing one concurrently.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        # PRAGMA does not accept bound parameters, hence the interpolation.
        # Both values come from rytp.constants, never from user input.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute(f"PRAGMA journal_mode = {C.SQLITE_JOURNAL_MODE}")
        self.conn.execute(f"PRAGMA busy_timeout = {int(C.SQLITE_BUSY_TIMEOUT_MS)}")

    # -- migrations --------------------------------------------------

    def schema_version(self) -> int:
        """The highest migration applied, or 0 on an untouched file."""
        try:
            row = self.conn.execute(
                "SELECT value FROM schema_version WHERE key = 'version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

    def migrate(self) -> int:
        """Apply every pending migration. Returns the version reached."""
        return self.migrate_to(LATEST_VERSION)

    def migrate_to(self, target: int) -> int:
        """Apply pending migrations up to and including ``target``.

        Only the tests stop short of :data:`LATEST_VERSION`; the runner
        needs the parameter so a partly-migrated database can be built
        deliberately.
        """
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_version (key, value) VALUES ('version', '0')"
        )
        current = self.schema_version()
        for version, sql in MIGRATIONS:
            if version <= current or version > target:
                continue
            # executescript() commits any open transaction before it runs,
            # so BEGIN/COMMIT has to live inside the script itself for the
            # DDL and the version bump to land together. `version` is an
            # int from MIGRATIONS, never user input.
            self.conn.executescript(
                f"BEGIN;\n{sql}\n"
                f"UPDATE schema_version SET value = '{int(version)}'"
                " WHERE key = 'version';\nCOMMIT;"
            )
        return self.schema_version()

    # -- transactions ------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """BEGIN / COMMIT, rolling back and re-raising on any exception.

        Re-entrant. SQLite has no nested transactions, and helpers that
        each wrap their own write are routinely called inside a larger
        one (``channel.sync`` upserts hundreds of videos in a single
        transaction). An inner block therefore joins the outer one; the
        outermost block owns the COMMIT and the ROLLBACK.
        """
        if self.conn.in_transaction:
            yield
            return
        self.conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # -- lifecycle ---------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
```

- [ ] **Step 5: Add the `db` fixture to `tests/conftest.py`**

Append to `tests/conftest.py`, and add the two imports at the top of the file (`from rytp.config import ensure_dir, paths` and `from rytp.db import Database`):

```python
@pytest.fixture()
def db(data_dir: Path) -> Iterator[Database]:
    """An opened, migrated Database on this test's data tree."""
    ensure_dir(paths().root)
    database = Database(paths().db)
    database.migrate()
    try:
        yield database
    finally:
        database.close()
```

- [ ] **Step 6: Run the test and watch it pass**

Run: `python -m pytest tests/test_db.py -v`
Expected: 13 passed.

- [ ] **Step 7: Commit**

```bash
git add rytp/db/__init__.py rytp/db/schema.py tests/conftest.py tests/test_db.py
git commit -m "feat: add the Database wrapper and the migration runner"
```

---

### Task 6: The rest of the schema, including FTS5 and its triggers

Every remaining table in contracts §3, which is explicit that **part 1 creates every table in that section**, including ones only later parts read or write. That includes columns part 1 never writes — `jobs.note`, which part 2's worker fills with a handler's non-fatal finding, and `video_speakers.engine`. `renders` is the clearest case: nothing in part 1 touches it, but part 6 builds `create_render`, a readiness predicate and `render.list` on top of it, and a missing table would surface as `no such table: renders` at render time. Later parts consume the schema; none of them add DDL. Note the two partial unique indexes on `assets` (one audio and one captions per video, renditions unconstrained) and `video_speakers.engine`, which records which diarizer produced a label. Two more details carry real weight and get their own tests: the FTS5 tokenizer must be `unicode61`, **never `porter`** (the Porter stemmer is English-only; Russian stemming happens in Python at write time), and `words.end_ms` may be NULL only for caption-sourced rows, because **cuttable is defined as `source = 'aligned'`** and nothing else may be cut.

**Files:**
- Modify: `rytp/db/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.db.schema.MIGRATIONS`.
- Produces: migrations 3–12 — `assets`, `speakers`, `video_speakers`, `words`, `utterances`, `utterances_fts` + `utterances_ai` / `utterances_ad` / `utterances_au`, `video_acoustics`, `jobs`, `settings`, `renders`. `LATEST_VERSION` becomes 12.

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema.py`:

```python
"""Contracts §3, as executable assertions."""

from __future__ import annotations

import sqlite3

import pytest

from rytp.db import Database

NOW = "2026-01-01T00:00:00+00:00"

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
    "words_stem",
    "words_speaker",
    "words_alignable",
    "utterances_video",
    "jobs_claim",
    "renders_cutlist",
}


def names(db: Database, kind: str) -> set[str]:
    return {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))
        # FTS5 keeps its own shadow tables (utterances_fts_data, _idx, …).
        if not row["name"].startswith("sqlite_")
    }


def seed_video(db: Database) -> int:
    db.conn.execute(
        "INSERT INTO channels (url, title) VALUES ('https://example.invalid/c/one', 'One')"
    )
    cur = db.conn.execute(
        "INSERT INTO videos (source, kind, channel_id, external_id, url, title,"
        " duration_ms, created_at)"
        " VALUES ('ytdlp', 'video', 1, 'VIDEO_A', 'https://example.invalid/w/VIDEO_A',"
        " 'A', 1000, ?)",
        (NOW,),
    )
    return int(cur.lastrowid)


def test_every_contract_table_exists(db: Database) -> None:
    assert names(db, "table") >= CONTRACT_TABLES


def test_every_contract_index_exists(db: Database) -> None:
    assert names(db, "index") >= CONTRACT_INDEXES


def test_no_dropped_table_survives(db: Database) -> None:
    """Design §4 "Dropped": these must not come back."""
    dropped = {
        "clips",
        "clip_features",
        "splice_runs",
        "splice_clips",
        "chunks",
        "queue_items",
        "videos_speaker_map",
        "transcribe_runs",
        "words_fts",
    }
    assert dropped & names(db, "table") == set()


def test_caption_words_may_have_no_end_time(db: Database) -> None:
    """Captions carry per-word starts and no ends (design §6)."""
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, NULL, 'да', 'да', 'да', 'caption', 'captions')",
        (video_id,),
    )
    assert db.conn.execute("SELECT COUNT(*) FROM words").fetchone()[0] == 1


def test_aligned_words_must_have_an_end_time(db: Database) -> None:
    """Cuttable is defined as source='aligned'; a cut needs both boundaries."""
    video_id = seed_video(db)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
            " source, engine) VALUES (?, 0, 0, NULL, 'да', 'да', 'да', 'aligned', 'mfa')",
            (video_id,),
        )


def test_the_three_transcript_tiers_are_accepted_and_nothing_else(db: Database) -> None:
    """Contracts §3: caption, timed, aligned. `timed` is searchable, not cuttable."""
    video_id = seed_video(db)
    for ord_, source in enumerate(("caption", "timed", "aligned")):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine) VALUES (?, ?, 0, 100, 'да', 'да', 'да', ?, 'e')",
            (video_id, ord_, source),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
            " stem, source, engine) VALUES (?, 9, 0, 100, 'да', 'да', 'да', 'guessed', 'e')",
            (video_id,),
        )


def test_the_alignable_index_covers_only_cuttable_words(db: Database) -> None:
    """The assembler reads cuttable words only; a full index would scale with captions."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'words_alignable'"
    ).fetchone()[0]
    assert "WHERE source = 'aligned'" in sql


def test_the_default_aligner_setting_ships_empty(db: Database) -> None:
    """Contracts §3: empty means ingest enqueues no align job at all."""
    row = db.conn.execute(
        "SELECT value FROM settings WHERE key = 'default_aligner'"
    ).fetchone()
    assert row is not None
    assert row["value"] == ""


def test_word_ordinals_are_unique_per_video(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')",
        (video_id,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        db.conn.execute(
            "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
            " source, engine) VALUES (?, 0, 100, 200, 'нет', 'нет', 'нет', 'aligned', 'mfa')",
            (video_id,),
        )


def test_the_fts_tokenizer_is_unicode61_not_porter(db: Database) -> None:
    """Porter is English-only; Russian stemming is done in Python (contracts §3)."""
    sql = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'utterances_fts'"
    ).fetchone()[0]
    assert "unicode61" in sql
    assert "remove_diacritics 0" in sql
    assert "porter" not in sql


def test_fts_mirrors_inserts_updates_and_deletes(db: Database) -> None:
    video_id = seed_video(db)

    def matches(expr: str) -> list[int]:
        return [
            row["rowid"]
            for row in db.conn.execute(
                "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?", (expr,)
            )
        ]

    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text)"
        " VALUES (?, 0, 500, 0, 1, 'Привет, мир!', 'привет мир', 'привет мир')",
        (video_id,),
    )
    assert matches('"привет мир"') == [1]

    db.conn.execute(
        "UPDATE utterances SET normalized_text = 'пока мир', stem_text = 'пок мир' WHERE id = 1"
    )
    assert matches('"привет мир"') == []
    assert matches('"пока мир"') == [1]

    db.conn.execute("DELETE FROM utterances WHERE id = 1")
    assert matches('"пока мир"') == []
    # An out-of-sync external-content index fails this; a clean one returns nothing.
    assert db.conn.execute(
        "INSERT INTO utterances_fts(utterances_fts) VALUES('integrity-check')"
    ).fetchall() == []


def test_fts_searches_the_two_columns_independently(db: Database) -> None:
    """Design §7: exact column first, stem column as the fallback."""
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text)"
        " VALUES (?, 0, 500, 0, 1, 'Сказали слово', 'сказали слово', 'сказа слов')",
        (video_id,),
    )
    exact = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сказали слово"',),
    ).fetchall()
    stemmed = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('stem_text : "сказа слов"',),
    ).fetchall()
    missing = db.conn.execute(
        "SELECT rowid FROM utterances_fts WHERE utterances_fts MATCH ?",
        ('normalized_text : "сказа слов"',),
    ).fetchall()
    assert len(exact) == 1
    assert len(stemmed) == 1
    assert missing == []


def test_deleting_a_video_cascades_to_its_rows(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text, stem,"
        " source, engine) VALUES (?, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')",
        (video_id,),
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord, last_word_ord,"
        " text, normalized_text, stem_text) VALUES (?, 0, 100, 0, 0, 'да', 'да', 'да')",
        (video_id,),
    )
    db.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    assert db.conn.execute("SELECT COUNT(*) FROM words").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM utterances").fetchone()[0] == 0


def test_a_video_has_at_most_one_audio_and_one_captions_asset(db: Database) -> None:
    """Design §4: one canonical audio, one captions, any number of renditions."""
    video_id = seed_video(db)

    def add_asset(role: str, path: str) -> None:
        db.conn.execute(
            "INSERT INTO assets (video_id, role, path, acquired_at) VALUES (?, ?, ?, ?)",
            (video_id, role, path, NOW),
        )

    add_asset("audio", "media/1/audio.m4a")
    add_asset("captions", "media/1/captions.json3")
    add_asset("video", "media/1/video-360.mp4")
    add_asset("video", "media/1/video-1080.mp4")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        add_asset("audio", "media/1/audio-better.opus")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        add_asset("captions", "media/1/captions-2.json3")
    assert db.conn.execute(
        "SELECT COUNT(*) FROM assets WHERE role = 'video'"
    ).fetchone()[0] == 2


def test_a_render_records_its_cut_list_and_state(db: Database) -> None:
    """Part 1 creates this table; part 6 is the only thing that writes to it."""
    db.conn.execute(
        "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at)"
        " VALUES ('monologue', 'pillarbox', 'planned', ?)",
        (NOW,),
    )
    row = db.conn.execute("SELECT * FROM renders").fetchone()
    assert row["cutlist_name"] == "monologue"
    assert row["output_path"] is None
    assert row["finished_at"] is None
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at)"
            " VALUES ('monologue', 'pillarbox', 'halfway', ?)",
            (NOW,),
        )


def test_a_job_is_unique_per_kind_and_target(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('download', 1, 'pending', 'network', ?)",
        (NOW,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 1, 'pending', 'network', ?)",
            (NOW,),
        )


def test_a_job_can_carry_a_non_fatal_note(db: Database) -> None:
    """Contracts §3: a handler's finding, recorded by the worker. `done`, not `failed`.

    Part 1 only creates the column; part 2's worker writes it and shows it
    in `jobs.list`. Without it a warning like "re-transcribing discarded
    this video's speaker mapping" reaches someone running the command by
    hand and nobody running the worker, which is the bulk path.
    """
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, note, created_at)"
        " VALUES ('transcribe', 1, 'done', 'gpu', 'discarded 2 speaker labels', ?)",
        (NOW,),
    )
    row = db.conn.execute("SELECT state, note FROM jobs").fetchone()
    assert row["state"] == "done"
    assert row["note"] == "discarded 2 speaker labels"


def test_a_job_note_is_optional(db: Database) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('download', 1, 'pending', 'network', ?)",
        (NOW,),
    )
    assert db.conn.execute("SELECT note FROM jobs").fetchone()["note"] is None


def test_job_state_and_pool_are_constrained(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 2, 'sleeping', 'network', ?)",
            (NOW,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
            " VALUES ('download', 3, 'pending', 'quantum', ?)",
            (NOW,),
        )


def test_a_speaker_label_is_unique_and_a_video_label_is_unique_per_video(db: Database) -> None:
    video_id = seed_video(db)
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
        " VALUES (?, 'S0', 1, 'pyannote')",
        (video_id,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, engine)"
            " VALUES (?, 'S0', 'pyannote')",
            (video_id,),
        )


def test_unmapping_a_speaker_keeps_the_video_label(db: Database) -> None:
    """Design §4: linking a label to a person updates one row, not thousands."""
    video_id = seed_video(db)
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Host', ?)", (NOW,))
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
        " VALUES (?, 'S0', 1, 'pyannote')",
        (video_id,),
    )
    db.conn.execute("DELETE FROM speakers WHERE id = 1")
    row = db.conn.execute("SELECT local_label, speaker_id FROM video_speakers").fetchone()
    assert row["local_label"] == "S0"
    assert row["speaker_id"] is None
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_schema.py -v`
Expected: most tests fail with `sqlite3.OperationalError: no such table: assets` / `words` / `utterances_fts`.

- [ ] **Step 3: Append migrations 3–12 to `rytp/db/schema.py`**

Append these tuples to `MIGRATIONS`, after the `videos` tuple and before the closing `]`. `speakers` deliberately precedes `video_speakers` so the foreign key is not a forward reference; the DDL text itself is verbatim from contracts §3.

```python
    (
        3,
        """
        CREATE TABLE assets (
            id          INTEGER PRIMARY KEY,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK (role IN ('audio','video','captions','container')),
            format_id   TEXT,
            path        TEXT NOT NULL,
            bytes       INTEGER,
            width       INTEGER,
            height      INTEGER,
            abr         REAL,
            acquired_at TEXT NOT NULL
        );
        CREATE INDEX assets_video_role ON assets(video_id, role);
        -- A video has at most one audio asset and at most one captions asset.
        -- Video renditions are unconstrained: many per video is the point.
        CREATE UNIQUE INDEX assets_one_audio    ON assets(video_id) WHERE role = 'audio';
        CREATE UNIQUE INDEX assets_one_captions ON assets(video_id) WHERE role = 'captions';
        """,
    ),
    (
        4,
        """
        CREATE TABLE speakers (
            id           INTEGER PRIMARY KEY,
            label        TEXT NOT NULL UNIQUE,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            notes        TEXT,
            created_at   TEXT NOT NULL
        );
        """,
    ),
    (
        5,
        """
        CREATE TABLE video_speakers (
            id          INTEGER PRIMARY KEY,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            local_label TEXT NOT NULL,
            speaker_id  INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
            embedding   BLOB,
            engine      TEXT NOT NULL,   -- which diarizer produced this label
            UNIQUE (video_id, local_label)
        );
        CREATE INDEX video_speakers_speaker ON video_speakers(speaker_id);
        """,
    ),
    (
        6,
        """
        CREATE TABLE words (
            id               INTEGER PRIMARY KEY,
            video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            ord              INTEGER NOT NULL,
            start_ms         INTEGER NOT NULL,
            end_ms           INTEGER,
            text             TEXT NOT NULL,
            normalized_text  TEXT NOT NULL,
            stem             TEXT NOT NULL,
            confidence       REAL,
            align_score      REAL,
            source           TEXT NOT NULL CHECK (source IN ('caption','timed','aligned')),
            engine           TEXT NOT NULL,
            video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
            UNIQUE (video_id, ord),
            CHECK (source = 'caption' OR end_ms IS NOT NULL)
        );
        CREATE INDEX words_video_ord  ON words(video_id, ord);
        CREATE INDEX words_normalized ON words(normalized_text);
        -- The assembler only ever looks at cuttable words, and most of the corpus
        -- will be caption-tier. Without this the hot lookup reads every row matching
        -- a token and filters afterwards, so its cost scales with the whole corpus
        -- rather than with the alignable part of it.
        CREATE INDEX words_alignable ON words(normalized_text) WHERE source = 'aligned';
        CREATE INDEX words_stem       ON words(stem);
        CREATE INDEX words_speaker    ON words(video_speaker_id);
        """,
    ),
    (
        7,
        """
        CREATE TABLE utterances (
            id               INTEGER PRIMARY KEY,
            video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
            start_ms         INTEGER NOT NULL,
            end_ms           INTEGER NOT NULL,
            first_word_ord   INTEGER NOT NULL,
            last_word_ord    INTEGER NOT NULL,
            text             TEXT NOT NULL,
            normalized_text  TEXT NOT NULL,
            stem_text        TEXT NOT NULL
        );
        CREATE INDEX utterances_video ON utterances(video_id, start_ms);
        """,
    ),
    (
        8,
        """
        CREATE VIRTUAL TABLE utterances_fts USING fts5(
            normalized_text, stem_text,
            content='utterances', content_rowid='id',
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TRIGGER utterances_ai AFTER INSERT ON utterances BEGIN
            INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
            VALUES (new.id, new.normalized_text, new.stem_text);
        END;
        CREATE TRIGGER utterances_ad AFTER DELETE ON utterances BEGIN
            INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
            VALUES ('delete', old.id, old.normalized_text, old.stem_text);
        END;
        CREATE TRIGGER utterances_au AFTER UPDATE ON utterances BEGIN
            INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
            VALUES ('delete', old.id, old.normalized_text, old.stem_text);
            INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
            VALUES (new.id, new.normalized_text, new.stem_text);
        END;
        """,
    ),
    (
        9,
        """
        CREATE TABLE video_acoustics (
            video_id             INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
            f0_mean              REAL,
            f0_std               REAL,
            spectral_tilt        REAL,
            noise_floor_db       REAL,
            reverb_proxy         REAL,
            loudness_lufs        REAL,
            computed_at          TEXT NOT NULL
        );
        """,
    ),
    (
        10,
        """
        CREATE TABLE jobs (
            id           INTEGER PRIMARY KEY,
            kind         TEXT NOT NULL,
            target_id    INTEGER NOT NULL,
            state        TEXT NOT NULL CHECK (state IN
                           ('pending','running','done','failed','blocked','cancelled')),
            pool         TEXT NOT NULL CHECK (pool IN ('network','gpu','cpu')),
            priority     INTEGER NOT NULL DEFAULT 0,
            attempts     INTEGER NOT NULL DEFAULT 0,
            not_before   TEXT,
            last_error   TEXT,
            note         TEXT,          -- non-fatal finding returned by the handler
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            finished_at  TEXT,
            UNIQUE (kind, target_id)
        );
        CREATE INDEX jobs_claim ON jobs(state, pool, priority, not_before);
        """,
    ),
    (
        11,
        """
        CREATE TABLE settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        -- `default_aligner` names the aligner that `ingest --transcribe` stamps onto
        -- the `align` jobs it creates. Empty means no alignment: ingest enqueues no
        -- `align` job at all and the words stay `timed` until you align them by hand.
        -- A job enqueued with an unregistered aligner name is rejected at enqueue
        -- time, not after five failed retries.
        INSERT INTO settings (key, value) VALUES ('default_aligner', '');
        """,
    ),
    (
        12,
        """
        -- A render targets a cut list, which is a named file with no integer
        -- identity, and jobs.target_id is an INTEGER. This table gives a render
        -- that identity, and gives render history somewhere to live.
        CREATE TABLE renders (
            id           INTEGER PRIMARY KEY,
            cutlist_name TEXT NOT NULL,
            output_path  TEXT,
            canvas_mode  TEXT NOT NULL,
            state        TEXT NOT NULL CHECK (state IN ('planned','rendered','failed')),
            created_at   TEXT NOT NULL,
            finished_at  TEXT
        );
        CREATE INDEX renders_cutlist ON renders(cutlist_name);
        """,
    ),
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_schema.py tests/test_db.py -v`
Expected: all pass; `test_migration_versions_are_contiguous_and_ascending` now covers 1–11.

- [ ] **Step 5: Commit**

```bash
git add rytp/db/schema.py tests/test_schema.py
git commit -m "feat: add the full contracts schema with the utterance FTS index"
```

---

### Task 7: Catalog queries

All SQL for channels, videos and settings in one place, so a command handler never concatenates SQL and the `videos.list` filters are written once. Later parts append their own sections to this file (assets, jobs, words); it is additive — never reorganise what is already there.

**Files:**
- Create: `rytp/db/queries.py`
- Test: `tests/test_queries.py`

**Interfaces:**
- Consumes: `rytp.db.Database`, `rytp.models` (`InvalidInputError`, `NotFoundError`, `utc_now_iso`), `rytp.constants` (`MAX_LIST_LIMIT`).
- Produces:
  - `insert_channel(db, *, url: str, title: str) -> tuple[int, bool]` — `(channel_id, created)`
  - `get_channel(db, channel_id: int) -> sqlite3.Row | None`
  - `find_channel(db, ref: str) -> sqlite3.Row | None` — by url, then by title
  - `list_channels(db, *, limit: int) -> list[sqlite3.Row]` — includes a derived `n_videos`
  - `mark_channel_synced(db, channel_id: int, when: str) -> None`
  - `upsert_video(db, *, source, kind, channel_id, external_id, url, title, duration_ms, published_at, metadata_json='{}') -> tuple[int, bool]`
  - `get_video(db, video_id: int) -> sqlite3.Row | None`
  - `list_videos(db, *, channel_id=None, kind=None, source=None, search=None, limit) -> list[sqlite3.Row]`
  - `get_setting(db, key, default=None) -> str | None`, `set_setting(db, key, value) -> None`
  - `clamp_limit(limit: int) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_queries.py`:

```python
"""Catalog SQL: channels, videos, settings."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import InvalidInputError

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def add_channel(db: Database, url: str = CHANNEL_URL, title: str = "Channel One") -> int:
    channel_id, _created = q.insert_channel(db, url=url, title=title)
    return channel_id


def add_video(db: Database, external_id: str, **overrides: object) -> int:
    kwargs: dict[str, object] = {
        "source": "ytdlp",
        "kind": "video",
        "channel_id": None,
        "external_id": external_id,
        "url": f"https://example.invalid/w/{external_id}",
        "title": f"Title {external_id}",
        "duration_ms": 60_000,
        "published_at": None,
    }
    kwargs.update(overrides)
    video_id, _created = q.upsert_video(db, **kwargs)  # type: ignore[arg-type]
    return video_id


def test_insert_channel_reports_creation_then_reuse(db: Database) -> None:
    first_id, created = q.insert_channel(db, url=CHANNEL_URL, title="Channel One")
    assert created is True
    second_id, created_again = q.insert_channel(db, url=CHANNEL_URL, title="Renamed")
    assert created_again is False
    assert second_id == first_id
    assert q.get_channel(db, first_id)["title"] == "Renamed"


def test_find_channel_by_url_then_by_title(db: Database) -> None:
    channel_id = add_channel(db)
    assert q.find_channel(db, CHANNEL_URL)["id"] == channel_id
    assert q.find_channel(db, "Channel One")["id"] == channel_id
    assert q.find_channel(db, "nothing like it") is None


def test_list_channels_counts_their_videos(db: Database) -> None:
    channel_id = add_channel(db)
    add_video(db, "VIDEO_A", channel_id=channel_id)
    add_video(db, "VIDEO_B", channel_id=channel_id)
    add_video(db, "VIDEO_C")
    rows = q.list_channels(db, limit=C.DEFAULT_LIST_LIMIT)
    assert len(rows) == 1
    assert rows[0]["n_videos"] == 2


def test_mark_channel_synced_records_the_timestamp(db: Database) -> None:
    channel_id = add_channel(db)
    assert q.get_channel(db, channel_id)["last_synced_at"] is None
    q.mark_channel_synced(db, channel_id, "2026-01-01T00:00:00+00:00")
    assert q.get_channel(db, channel_id)["last_synced_at"] == "2026-01-01T00:00:00+00:00"


def test_upsert_video_is_idempotent_on_source_and_external_id(db: Database) -> None:
    first_id, created = q.upsert_video(
        db,
        source="ytdlp",
        kind="video",
        channel_id=None,
        external_id="VIDEO_A",
        url="https://example.invalid/w/VIDEO_A",
        title="First title",
        duration_ms=1000,
        published_at=None,
    )
    assert created is True
    second_id, created_again = q.upsert_video(
        db,
        source="ytdlp",
        kind="livestream",
        channel_id=None,
        external_id="VIDEO_A",
        url="https://example.invalid/w/VIDEO_A",
        title="Corrected title",
        duration_ms=2000,
        published_at="2026-01-01T00:00:00+00:00",
    )
    assert created_again is False
    assert second_id == first_id
    row = q.get_video(db, first_id)
    assert row["title"] == "Corrected title"
    assert row["kind"] == "livestream"
    assert row["duration_ms"] == 2000
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_upsert_video_sets_created_at_once(db: Database) -> None:
    video_id = add_video(db, "VIDEO_A")
    first_seen = q.get_video(db, video_id)["created_at"]
    add_video(db, "VIDEO_A", title="Renamed")
    assert q.get_video(db, video_id)["created_at"] == first_seen


def test_upsert_video_rejects_a_missing_external_id(db: Database) -> None:
    """A NULL external_id defeats UNIQUE(source, external_id): every add duplicates."""
    with pytest.raises(InvalidInputError, match="external_id"):
        q.upsert_video(
            db,
            source="ytdlp",
            kind="video",
            channel_id=None,
            external_id=None,  # type: ignore[arg-type]
            url=None,
            title="No id",
            duration_ms=None,
            published_at=None,
        )


def test_the_same_external_id_can_exist_under_two_sources(db: Database) -> None:
    remote = add_video(db, "VIDEO_A")
    local, _ = q.upsert_video(
        db,
        source="local",
        kind="video",
        channel_id=None,
        external_id="VIDEO_A",
        url=None,
        title="A local file that happens to share a name",
        duration_ms=None,
        published_at=None,
    )
    assert remote != local


def test_list_videos_filters_by_channel_kind_source_and_search(db: Database) -> None:
    channel_id = add_channel(db)
    add_video(db, "VIDEO_A", channel_id=channel_id, title="Утренний эфир")
    add_video(db, "VIDEO_B", channel_id=channel_id, kind="livestream", title="Прямой эфир")
    add_video(db, "VIDEO_C", title="Unrelated")
    add_video(db, "VIDEO_D", source="local", title="From disk")

    def ids(**filters: object) -> list[str]:
        rows = q.list_videos(db, limit=C.DEFAULT_LIST_LIMIT, **filters)  # type: ignore[arg-type]
        return [row["external_id"] for row in rows]

    assert sorted(ids(channel_id=channel_id)) == ["VIDEO_A", "VIDEO_B"]
    assert ids(kind="livestream") == ["VIDEO_B"]
    assert ids(source="local") == ["VIDEO_D"]
    assert ids(search="эфир") == ["VIDEO_A", "VIDEO_B"] or ids(search="эфир") == [
        "VIDEO_B",
        "VIDEO_A",
    ]
    assert ids(channel_id=channel_id, kind="video") == ["VIDEO_A"]


def test_list_videos_orders_newest_first_and_respects_the_limit(db: Database) -> None:
    for name in ("VIDEO_A", "VIDEO_B", "VIDEO_C"):
        add_video(db, name)
    rows = q.list_videos(db, limit=2)
    assert [row["external_id"] for row in rows] == ["VIDEO_C", "VIDEO_B"]


def test_clamp_limit_keeps_the_query_bounded() -> None:
    assert q.clamp_limit(0) == 1
    assert q.clamp_limit(-5) == 1
    assert q.clamp_limit(10) == 10
    assert q.clamp_limit(C.MAX_LIST_LIMIT + 1) == C.MAX_LIST_LIMIT


def test_settings_round_trip(db: Database) -> None:
    assert q.get_setting(db, "download.daily_cap") is None
    assert q.get_setting(db, "download.daily_cap", "200") == "200"
    q.set_setting(db, "download.daily_cap", "50")
    assert q.get_setting(db, "download.daily_cap") == "50"
    q.set_setting(db, "download.daily_cap", "75")
    assert q.get_setting(db, "download.daily_cap") == "75"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_queries.py -v`
Expected: collection error — `ImportError: cannot import name 'queries' from 'rytp.db'`.

- [ ] **Step 3: Write `rytp/db/queries.py`**

```python
"""Parameterised SQL for the catalog tables.

Command handlers call these; they never build SQL themselves. Later plan
parts append their own sections (assets, jobs, words, utterances) — this
file is additive, so do not reorganise what is already here.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from rytp import constants as C
from rytp.db import Database
from rytp.models import InvalidInputError, utc_now_iso

__all__ = [
    "clamp_limit",
    "find_channel",
    "get_channel",
    "get_setting",
    "get_video",
    "insert_channel",
    "list_channels",
    "list_videos",
    "mark_channel_synced",
    "set_setting",
    "upsert_video",
]


def clamp_limit(limit: int) -> int:
    """Keep a caller from paging the whole corpus into memory (design §10)."""
    return max(1, min(int(limit), C.MAX_LIST_LIMIT))


# -- channels ---------------------------------------------------------


def insert_channel(db: Database, *, url: str, title: str) -> tuple[int, bool]:
    """Register a channel by URL. Returns ``(channel_id, created)``."""
    with db.transaction():
        existing = db.conn.execute(
            "SELECT id FROM channels WHERE url = ?", (url,)
        ).fetchone()
        if existing is not None:
            db.conn.execute(
                "UPDATE channels SET title = ? WHERE id = ?", (title, existing["id"])
            )
            return int(existing["id"]), False
        cur = db.conn.execute(
            "INSERT INTO channels (url, title) VALUES (?, ?)", (url, title)
        )
        new_id = cur.lastrowid
        assert new_id is not None  # an INSERT always sets it
        return new_id, True


def get_channel(db: Database, channel_id: int) -> sqlite3.Row | None:
    return db.conn.execute(
        "SELECT * FROM channels WHERE id = ?", (channel_id,)
    ).fetchone()


def find_channel(db: Database, ref: str) -> sqlite3.Row | None:
    """Look a channel up by URL, then by exact title."""
    row = db.conn.execute("SELECT * FROM channels WHERE url = ?", (ref,)).fetchone()
    if row is not None:
        return row
    return db.conn.execute("SELECT * FROM channels WHERE title = ?", (ref,)).fetchone()


def list_channels(db: Database, *, limit: int) -> list[sqlite3.Row]:
    """Channels newest first, each with the number of videos catalogued."""
    return db.conn.execute(
        """
        SELECT c.id, c.url, c.title, c.last_synced_at,
               (SELECT COUNT(*) FROM videos v WHERE v.channel_id = c.id) AS n_videos
        FROM channels c
        ORDER BY c.id DESC
        LIMIT ?
        """,
        (clamp_limit(limit),),
    ).fetchall()


def mark_channel_synced(db: Database, channel_id: int, when: str) -> None:
    db.conn.execute(
        "UPDATE channels SET last_synced_at = ? WHERE id = ?", (when, channel_id)
    )


# -- videos -----------------------------------------------------------


def upsert_video(
    db: Database,
    *,
    source: str,
    kind: str,
    channel_id: int | None,
    external_id: str,
    url: str | None,
    title: str,
    duration_ms: int | None,
    published_at: str | None,
    metadata_json: str = "{}",
) -> tuple[int, bool]:
    """Insert or refresh a catalog row. Returns ``(video_id, created)``.

    The natural key is ``(source, external_id)``. For a remote video that
    is the site's video id; for ``source='local'`` it is the resolved
    absolute path, which is what makes re-registering the same file
    idempotent — ``videos`` has no path column (design §4: "All path and
    download columns move out").

    ``created_at`` is written once and never touched again.
    """
    if not external_id:
        raise InvalidInputError(
            "a video needs an external_id: the site's video id, or the absolute "
            "path for a local file. Without one, UNIQUE (source, external_id) "
            "cannot dedupe and every add would create a new row."
        )
    with db.transaction():
        existing = db.conn.execute(
            "SELECT id FROM videos WHERE source = ? AND external_id = ?",
            (source, external_id),
        ).fetchone()
        if existing is not None:
            db.conn.execute(
                """
                UPDATE videos
                   SET kind = ?, channel_id = ?, url = ?, title = ?,
                       duration_ms = ?, published_at = ?, metadata_json = ?
                 WHERE id = ?
                """,
                (
                    kind,
                    channel_id,
                    url,
                    title,
                    duration_ms,
                    published_at,
                    metadata_json,
                    existing["id"],
                ),
            )
            return int(existing["id"]), False
        cur = db.conn.execute(
            """
            INSERT INTO videos (source, kind, channel_id, external_id, url, title,
                                duration_ms, published_at, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                kind,
                channel_id,
                external_id,
                url,
                title,
                duration_ms,
                published_at,
                metadata_json,
                utc_now_iso(),
            ),
        )
        new_id = cur.lastrowid
        assert new_id is not None  # an INSERT always sets it
        return new_id, True


def get_video(db: Database, video_id: int) -> sqlite3.Row | None:
    return db.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def list_videos(
    db: Database,
    *,
    channel_id: int | None = None,
    kind: str | None = None,
    source: str | None = None,
    search: str | None = None,
    limit: int,
) -> list[sqlite3.Row]:
    """Catalog rows newest first, narrowed by whichever filters are given.

    ``search`` is a substring match on the title — deliberately dumb.
    Real search is the utterance FTS index (design §7); this only helps
    you find a video you half remember.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if channel_id is not None:
        clauses.append("v.channel_id = ?")
        params.append(channel_id)
    if kind is not None:
        clauses.append("v.kind = ?")
        params.append(kind)
    if source is not None:
        clauses.append("v.source = ?")
        params.append(source)
    if search:
        clauses.append("v.title LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(clamp_limit(limit))
    return db.conn.execute(
        f"""
        SELECT v.*, c.title AS channel_title
        FROM videos v
        LEFT JOIN channels c ON c.id = v.channel_id
        {where}
        ORDER BY v.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


# -- settings ---------------------------------------------------------


def get_setting(db: Database, key: str, default: str | None = None) -> str | None:
    """Read one operational value (design §5: everything tunable lives here)."""
    row = db.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else default


def set_setting(db: Database, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_queries.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/db/queries.py tests/test_queries.py
git commit -m "feat: add catalog and settings queries"
```

---

### Task 8: The command registry

The most novel piece in part 1, and the one design §10 turns on: "Each operation is defined once — name, arguments, result shape — and both the CLI and the TUI are generated from that definition. Alignment between the two then holds by construction rather than by discipline."

Contracts §5 fixes the four types below, `Command.cli_only` included. Do not rename a field or add a required one. Everything else in this task is the validation that makes a registration a programming error rather than a surprise at the surface — plus the shared speaker resolver, which contracts §5 also puts in this module: there are two speaker identifier spaces, conflating them is a bug, and no part may roll its own resolution.

**Files:**
- Create: `rytp/commands/__init__.py`
- Test: `tests/test_registry.py`, `tests/test_speaker_filter.py`

**Interfaces:**
- Consumes: `rytp.models` (`InvalidInputError`, `NotFoundError`), `rytp.db.Database`.
- Produces: `REQUIRED`, `Param`, `CommandResult`, `Command` (with `long_running` and `cli_only`), `COMMANDS`, `register(cmd) -> Command`, `resolve(name) -> Command`, `leaf_name(cmd) -> str`, `cli_path(cmd) -> tuple[str, ...]`, `PARAM_TYPES`.
- Also produces the shared speaker machinery, whose signature contracts §5 pins because three parts had assumed three different ones: `PARAM_ALIASES`, `SPEAKER_PARAMS`, `SpeakerFilter(video_speaker_ids: frozenset[int], description: str)`, `resolve_speaker_filter(db, *, speaker=None, video_local_speaker=None, video_id=None) -> SpeakerFilter | None`. Parts 4, 5 and 7 splice `SPEAKER_PARAMS` into their commands and call the resolver; none of them define their own, and none of them write the join from `speakers.id` to `video_speakers.id` — this function has already done it.
- A handler's contract: `handler(db: Database, **kwargs) -> CommandResult`, keyword names matching `Param.name`. Handlers never print and never call `sys.exit`; they raise `RytpError` and the surface formats it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
"""The one definition both surfaces are generated from (contracts §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import commands
from rytp.commands import (
    REQUIRED,
    Command,
    CommandResult,
    Param,
    cli_path,
    leaf_name,
    register,
    resolve,
)
from rytp.models import NotFoundError


@pytest.fixture()
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Command]:
    """An empty COMMANDS for this test, so registrations don't leak."""
    fresh: dict[str, Command] = {}
    monkeypatch.setattr(commands, "COMMANDS", fresh)
    return fresh


def noop(db: object, **kwargs: object) -> CommandResult:
    return CommandResult(message="ok")


def make(
    name: str,
    group: str,
    *params: Param,
    long_running: bool = False,
    cli_only: bool = False,
) -> Command:
    return Command(
        name=name,
        group=group,
        summary=f"Summary for {name}.",
        params=params,
        handler=noop,
        long_running=long_running,
        cli_only=cli_only,
    )


def test_register_stores_and_returns_the_command(registry: dict[str, Command]) -> None:
    cmd = register(make("videos.list", "videos"))
    assert registry["videos.list"] is cmd
    assert resolve("videos.list") is cmd


def test_resolve_names_the_alternatives_when_it_fails(registry: dict[str, Command]) -> None:
    register(make("videos.list", "videos"))
    register(make("channel.list", "channel"))
    with pytest.raises(NotFoundError) as excinfo:
        resolve("video.lst")
    message = str(excinfo.value)
    assert "video.lst" in message
    assert "channel.list" in message
    assert "videos.list" in message


def test_a_top_level_command_has_an_empty_group(registry: dict[str, Command]) -> None:
    cmd = register(make("ingest", ""))
    assert cli_path(cmd) == ("ingest",)
    assert leaf_name(cmd) == "ingest"


def test_a_grouped_command_maps_to_two_path_segments(registry: dict[str, Command]) -> None:
    cmd = register(make("videos.add", "videos"))
    assert cli_path(cmd) == ("videos", "add")
    assert leaf_name(cmd) == "add"


def test_registering_the_same_name_twice_is_an_error(registry: dict[str, Command]) -> None:
    register(make("videos.list", "videos"))
    with pytest.raises(ValueError, match="already registered"):
        register(make("videos.list", "videos"))


def test_the_group_must_match_the_dotted_name(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="group"):
        register(make("videos.list", "channel"))
    with pytest.raises(ValueError, match="group"):
        register(make("ingest", "videos"))


def test_names_are_lowercase_dotted_identifiers(registry: dict[str, Command]) -> None:
    for bad in ("Videos.List", "videos..list", "videos.", ".list", "videos list", ""):
        with pytest.raises(ValueError, match="name"):
            register(make(bad, bad.split(".")[0]))


def test_parameter_names_must_be_unique(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="duplicate parameter"):
        register(
            make(
                "videos.list",
                "videos",
                Param("limit", int, "Rows.", default=50),
                Param("limit", int, "Rows again.", default=50),
            )
        )


def test_positional_parameters_come_first(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="positional"):
        register(
            make(
                "videos.add",
                "videos",
                Param("title", str, "Title.", default=None),
                Param("target", str, "URL or path.", positional=True),
            )
        )


def test_a_parameter_type_must_be_one_the_surfaces_can_render(
    registry: dict[str, Command],
) -> None:
    with pytest.raises(ValueError, match="type"):
        register(make("videos.add", "videos", Param("when", dict, "Nope.", default=None)))
    register(
        make(
            "videos.add",
            "videos",
            Param("target", Path, "A path.", positional=True),
            Param("limit", int, "Rows.", default=1),
            Param("ratio", float, "A ratio.", default=1.0),
            Param("force", bool, "A flag.", default=False),
        )
    )


def test_choices_are_only_meaningful_for_strings(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="choices"):
        register(
            make("videos.list", "videos", Param("kind", int, "Kind.", default=1, choices=("a",)))
        )


def test_a_short_flag_is_a_single_dash_letter(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="short"):
        register(make("videos.list", "videos", Param("limit", int, "Rows.", default=1, short="n")))
    with pytest.raises(ValueError, match="short"):
        register(
            make("videos.list", "videos", Param("limit", int, "Rows.", default=1, short="--n"))
        )


def test_a_positional_parameter_may_not_carry_a_short_flag(
    registry: dict[str, Command],
) -> None:
    with pytest.raises(ValueError, match="short"):
        register(
            make("videos.add", "videos", Param("target", str, "T.", positional=True, short="-t"))
        )


def test_every_command_needs_a_summary(registry: dict[str, Command]) -> None:
    with pytest.raises(ValueError, match="summary"):
        register(
            Command(
                name="videos.list",
                group="videos",
                summary="",
                params=(),
                handler=noop,
            )
        )


def test_required_is_distinguishable_from_none(registry: dict[str, Command]) -> None:
    """`default=None` means optional-and-empty; REQUIRED means the user must supply it."""
    required = Param("target", str, "T.", positional=True)
    optional = Param("title", str, "Title.", default=None)
    assert required.default is REQUIRED
    assert optional.default is None


def test_command_result_defaults_to_empty(registry: dict[str, Command]) -> None:
    result = CommandResult()
    assert result.columns == ()
    assert result.rows == ()
    assert result.message is None


def test_long_running_defaults_to_false(registry: dict[str, Command]) -> None:
    assert register(make("videos.list", "videos")).long_running is False
    assert register(make("channel.sync", "channel", long_running=True)).long_running is True


def test_cli_only_defaults_to_false(registry: dict[str, Command]) -> None:
    """Contracts §5: a surface launcher is a registry command the TUI hides."""
    assert register(make("videos.list", "videos")).cli_only is False
    assert register(make("tui", "", cli_only=True)).cli_only is True
```

- [ ] **Step 2: Write the failing resolver test**

Create `tests/test_speaker_filter.py`:

```python
"""Two speaker identifier spaces, one resolver (contracts §5)."""

from __future__ import annotations

import pytest

from rytp.commands import SPEAKER_PARAMS, SpeakerFilter, resolve_speaker_filter
from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture()
def roster(db: Database) -> Database:
    """Two videos, four diarized labels, three named people, one with aliases.

    `Ведущий` is mapped in both videos, `Гость` in one, `Призрак` in none
    — the roster entry that resolves but matches nothing.
    """
    for external_id in ("VIDEO_A", "VIDEO_B"):
        db.conn.execute(
            "INSERT INTO videos (source, kind, external_id, title, created_at)"
            " VALUES ('ytdlp', 'video', ?, ?, ?)",
            (external_id, external_id, NOW),
        )
    db.conn.execute(
        "INSERT INTO speakers (label, aliases_json, created_at)"
        " VALUES ('Ведущий', '[\"Host\", \"ведущий канала\"]', ?)",
        (NOW,),
    )
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Гость', ?)", (NOW,))
    db.conn.execute("INSERT INTO speakers (label, created_at) VALUES ('Призрак', ?)", (NOW,))
    rows = [
        (1, "SPEAKER_00", 1),  # video_speakers.id 1
        (1, "SPEAKER_01", 2),  # 2
        (1, "SPEAKER_02", None),  # 3 — an unnamed voice
        (2, "SPEAKER_00", 1),  # 4 — the host again, in the other video
    ]
    for video_id, label, speaker_id in rows:
        db.conn.execute(
            "INSERT INTO video_speakers (video_id, local_label, speaker_id, engine)"
            " VALUES (?, ?, ?, 'pyannote')",
            (video_id, label, speaker_id),
        )
    return db


# -- rule 1: nothing asked for means no filter ------------------------


def test_no_speaker_named_returns_none(roster: Database) -> None:
    """Contracts §5 rule 1: the "show everything" case, distinct from a miss."""
    assert resolve_speaker_filter(roster) is None


def test_a_video_alone_is_not_a_speaker_filter(roster: Database) -> None:
    assert resolve_speaker_filter(roster, video_id=1) is None


# -- rule 3: the expansion happens here -------------------------------


def test_a_roster_name_expands_to_every_video_speaker_row(roster: Database) -> None:
    """Consumers filter words.video_speaker_id; the join belongs here, not there."""
    result = resolve_speaker_filter(roster, speaker="Ведущий")
    assert result is not None
    assert result.video_speaker_ids == frozenset({1, 4})
    assert result.description == 'speaker "Ведущий"'


def test_an_alias_resolves_to_the_same_rows_and_the_canonical_label(
    roster: Database,
) -> None:
    by_alias = resolve_speaker_filter(roster, speaker="Host")
    by_label = resolve_speaker_filter(roster, speaker="Ведущий")
    assert by_alias is not None and by_label is not None
    assert by_alias.video_speaker_ids == by_label.video_speaker_ids
    assert by_alias.description == 'speaker "Ведущий"'


def test_a_roster_name_can_be_narrowed_to_one_video(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Ведущий", video_id=2)
    assert result is not None
    assert result.video_speaker_ids == frozenset({4})
    assert "in video 2" in result.description


# -- rule 2: empty means nothing, not everything ----------------------


def test_an_unmapped_person_resolves_to_an_empty_set(roster: Database) -> None:
    """Contracts §5 rule 2: a real person nobody has mapped to a video yet."""
    result = resolve_speaker_filter(roster, speaker="Призрак")
    assert result is not None
    assert result.video_speaker_ids == frozenset()
    assert result.description == 'speaker "Призрак"'


def test_an_empty_result_is_distinguishable_from_no_filter(roster: Database) -> None:
    """The bug this guards: `IN ()` silently matching everything."""
    unmapped = resolve_speaker_filter(roster, speaker="Призрак")
    assert unmapped is not None  # not None: a filter WAS requested
    assert not unmapped.video_speaker_ids  # and it matches nothing
    assert resolve_speaker_filter(roster) is None  # this is "everything"


def test_narrowing_to_the_wrong_video_is_also_empty(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Гость", video_id=2)
    assert result is not None
    assert result.video_speaker_ids == frozenset()


# -- the local-label space --------------------------------------------


def test_a_local_label_resolves_within_its_video(roster: Database) -> None:
    result = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_01", video_id=1)
    assert result is not None
    assert result.video_speaker_ids == frozenset({2})
    assert result.description == "local label SPEAKER_01 in video 1"


def test_the_same_local_label_in_another_video_is_another_row(
    roster: Database,
) -> None:
    """SPEAKER_00 exists in both videos and they are different voices."""
    first = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00", video_id=1)
    second = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00", video_id=2)
    assert first is not None and second is not None
    assert first.video_speaker_ids == frozenset({1})
    assert second.video_speaker_ids == frozenset({4})


def test_an_unnamed_voice_still_resolves_by_its_local_label(roster: Database) -> None:
    result = resolve_speaker_filter(roster, video_local_speaker="SPEAKER_02", video_id=1)
    assert result is not None
    assert result.video_speaker_ids == frozenset({3})


def test_a_local_label_without_a_video_is_rejected(roster: Database) -> None:
    """Contracts §5: alone it would match the first voice of every diarized video."""
    with pytest.raises(InvalidInputError, match="requires --video"):
        resolve_speaker_filter(roster, video_local_speaker="SPEAKER_00")


def test_the_two_identifier_spaces_cannot_be_combined(roster: Database) -> None:
    with pytest.raises(InvalidInputError, match="one or the other"):
        resolve_speaker_filter(
            roster, speaker="Ведущий", video_local_speaker="SPEAKER_00", video_id=1
        )


# -- rule 4: unresolvable input raises --------------------------------


def test_speaker_never_accepts_a_raw_diarizer_label(roster: Database) -> None:
    with pytest.raises(NotFoundError, match="no speaker matches"):
        resolve_speaker_filter(roster, speaker="SPEAKER_00")


def test_an_unknown_speaker_names_the_nearest_roster_entries(roster: Database) -> None:
    """Rule 4: "no such person" and "that person said nothing" are different answers."""
    with pytest.raises(NotFoundError) as excinfo:
        resolve_speaker_filter(roster, speaker="Ведущй")
    assert "Ведущий" in str(excinfo.value)


def test_an_unknown_speaker_against_an_empty_roster_says_so(db: Database) -> None:
    with pytest.raises(NotFoundError, match="roster is empty"):
        resolve_speaker_filter(db, speaker="Ведущий")


def test_an_unknown_local_label_lists_the_labels_that_video_has(
    roster: Database,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        resolve_speaker_filter(roster, video_local_speaker="SPEAKER_09", video_id=1)
    message = str(excinfo.value)
    assert "SPEAKER_00" in message
    assert "SPEAKER_02" in message


def test_the_result_is_frozen_and_hashable(roster: Database) -> None:
    result = resolve_speaker_filter(roster, speaker="Ведущий")
    assert result is not None
    assert isinstance(result.video_speaker_ids, frozenset)
    assert hash(result)


def test_the_shared_params_are_what_later_parts_splice_in() -> None:
    assert [param.name for param in SPEAKER_PARAMS] == [
        "speaker",
        "video_local_speaker",
        "video",
    ]
    assert all(param.default is None for param in SPEAKER_PARAMS)


def test_the_pinned_signature_has_not_drifted() -> None:
    """Parts 4, 5 and 7 call this; contracts §5 fixes the keywords."""
    import inspect

    parameters = inspect.signature(resolve_speaker_filter).parameters
    assert list(parameters) == ["db", "speaker", "video_local_speaker", "video_id"]
    assert [f.name for f in SpeakerFilter.__dataclass_fields__.values()] == [
        "video_speaker_ids",
        "description",
    ]
```

- [ ] **Step 3: Run both and watch them fail**

Run: `python -m pytest tests/test_registry.py tests/test_speaker_filter.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.commands'`.

- [ ] **Step 4: Write `rytp/commands/__init__.py`**

```python
"""Every rytp operation, defined exactly once (contracts §5).

Design §10: "Each operation is defined once — name, arguments, result
shape — and both the CLI and the TUI are generated from that definition.
Alignment between the two then holds by construction rather than by
discipline."

A handler takes an open :class:`rytp.db.Database` as its first positional
argument and the command's parameters as keyword arguments, and returns a
:class:`CommandResult`. Handlers never print and never call ``sys.exit``:
they raise :class:`rytp.models.RytpError` and the surface decides how a
failure looks.

Two flags name a speaker and they mean different things, so the
resolution lives here once rather than in every command that filters
(contracts §5). See :func:`resolve_speaker_filter`.

A command marked ``cli_only`` is still a registry command — the CLI
generates it like any other — but the TUI palette leaves it out, because
it launches a surface and cannot sensibly be invoked from inside the
surface it launches. ``tui`` is the one part 1 registers; part 7 marks
``speakers.map`` the same way.

Registration happens as a side effect of importing a command module, so
the imports at the bottom of this file are load-bearing — a command
module that nobody imports is a command that does not exist. Later plan
parts add their own line to that block.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rytp.db import Database
from rytp.models import InvalidInputError, NotFoundError

__all__ = [
    "COMMANDS",
    "PARAM_ALIASES",
    "PARAM_TYPES",
    "REQUIRED",
    "SPEAKER_PARAMS",
    "Command",
    "CommandResult",
    "Param",
    "SpeakerFilter",
    "cli_path",
    "leaf_name",
    "register",
    "resolve",
    "resolve_speaker_filter",
    "resolve_video_id",
]

#: Sentinel for "the user must supply this". Distinct from ``None``,
#: which is a perfectly good default meaning "left empty".
REQUIRED: Final = object()

#: Parameter types both surfaces know how to prompt for and parse.
PARAM_TYPES: Final = (str, int, float, bool, Path)

#: Extra long spellings a parameter also answers to, keyed by parameter
#: name. Contracts §5 names ``--global-speaker`` as an alias of
#: ``--speaker``; ``Param`` is fixed by that contract and has no alias
#: field, so the spellings live here and both surfaces read them.
PARAM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "speaker": ("--global-speaker",),
}

#: How many roster entries a failed ``--speaker`` lookup suggests.
_SPEAKER_SUGGESTIONS: Final = 3

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*(\.[a-z][a-z0-9-]*)?$")


@dataclass(frozen=True)
class Param:
    """One argument, described once for both surfaces."""

    name: str
    type: type
    help: str
    default: Any = REQUIRED
    choices: tuple[str, ...] | None = None
    short: str | None = None
    positional: bool = False


@dataclass(frozen=True)
class CommandResult:
    """What a handler returns: an optional table and an optional line.

    Strings only. Formatting a value for display is the handler's job,
    so the CLI and the TUI cannot render the same result differently.
    """

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class Command:
    """One operation."""

    name: str  # dotted: "videos.add", "search.words"
    group: str  # "videos"; "" for top level
    summary: str
    params: tuple[Param, ...]
    handler: Callable[..., CommandResult]
    long_running: bool = False  # TUI must not run these inline
    cli_only: bool = False  # surface launchers; absent from the TUI palette


COMMANDS: dict[str, Command] = {}


def leaf_name(cmd: Command) -> str:
    """The last segment of the dotted name — what the CLI calls the subcommand."""
    return cmd.name.rsplit(".", 1)[-1]


def cli_path(cmd: Command) -> tuple[str, ...]:
    """The argv path to this command: ``("videos", "add")`` or ``("ingest",)``."""
    return tuple(cmd.name.split("."))


def _validate(cmd: Command) -> None:
    """Reject a registration that either surface could not render.

    These are programming errors, not user errors, so they raise
    ``ValueError`` rather than a ``RytpError``: they fire at import time
    and must not be swallowed by the CLI's one-line error handler.
    """
    if not _NAME_RE.match(cmd.name):
        raise ValueError(
            f"bad command name {cmd.name!r}: expected 'verb' or 'group.verb' in "
            "lowercase, digits and hyphens"
        )
    expected_group = cmd.name.split(".")[0] if "." in cmd.name else ""
    if cmd.group != expected_group:
        raise ValueError(
            f"command {cmd.name!r} declares group {cmd.group!r}; the dotted name "
            f"implies {expected_group!r}"
        )
    if not cmd.summary.strip():
        raise ValueError(f"command {cmd.name!r} needs a summary: both surfaces show it")

    seen: set[str] = set()
    positional_finished = False
    for param in cmd.params:
        if param.name in seen:
            raise ValueError(f"command {cmd.name!r} has a duplicate parameter {param.name!r}")
        seen.add(param.name)
        if param.type not in PARAM_TYPES:
            names = ", ".join(t.__name__ for t in PARAM_TYPES)
            raise ValueError(
                f"parameter {cmd.name}.{param.name} has type {param.type!r}; "
                f"the surfaces can only render: {names}"
            )
        if param.choices is not None and param.type is not str:
            raise ValueError(
                f"parameter {cmd.name}.{param.name} has choices but is not a str"
            )
        if param.short is not None:
            if param.positional:
                raise ValueError(
                    f"parameter {cmd.name}.{param.name} is positional and cannot have "
                    "a short flag"
                )
            if not re.fullmatch(r"-[A-Za-z]", param.short):
                raise ValueError(
                    f"parameter {cmd.name}.{param.name} has short flag {param.short!r}; "
                    "expected a single dash and one letter, e.g. '-n'"
                )
        if param.positional:
            if positional_finished:
                raise ValueError(
                    f"command {cmd.name!r} lists positional parameter {param.name!r} "
                    "after an option; positionals must come first"
                )
        else:
            positional_finished = True


def register(cmd: Command) -> Command:
    """Validate and add a command. Returns it, so it can be assigned."""
    _validate(cmd)
    if cmd.name in COMMANDS:
        raise ValueError(f"command {cmd.name!r} is already registered")
    COMMANDS[cmd.name] = cmd
    return cmd


def resolve(name: str) -> Command:
    """Look a command up by dotted name."""
    try:
        return COMMANDS[name]
    except KeyError:
        available = ", ".join(sorted(COMMANDS)) or "(none registered)"
        raise NotFoundError(f"unknown command {name!r}; available: {available}") from None


# ---------------------------------------------------------------------
# Speaker filters — two flags, two identifier spaces (contracts §5)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SpeakerFilter:
    """A speaker selection, already expanded to the ids consumers filter on.

    Contracts §5 pins this shape. ``video_speaker_ids`` is **empty when
    the filter resolved but matches nothing** — a real person nobody has
    mapped to a video yet. An empty set must return no rows, never every
    row: that is the classic silent-wrong-answer bug in a SQL ``IN``
    clause, and it is what "show everything" (a ``None`` filter) is for.
    """

    video_speaker_ids: frozenset[int]  # already expanded, ready to filter on
    description: str  # for display, e.g. 'speaker "Иванов Иван Иванович"'


#: The three flags every speaker-filtering command shares. Splice these
#: into a command's ``params`` rather than restating them, so ``--speaker``
#: means the same thing everywhere. A handler converts its `video` string
#: with `resolve_video_id` before calling the resolver, whose signature is
#: fixed by contracts §5 and takes an int.
SPEAKER_PARAMS: Final[tuple[Param, ...]] = (
    Param(
        "speaker",
        str,
        "A named person from the roster: their label or one of their aliases.",
        default=None,
    ),
    Param(
        "video_local_speaker",
        str,
        "A raw per-video diarizer label such as SPEAKER_00. Requires --video.",
        default=None,
    ),
    Param(
        "video",
        str,
        "Video row id or external id. Required with --video-local-speaker.",
        default=None,
    ),
)


def resolve_speaker_filter(
    db: Database,
    *,
    speaker: str | None = None,
    video_local_speaker: str | None = None,
    video_id: int | None = None,
) -> SpeakerFilter | None:
    """Turn the speaker flags into the ``video_speakers.id`` set to filter on.

    Contracts §5 fixes this signature. There are two identifier spaces
    and conflating them is a bug:

    * ``--speaker`` (alias ``--global-speaker``, handled by
      :data:`PARAM_ALIASES`) names a person in the global roster, matched
      against ``speakers.label`` and then against ``speakers.aliases_json``.
      It never accepts a raw diarizer label.
    * ``--video-local-speaker`` names one diarized voice inside one video
      and **requires** a video. A bare ``SPEAKER_00`` would otherwise
      match the first-detected voice of every diarized video, which is
      not a person and not a useful answer.

    The expansion to ``video_speakers.id`` happens here, not in the
    caller: every consumer filters ``words.video_speaker_id`` or
    ``utterances.video_speaker_id``, and a resolver that stopped at
    ``speakers.id`` would push the same join into all of them.

    Returns:
        ``None`` when no speaker was named at all — the "show everything"
        case, which is distinct from a filter that matched nothing.

    Raises:
        InvalidInputError: if both identifier spaces are used at once, or
            a local label is given without a video.
        NotFoundError: if a name does not resolve. "No such person" and
            "that person said nothing" are different answers, so this
            never degrades to an empty result.
    """
    if speaker is not None and video_local_speaker is not None:
        raise InvalidInputError(
            "--speaker names a person and --video-local-speaker names one "
            "video's diarizer label; filter by one or the other"
        )

    if video_local_speaker is not None:
        if video_id is None:
            raise InvalidInputError(
                "--video-local-speaker requires --video: a label like "
                f"{video_local_speaker!r} is only meaningful inside one video, "
                "and alone it would match the first voice of every diarized one"
            )
        row = db.conn.execute(
            "SELECT id FROM video_speakers WHERE video_id = ? AND local_label = ?",
            (video_id, video_local_speaker),
        ).fetchone()
        if row is None:
            known = [
                str(other["local_label"])
                for other in db.conn.execute(
                    "SELECT local_label FROM video_speakers WHERE video_id = ?"
                    " ORDER BY local_label",
                    (video_id,),
                )
            ]
            listed = ", ".join(known) if known else "(this video is not diarized)"
            raise NotFoundError(
                f"video {video_id} has no speaker label {video_local_speaker!r}; "
                f"it has: {listed}"
            )
        return SpeakerFilter(
            video_speaker_ids=frozenset({int(row["id"])}),
            description=f"local label {video_local_speaker} in video {video_id}",
        )

    if speaker is None:
        return None

    speaker_id, label = _resolve_roster_speaker(db, speaker)
    sql = "SELECT id FROM video_speakers WHERE speaker_id = ?"
    params: list[Any] = [speaker_id]
    description = f'speaker "{label}"'
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
        description += f" in video {video_id}"
    # May legitimately come back empty: a roster entry nobody has mapped
    # to a video yet. Rule 2 — the caller must then return nothing.
    return SpeakerFilter(
        video_speaker_ids=frozenset(
            int(row["id"]) for row in db.conn.execute(sql, params)
        ),
        description=description,
    )


def _resolve_roster_speaker(db: Database, wanted: str) -> tuple[int, str]:
    """Match a roster entry by label, then by alias. Returns ``(id, label)``.

    The label comes back so a filter found through an alias still
    describes itself by the person's canonical name.
    """
    row = db.conn.execute(
        "SELECT id, label FROM speakers WHERE label = ?", (wanted,)
    ).fetchone()
    if row is not None:
        return int(row["id"]), str(row["label"])

    labels: list[str] = []
    for candidate in db.conn.execute("SELECT id, label, aliases_json FROM speakers"):
        labels.append(str(candidate["label"]))
        try:
            aliases = json.loads(candidate["aliases_json"])
        except json.JSONDecodeError:
            aliases = []
        if isinstance(aliases, list) and wanted in [str(alias) for alias in aliases]:
            return int(candidate["id"]), str(candidate["label"])

    near = difflib.get_close_matches(wanted, labels, n=_SPEAKER_SUGGESTIONS)
    if near:
        suggestion = f"did you mean: {', '.join(near)}?"
    elif labels:
        suggestion = f"the roster has: {', '.join(sorted(labels))}"
    else:
        suggestion = "the speaker roster is empty"
    raise NotFoundError(f"no speaker matches {wanted!r}; {suggestion}")


def resolve_video_id(db: Database, ref: str) -> int:
    """Find a video by row id or by external id.

    Public because `videos.remove` resolves the same way the speaker
    filter does, and two spellings of "which video?" would be one too
    many.
    """
    row = (
        db.conn.execute("SELECT id FROM videos WHERE id = ?", (int(ref),)).fetchone()
        if ref.isdigit()
        else db.conn.execute(
            "SELECT id FROM videos WHERE external_id = ?", (ref,)
        ).fetchone()
    )
    if row is None:
        raise NotFoundError(f"no video matches {ref!r}")
    return int(row["id"])
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `python -m pytest tests/test_registry.py tests/test_speaker_filter.py -v`
Expected: 38 passed.

- [ ] **Step 6: Lint the new module**

Run: `python -m ruff check rytp/commands/__init__.py tests/test_registry.py tests/test_speaker_filter.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add rytp/commands/__init__.py tests/test_registry.py tests/test_speaker_filter.py
git commit -m "feat: add the command registry and the shared speaker resolver"
```

---

### Task 9: Generating the CLI from the registry

`build_app()` synthesises one Typer callback per registered command by constructing an `inspect.Signature` and matching `__annotations__` — Typer reads both, so a command added to the registry appears on the CLI with no CLI-side code at all.

Every command is generated, `rytp tui` included. Contracts §5 gives `Command` a `cli_only` flag exactly for surface launchers, so the launcher is a registry entry like everything else — the CLI renders it, the TUI palette hides it (Task 10), and "neither surface may define a command of its own" holds literally.

**Files:**
- Create: `rytp/cli.py`, `rytp/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `rytp.commands` (`COMMANDS`, `REQUIRED`, `Command`, `CommandResult`, `Param`, `leaf_name`), `rytp.config` (`paths`, `ensure_dir`), `rytp.db.Database`, `rytp.models.RytpError`, `rytp.constants`.
- Produces: `rytp.cli.build_app(registry=None) -> typer.Typer`, `rytp.cli.render_result(result) -> str`, `rytp.cli.open_database() -> Database`, `rytp.cli.main() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""The CLI is generated; this checks the generator, not any one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database, schema
from rytp.models import NotFoundError
from tests.test_config import child_env

runner = CliRunner()

seen: dict[str, object] = {}


def record(db: Database, **kwargs: object) -> CommandResult:
    seen.clear()
    seen.update(kwargs)
    seen["db_type"] = type(db).__name__
    return CommandResult(message="recorded")


def explode(db: Database, **kwargs: object) -> CommandResult:
    raise NotFoundError("channel 7 is not in the catalog")


def crash(db: Database, **kwargs: object) -> CommandResult:
    raise ZeroDivisionError("this is a bug, not a user error")


def tabulate(db: Database, **kwargs: object) -> CommandResult:
    return CommandResult(
        columns=("id", "title"),
        rows=(("1", "Утренний эфир"), ("2", "Короткое")),
        message="2 videos",
    )


SAMPLE = {
    "videos.add": Command(
        name="videos.add",
        group="videos",
        summary="Register one video.",
        params=(
            Param("target", str, "URL or local path.", positional=True),
            Param("kind", str, "Kind.", default=None, choices=("video", "short")),
            Param("limit", int, "Rows.", default=10, short="-n"),
            Param("ratio", float, "A ratio.", default=1.0),
            Param("force", bool, "Overwrite.", default=False),
            Param("out", Path, "Where to write.", default=None),
            Param("speaker", str, "Roster name.", default=None),
        ),
        handler=record,
    ),
    "videos.boom": Command(
        name="videos.boom",
        group="videos",
        summary="Raise a domain error.",
        params=(),
        handler=explode,
    ),
    "videos.crash": Command(
        name="videos.crash",
        group="videos",
        summary="Raise a bug.",
        params=(),
        handler=crash,
    ),
    "videos.table": Command(
        name="videos.table",
        group="videos",
        summary="Return a table.",
        params=(),
        handler=tabulate,
    ),
    "doctor": Command(
        name="doctor",
        group="",
        summary="A top-level command.",
        params=(),
        handler=record,
    ),
}


@pytest.fixture()
def app(data_dir: Path):
    from rytp.cli import build_app

    return build_app(SAMPLE)


def test_a_grouped_command_is_reachable_at_group_then_leaf(app) -> None:
    result = runner.invoke(app, ["videos", "add", "https://example.invalid/w/VIDEO_A"])
    assert result.exit_code == 0, result.output
    assert seen["target"] == "https://example.invalid/w/VIDEO_A"
    assert seen["db_type"] == "Database"


def test_a_top_level_command_is_reachable_directly(app) -> None:
    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_defaults_are_applied_and_types_converted(app) -> None:
    result = runner.invoke(
        app,
        [
            "videos",
            "add",
            "https://example.invalid/w/VIDEO_A",
            "-n",
            "3",
            "--ratio",
            "2.5",
            "--force",
            "--out",
            "somewhere.mp4",
            "--kind",
            "short",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["limit"] == 3
    assert seen["ratio"] == 2.5
    assert seen["force"] is True
    assert seen["out"] == Path("somewhere.mp4")
    assert seen["kind"] == "short"


def test_an_omitted_optional_arrives_as_none(app) -> None:
    runner.invoke(app, ["videos", "add", "https://example.invalid/w/VIDEO_A"])
    assert seen["kind"] is None
    assert seen["out"] is None
    assert seen["force"] is False
    assert seen["limit"] == 10


def test_a_missing_required_argument_is_a_usage_error(app) -> None:
    result = runner.invoke(app, ["videos", "add"])
    assert result.exit_code == 2
    assert "Missing argument" in result.stderr


def test_a_value_outside_choices_is_rejected_before_the_handler(app) -> None:
    seen.clear()
    result = runner.invoke(
        app, ["videos", "add", "https://example.invalid/w/VIDEO_A", "--kind", "opera"]
    )
    assert result.exit_code == 2
    assert "must be one of" in result.stderr
    assert seen == {}


def test_a_parameter_alias_is_accepted_under_both_spellings(app) -> None:
    """Contracts §5: `--global-speaker` is another spelling of `--speaker`."""
    assert (
        runner.invoke(app, ["videos", "add", "x", "--speaker", "Ведущий"]).exit_code == 0
    )
    assert seen["speaker"] == "Ведущий"
    assert (
        runner.invoke(
            app, ["videos", "add", "x", "--global-speaker", "Ведущий"]
        ).exit_code
        == 0
    )
    assert seen["speaker"] == "Ведущий"


def test_help_lists_the_summary_and_every_parameter(app) -> None:
    result = runner.invoke(app, ["videos", "add", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Register one video." in result.stdout
    for flag in ("--kind", "--limit", "--ratio", "--force", "--out", "--global-speaker"):
        assert flag in result.stdout


def test_a_domain_error_is_one_line_on_stderr_and_exit_one(app) -> None:
    result = runner.invoke(app, ["videos", "boom"])
    assert result.exit_code == 1
    assert result.stderr.strip() == "channel 7 is not in the catalog"
    assert "Traceback" not in result.stderr


def test_an_unexpected_error_is_not_swallowed(app) -> None:
    result = runner.invoke(app, ["videos", "crash"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ZeroDivisionError)


def test_a_table_result_is_rendered_with_headers(app) -> None:
    result = runner.invoke(app, ["videos", "table"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == "2 videos"
    assert lines[1].split() == ["id", "title"]
    assert "Утренний эфир" in lines[3]


def test_render_result_handles_each_shape() -> None:
    from rytp.cli import render_result

    assert render_result(CommandResult()) == ""
    assert render_result(CommandResult(message="done")) == "done"
    rendered = render_result(
        CommandResult(columns=("a", "bb"), rows=(("1", "2"), ("333", "4")))
    )
    assert rendered.splitlines() == ["a    bb", "---  --", "1    2", "333  4"]


def test_the_database_is_created_and_migrated_on_first_use(app, data_dir: Path) -> None:
    assert not (data_dir / "rytp.db").exists()
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    assert (data_dir / "rytp.db").exists()
    database = Database(data_dir / "rytp.db")
    try:
        assert database.schema_version() == schema.LATEST_VERSION
    finally:
        database.close()


def test_the_data_flag_overrides_rytp_data_for_this_run(app, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    result = runner.invoke(app, ["--data", str(elsewhere), "doctor"])
    assert result.exit_code == 0, result.output
    assert (elsewhere / "rytp.db").exists()


def test_the_cli_exposes_exactly_the_registered_commands(app) -> None:
    """The generator adds nothing of its own — `tui` is a registry entry too."""
    from rytp.cli import command_paths

    assert command_paths(app) == set(SAMPLE)


def test_importing_the_cli_creates_no_directories(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.cli"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []


def test_bare_python_m_rytp_prints_help_and_exits_zero(tmp_path: Path) -> None:
    """Click exits 2 for a group with no arguments; `python -m rytp` must not."""
    proc = subprocess.run(
        [sys.executable, "-m", "rytp"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Usage" in proc.stdout
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.cli'`.

- [ ] **Step 3: Write `rytp/cli.py`**

```python
"""The command-line surface, generated from the registry (design §10).

Nothing in this module defines an operation. :func:`build_app` walks
``rytp.commands.COMMANDS`` and synthesises one Typer callback per entry,
so a command added to the registry appears on the CLI and in the TUI at
the same moment, with the same name, arguments and help.

Nothing is added by hand, ``rytp tui`` included: it is a registry command
flagged ``cli_only`` (contracts §5), because a TUI able to launch itself
would be nonsense. ``tests/test_surfaces.py`` asserts the CLI shows every
registered command and the TUI shows every one that is not ``cli_only``.

Non-ASCII in help text is deliberate: the design is cited as "§10" and
em dashes are used throughout. :func:`_configure_stdio` forces UTF-8 on
stdout and stderr because this project's owner runs Windows, where the
default code page turns those into mojibake.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from rytp import config
from rytp import constants as C
from rytp.commands import (
    COMMANDS,
    PARAM_ALIASES,
    REQUIRED,
    Command,
    CommandResult,
    Param,
    leaf_name,
)
from rytp.db import Database
from rytp.models import RytpError

__all__ = ["build_app", "command_paths", "main", "open_database", "render_result"]


def _configure_stdio() -> None:
    """Force UTF-8 on the standard streams; no-op where that is impossible."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def open_database() -> Database:
    """Open and migrate the database, creating the data root if absent.

    This is the only place in part 1 that creates a directory, which is
    what contracts §7 means by "created on demand by the command that
    needs it".
    """
    layout = config.paths()
    config.ensure_dir(layout.root)
    database = Database(layout.db)
    database.migrate()
    return database


def render_result(result: CommandResult) -> str:
    """Render a handler's result as plain text.

    Deliberately not a rich table: the output is asserted on in tests and
    piped into other tools, so column alignment by spaces beats box
    drawing that reflows with the terminal width.
    """
    lines: list[str] = []
    if result.message:
        lines.append(result.message)
    if result.columns:
        widths = [len(column) for column in result.columns]
        for row in result.rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(cells: tuple[str, ...]) -> str:
            return "  ".join(
                cell.ljust(widths[index]) for index, cell in enumerate(cells)
            ).rstrip()

        lines.append(line(result.columns))
        lines.append(line(tuple("-" * width for width in widths)))
        lines.extend(line(row) for row in result.rows)
    return "\n".join(lines)


def _annotation_for(param: Param) -> Any:
    """The type Typer should parse this parameter as."""
    if param.default is None:
        # `default=None` means "may be left out entirely".
        return param.type | None
    return param.type


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _choice_callback(param: Param) -> Callable[[Any], Any]:
    """Reject a value outside `choices` before the handler ever runs."""
    choices = param.choices or ()

    def check(value: Any) -> Any:
        if value is None or value in choices:
            return value
        raise typer.BadParameter(f"{param.name} must be one of: {', '.join(choices)}")

    return check


def _parameter_info(param: Param) -> Any:
    """The `typer.Argument` / `typer.Option` this parameter becomes."""
    help_text = param.help
    if param.choices:
        help_text = f"{help_text} One of: {', '.join(param.choices)}."
    callback = _choice_callback(param) if param.choices else None
    default = ... if param.default is REQUIRED else param.default

    if param.positional:
        return typer.Argument(default, help=help_text, callback=callback)

    if param.type is bool:
        declarations = [f"{_flag(param.name)}/--no-{param.name.replace('_', '-')}"]
    else:
        declarations = [_flag(param.name)]
    # Contracts §5 gives `--speaker` the alias `--global-speaker`; the
    # spellings live in the registry so both surfaces read the same list.
    declarations.extend(PARAM_ALIASES.get(param.name, ()))
    if param.short:
        declarations.append(param.short)
    return typer.Option(default, *declarations, help=help_text, callback=callback)


def _make_callback(cmd: Command) -> Callable[..., None]:
    """Synthesise the function Typer will introspect for this command.

    Typer reads both ``inspect.signature`` and ``__annotations__``, so
    both are set. The parameters are KEYWORD_ONLY; Typer decides argument
    versus option from the default object, not from the parameter kind.
    """
    annotations = {param.name: _annotation_for(param) for param in cmd.params}
    signature = inspect.Signature(
        [
            inspect.Parameter(
                param.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=_parameter_info(param),
                annotation=annotations[param.name],
            )
            for param in cmd.params
        ]
    )

    def run(**kwargs: Any) -> None:
        database = open_database()
        try:
            result = cmd.handler(database, **kwargs)
        except RytpError as exc:
            # Contracts §8: one line, exit 1, never a traceback.
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            database.close()
        text = render_result(result)
        if text:
            typer.echo(text)

    run.__name__ = cmd.name.replace(".", "_").replace("-", "_")
    run.__doc__ = cmd.summary
    run.__signature__ = signature  # type: ignore[attr-defined]
    run.__annotations__ = {**annotations, "return": None}
    return run


def _root_callback(
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Data tree root for this run; overrides RYTP_DATA."),
    ] = None,
) -> None:
    """rytp — find where words were said, then cut them into a new video."""
    if data is not None:
        os.environ[C.DATA_ROOT_ENV_VAR] = str(data)


def build_app(registry: Mapping[str, Command] | None = None) -> typer.Typer:
    """Build the Typer app for ``registry`` (the global one by default)."""
    commands = COMMANDS if registry is None else registry
    app = typer.Typer(
        name="rytp",
        help="Index a video archive by speaker and word, then cut new video from it.",
        no_args_is_help=True,
        add_completion=False,
    )
    app.callback()(_root_callback)

    groups: dict[str, typer.Typer] = {}
    for cmd in sorted(commands.values(), key=lambda c: c.name):
        callback = _make_callback(cmd)
        if not cmd.group:
            app.command(leaf_name(cmd), help=cmd.summary)(callback)
            continue
        group = groups.get(cmd.group)
        if group is None:
            group = typer.Typer(
                no_args_is_help=True, help=f"Commands in the {cmd.group} group."
            )
            groups[cmd.group] = group
            app.add_typer(group, name=cmd.group)
        group.command(leaf_name(cmd), help=cmd.summary)(callback)

    return app


def command_paths(app: typer.Typer) -> set[str]:
    """Every command the app exposes, as dotted names. Used by the parity test."""
    import typer.main

    def walk(group: Any, prefix: tuple[str, ...] = ()) -> set[str]:
        found: set[str] = set()
        for name, child in getattr(group, "commands", {}).items():
            if getattr(child, "commands", None):
                found |= walk(child, (*prefix, name))
            else:
                found.add(".".join((*prefix, name)))
        return found

    return walk(typer.main.get_command(app))


def main() -> None:
    """Console-script and ``python -m rytp`` entry point."""
    _configure_stdio()
    if len(sys.argv) == 1:
        # Click exits 2 when a group is invoked with no arguments, even
        # with no_args_is_help. Printing help is not an error.
        sys.argv.append("--help")
    build_app()()
```

- [ ] **Step 4: Write `rytp/__main__.py`**

```python
"""`python -m rytp` — the same app as the installed `rytp` console script.

Kept as its own module so a source checkout needs no install.
"""

from __future__ import annotations

from rytp.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test and watch it pass**

Two tests in this file exercise the TUI indirectly (`command_paths` includes `tui`) but do not import it; `tests/test_cli.py` must pass before `rytp/tui/` exists.

Run: `python -m pytest tests/test_cli.py -v`
Expected: 17 passed.

- [ ] **Step 6: Commit**

```bash
git add rytp/cli.py rytp/__main__.py tests/test_cli.py
git commit -m "feat: generate the Typer CLI from the command registry"
```

---

### Task 10: The TUI palette and result table

The second surface, generated from the same dict. Enough to prove registry-driven generation works: a filterable command palette, one line for arguments, a result table, and a refusal to run `long_running` commands inline (design §10: "The worker itself stays a separate process"). Workflow-specific screens — the speaker mapper, transcript reading, cut-list editing — belong to later parts.

`rytp/tui/palette.py` holds all the logic and imports no Textual; `rytp/tui/app.py` is the thin Textual shell. That split is what makes the palette unit-testable without a terminal.

**Files:**
- Create: `rytp/tui/__init__.py`, `rytp/tui/palette.py`, `rytp/tui/app.py`, `rytp/tui/screens/__init__.py`
- Modify: `rytp/commands/__init__.py` (register `tui` as `cli_only`)
- Test: `tests/test_palette.py`, `tests/test_tui_app.py`

**Interfaces:**
- Consumes: `rytp.commands` (`COMMANDS`, `REQUIRED`, `Command`, `CommandResult`, `Param`), `rytp.models` (`InvalidInputError`, `RytpError`), `rytp.db.Database`.
- Produces:
  - `rytp.tui.palette.PaletteEntry(name, group, summary, usage, long_running)`
  - `palette_entries(commands=None) -> list[PaletteEntry]`
  - `match_entries(entries, query) -> list[PaletteEntry]`
  - `usage_line(cmd) -> str`
  - `parse_arguments(cmd, text) -> dict[str, Any]`
  - `cli_invocation(cmd, values) -> str`
  - `rytp.tui.app.RytpApp(db)`, `run_tui(db) -> None`
  - the registered command `tui` (`cli_only=True`), whose handler lives in `rytp/commands/__init__.py`

- [ ] **Step 1: Write the failing palette test**

Create `tests/test_palette.py`:

```python
"""The registry, rendered for the interactive surface — no terminal involved."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp.commands import Command, CommandResult, Param
from rytp.models import InvalidInputError
from rytp.tui.palette import (
    cli_invocation,
    match_entries,
    palette_entries,
    parse_arguments,
    usage_line,
)


def noop(db: object, **kwargs: object) -> CommandResult:
    return CommandResult()


ADD = Command(
    name="videos.add",
    group="videos",
    summary="Register one video.",
    params=(
        Param("target", str, "URL or local path.", positional=True),
        Param("kind", str, "Kind.", default=None, choices=("video", "short")),
        Param("limit", int, "Rows.", default=10),
        Param("force", bool, "Overwrite.", default=False),
        Param("out", Path, "Where to write.", default=None),
        Param("speaker", str, "Roster name.", default=None),
    ),
    handler=noop,
)

SYNC = Command(
    name="channel.sync",
    group="channel",
    summary="Re-enumerate a channel.",
    params=(Param("channel", str, "Channel URL or title.", positional=True),),
    handler=noop,
    long_running=True,
)

LAUNCHER = Command(
    name="tui",
    group="",
    summary="Open the interactive surface.",
    params=(),
    handler=noop,
    cli_only=True,
)

SAMPLE = {"videos.add": ADD, "channel.sync": SYNC, "tui": LAUNCHER}


def test_every_registered_command_becomes_one_entry() -> None:
    entries = palette_entries(SAMPLE)
    assert [entry.name for entry in entries] == ["channel.sync", "videos.add"]
    assert entries[0].long_running is True
    assert entries[1].summary == "Register one video."
    assert entries[1].group == "videos"


def test_a_cli_only_command_is_absent_from_the_palette() -> None:
    """Contracts §5: a surface launcher cannot be invoked from the surface it launches."""
    assert "tui" not in {entry.name for entry in palette_entries(SAMPLE)}


def test_usage_shows_required_and_optional_parameters_differently() -> None:
    assert usage_line(ADD) == (
        "videos add <target> [kind=…] [limit=…] [force=…] [out=…] [speaker=…]"
    )
    assert usage_line(SYNC) == "channel sync <channel>"


def test_matching_is_case_insensitive_over_name_and_summary() -> None:
    entries = palette_entries(SAMPLE)
    assert [e.name for e in match_entries(entries, "SYNC")] == ["channel.sync"]
    assert [e.name for e in match_entries(entries, "register")] == ["videos.add"]
    assert [e.name for e in match_entries(entries, "")] == [e.name for e in entries]
    assert match_entries(entries, "nothing at all") == []


def test_parse_fills_positionals_in_order_then_keywords() -> None:
    values = parse_arguments(ADD, "https://example.invalid/w/VIDEO_A kind=short limit=3")
    assert values == {
        "target": "https://example.invalid/w/VIDEO_A",
        "kind": "short",
        "limit": 3,
        "force": False,
        "out": None,
        "speaker": None,
    }


def test_parse_converts_each_declared_type() -> None:
    values = parse_arguments(ADD, "x force=yes out=a/b.mp4 limit=7")
    assert values["force"] is True
    assert values["out"] == Path("a/b.mp4")
    assert values["limit"] == 7
    assert parse_arguments(ADD, "x force=no")["force"] is False


def test_parse_accepts_quoted_values() -> None:
    values = parse_arguments(SYNC, '"Channel One"')
    assert values["channel"] == "Channel One"


def test_parse_rejects_a_missing_required_parameter() -> None:
    with pytest.raises(InvalidInputError, match="target"):
        parse_arguments(ADD, "kind=short")


def test_the_global_speaker_alias_is_accepted_when_typing_arguments() -> None:
    """Contracts §5, via PARAM_ALIASES: both spellings reach the `speaker` param."""
    assert parse_arguments(ADD, "x global_speaker=Ведущий")["speaker"] == "Ведущий"
    assert parse_arguments(ADD, "x speaker=Ведущий")["speaker"] == "Ведущий"


def test_parse_rejects_an_unknown_parameter() -> None:
    with pytest.raises(InvalidInputError, match="nonsense"):
        parse_arguments(ADD, "x nonsense=1")


def test_parse_rejects_a_value_outside_choices() -> None:
    with pytest.raises(InvalidInputError, match="must be one of"):
        parse_arguments(ADD, "x kind=opera")


def test_parse_rejects_a_value_of_the_wrong_type() -> None:
    with pytest.raises(InvalidInputError, match="limit"):
        parse_arguments(ADD, "x limit=many")


def test_parse_rejects_extra_positional_values() -> None:
    with pytest.raises(InvalidInputError, match="too many"):
        parse_arguments(SYNC, "one two")


def test_a_url_with_a_query_string_is_still_a_positional_value() -> None:
    """`?v=…` must not be mistaken for a `name=value` argument."""
    values = parse_arguments(ADD, "https://example.invalid/watch?v=VIDEO_A")
    assert values["target"] == "https://example.invalid/watch?v=VIDEO_A"


def test_cli_invocation_is_copy_pasteable() -> None:
    values = parse_arguments(ADD, "https://example.invalid/w/VIDEO_A kind=short limit=3")
    assert cli_invocation(ADD, values) == (
        "rytp videos add https://example.invalid/w/VIDEO_A --kind short --limit 3"
    )


def test_cli_invocation_renders_flags_and_omits_defaults() -> None:
    values = parse_arguments(ADD, "x force=true")
    assert cli_invocation(ADD, values) == "rytp videos add x --force"
    assert cli_invocation(SYNC, {"channel": "Channel One"}) == (
        'rytp channel sync "Channel One"'
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_palette.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'rytp.tui'`.

- [ ] **Step 3: Write `rytp/tui/__init__.py` and `rytp/tui/screens/__init__.py`**

```python
"""The interactive surface. Generated from the same registry as the CLI."""

from __future__ import annotations
```

and, for `rytp/tui/screens/__init__.py`:

```python
"""Workflow screens. Part 7 adds the speaker mapper here."""

from __future__ import annotations
```

- [ ] **Step 4: Write `rytp/tui/palette.py`**

```python
"""The registry, rendered for the interactive surface (design §10).

Deliberately free of Textual imports: everything here is ordinary data
and string handling, so it can be tested without a terminal and reused
by whatever the TUI grows into. ``cli_only`` commands are filtered out
here, in one place, rather than in the Textual layer.

Arguments are typed as ``name=value``, with leading bare words filling
the positional parameters in order. That is enough for the catalog
commands and keeps the surface honest — there is exactly one definition
of what a command takes, and the TUI reads it rather than restating it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rytp.commands import COMMANDS, PARAM_ALIASES, REQUIRED, Command, Param
from rytp.models import InvalidInputError

__all__ = [
    "PaletteEntry",
    "cli_invocation",
    "match_entries",
    "palette_entries",
    "parse_arguments",
    "usage_line",
]

_TRUE = frozenset({"1", "true", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "no", "n", "off"})
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class PaletteEntry:
    """One row of the command palette."""

    name: str
    group: str
    summary: str
    usage: str
    long_running: bool


def usage_line(cmd: Command) -> str:
    """A one-line reminder of what this command takes."""
    parts = [cmd.name.replace(".", " ")]
    for param in cmd.params:
        if param.positional:
            parts.append(
                f"<{param.name}>" if param.default is REQUIRED else f"[{param.name}]"
            )
        elif param.default is REQUIRED:
            parts.append(f"{param.name}=…")
        else:
            parts.append(f"[{param.name}=…]")
    return " ".join(parts)


def palette_entries(commands: Mapping[str, Command] | None = None) -> list[PaletteEntry]:
    """Every registered command the TUI can show, sorted by name.

    ``cli_only`` commands are left out: they launch a surface, and the
    surface they launch is this one (contracts §5).
    """
    source = COMMANDS if commands is None else commands
    return [
        PaletteEntry(
            name=cmd.name,
            group=cmd.group,
            summary=cmd.summary,
            usage=usage_line(cmd),
            long_running=cmd.long_running,
        )
        for cmd in sorted(source.values(), key=lambda c: c.name)
        if not cmd.cli_only
    ]


def match_entries(entries: Sequence[PaletteEntry], query: str) -> list[PaletteEntry]:
    """Filter the palette by a case-insensitive substring of name or summary."""
    needle = query.strip().lower()
    if not needle:
        return list(entries)
    return [
        entry
        for entry in entries
        if needle in entry.name.lower() or needle in entry.summary.lower()
    ]


def _convert(param: Param, raw: str) -> Any:
    if param.type is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise InvalidInputError(f"{param.name}: expected true or false, got {raw!r}")
    if param.type is Path:
        return Path(raw)
    try:
        return param.type(raw)
    except ValueError as exc:
        raise InvalidInputError(
            f"{param.name}: expected {param.type.__name__}, got {raw!r}"
        ) from exc


def parse_arguments(cmd: Command, text: str) -> dict[str, Any]:
    """Turn a typed argument line into the handler's keyword arguments.

    Bare words fill positional parameters in order; ``name=value`` sets
    a named one. Missing optional parameters take their declared default.
    """
    by_name = {param.name: param for param in cmd.params}
    # `global_speaker=…` is `speaker=…` (contracts §5, PARAM_ALIASES).
    for canonical, spellings in PARAM_ALIASES.items():
        target = by_name.get(canonical)
        if target is None:
            continue
        for spelling in spellings:
            by_name.setdefault(spelling.lstrip("-").replace("-", "_"), target)
    positional = [param for param in cmd.params if param.positional]
    values: dict[str, Any] = {}
    next_positional = 0

    for token in shlex.split(text):
        name, sep, raw = token.partition("=")
        # A token counts as `name=value` only when the left side looks
        # like a parameter name. URLs contain '=' too, and a query string
        # must not be mistaken for a keyword argument.
        if sep and _NAME_RE.fullmatch(name):
            named = by_name.get(name)
            if named is None:
                raise InvalidInputError(
                    f"{cmd.name} has no parameter {name!r}: {usage_line(cmd)}"
                )
            param = named
        else:
            if next_positional >= len(positional):
                raise InvalidInputError(
                    f"too many values for {cmd.name}: {usage_line(cmd)}"
                )
            param = positional[next_positional]
            next_positional += 1
            raw = token
        if param.choices is not None and raw not in param.choices:
            raise InvalidInputError(
                f"{param.name} must be one of: {', '.join(param.choices)}"
            )
        values[param.name] = _convert(param, raw)

    for param in cmd.params:
        if param.name in values:
            continue
        if param.default is REQUIRED:
            raise InvalidInputError(f"{cmd.name} needs {param.name}: {usage_line(cmd)}")
        values[param.name] = param.default
    return values


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def cli_invocation(cmd: Command, values: Mapping[str, Any]) -> str:
    """The equivalent command line, for a long-running command's hint."""
    parts = ["rytp", *cmd.name.split(".")]
    for param in cmd.params:
        value = values.get(param.name, param.default)
        if value is None or value is REQUIRED:
            continue
        if param.positional:
            parts.append(_quote(str(value)))
            continue
        if param.type is bool:
            if value:
                parts.append("--" + param.name.replace("_", "-"))
            continue
        if value == param.default:
            continue
        parts.append("--" + param.name.replace("_", "-"))
        parts.append(_quote(str(value)))
    return " ".join(parts)
```

- [ ] **Step 5: Run the palette test and watch it pass**

Run: `python -m pytest tests/test_palette.py -v`
Expected: 16 passed.

- [ ] **Step 6: Write the failing app test**

Create `tests/test_tui_app.py`. Textual apps are async; `asyncio.run` drives them from a sync test so the suite needs no async plugin.

```python
"""The Textual shell, driven headlessly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from textual.widgets import DataTable, Input, OptionList

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.models import NotFoundError
from rytp.tui.app import RytpApp

calls: list[dict[str, Any]] = []


def listing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(kwargs)
    return CommandResult(
        columns=("id", "title"),
        rows=(("1", "Утренний эфир"),),
        message="1 video",
    )


def failing(db: Database, **kwargs: Any) -> CommandResult:
    raise NotFoundError("channel 7 is not in the catalog")


def syncing(db: Database, **kwargs: Any) -> CommandResult:  # pragma: no cover - never run
    raise AssertionError("a long-running command must not run inside the TUI")


SAMPLE = {
    "videos.list": Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(Param("limit", int, "Rows.", default=50),),
        handler=listing,
    ),
    "videos.boom": Command(
        name="videos.boom",
        group="videos",
        summary="Fail on purpose.",
        params=(),
        handler=failing,
    ),
    "channel.sync": Command(
        name="channel.sync",
        group="channel",
        summary="Re-enumerate a channel.",
        params=(Param("channel", str, "Channel.", positional=True),),
        handler=syncing,
        long_running=True,
    ),
}


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    """Run `body` against a mounted RytpApp, headlessly."""

    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_the_palette_lists_every_registered_command(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        palette = app.query_one("#palette", OptionList)
        assert palette.option_count == len(SAMPLE)
        ids = [palette.get_option_at_index(i).id for i in range(palette.option_count)]
        assert ids == sorted(SAMPLE)

    drive(db, body)


def test_typing_in_the_filter_narrows_the_palette(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.query_one("#filter", Input).value = "sync"
        await pilot.pause()
        palette = app.query_one("#palette", OptionList)
        assert palette.option_count == 1
        assert palette.get_option_at_index(0).id == "channel.sync"

    drive(db, body)


def test_running_a_command_fills_the_result_table(db: Database) -> None:
    calls.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.list")
        app.run_selected("limit=5")
        await pilot.pause()
        table = app.query_one("#results", DataTable)
        assert table.row_count == 1
        assert [str(c.label) for c in table.columns.values()] == ["id", "title"]
        assert app.status_text == "1 video"
        assert calls == [{"limit": 5}]

    drive(db, body)


def test_a_domain_error_shows_in_the_status_line_and_does_not_crash(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.boom")
        app.run_selected("")
        await pilot.pause()
        assert app.status_text == "channel 7 is not in the catalog"
        assert app.is_running

    drive(db, body)


def test_a_long_running_command_is_not_run_inline(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("channel.sync")
        app.run_selected("CHANNEL_ONE")
        await pilot.pause()
        assert "rytp channel sync CHANNEL_ONE" in app.status_text
        assert app.query_one("#results", DataTable).row_count == 0

    drive(db, body)


def test_a_bad_argument_line_is_reported_not_raised(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("videos.list")
        app.run_selected("limit=many")
        await pilot.pause()
        assert "limit" in app.status_text

    drive(db, body)


def test_importing_the_tui_creates_no_directories(tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.test_config import child_env

    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.tui.app"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 7: Run it and watch it fail**

Run: `python -m pytest tests/test_tui_app.py -v`
Expected: collection error — `ImportError: cannot import name 'RytpApp' from 'rytp.tui.app'`.

- [ ] **Step 8: Write `rytp/tui/app.py`**

Note `self.shown`, not `self.visible`: Textual's DOM already owns a `visible` property and assigning a list to it raises `TypeError: cannot use 'list' as a set element`.

```python
"""The Textual shell around the command registry (design §10).

A palette of every registered command, a line to type arguments into,
and a table for the result. Workflow screens — the speaker mapper above
all — belong to later plan parts; this is the proof that both surfaces
come from one definition.

Long-running commands are listed but never run here: design §5 puts them
in a separate worker process, so the TUI prints the equivalent command
line instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import DataTable, Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from rytp.commands import COMMANDS, Command, CommandResult
from rytp.db import Database
from rytp.models import RytpError
from rytp.tui.palette import (
    PaletteEntry,
    cli_invocation,
    match_entries,
    palette_entries,
    parse_arguments,
)

__all__ = ["RytpApp", "run_tui"]


class RytpApp(App[None]):
    """Command palette, argument line, result table."""

    TITLE = "rytp"
    SUB_TITLE = "find what was said, then cut it"

    CSS = """
    #palette { height: 1fr; }
    #results { height: 1fr; }
    #status  { height: auto; padding: 0 1; }
    """

    # `list[Binding]` is rejected by mypy: App declares the wider
    # `list[BindingType]` and list is invariant.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f2", "focus_filter", "Filter"),
    ]

    def __init__(
        self, db: Database, commands: Mapping[str, Command] | None = None
    ) -> None:
        super().__init__()
        self._db = db
        self._commands: Mapping[str, Command] = COMMANDS if commands is None else commands
        self._entries = palette_entries(self._commands)
        # NOT `self.visible`: Textual's DOM owns that name.
        self.shown: list[PaletteEntry] = list(self._entries)
        self.status_text = ""

    # -- layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="filter commands", id="filter")
        yield OptionList(id="palette")
        yield Input(placeholder="arguments: value name=value …", id="arguments")
        yield Static("", id="status")
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_palette("")

    # -- palette -----------------------------------------------------

    def refresh_palette(self, query: str) -> None:
        palette = self.query_one("#palette", OptionList)
        palette.clear_options()
        self.shown = match_entries(self._entries, query)
        for entry in self.shown:
            marker = " (worker)" if entry.long_running else ""
            palette.add_option(Option(f"{entry.name}  —  {entry.summary}{marker}", id=entry.name))
        if self.shown:
            palette.highlighted = 0

    def select(self, name: str) -> None:
        """Highlight a command by name. Used by the tests and by the palette."""
        for index, entry in enumerate(self.shown):
            if entry.name == name:
                self.query_one("#palette", OptionList).highlighted = index
                return

    def selected_command(self) -> Command | None:
        palette = self.query_one("#palette", OptionList)
        index = palette.highlighted
        if index is None or index >= len(self.shown):
            return None
        return self._commands[self.shown[index].name]

    # -- running -----------------------------------------------------

    def run_selected(self, argument_line: str | None = None) -> None:
        """Parse the argument line and run the highlighted command."""
        cmd = self.selected_command()
        if cmd is None:
            self.set_status("no command selected")
            return
        if argument_line is None:
            argument_line = self.query_one("#arguments", Input).value
        try:
            values = parse_arguments(cmd, argument_line)
        except RytpError as exc:
            self.set_status(str(exc))
            return
        if cmd.long_running:
            self.set_status(
                f"long-running; run it from a terminal: {cli_invocation(cmd, values)}"
            )
            return
        try:
            result = cmd.handler(self._db, **values)
        except RytpError as exc:
            self.set_status(str(exc))
            return
        self.show_result(result)

    def show_result(self, result: CommandResult) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=True)
        if result.columns:
            table.add_columns(*result.columns)
            for row in result.rows:
                table.add_row(*row)
        self.set_status(result.message or f"{len(result.rows)} rows")

    def set_status(self, text: str) -> None:
        self.status_text = text
        self.query_one("#status", Static).update(text)

    # -- events ------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.refresh_palette(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "arguments":
            self.run_selected(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.query_one("#arguments", Input).focus()

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()


def run_tui(db: Database) -> None:
    """Launch the app. Called only by `rytp tui`."""
    RytpApp(db).run()
```

- [ ] **Step 9: Register `tui` as a `cli_only` command**

`run_tui` now exists, so the launcher can join the registry. Append to `rytp/commands/__init__.py`, after `resolve` and above the module-import block that Task 11 adds:

```python
def _open_tui(db: Database, **_: object) -> CommandResult:
    """Hand the open database to the Textual app and block until it exits.

    Imported lazily so a plain `rytp videos list` never pays for Textual.
    """
    from rytp.tui.app import run_tui

    run_tui(db)
    return CommandResult(message="tui closed")


register(
    Command(
        name="tui",
        group="",
        summary="Open the interactive surface.",
        params=(),
        handler=_open_tui,
        cli_only=True,
    )
)
```

- [ ] **Step 10: Run the app test and watch it pass**

Run: `python -m pytest tests/test_tui_app.py tests/test_palette.py -v`
Expected: 23 passed.

- [ ] **Step 11: Commit**

```bash
git add rytp/tui/__init__.py rytp/tui/palette.py rytp/tui/app.py rytp/tui/screens/__init__.py \
  rytp/commands/__init__.py tests/test_palette.py tests/test_tui_app.py
git commit -m "feat: generate the TUI palette from the same command registry"
```

---

### Task 11: channel.add and channel.list

The first real commands, and the wiring that makes a command module's registrations happen: `rytp/commands/__init__.py` imports its siblings at the bottom, so importing `rytp.commands` is what makes `COMMANDS` non-empty. Later plan parts add one line each to that block.

**Files:**
- Create: `rytp/commands/catalog.py`
- Modify: `rytp/commands/__init__.py` (add the import block at the very bottom)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `rytp.commands` (`Command`, `CommandResult`, `Param`, `register`), `rytp.db.queries`, `rytp.constants`, `rytp.models`.
- Produces: handlers `channel_add(db, *, url, title=None)`, `channel_list(db, *, limit=DEFAULT_LIST_LIMIT)`; registered commands `channel.add`, `channel.list`; helper `truncate(text) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog.py`:

```python
"""Catalog commands, exercised both as handlers and through the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import InvalidInputError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def test_channel_add_registers_and_reports_the_id(db: Database) -> None:
    result = catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    assert result.rows[0][1] == CHANNEL_URL
    assert result.rows[0][2] == "Channel One"
    assert "added" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1


def test_channel_add_twice_updates_rather_than_duplicating(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    result = catalog.channel_add(db, url=CHANNEL_URL, title="Renamed")
    assert "updated" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1
    assert db.conn.execute("SELECT title FROM channels").fetchone()[0] == "Renamed"


def test_channel_add_falls_back_to_the_url_as_a_title(db: Database) -> None:
    """Probing a channel for its real title is acquisition, and belongs to part 2."""
    result = catalog.channel_add(db, url=CHANNEL_URL)
    assert result.rows[0][2] == CHANNEL_URL


def test_channel_add_rejects_an_empty_url(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="URL"):
        catalog.channel_add(db, url="   ")


def test_channel_list_is_empty_on_a_fresh_database(db: Database) -> None:
    result = catalog.channel_list(db)
    assert result.rows == ()
    assert result.message == "0 channels"


def test_channel_list_shows_the_video_count_and_sync_time(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    result = catalog.channel_list(db)
    assert result.columns == ("id", "title", "videos", "last synced", "url")
    assert result.rows[0][1] == "Channel One"
    assert result.rows[0][2] == "0"
    assert result.rows[0][3] == "-"
    assert result.message == "1 channel"


def test_channel_list_truncates_a_long_title(db: Database) -> None:
    long_title = "Д" * 200
    catalog.channel_add(db, url=CHANNEL_URL, title=long_title)
    cell = catalog.channel_list(db).rows[0][1]
    assert len(cell) <= 60
    assert cell.endswith("…")


def test_both_channel_commands_are_registered() -> None:
    assert "channel.add" in commands.COMMANDS
    assert "channel.list" in commands.COMMANDS
    assert commands.COMMANDS["channel.add"].handler is catalog.channel_add


def test_the_cli_runs_them_end_to_end(data_dir: Path) -> None:
    app = build_app()
    added = runner.invoke(app, ["channel", "add", CHANNEL_URL, "--title", "Channel One"])
    assert added.exit_code == 0, added.output
    listed = runner.invoke(app, ["channel", "list"], env={"COLUMNS": "200"})
    assert listed.exit_code == 0, listed.output
    assert "Channel One" in listed.stdout
    assert CHANNEL_URL in listed.stdout


def test_the_cli_creates_the_database_on_the_first_command(data_dir: Path) -> None:
    assert not (data_dir / "rytp.db").exists()
    assert runner.invoke(build_app(), ["channel", "list"]).exit_code == 0
    assert (data_dir / "rytp.db").exists()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_catalog.py -v`
Expected: collection error — `ImportError: cannot import name 'catalog' from 'rytp.commands'`.

- [ ] **Step 3: Write `rytp/commands/catalog.py`**

```python
"""Catalog commands: register channels and videos, and list them (design §5).

Cataloguing is deliberately separate from acquisition. These commands
write rows and nothing else — no downloads, no jobs, no files on disk.
Design §5: "`rytp videos add <url>` catalogs only. `rytp ingest <id>`
enqueues the chain."
"""

from __future__ import annotations

from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import InvalidInputError

__all__ = ["channel_add", "channel_list", "truncate"]


def truncate(text: str, limit: int = C.TITLE_TRUNCATE_CHARS) -> str:
    """Shorten a title for a table cell, marking that it was cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """`1 channel` / `3 channels`, so the message line reads like English."""
    word = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {word}"


# -- channels ---------------------------------------------------------


def channel_add(db: Database, *, url: str, title: str | None = None) -> CommandResult:
    """Register a channel by URL. Nothing is fetched."""
    url = url.strip()
    if not url:
        raise InvalidInputError("channel add needs a URL")
    resolved_title = (title or url).strip()
    channel_id, created = q.insert_channel(db, url=url, title=resolved_title)
    return CommandResult(
        columns=("id", "url", "title"),
        rows=((str(channel_id), url, resolved_title),),
        message=f"{'added' if created else 'updated'} channel {channel_id}",
    )


def channel_list(db: Database, *, limit: int = C.DEFAULT_LIST_LIMIT) -> CommandResult:
    """List registered channels, newest first, with their video counts."""
    rows = q.list_channels(db, limit=limit)
    return CommandResult(
        columns=("id", "title", "videos", "last synced", "url"),
        rows=tuple(
            (
                str(row["id"]),
                truncate(row["title"]),
                str(row["n_videos"]),
                row["last_synced_at"] or C.NULL_CELL,
                row["url"],
            )
            for row in rows
        ),
        message=plural(len(rows), "channel"),
    )


register(
    Command(
        name="channel.add",
        group="channel",
        summary="Register a channel by URL. Catalogs only; nothing is fetched.",
        params=(
            Param("url", str, "Channel URL.", positional=True),
            Param("title", str, "Title to store. Defaults to the URL.", default=None),
        ),
        handler=channel_add,
    )
)

register(
    Command(
        name="channel.list",
        group="channel",
        summary="List registered channels and how many videos each has.",
        params=(
            Param("limit", int, "Maximum rows.", default=C.DEFAULT_LIST_LIMIT, short="-n"),
        ),
        handler=channel_list,
    )
)
```

- [ ] **Step 4: Add the import block at the bottom of `rytp/commands/__init__.py`**

This is the load-bearing bit: a command module nobody imports registers nothing. Append at the very end of the file, after every definition:

```python
# Importing a command module is what registers its commands. Keep this
# block last: the modules below import names defined above.
from rytp.commands import catalog as _catalog  # noqa: E402,F401
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `python -m pytest tests/test_catalog.py -v`
Expected: 10 passed.

- [ ] **Step 6: Run the whole suite — the registry is no longer empty**

Run: `python -m pytest -q`
Expected: everything passes. `tests/test_registry.py` monkeypatches `COMMANDS`, so the real registrations do not leak into it.

- [ ] **Step 7: Commit**

```bash
git add rytp/commands/catalog.py rytp/commands/__init__.py tests/test_catalog.py
git commit -m "feat: add channel add and channel list"
```

---

### Task 12: videos.add and videos.list

`videos add` takes either a URL or a local path and writes exactly one catalog row. No downloads, no `assets`, no jobs — design §5 puts all of that behind `rytp ingest <id>` (plan part 2).

Two decisions this task locks in, both forced by the schema:

- **A local file's path is stored in `videos.external_id`.** Contracts §3 gives `videos` no path column (design §4: "All path and download columns move out"), and `UNIQUE (source, external_id)` is the only thing that makes re-registering the same file idempotent. Part 2's `acquire/local.py` reads it and creates the `container` asset.
- **Metadata for a URL comes from a seam, not from yt-dlp here.** `_probe_video` lazily imports `rytp.acquire.ytdlp.probe_video` (plan part 2) and raises a one-line install hint when it is absent. Tests monkeypatch `_probe_video`.

**Files:**
- Modify: `rytp/commands/catalog.py`
- Test: `tests/test_catalog_videos.py`

**Interfaces:**
- Consumes: `rytp.models.ChannelEntry`, `rytp.db.queries.upsert_video`, `rytp.constants` (`VIDEO_KINDS`, `REMOTE_SOURCE`, `LOCAL_SOURCE`, `DEFAULT_VIDEO_KIND`, `VIDEO_SOURCES`).
- Produces: `videos_add(db, *, target, title=None, kind=None, channel=None)`, `videos_list(db, *, channel=None, kind=None, source=None, search=None, limit=DEFAULT_LIST_LIMIT)`, `resolve_channel(db, ref) -> int`, `looks_local(target) -> bool`, `format_duration(ms) -> str`, `_probe_video(url) -> ChannelEntry`; registered commands `videos.add`, `videos.list`.
- Part 2 must expose `rytp.acquire.ytdlp.probe_video(url: str, *, runner: object | None = None) -> ChannelEntry`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog_videos.py`:

```python
"""videos add / videos list. yt-dlp is never called: the seam is faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError, RytpError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"
VIDEO_URL = "https://example.invalid/watch?v=VIDEO_A"

PROBED = ChannelEntry(
    external_id="VIDEO_A",
    title="Утренний эфир",
    url=VIDEO_URL,
    duration_ms=3_600_000,
    kind="video",
    published_at="2026-01-01T00:00:00+00:00",
)


@pytest.fixture()
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the yt-dlp seam. Part 2 supplies the real implementation."""
    seen: list[str] = []

    def probe(url: str) -> ChannelEntry:
        seen.append(url)
        return PROBED

    monkeypatch.setattr(catalog, "_probe_video", probe)
    return seen


def make_file(tmp_path: Path, name: str = "interview.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def test_adding_a_url_catalogs_what_the_probe_reported(
    db: Database, fake_probe: list[str]
) -> None:
    result = catalog.videos_add(db, target=VIDEO_URL)
    assert fake_probe == [VIDEO_URL]
    row = db.conn.execute("SELECT * FROM videos").fetchone()
    assert row["source"] == "ytdlp"
    assert row["external_id"] == "VIDEO_A"
    assert row["url"] == VIDEO_URL
    assert row["title"] == "Утренний эфир"
    assert row["duration_ms"] == 3_600_000
    assert row["kind"] == "video"
    assert "added" in (result.message or "")


def test_adding_a_url_creates_no_assets_and_no_jobs(
    db: Database, fake_probe: list[str]
) -> None:
    """Design §5: cataloguing only. `rytp ingest` starts the chain."""
    catalog.videos_add(db, target=VIDEO_URL)
    assert db.conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_explicit_title_and_kind_override_the_probe(
    db: Database, fake_probe: list[str]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL, title="My name for it", kind="livestream")
    row = db.conn.execute("SELECT title, kind FROM videos").fetchone()
    assert row["title"] == "My name for it"
    assert row["kind"] == "livestream"


def test_adding_the_same_url_twice_updates_one_row(
    db: Database, fake_probe: list[str]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL)
    result = catalog.videos_add(db, target=VIDEO_URL, title="Corrected")
    assert "updated" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert db.conn.execute("SELECT title FROM videos").fetchone()[0] == "Corrected"


def test_adding_a_local_file_stores_its_absolute_path_as_the_external_id(
    db: Database, tmp_path: Path
) -> None:
    path = make_file(tmp_path)
    catalog.videos_add(db, target=str(path))
    row = db.conn.execute("SELECT * FROM videos").fetchone()
    assert row["source"] == "local"
    assert row["external_id"] == str(path.resolve())
    assert row["url"] is None
    assert row["title"] == "interview"


def test_adding_the_same_local_file_twice_is_idempotent(
    db: Database, tmp_path: Path
) -> None:
    path = make_file(tmp_path)
    catalog.videos_add(db, target=str(path))
    catalog.videos_add(db, target=str(path))
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_a_local_path_never_reaches_the_probe(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(url: str) -> ChannelEntry:  # pragma: no cover - must not run
        raise AssertionError("a local file must not be probed")

    monkeypatch.setattr(catalog, "_probe_video", explode)
    catalog.videos_add(db, target=str(make_file(tmp_path)))
    assert db.conn.execute("SELECT source FROM videos").fetchone()[0] == "local"


def test_a_missing_local_path_that_is_not_a_url_is_an_error(db: Database) -> None:
    with pytest.raises(RytpError):
        catalog.videos_add(db, target="definitely/not/here.mp4")


def test_the_probe_seam_explains_the_missing_extra(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint must stay actionable — and this must never reach the network.

    Forcing `None` into sys.modules makes the import fail whether or not
    part 2 has landed. Without it, this test would quietly start making
    real yt-dlp calls the day `rytp/acquire/ytdlp.py` appears.
    """
    import sys

    monkeypatch.setitem(sys.modules, "rytp.acquire", None)
    monkeypatch.setitem(sys.modules, "rytp.acquire.ytdlp", None)
    with pytest.raises(RytpError, match="yt-dlp"):
        catalog._probe_video(VIDEO_URL)


def test_an_unknown_kind_is_rejected(db: Database, fake_probe: list[str]) -> None:
    with pytest.raises(InvalidInputError, match="kind"):
        catalog.videos_add(db, target=VIDEO_URL, kind="opera")


def test_a_video_can_be_filed_under_a_channel_by_id_or_by_url(
    db: Database, fake_probe: list[str], tmp_path: Path
) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    assert db.conn.execute("SELECT channel_id FROM videos").fetchone()[0] == 1
    catalog.videos_add(db, target=str(make_file(tmp_path)), channel=CHANNEL_URL)
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM videos WHERE channel_id = 1"
        ).fetchone()[0]
        == 2
    )


def test_an_unknown_channel_is_an_error(db: Database, fake_probe: list[str]) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.videos_add(db, target=VIDEO_URL, channel="no such channel")


def test_videos_list_shows_the_catalog(db: Database, fake_probe: list[str]) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    result = catalog.videos_list(db)
    assert result.columns == (
        "id",
        "source",
        "kind",
        "duration",
        "channel",
        "title",
        "external id",
    )
    row = result.rows[0]
    assert row[1] == "ytdlp"
    assert row[3] == "1:00:00"
    assert row[4] == "Channel One"
    assert result.message == "1 video"


def test_videos_list_filters(db: Database, fake_probe: list[str], tmp_path: Path) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    catalog.videos_add(db, target=str(make_file(tmp_path)), title="Домашняя запись")

    def ids(**filters: object) -> list[str]:
        return [row[6] for row in catalog.videos_list(db, **filters).rows]  # type: ignore[arg-type]

    assert len(ids()) == 2
    assert ids(source="local") == [str((tmp_path / "interview.mp4").resolve())]
    assert ids(channel="Channel One") == ["VIDEO_A"]
    assert ids(kind="video") == [
        str((tmp_path / "interview.mp4").resolve()),
        "VIDEO_A",
    ]
    assert ids(search="Домашняя") == [str((tmp_path / "interview.mp4").resolve())]
    assert ids(search="ничего") == []


def test_videos_list_rejects_an_unknown_source(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="source"):
        catalog.videos_list(db, source="vimeo")


def test_format_duration_handles_none_and_hours() -> None:
    assert catalog.format_duration(None) == "-"
    assert catalog.format_duration(0) == "0:00:00"
    assert catalog.format_duration(3_661_000) == "1:01:01"


def test_the_cli_adds_a_local_file_and_lists_it(
    data_dir: Path, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")
    app = build_app()
    added = runner.invoke(app, ["videos", "add", str(path)])
    assert added.exit_code == 0, added.output
    listed = runner.invoke(app, ["videos", "list"], env={"COLUMNS": "240"})
    assert listed.exit_code == 0, listed.output
    assert "clip" in listed.stdout
    assert "local" in listed.stdout


def test_the_cli_rejects_a_bad_kind_before_touching_the_database(
    data_dir: Path, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")
    result = runner.invoke(build_app(), ["videos", "add", str(path), "--kind", "opera"])
    assert result.exit_code == 2
    assert "must be one of" in result.stderr
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_catalog_videos.py -v`
Expected: every test errors with `AttributeError: module 'rytp.commands.catalog' has no attribute 'videos_add'`.

- [ ] **Step 3: Extend `rytp/commands/catalog.py`**

Replace the module's import block with this — it is the Task 11 block plus `Path` and three more model names:

```python
from pathlib import Path

from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError, RytpError
```

Extend `__all__` to `["channel_add", "channel_list", "format_duration", "looks_local", "resolve_channel", "truncate", "videos_add", "videos_list"]`, then append the following to the module, above the `register(...)` calls:

```python
# -- seams part 2 fills in --------------------------------------------


def _video_job_kinds() -> tuple[str, ...]:
    """Job kinds whose `jobs.target_id` is a `videos.id`.

    Derived from part 2's registry, never listed here: `target_kind` is a
    Python attribute of the registration (contracts §5), and the obvious
    eight kinds already miss part 3's `caption_words`.

    Part 1 ships before `rytp.jobs` exists. With no registry there are no
    job kinds and nothing creates job rows, so an empty tuple is the
    correct answer rather than a guess.
    """
    try:
        from rytp.jobs import JOB_KINDS
    except ImportError:
        return ()
    return tuple(
        sorted(
            name
            for name, kind in JOB_KINDS.items()
            if kind.target_kind == C.VIDEO_TARGET_KIND
        )
    )


def _probe_video(url: str) -> ChannelEntry:
    """Ask yt-dlp what a URL is. One metadata request, no download.

    Implemented by `rytp/acquire/ytdlp.py` (plan part 2). Kept behind a
    module-level function so part 1 can be tested without it and without
    a network.
    """
    try:
        from rytp.acquire.ytdlp import probe_video
    except ImportError as exc:
        raise RytpError(
            "registering a URL needs yt-dlp metadata; install the extra: "
            'pip install -e ".[yt-dlp]"'
        ) from exc
    return probe_video(url)


# -- videos -----------------------------------------------------------


def looks_local(target: str) -> bool:
    """True when `target` names a file on disk rather than a URL.

    A Windows path like `C:\\videos\\a.mp4` contains a colon, so the test
    is for a scheme separator, not for a colon.
    """
    if "://" in target:
        return False
    return Path(target).expanduser().exists()


def format_duration(ms: int | None) -> str:
    """Milliseconds as H:MM:SS, or the null placeholder."""
    if ms is None:
        return C.NULL_CELL
    seconds, _ = divmod(int(ms), C.MS_PER_SECOND)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def resolve_channel(db: Database, ref: str) -> int:
    """Find a channel by row id, URL or exact title."""
    row = q.get_channel(db, int(ref)) if ref.isdigit() else q.find_channel(db, ref)
    if row is None:
        raise NotFoundError(
            f"no channel matches {ref!r}; register it with: rytp channel add <url>"
        )
    return int(row["id"])


def check_choice(name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    """Guard a handler called directly, not through a surface."""
    if value is not None and value not in allowed:
        raise InvalidInputError(f"{name} must be one of: {', '.join(allowed)}")


def videos_add(
    db: Database,
    *,
    target: str,
    title: str | None = None,
    kind: str | None = None,
    channel: str | None = None,
) -> CommandResult:
    """Catalog one video from a URL or a local file. Nothing is downloaded."""
    target = target.strip()
    if not target:
        raise InvalidInputError("videos add needs a URL or a file path")
    check_choice("kind", kind, C.VIDEO_KINDS)
    channel_id = resolve_channel(db, channel) if channel else None

    if looks_local(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise NotFoundError(f"{path} is not a file")
        source = C.LOCAL_SOURCE
        # `videos` has no path column, so the resolved path is the
        # natural key that makes re-registration idempotent. Part 2's
        # acquire/local.py turns it into a 'container' asset.
        external_id = str(path)
        url: str | None = None
        resolved_title = title or path.stem
        resolved_kind = kind or C.DEFAULT_VIDEO_KIND
        duration_ms: int | None = None
        published_at: str | None = None
    elif "://" in target:
        entry = _probe_video(target)
        source = C.REMOTE_SOURCE
        external_id = entry.external_id
        url = entry.url
        resolved_title = title or entry.title
        resolved_kind = kind or entry.kind or C.DEFAULT_VIDEO_KIND
        duration_ms = entry.duration_ms
        published_at = entry.published_at
    else:
        raise NotFoundError(
            f"{target!r} is neither a URL nor an existing file"
        )

    video_id, created = q.upsert_video(
        db,
        source=source,
        kind=resolved_kind,
        channel_id=channel_id,
        external_id=external_id,
        url=url,
        title=resolved_title,
        duration_ms=duration_ms,
        published_at=published_at,
    )
    return CommandResult(
        columns=("id", "source", "kind", "title"),
        rows=((str(video_id), source, resolved_kind, truncate(resolved_title)),),
        message=f"{'added' if created else 'updated'} video {video_id}",
    )


def videos_list(
    db: Database,
    *,
    channel: str | None = None,
    kind: str | None = None,
    source: str | None = None,
    search: str | None = None,
    limit: int = C.DEFAULT_LIST_LIMIT,
) -> CommandResult:
    """List catalogued videos, newest first, narrowed by the given filters."""
    check_choice("kind", kind, C.VIDEO_KINDS)
    check_choice("source", source, C.VIDEO_SOURCES)
    channel_id = resolve_channel(db, channel) if channel else None
    rows = q.list_videos(
        db,
        channel_id=channel_id,
        kind=kind,
        source=source,
        search=search,
        limit=limit,
    )
    return CommandResult(
        columns=("id", "source", "kind", "duration", "channel", "title", "external id"),
        rows=tuple(
            (
                str(row["id"]),
                row["source"],
                row["kind"],
                format_duration(row["duration_ms"]),
                row["channel_title"] or C.NULL_CELL,
                truncate(row["title"]),
                row["external_id"] or C.NULL_CELL,
            )
            for row in rows
        ),
        message=plural(len(rows), "video"),
    )
```

Then append the two registrations at the bottom of the file:

```python
register(
    Command(
        name="videos.add",
        group="videos",
        summary="Catalog one video from a URL or a local file. Nothing is downloaded.",
        params=(
            Param("target", str, "A URL, or a path to a local media file.", positional=True),
            Param("title", str, "Title to store. Defaults to what the source reports.",
                  default=None),
            Param("kind", str, "Video kind.", default=None, choices=C.VIDEO_KINDS),
            Param("channel", str, "Channel id, URL or title to file it under.",
                  default=None),
        ),
        handler=videos_add,
    )
)

register(
    Command(
        name="videos.list",
        group="videos",
        summary="List catalogued videos.",
        params=(
            Param("channel", str, "Only this channel (id, URL or title).", default=None),
            Param("kind", str, "Only this kind.", default=None, choices=C.VIDEO_KINDS),
            Param("source", str, "Only this source.", default=None, choices=C.VIDEO_SOURCES),
            Param("search", str, "Substring of the title.", default=None, short="-s"),
            Param("limit", int, "Maximum rows.", default=C.DEFAULT_LIST_LIMIT, short="-n"),
        ),
        handler=videos_list,
    )
)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_catalog_videos.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/commands/catalog.py tests/test_catalog_videos.py
git commit -m "feat: add videos add and videos list"
```

---

### Task 13: channel.sync across the videos, streams and shorts listings

Design §13, "Catalog size is unverified": *"Enumeration of the main video listing found fewer videos than expected, most likely because live streams are listed separately. Cataloguing must cover the streams and shorts listings as well as the main one."* So `channel sync` walks three listings, not one, and the `kind` of each video comes from the listing it was found in.

Real enumeration is plan part 2's (`rytp/acquire/ytdlp.py::enumerate_channel`). This task defines the seam, walks the tabs, and fakes the enumerator in tests. `channel.sync` is `long_running=True`, so the TUI lists it but hands you the command line instead of running it.

**Files:**
- Modify: `rytp/commands/catalog.py`
- Test: `tests/test_catalog_sync.py`

**Interfaces:**
- Consumes: `rytp.constants.CHANNEL_TABS`, `rytp.models.ChannelEntry`, `rytp.db.queries` (`upsert_video`, `mark_channel_synced`).
- Produces: `channel_sync(db, *, channel, tabs=…)`, `parse_tabs(text) -> tuple[str, ...]`, `_enumerate_channel(channel_url, tabs) -> list[ChannelEntry]`; registered command `channel.sync` with `long_running=True`.
- Part 2 must expose `rytp.acquire.ytdlp.enumerate_channel(channel_url: str, *, tabs: Sequence[str] = CHANNEL_TABS, runner: object | None = None) -> list[ChannelEntry]`, returning entries **in tab order and without deduplicating** — the collision rule below depends on it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog_sync.py`:

```python
"""channel sync walks three listings. yt-dlp is never called."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError, RytpError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def entry(external_id: str, kind: str, title: str | None = None) -> ChannelEntry:
    return ChannelEntry(
        external_id=external_id,
        title=title or f"Title {external_id}",
        url=f"https://example.invalid/watch?v={external_id}",
        duration_ms=60_000,
        kind=kind,
        published_at=None,
    )


LISTINGS = {
    "videos": [entry("VIDEO_A", "video"), entry("VIDEO_B", "video")],
    "streams": [entry("VIDEO_B", "livestream"), entry("VIDEO_C", "livestream")],
    "shorts": [entry("VIDEO_D", "short")],
}


@pytest.fixture()
def fake_enumerate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Replace the yt-dlp seam with the fixed listings above."""
    seen: list[tuple[str, tuple[str, ...]]] = []

    def enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
        seen.append((channel_url, tuple(tabs)))
        found: list[ChannelEntry] = []
        for tab in tabs:
            found.extend(LISTINGS[tab])
        return found

    monkeypatch.setattr(catalog, "_enumerate_channel", enumerate_channel)
    return seen


@pytest.fixture()
def channel(db: Database) -> int:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    return 1


def test_sync_catalogs_every_listing_not_just_the_main_one(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    result = catalog.channel_sync(db, channel="1")
    assert fake_enumerate == [(CHANNEL_URL, ("videos", "streams", "shorts"))]
    kinds = dict(
        db.conn.execute("SELECT kind, COUNT(*) FROM videos GROUP BY kind").fetchall()
    )
    assert kinds == {"video": 1, "livestream": 2, "short": 1}
    assert "4" in (result.message or "")


def test_a_video_listed_in_two_tabs_takes_the_later_tabs_kind(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    """A live stream appears under /videos too; /streams is the truth."""
    catalog.channel_sync(db, channel="1")
    row = db.conn.execute(
        "SELECT kind FROM videos WHERE external_id = 'VIDEO_B'"
    ).fetchone()
    assert row["kind"] == "livestream"
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 4


def test_sync_files_everything_under_the_channel(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    assert (
        db.conn.execute("SELECT COUNT(*) FROM videos WHERE channel_id = 1").fetchone()[0]
        == 4
    )


def test_sync_records_when_it_ran(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    stamp = db.conn.execute("SELECT last_synced_at FROM channels").fetchone()[0]
    assert stamp is not None
    assert stamp.endswith("+00:00")


def test_re_syncing_updates_rather_than_duplicating(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    result = catalog.channel_sync(db, channel="1")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 4
    assert "0 new" in (result.message or "")


def test_the_result_breaks_the_count_down_by_kind(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    result = catalog.channel_sync(db, channel="1")
    assert result.columns == ("kind", "catalogued")
    assert dict((row[0], row[1]) for row in result.rows) == {
        "video": "1",
        "livestream": "2",
        "short": "1",
    }


def test_tabs_can_be_narrowed(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1", tabs="shorts")
    assert fake_enumerate == [(CHANNEL_URL, ("shorts",))]
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_tabs_are_always_walked_in_the_canonical_order(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    """Typed in any order, they run videos → streams → shorts, so kinds settle the same way."""
    catalog.channel_sync(db, channel="1", tabs="shorts, streams ,videos")
    assert fake_enumerate == [(CHANNEL_URL, ("videos", "streams", "shorts"))]


def test_an_unknown_tab_is_rejected(db: Database, channel: int) -> None:
    with pytest.raises(InvalidInputError, match="tab"):
        catalog.channel_sync(db, channel="1", tabs="videos,playlists")


def test_parse_tabs_rejects_an_empty_selection() -> None:
    with pytest.raises(InvalidInputError, match="tab"):
        catalog.parse_tabs("  ,  ")


def test_syncing_an_unknown_channel_is_an_error(db: Database) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.channel_sync(db, channel="7")


def test_the_enumerate_seam_explains_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced absent, so this keeps testing the hint instead of calling part 2."""
    import sys

    monkeypatch.setitem(sys.modules, "rytp.acquire", None)
    monkeypatch.setitem(sys.modules, "rytp.acquire.ytdlp", None)
    with pytest.raises(RytpError, match="yt-dlp"):
        catalog._enumerate_channel(CHANNEL_URL, ("videos",))


def test_sync_is_registered_as_long_running() -> None:
    assert commands.COMMANDS["channel.sync"].long_running is True


def test_the_cli_runs_a_sync(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
        return [entry("VIDEO_A", "video")]

    monkeypatch.setattr(catalog, "_enumerate_channel", enumerate_channel)
    app = build_app()
    assert runner.invoke(app, ["channel", "add", CHANNEL_URL]).exit_code == 0
    synced = runner.invoke(app, ["channel", "sync", "1"], env={"COLUMNS": "200"})
    assert synced.exit_code == 0, synced.output
    assert "1 new" in synced.stdout
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_catalog_sync.py -v`
Expected: `AttributeError: module 'rytp.commands.catalog' has no attribute 'channel_sync'`.

- [ ] **Step 3: Extend `rytp/commands/catalog.py`**

Replace the module's import block again — Task 12's block plus three names:

```python
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import (
    ChannelEntry,
    InvalidInputError,
    NotFoundError,
    RytpError,
    utc_now_iso,
)
```

Add `"channel_sync"` and `"parse_tabs"` to `__all__`, then append the seam and the handler:

```python
def _enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
    """List a channel's videos across the given tabs.

    Implemented by `rytp/acquire/ytdlp.py` (plan part 2). It must return
    entries in tab order and must not deduplicate: `channel_sync` relies
    on a later tab overwriting an earlier one's `kind`.
    """
    try:
        from rytp.acquire.ytdlp import enumerate_channel
    except ImportError as exc:
        raise RytpError(
            "channel sync needs yt-dlp to enumerate the channel; install the "
            'extra: pip install -e ".[yt-dlp]"'
        ) from exc
    return enumerate_channel(channel_url, tabs=tuple(tabs))


def parse_tabs(text: str) -> tuple[str, ...]:
    """Parse `--tabs`, always returning them in the canonical order.

    Order matters: a live stream is listed under both /videos and
    /streams, and the last listing to mention a video decides its kind.
    Sorting the user's selection into `constants.CHANNEL_TABS` order
    makes that outcome independent of how the flag was typed.
    """
    requested = {part.strip().lower() for part in text.split(",") if part.strip()}
    unknown = requested - set(C.CHANNEL_TABS)
    if unknown:
        raise InvalidInputError(
            f"unknown channel tab(s): {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(C.CHANNEL_TABS)}"
        )
    if not requested:
        raise InvalidInputError(
            f"channel sync needs at least one tab: {', '.join(C.CHANNEL_TABS)}"
        )
    return tuple(tab for tab in C.CHANNEL_TABS if tab in requested)


def channel_sync(
    db: Database, *, channel: str, tabs: str = ",".join(C.CHANNEL_TABS)
) -> CommandResult:
    """Re-enumerate a channel's listings and catalog everything found.

    Design §13: the main video listing alone undercounts the corpus,
    because live streams are listed separately. All three listings are
    walked by default.
    """
    channel_id = resolve_channel(db, channel)
    row = q.get_channel(db, channel_id)
    assert row is not None  # resolve_channel raised if it were missing
    selected = parse_tabs(tabs)

    entries = _enumerate_channel(row["url"], selected)
    added = 0
    refreshed = 0
    with db.transaction():
        for found in entries:
            _video_id, created = q.upsert_video(
                db,
                source=C.REMOTE_SOURCE,
                kind=found.kind or C.DEFAULT_VIDEO_KIND,
                channel_id=channel_id,
                external_id=found.external_id,
                url=found.url,
                title=found.title,
                duration_ms=found.duration_ms,
                published_at=found.published_at,
            )
            if created:
                added += 1
            else:
                refreshed += 1
        q.mark_channel_synced(db, channel_id, utc_now_iso())

    by_kind = Counter(
        kind
        for (kind,) in db.conn.execute(
            "SELECT kind FROM videos WHERE channel_id = ?", (channel_id,)
        )
    )
    return CommandResult(
        columns=("kind", "catalogued"),
        rows=tuple(
            (kind, str(by_kind[kind]))
            for kind in C.VIDEO_KINDS
            if by_kind.get(kind)
        ),
        message=(
            f"{added} new, {refreshed} refreshed from "
            f"{len(entries)} listing entries across {', '.join(selected)}"
        ),
    )
```

And the registration, at the bottom of the file:

```python
register(
    Command(
        name="channel.sync",
        group="channel",
        summary="Re-enumerate a channel's videos, live streams and shorts listings.",
        params=(
            Param("channel", str, "Channel id, URL or title.", positional=True),
            Param(
                "tabs",
                str,
                "Comma-separated listings to walk.",
                default=",".join(C.CHANNEL_TABS),
            ),
        ),
        handler=channel_sync,
        long_running=True,
    )
)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_catalog_sync.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/commands/catalog.py tests/test_catalog_sync.py
git commit -m "feat: sync a channel across its video, stream and short listings"
```

---

### Task 14: Removing a channel, and removing a video

Contracts §5 "Deletion": every entity gets a `remove`, so there is nothing you can create and not get rid of. Part 1 owns the two catalog ones, and they are deliberately asymmetric.

`channel.remove` takes the channel registration and **nothing else**. `videos.channel_id` is nullable for exactly this reason: someone dropping a channel has not asked to lose 1,600 downloaded videos, so its videos are orphaned. The foreign key has no `ON DELETE` clause, so the orphaning is an explicit `UPDATE` in the same transaction — without it SQLite refuses the delete.

`videos.remove` is the consequential one, and three things make it harder than it looks:

- **Cascades cover rows, not files.** `words`, `utterances`, `video_speakers`, `assets` and `video_acoustics` go by `ON DELETE CASCADE`. `media/{video_id}/`, `cache/wav/{video_id}.wav` and `transcripts/{video_id}.md` must be removed by hand.
- **`jobs` has no foreign key.** `jobs.target_id` is disambiguated by the row's `kind`, so a `transcribe` job for video 7 survives video 7 and a worker would later run it against nothing. Its jobs go in the same transaction as the row — and the *set* of kinds to cancel is derived from part 2's `JOB_KINDS` registry, never listed here. A hardcoded list went stale before any code existed: the eight obvious kinds miss part 3's `caption_words`.
- **A cut list on disk may name the video.** Parts 5 and 6 treat a dangling `video_id` as corrupt input and refuse to render, which is right, but the owner should hear about it now rather than at render time. So removal *warns*, naming the cut lists — it never blocks and never edits them.

Per contracts: `--dry-run` prints what would go, counting rows and bytes; `--yes` is required because files are deleted; removal is synchronous, never a job.

**Files:**
- Modify: `rytp/commands/catalog.py`
- Test: `tests/test_remove.py`

**Interfaces:**
- Consumes: `rytp.commands.resolve_video_id`, `rytp.config` (`paths`), `rytp.constants` (`VIDEO_TARGET_KIND`, `VIDEO_CASCADE_TABLES`, `CUTLISTS_DIRNAME`).
- Produces: `channel_remove(db, *, channel)`, `videos_remove(db, *, video, dry_run=False, yes=False)`, `video_files(video_id) -> list[Path]`, `cutlists_naming_video(video_id) -> list[str]`, `directory_bytes(path) -> int`, `_video_job_kinds() -> tuple[str, ...]`; registered commands `channel.remove`, `videos.remove`.
- Part 2 must expose `rytp.jobs.JOB_KINDS`, whose values carry `target_kind`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_remove.py`:

```python
"""Removing a channel, and removing a video with its rows and its files."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"
VIDEO_URL = "https://example.invalid/watch?v=VIDEO_A"
NOW = "2026-01-01T00:00:00+00:00"

PROBED = ChannelEntry(
    external_id="VIDEO_A",
    title="Утренний эфир",
    url=VIDEO_URL,
    duration_ms=3_600_000,
    kind="video",
    published_at=None,
)


@pytest.fixture()
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_probe_video", lambda url: PROBED)


@pytest.fixture()
def job_kinds(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """Stand in for part 2's JOB_KINDS registry.

    `caption_words` is here on purpose: it is the kind that a hardcoded
    list of the obvious eight would have missed, and the reason removal
    derives this set instead of listing it.
    """
    kinds = ("caption_words", "download", "transcribe")
    monkeypatch.setattr(catalog, "_video_job_kinds", lambda: kinds)
    return kinds


@pytest.fixture()
def video(db: Database, data_dir: Path, fake_probe: None, job_kinds: tuple[str, ...]) -> int:
    """One catalogued video with rows, files and a queued job."""
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
        " stem, source, engine) VALUES (1, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')"
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord,"
        " last_word_ord, text, normalized_text, stem_text)"
        " VALUES (1, 0, 100, 0, 0, 'да', 'да', 'да')"
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (1, 'SPEAKER_00', 'pyannote')"
    )
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, bytes, acquired_at)"
        " VALUES (1, 'audio', 'media/1/audio.m4a', 10, ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO video_acoustics (video_id, computed_at) VALUES (1, ?)", (NOW,)
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('transcribe', 1, 'pending', 'gpu', ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('caption_words', 1, 'pending', 'cpu', ?)",
        (NOW,),
    )
    layout = config.paths()
    media = config.ensure_dir(layout.media_dir(1))
    (media / "audio.m4a").write_bytes(b"0123456789")
    config.ensure_dir(layout.cache_wav(1).parent)
    layout.cache_wav(1).write_bytes(b"01234")
    config.ensure_dir(layout.transcript(1).parent)
    layout.transcript(1).write_text("# transcript\n", encoding="utf-8")
    return 1


# -- channel.remove ---------------------------------------------------


def test_removing_a_channel_orphans_its_videos(
    db: Database, data_dir: Path, fake_probe: None
) -> None:
    """Contracts §5: videos.channel_id is nullable for exactly this."""
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    result = catalog.channel_remove(db, channel="1")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert db.conn.execute("SELECT channel_id FROM videos").fetchone()[0] is None
    assert "1 video orphaned" in (result.message or "")


def test_removing_a_channel_with_no_videos_says_so(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    assert "0 videos orphaned" in (catalog.channel_remove(db, channel="1").message or "")


def test_removing_a_channel_by_url_or_title(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.channel_remove(db, channel=CHANNEL_URL)
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_removing_an_unknown_channel_is_an_error(db: Database) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.channel_remove(db, channel="7")


# -- videos.remove ----------------------------------------------------


def test_a_dry_run_deletes_nothing_and_counts_everything(
    db: Database, video: int, data_dir: Path
) -> None:
    result = catalog.videos_remove(db, video="1", dry_run=True)
    counts = dict((row[0], row[1]) for row in result.rows)
    assert counts["words"] == "1"
    assert counts["utterances"] == "1"
    assert counts["video_speakers"] == "1"
    assert counts["assets"] == "1"
    assert counts["video_acoustics"] == "1"
    assert counts["jobs"] == "2"
    assert counts["files"] == "3"
    # 10 bytes of audio + 5 of wav + the transcript.
    assert "bytes" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert config.paths().cache_wav(1).exists()


def test_removal_refuses_without_yes(db: Database, video: int, data_dir: Path) -> None:
    """Contracts §5: anything that deletes a file requires --yes."""
    with pytest.raises(InvalidInputError, match="--yes"):
        catalog.videos_remove(db, video="1")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_removal_takes_the_row_the_cascades_and_the_files(
    db: Database, video: int, data_dir: Path
) -> None:
    layout = config.paths()
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    for table in ("words", "utterances", "video_speakers", "assets", "video_acoustics"):
        assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert not layout.media_dir(1).exists()
    assert not layout.cache_wav(1).exists()
    assert not layout.transcript(1).exists()


def test_removal_cancels_every_kind_of_job_that_targeted_the_video(
    db: Database, video: int, data_dir: Path
) -> None:
    """`jobs` has no foreign key, so a worker would otherwise run against nothing.

    `caption_words` is part 3's, and is cancelled because the kinds come
    from the registry rather than from a list written here.
    """
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_the_video_job_kinds_come_from_part_twos_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contracts §5: `target_kind` is an attribute of the registration."""
    import sys
    import types

    registry = types.SimpleNamespace(
        JOB_KINDS={
            "download": types.SimpleNamespace(target_kind="video"),
            "caption_words": types.SimpleNamespace(target_kind="video"),
            "render": types.SimpleNamespace(target_kind="render"),
        }
    )
    monkeypatch.setitem(sys.modules, "rytp.jobs", registry)
    assert catalog._video_job_kinds() == ("caption_words", "download")


def test_with_no_job_registry_there_are_no_job_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Part 1 ships before rytp.jobs; nothing creates job rows either."""
    import sys

    monkeypatch.setitem(sys.modules, "rytp.jobs", None)
    assert catalog._video_job_kinds() == ()


def test_removal_leaves_another_videos_jobs_alone(
    db: Database, video: int, data_dir: Path, job_kinds: tuple[str, ...]
) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('transcribe', 2, 'pending', 'gpu', ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('render', 1, 'pending', 'cpu', ?)",
        (NOW,),
    )
    catalog.videos_remove(db, video="1", yes=True)
    remaining = {
        (row["kind"], row["target_id"])
        for row in db.conn.execute("SELECT kind, target_id FROM jobs")
    }
    # `render` targets a renders row, not a video, so its target_kind keeps
    # it out of the derived set and it is not ours to cancel.
    assert remaining == {("transcribe", 2), ("render", 1)}


def test_removal_leaves_the_channel_alone(
    db: Database, video: int, data_dir: Path
) -> None:
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1


def test_removal_warns_about_cut_lists_naming_the_video(
    db: Database, video: int, data_dir: Path
) -> None:
    """Parts 5 and 6 refuse to render a dangling video_id; say so now, not then."""
    cutlists = config.ensure_dir(config.paths().root / "cutlists")
    (cutlists / "monologue.toml").write_text(
        '[[fragment]]\nvideo_id = 1\nstart_ms = 0\nend_ms = 100\n', encoding="utf-8"
    )
    (cutlists / "other.toml").write_text(
        '[[fragment]]\nvideo_id = 2\nstart_ms = 0\nend_ms = 100\n', encoding="utf-8"
    )
    result = catalog.videos_remove(db, video="1", dry_run=True)
    assert "monologue" in (result.message or "")
    assert "other" not in (result.message or "")


def test_an_unreadable_cut_list_is_skipped_not_fatal(
    db: Database, video: int, data_dir: Path
) -> None:
    cutlists = config.ensure_dir(config.paths().root / "cutlists")
    (cutlists / "broken.toml").write_text("this is not toml = = =", encoding="utf-8")
    assert catalog.videos_remove(db, video="1", dry_run=True).rows


def test_removal_works_when_the_files_were_never_downloaded(
    db: Database, data_dir: Path, fake_probe: None, job_kinds: tuple[str, ...]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL)
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_a_video_may_be_named_by_external_id(
    db: Database, video: int, data_dir: Path
) -> None:
    catalog.videos_remove(db, video="VIDEO_A", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_removing_an_unknown_video_is_an_error(
    db: Database, data_dir: Path, job_kinds: tuple[str, ...]
) -> None:
    with pytest.raises(NotFoundError, match="no video matches"):
        catalog.videos_remove(db, video="404", yes=True)


def test_directory_bytes_sums_a_tree(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"12345")
    (tmp_path / "two").write_bytes(b"123")
    assert catalog.directory_bytes(tmp_path) == 8
    assert catalog.directory_bytes(tmp_path / "missing") == 0


def test_both_remove_commands_are_registered() -> None:
    from rytp import commands

    assert "channel.remove" in commands.COMMANDS
    assert "videos.remove" in commands.COMMANDS
    assert {"dry_run", "yes"} <= {
        param.name for param in commands.COMMANDS["videos.remove"].params
    }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_remove.py -v`
Expected: `AttributeError: module 'rytp.commands.catalog' has no attribute 'channel_remove'`.

- [ ] **Step 3: Extend `rytp/commands/catalog.py`**

Replace the import block one last time — Task 13's block plus the filesystem and resolver names removal needs:

```python
import shutil
import tomllib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rytp import config
from rytp import constants as C
from rytp.commands import Command, CommandResult, Param, register, resolve_video_id
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import (
    ChannelEntry,
    InvalidInputError,
    NotFoundError,
    RytpError,
    utc_now_iso,
)
```

Add `"channel_remove"`, `"cutlists_naming_video"`, `"directory_bytes"`, `"video_files"` and `"videos_remove"` to `__all__`, then append:

```python
# -- removal (contracts §5 "Deletion") --------------------------------


def channel_remove(db: Database, *, channel: str) -> CommandResult:
    """Remove a channel registration. Its videos are orphaned, not deleted.

    `videos.channel_id` is nullable for exactly this case (contracts §5):
    dropping a channel is not a request to lose its videos. The foreign
    key has no `ON DELETE` clause, so the orphaning is explicit — without
    it SQLite refuses the delete. No files are touched, so this needs
    neither `--dry-run` nor `--yes`.
    """
    channel_id = resolve_channel(db, channel)
    with db.transaction():
        orphaned = db.conn.execute(
            "UPDATE videos SET channel_id = NULL WHERE channel_id = ?", (channel_id,)
        ).rowcount
        db.conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
    return CommandResult(
        message=(
            f"removed channel {channel_id}; "
            f"{plural(orphaned, 'video')} orphaned, none deleted"
        )
    )


def directory_bytes(path: Path) -> int:
    """Total size of a file or of everything under a directory. 0 if absent."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def video_files(video_id: int) -> list[Path]:
    """Every path on disk that belongs to one video and exists right now.

    Cascades do not reach the filesystem (contracts §5), so these are
    removed by hand. `transcripts/{video_id}.md` is included with the
    other two: it is named after the video and would otherwise be a file
    about something that no longer exists.
    """
    layout = config.paths()
    candidates = [
        layout.media_dir(video_id),
        layout.cache_wav(video_id),
        layout.transcript(video_id),
    ]
    return [path for path in candidates if path.exists()]


def cutlists_naming_video(video_id: int) -> list[str]:
    """Names of cut lists that mention this video id.

    A warning, never a block. Parts 5 and 6 treat a dangling `video_id`
    as corrupt input and refuse to render; the owner should hear about
    it at removal time instead of at render time. A cut list that will
    not parse is skipped rather than guessed at.
    """

    def mentions(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("video_id") == video_id:
                return True
            return any(mentions(value) for value in node.values())
        if isinstance(node, list):
            return any(mentions(item) for item in node)
        return False

    directory = config.paths().root / C.CUTLISTS_DIRNAME
    if not directory.is_dir():
        return []
    found: list[str] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            with path.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if mentions(document):
            found.append(path.stem)
    return found


def videos_remove(
    db: Database, *, video: str, dry_run: bool = False, yes: bool = False
) -> CommandResult:
    """Remove a video: its row, everything that cascades from it, and its files.

    Synchronous and immediate — contracts §5 is explicit that removal is
    never a job, because a half-deleted entity recovered from a crashed
    queue is worse than a slow command.
    """
    video_id = resolve_video_id(db, video)

    counts = {
        table: int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        )
        for table in C.VIDEO_CASCADE_TABLES
    }
    job_kinds = _video_job_kinds()
    placeholders = ", ".join("?" for _ in job_kinds)
    counts["jobs"] = (
        int(
            db.conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE target_id = ? AND kind IN"
                f" ({placeholders})",
                (video_id, *job_kinds),
            ).fetchone()[0]
        )
        if job_kinds
        else 0
    )
    files = video_files(video_id)
    total_bytes = sum(directory_bytes(path) for path in files)
    counts["files"] = len(files)
    affected_cutlists = cutlists_naming_video(video_id)

    rows = tuple((name, str(value)) for name, value in counts.items())
    warning = ""
    if affected_cutlists:
        warning = (
            f"; warning: cut list(s) {', '.join(affected_cutlists)} reference this "
            "video and will not render until you edit them"
        )

    if dry_run:
        return CommandResult(
            columns=("what", "count"),
            rows=rows,
            message=(
                f"dry run: video {video_id} would be removed with "
                f"{total_bytes} bytes across {plural(len(files), 'path')}{warning}"
            ),
        )
    if not yes:
        raise InvalidInputError(
            f"videos remove deletes {total_bytes} bytes from disk and cannot be "
            "undone; re-run with --yes, or with --dry-run to see what would go"
        )

    # The database first, in one transaction: if a file then refuses to
    # go, the catalog is still consistent and the leftover is named.
    with db.transaction():
        if job_kinds:
            db.conn.execute(
                f"DELETE FROM jobs WHERE target_id = ? AND kind IN ({placeholders})",
                (video_id, *job_kinds),
            )
        db.conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))

    stubborn: list[str] = []
    for path in files:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            stubborn.append(f"{path} ({exc.strerror or exc})")
    if stubborn:
        warning += f"; could not delete: {', '.join(stubborn)}"

    return CommandResult(
        columns=("what", "count"),
        rows=rows,
        message=(
            f"removed video {video_id} and {total_bytes} bytes across "
            f"{plural(len(files), 'path')}{warning}"
        ),
    )
```

and register both at the bottom of the file:

```python
register(
    Command(
        name="channel.remove",
        group="channel",
        summary="Remove a channel registration. Its videos are orphaned, not deleted.",
        params=(Param("channel", str, "Channel id, URL or title.", positional=True),),
        handler=channel_remove,
    )
)

register(
    Command(
        name="videos.remove",
        group="videos",
        summary="Remove a video, everything derived from it, and its files on disk.",
        params=(
            Param("video", str, "Video id or external id.", positional=True),
            Param(
                "dry_run",
                bool,
                "Print what would be removed and change nothing.",
                default=False,
            ),
            Param(
                "yes",
                bool,
                "Confirm. Required, because this deletes files.",
                default=False,
            ),
        ),
        handler=videos_remove,
    )
)
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_remove.py -v`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/commands/catalog.py rytp/constants.py tests/test_remove.py
git commit -m "feat: add channel remove and videos remove"
```

---

### Task 15: `doctor`, and the health-check registry the other parts plug into

Contracts §5 "Health checks". One top-level command reports on everything the tool needs, and it must not know what those things are: parts 2, 3, 4 and 7 register their own checks the way they register job handlers. Part 1 owns the registry, the command, and three checks of its own — the interpreter, the data tree, and the schema version, all `required=True`.

Three rules the other parts inherit from this task, so they are worth stating exactly:

- **`ok` always tells the truth; `required` decides whether it matters.** A missing optional engine is `ok=False`, because it is in fact missing. Reporting it as `ok=True` to protect the exit code would make the output lie about the thing `doctor` exists to tell you. `required=False` on the check is what makes that absence non-fatal, and the report shows it as `advisory` rather than `FAILED`. `required=True` is right for ffmpeg, ffprobe, FTS5, the data tree and the schema; `required=False` for yt-dlp, ffplay, every engine, the GPU and `HF_TOKEN`.
- **A check never raises and never blocks.** `doctor` turns a check that raises anyway into an `ok=False` result rather than dying — a broken check is itself a finding.
- **Every failing check carries a `remedy`**: the exact command that would fix it, not a description of the problem. That applies to advisory failures too — "not installed" is only useful next to "install it with …".

`doctor` exits non-zero when any `required` check reports `ok=False`, and never otherwise. A handler cannot set an exit code (contracts §8: handlers raise, the surface formats), so the failure path raises `HealthCheckError`, whose message is the whole report — every check, passing, advisory and failing, with remedies. Nothing is lost on the way to stderr.

**Files:**
- Modify: `rytp/commands/__init__.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `rytp.config.paths`, `rytp.constants` (`MIN_PYTHON_VERSION`, `WRITE_PROBE_FILENAME`, `CHECK_OK`, `CHECK_FAILED`, `CHECK_ADVISORY`), `rytp.db.LATEST_VERSION`.
- Produces: `HealthCheck(name, summary, run, required=True)`, `HealthResult(ok, detail, remedy=None)`, `HEALTH_CHECKS`, `register_check(check) -> HealthCheck`, `HealthCheckError(RytpError)`, `run_health_checks(db) -> list[tuple[HealthCheck, HealthResult]]`, `health_status(check, result) -> str`, the `doctor` command, and the three part-1 checks `python`, `data-tree`, `schema`, all `required=True`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_doctor.py`:

```python
"""`doctor` and the registry parts 2, 3, 4 and 7 plug into."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import (
    HEALTH_CHECKS,
    HealthCheck,
    HealthCheckError,
    HealthResult,
    doctor,
    register_check,
    run_health_checks,
)
from rytp.db import Database, schema

runner = CliRunner()


@pytest.fixture()
def checks(monkeypatch: pytest.MonkeyPatch) -> dict[str, HealthCheck]:
    """An empty registry, so a test's checks do not leak into the next."""
    fresh: dict[str, HealthCheck] = {}
    monkeypatch.setattr(commands, "HEALTH_CHECKS", fresh)
    return fresh


def passing(name: str, detail: str = "found") -> HealthCheck:
    return HealthCheck(
        name=name, summary=f"Check {name}.", run=lambda db: HealthResult(True, detail)
    )


def failing(name: str, *, required: bool = True) -> HealthCheck:
    """A check that truthfully reports an absence. `required` decides the exit code."""
    return HealthCheck(
        name=name,
        summary=f"Check {name}.",
        run=lambda db: HealthResult(False, "not installed", remedy=f"install {name}"),
        required=required,
    )


def test_register_check_stores_and_returns_it(checks: dict[str, HealthCheck]) -> None:
    check = register_check(passing("ffmpeg"))
    assert checks["ffmpeg"] is check


def test_registering_the_same_name_twice_is_an_error(
    checks: dict[str, HealthCheck],
) -> None:
    register_check(passing("ffmpeg"))
    with pytest.raises(ValueError, match="already registered"):
        register_check(passing("ffmpeg"))


def test_checks_run_in_name_order(db: Database, checks: dict[str, HealthCheck]) -> None:
    for name in ("zulu", "alpha", "mike"):
        register_check(passing(name))
    assert [check.name for check, _ in run_health_checks(db)] == ["alpha", "mike", "zulu"]


def test_a_check_that_raises_becomes_a_finding_not_a_crash(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    def explode(database: Database) -> HealthResult:
        raise RuntimeError("the engine exploded")

    register_check(HealthCheck(name="engine", summary="Check.", run=explode))
    (check, result), = run_health_checks(db)
    assert check.name == "engine"
    assert result.ok is False
    assert "the engine exploded" in result.detail


def test_doctor_reports_every_check_when_all_pass(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    register_check(passing("ffmpeg", "ffmpeg 7.1"))
    register_check(passing("gpu", "RTX 3080"))
    result = doctor(db)
    assert result.columns == ("check", "status", "detail", "remedy")
    statuses = {row[0]: row[1] for row in result.rows}
    assert statuses == {"ffmpeg": "ok", "gpu": "ok"}
    assert result.message == "2 checks, all required ok"


def test_a_check_is_required_by_default(checks: dict[str, HealthCheck]) -> None:
    assert register_check(passing("ffmpeg")).required is True


def test_a_missing_optional_engine_is_reported_truthfully_and_is_not_fatal(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    """Contracts §5: ok tells the truth; `required=False` makes it non-fatal."""
    register_check(passing("ffmpeg"))
    register_check(failing("transcriber:gigaam", required=False))
    result = doctor(db)
    statuses = {row[0]: row[1] for row in result.rows}
    assert statuses == {"ffmpeg": "ok", "transcriber:gigaam": "advisory"}
    assert "1 advisory" in (result.message or "")


def test_an_advisory_failure_still_shows_its_remedy(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    """"Not installed" is only useful next to "install it with …"."""
    register_check(failing("yt-dlp", required=False))
    remedies = {row[0]: row[3] for row in doctor(db).rows}
    assert remedies["yt-dlp"] == "install yt-dlp"


def test_advisory_failures_alone_never_fail_the_command(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    for name in ("yt-dlp", "ffplay", "gpu"):
        register_check(failing(name, required=False))
    assert doctor(db).message == "3 checks, all required ok (3 advisory)"


def test_health_status_maps_the_three_cases(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    from rytp.commands import health_status

    required_check = failing("ffmpeg")
    advisory_check = failing("gpu", required=False)
    good = passing("schema")
    assert health_status(good, good.run(db)) == "ok"
    assert health_status(required_check, required_check.run(db)) == "FAILED"
    assert health_status(advisory_check, advisory_check.run(db)) == "advisory"


def test_doctor_raises_when_a_required_check_fails(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    register_check(passing("ffmpeg"))
    register_check(failing("ffprobe"))
    register_check(failing("gpu", required=False))
    with pytest.raises(HealthCheckError) as excinfo:
        doctor(db)
    report = str(excinfo.value)
    assert "1 of 3 required checks failed: ffprobe" in report
    assert "install ffprobe" in report
    # Passing and advisory checks survive into the report; nothing is lost.
    assert "ffmpeg" in report
    assert "advisory" in report


def test_the_interpreter_check_passes_on_a_supported_python(db: Database) -> None:
    result = HEALTH_CHECKS["python"].run(db)
    assert result.ok is True
    assert str(sys.version_info.major) in result.detail


def test_the_data_tree_check_passes_on_a_writable_tree(
    db: Database, data_dir: Path
) -> None:
    result = HEALTH_CHECKS["data-tree"].run(db)
    assert result.ok is True
    assert str(data_dir) in result.detail


def test_the_data_tree_check_fails_when_the_tree_is_absent(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RYTP_DATA", str(tmp_path / "nowhere"))
    result = HEALTH_CHECKS["data-tree"].run(db)
    assert result.ok is False
    assert result.remedy is not None
    assert "rytp" in result.remedy


def test_the_data_tree_check_leaves_no_probe_file_behind(
    db: Database, data_dir: Path
) -> None:
    HEALTH_CHECKS["data-tree"].run(db)
    assert [item.name for item in data_dir.iterdir() if item.name.startswith(".")] == []


def test_the_schema_check_passes_on_a_migrated_database(db: Database) -> None:
    result = HEALTH_CHECKS["schema"].run(db)
    assert result.ok is True
    assert str(schema.LATEST_VERSION) in result.detail


def test_the_schema_check_fails_on_a_database_left_behind(
    data_dir: Path,
) -> None:
    from rytp.config import paths

    database = Database(paths().db)
    try:
        database.migrate_to(1)
        result = HEALTH_CHECKS["schema"].run(database)
        assert result.ok is False
        assert result.remedy is not None
    finally:
        database.close()


def test_part_one_registers_exactly_its_three_checks() -> None:
    assert set(HEALTH_CHECKS) >= {"python", "data-tree", "schema"}


def test_every_part_one_check_is_required() -> None:
    """The interpreter, the data tree and the schema are not optional."""
    for name in ("python", "data-tree", "schema"):
        assert HEALTH_CHECKS[name].required is True


def test_every_registered_check_has_a_summary() -> None:
    for name, check in HEALTH_CHECKS.items():
        assert check.summary.strip(), name


def test_doctor_is_a_registered_top_level_command() -> None:
    assert commands.COMMANDS["doctor"].group == ""
    assert commands.COMMANDS["doctor"].cli_only is False


def test_the_cli_runs_doctor_and_exits_zero_on_a_healthy_install(
    data_dir: Path,
) -> None:
    result = runner.invoke(build_app(), ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "python" in result.stdout
    assert "schema" in result.stdout


def test_the_cli_exits_one_when_a_required_check_fails(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = dict(HEALTH_CHECKS)
    broken["ffmpeg"] = failing("ffmpeg")
    monkeypatch.setattr(commands, "HEALTH_CHECKS", broken)
    result = runner.invoke(build_app(), ["doctor"])
    assert result.exit_code == 1
    assert "install ffmpeg" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_cli_stays_at_zero_when_only_advisory_checks_fail(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh machine with no yt-dlp is healthy, and says so out loud."""
    advisory = dict(HEALTH_CHECKS)
    advisory["yt-dlp"] = failing("yt-dlp", required=False)
    monkeypatch.setattr(commands, "HEALTH_CHECKS", advisory)
    result = runner.invoke(build_app(), ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "advisory" in result.stdout
    assert "install yt-dlp" in result.stdout
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_doctor.py -v`
Expected: collection error — `ImportError: cannot import name 'HealthCheck' from 'rytp.commands'`.

- [ ] **Step 3: Add the health registry and `doctor` to `rytp/commands/__init__.py`**

Append after the speaker resolver and before the `_open_tui` block. `__all__` becomes, in full — ruff's `RUF022` wants isort order, which is CONSTANT_CASE, then CamelCase, then snake_case:

```python
__all__ = [
    "COMMANDS",
    "HEALTH_CHECKS",
    "PARAM_ALIASES",
    "PARAM_TYPES",
    "REQUIRED",
    "SPEAKER_PARAMS",
    "Command",
    "CommandResult",
    "HealthCheck",
    "HealthCheckError",
    "HealthResult",
    "Param",
    "SpeakerFilter",
    "cli_path",
    "doctor",
    "health_status",
    "leaf_name",
    "register",
    "register_check",
    "resolve",
    "resolve_speaker_filter",
    "resolve_video_id",
    "run_health_checks",
]
```

```python
# ---------------------------------------------------------------------
# Health checks (contracts §5 "Health checks")
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HealthResult:
    """What one check found.

    ``ok`` always tells the truth about what was found. A missing
    optional engine is ``ok=False``, because it is in fact missing — the
    alternative, reporting it as ``ok=True`` to protect the exit code,
    makes the output lie about the thing `doctor` exists to tell you.
    What makes an absence non-fatal is ``required=False`` on the check
    (contracts §5).
    """

    ok: bool
    detail: str
    remedy: str | None = None


@dataclass(frozen=True)
class HealthCheck:
    """One thing `doctor` looks at.

    ``required=True`` for anything the base install genuinely needs —
    ffmpeg, ffprobe, FTS5, the data tree, the schema version.
    ``required=False`` for yt-dlp, ffplay, every transcriber, aligner and
    diarizer, the GPU and ``HF_TOKEN``: their absence is reported, and
    does not affect the exit status.
    """

    name: str  # "ffmpeg", "transcriber:gigaam"
    summary: str
    run: Callable[[Database], HealthResult]
    required: bool = True  # False = advisory; absence is not a failure


HEALTH_CHECKS: dict[str, HealthCheck] = {}


class HealthCheckError(RytpError):
    """`doctor` found something broken. The message is the whole report."""


def register_check(check: HealthCheck) -> HealthCheck:
    """Add a check. Parts 2, 3, 4 and 7 call this from their own modules."""
    if not check.summary.strip():
        raise ValueError(f"health check {check.name!r} needs a summary")
    if check.name in HEALTH_CHECKS:
        raise ValueError(f"health check {check.name!r} is already registered")
    HEALTH_CHECKS[check.name] = check
    return check


def run_health_checks(db: Database) -> list[tuple[HealthCheck, HealthResult]]:
    """Run every registered check in name order. Never raises.

    A check that raises anyway becomes an ``ok=False`` result rather than
    taking `doctor` down with it: a broken check is itself a finding, and
    whether it is fatal is still decided by the check's ``required``.
    """
    results: list[tuple[HealthCheck, HealthResult]] = []
    for name in sorted(HEALTH_CHECKS):
        check = HEALTH_CHECKS[name]
        try:
            result = check.run(db)
        # Broad on purpose: a check must never take `doctor` down.
        except Exception as exc:
            result = HealthResult(
                ok=False,
                detail=f"the check itself raised: {exc}",
                remedy=f"fix or unregister the {name!r} health check",
            )
        results.append((check, result))
    return results


def _check_python(db: Database) -> HealthResult:
    major, minor = C.MIN_PYTHON_VERSION
    running = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= C.MIN_PYTHON_VERSION:
        return HealthResult(True, f"Python {running}")
    return HealthResult(
        False,
        f"Python {running}, need {major}.{minor} or newer",
        remedy=f"install Python {major}.{minor}+ and re-run scripts/bootstrap.sh",
    )


def _check_data_tree(db: Database) -> HealthResult:
    root = paths().root
    if not root.is_dir():
        return HealthResult(
            False,
            f"{root} does not exist",
            remedy="run any rytp command, for example `rytp channel list`, to create it",
        )
    probe = root / C.WRITE_PROBE_FILENAME
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return HealthResult(
            False,
            f"{root} is not writable ({exc.strerror or exc})",
            remedy=f"grant write access to {root}, or point RYTP_DATA somewhere writable",
        )
    return HealthResult(True, f"{root} exists and is writable")


def _check_schema(db: Database) -> HealthResult:
    found = db.schema_version()
    if found == LATEST_VERSION:
        return HealthResult(True, f"schema version {found}")
    if found < LATEST_VERSION:
        return HealthResult(
            False,
            f"schema version {found}, code expects {LATEST_VERSION}",
            remedy="run any rytp command, for example `rytp channel list`, to migrate",
        )
    return HealthResult(
        False,
        f"schema version {found} is newer than this code's {LATEST_VERSION}",
        remedy="update rytp: this database was written by a newer version",
    )


register_check(
    HealthCheck(
        name="python",
        summary="The interpreter is new enough.",
        run=_check_python,
    )
)
register_check(
    HealthCheck(
        name="data-tree",
        summary="The data tree exists and can be written to.",
        run=_check_data_tree,
    )
)
register_check(
    HealthCheck(
        name="schema",
        summary="The database is migrated to the version this code expects.",
        run=_check_schema,
    )
)


def health_status(check: HealthCheck, result: HealthResult) -> str:
    """One of the three status words. Only ``FAILED`` affects the exit code."""
    if result.ok:
        return C.CHECK_OK
    return C.CHECK_FAILED if check.required else C.CHECK_ADVISORY


def _health_rows(
    results: list[tuple[HealthCheck, HealthResult]],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            check.name,
            health_status(check, result),
            result.detail,
            result.remedy or C.NULL_CELL,
        )
        for check, result in results
    )


def doctor(db: Database, **_: object) -> CommandResult:
    """Report on everything rytp needs in order to work.

    An advisory check that reports an absence is shown as ``advisory``,
    with its remedy, and does not affect the exit status. Only a
    ``required`` check reporting ``ok=False`` is fatal (contracts §5).

    Raises:
        HealthCheckError: if a required check failed. Its message is the
            full report, so the surface's one-line error handling still
            shows every check and every remedy.
    """
    results = run_health_checks(db)
    rows = _health_rows(results)
    broken = [check.name for check, result in results if not result.ok and check.required]
    advisory = [
        check.name for check, result in results if not result.ok and not check.required
    ]
    columns = ("check", "status", "detail", "remedy")

    def summary() -> str:
        tail = f" ({len(advisory)} advisory)" if advisory else ""
        return f"{len(results)} checks, all required ok{tail}"

    if not broken:
        return CommandResult(columns=columns, rows=rows, message=summary())

    lines = [
        f"{len(broken)} of {len(results)} required checks failed: {', '.join(broken)}"
    ]
    for check, result in results:
        lines.append(
            f"  {check.name}: {health_status(check, result)} — {result.detail}"
        )
        if result.remedy and not result.ok:
            lines.append(f"      fix: {result.remedy}")
    raise HealthCheckError("\n".join(lines))


register(
    Command(
        name="doctor",
        group="",
        summary="Check that everything rytp needs is installed and working.",
        params=(),
        handler=doctor,
    )
)
```

The module's import block becomes, in full:

```python
import difflib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rytp import constants as C
from rytp.config import paths
from rytp.db import LATEST_VERSION, Database
from rytp.models import InvalidInputError, NotFoundError, RytpError
```

- [ ] **Step 4: Run the test and watch it pass**

Run: `python -m pytest tests/test_doctor.py -v`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add rytp/commands/__init__.py rytp/constants.py tests/test_doctor.py
git commit -m "feat: add doctor and the health-check registry"
```

---

### Task 16: Surface parity, lint, types, and a real end-to-end run

The test this whole design exists for: **the CLI exposes every registered command, the TUI exposes every one that is not `cli_only`, and neither surface invents one of its own.** Design §10 asks for alignment "by construction rather than by discipline"; this is the assertion that proves the construction works and will keep failing as parts 2–7 add commands, which is exactly what it is for. Part 7 registers `speakers.map` as `cli_only` and asserts the same behaviour, so the two tests are written to agree.

Then the gate. `CLAUDE.md` is blunt about this: "a green test run says almost nothing about whether the real pipeline works. Verify end-to-end behaviour by actually running `python -m rytp` against a real file." Step 6 does that.

**Files:**
- Create: `tests/test_surfaces.py`
- Modify: whichever files lint or type-check flags

**Interfaces:**
- Consumes: `rytp.cli.build_app`, `rytp.cli.command_paths`, `rytp.tui.palette.palette_entries`, `rytp.commands.COMMANDS`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Create `tests/test_surfaces.py`:

```python
"""The CLI and the TUI are generated from one dict, and this proves it.

This test is meant to fail whenever a later plan part adds a command to
one surface and not the other. That is the point of it. Part 7 asserts
the same `cli_only` behaviour for `speakers.map`, so the two must agree.
"""

from __future__ import annotations

from rytp.cli import build_app, command_paths
from rytp.commands import COMMANDS
from rytp.tui.palette import palette_entries


def cli_only_names() -> set[str]:
    """Surface launchers: on the CLI, deliberately absent from the palette."""
    return {name for name, cmd in COMMANDS.items() if cmd.cli_only}


def test_part_one_registers_the_commands_it_owns() -> None:
    """A registry that never got imported would make every assertion below vacuous."""
    assert set(COMMANDS) == {
        "channel.add",
        "channel.list",
        "channel.remove",
        "channel.sync",
        "videos.add",
        "videos.list",
        "videos.remove",
        "doctor",
        "tui",
    }


def test_every_entity_part_one_creates_can_also_be_removed() -> None:
    """Contracts §5: there is no entity you can create but not get rid of."""
    for group in ("channel", "videos"):
        assert f"{group}.add" in COMMANDS
        assert f"{group}.remove" in COMMANDS


def test_every_registered_command_is_reachable_from_the_cli() -> None:
    """Including the cli_only ones: the CLI hides nothing."""
    assert command_paths(build_app()) == set(COMMANDS)


def test_every_command_that_is_not_cli_only_is_reachable_from_the_tui() -> None:
    assert {entry.name for entry in palette_entries()} == set(COMMANDS) - cli_only_names()


def test_neither_surface_invents_a_command() -> None:
    cli = command_paths(build_app())
    tui = {entry.name for entry in palette_entries()}
    assert cli - tui == cli_only_names()
    assert tui - cli == set()


def test_the_tui_launcher_is_the_cli_only_command_part_one_registers() -> None:
    assert cli_only_names() == {"tui"}


def test_the_two_surfaces_agree_on_summaries() -> None:
    for entry in palette_entries():
        assert entry.summary == COMMANDS[entry.name].summary


def test_every_command_documents_every_parameter() -> None:
    """Both surfaces show `help`; an empty one is a hole in the UI."""
    for name, cmd in COMMANDS.items():
        for param in cmd.params:
            assert param.help.strip(), f"{name}.{param.name} has no help text"


def test_the_cli_help_lists_every_group() -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), ["--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for group in {cmd.group for cmd in COMMANDS.values() if cmd.group}:
        assert group in result.stdout
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_surfaces.py -v`
Expected: 9 passed. If `test_every_registered_command_is_reachable_from_the_cli` fails, the generator dropped a command — fix `build_app`, never the test's expectation.

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass, no warnings about unclosed databases.

- [ ] **Step 4: Lint**

Run: `python -m ruff check rytp tests`
Expected: `All checks passed!`

The module code in this plan was checked against ruff 0.16 with exactly the Task 2 configuration and is clean. If something does fire, the fix is almost never a `noqa`:
- `B008` — a `typer.Option(...)` written as a literal default. The generated callbacks build theirs at runtime, so only `_root_callback` could trip it; it uses `Annotated[T, typer.Option(...)] = default` for that reason. Keep that form.
- `RUF012` — a mutable class attribute needs `ClassVar`. Textual's `BINDINGS` is already annotated.
- `E501` over 100 columns in a migration — reflow the surrounding Python, never the DDL text copied from contracts §3.
- `RUF001/2/3` on a Cyrillic literal — already ignored in `pyproject.toml`; do not ASCII-fy the text instead.
- `SIM300` (Yoda condition) on `assert CONSTANT <= computed()` — flip it to `assert computed() >= CONSTANT`.

- [ ] **Step 5: Type-check**

Run: `python -m mypy rytp`
Expected: `Success: no issues found`.

The module code here was checked against mypy 2.3 with the Task 2 configuration and is clean. Four things it objected to are already handled, and re-introducing any of them will fail the gate — `warn_unused_ignores = true` means a *surplus* `type: ignore` is an error too:
- `cursor.lastrowid` is `int | None`. `rytp/db/queries.py` asserts it is not `None` after an INSERT rather than casting.
- `rytp.acquire.ytdlp` does not exist yet; the `[[tool.mypy.overrides]]` block in `pyproject.toml` covers it, not a per-line ignore that would go stale when part 2 lands.
- Textual's `App.BINDINGS` is `list[BindingType]`, and `list` is invariant, so `ClassVar[list[Binding]]` is rejected. Use `ClassVar[list[BindingType]]`.
- `snowballstemmer` ships no type stubs; the second `[[tool.mypy.overrides]]` block covers it.
- `run.__signature__ = …` in `rytp/cli.py` genuinely needs `# type: ignore[attr-defined]`. That one stays.

`sqlite3.Row` indexing returns `Any`, so wrap ids in `int(...)` at the boundary rather than casting.

- [ ] **Step 6: Drive the real CLI, not the test doubles**

A green suite proves the fakes agree with each other. Run the thing. Everything below happens under `.smoke/` at the repository root (gitignored alongside `.venv/` and `.pytest_tmp/`); add `.smoke/` to `.gitignore` if it is not there.

On macOS or Linux, with the virtualenv activated and the repository root as the working directory:

```sh
rm -rf .smoke && mkdir -p .smoke
printf 'not a real video' > .smoke/sample.mp4
export RYTP_DATA=.smoke/data
python -m rytp                                                        # help, exit 0
python -m rytp channel add https://example.invalid/c/CHANNEL_ONE --title "Channel One"
python -m rytp channel list
python -m rytp videos add .smoke/sample.mp4 --title "Sample"
python -m rytp videos add .smoke/sample.mp4 --title "Sample"          # idempotent
python -m rytp videos list
python -m rytp videos list --source local -n 5
python -m rytp videos add "https://example.invalid/watch?v=VIDEO_A"   # yt-dlp hint
python -m rytp videos add .smoke/sample.mp4 --kind opera              # usage error
python -m rytp channel sync 1                                         # yt-dlp hint
ls -la .smoke/data
unset RYTP_DATA
```

On Windows:

```powershell
Remove-Item -Recurse -Force .smoke -ErrorAction Ignore
New-Item -ItemType Directory .smoke | Out-Null
Set-Content -NoNewline .smoke\sample.mp4 'not a real video'
$env:RYTP_DATA = ".smoke\data"
python -m rytp
python -m rytp channel add https://example.invalid/c/CHANNEL_ONE --title "Channel One"
python -m rytp channel list
python -m rytp videos add .smoke\sample.mp4 --title "Sample"
python -m rytp videos add .smoke\sample.mp4 --title "Sample"
python -m rytp videos list
python -m rytp videos list --source local -n 5
python -m rytp videos add "https://example.invalid/watch?v=VIDEO_A"
python -m rytp videos add .smoke\sample.mp4 --kind opera
python -m rytp channel sync 1
python -m rytp doctor
python -m rytp videos remove 1 --dry-run
python -m rytp videos remove 1
python -m rytp videos remove 1 --yes
python -m rytp channel remove 1
Get-ChildItem .smoke\data
Remove-Item Env:\RYTP_DATA
```

Check, by eye:
- `python -m rytp` with no arguments prints help and exits 0.
- `videos add` twice leaves **one** row (`videos list` shows one).
- The two yt-dlp commands print one line naming the extra, exit 1, and show no traceback.
- `--kind opera` exits 2 and names the allowed kinds.
- `.smoke/data/` contains `rytp.db` and nothing but SQLite's own `-wal` / `-shm` siblings (which disappear on a clean close) — no `media/`, no `cache/`, no `output/`. Directories are created on demand, and nothing in part 1 demands one.
- `git status --porcelain` shows no new `data/` directory in the repository.

- [ ] **Step 7: Open the TUI once, by hand**

```sh
RYTP_DATA=.smoke/data python -m rytp tui
```

On Windows: `$env:RYTP_DATA = ".smoke\data"; python -m rytp tui`

Check: the palette lists all five catalog commands and **not** `tui` itself, which is `cli_only`; typing `videos` narrows it; highlighting `videos.list`, tabbing to the argument line and pressing Enter fills the table; highlighting `channel.sync` and pressing Enter prints the `rytp channel sync …` hint instead of running it. `ctrl+q` quits.

- [ ] **Step 8: Commit**

```bash
git add tests/test_surfaces.py
git add -u rytp tests
git commit -m "test: assert both surfaces expose exactly the registered commands"
```

---

## Notes for whoever executes this

**The code in this plan was run, not just written.** Every module and test block was extracted verbatim into a scratch package and checked against the dev venv (Python 3.14.6, typer 0.27.2, textual 8.2.8, ruff 0.16.8, mypy 2.3.1): 258 tests pass, `ruff check rytp tests` reports `All checks passed!`, `mypy rytp` reports `Success: no issues found`, and the Task 16 Step 6 command sequence produces exactly the output described there. The per-task "Expected: N passed" counts are measured, not estimated. If a step does not behave as written, suspect a transcription slip before suspecting the plan.

**Where the old code went.** Task 2 deletes it. Anything worth cribbing is at `git show 44fc214c:rytp/<path>` — the yt-dlp fake-runner split in `download/ytdlp.py`, the ffmpeg wrapper in `transcribe/extract.py`, the loudness normalizer, the migration runner. Design §12 lists what is worth keeping, and in every case it is the pattern, not the file.

**What part 1 deliberately does not do.** No downloading, no job queue, no assets, no audio, no words, no search, no assembly, no rendering. Every table for all of that exists from Task 6 and stays empty; parts 2–7 fill them. Two things part 1 *does* own that later parts extend rather than replace: the health-check registry (parts 2, 3, 4 and 7 register their own checks) and the shared speaker resolver (parts 4, 5 and 7 splice in `SPEAKER_PARAMS` and call `resolve_speaker_filter`).

**Do not put a real video id, channel id, channel name or URL in any file, including a test fixture or a commit message.** `VIDEO_A`, `CHANNEL_ONE` and `https://example.invalid/...` exist for this. The repository root currently holds a few untracked scratch files whose names contain a real id; never `git add -A`, and never commit them.
