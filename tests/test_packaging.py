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
#
# "pyannote.audio" (not the older "pyannote-audio" spelling): Part 7's
# `pyannote` extra names the real PyPI distribution behind
# speaker-diarization-community-1 (design §6), whose name has a dot, not
# a hyphen.
HEAVY_DEPENDENCIES = {"yt-dlp", "faster-whisper", "pyannote.audio", "torch"}


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
