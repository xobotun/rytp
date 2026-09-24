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
