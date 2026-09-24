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
