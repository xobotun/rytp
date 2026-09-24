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

from rytp.config import ensure_dir, paths
from rytp.db import Database


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
