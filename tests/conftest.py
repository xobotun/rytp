"""Shared pytest fixtures.

A couple of things every test suite in this project wants:

* An isolated, on-disk SQLite database in a tmpdir — gives every test a
  fresh schema and avoids any cross-test bleed.
* A small, in-process monkeypatch for ``rytp.config.paths`` so tests
  don't accidentally write into the user's real ``data/`` tree.
* A handful of fake objects (videos, words, diarizer segments) used by
  the unit tests in each segment.

Note on the ``tmp_path`` fixture: this codebase ships a custom override
of pytest's built-in ``tmp_path``. The reason is that ``tempfile.mkdtemp``
followed by ``pathlib.Path.mkdir`` fails with ``PermissionError`` under
the development sandbox on Windows. Using ``os.makedirs`` directly avoids
the syscall mismatch. The custom fixture also manages cleanup with
``shutil.rmtree(ignore_errors=True)`` so a stale basetemp can never
break the next session.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from rytp import config
from rytp.db import Database


_PYTEST_TMP_ROOT = Path(__file__).resolve().parent.parent / ".pytest_tmp"


@pytest.fixture()
def tmp_path() -> Iterator[Path]:  # type: ignore[override]
    """Per-test scratch dir under a workspace-local root.

    Overrides pytest's built-in ``tmp_path`` to avoid the sandbox issues
    described at the top of this file.
    """
    os.makedirs(str(_PYTEST_TMP_ROOT), exist_ok=True)
    # Use a uuid-based name; tempfile.mkdtemp races badly under concurrent
    # collection and we don't need anything fancy here.
    name = f"t-{uuid.uuid4().hex[:8]}"
    d = _PYTEST_TMP_ROOT / name
    os.makedirs(str(d))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Iterator[Path]:
    """Replace the global data tree with a fresh tmp dir for this test."""
    paths = config.Paths(
        root=tmp_path,
        db=tmp_path / "rytp.db",
        media=tmp_path / "media",
        audio=tmp_path / "audio",
        output=tmp_path / "output",
        normalized=tmp_path / "output" / "normalized",
        transcripts=tmp_path / "transcripts",
    )
    # Use os.makedirs to dodge the pathlib/sandbox PermissionError on
    # freshly-mkdtemp'd parents (see module docstring).
    for sub in (
        paths.media,
        paths.audio,
        paths.output,
        paths.normalized,
        paths.transcripts,
    ):
        os.makedirs(str(sub), exist_ok=True)

    # Monkey-patch the module-level singleton for the duration of the test.
    old = config.paths
    config.paths = paths
    try:
        yield tmp_path
    finally:
        config.paths = old


@pytest.fixture()
def db(data_dir: Path) -> Iterator[Database]:
    """An opened, migrated Database tied to the tmp ``data_dir``."""
    database = Database(config.paths.db)
    database.migrate()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture()
def fake_video_row() -> dict[str, object]:
    """A minimal videos-row dict used by tests that need a video to exist."""
    return {
        "source": "youtube",
        "kind": "video",
        "channel_id": None,
        "youtube_id": "sample_youtube_id",
        "url": "https://www.youtube.com/watch?v=sample_youtube_id",
        "local_path": None,
        "title": "Sample video",
        "duration": 600,
        "published_at": None,
        "downloaded": False,
        "downloaded_path": None,
        "metadata_json": "{}",
    }