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
