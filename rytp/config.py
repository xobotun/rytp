"""Runtime configuration: paths, env vars, defaults."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Paths:
    """Filesystem layout rooted at ``root``."""

    root: Path
    db: Path
    media: Path
    audio: Path
    output: Path
    normalized: Path
    transcripts: Path

    @classmethod
    def from_root(cls, root: Path) -> Paths:
        """Construct a full ``Paths`` from a single root directory."""
        return cls(
            root=root,
            db=root / "rytp.db",
            media=root / "media",
            audio=root / "audio",
            output=root / "output",
            normalized=root / "output" / "normalized",
            transcripts=root / "transcripts",
        )

    def ensure(self) -> Paths:
        """Create every directory on disk and return self for chaining."""
        for p in (self.media, self.audio, self.output, self.normalized, self.transcripts):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _default_paths() -> Paths:
    """Compute the default paths from the environment."""
    root_env = os.environ.get("RYTP_DATA")
    root = Path(root_env) if root_env else Path.cwd() / "data"
    return Paths.from_root(root).ensure()


paths: Paths = _default_paths()
"""Module-level singleton used by the rest of the codebase."""


def hf_token() -> str | None:
    """Return the Hugging Face token from the environment.

    Checks ``HF_TOKEN`` first, then ``HUGGINGFACE_TOKEN`` as a fallback.
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


@dataclass(slots=True)
class Settings:
    """User-tunable defaults, persisted in the ``settings`` table (later)."""

    queue_paused: bool = False
    default_stt_engine: str = "faster-whisper"
    default_diarizer: str = "none"
    default_combined_engine: str | None = None
    default_splice_mode: str = "concat"
    worker_concurrency: int = 1


def load_settings() -> Settings:
    """Return the current settings.

    For now this returns hard-coded defaults. Later segments will wire this
    through the ``settings`` table in the database.
    """
    return Settings()


def ffmpeg_binary() -> str | None:
    """Return the path to ``ffmpeg`` if found on ``PATH``, else ``None``."""
    return shutil.which("ffmpeg")