"""Tests for ``rytp.config``."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rytp import config


def test_paths_from_root() -> None:
    root = Path("/tmp/rytp-test")
    p = config.Paths.from_root(root)
    assert p.root == root
    assert p.db == root / "rytp.db"
    assert p.media == root / "media"
    assert p.audio == root / "audio"
    assert p.output == root / "output"
    assert p.normalized == root / "output" / "normalized"


def test_paths_ensure_creates_dirs(tmp_path: Path) -> None:
    p = config.Paths.from_root(tmp_path / "data")
    # Pre-state: directories don't exist
    assert not p.media.exists()
    assert not p.audio.exists()
    assert not p.output.exists()
    assert not p.normalized.exists()

    result = p.ensure()

    # All directories exist after ensure()
    assert p.media.is_dir()
    assert p.audio.is_dir()
    assert p.output.is_dir()
    assert p.normalized.is_dir()
    # Idempotent and chainable
    assert result is p
    p.ensure()  # does not raise


def test_paths_ensure_is_idempotent(tmp_path: Path) -> None:
    p = config.Paths.from_root(tmp_path / "data")
    p.ensure()
    p.ensure()  # second call must not raise
    assert p.media.is_dir()


def test_default_paths_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RYTP_DATA", str(tmp_path / "alt"))
    # Re-evaluate the default by calling the internal helper
    fresh = config._default_paths()
    assert fresh.root == tmp_path / "alt"
    assert fresh.db == tmp_path / "alt" / "rytp.db"
    # The directories were created by _default_paths().ensure()
    assert (tmp_path / "alt" / "media").is_dir()


def test_default_paths_fallback_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RYTP_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    fresh = config._default_paths()
    assert fresh.root == tmp_path / "data"


def test_hf_token_prefers_HF_TOKEN(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_aaaa")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_bbbb")
    assert config.hf_token() == "hf_aaaa"


def test_hf_token_falls_back_to_HUGGINGFACE_TOKEN(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_cccc")
    assert config.hf_token() == "hf_cccc"


def test_hf_token_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert config.hf_token() is None


def test_load_settings_defaults() -> None:
    s = config.load_settings()
    assert s.queue_paused is False
    assert s.default_stt_engine == "faster-whisper"
    assert s.default_diarizer == "none"
    assert s.default_combined_engine is None
    assert s.default_splice_mode == "concat"
    assert s.worker_concurrency == 1


def test_ffmpeg_binary_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.shutil, "which", lambda _: None)
    assert config.ffmpeg_binary() is None


def test_module_singleton_paths_is_a_Paths() -> None:
    assert isinstance(config.paths, config.Paths)