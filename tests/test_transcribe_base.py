"""Tests for ``rytp.transcribe.base`` (re-exports + NullSTTEngine)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rytp import engines
from rytp.transcribe.base import NullSTTEngine, chunk_audio


def test_reexports_work() -> None:
    from rytp.transcribe.base import STTEngine, Word

    assert STTEngine is engines.STTEngine
    assert callable(chunk_audio)


def test_null_engine_metadata() -> None:
    assert NullSTTEngine.name == "null"
    assert NullSTTEngine.requires_hf_token is False


def test_null_engine_transcribe_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no STT engine"):
        list(NullSTTEngine().transcribe(tmp_path / "audio.wav"))


def test_null_engine_is_registered() -> None:
    assert "null" in engines.STTS
    assert engines.STTS["null"] is NullSTTEngine