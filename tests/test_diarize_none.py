"""The null diarizer: one label, the real duration, no model."""

from __future__ import annotations

import wave
from pathlib import Path

from rytp import constants as C
from rytp.diarize import base
from rytp.diarize.none import NullDiarizer


def _wav(path: Path, ms: int) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * (16 * ms))
    return path


def test_one_segment_covering_the_whole_file(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "a.wav", 2_500)
    segments = list(NullDiarizer().diarize(audio))
    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 2_500
    assert segments[0].local_label == C.NULL_DIARIZER_LABEL


def test_importing_the_package_registers_the_built_in_names() -> None:
    import rytp.diarize  # noqa: F401 - the import is the point

    assert base.resolve_diarizer("none") is NullDiarizer
    assert "pyannote" in base.DIARIZERS


def test_the_default_diarizer_constant_names_a_registered_engine() -> None:
    import rytp.diarize  # noqa: F401

    assert C.DEFAULT_DIARIZER in base.DIARIZERS
