"""STT engine interface — see DESIGN §5.1.

This module re-exports the engine protocol and adds a built-in
:class:`NullSTTEngine` so the rest of the pipeline has a sensible
default that errors loudly when invoked (instead of silently doing
nothing).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rytp import engines
from rytp.engines import STTEngine, Word
from rytp.transcribe.chunking import Chunk, chunk_audio

__all__ = [
    "STTEngine",
    "Word",
    "Chunk",
    "chunk_audio",
    "NullSTTEngine",
]


class NullSTTEngine:
    """STT engine that errors loudly when invoked.

    The default STT engine in v1. Users who install ``rytp[stt]`` get
    :class:`FasterWhisperEngine` registered automatically and can pick
    it with ``--stt faster-whisper``.
    """

    name = "null"
    requires_hf_token = False

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> Iterable[Word]:
        raise RuntimeError(
            "no STT engine configured. Install faster-whisper "
            "(`pip install rytp[stt]`) and pass --stt faster-whisper"
        )


# Register so ``engines.resolve_stt("null")`` works.
engines.register_stt(NullSTTEngine)