"""Transcribe subsystem — audio extraction, chunking, STT engine plugins.

Public re-exports:

* :func:`chunk_audio` — plan STT chunks (and slice WAVs if requested).
* :class:`FasterWhisperEngine` — recommended STT engine. Importing this
  module does NOT require faster-whisper to be installed (it lazy-imports
  at instantiation time); however, importing the class itself is fine.
* :class:`STTEngine` — the protocol other engines conform to.
"""
from rytp.transcribe.base import Chunk, NullSTTEngine, STTEngine, Word, chunk_audio
from rytp.transcribe.chunking import Chunk as _Chunk
from rytp.transcribe.chunking import (
    ProbeError,
    chunk_audio as _chunk_audio,
    probe_duration_ms,
)
from rytp.transcribe.faster_whisper import FasterWhisperEngine

__all__ = [
    "STTEngine",
    "Word",
    "Chunk",
    "chunk_audio",
    "probe_duration_ms",
    "ProbeError",
    "NullSTTEngine",
    "FasterWhisperEngine",
]