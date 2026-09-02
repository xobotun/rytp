"""Concrete STT engine backed by ``faster-whisper``.

DESIGN §5.1: ``FasterWhisperEngine`` is the recommended default.

The ``faster_whisper`` package is an optional dependency installed via
``pip install rytp[stt]``. This module **lazy-imports** it inside
``__init__`` so the rest of the codebase can be imported in
environments where faster-whisper isn't installed — the user only pays
the import cost when they actually try to run transcription.

Public surface:

* :class:`FasterWhisperEngine` — STT engine that yields
  :class:`Word` records with absolute timestamps in milliseconds.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import engines
from rytp.engines import Word


class FasterWhisperEngine:
    """STT engine backed by the ``faster_whisper`` package.

    Args:
        model_size: Name of the Whisper model to load (e.g. ``"small"``,
            ``"medium"``, ``"large-v3"``). Passed to
            :class:`faster_whisper.WhisperModel`.
        device: ``"auto"`` (default), ``"cpu"``, ``"cuda"``, etc.
        compute_type: ``"default"``, ``"int8"``, ``"float16"``, etc.
            ``"int8"`` is a sensible default for laptops; the constructor
            doesn't pick for you so you stay in control.

    Raises:
        ImportError: if ``faster_whisper`` is not installed. The error
            message names the extra to install.
    """

    name = "faster-whisper"
    requires_hf_token = False

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Install with "
                "`pip install rytp[stt]`."
            ) from e

        self._model: Any = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> Iterable[Word]:
        """Yield :class:`Word` objects with absolute timestamps.

        Internally calls ``WhisperModel.transcribe`` with
        ``word_timestamps=True``. Yields one Word per token, with
        ``start_ms``/``end_ms`` computed from the underlying seconds
        and confidence taken from ``word.probability`` (defaulting
        to 1.0 when the attribute is absent).
        """
        segments, _info = self._model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
        )
        for segment in segments:
            words = getattr(segment, "words", None) or []
            for w in words:
                yield Word(
                    start_ms=int(w.start * 1000),
                    end_ms=int(w.end * 1000),
                    text=(w.word or "").strip(),
                    confidence=float(getattr(w, "probability", 1.0)),
                )


# Register so ``engines.resolve_stt("faster-whisper")`` works. The class
# itself is safe to import (it has no top-level faster_whisper
# dependency); instantiation is what fails when faster_whisper isn't
# installed.
engines.register_stt(FasterWhisperEngine)