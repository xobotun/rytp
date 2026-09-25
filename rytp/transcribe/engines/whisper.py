"""faster-whisper as a transcriber (design §6, "Recommended stack").

Whisper is middling at Russian — about 16.2 WER against GigaAM's 8.4 on clean
benchmarks — but it installs anywhere, needs no separate interpreter, and the
one published test on *noisy YouTube* audio reversed that ranking. It is the
default until `transcribe compare` says otherwise on this corpus.

Its word timestamps are the ones measured at 78.7% zero gaps, so pair it with
an aligner and boundary refinement before cutting anything.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RawWord
from rytp.transcribe.base import EngineUnavailable, shift_words, slice_wav_window
from rytp.transcribe.registry import register_transcriber


def words_from_segments(segments: Iterable[Any]) -> list[RawWord]:
    """Flatten faster-whisper segments into words, converting seconds to ms.

    The only part of this adapter testable without the model, which is why it
    is a module-level function rather than a method.
    """
    words: list[RawWord] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or ():
            text = str(getattr(word, "word", "") or "").strip()
            if not text:
                continue
            words.append(
                RawWord(
                    start_ms=int(float(word.start) * 1000),
                    end_ms=int(float(word.end) * 1000),
                    text=text,
                    confidence=float(getattr(word, "probability", 1.0)),
                )
            )
    return words


@register_transcriber
class FasterWhisperTranscriber:
    """Whisper through faster-whisper, in this process."""

    name = "whisper"
    requires_hf_token = False
    out_of_process = False
    required_module = "faster_whisper"
    extra = "whisper"

    #: Class-level default (plan §1b, contracts §6). Unlike the other four
    #: engines (entry 34), whisper already selects a device — ctranslate2's
    #: own ``"auto"`` resolves to CUDA when available — so this adapter only
    #: needed to *report* the resolved choice, not add placement.
    device = "auto"
    #: Contracts §6: a transcriber has no `words.align_score` to report, so
    #: this is a fixed, honest placeholder — not "unknown", which contracts
    #: §6 reserves for an engine that predates the attribute entirely.
    score_scale = C.ALIGN_SCALE_NONE

    def __init__(
        self,
        model: str = C.WHISPER_DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        self._model_name = model
        self._compute_type = compute_type
        self._model: Any = None
        #: Requested device, then — once the model has loaded — the
        #: concrete device ctranslate2 actually resolved ``"auto"`` to.
        #: Callers read this rather than re-deriving it (BUGS.md entry 34).
        self.device = device
        #: Non-fatal findings from the last :meth:`transcribe` call (contracts
        #: §6's engine notes channel). A genuine instance attribute, never a
        #: shared `ClassVar` — this engine never has anything to report, but
        #: the channel still has to be a real per-instance list to satisfy
        #: `Transcriber.notes` the same way as every other engine.
        self.notes: list[str] = []

    def _load(self) -> Any:
        if self._model is None:
            # BUGS.md entry 11: silence huggingface_hub's per-process
            # symlink warning before it can print it; `doctor`'s hf-cache
            # check says the same thing once, with the disk cost.
            os.environ.setdefault(C.HF_SYMLINK_WARNING_ENV, "1")
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise EngineUnavailable(
                    "faster-whisper is not installed: pip install rytp[whisper]"
                ) from exc
            self._model = WhisperModel(
                self._model_name, device=self.device, compute_type=self._compute_type
            )
            # ctranslate2 resolves "auto" internally; the constructed model
            # exposes the concrete choice it made, and that — not the
            # request — is the fact worth surfacing.
            resolved = getattr(self._model.model, "device", None)
            if resolved:
                self.device = str(resolved)
        return self._model

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        """Decode one window and return words on the whole-file timeline."""
        model = self._load()
        with tempfile.TemporaryDirectory(prefix="rytp-whisper-") as tmp:
            window = slice_wav_window(audio, Path(tmp) / "window.wav", start_ms, end_ms)
            segments, _info = model.transcribe(
                str(window), language=language, word_timestamps=True
            )
            # The generator must be drained before the temporary file goes.
            words = words_from_segments(segments)
        return shift_words(words, start_ms)
