"""GigaAM v3 as a transcriber, out of process (design §6).

Russian-specific, MIT licensed, roughly half Whisper's Russian error rate on
clean benchmarks, 2-4 GB of VRAM. Two properties shape this adapter:

* It caps a call at 25 seconds, so the caller must chunk with
  :func:`rytp.audio.vad.plan_chunks`. A longer window is an error here rather
  than a silent truncation inside the model.
* It pins versions that conflict with the diarizer's, so it declares
  ``out_of_process = True`` and runs under the interpreter named by the
  ``engine.interpreter.gigaam`` setting.

:func:`child_main` runs inside *that* interpreter and is the only place the
``gigaam`` package is imported. Its call into the library targets the
documented ``load_model`` / ``transcribe`` API; if the installed version
differs, this function is the only thing that changes and its contract is the
dictionary it returns. **Verify per-word timestamp support when installing.**
If the installed model returns text without word times, treat gigaam as
text-only and always pair it with an aligner.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RawWord, RytpError
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_transcriber
from rytp.transcribe.subproc import load_cached, resolve_device, run_child


def words_from_gigaam(result: dict[str, Any], offset_ms: int) -> list[dict[str, Any]]:
    """Gigaam's word list to the serialisable shape the parent expects."""
    words: list[dict[str, Any]] = []
    for item in result.get("words") or ():
        text = str(item.get("word", "")).strip()
        if not text:
            continue
        words.append(
            {
                "start_ms": offset_ms + round(float(item["start"]) * 1000),
                "end_ms": offset_ms + round(float(item["end"]) * 1000),
                "text": text,
                "confidence": item.get("confidence"),
            }
        )
    return words


@register_transcriber
class GigaAMTranscriber:
    """GigaAM under its own interpreter, one window per call."""

    name = "gigaam"
    requires_hf_token = False
    out_of_process = True
    required_module = "gigaam"
    extra = "gigaam"
    max_window_ms = C.VAD_CHUNK_MAX_MS
    #: Class-level default (plan §1b, contracts §6). An instance's own
    #: ``device`` starts as whatever was requested and, after the first
    #: call, holds the concrete device the child actually used.
    device = C.ENGINE_DEFAULT_DEVICE
    #: Contracts §6: a transcriber has no `words.align_score` to report, so
    #: this is a fixed, honest placeholder — see
    #: :class:`rytp.transcribe.engines.whisper.FasterWhisperTranscriber`.
    score_scale = C.ALIGN_SCALE_NONE

    def __init__(
        self,
        interpreter: str,
        model: str = C.GIGAAM_DEFAULT_MODEL,
        device: str = C.ENGINE_DEFAULT_DEVICE,
    ) -> None:
        self._interpreter = interpreter
        self._model = model
        #: Requested device, then — once a call has returned — the concrete
        #: device that call actually used. Callers read this rather than
        #: re-deriving it, so a foreground result and a queued job's note
        #: both say what ran (BUGS.md entry 34).
        self.device = device
        #: Non-fatal findings from the last :meth:`transcribe` call (contracts
        #: §6's engine notes channel). A genuine instance attribute — see
        #: :class:`rytp.transcribe.engines.whisper.FasterWhisperTranscriber`.
        self.notes: list[str] = []

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        if end_ms is not None and end_ms - start_ms > self.max_window_ms:
            raise RytpError(
                f"gigaam accepts at most {self.max_window_ms} ms per call; got "
                f"{end_ms - start_ms} ms — chunk with rytp.audio.vad.plan_chunks"
            )
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.engines.gigaam",
            request={
                "audio": str(audio),
                "model": self._model,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "device": self.device,
            },
        )
        self.device = str(result.get("device") or self.device)
        return [
            RawWord(
                start_ms=int(word["start_ms"]),
                end_ms=None if word["end_ms"] is None else int(word["end_ms"]),
                text=str(word["text"]),
                confidence=word.get("confidence"),
            )
            for word in result["words"]
        ]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside gigaam's own interpreter. The only import of the library.

    **Verify the ``device`` kwarg per installed version** (module docstring):
    ``load_model``'s documented signature accepts it; if the installed
    version spells device placement differently, this function — and only
    this function — is what changes.

    The model load is cached across calls in this process via
    :func:`load_cached`, keyed by ``(model, device)`` — the whole point of
    the persistent worker seam (``rytp.transcribe.subproc``'s module
    docstring). Only the load is cached; each call still transcribes its own
    window.
    """
    import tempfile

    import gigaam

    device = resolve_device(str(request.get("device") or C.ENGINE_DEFAULT_DEVICE))
    start_ms = int(request["start_ms"])
    end_ms = request["end_ms"]
    model_name = str(request["model"])
    model = load_cached(
        ("gigaam", model_name, device), lambda: gigaam.load_model(model_name, device=device)
    )
    with tempfile.TemporaryDirectory(prefix="rytp-gigaam-") as tmp:
        window = slice_wav_window(
            Path(request["audio"]), Path(tmp) / "window.wav", start_ms, end_ms
        )
        result = model.transcribe(str(window))
    return {"words": words_from_gigaam(dict(result), start_ms), "device": device}
