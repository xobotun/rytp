"""pyannote `speaker-diarization-community-1`, out of process (design §6).

Roughly half the speaker confusion of pyannote 3.x, which is worth having:
the confusion 3.x makes is host-versus-guest, and this corpus is mostly
host and guest.

Two properties shape the adapter. It needs a Hugging Face token, checked
against the *class* before anything is constructed, because building the
pipeline without one downloads two gigabytes and then fails. And it pins
versions that conflict with the transcriber's, so it runs under the
interpreter named by the `engine.interpreter.pyannote` setting and never
imports its dependency in this process at all.

:func:`child_main` runs inside that interpreter and is the only place
`pyannote.audio` is imported. **The library call is unverified** — nobody
here has run community-1. Its *contract*, the dictionary it returns, is
what the rest of this part depends on, and that is covered by tests.
Expect to rewrite the body of one function when you install the real thing.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.diarize.base import register_diarizer
from rytp.models import DiarSegment
from rytp.transcribe.subproc import load_cached, resolve_device, run_child


def segments_from_tracks(
    tracks: Sequence[tuple[float, float, str]],
) -> list[dict[str, Any]]:
    """pyannote's ``(start_s, end_s, label)`` triples to the wire shape.

    Pure, so the conversion is tested without the library. A zero-length
    turn is dropped: pyannote emits them at the edges of some pipelines and
    a segment with no duration cannot own a word.
    """
    out: list[dict[str, Any]] = []
    for start_s, end_s, label in tracks:
        start_ms = round(float(start_s) * 1000)
        end_ms = round(float(end_s) * 1000)
        if end_ms <= start_ms:
            continue
        out.append({"start_ms": start_ms, "end_ms": end_ms, "local_label": str(label)})
    return out


@register_diarizer
class PyannoteDiarizer:
    """Diarization under pyannote's own interpreter."""

    name = "pyannote"
    requires_hf_token = True
    out_of_process = True
    required_module = "pyannote.audio"
    extra = "pyannote"
    #: Class-level default (plan §1b, contracts §6); see :class:`GigaAMTranscriber`.
    device = C.ENGINE_DEFAULT_DEVICE
    #: Contracts §6: a diarizer has no `words.align_score` to report, so
    #: this is a fixed, honest placeholder — see
    #: :class:`rytp.transcribe.engines.whisper.FasterWhisperTranscriber`.
    score_scale = C.ALIGN_SCALE_NONE

    def __init__(
        self,
        interpreter: str,
        model: str = C.PYANNOTE_DIARIZATION_MODEL,
        hf_token: str | None = None,
        device: str = C.ENGINE_DEFAULT_DEVICE,
    ) -> None:
        self._interpreter = interpreter
        self._model = model
        self._hf_token = (
            hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        )
        #: Requested device, then the concrete device the last call used
        #: (BUGS.md entry 34) — see :class:`GigaAMTranscriber`.
        self.device = device
        #: Non-fatal findings from the last :meth:`diarize` call (contracts
        #: §6's engine notes channel). A genuine instance attribute — see
        #: :class:`rytp.transcribe.engines.whisper.FasterWhisperTranscriber`.
        self.notes: list[str] = []

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        """Whole file in, labelled turns out.

        Always the whole file, never a chunk: speaker labels are
        context-dependent and chunking makes them inconsistent between
        chunks. Only the transcriber chunks (design §6).
        """
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.diarize.pyannote",
            request={
                "audio": str(audio),
                "model": self._model,
                "hf_token": self._hf_token,
                "device": self.device,
            },
        )
        self.device = str(result.get("device") or self.device)
        return [
            DiarSegment(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                local_label=str(item["local_label"]),
            )
            for item in result["segments"]
        ]


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside pyannote's own interpreter. The only import of the library.

    The pipeline is cached across calls in this process, keyed by
    ``(model, device)`` — a diarizer already ran once per video before the
    persistent worker seam, but this now also spares every video after the
    first in one run from reloading it (the seam's "helps all four engines
    uniformly" goal). The HF token used to build a cached pipeline is
    whichever call first constructed it for that key; a token change
    mid-process would not take effect until the worker is torn down.
    """
    import torch
    from pyannote.audio import Pipeline

    device = resolve_device(str(request.get("device") or C.ENGINE_DEFAULT_DEVICE))
    model_name = str(request["model"])

    def _load() -> Any:
        built = Pipeline.from_pretrained(model_name, token=request["hf_token"])
        built.to(torch.device(device))
        return built

    pipeline = load_cached(("pyannote", model_name, device), _load)
    annotation = pipeline(request["audio"])
    tracks = [
        (segment.start, segment.end, label)
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]
    return {"segments": segments_from_tracks(tracks), "device": device}
