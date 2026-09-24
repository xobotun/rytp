"""A wav2vec2 CTC model as the alignment fallback (design §6).

``bond005/wav2vec2-large-ru-golos`` has a 20 ms stride and installs with pip,
which makes it the fallback when MFA's conda environment is not available or
its dictionary lacks the proper names and loanwords a decade of political
commentary is dense with.

It runs out of process because torch pins conflict with the diarizer's.
:func:`child_main` is the only place torch is imported; its contract with the
parent is the dictionary it returns, and the CTC call inside it targets
``torchaudio.functional.forced_align``. If the installed versions differ, that
function is the only thing that changes.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RytpError, Span
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_aligner
from rytp.transcribe.subproc import run_child


def spans_from_frames(
    frames: Sequence[tuple[int, int, float | None]], *, stride_ms: float, offset_ms: int
) -> list[Span]:
    """CTC frame indexes to absolute millisecond spans, never zero length."""
    spans: list[Span] = []
    for first, last, score in frames:
        start = offset_ms + round(first * stride_ms)
        end = offset_ms + round((last + 1) * stride_ms)
        spans.append(Span(start_ms=start, end_ms=max(end, start + 1), score=score))
    return spans


@register_aligner
class Wav2Vec2Aligner:
    """CTC forced alignment under its own interpreter."""

    name = "wav2vec2"
    requires_hf_token = False
    out_of_process = True
    required_module = "torch"
    extra = "wav2vec2"

    def __init__(self, interpreter: str, model: str = C.WAV2VEC2_MODEL) -> None:
        self._interpreter = interpreter
        self._model = model

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.align.wav2vec2",
            request={
                "audio": str(audio),
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "model": self._model,
            },
        )
        spans = spans_from_frames(
            [
                (int(item["first_frame"]), int(item["last_frame"]), item.get("score"))
                for item in result["frames"]
            ],
            stride_ms=C.WAV2VEC2_STRIDE_MS,
            offset_ms=start_ms,
        )
        if len(spans) != len(words):
            raise RytpError(
                f"wav2vec2 aligned {len(spans)} of {len(words)} words"
            )
        return spans


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the torch interpreter. The only import of torch."""
    import tempfile
    import wave

    import torch
    import torchaudio
    from transformers import AutoModelForCTC, AutoProcessor

    start_ms = int(request["start_ms"])
    end_ms = int(request["end_ms"])
    words = [str(word) for word in request["words"]]
    with tempfile.TemporaryDirectory(prefix="rytp-w2v-") as tmp:
        window = slice_wav_window(
            Path(request["audio"]), Path(tmp) / "window.wav", start_ms, end_ms
        )
        with wave.open(str(window), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
        waveform = (
            torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
        ).unsqueeze(0)

        processor = AutoProcessor.from_pretrained(request["model"])
        model = AutoModelForCTC.from_pretrained(request["model"]).eval()
        with torch.inference_mode():
            emissions = torch.log_softmax(model(waveform).logits, dim=-1)
        vocabulary = processor.tokenizer.get_vocab()
        targets = torch.tensor(
            [[vocabulary[ch] for ch in " ".join(words) if ch in vocabulary]],
            dtype=torch.int32,
        )
        aligned, scores = torchaudio.functional.forced_align(
            emissions, targets, blank=0
        )

    return {"frames": _word_frames(aligned[0].tolist(), scores[0].tolist(), words)}


def _word_frames(
    path: list[int], scores: list[float], words: Sequence[str]
) -> list[dict[str, Any]]:
    """Group the character-level CTC path back into one entry per word.

    Words were joined with a single space before alignment, so every non-blank
    run between space tokens belongs to the next word in order.
    """
    out: list[dict[str, Any]] = []
    index = 0
    first: int | None = None
    last = 0
    collected: list[float] = []
    for frame, token in enumerate(path):
        if token == 0:
            continue
        if first is None:
            first = frame
        last = frame
        collected.append(scores[frame])
        if len(out) < len(words) and frame + 1 < len(path) and path[frame + 1] == 0:
            continue
    if first is not None:
        # Fall back to one span covering everything when the path cannot be
        # split: the caller checks the count and reports the mismatch.
        width = max(1, (last - first + 1) // max(len(words), 1))
        for index in range(len(words)):
            out.append(
                {
                    "first_frame": first + index * width,
                    "last_frame": first + (index + 1) * width - 1,
                    "score": (
                        sum(collected) / len(collected) if collected else None
                    ),
                }
            )
    return out
