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

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RytpError, Span
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_aligner
from rytp.transcribe.subproc import resolve_device, run_child


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
    #: `torchaudio.functional.forced_align` reports log-probabilities,
    #: unbounded and negative by definition (plan §1a, BUGS.md entry 13).
    score_scale = C.ALIGN_SCALE_LOGPROB

    def __init__(
        self,
        interpreter: str,
        model: str = C.WAV2VEC2_MODEL,
        device: str = C.ENGINE_DEFAULT_DEVICE,
    ) -> None:
        self._interpreter = interpreter
        self._model = model
        self._requested_device = device
        #: Class default until the first `align()` call resolves it to the
        #: concrete device the child actually used (plan §1b).
        self.device = device
        #: Non-fatal findings from the last :meth:`align` call (contracts §6's
        #: engine notes channel). The pipeline drains this via
        #: ``getattr(engine, "notes", [])`` into ``jobs.note``.
        self.notes: list[str] = []

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        self.notes = []
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.align.wav2vec2",
            request={
                "audio": str(audio),
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "model": self._model,
                "device": self._requested_device,
            },
        )
        self.notes.extend(str(note) for note in result.get("notes", []))
        self.device = str(result.get("device", self._requested_device))
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
        _guard_spans(spans, words)
        return spans


def _guard_spans(spans: Sequence[Span], words: Sequence[str]) -> None:
    """Entry 36's second guard: the span count was always right, never the shape.

    Catches a regression to equal division directly: spans must not go
    backwards in time, and — whenever the words themselves differ in length —
    the spans must not all be the same width either.
    """
    starts = [span.start_ms for span in spans]
    if any(later < earlier for earlier, later in pairwise(starts)):
        raise RytpError("wav2vec2 produced out-of-order spans")
    if len(spans) > 1 and len({len(word) for word in words}) > 1:
        widths = {span.end_ms - span.start_ms for span in spans}
        if len(widths) == 1:
            raise RytpError(
                "wav2vec2 produced equal-width spans for words of differing "
                "length; this looks like a fallback to equal division rather "
                "than a real alignment"
            )


def _vocabulary_case(vocabulary: Mapping[str, int]) -> str:
    """Whether the model's vocabulary spells Cyrillic letters lower or upper case.

    Wav2vec2 vocabularies are case-sensitive token maps. Encoding a word in the
    wrong case silently drops every one of its characters (entry 36's second
    defect was exactly this shape, for the delimiter). Probed on a common
    Cyrillic letter rather than assumed.
    """
    has_lower = "а" in vocabulary
    has_upper = "А" in vocabulary
    if has_lower and not has_upper:
        return "lower"
    if has_upper and not has_lower:
        return "upper"
    return "asis"


def _encode_word(word: str, vocabulary: Mapping[str, int], case: str) -> list[int]:
    if case == "lower":
        word = word.lower()
    elif case == "upper":
        word = word.upper()
    return [vocabulary[ch] for ch in word if ch in vocabulary]


def _resolve_delimiter(tokenizer: Any, vocabulary: Mapping[str, int], model: str) -> int:
    delimiter = getattr(tokenizer, "word_delimiter_token_id", None)
    if delimiter is not None:
        return int(delimiter)
    delimiter = vocabulary.get("|")
    if delimiter is not None:
        return int(delimiter)
    raise RytpError(
        f"wav2vec2 model '{model}' has no word-delimiter token in its "
        "vocabulary; cannot separate words for alignment"
    )


def build_target_ids(
    words: Sequence[str], vocabulary: Mapping[str, int], *, delimiter: int
) -> list[int]:
    """The full CTC target sequence: each word's own tokens, delimiter between.

    No leading or trailing delimiter — :func:`word_frames` infers a frame's
    word index from how many delimiter targets came before it, which only
    works if delimiters appear strictly *between* words. Pure Python and
    imports nothing heavy, so it is testable without torch.
    """
    case = _vocabulary_case(vocabulary)
    target_ids: list[int] = []
    for index, word in enumerate(words):
        tokens = _encode_word(word, vocabulary, case)
        if not tokens:
            raise RytpError(
                f"wav2vec2 cannot align the word '{word}': none of its "
                "characters are in the model's vocabulary"
            )
        if index:
            target_ids.append(delimiter)
        target_ids.extend(tokens)
    return target_ids


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the torch interpreter. The only import of torch."""
    import tempfile
    import wave

    import torch
    import torchaudio
    from transformers import AutoModelForCTC, AutoProcessor

    device = resolve_device(str(request.get("device", C.ENGINE_DEFAULT_DEVICE)))

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
        ).unsqueeze(0).to(device)

        processor = AutoProcessor.from_pretrained(request["model"])
        model = AutoModelForCTC.from_pretrained(request["model"]).eval().to(device)
        with torch.inference_mode():
            emissions = torch.log_softmax(model(waveform).logits, dim=-1)
        vocabulary = processor.tokenizer.get_vocab()
        delimiter = _resolve_delimiter(processor.tokenizer, vocabulary, request["model"])
        target_ids = build_target_ids(words, vocabulary, delimiter=delimiter)

        targets = torch.tensor([target_ids], dtype=torch.int32, device=device)
        aligned, scores = torchaudio.functional.forced_align(
            emissions, targets, blank=0
        )

    frames_out, notes = word_frames(
        aligned[0].tolist(), scores[0].tolist(), targets=target_ids, delimiter=delimiter
    )
    return {"frames": frames_out, "notes": notes, "device": device}


def word_frames(
    path: Sequence[int],
    scores: Sequence[float],
    *,
    targets: Sequence[int],
    delimiter: int,
    blank: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Group the character-level CTC path back into one entry per word.

    ``path`` and ``scores`` are :func:`torchaudio.functional.forced_align`'s
    per-frame token ids and per-frame scores. ``targets`` is the full target
    sequence that was aligned against — each word's own tokens, in order,
    with a single ``delimiter`` token between (not around) words.

    The path is a valid alignment of ``targets``, so the target position a
    frame belongs to is recoverable by walking it with a cursor: skip blanks,
    and advance the cursor when the emitted token differs from the previously
    emitted one, or when a blank came between two frames that emitted the same
    token (the standard CTC token-merge rule; this is what lets a doubled
    letter such as "аа" — two adjacent, identical targets separated by a
    forced-align blank — advance the cursor exactly once per occurrence
    rather than collapsing or double-counting them). A word's index is then
    simply how many delimiter targets the cursor has passed.

    Returns the per-word frame entries, pure Python, and a list of notes for
    any word that received no frames of its own — a narrow, observable
    fallback, not the silent equal-division this replaces.
    """
    num_words = list(targets).count(delimiter) + 1 if targets else 1

    buckets: list[list[tuple[int, float]]] = [[] for _ in range(num_words)]
    cursor = -1
    prev_token: int | None = None
    blank_since_prev = True
    for frame, token in enumerate(path):
        if token == blank:
            blank_since_prev = True
            continue
        if cursor == -1 or token != prev_token or blank_since_prev:
            cursor += 1
        prev_token = token
        blank_since_prev = False
        if cursor >= len(targets):
            # More non-blank runs than targets: alignment is corrupt. Ignore
            # the overflow; the caller's span-count guard reports the mismatch.
            continue
        if targets[cursor] == delimiter:
            continue
        word_index = targets[:cursor].count(delimiter)
        if word_index < num_words:
            buckets[word_index].append((frame, scores[frame]))

    resolved: list[dict[str, Any] | None] = [None] * num_words
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        frame_numbers = [frame for frame, _score in bucket]
        word_scores = [score for _frame, score in bucket]
        resolved[index] = {
            "first_frame": min(frame_numbers),
            "last_frame": max(frame_numbers),
            "score": sum(word_scores) / len(word_scores),
        }

    notes: list[str] = []
    out: list[dict[str, Any]] = []
    for index in range(num_words):
        entry = resolved[index]
        if entry is not None:
            out.append(entry)
            continue
        notes.append(
            f"wav2vec2: word {index} received no CTC frames; its boundary was "
            "interpolated between its neighbours"
        )
        left = next((resolved[j] for j in range(index - 1, -1, -1) if resolved[j]), None)
        right = next(
            (resolved[j] for j in range(index + 1, num_words) if resolved[j]), None
        )
        if left is not None and right is not None:
            first = left["last_frame"] + 1
            last = max(first, right["first_frame"] - 1)
        elif left is not None:
            first = left["last_frame"] + 1
            last = first
        elif right is not None:
            last = max(0, right["first_frame"] - 1)
            first = last
        else:
            first = last = 0
        out.append({"first_frame": first, "last_frame": last, "score": None})
    return out, notes
