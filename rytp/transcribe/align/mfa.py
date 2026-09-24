"""Montreal Forced Aligner as an aligner (design §6, "Recommended stack").

MFA is the best free forced aligner and the only one with a Russian acoustic
model, dictionary and G2P (``russian_mfa`` v3.1.0). Its published boundary
error is 12.5 ms median — on English; no aligner's Russian boundary error has
ever been measured, which is why `transcribe compare` exists.

MFA installs through conda into its own environment, so it is invoked out of
process: point the ``engine.interpreter.mfa`` setting at the Python inside
that environment and the ``mfa`` binary beside it is found automatically.
Download the models once with ``mfa model download acoustic russian_mfa`` and
``mfa model download dictionary russian_mfa``.

MFA reports no per-word confidence, so spans come back with ``score=None``
and the pipeline substitutes the measured boundary quality instead.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rytp import constants as C
from rytp.models import RytpError, Span
from rytp.transcribe.base import slice_wav_window
from rytp.transcribe.registry import register_aligner
from rytp.transcribe.subproc import run_child


def mfa_binary(interpreter: str) -> str:
    """The ``mfa`` executable that lives beside a configured interpreter."""
    name = "mfa.exe" if os.name == "nt" else "mfa"
    return str(Path(interpreter).parent / name)


def _quoted(line: str) -> str:
    first = line.find('"')
    last = line.rfind('"')
    return line[first + 1 : last] if 0 <= first < last else ""


#: Labels MFA can write into the ``words`` tier itself that are not words:
#: silence and short-pause markers, and out-of-vocabulary speech ("spn" —
#: "speech, non-word"). Audited for entry 36: MFA's mismatch check compares
#: interval count to word count, so any of these sitting in the words tier
#: would shift the count and either raise a spurious mismatch or, worse,
#: silently pair the wrong span with the wrong word.
_NON_WORD_LABELS = {"sil", "sp", "spn"}


def parse_textgrid(text: str) -> list[tuple[float, float, str]]:
    """Intervals of the ``words`` tier of a long-form Praat TextGrid.

    MFA writes one interval per word plus empty intervals for the silences
    between them; the empty ones are dropped, and so are ``sil``/``sp``/
    ``spn`` markers that MFA sometimes writes as labelled (non-empty)
    intervals in the words tier rather than as empty ones. A tier's own
    ``xmin``/``xmax`` are always overwritten by its first interval's before
    any ``text`` line appears, so no special case is needed for the header.
    """
    intervals: list[tuple[float, float, str]] = []
    in_words = False
    xmin: float | None = None
    xmax: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name ="):
            in_words = _quoted(line) == "words"
            continue
        if not in_words:
            continue
        try:
            if line.startswith("xmin ="):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax ="):
                xmax = float(line.split("=", 1)[1])
            elif line.startswith("text ="):
                label = _quoted(line).strip()
                if xmin is not None and xmax is not None and label and (
                    label not in _NON_WORD_LABELS
                ):
                    intervals.append((xmin, xmax, label))
                xmin = xmax = None
        except ValueError:
            xmin = xmax = None
    return intervals


def spans_from_intervals(
    intervals: Sequence[tuple[float, float, str]],
    *,
    offset_ms: int,
    score: float | None = None,
) -> list[Span]:
    """TextGrid seconds, relative to the window, to absolute millisecond spans."""
    return [
        Span(
            start_ms=offset_ms + round(start * 1000),
            end_ms=offset_ms + round(end * 1000),
            score=score,
        )
        for start, end, _label in intervals
    ]


@register_aligner
class MfaAligner:
    """Forced alignment through the MFA command line, out of process."""

    name = "mfa"
    requires_hf_token = False
    out_of_process = True
    required_module = None
    #: MFA is a conda-installed *binary*, not a `pip`-importable module, so
    #: `required_module` can never name it (BUGS.md entry 7). Bare name,
    #: platform suffix resolved by `rytp.transcribe.registry` the same way
    #: :func:`mfa_binary` resolves it for a real run.
    required_binary = "mfa"
    extra = "mfa"
    #: MFA reports no per-word confidence at all (contracts §3's score table).
    score_scale = C.ALIGN_SCALE_NONE
    #: MFA has no GPU path; plan §1b's "an engine with no torch" case.
    device = "n/a"

    def __init__(
        self,
        interpreter: str,
        acoustic_model: str = C.MFA_ACOUSTIC_MODEL,
        dictionary: str = C.MFA_DICTIONARY,
        device: str = C.ENGINE_DEFAULT_DEVICE,
    ) -> None:
        self._interpreter = interpreter
        self._acoustic_model = acoustic_model
        self._dictionary = dictionary
        # Accepted and ignored: plan §1b has every adapter take the parameter
        # uniformly (a caller may pass device= without special-casing MFA),
        # but MFA never runs on a GPU, so the instance attribute stays "n/a".
        del device
        #: MFA never has a non-fatal finding to report. Set per instance
        #: (never a class-level default) so it satisfies the `Aligner`
        #: protocol's `notes: list[str]` as a genuine, mutable instance
        #: attribute, exactly like the other two engines'; it simply never
        #: gets appended to.
        self.notes: list[str] = []

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        result = run_child(
            interpreter=self._interpreter,
            module="rytp.transcribe.align.mfa",
            request={
                "audio": str(audio),
                "words": list(words),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "binary": mfa_binary(self._interpreter),
                "acoustic_model": self._acoustic_model,
                "dictionary": self._dictionary,
            },
        )
        spans = [
            Span(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                score=item.get("score"),
            )
            for item in result["spans"]
        ]
        if len(spans) != len(words):
            raise RytpError(
                f"mfa aligned {len(spans)} of {len(words)} words; the dictionary "
                "is probably missing entries for this text"
            )
        return spans


def child_main(request: dict[str, Any]) -> dict[str, Any]:
    """Runs inside MFA's own environment. Shells out to the ``mfa`` binary."""
    import subprocess
    import tempfile

    start_ms = int(request["start_ms"])
    end_ms = int(request["end_ms"])
    words = [str(word) for word in request["words"]]
    with tempfile.TemporaryDirectory(prefix="rytp-mfa-") as tmp:
        root = Path(tmp)
        corpus = root / "corpus"
        corpus.mkdir()
        slice_wav_window(Path(request["audio"]), corpus / "window.wav", start_ms, end_ms)
        (corpus / "window.lab").write_text(" ".join(words), encoding="utf-8")
        output = root / "aligned"
        command = [
            str(request["binary"]),
            "align",
            str(corpus),
            str(request["dictionary"]),
            str(request["acoustic_model"]),
            str(output),
            "--clean",
            "--quiet",
            "--single_speaker",
            "--output_format",
            "long_textgrid",
        ]
        proc = subprocess.run(  # fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        grid = output / "window.TextGrid"
        if not grid.exists():
            raise RuntimeError(
                f"mfa align wrote no TextGrid (exit {proc.returncode}): "
                f"{proc.stderr.strip()[-800:]}"
            )
        intervals = parse_textgrid(grid.read_text(encoding="utf-8"))
    spans = spans_from_intervals(intervals, offset_ms=start_ms)
    return {
        "spans": [
            {"start_ms": span.start_ms, "end_ms": span.end_ms, "score": span.score}
            for span in spans
        ]
    }
