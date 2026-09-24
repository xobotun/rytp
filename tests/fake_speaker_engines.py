"""Fake diarizers and embedders: no model, no network, no optional extra.

Registering an engine mutates module-level dicts, so always go through
:func:`registered`, which undoes exactly what it did. Snapshotting the whole
registry and restoring it would be wrong: a real engine registers itself at
import time and Python imports a module once, so wiping its name here would
leave it unregistered for every later test file.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rytp import constants as C
from rytp.diarize import base, embed
from rytp.models import DiarSegment


class FakeDiarizer:
    """Replays a fixed script of (start_ms, end_ms, local_label) triples."""

    name = "fake-diarizer"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    script: tuple[tuple[int, int, str], ...] = (
        (0, 1_000, "SPEAKER_00"),
        (1_000, 2_000, "SPEAKER_01"),
        (2_000, 3_000, "SPEAKER_00"),
    )

    def __init__(self, script: Sequence[tuple[int, int, str]] | None = None) -> None:
        self._script = tuple(script) if script is not None else self.script

    def diarize(self, audio: Path) -> Iterable[DiarSegment]:
        for start, end, label in self._script:
            yield DiarSegment(start_ms=start, end_ms=end, local_label=label)


class SingleLabelDiarizer(FakeDiarizer):
    """Everything is one voice — what the null diarizer does, without a WAV."""

    name = "fake-single"
    script = ((0, 5_000, C.NULL_DIARIZER_LABEL),)


class GatedDiarizer(FakeDiarizer):
    name = "fake-gated-diarizer"
    requires_hf_token = True


class OutOfProcessDiarizer(FakeDiarizer):
    name = "fake-remote-diarizer"
    out_of_process = True

    def __init__(
        self,
        interpreter: str,
        script: Sequence[tuple[int, int, str]] | None = None,
    ) -> None:
        super().__init__(script)
        self.interpreter = interpreter


class FakeEmbedder:
    """Returns a vector derived from the windows, so tests can predict it.

    The first component is the total window length in seconds and the rest
    are fixed, which makes two labels with different amounts of speech land
    in different directions without any model involved.
    """

    name = "fake-embedder"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    dim = 4

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> list[float]:
        total_s = sum(end - start for start, end in windows) / 1000.0
        return [total_s, 1.0, 0.0, 0.0]


class ConstantEmbedder(FakeEmbedder):
    """Always the same vector — two labels then look like the same person."""

    name = "fake-constant-embedder"
    vector: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)

    def embed(self, audio: Path, windows: Sequence[tuple[int, int]]) -> list[float]:
        return list(self.vector)


_MISSING = object()


@contextmanager
def registered(*classes: type) -> Iterator[None]:
    """Register engines for the duration of a test, then undo exactly that."""
    undo: list[tuple[dict[str, type], str, object]] = []
    try:
        for cls in classes:
            table = base.DIARIZERS if hasattr(cls, "diarize") else embed.EMBEDDERS
            undo.append((table, cls.name, table.get(cls.name, _MISSING)))
            table[cls.name] = cls
        yield
    finally:
        for table, name, previous in reversed(undo):
            if previous is _MISSING:
                table.pop(name, None)
            else:
                table[name] = previous  # type: ignore[assignment]
