"""Fake engines for tests: no model, no network, no optional extra.

Registering an engine mutates module-level dicts, so always go through
:func:`registered`, which restores both registries afterwards.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from rytp.models import RawWord, Span
from rytp.transcribe import registry


class FakeTranscriber:
    """Replays a fixed script of (start_ms, end_ms, text) triples."""

    name = "fake"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    script: tuple[tuple[int, int | None, str], ...] = (
        (0, 200, "один"),
        (200, 460, "два"),
        (460, 700, "три"),
    )

    def __init__(self, script: Sequence[tuple[int, int | None, str]] | None = None) -> None:
        self._script = tuple(script) if script is not None else self.script

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        for start, end, text in self._script:
            if start < start_ms:
                continue
            if end_ms is not None and start >= end_ms:
                continue
            yield RawWord(start_ms=start, end_ms=end, text=text, confidence=0.9)


class NoEndTimesTranscriber(FakeTranscriber):
    """A transcriber that emits starts only — the pipeline must demand an aligner."""

    name = "fake-no-ends"
    script = ((0, None, "один"), (200, None, "два"))


class TextOnlyTranscriber:
    """Text and nothing else.

    Legal since contracts §4 made both ``RawWord`` timings optional, and the
    reason the pipeline may never assume ``start_ms`` is an ``int``. Must be
    paired with an aligner.
    """

    name = "fake-text-only"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None
    words: tuple[str, ...] = ("один", "два", "три")

    def __init__(self, words: Sequence[str] | None = None) -> None:
        self._words = tuple(words) if words is not None else self.words

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        start_ms: int = 0,
        end_ms: int | None = None,
    ) -> Iterable[RawWord]:
        for text in self._words:
            yield RawWord(text=text)


class FakeAligner:
    """Spreads the words evenly across the window, one span each."""

    name = "fake-aligner"
    requires_hf_token = False
    out_of_process = False
    required_module: str | None = None
    extra: str | None = None

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        n = len(words)
        if n == 0:
            return []
        step = max(1, (end_ms - start_ms) // n)
        return [
            Span(start_ms=start_ms + i * step, end_ms=start_ms + (i + 1) * step, score=0.8)
            for i in range(n)
        ]


class ShortAligner(FakeAligner):
    """Returns one span too few — the count mismatch the pipeline must refuse."""

    name = "fake-short-aligner"

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        return super().align(audio, words, start_ms=start_ms, end_ms=end_ms)[:-1]


class GatedTranscriber(FakeTranscriber):
    name = "fake-gated"
    requires_hf_token = True


class OutOfProcessTranscriber(FakeTranscriber):
    name = "fake-remote"
    out_of_process = True

    def __init__(
        self,
        interpreter: str,
        script: Sequence[tuple[int, int | None, str]] | None = None,
    ) -> None:
        super().__init__(script)
        self.interpreter = interpreter


class MissingModuleTranscriber(FakeTranscriber):
    name = "fake-missing"
    required_module = "definitely_not_installed_xyz"
    extra = "fakeextra"


_MISSING = object()


@contextmanager
def registered(*classes: type) -> Iterator[None]:
    """Register engines for the duration of a test, then undo exactly that.

    Only the names this call added are removed afterwards. Snapshotting and
    restoring the whole registry would be wrong: a module registers itself at
    import time and Python imports it once, so wiping a real engine's name
    here would leave it unregistered for every later test file.
    """
    undo: list[tuple[dict[str, type], str, object]] = []
    try:
        for cls in classes:
            table = registry.ALIGNERS if hasattr(cls, "align") else registry.TRANSCRIBERS
            undo.append((table, cls.name, table.get(cls.name, _MISSING)))
            table[cls.name] = cls
        yield
    finally:
        for table, name, previous in reversed(undo):
            if previous is _MISSING:
                table.pop(name, None)
            else:
                table[name] = previous  # type: ignore[assignment]


class ScorelessAligner(FakeAligner):
    """An aligner that reports no per-word confidence — MFA behaves this way."""

    name = "fake-scoreless-aligner"

    def align(
        self, audio: Path, words: Sequence[str], *, start_ms: int, end_ms: int
    ) -> list[Span]:
        return [
            Span(start_ms=span.start_ms, end_ms=span.end_ms, score=None)
            for span in super().align(audio, words, start_ms=start_ms, end_ms=end_ms)
        ]
