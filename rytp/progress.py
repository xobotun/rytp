"""The progress seam. plan §1c, BUGS.md entries 3, 10, 22.

Nothing in this system reported that it was working: a download, a
first-run model fetch of several gigabytes, a WAV decode and a diarizer's
segment pass were all indistinguishable from a hang. This module is the one
seam every long-running stage reports through, so the fix lives here once
rather than four times.

Three consumers must not disagree, and none of them is visible to an
emitter:

* A **CLI run in a terminal** gets the default sink below with no change to
  ``rytp/cli.py`` — it is simply what :func:`report` resolves to until
  something else is installed.
* A **queued job** gets a sink the worker installs for the duration of one
  handler call, writing ``jobs.progress`` (contracts §5: advisory, lossy,
  cleared the moment the job leaves ``running``, never affecting job
  state — it is not ``jobs.note``).
* The **TUI** installs a sink that posts a message from the worker thread,
  so the event loop is never blocked (Task 11's territory; this module only
  has to make that installable).

Emitters call :func:`report` and know nothing about any of this. The unit is
whatever the work already counts — for transcription and alignment that is
the chunk, which already exists (``plan_chunks``, and the completion table
already prints ``chunks = 38``); for a download it is bytes; for a single
blocking call it is nothing more than "this stage started" and "this stage
finished".
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from rytp import constants as C

#: ``stage`` names the phase ("download", "transcribe", ...); ``done`` and
#: ``total`` are a count in whatever unit the caller already has (``None``
#: when there is none); ``detail`` is a short free-text tail (a filename, an
#: engine name). A sink must not raise — :func:`report` tolerates one that
#: does, but a well-behaved sink does its own trimming and formatting.
Sink = Callable[[str, "int | None", "int | None", str], None]


def format_line(stage: str, done: int | None, total: int | None, detail: str) -> str:
    """The one-line rendering every sink in this project shares.

    Public so the worker's database sink and the TUI's sink (Tasks 8b, 11)
    can produce the same text a terminal would have shown, instead of each
    inventing its own.
    """
    bits = [stage]
    if done is not None and total is not None:
        bits.append(f"{done}/{total}")
    elif done is not None:
        bits.append(str(done))
    if detail:
        bits.append(detail)
    return " ".join(bits)


def _is_final(done: int | None, total: int | None) -> bool:
    return done is not None and total is not None and done >= total


class TtySink:
    """The default sink: one carriage-return-updated line on ``stderr``.

    Silent whenever ``sys.stderr.isatty()`` is false, checked on every call
    rather than once at construction — piped output and pytest's captured
    stderr must see nothing, ever, so no existing assertion in the suite
    changes. Throttled to :data:`rytp.constants.PROGRESS_TTY_INTERVAL_MS`,
    except that the call which completes a count (``done >= total``) always
    goes through, so a fast final chunk is never swallowed by the throttle,
    and is followed by a newline so the next thing printed does not run into
    the progress line's tail.
    """

    def __init__(self, *, interval_ms: int = C.PROGRESS_TTY_INTERVAL_MS) -> None:
        self._interval_ms = interval_ms
        self._last_ms = float("-inf")
        self._last_len = 0

    def __call__(
        self, stage: str, done: int | None, total: int | None, detail: str
    ) -> None:
        if not sys.stderr.isatty():
            return
        finished = _is_final(done, total)
        now = time.monotonic() * 1000
        if now - self._last_ms < self._interval_ms and not finished:
            return
        self._last_ms = now
        text = format_line(stage, done, total, detail)
        pad = max(0, self._last_len - len(text))
        sys.stderr.write("\r" + text + (" " * pad))
        sys.stderr.flush()
        self._last_len = len(text)
        if finished:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._last_len = 0


#: The sink used until something installs another one. A single shared
#: instance, not a fresh one per call, so its throttle state persists across
#: an emitter's whole loop.
_default_sink: Sink = TtySink()

#: Resolved per call from a ``ContextVar`` (plan §1c), exactly like
#: ``rytp/config.py``'s ``paths()`` is resolved per call rather than cached
#: at import time — a worker thread and the main thread must never share a
#: sink by accident, and each thread's default is the module default until
#: that thread installs its own.
_current: ContextVar[Sink | None] = ContextVar("rytp_progress_sink", default=_default_sink)


def report(
    stage: str, *, done: int | None = None, total: int | None = None, detail: str = ""
) -> None:
    """Tell whichever sink is installed where one piece of work has gotten to.

    Emitters call this and know nothing about sinks. A sink that is missing
    (``None`` was installed, e.g. to silence a scope entirely) or one that
    raises must never break the work it is reporting on, so both are
    swallowed here rather than by every call site.
    """
    sink = _current.get()
    if sink is None:
        return
    with contextlib.suppress(Exception):
        sink(stage, done, total, detail)


@contextmanager
def install(sink: Sink | None) -> Iterator[None]:
    """Install ``sink`` for the dynamic extent of the ``with`` block.

    Re-entrant: an ``install`` nested inside another restores exactly the
    sink that was current before it on exit — the outer install, not the
    module default — so the worker's per-job database sink and a future
    TUI sink can each wrap a narrower scope without clobbering each other.
    Pass ``None`` to silence :func:`report` entirely for a scope.
    """
    token = _current.set(sink)
    try:
        yield
    finally:
        _current.reset(token)
