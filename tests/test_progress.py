"""Tests for the progress seam (plan §1c, Task 8a; BUGS.md entries 3, 10, 22)."""

from __future__ import annotations

import io

import pytest

from rytp import constants as C
from rytp import progress


class _FakeTty(io.StringIO):
    """A stream that claims to be a terminal, so the default sink writes."""

    def isatty(self) -> bool:  # type: ignore[override]
        return True


class _FakeNonTty(io.StringIO):
    """A stream that is explicitly not a terminal."""

    def isatty(self) -> bool:  # type: ignore[override]
        return False


def test_format_line_with_done_and_total() -> None:
    assert progress.format_line("transcribe", 12, 38, "") == "transcribe 12/38"


def test_format_line_with_done_only() -> None:
    assert progress.format_line("download", 4096, None, "") == "download 4096"


def test_format_line_with_detail() -> None:
    assert (
        progress.format_line("download", 1, 2, "clip.mp4") == "download 1/2 clip.mp4"
    )


def test_format_line_bare_stage() -> None:
    assert progress.format_line("captions", None, None, "") == "captions"


def test_report_never_raises_with_no_sink_installed() -> None:
    with progress.install(None):
        progress.report("stage", done=1, total=2, detail="x")  # must not raise


def test_report_never_raises_when_the_sink_itself_raises() -> None:
    def boom(stage, done, total, detail):
        raise RuntimeError("a broken sink must not break the work it reports on")

    with progress.install(boom):
        progress.report("stage")  # must not raise


def test_install_overrides_the_current_sink() -> None:
    calls: list[tuple[str, int | None, int | None, str]] = []

    def sink(stage, done, total, detail):
        calls.append((stage, done, total, detail))

    with progress.install(sink):
        progress.report("download", done=1, total=2, detail="clip.mp4")
    assert calls == [("download", 1, 2, "clip.mp4")]


def test_install_restores_the_previous_sink_on_exit() -> None:
    calls: list[str] = []

    def outer(stage, done, total, detail):
        calls.append(f"outer:{stage}")

    with progress.install(outer):
        progress.report("before")
        with progress.install(lambda s, d, t, x: calls.append(f"inner:{s}")):
            progress.report("during")
        progress.report("after")
    assert calls == ["outer:before", "inner:during", "outer:after"]


def test_install_nesting_three_deep_unwinds_in_order() -> None:
    calls: list[str] = []
    with progress.install(lambda s, d, t, x: calls.append(f"1:{s}")):
        progress.report("a")
        with progress.install(lambda s, d, t, x: calls.append(f"2:{s}")):
            progress.report("b")
            with progress.install(lambda s, d, t, x: calls.append(f"3:{s}")):
                progress.report("c")
            progress.report("d")  # back to level 2, not level 1 or the default
        progress.report("e")     # back to level 1
    assert calls == ["1:a", "2:b", "3:c", "2:d", "1:e"]


def test_install_none_after_a_real_sink_silences_it_for_that_scope() -> None:
    calls: list[str] = []
    with progress.install(lambda s, d, t, x: calls.append(s)):
        with progress.install(None):
            progress.report("silenced")  # must not raise, must not be recorded
        progress.report("audible")
    assert calls == ["audible"]


def test_tty_sink_silent_when_stderr_is_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeNonTty()
    monkeypatch.setattr(progress.sys, "stderr", fake)
    sink = progress.TtySink()
    sink("stage", 1, 2, "")
    assert fake.getvalue() == ""


def test_tty_sink_writes_when_stderr_is_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTty()
    monkeypatch.setattr(progress.sys, "stderr", fake)
    sink = progress.TtySink()
    sink("download", 1, 2, "clip.mp4")
    assert "download 1/2 clip.mp4" in fake.getvalue()
    # No log-filling control characters when this is inspected later, only a
    # carriage return meant for a live terminal.
    assert "\x1b" not in fake.getvalue()


def test_tty_sink_throttles_repaints(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTty()
    monkeypatch.setattr(progress.sys, "stderr", fake)
    clock = {"t": 0.0}
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
    sink = progress.TtySink(interval_ms=C.PROGRESS_TTY_INTERVAL_MS)
    sink("stage", 1, 10, "")
    first = fake.getvalue()
    clock["t"] += 0.001  # 1 ms later: well under the 250 ms default interval
    sink("stage", 2, 10, "")
    assert fake.getvalue() == first  # the second repaint was suppressed


def test_tty_sink_always_writes_the_final_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeTty()
    monkeypatch.setattr(progress.sys, "stderr", fake)
    clock = {"t": 0.0}
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
    sink = progress.TtySink(interval_ms=C.PROGRESS_TTY_INTERVAL_MS)
    sink("stage", 1, 10, "")
    clock["t"] += 0.001
    sink("stage", 10, 10, "")  # done >= total: must go through despite the throttle
    assert "stage 10/10" in fake.getvalue()


def test_tty_sink_reevaluates_isatty_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirecting output mid-run (a log file) must silence it immediately."""
    fake = _FakeTty()
    monkeypatch.setattr(progress.sys, "stderr", fake)
    sink = progress.TtySink()
    sink("stage", 1, 2, "")
    assert fake.getvalue() != ""

    redirected = _FakeNonTty()
    monkeypatch.setattr(progress.sys, "stderr", redirected)
    sink("stage", 2, 2, "")
    assert redirected.getvalue() == ""
