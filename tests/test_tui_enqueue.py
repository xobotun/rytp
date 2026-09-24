"""Queue it, do not run it (design §5, §10)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rytp.commands import Command, CommandResult, Param
from rytp.db import Database
from rytp.tui import enqueue as E
from rytp.tui.app import RytpApp
from tests.consistency import load_registries

REGISTERED, _ = load_registries()

calls: list[dict[str, Any]] = []


def queueing(db: Database, **kwargs: Any) -> CommandResult:
    calls.append(dict(kwargs))
    if not kwargs.get("enqueue"):  # pragma: no cover - the bug this guards
        raise AssertionError("the TUI ran long work inline")
    return CommandResult(message="queued transcribe job 4 for video 1")


def never(db: Database, **kwargs: Any) -> CommandResult:  # pragma: no cover
    raise AssertionError("this command must never run from the TUI")


QUEUEABLE = Command(
    name="transcribe.run",
    group="transcribe",
    summary="Transcribe one video.",
    params=(
        Param("video", str, "Video id.", positional=True),
        Param("enqueue", bool, "Queue it for the worker.", default=False),
    ),
    handler=queueing,
    long_running=True,
)

FOREGROUND = Command(
    name="worker",
    group="",
    summary="Run the worker pools.",
    params=(Param("once", bool, "Drain and exit.", default=False),),
    handler=never,
    long_running=True,
)

SAMPLE = {"transcribe.run": QUEUEABLE, "worker": FOREGROUND}


def test_a_command_with_an_enqueue_flag_gets_a_plan() -> None:
    plan = E.enqueue_plan(QUEUEABLE, {"video": "VIDEO_A", "enqueue": False})
    assert plan is not None
    assert plan.command == "transcribe.run"
    assert plan.flag == "enqueue"
    assert plan.values["enqueue"] is True
    assert plan.values["video"] == "VIDEO_A"


def test_the_plan_does_not_mutate_what_the_user_typed() -> None:
    typed = {"video": "VIDEO_A", "enqueue": False}
    E.enqueue_plan(QUEUEABLE, typed)
    assert typed["enqueue"] is False


def test_the_older_spelling_is_accepted_too() -> None:
    """`render.run` calls it `--queue`. Accepting both is what lets this work
    before or after the vocabulary is unified."""
    render = Command(
        name="render.run",
        group="render",
        summary="Render a cut list.",
        params=(
            Param("name", str, "Cut list.", positional=True),
            Param("queue", bool, "Enqueue it.", default=False),
        ),
        handler=queueing,
        long_running=True,
    )
    plan = E.enqueue_plan(render, {"name": "demo", "queue": False})
    assert plan is not None and plan.flag == "queue"


def test_a_command_with_no_queued_form_gets_no_plan() -> None:
    assert E.enqueue_plan(FOREGROUND, {"once": True}) is None


def test_a_command_that_is_not_long_running_gets_no_plan() -> None:
    quick = Command(
        name="videos.list",
        group="videos",
        summary="List videos.",
        params=(Param("enqueue", bool, "Nonsense.", default=False),),
        handler=queueing,
    )
    assert E.enqueue_plan(quick, {"enqueue": False}) is None


def test_the_hint_for_a_foreground_command_says_why_and_what_to_type() -> None:
    hint = E.foreground_hint(FOREGROUND, {"once": True})
    assert "rytp worker" in hint
    assert E.FOREGROUND_ONLY["worker"] in hint


def test_every_long_running_command_is_classified() -> None:
    """The staleness guard. A new long-running command must either offer a
    queued form or say here why it cannot have one — the alternative is a
    command the TUI silently cannot start, which is the gap this task closes.
    """
    assert E.unclassified_long_running(REGISTERED) == ()


def test_the_foreground_table_names_no_command_that_does_not_exist() -> None:
    assert set(E.FOREGROUND_ONLY) <= set(REGISTERED)


def test_every_foreground_reason_is_a_sentence() -> None:
    for name, reason in E.FOREGROUND_ONLY.items():
        assert reason.strip() and not reason.endswith("."), name


# --- the wiring ------------------------------------------------------


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    async def main() -> None:
        app = RytpApp(db, commands=SAMPLE)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_running_a_queueable_command_enqueues_it(db: Database) -> None:
    calls.clear()

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("transcribe.run")
        app.run_selected("VIDEO_A")
        await pilot.pause()
        assert calls == [{"video": "VIDEO_A", "enqueue": True}]
        assert "queued" in app.status_text

    drive(db, body)


def test_a_foreground_command_still_hands_over_a_command_line(db: Database) -> None:
    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("worker")
        app.run_selected("once=true")
        await pilot.pause()
        assert "rytp worker" in app.status_text
        assert E.FOREGROUND_ONLY["worker"] in app.status_text

    drive(db, body)


def test_the_status_line_points_at_the_queue_screen(db: Database) -> None:
    """Enqueuing without a way to watch is worse than not enqueuing."""

    async def body(app: RytpApp, pilot: Any) -> None:
        app.select("transcribe.run")
        app.run_selected("VIDEO_A")
        await pilot.pause()
        assert "F7" in app.status_text

    drive(db, body)
