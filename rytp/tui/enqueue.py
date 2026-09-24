"""Turning a long-running command into queued work (design §5, §10).

design §10: "Long-running work is a CLI worker process… The worker itself stays
a separate process." Part 1 honours that by refusing to run a `long_running`
command inline and printing the command line instead, which is correct and also
means the TUI cannot start an ingest.

The way out is not to relax the rule. Parts 3, 4 and 6 each gave their long
commands a flag that enqueues rather than runs — `transcribe run --enqueue`,
`index build --enqueue`, `render run --queue` — because contracts §5 already
said the TUI must not run them. Calling the handler with that flag forced true
writes one row to `jobs` and returns; the GPU is still touched only by a worker
the owner started himself.

Commands with no queued form keep Part 1's behaviour, with a reason attached.
That table's *domain* is derived from the registry rather than written down, so
a new long-running command with neither a flag nor a reason fails the suite by
name instead of quietly becoming unreachable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from rytp.commands import Command
from rytp.tui.palette import cli_invocation

__all__ = [
    "ENQUEUE_FLAGS",
    "FOREGROUND_ONLY",
    "EnqueuePlan",
    "enqueue_plan",
    "foreground_hint",
    "unclassified_long_running",
]

#: Boolean parameter names that mean "write the job row instead of doing the
#: work". `enqueue` is the spelling four of the five use and the one the
#: vocabulary test argues for; `queue` is `render.run`'s and is accepted so
#: this works either side of that rename.
ENQUEUE_FLAGS: Final[tuple[str, ...]] = ("enqueue", "queue")

#: Long-running commands with no queued form, and why. Each value is a clause
#: that reads after "cannot be queued: ". Keeping the reason beside the name
#: is what stops this becoming a list nobody can audit.
FOREGROUND_ONLY: Final[dict[str, str]] = {
    "worker": "it is the process that drains the queue",
    "channel.sync": "enumerating a channel is one network call, not per-video work",
    "fetch-video": "this is the do-it-now path; `rytp ingest` is its queued twin",
    "search.play": "playback is interactive — press F4 and play the hit there",
    "search.export": "it writes files into the directory you name, now",
    "transcribe.compare": "it is a measurement you read, not a batch",
    "speakers.diarize": "queue it with `speakers enqueue`, which is in the palette",
    "speakers.embed": "queue it with `speakers enqueue`, which embeds as it diarizes",
}


@dataclass(frozen=True)
class EnqueuePlan:
    """How to ask for one long-running command without running it."""

    command: str
    flag: str
    values: dict[str, Any]


def _flag_for(cmd: Command) -> str | None:
    """The name of this command's enqueue flag, if it has one."""
    boolean = {param.name for param in cmd.params if param.type is bool}
    return next((flag for flag in ENQUEUE_FLAGS if flag in boolean), None)


def enqueue_plan(cmd: Command, values: Mapping[str, Any]) -> EnqueuePlan | None:
    """The values to call this command's handler with, to queue it.

    Returns None when there is nothing to queue — either the command is not
    long-running, or it has no queued form and belongs in
    :data:`FOREGROUND_ONLY`.
    """
    if not cmd.long_running:
        return None
    flag = _flag_for(cmd)
    if flag is None:
        return None
    return EnqueuePlan(command=cmd.name, flag=flag, values={**values, flag: True})


def foreground_hint(cmd: Command, values: Mapping[str, Any]) -> str:
    """What to say about a long command that cannot be queued."""
    reason = FOREGROUND_ONLY.get(cmd.name, "it runs in the foreground")
    return f"{reason}; run it in a terminal: {cli_invocation(cmd, values)}"


def unclassified_long_running(commands: Mapping[str, Command]) -> tuple[str, ...]:
    """Long-running commands that neither queue nor explain why they cannot.

    The whole point of this function is to fail a test. A command that reaches
    this list is one the TUI lists and cannot start, which is the gap review
    finding 3.3 recorded.
    """
    return tuple(
        sorted(
            name
            for name, cmd in commands.items()
            if cmd.long_running and _flag_for(cmd) is None and name not in FOREGROUND_ONLY
        )
    )
