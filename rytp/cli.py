"""The command-line surface, generated from the registry (design §10).

Nothing in this module defines an operation. :func:`build_app` walks
``rytp.commands.COMMANDS`` and synthesises one Typer callback per entry,
so a command added to the registry appears on the CLI and in the TUI at
the same moment, with the same name, arguments and help.

Nothing is added by hand, ``rytp tui`` included: it is a registry command
flagged ``cli_only`` (contracts §5), because a TUI able to launch itself
would be nonsense. ``tests/test_surfaces.py`` asserts the CLI shows every
registered command and the TUI shows every one that is not ``cli_only``.

Non-ASCII in help text is deliberate: the design is cited as "§10" and
em dashes are used throughout. :func:`_configure_stdio` forces UTF-8 on
stdout and stderr because this project's owner runs Windows, where the
default code page turns those into mojibake.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from rytp import config
from rytp import constants as C
from rytp.commands import (
    COMMANDS,
    PARAM_ALIASES,
    REQUIRED,
    Command,
    CommandResult,
    Param,
    leaf_name,
)
from rytp.db import Database
from rytp.models import RytpError

__all__ = ["build_app", "command_paths", "main", "open_database", "render_result"]


def _configure_stdio() -> None:
    """Force UTF-8 on the standard streams; no-op where that is impossible."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def open_database() -> Database:
    """Open and migrate the database, creating the data root if absent.

    This is the only place in part 1 that creates a directory, which is
    what contracts §7 means by "created on demand by the command that
    needs it".
    """
    layout = config.paths()
    config.ensure_dir(layout.root)
    database = Database(layout.db)
    database.migrate()
    return database


def render_result(result: CommandResult) -> str:
    """Render a handler's result as plain text.

    Deliberately not a rich table: the output is asserted on in tests and
    piped into other tools, so column alignment by spaces beats box
    drawing that reflows with the terminal width.
    """
    lines: list[str] = []
    if result.message:
        lines.append(result.message)
    if result.columns:
        widths = [len(column) for column in result.columns]
        for row in result.rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(cells: tuple[str, ...]) -> str:
            return "  ".join(
                cell.ljust(widths[index]) for index, cell in enumerate(cells)
            ).rstrip()

        lines.append(line(result.columns))
        lines.append(line(tuple("-" * width for width in widths)))
        lines.extend(line(row) for row in result.rows)
    return "\n".join(lines)


def _annotation_for(param: Param) -> Any:
    """The type Typer should parse this parameter as."""
    if param.default is None:
        # `default=None` means "may be left out entirely".
        return param.type | None
    return param.type


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _choice_callback(param: Param) -> Callable[[Any], Any]:
    """Reject a value outside `choices` before the handler ever runs."""
    choices = param.choices or ()

    def check(value: Any) -> Any:
        if value is None or value in choices:
            return value
        raise typer.BadParameter(f"{param.name} must be one of: {', '.join(choices)}")

    return check


def _parameter_info(param: Param) -> Any:
    """The `typer.Argument` / `typer.Option` this parameter becomes."""
    help_text = param.help
    if param.choices:
        help_text = f"{help_text} One of: {', '.join(param.choices)}."
    callback = _choice_callback(param) if param.choices else None
    default = ... if param.default is REQUIRED else param.default

    if param.positional:
        return typer.Argument(default, help=help_text, callback=callback)

    if param.type is bool:
        declarations = [f"{_flag(param.name)}/--no-{param.name.replace('_', '-')}"]
    else:
        declarations = [_flag(param.name)]
    # Contracts §5 gives `--speaker` the alias `--global-speaker`; the
    # spellings live in the registry so both surfaces read the same list.
    declarations.extend(PARAM_ALIASES.get(param.name, ()))
    if param.short:
        declarations.append(param.short)
    return typer.Option(default, *declarations, help=help_text, callback=callback)


def _make_callback(cmd: Command) -> Callable[..., None]:
    """Synthesise the function Typer will introspect for this command.

    Typer reads both ``inspect.signature`` and ``__annotations__``, so
    both are set. The parameters are KEYWORD_ONLY; Typer decides argument
    versus option from the default object, not from the parameter kind.
    """
    annotations = {param.name: _annotation_for(param) for param in cmd.params}
    signature = inspect.Signature(
        [
            inspect.Parameter(
                param.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=_parameter_info(param),
                annotation=annotations[param.name],
            )
            for param in cmd.params
        ]
    )

    def run(**kwargs: Any) -> None:
        database = open_database()
        try:
            result = cmd.handler(database, **kwargs)
        except RytpError as exc:
            # Contracts §8: one line, exit 1, never a traceback.
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        finally:
            database.close()
        text = render_result(result)
        if text:
            typer.echo(text)

    run.__name__ = cmd.name.replace(".", "_").replace("-", "_")
    run.__doc__ = cmd.summary
    run.__signature__ = signature  # type: ignore[attr-defined]
    run.__annotations__ = {**annotations, "return": None}
    return run


def _root_callback(
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Data tree root for this run; overrides RYTP_DATA."),
    ] = None,
) -> None:
    """rytp — find where words were said, then cut them into a new video."""
    if data is not None:
        os.environ[C.DATA_ROOT_ENV_VAR] = str(data)


def build_app(registry: Mapping[str, Command] | None = None) -> typer.Typer:
    """Build the Typer app for ``registry`` (the global one by default)."""
    commands = COMMANDS if registry is None else registry
    app = typer.Typer(
        name="rytp",
        help="Index a video archive by speaker and word, then cut new video from it.",
        no_args_is_help=True,
        add_completion=False,
    )
    app.callback()(_root_callback)

    groups: dict[str, typer.Typer] = {}
    for cmd in sorted(commands.values(), key=lambda c: c.name):
        callback = _make_callback(cmd)
        if not cmd.group:
            app.command(leaf_name(cmd), help=cmd.summary)(callback)
            continue
        group = groups.get(cmd.group)
        if group is None:
            group = typer.Typer(
                no_args_is_help=True, help=f"Commands in the {cmd.group} group."
            )
            groups[cmd.group] = group
            app.add_typer(group, name=cmd.group)
        group.command(leaf_name(cmd), help=cmd.summary)(callback)

    return app


def command_paths(app: typer.Typer) -> set[str]:
    """Every command the app exposes, as dotted names. Used by the parity test."""
    import typer.main

    def walk(group: Any, prefix: tuple[str, ...] = ()) -> set[str]:
        found: set[str] = set()
        for name, child in getattr(group, "commands", {}).items():
            if getattr(child, "commands", None):
                found |= walk(child, (*prefix, name))
            else:
                found.add(".".join((*prefix, name)))
        return found

    return walk(typer.main.get_command(app))


def main() -> None:
    """Console-script and ``python -m rytp`` entry point."""
    _configure_stdio()
    if len(sys.argv) == 1:
        # Click exits 2 when a group is invoked with no arguments, even
        # with no_args_is_help. Printing help is not an error.
        sys.argv.append("--help")
    build_app()()
