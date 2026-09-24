"""The registry, rendered for the interactive surface (design §10).

Deliberately free of Textual imports: everything here is ordinary data
and string handling, so it can be tested without a terminal and reused
by whatever the TUI grows into. ``cli_only`` commands are filtered out
here, in one place, rather than in the Textual layer.

Arguments are typed as ``name=value``, with leading bare words filling
the positional parameters in order. That is enough for the catalog
commands and keeps the surface honest — there is exactly one definition
of what a command takes, and the TUI reads it rather than restating it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rytp.commands import COMMANDS, PARAM_ALIASES, REQUIRED, Command, Param
from rytp.models import InvalidInputError

__all__ = [
    "PaletteEntry",
    "cli_invocation",
    "match_entries",
    "palette_entries",
    "parse_arguments",
    "usage_line",
]

_TRUE = frozenset({"1", "true", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "no", "n", "off"})
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class PaletteEntry:
    """One row of the command palette."""

    name: str
    group: str
    summary: str
    usage: str
    long_running: bool


def usage_line(cmd: Command) -> str:
    """A one-line reminder of what this command takes."""
    parts = [cmd.name.replace(".", " ")]
    for param in cmd.params:
        if param.positional:
            parts.append(
                f"<{param.name}>" if param.default is REQUIRED else f"[{param.name}]"
            )
        elif param.default is REQUIRED:
            parts.append(f"{param.name}=…")
        else:
            parts.append(f"[{param.name}=…]")
    return " ".join(parts)


def palette_entries(commands: Mapping[str, Command] | None = None) -> list[PaletteEntry]:
    """Every registered command the TUI can show, sorted by name.

    ``cli_only`` commands are left out: they launch a surface, and the
    surface they launch is this one (contracts §5).
    """
    source = COMMANDS if commands is None else commands
    return [
        PaletteEntry(
            name=cmd.name,
            group=cmd.group,
            summary=cmd.summary,
            usage=usage_line(cmd),
            long_running=cmd.long_running,
        )
        for cmd in sorted(source.values(), key=lambda c: c.name)
        if not cmd.cli_only
    ]


def match_entries(entries: Sequence[PaletteEntry], query: str) -> list[PaletteEntry]:
    """Filter the palette by a case-insensitive substring of name or summary."""
    needle = query.strip().lower()
    if not needle:
        return list(entries)
    return [
        entry
        for entry in entries
        if needle in entry.name.lower() or needle in entry.summary.lower()
    ]


def _convert(param: Param, raw: str) -> Any:
    if param.type is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise InvalidInputError(f"{param.name}: expected true or false, got {raw!r}")
    if param.type is Path:
        return Path(raw)
    try:
        return param.type(raw)
    except ValueError as exc:
        raise InvalidInputError(
            f"{param.name}: expected {param.type.__name__}, got {raw!r}"
        ) from exc


def parse_arguments(cmd: Command, text: str) -> dict[str, Any]:
    """Turn a typed argument line into the handler's keyword arguments.

    Bare words fill positional parameters in order; ``name=value`` sets
    a named one. Missing optional parameters take their declared default.
    """
    by_name = {param.name: param for param in cmd.params}
    # `global_speaker=…` is `speaker=…` (contracts §5, PARAM_ALIASES).
    for canonical, spellings in PARAM_ALIASES.items():
        target = by_name.get(canonical)
        if target is None:
            continue
        for spelling in spellings:
            by_name.setdefault(spelling.lstrip("-").replace("-", "_"), target)
    positional = [param for param in cmd.params if param.positional]
    values: dict[str, Any] = {}
    next_positional = 0

    for token in shlex.split(text):
        name, sep, raw = token.partition("=")
        # A token counts as `name=value` only when the left side looks
        # like a parameter name. URLs contain '=' too, and a query string
        # must not be mistaken for a keyword argument.
        if sep and _NAME_RE.fullmatch(name):
            named = by_name.get(name)
            if named is None:
                raise InvalidInputError(
                    f"{cmd.name} has no parameter {name!r}: {usage_line(cmd)}"
                )
            param = named
        else:
            if next_positional >= len(positional):
                raise InvalidInputError(
                    f"too many values for {cmd.name}: {usage_line(cmd)}"
                )
            param = positional[next_positional]
            next_positional += 1
            raw = token
        if param.choices is not None and raw not in param.choices:
            raise InvalidInputError(
                f"{param.name} must be one of: {', '.join(param.choices)}"
            )
        values[param.name] = _convert(param, raw)

    for param in cmd.params:
        if param.name in values:
            continue
        if param.default is REQUIRED:
            raise InvalidInputError(f"{cmd.name} needs {param.name}: {usage_line(cmd)}")
        values[param.name] = param.default
    return values


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def cli_invocation(cmd: Command, values: Mapping[str, Any]) -> str:
    """The equivalent command line, for a long-running command's hint."""
    parts = ["rytp", *cmd.name.split(".")]
    for param in cmd.params:
        value = values.get(param.name, param.default)
        if value is None or value is REQUIRED:
            continue
        if param.positional:
            parts.append(_quote(str(value)))
            continue
        if param.type is bool:
            if value:
                parts.append("--" + param.name.replace("_", "-"))
            continue
        if value == param.default:
            continue
        parts.append("--" + param.name.replace("_", "-"))
        parts.append(_quote(str(value)))
    return " ".join(parts)
