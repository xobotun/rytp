"""The registry, rendered for the interactive surface (design §10).

Deliberately free of Textual imports: everything here is ordinary data
and string handling, so it can be tested without a terminal and reused
by whatever the TUI grows into. ``cli_only`` commands are filtered out
here, in one place, rather than in the Textual layer.

Arguments are typed as ``name=value``, with leading bare words filling
the positional parameters in order; ``--name``, ``--name=value`` and
``--no-name`` are accepted as synonyms of the named form, for a user
who spent the day in the CLI (entry 24). That is enough for the catalog
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


def _tokenize(cmd: Command, text: str) -> list[str]:
    """Split a typed argument line the way a shell would, minus escaping.

    Plain POSIX-mode ``shlex`` (its default ``escape``) would treat backslash as
    an escape character, silently mangling an unquoted Windows path such as
    ``D:\\Work\\file.mp4`` into ``D:Workfile.mp4`` (entry 21) — wrong rather
    than refused, and on the target platform's most natural input. Plain
    non-POSIX ``shlex`` (``posix=False``) avoids that, but its quoted state
    is entered only from whitespace, never mid-token, so it cannot handle
    ``name="value with spaces"`` — a form that worked before entry 21 was
    ever a problem. Emptying ``escape`` on an otherwise-POSIX lexer keeps
    both: POSIX mode enters quoted state mid-token (``name="a b"`` and
    ``"a b"`` both parse to one token, quotes stripped), and no character
    is treated as an escape, so backslashes survive untouched.
    """
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        # Unbalanced quote (`shlex` raises a bare `ValueError`) — funnel it
        # through the same one-line error channel as every other rejected
        # argument line, rather than letting it become a traceback (the
        # class of defect in BUGS.md entry 16).
        raise InvalidInputError(f"{cmd.name}: {exc}: {usage_line(cmd)}") from exc


def parse_arguments(cmd: Command, text: str) -> dict[str, Any]:
    """Turn a typed argument line into the handler's keyword arguments.

    Bare words fill positional parameters in order; ``name=value`` sets a
    named one. ``--name``, ``--name=value`` and ``--no-name`` are accepted
    as synonyms for the CLI user who types the flag spelling instead
    (entry 24) — ``--name`` on a boolean means ``true``, ``--no-name``
    means ``false``, and ``--name`` on any other parameter takes the next
    token as its value. Missing optional parameters take their declared
    default.

    See :func:`_tokenize` for how the line is split (entry 21).
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

    tokens = _tokenize(cmd, text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1

        if token.startswith("--") and len(token) > 2:
            body = token[2:]
            flag_name, flag_sep, flag_value = body.partition("=")
            normalized = flag_name.replace("-", "_")
            negate = False
            lookup = normalized
            if (
                not flag_sep
                and normalized.startswith("no_")
                and normalized not in by_name
                and normalized[3:] in by_name
                and by_name[normalized[3:]].type is bool
            ):
                negate = True
                lookup = normalized[3:]
            param = by_name.get(lookup)
            if param is None:
                raise InvalidInputError(
                    f"{cmd.name} has no parameter {flag_name!r}: {usage_line(cmd)}"
                )
            if negate:
                raw = "false"
            elif flag_sep:
                raw = flag_value
            elif param.type is bool:
                raw = "true"
            else:
                if index >= len(tokens):
                    raise InvalidInputError(
                        f"--{flag_name} needs a value: {usage_line(cmd)}"
                    )
                raw = tokens[index]
                index += 1
            if param.choices is not None and raw not in param.choices:
                raise InvalidInputError(
                    f"{param.name} must be one of: {', '.join(param.choices)}"
                )
            values[param.name] = _convert(param, raw)
            continue

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
