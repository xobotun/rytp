"""``python -m rytp`` entry point.

Lets you run the CLI without installing the package: from the repo
root (``D:\\Work\\rytp`` or wherever you cloned it), run
``python -m rytp <subcommand> ...`` and Typer takes over.

This file exists specifically so the project is runnable from a
source checkout — useful during development, in CI, or when you
just want to try the CLI without committing to a ``pip install``.
The installed ``rytp`` console script does the same thing once
the package is installed.
"""
from __future__ import annotations

import sys

import typer

from rytp.cli import app


def main() -> None:
    """Entry point for ``python -m rytp``.

    Two special cases are handled here rather than in :mod:`rytp.cli`:

    1. ``python -m rytp`` with no subcommand prints help and exits 0
       (Typer alone exits 2, which surprises CI scripts that just
       check ``$LASTEXITCODE``).
    2. ``sys.stdout``/``sys.stderr`` are reconfigured to UTF-8 so
       the ``§`` and ``—`` characters used throughout the docstrings
       survive the round-trip to whatever console is attached
       (Windows defaults to a legacy code page that would otherwise
       replace them with ``?`` or the U+FFFD replacement character).
    """
    # Force UTF-8 on stdio. No-op on streams that can't be reconfigured.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    if not _has_subcommand(sys.argv[1:]):
        # Bare invocation: synthesize an explicit ``--help`` call so
        # the user gets the same banner as ``python -m rytp --help``,
        # but force the exit code to 0. ``no_args_is_help`` triggers
        # Click's "missing command" error path which would otherwise
        # exit non-zero.
        sys.argv = [sys.argv[0], "--help"]
        try:
            app(standalone_mode=False)
        except typer.Exit:
            # ``--help`` raises typer.Exit(0) after printing; we
            # swallow that here and exit cleanly ourselves.
            return 0
        return 0

    app(standalone_mode=True)
    return 0


def _has_subcommand(argv: list[str]) -> bool:
    """True if ``argv`` contains a positional command after options.

    Typer/Click tokens that start with ``-`` are options or flags;
    everything else is treated as a subcommand. We deliberately
    ignore ``--help`` and ``-h`` here so the bare ``python -m rytp -h``
    still routes through the help path.
    """
    for tok in argv:
        if not tok.startswith("-"):
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main() or 0)