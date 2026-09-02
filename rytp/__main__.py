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

from rytp.cli import app


def main() -> None:
    """Entry point for ``python -m rytp``."""
    app()


if __name__ == "__main__":
    main()