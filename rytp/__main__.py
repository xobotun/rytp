"""`python -m rytp` — the same app as the installed `rytp` console script.

Kept as its own module so a source checkout needs no install.
"""

from __future__ import annotations

from rytp.cli import main

if __name__ == "__main__":
    main()
