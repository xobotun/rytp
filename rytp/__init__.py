"""rytp — a searchable, cuttable index over a personal video archive.

Design: `docs/superpowers/specs/2026-09-21-rytp-redesign-design.md`.
Binding interfaces: `docs/superpowers/specs/2026-09-21-rytp-contracts.md`.

This module deliberately imports nothing. Importing `rytp` must not read
the environment, touch the filesystem, or pull in an optional dependency.
"""

from __future__ import annotations

__version__ = "0.2.0"
