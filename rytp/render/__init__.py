"""Render: cut list in, one uploadable file plus a report out. design §9.

Deliberately empty of imports. ``rytp/jobs/__init__.py`` imports
``rytp.render.readiness`` at module level, and importing a submodule
executes this file first — so anything imported here would be imported by
every CLI invocation, which contracts §1 forbids. The stage entry points
live in :mod:`rytp.render.run`.
"""

from __future__ import annotations
