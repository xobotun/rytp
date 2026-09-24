"""Every shipped module must still parse as Python 3.11.

`pyproject.toml` sets mypy's `python_version = "3.12"`, but only because the
installed numpy's bundled `.pyi` uses a PEP 695 `type` statement that mypy
refuses to parse under a 3.11 target — it crashes before checking any of
rytp's own code. That bump is a static-analysis workaround, not a change of
runtime floor: `requires-python` is still `>=3.11` and the pipeline runs on
the owner's Windows machine.

Raising mypy's target quietly removed the one thing that would have caught
3.12-only syntax reaching a 3.11 interpreter. This puts that guarantee back,
and checks it more directly than mypy did.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "rytp"

#: The floor `pyproject.toml` promises in `requires-python`.
MINIMUM_FEATURE_VERSION = (3, 11)


def _sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_there_is_something_to_check() -> None:
    """A glob that silently matched nothing would make this file vacuous."""
    assert len(_sources()) > 10


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_module_parses_as_python_311(path: Path) -> None:
    """3.12-only syntax here would import fine on 3.14 and fail on the target."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=MINIMUM_FEATURE_VERSION)
    except SyntaxError as exc:  # pragma: no cover - the failure message is the point
        pytest.fail(
            f"{path.relative_to(PACKAGE_ROOT.parent)} needs Python "
            f"{'.'.join(str(n) for n in MINIMUM_FEATURE_VERSION)} syntax, "
            f"but line {exc.lineno} is newer: {exc.msg}"
        )
