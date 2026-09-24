"""The render job kind, contributed to Part 2's registry (contracts §5)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rytp.config import paths
from rytp.db import Database
from rytp.jobs import JOB_HANDLERS, Readiness, resolve_job_kind
from rytp.render.readiness import render_readiness

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_render_kind_is_registered_on_the_cpu_pool() -> None:
    kind = resolve_job_kind("render")
    assert kind.pool == "cpu"
    assert kind.target_kind == "render"
    assert kind.summary


def test_the_handler_is_in_the_contracted_flat_mapping() -> None:
    assert "render" in JOB_HANDLERS
    assert callable(JOB_HANDLERS["render"])


def _render_row(db: Database, name: str = "demo") -> int:
    cursor = db.conn.execute(
        "INSERT INTO renders (cutlist_name, canvas_mode, state, created_at) "
        "VALUES (?, '16:9', 'planned', '2026-01-01T00:00:00+00:00')",
        (name,),
    )
    db.conn.commit()
    return int(cursor.lastrowid)


def test_an_unknown_render_is_blocked(db: Database) -> None:
    assert render_readiness(db, 999_999) is Readiness.BLOCKED


def test_a_planned_render_with_its_cut_list_on_disk_is_ready(
    db: Database, data_dir: None
) -> None:
    render_id = _render_row(db)
    cutlist = paths().cutlist("demo")
    cutlist.parent.mkdir(parents=True, exist_ok=True)
    cutlist.write_text("schema_version = 1\n", encoding="utf-8")
    assert render_readiness(db, render_id) is Readiness.READY


def test_a_render_whose_cut_list_is_gone_is_blocked(
    db: Database, data_dir: None
) -> None:
    assert render_readiness(db, _render_row(db)) is Readiness.BLOCKED


def test_a_finished_render_is_satisfied_until_its_output_disappears(
    db: Database, tmp_path: Path, data_dir: None
) -> None:
    render_id = _render_row(db)
    cutlist = paths().cutlist("demo")
    cutlist.parent.mkdir(parents=True, exist_ok=True)
    cutlist.write_text("schema_version = 1\n", encoding="utf-8")
    output = tmp_path / "output.mp4"
    output.write_bytes(b"\x00")
    db.conn.execute(
        "UPDATE renders SET state = 'rendered', output_path = ? WHERE id = ?",
        (str(output), render_id),
    )
    db.conn.commit()
    assert render_readiness(db, render_id) is Readiness.SATISFIED
    output.unlink()
    # Self-healing, design §5: the world changed, so the job is work again.
    assert render_readiness(db, render_id) is Readiness.READY


def test_the_handler_delegates_to_the_stage(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, dict]] = []
    import rytp.render.run as run_module

    monkeypatch.setattr(
        run_module,
        "run_render_job",
        lambda _db, target_id, payload: seen.append((target_id, payload)),
    )
    result = JOB_HANDLERS["render"](db, 7, {"cutlist": "demo"})
    assert result is None  # contracts §5: handlers return None
    assert seen == [(7, {"cutlist": "demo"})]


def test_the_predicate_module_imports_cold(tmp_path: Path) -> None:
    """`rytp.render` imports nothing, so this module must not either."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import rytp.render.readiness as m; print(m.render_readiness.__name__)"],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "render_readiness" in proc.stdout


def test_importing_the_job_registry_does_not_import_the_render_stage() -> None:
    """A lazy thunk, so `rytp <anything>` does not pull in the render stage."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rytp.jobs; "
            "assert 'rytp.render.run' not in sys.modules, sorted(sys.modules); "
            "print('clean')",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
