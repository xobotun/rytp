"""`doctor` and the registry parts 2, 3, 4 and 7 plug into."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import (
    HEALTH_CHECKS,
    HealthCheck,
    HealthCheckError,
    HealthResult,
    doctor,
    register_check,
    run_health_checks,
)
from rytp.db import Database, schema

runner = CliRunner()


@pytest.fixture()
def checks(monkeypatch: pytest.MonkeyPatch) -> dict[str, HealthCheck]:
    """An empty registry, so a test's checks do not leak into the next."""
    fresh: dict[str, HealthCheck] = {}
    monkeypatch.setattr(commands, "HEALTH_CHECKS", fresh)
    return fresh


def passing(name: str, detail: str = "found") -> HealthCheck:
    return HealthCheck(
        name=name, summary=f"Check {name}.", run=lambda db: HealthResult(True, detail)
    )


def failing(name: str, *, required: bool = True) -> HealthCheck:
    """A check that truthfully reports an absence. `required` decides the exit code."""
    return HealthCheck(
        name=name,
        summary=f"Check {name}.",
        run=lambda db: HealthResult(False, "not installed", remedy=f"install {name}"),
        required=required,
    )


def test_register_check_stores_and_returns_it(checks: dict[str, HealthCheck]) -> None:
    check = register_check(passing("ffmpeg"))
    assert checks["ffmpeg"] is check


def test_registering_the_same_name_twice_is_an_error(
    checks: dict[str, HealthCheck],
) -> None:
    register_check(passing("ffmpeg"))
    with pytest.raises(ValueError, match="already registered"):
        register_check(passing("ffmpeg"))


def test_checks_run_in_name_order(db: Database, checks: dict[str, HealthCheck]) -> None:
    for name in ("zulu", "alpha", "mike"):
        register_check(passing(name))
    assert [check.name for check, _ in run_health_checks(db)] == ["alpha", "mike", "zulu"]


def test_a_check_that_raises_becomes_a_finding_not_a_crash(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    def explode(database: Database) -> HealthResult:
        raise RuntimeError("the engine exploded")

    register_check(HealthCheck(name="engine", summary="Check.", run=explode))
    (check, result), = run_health_checks(db)
    assert check.name == "engine"
    assert result.ok is False
    assert "the engine exploded" in result.detail


def test_doctor_reports_every_check_when_all_pass(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    register_check(passing("ffmpeg", "ffmpeg 7.1"))
    register_check(passing("gpu", "RTX 3080"))
    result = doctor(db)
    assert result.columns == ("check", "status", "detail", "remedy")
    statuses = {row[0]: row[1] for row in result.rows}
    assert statuses == {"ffmpeg": "ok", "gpu": "ok"}
    assert result.message == "2 checks, all required ok"


def test_a_check_is_required_by_default(checks: dict[str, HealthCheck]) -> None:
    assert register_check(passing("ffmpeg")).required is True


def test_a_missing_optional_engine_is_reported_truthfully_and_is_not_fatal(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    """Contracts §5: ok tells the truth; `required=False` makes it non-fatal."""
    register_check(passing("ffmpeg"))
    register_check(failing("transcriber:gigaam", required=False))
    result = doctor(db)
    statuses = {row[0]: row[1] for row in result.rows}
    assert statuses == {"ffmpeg": "ok", "transcriber:gigaam": "advisory"}
    assert "1 advisory" in (result.message or "")


def test_an_advisory_failure_still_shows_its_remedy(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    """"Not installed" is only useful next to "install it with …"."""
    register_check(failing("yt-dlp", required=False))
    remedies = {row[0]: row[3] for row in doctor(db).rows}
    assert remedies["yt-dlp"] == "install yt-dlp"


def test_advisory_failures_alone_never_fail_the_command(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    for name in ("yt-dlp", "ffplay", "gpu"):
        register_check(failing(name, required=False))
    assert doctor(db).message == "3 checks, all required ok (3 advisory)"


def test_health_status_maps_the_three_cases(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    from rytp.commands import health_status

    required_check = failing("ffmpeg")
    advisory_check = failing("gpu", required=False)
    good = passing("schema")
    assert health_status(good, good.run(db)) == "ok"
    assert health_status(required_check, required_check.run(db)) == "FAILED"
    assert health_status(advisory_check, advisory_check.run(db)) == "advisory"


def test_doctor_raises_when_a_required_check_fails(
    db: Database, checks: dict[str, HealthCheck]
) -> None:
    register_check(passing("ffmpeg"))
    register_check(failing("ffprobe"))
    register_check(failing("gpu", required=False))
    with pytest.raises(HealthCheckError) as excinfo:
        doctor(db)
    report = str(excinfo.value)
    assert "1 of 3 required checks failed: ffprobe" in report
    assert "install ffprobe" in report
    # Passing and advisory checks survive into the report; nothing is lost.
    assert "ffmpeg" in report
    assert "advisory" in report


def test_the_interpreter_check_passes_on_a_supported_python(db: Database) -> None:
    result = HEALTH_CHECKS["python"].run(db)
    assert result.ok is True
    assert str(sys.version_info.major) in result.detail


def test_the_data_tree_check_passes_on_a_writable_tree(
    db: Database, data_dir: Path
) -> None:
    result = HEALTH_CHECKS["data-tree"].run(db)
    assert result.ok is True
    assert str(data_dir) in result.detail


def test_the_data_tree_check_fails_when_the_tree_is_absent(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RYTP_DATA", str(tmp_path / "nowhere"))
    result = HEALTH_CHECKS["data-tree"].run(db)
    assert result.ok is False
    assert result.remedy is not None
    assert "rytp" in result.remedy


def test_the_data_tree_check_leaves_no_probe_file_behind(
    db: Database, data_dir: Path
) -> None:
    HEALTH_CHECKS["data-tree"].run(db)
    assert [item.name for item in data_dir.iterdir() if item.name.startswith(".")] == []


def test_the_schema_check_passes_on_a_migrated_database(db: Database) -> None:
    result = HEALTH_CHECKS["schema"].run(db)
    assert result.ok is True
    assert str(schema.LATEST_VERSION) in result.detail


def test_the_schema_check_fails_on_a_database_left_behind(
    data_dir: Path,
) -> None:
    from rytp.config import paths

    database = Database(paths().db)
    try:
        database.migrate_to(1)
        result = HEALTH_CHECKS["schema"].run(database)
        assert result.ok is False
        assert result.remedy is not None
    finally:
        database.close()


def test_part_one_registers_exactly_its_three_checks() -> None:
    assert set(HEALTH_CHECKS) >= {"python", "data-tree", "schema"}


def test_every_part_one_check_is_required() -> None:
    """The interpreter, the data tree and the schema are not optional."""
    for name in ("python", "data-tree", "schema"):
        assert HEALTH_CHECKS[name].required is True


def test_every_registered_check_has_a_summary() -> None:
    for name, check in HEALTH_CHECKS.items():
        assert check.summary.strip(), name


def test_doctor_is_a_registered_top_level_command() -> None:
    assert commands.COMMANDS["doctor"].group == ""
    assert commands.COMMANDS["doctor"].cli_only is False


def test_the_cli_runs_doctor_and_exits_zero_on_a_healthy_install(
    data_dir: Path,
) -> None:
    result = runner.invoke(build_app(), ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "python" in result.stdout
    assert "schema" in result.stdout


def test_the_cli_exits_one_when_a_required_check_fails(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = dict(HEALTH_CHECKS)
    broken["ffmpeg"] = failing("ffmpeg")
    monkeypatch.setattr(commands, "HEALTH_CHECKS", broken)
    result = runner.invoke(build_app(), ["doctor"])
    assert result.exit_code == 1
    assert "install ffmpeg" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_cli_stays_at_zero_when_only_advisory_checks_fail(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh machine with no yt-dlp is healthy, and says so out loud."""
    advisory = dict(HEALTH_CHECKS)
    advisory["yt-dlp"] = failing("yt-dlp", required=False)
    monkeypatch.setattr(commands, "HEALTH_CHECKS", advisory)
    result = runner.invoke(build_app(), ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "advisory" in result.stdout
    assert "install yt-dlp" in result.stdout
