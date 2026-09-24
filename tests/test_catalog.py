"""Catalog commands, exercised both as handlers and through the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import InvalidInputError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def test_channel_add_registers_and_reports_the_id(db: Database) -> None:
    result = catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    assert result.rows[0][1] == CHANNEL_URL
    assert result.rows[0][2] == "Channel One"
    assert "added" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1


def test_channel_add_twice_updates_rather_than_duplicating(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    result = catalog.channel_add(db, url=CHANNEL_URL, title="Renamed")
    assert "updated" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1
    assert db.conn.execute("SELECT title FROM channels").fetchone()[0] == "Renamed"


def test_channel_add_falls_back_to_the_url_as_a_title(db: Database) -> None:
    """Probing a channel for its real title is acquisition, and belongs to part 2."""
    result = catalog.channel_add(db, url=CHANNEL_URL)
    assert result.rows[0][2] == CHANNEL_URL


def test_channel_add_rejects_an_empty_url(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="URL"):
        catalog.channel_add(db, url="   ")


def test_channel_list_is_empty_on_a_fresh_database(db: Database) -> None:
    result = catalog.channel_list(db)
    assert result.rows == ()
    assert result.message == "0 channels"


def test_channel_list_shows_the_video_count_and_sync_time(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    result = catalog.channel_list(db)
    assert result.columns == ("id", "title", "videos", "last synced", "url")
    assert result.rows[0][1] == "Channel One"
    assert result.rows[0][2] == "0"
    assert result.rows[0][3] == "-"
    assert result.message == "1 channel"


def test_channel_list_truncates_a_long_title(db: Database) -> None:
    long_title = "Д" * 200
    catalog.channel_add(db, url=CHANNEL_URL, title=long_title)
    cell = catalog.channel_list(db).rows[0][1]
    assert len(cell) <= 60
    assert cell.endswith("…")


def test_both_channel_commands_are_registered() -> None:
    assert "channel.add" in commands.COMMANDS
    assert "channel.list" in commands.COMMANDS
    assert commands.COMMANDS["channel.add"].handler is catalog.channel_add


def test_the_cli_runs_them_end_to_end(data_dir: Path) -> None:
    app = build_app()
    added = runner.invoke(app, ["channel", "add", CHANNEL_URL, "--title", "Channel One"])
    assert added.exit_code == 0, added.output
    listed = runner.invoke(app, ["channel", "list"], env={"COLUMNS": "200"})
    assert listed.exit_code == 0, listed.output
    assert "Channel One" in listed.stdout
    assert CHANNEL_URL in listed.stdout


def test_the_cli_creates_the_database_on_the_first_command(data_dir: Path) -> None:
    assert not (data_dir / "rytp.db").exists()
    assert runner.invoke(build_app(), ["channel", "list"]).exit_code == 0
    assert (data_dir / "rytp.db").exists()
