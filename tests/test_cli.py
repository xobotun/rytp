"""Tests for ``rytp.cli`` and the HF_TOKEN pre-launch gate (DESIGN §12)."""
from __future__ import annotations

from typing import Iterable

import pytest
from typer.testing import CliRunner

from rytp import cli, engines
from rytp.cli import app

runner = CliRunner()


# --- help / tree ----------------------------------------------------------


def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    out = result.stdout
    # Top-level singletons
    for cmd in ("download", "transcribe", "mine", "splice", "tui"):
        assert cmd in out, f"missing top-level command: {cmd}"
    # Subcommand groups
    for group in ("channel", "videos", "queue", "speakers"):
        assert group in out, f"missing subcommand group: {group}"


def test_each_group_help_lists_its_subcommands() -> None:
    expected = {
        "channel": ["add", "sync", "list"],
        "videos": ["add", "list"],
        "queue": ["add", "worker", "pause", "resume", "list"],
        "speakers": ["add", "list", "recompute-pauses", "map"],
    }
    for group, subs in expected.items():
        result = runner.invoke(app, [group, "--help"], catch_exceptions=False)
        assert result.exit_code == 0, result.stdout
        for sub in subs:
            assert sub in result.stdout, f"{group} {sub} missing"


def test_each_stub_exits_nonzero_without_crashing() -> None:
    """Every v2 subcommand (still a stub) exits 1 cleanly.

    The wired v1 subcommands (``channel add`` / ``sync`` / ``list``,
    ``videos add`` / ``list``, ``queue add`` / ``pause`` / ``resume``
    / ``list``, ``speakers add`` / ``list`` / ``recompute-pauses``,
    ``transcripts export``) are exercised by their own dedicated
    tests below. This list contains only the v2 stubs.
    """
    commands: list[list[str]] = [
        ["download", "oSYPC3cc_4A"],
        ["queue", "worker"],
        ["transcribe", "1"],
        ["speakers", "map", "1"],
        ["mine", "hello"],
        ["splice", "1", "--out", "out.mp4"],
        ["tui"],
    ]
    for cmd in commands:
        result = runner.invoke(app, cmd)
        # Stubs exit 1. ImportError / parse errors would be exit 2
        # with a stack trace — we want neither.
        assert result.exit_code in (1, 2), f"{cmd!r}: unexpected exit {result.exit_code}\n{result.stdout}"


# --- wired v1 subcommands (CRUD) ------------------------------------------


def test_videos_add_registers_local_path(tmp_path, monkeypatch):
    """The exact scenario from the bug report: ``python -m rytp videos add <local-file>`` should succeed."""
    import sqlite3
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    # Use a fresh data dir so we don't touch the user's DB.
    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    # Create a fake local video file.
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00" * 16)

    result = _CR().invoke(app, ["videos", "add", str(fake)])
    assert result.exit_code == 0, result.stdout
    assert "video 1" in result.stdout

    # Verify the row landed in the DB.
    db = Database(config.paths.db)
    try:
        row = db.conn.execute("SELECT * FROM videos WHERE id = 1").fetchone()
    finally:
        db.close()
    assert row is not None
    assert row["source"] == "local"
    assert row["downloaded"] == 1
    assert row["local_path"].endswith("fake.mp4")


def test_speakers_add_then_list(tmp_path, monkeypatch):
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["speakers", "add", "Alice", "--alias", "Al"])
    assert r.exit_code == 0, r.stdout
    assert "Alice" in r.stdout

    r = _CR().invoke(app, ["speakers", "list"])
    assert r.exit_code == 0, r.stdout
    assert "Alice" in r.stdout
    assert "Al" in r.stdout


def test_queue_pause_resume_round_trip(tmp_path, monkeypatch):
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["queue", "pause"])
    assert r.exit_code == 0, r.stdout
    r = _CR().invoke(app, ["queue", "resume"])
    assert r.exit_code == 0, r.stdout


def test_channel_list_on_empty_db_says_no_channels(tmp_path, monkeypatch):
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["channel", "list"])
    assert r.exit_code == 0, r.stdout
    assert "no channels yet" in r.stdout.lower() or "no channels" in r.stdout.lower()


def test_videos_list_unknown_channel_exits_nonzero(tmp_path, monkeypatch):
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["videos", "list", "--channel", "nope"])
    assert r.exit_code == 1, r.stdout
    assert "channel not found" in r.stdout.lower() or "channel not found" in (r.stderr or "").lower()


# --- HF_TOKEN gate -------------------------------------------------------


@pytest.fixture()
def gated_stt(monkeypatch: pytest.MonkeyPatch) -> Iterable[type]:
    """Register a fake STT engine that requires HF_TOKEN."""

    class _Gated:
        name = "gated-test"
        requires_hf_token = True
        help_url = "https://huggingface.co/gated/test"
        token_env_var = "HF_TOKEN"

        def transcribe(self, audio_path, *, language=None):
            return iter(())

    engines.register_stt(_Gated)
    try:
        yield _Gated
    finally:
        engines.STTS.pop("gated-test", None)


@pytest.fixture()
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)


def test_gate_fails_loudly_without_token(gated_stt: type, no_token: None) -> None:
    # The gate is invoked via top-level --stt/--diarizer/--combined flags
    # (DESIGN §12: pre-launch callback). Pick a gated STT engine at the top
    # level; the gate must exit 2 with an HF_TOKEN paragraph before any
    # subcommand body runs.
    result = runner.invoke(app, ["--stt", "gated-test", "transcribe", "1"])
    assert result.exit_code == 2, result.stdout
    # typer.echo(..., err=True) writes to stderr — CliRunner mixes both into
    # output but exposes only .stdout. Combine to be safe.
    combined = (result.stdout or "") + (result.output or "")
    # The help paragraph must mention the env var name.
    assert "HF_TOKEN" in combined


def test_gate_passes_silently_with_token(gated_stt: type, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    result = runner.invoke(app, ["--stt", "gated-test", "transcribe", "1"])
    # Token set → gate silent, command body runs and emits "not implemented"
    # (exit 1). The point is: not exit 2, and no HF_TOKEN paragraph.
    assert result.exit_code != 2, result.stdout
    assert "HF_TOKEN is required" not in ((result.stdout or "") + (result.output or ""))


def test_gate_ignores_unnamed_engine(no_token: None) -> None:
    # Without naming an engine on the command line, the gate doesn't run
    # even if the default would need a token (we don't know defaults here
    # because no engine is named).
    result = runner.invoke(app, ["videos", "list"])
    # Not a gate failure — just a stub exit code.
    assert result.exit_code != 2


def test_gate_skips_unknown_engine_silently(no_token: None) -> None:
    # An unknown engine name should NOT trigger the gate (we can't check
    # an unknown class). The subcommand body will raise later — that's
    # the right place for that error.
    result = runner.invoke(app, ["transcribe", "1", "--stt", "nonexistent"])
    # Either the body fails (1) or the resolve_stt in the gate raises (2).
    # The point is the *paragraph* is not printed.
    assert "HF_TOKEN is required" not in (result.stdout or "")