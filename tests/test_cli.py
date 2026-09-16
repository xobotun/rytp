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


def test_no_stubs_remain() -> None:
    """All subcommands are wired in v1.

    The actual behavior of each heavy-lift command is covered by
    its own test below. This test only confirms the v1 wiring is
    complete: every command group has a real body, not the
    ``_not_implemented`` stub. We check by inspecting the source of
    :mod:`rytp.cli` for the marker string.
    """
    import inspect

    src = inspect.getsource(cli)
    # There should be no calls to _not_implemented still in the file
    # (only the function definition itself).
    function_def_count = src.count("def _not_implemented(")
    call_count = src.count("_not_implemented(")
    # One occurrence is the def itself; the rest would be calls.
    assert call_count <= function_def_count, (
        f"unexpected _not_implemented calls left in cli.py: "
        f"{call_count - function_def_count}"
    )


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


def test_speakers_add_idempotent_warns_on_alias_drop(tmp_path, monkeypatch):
    """Re-running ``speakers add`` for an existing label must warn
    that --alias / --notes were ignored, instead of silently dropping
    the user's input.

    Regression test for ANALYSIS-2.md issue A: the previous behavior
    was a silent no-op.
    """
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    cr = _CR()
    r1 = cr.invoke(app, ["speakers", "add", "Liam", "--alias", "L1"])
    assert r1.exit_code == 0, r1.stdout

    # Re-add the same label with a fresh alias. The CLI must
    # explicitly tell the user the alias was dropped, otherwise the
    # "idempotent" promise is misleading.
    r2 = cr.invoke(app, ["speakers", "add", "Liam", "--alias", "L2"])
    assert r2.exit_code == 0, r2.stdout
    combined = (r2.stdout or "") + (r2.output or "")
    assert "already exists" in combined.lower(), (
        f"expected a 'already exists' warning; got: {combined!r}"
    )
    assert "--alias" in combined
    assert "L2" not in combined or "not applied" in combined

    # The alias really is not in the DB.
    db = Database(config.paths.db)
    try:
        row = db.conn.execute(
            "SELECT aliases_json FROM speakers WHERE label = ?", ("Liam",)
        ).fetchone()
    finally:
        db.close()
    import json as _json
    assert _json.loads(row["aliases_json"]) == ["L1"]


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


def test_videos_list_invalid_source_errors(tmp_path, monkeypatch) -> None:
    """``videos list --source <unknown>`` should fail fast with the list
    of allowed sources -- not silently return an empty result.

    Regression test for ANALYSIS-2.md observation C-2.
    """
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["videos", "list", "--source", "bogus"])
    assert r.exit_code == 2, r.stdout  # Typer BadParameter exits 2.
    combined = (r.stdout or "") + (r.output or "")
    assert "bogus" in combined
    assert "youtube" in combined  # the allowed list


def test_videos_list_invalid_kind_errors(tmp_path, monkeypatch) -> None:
    """Same as ``--source`` above, but for ``--kind``."""
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["videos", "list", "--kind", "bogus"])
    assert r.exit_code == 2, r.stdout
    combined = (r.stdout or "") + (r.output or "")
    assert "bogus" in combined
    assert "video" in combined  # the allowed list mentions "video"


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
    # Token set → gate silent, command body runs. v1's body fails
    # with a "no media" or "video not found" error (exit 1) because
    # no real video row exists. The point is: not exit 2, and no
    # HF_TOKEN paragraph.
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


# --- heavy-lift subcommands (v1 wired) ------------------------------------


def test_download_unknown_id_exits_nonzero(tmp_path, monkeypatch) -> None:
    """`rytp download 9999` should exit 1 with a clear message."""
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["download", "9999"])
    assert r.exit_code == 1, r.stdout
    combined = (r.stdout or "") + (r.output or "")
    assert "not found" in combined.lower() or "9999" in combined


def test_transcribe_unknown_id_exits_nonzero(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["transcribe", "9999"])
    assert r.exit_code == 1, r.stdout


def test_mine_no_matches_says_so(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["mine", "no-such-string-xyz"])
    assert r.exit_code == 0, r.stdout
    assert "no matches" in r.stdout.lower() or "0" in r.stdout


def test_speakers_map_lists_empty(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    # Insert a video so the video_id lookup doesn't bail early.
    db.conn.execute(
        "INSERT INTO videos (source, kind, title, duration) "
        "VALUES ('local', 'video', 'v', 1000)"
    )
    db.conn.commit()
    db.close()

    r = _CR().invoke(app, ["speakers", "map", "1"])
    assert r.exit_code == 0, r.stdout
    assert "no diarizer labels" in r.stdout.lower() or "video 1 has no" in r.stdout.lower()


def test_speakers_map_unknown_video_errors(tmp_path, monkeypatch) -> None:
    """``speakers map <unknown-id>`` must error out, not silently dump
    the global roster. Regression test for ANALYSIS-2.md issue D: the
    previous implementation ignored the video_id and printed the
    entire roster as if every label belonged to the requested video.
    """
    from typer.testing import CliRunner as _CR

    from rytp import config
    from rytp.db import Database

    monkeypatch.setattr(config, "paths", config.Paths.from_root(tmp_path / "data").ensure())
    db = Database(config.paths.db)
    db.migrate()
    db.close()

    r = _CR().invoke(app, ["speakers", "map", "999"])
    assert r.exit_code == 1, r.stdout
    combined = (r.stdout or "") + (r.output or "")
    assert "not found" in combined.lower()