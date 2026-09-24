"""channel sync walks three listings. yt-dlp is never called."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import commands
from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError, RytpError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def entry(external_id: str, kind: str, title: str | None = None) -> ChannelEntry:
    return ChannelEntry(
        external_id=external_id,
        title=title or f"Title {external_id}",
        url=f"https://example.invalid/watch?v={external_id}",
        duration_ms=60_000,
        kind=kind,
        published_at=None,
    )


LISTINGS = {
    "videos": [entry("VIDEO_A", "video"), entry("VIDEO_B", "video")],
    "streams": [entry("VIDEO_B", "livestream"), entry("VIDEO_C", "livestream")],
    "shorts": [entry("VIDEO_D", "short")],
}


@pytest.fixture()
def fake_enumerate(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Replace the yt-dlp seam with the fixed listings above."""
    seen: list[tuple[str, tuple[str, ...]]] = []

    def enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
        seen.append((channel_url, tuple(tabs)))
        found: list[ChannelEntry] = []
        for tab in tabs:
            found.extend(LISTINGS[tab])
        return found

    monkeypatch.setattr(catalog, "_enumerate_channel", enumerate_channel)
    return seen


@pytest.fixture()
def channel(db: Database) -> int:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    return 1


def test_sync_catalogs_every_listing_not_just_the_main_one(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    result = catalog.channel_sync(db, channel="1")
    assert fake_enumerate == [(CHANNEL_URL, ("videos", "streams", "shorts"))]
    kinds = dict(
        db.conn.execute("SELECT kind, COUNT(*) FROM videos GROUP BY kind").fetchall()
    )
    assert kinds == {"video": 1, "livestream": 2, "short": 1}
    assert "4" in (result.message or "")


def test_a_video_listed_in_two_tabs_takes_the_later_tabs_kind(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    """A live stream appears under /videos too; /streams is the truth."""
    catalog.channel_sync(db, channel="1")
    row = db.conn.execute(
        "SELECT kind FROM videos WHERE external_id = 'VIDEO_B'"
    ).fetchone()
    assert row["kind"] == "livestream"
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 4


def test_sync_files_everything_under_the_channel(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    assert (
        db.conn.execute("SELECT COUNT(*) FROM videos WHERE channel_id = 1").fetchone()[0]
        == 4
    )


def test_sync_records_when_it_ran(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    stamp = db.conn.execute("SELECT last_synced_at FROM channels").fetchone()[0]
    assert stamp is not None
    assert stamp.endswith("+00:00")


def test_re_syncing_updates_rather_than_duplicating(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1")
    result = catalog.channel_sync(db, channel="1")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 4
    assert "0 new" in (result.message or "")


def test_the_result_breaks_the_count_down_by_kind(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    result = catalog.channel_sync(db, channel="1")
    assert result.columns == ("kind", "catalogued")
    assert dict((row[0], row[1]) for row in result.rows) == {
        "video": "1",
        "livestream": "2",
        "short": "1",
    }


def test_tabs_can_be_narrowed(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    catalog.channel_sync(db, channel="1", tabs="shorts")
    assert fake_enumerate == [(CHANNEL_URL, ("shorts",))]
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_tabs_are_always_walked_in_the_canonical_order(
    db: Database, channel: int, fake_enumerate: list[tuple[str, tuple[str, ...]]]
) -> None:
    """Typed in any order, they run videos → streams → shorts, so kinds settle the same way."""
    catalog.channel_sync(db, channel="1", tabs="shorts, streams ,videos")
    assert fake_enumerate == [(CHANNEL_URL, ("videos", "streams", "shorts"))]


def test_an_unknown_tab_is_rejected(db: Database, channel: int) -> None:
    with pytest.raises(InvalidInputError, match="tab"):
        catalog.channel_sync(db, channel="1", tabs="videos,playlists")


def test_parse_tabs_rejects_an_empty_selection() -> None:
    with pytest.raises(InvalidInputError, match="tab"):
        catalog.parse_tabs("  ,  ")


def test_syncing_an_unknown_channel_is_an_error(db: Database) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.channel_sync(db, channel="7")


def test_the_enumerate_seam_explains_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced absent, so this keeps testing the hint instead of calling part 2."""
    import sys

    monkeypatch.setitem(sys.modules, "rytp.acquire", None)
    monkeypatch.setitem(sys.modules, "rytp.acquire.ytdlp", None)
    with pytest.raises(RytpError, match="yt-dlp"):
        catalog._enumerate_channel(CHANNEL_URL, ("videos",))


def test_sync_is_registered_as_long_running() -> None:
    assert commands.COMMANDS["channel.sync"].long_running is True


def test_the_cli_runs_a_sync(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def enumerate_channel(channel_url: str, tabs: Sequence[str]) -> list[ChannelEntry]:
        return [entry("VIDEO_A", "video")]

    monkeypatch.setattr(catalog, "_enumerate_channel", enumerate_channel)
    app = build_app()
    assert runner.invoke(app, ["channel", "add", CHANNEL_URL]).exit_code == 0
    synced = runner.invoke(app, ["channel", "sync", "1"], env={"COLUMNS": "200"})
    assert synced.exit_code == 0, synced.output
    assert "1 new" in synced.stdout
