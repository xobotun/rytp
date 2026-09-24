"""videos add / videos list. yt-dlp is never called: the seam is faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError, RytpError

runner = CliRunner()

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"
VIDEO_URL = "https://example.invalid/watch?v=VIDEO_A"

PROBED = ChannelEntry(
    external_id="VIDEO_A",
    title="Утренний эфир",
    url=VIDEO_URL,
    duration_ms=3_600_000,
    kind="video",
    published_at="2026-01-01T00:00:00+00:00",
)


@pytest.fixture()
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the yt-dlp seam. Part 2 supplies the real implementation."""
    seen: list[str] = []

    def probe(url: str) -> ChannelEntry:
        seen.append(url)
        return PROBED

    monkeypatch.setattr(catalog, "_probe_video", probe)
    return seen


def make_file(tmp_path: Path, name: str = "interview.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def test_adding_a_url_catalogs_what_the_probe_reported(
    db: Database, fake_probe: list[str]
) -> None:
    result = catalog.videos_add(db, target=VIDEO_URL)
    assert fake_probe == [VIDEO_URL]
    row = db.conn.execute("SELECT * FROM videos").fetchone()
    assert row["source"] == "ytdlp"
    assert row["external_id"] == "VIDEO_A"
    assert row["url"] == VIDEO_URL
    assert row["title"] == "Утренний эфир"
    assert row["duration_ms"] == 3_600_000
    assert row["kind"] == "video"
    assert "added" in (result.message or "")


def test_adding_a_url_creates_no_assets_and_no_jobs(
    db: Database, fake_probe: list[str]
) -> None:
    """Design §5: cataloguing only. `rytp ingest` starts the chain."""
    catalog.videos_add(db, target=VIDEO_URL)
    assert db.conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_explicit_title_and_kind_override_the_probe(
    db: Database, fake_probe: list[str]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL, title="My name for it", kind="livestream")
    row = db.conn.execute("SELECT title, kind FROM videos").fetchone()
    assert row["title"] == "My name for it"
    assert row["kind"] == "livestream"


def test_adding_the_same_url_twice_updates_one_row(
    db: Database, fake_probe: list[str]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL)
    result = catalog.videos_add(db, target=VIDEO_URL, title="Corrected")
    assert "updated" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert db.conn.execute("SELECT title FROM videos").fetchone()[0] == "Corrected"


def test_adding_a_local_file_stores_its_absolute_path_as_the_external_id(
    db: Database, tmp_path: Path
) -> None:
    path = make_file(tmp_path)
    catalog.videos_add(db, target=str(path))
    row = db.conn.execute("SELECT * FROM videos").fetchone()
    assert row["source"] == "local"
    assert row["external_id"] == str(path.resolve())
    assert row["url"] is None
    assert row["title"] == "interview"


def test_adding_the_same_local_file_twice_is_idempotent(
    db: Database, tmp_path: Path
) -> None:
    path = make_file(tmp_path)
    catalog.videos_add(db, target=str(path))
    catalog.videos_add(db, target=str(path))
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_a_local_path_never_reaches_the_probe(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(url: str) -> ChannelEntry:  # pragma: no cover - must not run
        raise AssertionError("a local file must not be probed")

    monkeypatch.setattr(catalog, "_probe_video", explode)
    catalog.videos_add(db, target=str(make_file(tmp_path)))
    assert db.conn.execute("SELECT source FROM videos").fetchone()[0] == "local"


def test_a_missing_local_path_that_is_not_a_url_is_an_error(db: Database) -> None:
    with pytest.raises(RytpError):
        catalog.videos_add(db, target="definitely/not/here.mp4")


def test_the_probe_seam_explains_the_missing_extra(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint must stay actionable — and this must never reach the network.

    Forcing `None` into sys.modules makes the import fail whether or not
    part 2 has landed. Without it, this test would quietly start making
    real yt-dlp calls the day `rytp/acquire/ytdlp.py` appears.
    """
    import sys

    monkeypatch.setitem(sys.modules, "rytp.acquire", None)
    monkeypatch.setitem(sys.modules, "rytp.acquire.ytdlp", None)
    with pytest.raises(RytpError, match="yt-dlp"):
        catalog._probe_video(VIDEO_URL)


def test_an_unknown_kind_is_rejected(db: Database, fake_probe: list[str]) -> None:
    with pytest.raises(InvalidInputError, match="kind"):
        catalog.videos_add(db, target=VIDEO_URL, kind="opera")


def test_a_video_can_be_filed_under_a_channel_by_id_or_by_url(
    db: Database, fake_probe: list[str], tmp_path: Path
) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    assert db.conn.execute("SELECT channel_id FROM videos").fetchone()[0] == 1
    catalog.videos_add(db, target=str(make_file(tmp_path)), channel=CHANNEL_URL)
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM videos WHERE channel_id = 1"
        ).fetchone()[0]
        == 2
    )


def test_an_unknown_channel_is_an_error(db: Database, fake_probe: list[str]) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.videos_add(db, target=VIDEO_URL, channel="no such channel")


def test_videos_list_shows_the_catalog(db: Database, fake_probe: list[str]) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    result = catalog.videos_list(db)
    assert result.columns == (
        "id",
        "source",
        "kind",
        "duration",
        "channel",
        "title",
        "external id",
    )
    row = result.rows[0]
    assert row[1] == "ytdlp"
    assert row[3] == "1:00:00"
    assert row[4] == "Channel One"
    assert result.message == "1 video"


def test_videos_list_filters(db: Database, fake_probe: list[str], tmp_path: Path) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    catalog.videos_add(db, target=str(make_file(tmp_path)), title="Домашняя запись")

    def ids(**filters: object) -> list[str]:
        return [row[6] for row in catalog.videos_list(db, **filters).rows]  # type: ignore[arg-type]

    assert len(ids()) == 2
    assert ids(source="local") == [str((tmp_path / "interview.mp4").resolve())]
    assert ids(channel="Channel One") == ["VIDEO_A"]
    assert ids(kind="video") == [
        str((tmp_path / "interview.mp4").resolve()),
        "VIDEO_A",
    ]
    assert ids(search="Домашняя") == [str((tmp_path / "interview.mp4").resolve())]
    assert ids(search="ничего") == []


def test_videos_list_rejects_an_unknown_source(db: Database) -> None:
    with pytest.raises(InvalidInputError, match="source"):
        catalog.videos_list(db, source="vimeo")


def test_format_duration_handles_none_and_hours() -> None:
    assert catalog.format_duration(None) == "-"
    assert catalog.format_duration(0) == "0:00:00"
    assert catalog.format_duration(3_661_000) == "1:01:01"


def test_the_cli_adds_a_local_file_and_lists_it(
    data_dir: Path, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")
    app = build_app()
    added = runner.invoke(app, ["videos", "add", str(path)])
    assert added.exit_code == 0, added.output
    listed = runner.invoke(app, ["videos", "list"], env={"COLUMNS": "240"})
    assert listed.exit_code == 0, listed.output
    assert "clip" in listed.stdout
    assert "local" in listed.stdout


def test_the_cli_rejects_a_bad_kind_before_touching_the_database(
    data_dir: Path, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")
    result = runner.invoke(build_app(), ["videos", "add", str(path), "--kind", "opera"])
    assert result.exit_code == 2
    assert "must be one of" in result.stderr
