"""videos add / videos list. yt-dlp is never called: the seam is faked."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rytp import constants as C
from rytp.cli import build_app
from rytp.commands import catalog
from rytp.db import Database
from rytp.db.queries import insert_asset
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


def test_register_only_creates_no_assets_and_no_jobs(
    db: Database, fake_probe: list[str]
) -> None:
    """`--register-only` is the old default: catalog, and nothing else."""
    catalog.videos_add(db, target=VIDEO_URL, register_only=True)
    assert db.conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_adding_a_url_enqueues_the_ingest_chain_by_default(
    db: Database, fake_probe: list[str]
) -> None:
    """BUGS.md entry 1: the owner's first command failed because
    registering and acquiring were separate steps and nothing said so.
    The default now enqueues the same chain `rytp ingest <id>` would."""
    result = catalog.videos_add(db, target=VIDEO_URL)
    video_id = int(result.rows[0][0])
    kinds = {
        row[0]
        for row in db.conn.execute(
            "SELECT kind FROM jobs WHERE target_id = ?", (video_id,)
        ).fetchall()
    }
    assert kinds == set(C.INGEST_CHAIN_REMOTE)
    # Readiness, not a dependency graph (contracts §5): the first hop
    # (`download`) is `pending`, and everything downstream of it is
    # `blocked` until it runs — neither state is `done`, because nothing
    # has run yet.
    assert not any(state == "done" for (state,) in db.conn.execute(
        "SELECT state FROM jobs WHERE target_id = ?", (video_id,)
    ).fetchall())
    # "Queued" must not read as "done" — no worker has run in this test.
    assert "queued" in (result.message or "")
    assert "downloaded" not in (result.message or "")


def test_adding_a_local_file_enqueues_extract_wav_not_a_download(
    db: Database, tmp_path: Path
) -> None:
    """A local file is coherent with the default: `ingest()` registers the
    container synchronously (no ffprobe, no network — just the row) and
    enqueues `extract_wav`/`fingerprint`; the WAV itself is only cached
    once a worker actually runs `extract_wav`."""
    path = make_file(tmp_path)
    result = catalog.videos_add(db, target=str(path))
    video_id = int(result.rows[0][0])
    kinds = {
        row[0]
        for row in db.conn.execute(
            "SELECT kind FROM jobs WHERE target_id = ?", (video_id,)
        ).fetchall()
    }
    assert kinds == set(C.INGEST_CHAIN_LOCAL)
    assert "download" not in kinds
    container = db.conn.execute(
        "SELECT role, path FROM assets WHERE video_id = ?", (video_id,)
    ).fetchone()
    assert container["role"] == "container"
    assert container["path"] == str(path)


def test_adding_the_same_video_twice_does_not_duplicate_jobs(
    db: Database, fake_probe: list[str]
) -> None:
    """`enqueue` is idempotent by `UNIQUE (kind, target_id)`; re-adding the
    same video (e.g. to correct its title) must not pile up a second set
    of jobs alongside the first."""
    catalog.videos_add(db, target=VIDEO_URL)
    result = catalog.videos_add(db, target=VIDEO_URL, title="Corrected")
    video_id = int(result.rows[0][0])
    counts = db.conn.execute(
        "SELECT kind, COUNT(*) FROM jobs WHERE target_id = ? GROUP BY kind",
        (video_id,),
    ).fetchall()
    assert all(count == 1 for _, count in counts)
    assert {kind for kind, _ in counts} == set(C.INGEST_CHAIN_REMOTE)


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


# -- `videos list --long` (BUGS.md entry 25) ---------------------------


def _add_local_video(db: Database, tmp_path: Path, name: str = "clip.mp4") -> int:
    result = catalog.videos_add(db, target=str(make_file(tmp_path, name)))
    return int(result.rows[0][0])


def _insert_word(
    db: Database,
    video_id: int,
    *,
    ord_: int,
    source: str,
    engine: str,
    end_ms: int | None = 1000,
    align_scale: str | None = None,
) -> None:
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, "
        "normalized_text, stem, source, engine, align_scale) "
        "VALUES (?, ?, 0, ?, 'a', 'a', 'a', ?, ?, ?)",
        (video_id, ord_, end_ms, source, engine, align_scale),
    )


@pytest.mark.parametrize(
    "count, expected",
    [(0, C.CELL_CROSS), (1, C.CELL_TICK), (2, "2"), (5, "5")],
)
def test_render_cell_ticks_crosses_and_counts(count: int, expected: str) -> None:
    """`0` a cross, `1` a tick, `2` and above the number: one rule for every
    column, asset counts and stage flags alike (entry 25, decided)."""
    assert catalog.render_cell(count) == expected


def test_videos_list_default_still_prints_exactly_what_it_prints_today(
    db: Database, tmp_path: Path
) -> None:
    """`--long` is additive; a script parsing the default listing must see
    no change in shape."""
    video_id = _add_local_video(db, tmp_path)
    insert_asset(db, video_id=video_id, role="audio", path=str(tmp_path / "a.wav"))
    _insert_word(db, video_id, ord_=0, source="timed", engine="whisper")
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


def test_videos_list_long_adds_asset_and_stage_columns(
    db: Database, tmp_path: Path
) -> None:
    video_id = _add_local_video(db, tmp_path)
    insert_asset(db, video_id=video_id, role="audio", path=str(tmp_path / "a.wav"))
    insert_asset(db, video_id=video_id, role="captions", path=str(tmp_path / "c.json3"))
    insert_asset(
        db, video_id=video_id, role="video", format_id="720p", path=str(tmp_path / "v1.mp4")
    )
    result = catalog.videos_list(db, long=True)
    assert result.columns == (
        "id",
        "source",
        "kind",
        "duration",
        "channel",
        "title",
        "external id",
        "audio",
        "captions",
        "videos",
        "transcribed",
        "aligned",
        "indexed",
        "diarized",
        "tier",
        "engine",
        "align scale",
    )
    row = result.rows[0]
    idx = {name: i for i, name in enumerate(result.columns)}
    assert row[idx["audio"]] == C.CELL_TICK
    assert row[idx["captions"]] == C.CELL_TICK
    assert row[idx["videos"]] == C.CELL_TICK  # exactly one rendition: a tick
    assert row[idx["transcribed"]] == C.CELL_CROSS
    assert row[idx["aligned"]] == C.CELL_CROSS
    assert row[idx["indexed"]] == C.CELL_CROSS
    assert row[idx["diarized"]] == C.CELL_CROSS
    assert row[idx["tier"]] == C.NULL_CELL
    assert row[idx["engine"]] == C.NULL_CELL
    assert row[idx["align scale"]] == C.NULL_CELL


def test_videos_list_long_two_renditions_is_a_count_not_a_tick(
    db: Database, tmp_path: Path
) -> None:
    """`videos` is the one asset role allowed more than one; `audio` never
    is (`C.SINGLETON_ASSET_ROLES`)."""
    video_id = _add_local_video(db, tmp_path)
    insert_asset(db, video_id=video_id, role="audio", path=str(tmp_path / "a.wav"))
    insert_asset(
        db, video_id=video_id, role="video", format_id="480p", path=str(tmp_path / "v1.mp4")
    )
    insert_asset(
        db, video_id=video_id, role="video", format_id="720p", path=str(tmp_path / "v2.mp4")
    )
    row = catalog.videos_list(db, long=True).rows[0]
    columns = catalog.videos_list(db, long=True).columns
    idx = {name: i for i, name in enumerate(columns)}
    assert row[idx["videos"]] == "2"
    assert row[idx["audio"]] == C.CELL_TICK


def test_videos_list_long_shows_the_best_tier_a_video_holds(
    db: Database, tmp_path: Path
) -> None:
    """A video can hold caption-tier and a later transcriber's tier at once
    (captions ingested, then a transcriber run); the column shows the best
    one present, because that answers "can I cut this yet"."""
    video_id = _add_local_video(db, tmp_path)
    _insert_word(db, video_id, ord_=0, source="caption", engine="ytdlp-captions")
    columns = catalog.videos_list(db, long=True).columns
    idx = {name: i for i, name in enumerate(columns)}
    row = catalog.videos_list(db, long=True).rows[0]
    assert row[idx["tier"]] == "caption"
    assert row[idx["aligned"]] == C.CELL_CROSS

    _insert_word(db, video_id, ord_=1, source="timed", engine="whisper")
    row = catalog.videos_list(db, long=True).rows[0]
    assert row[idx["tier"]] == "timed"
    assert row[idx["transcribed"]] == C.CELL_TICK
    assert row[idx["aligned"]] == C.CELL_CROSS

    _insert_word(
        db, video_id, ord_=2, source="aligned", engine="whisper+wav2vec2",
        align_scale="logprob",
    )
    row = catalog.videos_list(db, long=True).rows[0]
    assert row[idx["tier"]] == "aligned"
    assert row[idx["aligned"]] == C.CELL_TICK
    assert row[idx["engine"]] == "whisper, whisper+wav2vec2, ytdlp-captions"
    assert row[idx["align scale"]] == "logprob"


def test_videos_list_long_aligned_is_a_tier_projection_not_a_readiness_call(
    db: Database, tmp_path: Path
) -> None:
    """`align_readiness` is documented to never return `SATISFIED` — the
    `align` job kind is `reopenable=False` and re-aligning is always offered
    again — so the `aligned` cell is sourced from `words.source = 'aligned'`
    (the same fact the `tier` column reports), not from that predicate."""
    from rytp.jobs import Readiness
    from rytp.transcribe.readiness import align_readiness

    video_id = _add_local_video(db, tmp_path)
    _insert_word(
        db, video_id, ord_=0, source="aligned", engine="whisper+wav2vec2",
        align_scale="unknown",
    )
    assert align_readiness(db, video_id) != Readiness.SATISFIED
    row = catalog.videos_list(db, long=True).rows[0]
    columns = catalog.videos_list(db, long=True).columns
    idx = {name: i for i, name in enumerate(columns)}
    assert row[idx["aligned"]] == C.CELL_TICK


def test_videos_list_long_stage_columns_agree_with_the_readiness_predicates(
    db: Database, tmp_path: Path
) -> None:
    """Assert against the registered predicates themselves, not a duplicated
    rule — the whole point of entry 25 is one definition of "done"."""
    from rytp.index.utterances import index_video
    from rytp.jobs import JOB_KINDS, Readiness

    video_id = _add_local_video(db, tmp_path)
    _insert_word(db, video_id, ord_=0, source="timed", engine="whisper")
    index_video(db, video_id)
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (?, 'SPEAKER_00', 'fake')",
        (video_id,),
    )

    result = catalog.videos_list(db, long=True)
    idx = {name: i for i, name in enumerate(result.columns)}
    row = result.rows[0]

    for column, kind_name in (
        ("transcribed", "transcribe"),
        ("indexed", "index"),
        ("diarized", "diarize"),
    ):
        expected = (
            C.CELL_TICK
            if JOB_KINDS[kind_name].readiness(db, video_id) == Readiness.SATISFIED
            else C.CELL_CROSS
        )
        assert row[idx[column]] == expected, column


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
