"""Removing a channel, and removing a video with its rows and its files."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp.commands import catalog
from rytp.db import Database
from rytp.models import ChannelEntry, InvalidInputError, NotFoundError

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"
VIDEO_URL = "https://example.invalid/watch?v=VIDEO_A"
NOW = "2026-01-01T00:00:00+00:00"

PROBED = ChannelEntry(
    external_id="VIDEO_A",
    title="Утренний эфир",
    url=VIDEO_URL,
    duration_ms=3_600_000,
    kind="video",
    published_at=None,
)


@pytest.fixture()
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalog, "_probe_video", lambda url: PROBED)


@pytest.fixture()
def job_kinds(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """Stand in for part 2's JOB_KINDS registry.

    `caption_words` is here on purpose: it is the kind that a hardcoded
    list of the obvious eight would have missed, and the reason removal
    derives this set instead of listing it.
    """
    kinds = ("caption_words", "download", "transcribe")
    monkeypatch.setattr(catalog, "_video_job_kinds", lambda: kinds)
    return kinds


@pytest.fixture()
def video(db: Database, data_dir: Path, fake_probe: None, job_kinds: tuple[str, ...]) -> int:
    """One catalogued video with rows, files and a queued job."""
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, normalized_text,"
        " stem, source, engine) VALUES (1, 0, 0, 100, 'да', 'да', 'да', 'aligned', 'mfa')"
    )
    db.conn.execute(
        "INSERT INTO utterances (video_id, start_ms, end_ms, first_word_ord,"
        " last_word_ord, text, normalized_text, stem_text)"
        " VALUES (1, 0, 100, 0, 0, 'да', 'да', 'да')"
    )
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (1, 'SPEAKER_00', 'pyannote')"
    )
    db.conn.execute(
        "INSERT INTO assets (video_id, role, path, bytes, acquired_at)"
        " VALUES (1, 'audio', 'media/1/audio.m4a', 10, ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO video_acoustics (video_id, computed_at) VALUES (1, ?)", (NOW,)
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('transcribe', 1, 'pending', 'gpu', ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('caption_words', 1, 'pending', 'cpu', ?)",
        (NOW,),
    )
    layout = config.paths()
    media = config.ensure_dir(layout.media_dir(1))
    (media / "audio.m4a").write_bytes(b"0123456789")
    config.ensure_dir(layout.cache_wav(1).parent)
    layout.cache_wav(1).write_bytes(b"01234")
    config.ensure_dir(layout.transcript(1).parent)
    layout.transcript(1).write_text("# transcript\n", encoding="utf-8")
    return 1


# -- channel.remove ---------------------------------------------------


def test_removing_a_channel_orphans_its_videos(
    db: Database, data_dir: Path, fake_probe: None
) -> None:
    """Contracts §5: videos.channel_id is nullable for exactly this."""
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.videos_add(db, target=VIDEO_URL, channel="1")
    result = catalog.channel_remove(db, channel="1")
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert db.conn.execute("SELECT channel_id FROM videos").fetchone()[0] is None
    assert "1 video orphaned" in (result.message or "")


def test_removing_a_channel_with_no_videos_says_so(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    assert "0 videos orphaned" in (catalog.channel_remove(db, channel="1").message or "")


def test_removing_a_channel_by_url_or_title(db: Database) -> None:
    catalog.channel_add(db, url=CHANNEL_URL, title="Channel One")
    catalog.channel_remove(db, channel=CHANNEL_URL)
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0


def test_removing_an_unknown_channel_is_an_error(db: Database) -> None:
    with pytest.raises(NotFoundError, match="channel"):
        catalog.channel_remove(db, channel="7")


# -- videos.remove ----------------------------------------------------


def test_a_dry_run_deletes_nothing_and_counts_everything(
    db: Database, video: int, data_dir: Path
) -> None:
    result = catalog.videos_remove(db, video="1", dry_run=True)
    counts = dict((row[0], row[1]) for row in result.rows)
    assert counts["words"] == "1"
    assert counts["utterances"] == "1"
    assert counts["video_speakers"] == "1"
    assert counts["assets"] == "1"
    assert counts["video_acoustics"] == "1"
    assert counts["jobs"] == "2"
    assert counts["files"] == "3"
    # 10 bytes of audio + 5 of wav + the transcript.
    assert "bytes" in (result.message or "")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert config.paths().cache_wav(1).exists()


def test_removal_refuses_without_yes(db: Database, video: int, data_dir: Path) -> None:
    """Contracts §5: anything that deletes a file requires --yes."""
    with pytest.raises(InvalidInputError, match="--yes"):
        catalog.videos_remove(db, video="1")
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_removal_takes_the_row_the_cascades_and_the_files(
    db: Database, video: int, data_dir: Path
) -> None:
    layout = config.paths()
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0
    for table in ("words", "utterances", "video_speakers", "assets", "video_acoustics"):
        assert db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert not layout.media_dir(1).exists()
    assert not layout.cache_wav(1).exists()
    assert not layout.transcript(1).exists()


def test_removal_cancels_every_kind_of_job_that_targeted_the_video(
    db: Database, video: int, data_dir: Path
) -> None:
    """`jobs` has no foreign key, so a worker would otherwise run against nothing.

    `caption_words` is part 3's, and is cancelled because the kinds come
    from the registry rather than from a list written here.
    """
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_the_video_job_kinds_come_from_part_twos_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contracts §5: `target_kind` is an attribute of the registration."""
    import sys
    import types

    registry = types.SimpleNamespace(
        JOB_KINDS={
            "download": types.SimpleNamespace(target_kind="video"),
            "caption_words": types.SimpleNamespace(target_kind="video"),
            "render": types.SimpleNamespace(target_kind="render"),
        }
    )
    monkeypatch.setitem(sys.modules, "rytp.jobs", registry)
    assert catalog._video_job_kinds() == ("caption_words", "download")


def test_with_no_job_registry_there_are_no_job_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Part 1 ships before rytp.jobs; nothing creates job rows either."""
    import sys

    monkeypatch.setitem(sys.modules, "rytp.jobs", None)
    assert catalog._video_job_kinds() == ()


def test_removal_leaves_another_videos_jobs_alone(
    db: Database, video: int, data_dir: Path, job_kinds: tuple[str, ...]
) -> None:
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('transcribe', 2, 'pending', 'gpu', ?)",
        (NOW,),
    )
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, created_at)"
        " VALUES ('render', 1, 'pending', 'cpu', ?)",
        (NOW,),
    )
    catalog.videos_remove(db, video="1", yes=True)
    remaining = {
        (row["kind"], row["target_id"])
        for row in db.conn.execute("SELECT kind, target_id FROM jobs")
    }
    # `render` targets a renders row, not a video, so its target_kind keeps
    # it out of the derived set and it is not ours to cancel.
    assert remaining == {("transcribe", 2), ("render", 1)}


def test_removal_leaves_the_channel_alone(
    db: Database, video: int, data_dir: Path
) -> None:
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 1


def test_removal_warns_about_cut_lists_naming_the_video(
    db: Database, video: int, data_dir: Path
) -> None:
    """Parts 5 and 6 refuse to render a dangling video_id; say so now, not then."""
    cutlists = config.ensure_dir(config.paths().root / "cutlists")
    (cutlists / "monologue.toml").write_text(
        '[[fragment]]\nvideo_id = 1\nstart_ms = 0\nend_ms = 100\n', encoding="utf-8"
    )
    (cutlists / "other.toml").write_text(
        '[[fragment]]\nvideo_id = 2\nstart_ms = 0\nend_ms = 100\n', encoding="utf-8"
    )
    result = catalog.videos_remove(db, video="1", dry_run=True)
    assert "monologue" in (result.message or "")
    assert "other" not in (result.message or "")


def test_an_unreadable_cut_list_is_skipped_not_fatal(
    db: Database, video: int, data_dir: Path
) -> None:
    cutlists = config.ensure_dir(config.paths().root / "cutlists")
    (cutlists / "broken.toml").write_text("this is not toml = = =", encoding="utf-8")
    assert catalog.videos_remove(db, video="1", dry_run=True).rows


def test_removal_works_when_the_files_were_never_downloaded(
    db: Database, data_dir: Path, fake_probe: None, job_kinds: tuple[str, ...]
) -> None:
    catalog.videos_add(db, target=VIDEO_URL)
    catalog.videos_remove(db, video="1", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_a_video_may_be_named_by_external_id(
    db: Database, video: int, data_dir: Path
) -> None:
    catalog.videos_remove(db, video="VIDEO_A", yes=True)
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


def test_removing_an_unknown_video_is_an_error(
    db: Database, data_dir: Path, job_kinds: tuple[str, ...]
) -> None:
    with pytest.raises(NotFoundError, match="no video matches"):
        catalog.videos_remove(db, video="404", yes=True)


def test_directory_bytes_sums_a_tree(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"12345")
    (tmp_path / "two").write_bytes(b"123")
    assert catalog.directory_bytes(tmp_path) == 8
    assert catalog.directory_bytes(tmp_path / "missing") == 0


def test_both_remove_commands_are_registered() -> None:
    from rytp import commands

    assert "channel.remove" in commands.COMMANDS
    assert "videos.remove" in commands.COMMANDS
    assert {"dry_run", "yes"} <= {
        param.name for param in commands.COMMANDS["videos.remove"].params
    }
