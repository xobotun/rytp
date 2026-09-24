"""Catalog SQL: channels, videos, settings."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.db import queries as q
from rytp.models import InvalidInputError

CHANNEL_URL = "https://example.invalid/c/CHANNEL_ONE"


def add_channel(db: Database, url: str = CHANNEL_URL, title: str = "Channel One") -> int:
    channel_id, _created = q.insert_channel(db, url=url, title=title)
    return channel_id


def add_video(db: Database, external_id: str, **overrides: object) -> int:
    kwargs: dict[str, object] = {
        "source": "ytdlp",
        "kind": "video",
        "channel_id": None,
        "external_id": external_id,
        "url": f"https://example.invalid/w/{external_id}",
        "title": f"Title {external_id}",
        "duration_ms": 60_000,
        "published_at": None,
    }
    kwargs.update(overrides)
    video_id, _created = q.upsert_video(db, **kwargs)  # type: ignore[arg-type]
    return video_id


def test_insert_channel_reports_creation_then_reuse(db: Database) -> None:
    first_id, created = q.insert_channel(db, url=CHANNEL_URL, title="Channel One")
    assert created is True
    second_id, created_again = q.insert_channel(db, url=CHANNEL_URL, title="Renamed")
    assert created_again is False
    assert second_id == first_id
    assert q.get_channel(db, first_id)["title"] == "Renamed"


def test_find_channel_by_url_then_by_title(db: Database) -> None:
    channel_id = add_channel(db)
    assert q.find_channel(db, CHANNEL_URL)["id"] == channel_id
    assert q.find_channel(db, "Channel One")["id"] == channel_id
    assert q.find_channel(db, "nothing like it") is None


def test_list_channels_counts_their_videos(db: Database) -> None:
    channel_id = add_channel(db)
    add_video(db, "VIDEO_A", channel_id=channel_id)
    add_video(db, "VIDEO_B", channel_id=channel_id)
    add_video(db, "VIDEO_C")
    rows = q.list_channels(db, limit=C.DEFAULT_LIST_LIMIT)
    assert len(rows) == 1
    assert rows[0]["n_videos"] == 2


def test_mark_channel_synced_records_the_timestamp(db: Database) -> None:
    channel_id = add_channel(db)
    assert q.get_channel(db, channel_id)["last_synced_at"] is None
    q.mark_channel_synced(db, channel_id, "2026-01-01T00:00:00+00:00")
    assert q.get_channel(db, channel_id)["last_synced_at"] == "2026-01-01T00:00:00+00:00"


def test_upsert_video_is_idempotent_on_source_and_external_id(db: Database) -> None:
    first_id, created = q.upsert_video(
        db,
        source="ytdlp",
        kind="video",
        channel_id=None,
        external_id="VIDEO_A",
        url="https://example.invalid/w/VIDEO_A",
        title="First title",
        duration_ms=1000,
        published_at=None,
    )
    assert created is True
    second_id, created_again = q.upsert_video(
        db,
        source="ytdlp",
        kind="livestream",
        channel_id=None,
        external_id="VIDEO_A",
        url="https://example.invalid/w/VIDEO_A",
        title="Corrected title",
        duration_ms=2000,
        published_at="2026-01-01T00:00:00+00:00",
    )
    assert created_again is False
    assert second_id == first_id
    row = q.get_video(db, first_id)
    assert row["title"] == "Corrected title"
    assert row["kind"] == "livestream"
    assert row["duration_ms"] == 2000
    assert db.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_upsert_video_sets_created_at_once(db: Database) -> None:
    video_id = add_video(db, "VIDEO_A")
    first_seen = q.get_video(db, video_id)["created_at"]
    add_video(db, "VIDEO_A", title="Renamed")
    assert q.get_video(db, video_id)["created_at"] == first_seen


def test_upsert_video_rejects_a_missing_external_id(db: Database) -> None:
    """A NULL external_id defeats UNIQUE(source, external_id): every add duplicates."""
    with pytest.raises(InvalidInputError, match="external_id"):
        q.upsert_video(
            db,
            source="ytdlp",
            kind="video",
            channel_id=None,
            external_id=None,  # type: ignore[arg-type]
            url=None,
            title="No id",
            duration_ms=None,
            published_at=None,
        )


def test_the_same_external_id_can_exist_under_two_sources(db: Database) -> None:
    remote = add_video(db, "VIDEO_A")
    local, _ = q.upsert_video(
        db,
        source="local",
        kind="video",
        channel_id=None,
        external_id="VIDEO_A",
        url=None,
        title="A local file that happens to share a name",
        duration_ms=None,
        published_at=None,
    )
    assert remote != local


def test_list_videos_filters_by_channel_kind_source_and_search(db: Database) -> None:
    channel_id = add_channel(db)
    add_video(db, "VIDEO_A", channel_id=channel_id, title="Утренний эфир")
    add_video(db, "VIDEO_B", channel_id=channel_id, kind="livestream", title="Прямой эфир")
    add_video(db, "VIDEO_C", title="Unrelated")
    add_video(db, "VIDEO_D", source="local", title="From disk")

    def ids(**filters: object) -> list[str]:
        rows = q.list_videos(db, limit=C.DEFAULT_LIST_LIMIT, **filters)  # type: ignore[arg-type]
        return [row["external_id"] for row in rows]

    assert sorted(ids(channel_id=channel_id)) == ["VIDEO_A", "VIDEO_B"]
    assert ids(kind="livestream") == ["VIDEO_B"]
    assert ids(source="local") == ["VIDEO_D"]
    assert ids(search="эфир") == ["VIDEO_A", "VIDEO_B"] or ids(search="эфир") == [
        "VIDEO_B",
        "VIDEO_A",
    ]
    assert ids(channel_id=channel_id, kind="video") == ["VIDEO_A"]


def test_list_videos_orders_newest_first_and_respects_the_limit(db: Database) -> None:
    for name in ("VIDEO_A", "VIDEO_B", "VIDEO_C"):
        add_video(db, name)
    rows = q.list_videos(db, limit=2)
    assert [row["external_id"] for row in rows] == ["VIDEO_C", "VIDEO_B"]


def test_list_videos_folds_asset_and_word_state_into_one_join(db: Database) -> None:
    """BUGS.md entry 25: `videos list --long` needs asset counts, the best
    tier, the engine chain and the align scales — folded into this one
    statement rather than a query per video."""
    video_id = add_video(db, "VIDEO_A")
    q.insert_asset(db, video_id=video_id, role="audio", path="a.wav")
    q.insert_asset(db, video_id=video_id, role="video", format_id="480p", path="v1.mp4")
    q.insert_asset(db, video_id=video_id, role="video", format_id="720p", path="v2.mp4")
    db.conn.execute(
        "INSERT INTO words (video_id, ord, start_ms, end_ms, text, "
        "normalized_text, stem, source, engine, align_scale) VALUES "
        "(?, 0, 0, 1000, 'a', 'a', 'a', 'timed', 'whisper', NULL), "
        "(?, 1, 0, 1000, 'b', 'b', 'b', 'aligned', 'whisper+wav2vec2', 'logprob')",
        (video_id, video_id),
    )
    (row,) = q.list_videos(db, limit=C.DEFAULT_LIST_LIMIT)
    assert row["n_audio"] == 1
    assert row["n_captions"] == 0
    assert row["n_video_assets"] == 2
    assert row["best_tier"] == "aligned"
    assert row["engines"] == "whisper, whisper+wav2vec2"
    assert row["align_scales"] == "logprob"


def test_list_videos_reports_no_tier_or_engine_for_an_untranscribed_video(
    db: Database,
) -> None:
    video_id = add_video(db, "VIDEO_A")
    (row,) = q.list_videos(db, limit=C.DEFAULT_LIST_LIMIT)
    assert video_id == row["id"]
    assert row["n_audio"] == 0
    assert row["best_tier"] is None
    assert row["engines"] is None
    assert row["align_scales"] is None


def test_clamp_limit_keeps_the_query_bounded() -> None:
    assert q.clamp_limit(0) == 1
    assert q.clamp_limit(-5) == 1
    assert q.clamp_limit(10) == 10
    assert q.clamp_limit(C.MAX_LIST_LIMIT + 1) == C.MAX_LIST_LIMIT


def test_settings_round_trip(db: Database) -> None:
    assert q.get_setting(db, "download.daily_cap") is None
    assert q.get_setting(db, "download.daily_cap", "200") == "200"
    q.set_setting(db, "download.daily_cap", "50")
    assert q.get_setting(db, "download.daily_cap") == "50"
    q.set_setting(db, "download.daily_cap", "75")
    assert q.get_setting(db, "download.daily_cap") == "75"
