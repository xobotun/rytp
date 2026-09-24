"""Tests for the download politeness policy (design §5)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from rytp import constants as C
from rytp.acquire import policy as P
from rytp.db import Database
from rytp.db.queries import set_setting

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def test_defaults_come_from_constants(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert p.delay_min_s == C.DOWNLOAD_DELAY_MIN_S
    assert p.delay_max_s == C.DOWNLOAD_DELAY_MAX_S
    assert p.daily_cap == C.DOWNLOAD_DAILY_CAP
    assert p.rate_limit_bps == C.DOWNLOAD_RATE_LIMIT_BPS
    assert p.caption_langs == C.CAPTION_LANGS
    assert p.backoff_ladder_s == C.THROTTLE_BACKOFF_LADDER_S


def test_settings_override_every_value(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MIN, "1.5")
    set_setting(db, C.SETTING_DOWNLOAD_DELAY_MAX, "2.5")
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "7")
    set_setting(db, C.SETTING_DOWNLOAD_RATE_LIMIT, "0")
    set_setting(db, C.SETTING_CAPTION_LANGS, "ru-orig, ru, en")
    p = P.DownloadPolicy.from_settings(db)
    assert (p.delay_min_s, p.delay_max_s) == (1.5, 2.5)
    assert p.daily_cap == 7
    assert p.rate_limit_bps is None  # 0 means "no limit"
    assert p.caption_langs == ("ru-orig", "ru", "en")


def test_next_delay_is_inside_the_configured_band(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    rng = random.Random(1234)
    for _ in range(50):
        d = P.next_delay_s(p, rng)
        assert C.DOWNLOAD_DELAY_MIN_S <= d <= C.DOWNLOAD_DELAY_MAX_S


def test_backoff_ladder_climbs_then_plateaus(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert [P.backoff_seconds(p, n) for n in (1, 2, 3, 4, 5, 99)] == [
        300, 900, 2700, 7200, 7200, 7200
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ERROR: HTTP Error 429: Too Many Requests", P.ErrorKind.THROTTLED),
        ("ERROR: HTTP Error 403: Forbidden", P.ErrorKind.THROTTLED),
        ("Sign in to confirm you're not a bot", P.ErrorKind.THROTTLED),
        ("ERROR: Private video. Sign in if you've been granted access", P.ErrorKind.UNAVAILABLE),
        ("ERROR: Video unavailable", P.ErrorKind.UNAVAILABLE),
        ("ERROR: This video has been removed by the uploader", P.ErrorKind.UNAVAILABLE),
        ("ERROR: Requested format is not available", P.ErrorKind.UNAVAILABLE),
        ("ERROR: unable to download video data: timed out", P.ErrorKind.TRANSIENT),
    ],
)
def test_classify_error(message: str, expected: P.ErrorKind) -> None:
    assert P.classify_error(message) is expected


def test_unavailable_wins_over_a_403_in_the_same_message() -> None:
    # A delisted video often answers 403. Treating that as a throttle would
    # freeze the whole network pool for two hours over one dead video.
    msg = "ERROR: HTTP Error 403: Forbidden. Video unavailable"
    assert P.classify_error(msg) is P.ErrorKind.UNAVAILABLE


def test_throttle_sets_a_pool_wide_cooldown_that_climbs(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    assert P.is_cooling_down(db, now=NOW) is False

    first = P.record_throttle(db, p, now=NOW)
    assert first == (NOW + timedelta(seconds=300)).isoformat()
    assert P.is_cooling_down(db, now=NOW) is True
    assert P.is_cooling_down(db, now=NOW + timedelta(seconds=301)) is False

    second = P.record_throttle(db, p, now=NOW)
    assert second == (NOW + timedelta(seconds=900)).isoformat()


def test_success_clears_the_cooldown_and_the_streak(db: Database) -> None:
    p = P.DownloadPolicy.from_settings(db)
    P.record_throttle(db, p, now=NOW)
    P.record_throttle(db, p, now=NOW)
    P.clear_throttle(db)
    assert P.is_cooling_down(db, now=NOW) is False
    assert P.cooldown_until(db) is None
    # The ladder restarts from the bottom after a clean download.
    assert P.record_throttle(db, p, now=NOW) == (NOW + timedelta(seconds=300)).isoformat()


def test_daily_cap_counts_downloads_started_today(db: Database) -> None:
    yesterday = (NOW - timedelta(days=1)).isoformat()
    today = NOW.isoformat()
    for i, started in enumerate([yesterday, today, today]):
        db.conn.execute(
            "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
            "created_at, started_at) VALUES "
            "('download', ?, 'done', 'network', '{}', ?, ?)",
            (i + 1, started, started),
        )
    assert P.downloads_started_today(db, now=NOW) == 2


def test_check_daily_cap_raises_when_reached(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "1")
    p = P.DownloadPolicy.from_settings(db)
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
        "created_at, started_at) VALUES ('download', 1, 'done', 'network', "
        "'{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    with pytest.raises(P.DailyCapReached, match="daily cap"):
        P.check_daily_cap(db, p, now=NOW)


def test_zero_daily_cap_means_unlimited(db: Database) -> None:
    set_setting(db, C.SETTING_DOWNLOAD_DAILY_CAP, "0")
    p = P.DownloadPolicy.from_settings(db)
    db.conn.execute(
        "INSERT INTO jobs (kind, target_id, state, pool, payload_json, "
        "created_at, started_at) VALUES ('download', 1, 'done', 'network', "
        "'{}', ?, ?)",
        (NOW.isoformat(), NOW.isoformat()),
    )
    P.check_daily_cap(db, p, now=NOW)  # does not raise


def test_pool_size_defaults_and_overrides(db: Database) -> None:
    assert P.pool_size(db, "network") == 1
    assert P.pool_size(db, "cpu") == C.POOL_DEFAULT_SIZE["cpu"]
    set_setting(db, C.SETTING_POOL_SIZE.format(pool="cpu"), "4")
    assert P.pool_size(db, "cpu") == 4
