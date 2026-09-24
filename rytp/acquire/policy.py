"""Download politeness. design §5.

Every value here is a default that ``settings`` can override, and the
throttle backoff deliberately applies to the whole **network pool** rather
than to the job that tripped it: deferring one job would just let the next
download hit the same throttled server a second later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from rytp import constants as C
from rytp.db import Database
from rytp.db.queries import get_setting, set_setting
from rytp.models import RytpError


class AcquireError(RytpError):
    """Acquisition failed for a reason worth one line on stderr."""


class PermanentAcquireError(AcquireError):
    """Retrying will not help. The worker parks the job in ``failed``."""


class VideoUnavailable(PermanentAcquireError):
    """Private, removed, geo-blocked, or members-only."""


class NoCaptions(PermanentAcquireError):
    """The video has no caption track in any requested language."""


class RateLimited(AcquireError):
    """HTTP 429 / 403-forbidden. The whole network pool must back off."""


class DailyCapReached(AcquireError):
    """The per-day download cap in settings has been reached."""


class MissingAssetError(AcquireError):
    """A required asset row or its file on disk is absent."""


class ErrorKind(Enum):
    THROTTLED = "throttled"
    UNAVAILABLE = "unavailable"
    TRANSIENT = "transient"


# Checked before the throttle pattern on purpose: a delisted video often
# answers 403, and treating that as a throttle would freeze the pool.
_UNAVAILABLE_RE = re.compile(
    r"private video|video unavailable|removed by the uploader"
    r"|account associated with this video has been terminated"
    r"|members[- ]only|not available in your country|no video formats found"
    # An audio-only source answers this. Retrying it five times against the
    # network is the one thing the whole politeness budget exists to avoid.
    r"|requested format is not available",
    re.IGNORECASE,
)
_THROTTLE_RE = re.compile(
    r"http error 429|http error 403|too many requests|rate[- ]?limit"
    r"|sign in to confirm",
    re.IGNORECASE,
)


def classify_error(message: str) -> ErrorKind:
    """Decide what a failure message means for retry behaviour."""
    if _UNAVAILABLE_RE.search(message):
        return ErrorKind.UNAVAILABLE
    if _THROTTLE_RE.search(message):
        return ErrorKind.THROTTLED
    return ErrorKind.TRANSIENT


def _float(db: Database, key: str, default: float) -> float:
    raw = get_setting(db, key)
    return default if raw is None else float(raw)


def _int(db: Database, key: str, default: int) -> int:
    raw = get_setting(db, key)
    return default if raw is None else int(raw)


def _str(db: Database, key: str, default: str) -> str:
    raw = get_setting(db, key)
    return default if raw is None else raw


def _tuple(db: Database, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = get_setting(db, key)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class DownloadPolicy:
    """The tunable half of acquisition, read once per stage invocation."""

    delay_min_s: float
    delay_max_s: float
    file_delay_s: float
    rate_limit_bps: int | None
    sleep_requests_s: float
    daily_cap: int
    backoff_ladder_s: tuple[int, ...]
    format_audio: str
    format_video: str
    caption_langs: tuple[str, ...]
    caption_format: str

    @classmethod
    def from_settings(cls, db: Database) -> DownloadPolicy:
        rate = _int(db, C.SETTING_DOWNLOAD_RATE_LIMIT, C.DOWNLOAD_RATE_LIMIT_BPS)
        return cls(
            delay_min_s=_float(db, C.SETTING_DOWNLOAD_DELAY_MIN, C.DOWNLOAD_DELAY_MIN_S),
            delay_max_s=_float(db, C.SETTING_DOWNLOAD_DELAY_MAX, C.DOWNLOAD_DELAY_MAX_S),
            file_delay_s=_float(db, C.SETTING_DOWNLOAD_FILE_DELAY, C.DOWNLOAD_FILE_DELAY_S),
            rate_limit_bps=rate or None,
            sleep_requests_s=_float(
                db, C.SETTING_DOWNLOAD_SLEEP_REQUESTS, C.DOWNLOAD_SLEEP_REQUESTS_S
            ),
            daily_cap=_int(db, C.SETTING_DOWNLOAD_DAILY_CAP, C.DOWNLOAD_DAILY_CAP),
            backoff_ladder_s=C.THROTTLE_BACKOFF_LADDER_S,
            format_audio=_str(db, C.SETTING_DOWNLOAD_FORMAT_AUDIO, C.DOWNLOAD_FORMAT_AUDIO),
            format_video=_str(db, C.SETTING_DOWNLOAD_FORMAT_VIDEO, C.DOWNLOAD_FORMAT_VIDEO),
            caption_langs=_tuple(db, C.SETTING_CAPTION_LANGS, C.CAPTION_LANGS),
            caption_format=_str(db, C.SETTING_CAPTION_FORMAT, C.CAPTION_FORMAT),
        )


def next_delay_s(policy: DownloadPolicy, rng: random.Random) -> float:
    """Randomised pause before the next video. design §5."""
    return rng.uniform(policy.delay_min_s, policy.delay_max_s)


def backoff_seconds(policy: DownloadPolicy, streak: int) -> int:
    """Ladder position for a throttle streak; the last rung repeats."""
    ladder = policy.backoff_ladder_s
    return ladder[min(max(streak, 1), len(ladder)) - 1]


def cooldown_until(db: Database) -> str | None:
    """ISO timestamp the network pool is frozen until, or None."""
    return get_setting(db, C.SETTING_COOLDOWN_UNTIL) or None


def is_cooling_down(db: Database, *, now: datetime) -> bool:
    until = cooldown_until(db)
    return until is not None and now.isoformat() < until


def record_throttle(db: Database, policy: DownloadPolicy, *, now: datetime) -> str:
    """Advance the throttle streak and freeze the pool. Returns the ISO end."""
    streak = _int(db, C.SETTING_THROTTLE_STREAK, 0) + 1
    until = (now + timedelta(seconds=backoff_seconds(policy, streak))).isoformat()
    # Not wrapped in db.transaction(): set_setting manages its own write and
    # there is nothing here that has to land atomically — a streak without a
    # cooldown, or the reverse, is self-correcting on the next tick.
    set_setting(db, C.SETTING_THROTTLE_STREAK, str(streak))
    set_setting(db, C.SETTING_COOLDOWN_UNTIL, until)
    return until


def clear_throttle(db: Database) -> None:
    """A clean download resets the ladder to the bottom."""
    set_setting(db, C.SETTING_THROTTLE_STREAK, "0")
    set_setting(db, C.SETTING_COOLDOWN_UNTIL, "")


def downloads_started_today(db: Database, *, now: datetime) -> int:
    """Download jobs started since UTC midnight — attempts, not successes."""
    day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE kind = 'download' AND started_at >= ?",
        (day_start.isoformat(),),
    ).fetchone()
    return int(row["n"])


def check_daily_cap(db: Database, policy: DownloadPolicy, *, now: datetime) -> None:
    """Raise :class:`DailyCapReached` if today's budget is spent."""
    if policy.daily_cap <= 0:
        return
    started = downloads_started_today(db, now=now)
    if started >= policy.daily_cap:
        raise DailyCapReached(
            f"daily cap reached: {started} downloads started today, cap is "
            f"{policy.daily_cap} (setting {C.SETTING_DOWNLOAD_DAILY_CAP})"
        )


def pool_size(db: Database, pool: str) -> int:
    """Slot count for one pool, from settings or the built-in default."""
    return _int(db, C.SETTING_POOL_SIZE.format(pool=pool), C.POOL_DEFAULT_SIZE[pool])
