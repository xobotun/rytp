"""Tests for the yt-dlp seam, enumeration and probing. No network, ever."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.acquire.policy import RateLimited, VideoUnavailable
from rytp.acquire.ytdlp import YtDlpRunner, enumerate_channel, probe_video, tab_url
from tests.fakes import CHANNEL_ONE_URL, FakeYtDlpRunner


def _entry(external_id: str, title: str, duration: float | None = 3600.0) -> dict:
    return {
        "id": external_id,
        "title": title,
        "url": f"https://example.invalid/watch/{external_id}",
        "duration": duration,
        "upload_date": "20200102",
    }


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeYtDlpRunner(), YtDlpRunner)


@pytest.mark.parametrize(
    ("given", "tab", "expected"),
    [
        (CHANNEL_ONE_URL, "videos", f"{CHANNEL_ONE_URL}/videos"),
        (CHANNEL_ONE_URL + "/", "streams", f"{CHANNEL_ONE_URL}/streams"),
        (CHANNEL_ONE_URL + "/videos", "shorts", f"{CHANNEL_ONE_URL}/shorts"),
    ],
)
def test_tab_url_normalises_whatever_the_user_pasted(
    given: str, tab: str, expected: str
) -> None:
    assert tab_url(given, tab) == expected


def test_enumeration_covers_videos_streams_and_shorts() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [_entry("VIDEO_B", "B", 10800.0)],
        f"{CHANNEL_ONE_URL}/shorts": [_entry("VIDEO_C", "C", 45.0)],
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [(e.external_id, e.kind) for e in got] == [
        ("VIDEO_A", "video"), ("VIDEO_B", "livestream"), ("VIDEO_C", "short")
    ]
    assert got[0].duration_ms == 3_600_000
    assert got[0].published_at == "2020-01-02"


def test_enumeration_keeps_tab_order_and_does_not_dedupe() -> None:
    # A live stream listed in both /videos and /streams must come back twice,
    # streams last, so the caller's upsert lands on kind='livestream'.
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_B", "B")],
        f"{CHANNEL_ONE_URL}/streams": [_entry("VIDEO_B", "B")],
        f"{CHANNEL_ONE_URL}/shorts": [],
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [e.kind for e in got] == ["video", "livestream"]


def test_a_channel_without_a_shorts_tab_is_not_an_error() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [_entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [],
        # no /shorts key at all -> the fake raises "does not have a videos tab"
    })
    got = enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [e.external_id for e in got] == ["VIDEO_A"]


def test_enumeration_stops_on_a_throttle_instead_of_hammering_the_next_tab() -> None:
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429: Too Many Requests"))
    with pytest.raises(RateLimited):
        enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert len(runner.calls) == 1


def test_entries_without_an_id_are_skipped() -> None:
    runner = FakeYtDlpRunner(entries={
        f"{CHANNEL_ONE_URL}/videos": [{"title": "orphan"}, _entry("VIDEO_A", "A")],
        f"{CHANNEL_ONE_URL}/streams": [],
        f"{CHANNEL_ONE_URL}/shorts": [],
    })
    assert [e.external_id for e in enumerate_channel(CHANNEL_ONE_URL, runner=runner)] == [
        "VIDEO_A"
    ]


def test_probe_video_returns_one_channel_entry() -> None:
    runner = FakeYtDlpRunner(info={
        "id": "VIDEO_A", "title": "A", "duration": 90.5,
        "webpage_url": "https://example.invalid/watch/VIDEO_A",
        "upload_date": "20240315", "live_status": "not_live",
    })
    entry = probe_video("https://example.invalid/watch/VIDEO_A", runner=runner)
    assert entry.external_id == "VIDEO_A"
    assert entry.kind == "video"
    assert entry.duration_ms == 90_500
    assert entry.published_at == "2024-03-15"


def test_probe_video_recognises_a_past_live_stream() -> None:
    runner = FakeYtDlpRunner(info={
        "id": "VIDEO_B", "title": "B", "duration": 10800.0,
        "webpage_url": "https://example.invalid/watch/VIDEO_B",
        "live_status": "was_live",
    })
    assert probe_video("https://example.invalid/watch/VIDEO_B", runner=runner).kind == (
        "livestream"
    )


def test_probe_video_maps_a_dead_video_to_video_unavailable() -> None:
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: Private video"))
    with pytest.raises(VideoUnavailable):
        probe_video("https://example.invalid/watch/VIDEO_A", runner=runner)


def test_enumeration_asks_for_every_configured_tab() -> None:
    runner = FakeYtDlpRunner(entries={f"{CHANNEL_ONE_URL}/{t}": [] for t in C.CHANNEL_TABS})
    enumerate_channel(CHANNEL_ONE_URL, runner=runner)
    assert [c["url"] for c in runner.calls] == [
        f"{CHANNEL_ONE_URL}/{t}" for t in C.CHANNEL_TABS
    ]
