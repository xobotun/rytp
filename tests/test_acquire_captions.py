"""Tests for caption acquisition (design §6). No network, ever."""

from __future__ import annotations

import pytest

from rytp import config
from rytp import constants as C
from rytp.acquire.captions import acquire_captions
from rytp.acquire.policy import NoCaptions, RateLimited
from rytp.db import Database
from rytp.db.queries import asset_for
from tests.fakes import CAPTION_FILE_SPEC, FakeYtDlpRunner, make_video


def test_the_track_becomes_a_captions_asset(db: Database) -> None:
    vid = make_video(db)
    message = acquire_captions(db, vid, runner=FakeYtDlpRunner()).message
    row = asset_for(db, vid, "captions")
    assert row is not None
    assert row["path"] == str(config.paths().media_dir(vid) / "captions.json3")
    assert row["format_id"] == "ru-orig"
    assert "captions.json3" in message


def test_auto_captions_are_requested_because_ru_orig_is_auto(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner()
    acquire_captions(db, vid, runner=runner)
    call = next(c for c in runner.calls if c["op"] == "download_captions")
    assert call["langs"] == C.CAPTION_LANGS
    assert call["sub_format"] == C.CAPTION_FORMAT


def test_the_most_preferred_language_wins(db: Database) -> None:
    vid = make_video(db)
    plain_ru = dict(CAPTION_FILE_SPEC, name="captions.ru.json3",
                    format_id="ru", language="ru")
    runner = FakeYtDlpRunner(captions=[plain_ru, dict(CAPTION_FILE_SPEC)])
    result = acquire_captions(db, vid, runner=runner)
    assert asset_for(db, vid, "captions")["format_id"] == "ru-orig"
    assert result.fallback_note is None


def test_a_fallback_language_leaves_a_note_for_the_job_row(db: Database) -> None:
    vid = make_video(db)
    plain_ru = dict(CAPTION_FILE_SPEC, name="captions.ru.json3",
                    format_id="ru", language="ru")
    result = acquire_captions(db, vid, runner=FakeYtDlpRunner(captions=[plain_ru]))
    assert asset_for(db, vid, "captions")["format_id"] == "ru"
    assert result.fallback_note and "ru-orig" in result.fallback_note


def test_a_video_with_no_track_fails_permanently(db: Database) -> None:
    vid = make_video(db)
    with pytest.raises(NoCaptions, match="no caption track"):
        acquire_captions(db, vid, runner=FakeYtDlpRunner(captions=[]))


def test_runner_failures_are_translated(db: Database) -> None:
    vid = make_video(db)
    runner = FakeYtDlpRunner(raises=RuntimeError("ERROR: HTTP Error 429"))
    with pytest.raises(RateLimited):
        acquire_captions(db, vid, runner=runner)


def test_refetching_replaces_rather_than_duplicates(db: Database) -> None:
    from rytp.db.queries import assets_for

    vid = make_video(db)
    acquire_captions(db, vid, runner=FakeYtDlpRunner())
    acquire_captions(db, vid, runner=FakeYtDlpRunner())
    assert len(assets_for(db, vid, "captions")) == 1
