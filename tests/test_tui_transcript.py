"""Reading a transcript in the TUI (design §10).

The screens are shells: every assertion that is not about widgets is an
assertion about the command they call.
"""

from __future__ import annotations

import asyncio

import pytest

import rytp.commands.search  # noqa: F401 - importing registers the commands
from rytp.commands import resolve
from rytp.db import Database
from rytp.index.utterances import indexed_videos
from rytp.models import RytpError
from rytp.tui.app import RytpApp
from rytp.tui.screens.transcript import TranscriptScreen, TranscriptVideosScreen
from tests.test_index_search import add_words, corpus
from tests.test_index_utterances import make_speaker, make_video

# --- headless: the listing ------------------------------------------


def test_indexed_videos_lists_only_what_has_utterances(db: Database) -> None:
    indexed = corpus(db, "Добрый вечер", title="Indexed")
    bare = make_video(db, "Bare")
    add_words(db, bare, "Здравствуйте друзья")
    assert [row[0] for row in indexed_videos(db)] == [indexed]


def test_indexed_videos_carries_the_title_and_a_count(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер", title="Evening")
    assert indexed_videos(db) == [(video_id, "Evening", 1)]


def test_indexed_videos_is_empty_before_anything_is_indexed(db: Database) -> None:
    make_video(db, "Nothing")
    assert indexed_videos(db) == []


# --- headless: the command the screen renders ------------------------


def test_transcript_show_returns_the_blocks_as_a_table(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")
    result = resolve("transcript.show").handler(db, video=str(video_id))
    assert result.columns == ("anchor", "start", "end", "speaker", "text")
    assert result.rows[0][0] == f"v{video_id}:0-3"
    assert result.rows[0][1] == "00:00:00.000"
    assert result.rows[0][4] == "Добрый вечер дорогие друзья"


def test_transcript_show_writes_no_file(db: Database) -> None:
    """The difference from `transcript.build`, and the reason both exist."""
    from rytp.config import paths

    video_id = corpus(db, "Добрый вечер")
    resolve("transcript.show").handler(db, video=str(video_id))
    assert not paths().transcript(video_id).exists()


def test_transcript_show_names_the_speaker(db: Database) -> None:
    video_id = make_video(db, "Evening")
    local = make_speaker(db, video_id, "SPEAKER_00")
    add_words(db, video_id, "Добрый вечер", speaker_id=local)
    from rytp.index.utterances import index_video

    index_video(db, video_id)
    assert resolve("transcript.show").handler(db, video=str(video_id)).rows[0][3] == (
        "SPEAKER_00"
    )


def test_transcript_show_tells_you_to_index_first(db: Database) -> None:
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")
    with pytest.raises(RytpError, match="index"):
        resolve("transcript.show").handler(db, video=str(video_id))


def test_transcript_show_is_not_long_running() -> None:
    """It reads rows. The TUI may run it inline; that is the point."""
    assert resolve("transcript.show").long_running is False


# --- the shells ------------------------------------------------------


def test_the_reader_screen_shows_every_block(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptScreen(db, video_id)
            await app.push_screen(screen)
            table = screen.query_one("#transcript-blocks")
            assert table.row_count == 1
            assert "Evening" in str(screen.query_one("#transcript-status").content)

    asyncio.run(scenario())


def test_the_reader_screen_reports_an_unindexed_video_instead_of_crashing(
    db: Database,
) -> None:
    """A TUI that raises on a normal mistake is worse than one that says so."""
    video_id = make_video(db)
    add_words(db, video_id, "Добрый вечер")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptScreen(db, video_id)
            await app.push_screen(screen)
            assert "index" in str(screen.query_one("#transcript-status").content)

    asyncio.run(scenario())


def test_the_picker_lists_indexed_videos(db: Database) -> None:
    corpus(db, "Добрый вечер", title="Evening")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptVideosScreen(db)
            await app.push_screen(screen)
            assert screen.query_one("#transcript-videos").row_count == 1

    asyncio.run(scenario())


def test_the_picker_says_so_when_nothing_is_indexed(db: Database) -> None:
    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = TranscriptVideosScreen(db)
            await app.push_screen(screen)
            assert "index build" in str(
                screen.query_one("#transcript-videos-status").content
            )

    asyncio.run(scenario())


def test_the_app_binds_a_key_to_the_reader() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f6" in keys
