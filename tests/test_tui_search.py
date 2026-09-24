"""Searching and playing in the TUI (design §7, §10).

The session is headless and carries every assertion that is not about a
widget. No test reaches a real player: `search.play` goes through the
same `_run` seam Task 6 built, and it is replaced here the same way.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

import rytp.commands.search  # noqa: F401 - importing registers the commands
from rytp import constants as C
from rytp.db import Database
from rytp.index import export
from rytp.tui.app import RytpApp
from rytp.tui.screens.search import SearchScreen, SearchSession
from tests.test_index_search import add_words, corpus, name_speaker
from tests.test_index_utterances import make_speaker, make_video


@pytest.fixture(autouse=True)
def never_spawn_a_player(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Task 6's seam, replaced the same way. Autouse, so a new test that
    forgets cannot open a window."""
    recorded: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        recorded.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(export, "_run", fake_run)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: "/usr/bin/ffplay")
    return recorded


def cached_wav(video_id: int) -> Path:
    import struct
    import wave

    from rytp.config import paths

    path = paths().cache_wav(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(C.AUDIO_CHANNELS)
        handle.setsampwidth(2)
        handle.setframerate(C.AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(struct.pack("<h", 0) * C.AUDIO_SAMPLE_RATE_HZ)
    return path


# --- headless: the session ------------------------------------------


def test_a_search_fills_the_session_from_the_command(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья", title="Evening")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert session.columns == ("anchor", "video", "time", "speaker", "tier", "text")
    assert session.rows[0][0] == f"v{video_id}:0-1"
    assert session.error is None


def test_the_session_reports_the_tier_because_an_inflection_is_not_what_was_typed(
    db: Database,
) -> None:
    corpus(db, "Сегодня было странное ощущение")
    session = SearchSession(db)
    session.run("ощущения")
    assert "stem match" in session.status


def test_the_session_names_the_transcript_tier_per_row(db: Database) -> None:
    """Contracts §3 has three tiers and the screen has to distinguish them:
    a `timed` hit needs `transcribe align`, a caption hit needs a
    transcriber, and only `aligned` can be cut."""
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    corpus(db, "Добрый вечер", source="timed", title="Timed")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert sorted(row[4] for row in session.rows) == ["caption", "timed"]


def test_no_hits_is_a_status_line_not_an_error(db: Database) -> None:
    corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("совершенно другое")
    assert session.rows == ()
    assert "no hits" in session.status
    assert session.error is None


def test_a_bad_query_is_shown_rather_than_raised(db: Database) -> None:
    """A TUI that dies on an empty search box is not a TUI."""
    corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("   ")
    assert session.error is not None
    assert session.rows == ()


def test_an_unknown_speaker_is_shown_rather_than_raised(db: Database) -> None:
    """The resolver raises — right for the CLI, fatal for a screen. Showing
    it is also what keeps "unknown person" from reading as "never said it"."""
    corpus(db, "Добрый вечер")
    session = SearchSession(db, speaker="Никто")
    session.run("добрый вечер")
    assert session.error is not None


def test_the_session_passes_the_filters_to_the_command(db: Database) -> None:
    """It hands `search.words` a speaker string exactly as the CLI does, so
    the shared resolver is settled in one place, not two."""
    video_id = make_video(db)
    host = make_speaker(db, video_id, "SPEAKER_00")
    name_speaker(db, host, "Ведущий")
    add_words(db, video_id, "Добрый вечер", speaker_id=host)
    from rytp.index.utterances import index_video

    index_video(db, video_id)
    corpus(db, "Добрый вечер", title="Anonymous")

    assert len(SearchSession(db).also("добрый вечер")) == 2
    assert len(SearchSession(db, speaker="Ведущий").also("добрый вечер")) == 1


def test_cuttable_only_filters_through_the_same_command(db: Database) -> None:
    aligned = corpus(db, "Добрый вечер", title="Aligned")
    corpus(db, "Добрый вечер", source="caption", title="Captioned")
    assert len(SearchSession(db).also("добрый вечер")) == 2
    rows = SearchSession(db, cuttable=True).also("добрый вечер")
    assert [row[0] for row in rows] == [f"v{aligned}:0-1"]


def test_cuttable_only_status_explains_what_cuttable_means(db: Database) -> None:
    """BUGS.md entry 19: "cuttable only" said nothing about what it meant."""
    corpus(db, "Добрый вечер")
    session = SearchSession(db, cuttable=True)
    session.run("добрый вечер")
    assert "aligned" in session.status
    assert "--allow-timed" in session.status


def test_play_routes_the_highlighted_row_through_the_play_command(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер дорогие друзья")
    cached_wav(video_id)
    session = SearchSession(db)
    session.run("дорогие друзья")
    message = session.play(0)
    assert never_spawn_a_player[0][0] == "/usr/bin/ffplay"
    assert f"v{video_id}:2-3" in message


def test_play_on_an_empty_result_says_so_rather_than_indexing_into_nothing(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    session = SearchSession(db)
    session.run("совершенно другое")
    session.play(0)
    assert never_spawn_a_player == []
    assert session.error is not None


def test_play_reports_a_missing_ffplay_instead_of_raising(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)
    monkeypatch.setattr(export, "_ffplay_binary", lambda: None)
    session = SearchSession(db)
    session.run("добрый вечер")
    session.play(0)
    assert session.error is not None
    assert "ffplay" in session.error


def test_anchor_at_is_the_column_search_play_takes(db: Database) -> None:
    video_id = corpus(db, "Добрый вечер")
    session = SearchSession(db)
    session.run("добрый вечер")
    assert session.anchor_at(0) == f"v{video_id}:0-1"
    assert session.anchor_at(99) is None


# --- the shell -------------------------------------------------------


def test_the_screen_fills_its_table_from_a_query(db: Database) -> None:
    corpus(db, "Добрый вечер дорогие друзья")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("добрый вечер")
            assert screen.query_one("#search-hits").row_count == 1

    asyncio.run(scenario())


def test_the_screen_shows_the_tier_in_its_status_line(db: Database) -> None:
    corpus(db, "Сегодня было странное ощущение")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("ощущения")
            assert "stem match" in str(screen.query_one("#search-status").content)

    asyncio.run(scenario())


def test_the_screen_plays_the_highlighted_hit(
    db: Database, never_spawn_a_player: list[list[str]]
) -> None:
    video_id = corpus(db, "Добрый вечер")
    cached_wav(video_id)

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("добрый вечер")
            screen.action_play()
            assert never_spawn_a_player and never_spawn_a_player[0][0] == "/usr/bin/ffplay"

    asyncio.run(scenario())


def test_the_screen_survives_a_bad_query(db: Database) -> None:
    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("   ")
            assert screen.query_one("#search-hits").row_count == 0
            assert str(screen.query_one("#search-status").content)

    asyncio.run(scenario())


def test_the_screen_binds_play_and_the_filters(db: Database) -> None:
    keys = {binding.key for binding in SearchScreen.BINDINGS}
    assert {"ctrl+p", "ctrl+t", "escape"} <= keys


def test_the_app_binds_a_key_to_search() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f4" in keys
