"""The mapper screen: a shell over MappingSession, driven headlessly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from textual.widgets import DataTable, Input

from rytp.db import Database
from rytp.diarize import store
from rytp.tui.app import RytpApp
from rytp.tui.screens.speakers import SpeakerMapperScreen, SpeakerVideosScreen
from tests.fakes import make_video
from tests.test_speakers_labels import add_label, add_words


def a_diarized_video(db: Database) -> int:
    video_id = make_video(db, title="An Interview")
    word_ids = add_words(db, video_id, [(0, 4_000), (4_000, 8_000)])
    host = add_label(db, video_id, "SPEAKER_00")
    guest = add_label(db, video_id, "SPEAKER_01")
    store.stamp_words(db, video_id, {word_ids[0]: host, word_ids[1]: guest})
    store.add_speaker(db, "Host One", aliases=("h1",))
    store.add_speaker(db, "Guest Two", aliases=("g2",))
    return video_id


def drive(db: Database, body: Callable[[RytpApp, Any], Awaitable[None]]) -> None:
    """Run `body` against a mounted RytpApp, headlessly."""

    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            await body(app, pilot)

    asyncio.run(main())


def test_the_mapper_shows_both_panes(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#labels", DataTable).row_count == 2
        assert screen.query_one("#roster", DataTable).row_count == 2

    drive(db, body)


def test_typing_in_the_filter_narrows_the_roster_pane_only(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "h1"
        await pilot.pause()
        assert screen.query_one("#roster", DataTable).row_count == 1
        assert screen.query_one("#labels", DataTable).row_count == 2

    drive(db, body)


def test_assigning_names_the_label_and_says_so(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "Host One"
        await pilot.pause()
        screen.action_assign()
        await pilot.pause()
        assert "Host One" in screen.session.status
        assert store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Host One"

    drive(db, body)


def test_typing_a_new_name_and_creating_links_in_one_gesture(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        screen.query_one("#roster-filter", Input).value = "Каспар Хаузер"
        await pilot.pause()
        screen.action_new_speaker()
        await pilot.pause()
        assert (
            store.find_label(db, video_id, "SPEAKER_00").speaker_label == "Каспар Хаузер"
        )

    drive(db, body)


def test_the_suggestion_pane_is_hidden_until_asked_for(db: Database) -> None:
    video_id = a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        suggestions = screen.query_one("#suggestions", DataTable)
        assert suggestions.display is False
        screen.action_toggle_suggestions()
        await pilot.pause()
        assert suggestions.display is True

    drive(db, body)


def test_the_video_chooser_lists_diarized_videos(db: Database) -> None:
    a_diarized_video(db)
    make_video(db, external_id="VIDEO_B", url="https://example.invalid/b")

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerVideosScreen(db)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#videos", DataTable).row_count == 1

    drive(db, body)


def test_f3_opens_the_speakers_screen(db: Database) -> None:
    a_diarized_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        await pilot.press("f3")
        await pilot.pause()
        assert isinstance(app.screen, SpeakerVideosScreen)

    drive(db, body)


def test_a_video_with_no_labels_still_mounts(db: Database) -> None:
    video_id = make_video(db)

    async def body(app: RytpApp, pilot: Any) -> None:
        screen = SpeakerMapperScreen(db, video_id)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#labels", DataTable).row_count == 0
        screen.action_assign()
        await pilot.pause()
        assert "no label" in screen.session.status

    drive(db, body)


def test_importing_the_screen_creates_no_directories(tmp_path: Path) -> None:
    import subprocess
    import sys

    from tests.test_config import child_env

    proc = subprocess.run(
        [sys.executable, "-c", "import rytp.tui.screens.speakers"],
        cwd=str(tmp_path),
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.iterdir()) == []
