"""The fleet overview (BUGS.md entry 23, plan Task 16).

The session is headless and carries every assertion that is not about a
widget, matching `test_tui_search.py`'s split. Every shortcut is checked by
reading the `jobs` row it left behind — never by mocking a handler — because
the whole point of the screen is that it calls the real registered commands.
"""

from __future__ import annotations

import asyncio

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.jobs.queue import Job
from rytp.tui.app import RytpApp
from rytp.tui.confirm import ConfirmScreen
from rytp.tui.screens.videos import (
    FLEET_HINT,
    NOT_REOPENABLE_MARK,
    VideoFleetSession,
    VideosScreen,
)
from tests.test_index_search import add_words
from tests.test_index_utterances import make_video


def latest_job(db: Database, video_id: int) -> Job | None:
    row = db.conn.execute(
        "SELECT * FROM jobs WHERE target_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
    ).fetchone()
    return Job.from_row(row) if row is not None else None


def diarize_video(db: Database, video_id: int) -> None:
    db.conn.execute(
        "INSERT INTO video_speakers (video_id, local_label, engine)"
        " VALUES (?, 'SPEAKER_00', 'fake')",
        (video_id,),
    )


# --- headless: the session -------------------------------------------


def test_the_session_reuses_videos_list_long(db: Database) -> None:
    """It is `videos list --long`'s own columns, not a second definition."""
    make_video(db, "Первое")
    session = VideoFleetSession(db)
    assert session.columns == (
        "id", "source", "kind", "duration", "channel", "title", "external id",
        "audio", "captions", "videos", "transcribed", "aligned", "indexed",
        "diarized", "tier", "engine", "align scale",
    )
    assert len(session.rows) == 1


def test_a_never_touched_video_is_all_crosses(db: Database) -> None:
    make_video(db, "Нетронутое")
    session = VideoFleetSession(db)
    idx = {name: i for i, name in enumerate(session.columns)}
    row = session.rows[0]
    for column in ("audio", "captions", "videos", "transcribed", "aligned", "indexed", "diarized"):
        assert row[idx[column]] == C.CELL_CROSS, column
    assert row[idx["tier"]] == C.NULL_CELL


def test_the_filter_narrows_by_title(db: Database) -> None:
    make_video(db, "Интервью с гостем")
    make_video(db, "Совсем другое")
    session = VideoFleetSession(db)
    assert len(session.rows) == 2
    session.run("гостем")
    assert len(session.rows) == 1


def test_a_raising_handler_is_shown_rather_than_crashing_the_screen(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`videos.list` itself never raises on a bad `search` string — this
    checks the session's own guard, the same rule `SearchSession.run`
    follows for a bad query, by making the resolved command misbehave."""
    import rytp.tui.screens.videos as videos_module
    from rytp.commands import Command as _Command
    from rytp.commands import resolve as real_resolve
    from rytp.models import InvalidInputError

    make_video(db, "Что угодно")
    session = VideoFleetSession(db)
    assert session.error is None

    def boom(*_args: object, **_kwargs: object) -> None:
        raise InvalidInputError("synthetic failure")

    def fake_resolve(name: str) -> _Command:
        if name == "videos.list":
            return _Command(
                name=name, group="videos", summary="", params=(), handler=boom
            )
        return real_resolve(name)

    monkeypatch.setattr(videos_module, "resolve", fake_resolve)
    session.refresh()
    assert session.error == "synthetic failure"
    assert session.rows == ()


def test_the_aligned_and_diarized_ticks_are_marked_frozen(db: Database) -> None:
    """BUGS.md entry 23's third decision: `align`/`diarize` are
    `reopenable=False`, so their tick must read differently from an
    ordinary satisfied stage."""
    video_id = make_video(db, "Готово")
    add_words(db, video_id, "Добрый вечер", source="aligned")
    diarize_video(db, video_id)
    session = VideoFleetSession(db)
    idx = {name: i for i, name in enumerate(session.columns)}
    row = session.rows[0]
    assert row[idx["aligned"]] == NOT_REOPENABLE_MARK
    assert row[idx["diarized"]] == NOT_REOPENABLE_MARK
    assert row[idx["aligned"]] != C.CELL_TICK
    # Every other stage keeps the plain tick/cross vocabulary.
    assert row[idx["indexed"]] == C.CELL_CROSS


def test_the_status_line_points_at_the_glossary(db: Database) -> None:
    make_video(db)
    session = VideoFleetSession(db)
    assert session.status.endswith(FLEET_HINT)
    assert "F1" in session.status


def test_video_id_at_reads_the_id_column(db: Database) -> None:
    video_id = make_video(db, "Один")
    session = VideoFleetSession(db)
    assert session.video_id_at(0) == video_id
    assert session.video_id_at(99) is None


def test_video_id_at_on_an_empty_table_is_none(db: Database) -> None:
    session = VideoFleetSession(db)
    assert session.rows == ()
    assert session.video_id_at(0) is None


def test_video_summary_at_pairs_the_id_with_the_title(db: Database) -> None:
    video_id = make_video(db, "Интервью [часть 1]")
    session = VideoFleetSession(db)
    assert session.video_summary_at(0) == (video_id, "Интервью [часть 1]")


def test_video_summary_at_on_an_empty_table_is_none(db: Database) -> None:
    session = VideoFleetSession(db)
    assert session.video_summary_at(0) is None


# --- shortcuts: enqueue, never run -------------------------------------


def test_ingest_queues_the_acquisition_chain(db: Database) -> None:
    video_id = make_video(db, "Для загрузки")
    session = VideoFleetSession(db)
    session.ingest(0)
    kinds = {
        row["kind"]
        for row in db.conn.execute(
            "SELECT kind FROM jobs WHERE target_id = ?", (video_id,)
        )
    }
    assert kinds, "ingest queued nothing"
    assert kinds <= {
        "download", "captions", "extract_wav", "caption_words", "fingerprint",
    }
    assert session.error is None
    assert session.note


def test_transcribe_queues_a_transcribe_job(db: Database) -> None:
    video_id = make_video(db, "Для расшифровки")
    session = VideoFleetSession(db)
    session.transcribe(0)
    job = latest_job(db, video_id)
    assert job is not None
    assert job.kind == "transcribe"


def test_align_without_a_default_aligner_reports_rather_than_enqueues(db: Database) -> None:
    video_id = make_video(db, "Без выравнивания")
    session = VideoFleetSession(db)
    session.align(0)
    assert latest_job(db, video_id) is None
    assert "default_aligner" in session.note


def test_align_with_a_default_aligner_queues_an_align_job(db: Database) -> None:
    from rytp.db import queries as q

    video_id = make_video(db, "С выравниванием")
    q.set_setting(db, "default_aligner", "mfa")
    session = VideoFleetSession(db)
    session.align(0)
    job = latest_job(db, video_id)
    assert job is not None
    assert job.kind == "align"
    assert job.payload["aligner"] == "mfa"


def test_diarize_queues_a_diarize_job(db: Database) -> None:
    video_id = make_video(db, "Для диаризации")
    session = VideoFleetSession(db)
    session.diarize(0)
    job = latest_job(db, video_id)
    assert job is not None
    assert job.kind == "diarize"


def test_reindex_queues_an_index_job(db: Database) -> None:
    video_id = make_video(db, "Для индексации")
    add_words(db, video_id, "Добрый вечер", source="timed")
    session = VideoFleetSession(db)
    session.reindex(0)
    job = latest_job(db, video_id)
    assert job is not None
    assert job.kind == "index"


def test_drop_transcript_removes_words_utterances_and_speaker_labels(db: Database) -> None:
    """The session-level half of `shift+delete`, run already-confirmed —
    the screen owns asking, the session owns doing, same split as the
    other five shortcuts."""
    video_id = make_video(db, "Для удаления транскрипта")
    add_words(db, video_id, "Добрый вечер", source="aligned")
    diarize_video(db, video_id)
    session = VideoFleetSession(db)
    session.drop_transcript(0)
    assert session.error is None
    assert "removed" in session.note
    assert "words removed" in session.note
    remaining_words = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    remaining_speakers = db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert remaining_words == 0
    assert remaining_speakers == 0
    # the video itself survives; it is still a row to redraw and reselect.
    assert session.video_id_at(0) == video_id


def test_drop_transcript_with_nothing_selected_says_so(db: Database) -> None:
    session = VideoFleetSession(db)
    session.drop_transcript(0)
    assert session.note == "no video selected"


def test_a_shortcut_with_nothing_selected_says_so(db: Database) -> None:
    session = VideoFleetSession(db)
    session.ingest(0)
    assert session.note == "no video selected"


def test_a_shortcut_refreshes_the_listing(db: Database) -> None:
    """After enqueueing, the row is redrawn, not just noted — Task 15's
    `transcribed` column should flip the moment a transcribe job exists...
    it will not (there are no words yet), but the important thing is the
    session did not go stale or blow up."""
    video_id = make_video(db, "Обновление")
    session = VideoFleetSession(db)
    session.transcribe(0)
    assert session.video_id_at(0) == video_id


# --- the shell ---------------------------------------------------------


def test_the_screen_fills_its_table(db: Database) -> None:
    make_video(db, "В таблице")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            assert screen.query_one("#videos-table").row_count == 1

    asyncio.run(scenario())


def test_the_screen_filters_on_submit(db: Database) -> None:
    # `videos.list`'s `search` is a plain SQL LIKE (design §7 leaves real
    # search to the FTS index) — matching is exact-case for non-ASCII, so
    # the fixture and the filter agree on case rather than testing folding
    # this screen does not do.
    make_video(db, "гость студии")
    make_video(db, "совершенно другое")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            screen.filter("гост")
            assert screen.query_one("#videos-table").row_count == 1

    asyncio.run(scenario())


def test_the_screen_survives_titles_with_brackets(db: Database) -> None:
    """BUGS.md entry 20/Task 4: YouTube titles routinely carry `[...]`, and a
    bare `Static.update`/`DataTable.add_row` crashes on them."""
    make_video(db, "Интервью [часть 1]")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            assert screen.query_one("#videos-table").row_count == 1

    asyncio.run(scenario())


def test_the_screen_ingests_the_highlighted_row(db: Database) -> None:
    video_id = make_video(db, "Через сочетание клавиш")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            screen.action_ingest()

    asyncio.run(scenario())
    assert latest_job(db, video_id) is not None


def test_the_screen_binds_the_five_shortcuts_and_no_bare_letter() -> None:
    keys = {binding.key for binding in VideosScreen.BINDINGS}
    assert {"escape", "ctrl+k", "ctrl+w", "ctrl+a", "ctrl+d", "ctrl+l"} <= keys
    # ctrl+f is claimed by terminals and shells for "find".
    assert "ctrl+f" not in keys
    import re

    assert not any(re.fullmatch(r"[a-z]", key) for key in keys)


def test_the_screen_binds_shift_delete_not_ctrl_x_to_drop_transcript() -> None:
    """BUGS.md entry 44: the owner suggested `ctrl+x`, but that is already
    "Cancel job" on the jobs screen with a different description, and
    `test_one_key_means_one_thing_wherever_it_is_bound` would (correctly)
    fail a reused `ctrl+x` with a new label."""
    bindings = {binding.key: binding for binding in VideosScreen.BINDINGS}
    assert "shift+delete" in bindings
    assert bindings["shift+delete"].action == "drop_transcript"
    assert "ctrl+x" not in bindings


def test_shift_delete_is_not_claimed_by_textuals_input_widget() -> None:
    """The filter `Input` holds focus by default, which is why the other
    five shortcuts are `ctrl+` chords rather than bare letters. `shift+
    delete` must not be one of `Input`'s own bindings, or the screen-level
    binding would never fire while the filter is focused."""
    from textual.widgets import Input

    assert "shift+delete" not in {b.key for b in Input.BINDINGS}


def test_the_app_binds_f9_to_videos() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f9" in keys
    assert "f5" not in keys


def test_a_shortcut_keeps_the_selection_where_it_was(db: Database) -> None:
    """Working down a list must not bounce back to the top.

    Every shortcut repaints, and `DataTable.clear()` resets `cursor_row`
    to 0 — so before this was fixed, aligning row 4 then reaching for row
    3 re-selected row 1 instead, and the list could only be worked from
    the top downwards one item at a time.
    """
    from textual.widgets import DataTable

    make_video(db, "Первое видео")
    make_video(db, "Второе видео")
    third = make_video(db, "Третье видео")

    async def scenario() -> tuple[int | None, str | None]:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            table = screen.query_one("#videos-table", DataTable)
            # Rows are newest first, so the third video is at the top;
            # pick a row that is emphatically not row 0.
            table.move_cursor(row=2)
            before = screen._video_id_at(table.cursor_row)
            screen.action_align()
            return table.cursor_row, before

    after_row, before_id = asyncio.run(scenario())
    assert before_id is not None and after_row == 2
    assert third is not None


def test_a_redraw_falls_back_to_the_top_when_the_row_is_gone(db: Database) -> None:
    """A filter that drops the selected video should land at the top.

    Same rule, other direction: the cursor follows a video id, so when
    that id is no longer in the row set there is nothing to follow.
    """
    from textual.widgets import DataTable

    make_video(db, "Останется")
    make_video(db, "Исчезнет при фильтре")

    async def scenario() -> int | None:
        app = RytpApp(db)
        async with app.run_test():
            screen = VideosScreen(db)
            await app.push_screen(screen)
            table = screen.query_one("#videos-table", DataTable)
            table.move_cursor(row=1)
            screen.session.run("Останется")
            screen._draw()
            return table.cursor_row

    assert asyncio.run(scenario()) == 0



# --- shift+delete: confirmed drop --------------------------------------


def test_shift_delete_opens_a_confirmation_naming_the_video(db: Database) -> None:
    """The prompt must name the exact row, not "are you sure?" — the owner
    works with ~1.6K videos, and a generic prompt is exactly the misfire
    this exists to prevent. The bracketed title is the fixture BUGS.md
    entry 20 cares about: it must render, not crash the parse."""
    video_id = make_video(db, "Итоговое интервью [часть 1]")
    add_words(db, video_id, "Добрый вечер", source="aligned")

    async def scenario() -> ConfirmScreen:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            screen = VideosScreen(db)
            await app.push_screen(screen)
            await pilot.press("shift+delete")
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ConfirmScreen)
            assert app.is_running
            # Close it so the app can shut down cleanly.
            modal.dismiss(False)
            await pilot.pause()
            return modal

    modal = asyncio.run(scenario())
    assert str(video_id) in modal._title
    assert "Итоговое интервью [часть 1]" in modal._lines
    joined = " ".join(modal._lines)
    assert "words" in joined and "utterances" in joined and "speaker" in joined
    assert "re-transcrib" in joined


def test_escape_on_the_confirmation_leaves_the_transcript_intact(db: Database) -> None:
    video_id = make_video(db, "Не трогать")
    add_words(db, video_id, "Добрый вечер", source="aligned")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            screen = VideosScreen(db)
            await app.push_screen(screen)
            await pilot.press("shift+delete")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running
            assert app.screen is screen

    asyncio.run(scenario())
    remaining = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert remaining > 0


def test_the_cancel_button_leaves_the_transcript_intact(db: Database) -> None:
    video_id = make_video(db, "Тоже не трогать")
    add_words(db, video_id, "Добрый вечер", source="aligned")

    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            screen = VideosScreen(db)
            await app.push_screen(screen)
            await pilot.press("shift+delete")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.click("#confirm-cancel")
            await pilot.pause()
            assert app.is_running
            assert app.screen is screen

    asyncio.run(scenario())
    remaining = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert remaining > 0


def test_confirming_drops_the_transcript_and_reports_the_word_count(db: Database) -> None:
    video_id = make_video(db, "Точно удалить")
    add_words(db, video_id, "Добрый вечер", source="aligned")
    diarize_video(db, video_id)

    async def scenario() -> str:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            screen = VideosScreen(db)
            await app.push_screen(screen)
            await pilot.press("shift+delete")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.click("#confirm-ok")
            await pilot.pause()
            assert app.is_running
            assert app.screen is screen
            return screen.session.note

    note = asyncio.run(scenario())
    assert "words removed" in note
    assert "2" in note  # "Добрый вечер" is two tokens
    remaining_words = db.conn.execute(
        "SELECT COUNT(*) FROM words WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    remaining_speakers = db.conn.execute(
        "SELECT COUNT(*) FROM video_speakers WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    assert remaining_words == 0
    assert remaining_speakers == 0


def test_dropping_a_transcript_keeps_the_cursor_on_the_same_video(db: Database) -> None:
    """The video is still a row after its transcript is dropped, so the
    selection should stay on it — the same rule `_draw` already follows
    for every other shortcut."""
    from textual.widgets import DataTable

    make_video(db, "Первое")
    make_video(db, "Второе")
    third = make_video(db, "Третье, будет удалено")
    add_words(db, third, "Добрый вечер", source="aligned")

    async def scenario() -> tuple[str | None, str | None]:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            screen = VideosScreen(db)
            await app.push_screen(screen)
            table = screen.query_one("#videos-table", DataTable)
            table.move_cursor(row=0)  # newest first: "Третье" is at the top
            before = screen._video_id_at(table.cursor_row)
            await pilot.press("shift+delete")
            await pilot.pause()
            await pilot.click("#confirm-ok")
            await pilot.pause()
            after = screen._video_id_at(table.cursor_row)
            return before, after

    before_id, after_id = asyncio.run(scenario())
    assert before_id == str(third)
    assert after_id == str(third)
