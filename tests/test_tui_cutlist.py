"""Editing a cut list without leaving the tool (design §8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rytp import config
from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistParams,
    Slot,
    Substitution,
    load_cutlist,
    write_cutlist,
)
from rytp.audio.extract import wav_path
from rytp.tui.cutlist_view import CutlistView, available_cutlists
from tests.synth_audio import tone_gap_tone, write_wav

CREATED = "2026-09-21T09:00:00+00:00"
PARAMS = CutlistParams(
    consistency=0.25, seed=0, pad_ms=0, speaker="", exclude=(), min_align_score=0.0
)


def sample() -> CutList:
    """One fragment with an alternative, one gap with a substitution."""
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="demo",
        target="мы всё исправим",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=1,
                text="мы всё",
                video_id=3,
                first_word_ord=1204,
                last_word_ord=1205,
                start_ms=612_340,
                end_ms=613_100,
                align_score=0.81,
                cost=1.42,
                video_speaker_id=11,
                speaker_label="Ведущий",
                alternatives=(
                    Alternative(
                        video_id=7,
                        first_word_ord=88,
                        last_word_ord=89,
                        start_ms=10_500,
                        end_ms=11_220,
                        text="мы всё",
                        cost=1.77,
                    ),
                ),
            ),
            Slot(
                kind="gap",
                target_first=2,
                target_last=2,
                text="исправим",
                substitutions=(
                    Substitution(
                        text="исправит",
                        reason="edit",
                        distance=1,
                        occurrences=3,
                        video_id=9,
                        first_word_ord=502,
                        last_word_ord=502,
                        start_ms=220_100,
                        end_ms=220_780,
                    ),
                ),
            ),
        ),
    )


@pytest.fixture()
def written(data_dir: object, tmp_path: Path) -> Path:
    from rytp.assemble.cutlist import cutlist_path

    return write_cutlist(sample(), cutlist_path("demo"))


def test_the_picker_lists_what_is_on_disk(written: Path) -> None:
    """The mapper has `SpeakerVideosScreen` for the same reason: nobody should
    have to remember a name to edit the thing they just planned."""
    listed = available_cutlists()
    assert [item.name for item in listed] == ["demo"]
    assert listed[0].target == "мы всё исправим"
    assert listed[0].gaps == 1
    assert listed[0].sources == 1
    assert listed[0].error is None


def test_a_broken_file_is_listed_with_its_complaint_not_hidden(
    written: Path, data_dir: object
) -> None:
    """A cut list the owner mistyped is exactly the one he needs to find."""
    written.with_name("broken.toml").write_text("this is not toml [", encoding="utf-8")
    broken = next(item for item in available_cutlists() if item.name == "broken")
    assert broken.error is not None
    assert broken.slots == 0


def test_the_table_is_the_timeline_in_document_order(written: Path) -> None:
    """design §8: one ordered slot array, "document order is the timeline"."""
    view = CutlistView(written)
    assert view.columns == (
        "#", "kind", "text", "source", "in", "out", "gap", "alts",
    )
    assert [row[1] for row in view.rows] == ["fragment", "gap"]
    assert view.rows[0][3] == "v3"
    assert view.rows[1][3] == C.NULL_CELL


def test_the_options_of_a_fragment_are_its_ranked_alternatives(written: Path) -> None:
    view = CutlistView(written)
    assert view.option_columns(0) == ("#", "text", "source", "in", "out", "cost")
    assert view.option_rows(0)[0][2] == "v7"


def test_the_options_of_a_gap_are_its_ranked_substitutions(written: Path) -> None:
    """design §8: "Offer ranked substitutions… but never apply one silently"."""
    view = CutlistView(written)
    assert view.option_columns(1) == ("#", "text", "reason", "source", "in", "out")
    assert view.option_rows(1)[0][1] == "исправит"
    assert view.option_rows(1)[0][2] == "edit"


def test_swapping_promotes_the_alternative_and_keeps_the_old_choice(
    written: Path,
) -> None:
    """Reversible by construction: the displaced fragment becomes an
    alternative in the slot it was displaced from."""
    view = CutlistView(written)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (7, 10_500, 11_220)
    assert [alt.video_id for alt in slot.alternatives] == [3]
    assert view.dirty is True


def test_swapping_twice_returns_to_where_it_started(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (3, 612_340, 613_100)


def test_a_promoted_alternative_admits_it_does_not_know_the_speaker(
    written: Path,
) -> None:
    """The one lossy edge, stated rather than papered over. `Alternative` has
    no `align_score`, `video_speaker_id` or `speaker_label`, so keeping the
    displaced fragment's would claim that a clip from video 7 was said by the
    person identified in video 3."""
    view = CutlistView(written)
    view.swap(0, 0)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.align_score is None
    assert slot.video_speaker_id is None
    assert slot.speaker_label is None
    assert "speaker" in view.status.lower()


def test_adopting_a_substitution_turns_a_gap_into_a_fragment(written: Path) -> None:
    view = CutlistView(written)
    view.swap(1, 0)
    slot = view.slot_at(1)
    assert slot is not None
    assert slot.kind == "fragment"
    assert slot.text == "исправит"
    assert (slot.video_id, slot.start_ms, slot.end_ms) == (9, 220_100, 220_780)
    assert slot.substitutions, "the ranked list survives so the choice can change"


def test_nudging_moves_one_boundary_by_one_frame(written: Path) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="start", delta_ms=-C.TUI_CUTLIST_NUDGE_MS)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 612_340 - C.TUI_CUTLIST_NUDGE_MS
    assert slot.end_ms == 613_100


def test_a_nudge_may_not_turn_a_fragment_inside_out(written: Path) -> None:
    view = CutlistView(written)
    message = view.nudge(0, edge="start", delta_ms=10_000)
    assert "shorter than" in message or "would not leave" in message
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 612_340


def test_a_nudge_may_not_move_a_boundary_before_the_start_of_the_video(
    written: Path,
) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="start", delta_ms=-999_999)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 0


def test_a_gap_slot_has_no_boundary_to_nudge(written: Path) -> None:
    view = CutlistView(written)
    assert "gap" in view.nudge(1, edge="start", delta_ms=40).lower()


def test_the_shift_step_is_the_coarse_constant(written: Path) -> None:
    """BUGS.md entry 30: Shift is coarser, not finer — a wrong fragment is
    usually wrong by a syllable, not by a millisecond."""
    view = CutlistView(written)
    view.nudge(0, edge="start", delta_ms=-C.TUI_CUTLIST_SHIFT_NUDGE_MS)
    slot = view.slot_at(0)
    assert slot is not None
    assert slot.start_ms == 612_340 - C.TUI_CUTLIST_SHIFT_NUDGE_MS


# --- the snap-to-clean-boundary key (BUGS.md entry 30) -------------------


def wav_for(video_id: int, data_dir: object) -> tuple[int, int]:
    """Write a synthesised WAV to the cache path `snap` reads, and return the
    measured silence's `(gap_start_ms, gap_end_ms)` — the interval a snap is
    required to land in."""
    samples, gap_start, gap_end = tone_gap_tone(lead_ms=200, gap_ms=120, tail_ms=200)
    path = wav_path(video_id)
    config.ensure_dir(path.parent)
    write_wav(path, samples)
    return gap_start, gap_end


def cutlist_with_boundary_near(video_id: int, end_ms: int) -> CutList:
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="snap-demo",
        target="слово",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=0,
                text="слово",
                video_id=video_id,
                first_word_ord=0,
                last_word_ord=0,
                start_ms=0,
                end_ms=end_ms,
            ),
        ),
    )


def test_the_snap_key_moves_the_boundary_into_the_measured_silence(
    data_dir: object, tmp_path: Path
) -> None:
    gap_start, gap_end = wav_for(101, data_dir)
    path = write_cutlist(cutlist_with_boundary_near(101, 230), tmp_path / "snap.toml")
    view = CutlistView(path)
    message = view.snap(0, edge="end")
    slot = view.slot_at(0)
    assert slot is not None
    assert gap_start <= slot.end_ms <= gap_end
    assert "snapped" in message.lower()


def test_snapping_twice_is_a_no_op_the_second_time(
    data_dir: object, tmp_path: Path
) -> None:
    wav_for(102, data_dir)
    path = write_cutlist(cutlist_with_boundary_near(102, 230), tmp_path / "snap2.toml")
    view = CutlistView(path)
    view.snap(0, edge="end")
    moved = view.slot_at(0)
    assert moved is not None
    message = view.snap(0, edge="end")
    assert "already" in message.lower()
    assert view.slot_at(0) == moved


def test_snapping_with_no_cached_audio_says_so_instead_of_crashing(
    data_dir: object, tmp_path: Path
) -> None:
    path = write_cutlist(cutlist_with_boundary_near(999, 230), tmp_path / "snap3.toml")
    view = CutlistView(path)
    message = view.snap(0, edge="end")
    assert "no cached audio" in message.lower()
    slot = view.slot_at(0)
    assert slot is not None and slot.end_ms == 230


def test_snapping_a_gap_has_no_boundary_either(written: Path) -> None:
    view = CutlistView(written)
    assert "gap" in view.snap(1, edge="start").lower()


def test_setting_a_pause_writes_the_key_the_renderer_reads(written: Path) -> None:
    """design §9's gap is "Configurable… Disableable if it doesn't sound
    right", and contracts §7's cut list carries `gap_before_ms` for exactly
    that."""
    view = CutlistView(written)
    view.set_gap_before(1, 320)
    slot = view.slot_at(1)
    assert slot is not None and slot.gap_before_ms == 320
    view.clear_gap(1)
    slot = view.slot_at(1)
    assert slot is not None and slot.gap_before_ms is None


def test_a_negative_pause_is_refused(written: Path) -> None:
    view = CutlistView(written)
    assert "negative" in view.set_gap_before(1, -5).lower()


def test_undo_walks_back_through_every_kind_of_edit(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    view.nudge(0, edge="end", delta_ms=C.TUI_CUTLIST_NUDGE_MS)
    view.set_gap_before(1, 200)
    view.undo()
    view.undo()
    view.undo()
    assert view.cutlist == load_cutlist(written)
    assert view.dirty is False
    assert "nothing to undo" in view.undo().lower()


def test_saving_round_trips_through_the_file_part_five_wrote(written: Path) -> None:
    """The file is the durable representation (design §8), so what the screen
    saves must be what Part 6 can read."""
    view = CutlistView(written)
    view.swap(0, 0)
    view.set_gap_before(1, 250)
    path = view.save()
    assert path == written
    assert view.dirty is False
    reloaded = load_cutlist(path)
    assert reloaded.slots[0].video_id == 7
    assert reloaded.slots[1].gap_before_ms == 250
    assert CutlistView(path).cutlist == reloaded


def test_saving_keeps_the_name_and_the_target(written: Path) -> None:
    view = CutlistView(written)
    view.nudge(0, edge="end", delta_ms=40)
    reloaded = load_cutlist(view.save())
    assert reloaded.name == "demo"
    assert reloaded.target == "мы всё исправим"


def test_reloading_discards_unsaved_edits_and_says_so(written: Path) -> None:
    view = CutlistView(written)
    view.swap(0, 0)
    message = view.reload()
    assert "discarded" in message.lower()
    assert view.dirty is False
    assert view.cutlist == load_cutlist(written)


def test_the_status_line_says_what_will_be_rendered(written: Path) -> None:
    status = CutlistView(written).status
    assert "1 fragment" in status
    assert "1 gap" in status
    assert "unsaved" not in status.lower()


def test_an_index_off_the_end_is_reported_not_raised(written: Path) -> None:
    view = CutlistView(written)
    assert view.slot_at(99) is None
    assert view.option_rows(99) == ()
    assert "no slot" in view.swap(99, 0).lower()


def test_swapping_to_an_option_that_is_not_there_is_reported(written: Path) -> None:
    view = CutlistView(written)
    assert "no alternative" in view.swap(0, 9).lower()


def test_a_fragment_with_no_alternatives_says_so(written: Path, tmp_path: Path) -> None:
    """The realistic case: the assembler found one occurrence and nothing else."""
    bare = CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="bare",
        target="раз",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=0,
                text="раз",
                video_id=1,
                first_word_ord=0,
                last_word_ord=0,
                start_ms=0,
                end_ms=500,
            ),
        ),
    )
    path = write_cutlist(bare, tmp_path / "bare.toml")
    view = CutlistView(path)
    assert view.option_rows(0) == ()
    assert "no alternative" in view.swap(0, 0).lower()


# --- the screens ---------------------------------------------------------

import asyncio  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import Any  # noqa: E402

from textual.app import App, ComposeResult  # noqa: E402
from textual.widgets import DataTable, Footer, Static  # noqa: E402

from rytp.db import Database  # noqa: E402
from rytp.tui.screens.cutlist import CutlistPickerScreen, CutlistScreen  # noqa: E402


def drive(
    screen_factory: Callable[[], Any], body: Callable[[Any, Any], Awaitable[None]]
) -> None:
    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            return iter(())

        def on_mount(self) -> None:
            self.push_screen(screen_factory())

    async def main() -> None:
        app = Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            await body(app.screen, pilot)

    asyncio.run(main())


def test_the_picker_shows_what_is_on_disk(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        table = screen.query_one("#cutlist-picker", DataTable)
        assert table.row_count == 1
        assert "demo" in str(table.get_cell_at((0, 0)))

    drive(lambda: CutlistPickerScreen(db), body)


def test_the_picker_says_so_when_there_is_nothing_to_edit(
    db: Database, data_dir: object
) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        text = str(screen.query_one("#cutlist-picker-status", Static).content)
        assert "assemble plan" in text

    drive(lambda: CutlistPickerScreen(db), body)


def test_the_editor_shows_the_slots_and_the_options_of_the_first(
    db: Database, written: Path
) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        slots = screen.query_one("#cutlist-slots", DataTable)
        options = screen.query_one("#cutlist-options", DataTable)
        assert slots.row_count == 2
        assert options.row_count == 1

    drive(lambda: CutlistScreen(db, written), body)


def test_moving_to_the_gap_shows_its_substitutions(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        screen.query_one("#cutlist-slots", DataTable).move_cursor(row=1)
        await pilot.pause()
        options = screen.query_one("#cutlist-options", DataTable)
        assert [str(c.label) for c in options.columns.values()][2] == "reason"

    drive(lambda: CutlistScreen(db, written), body)


def test_one_key_swaps_the_highlighted_option_in(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.video_id == 7

    drive(lambda: CutlistScreen(db, written), body)


def test_the_bracket_keys_move_the_boundaries(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("[")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.start_ms < 612_340

    drive(lambda: CutlistScreen(db, written), body)


def test_shift_bracket_moves_by_the_coarse_step(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("shift+[")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None
        assert slot.start_ms == 612_340 - C.TUI_CUTLIST_SHIFT_NUDGE_MS

    drive(lambda: CutlistScreen(db, written), body)


def test_ctrl_shift_arrow_moves_the_end_by_the_coarse_step(
    db: Database, written: Path
) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("ctrl+shift+right")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None
        assert slot.end_ms == 613_100 + C.TUI_CUTLIST_SHIFT_NUDGE_MS

    drive(lambda: CutlistScreen(db, written), body)


def test_the_snap_key_reaches_the_view(db: Database, written: Path) -> None:
    """No cached audio for `written`'s video 3, so this exercises the no-op
    path rather than a real snap — the screen-level contract is only that
    `z` reaches `CutlistView.snap`, not the measurement itself (covered at
    the view level with a synthesised waveform)."""

    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("z")
        await pilot.pause()
        status = str(screen.query_one("#cutlist-status", Static).content)
        assert "no cached audio" in status.lower()

    drive(lambda: CutlistScreen(db, written), body)


def test_the_help_key_opens_the_help_screen(db: Database, written: Path) -> None:
    """`?` is this screen's shortcut to the same help screen F1 already
    opens app-wide (`RytpApp.action_help`) — the smallest honest way to point
    at the full binding list once the footer stops showing it.

    Driven through the real `RytpApp`, not the bare `drive()` harness: `?`
    resolves to `app.help`, and only the real app defines `action_help`."""
    from rytp.tui.app import RytpApp
    from rytp.tui.screens.help import HelpScreen

    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(CutlistScreen(db, written))
            await pilot.pause()
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)

    asyncio.run(main())


def test_the_screens_own_footer_keys_fit_an_80_column_terminal(
    db: Database, written: Path
) -> None:
    """BUGS.md entry 29: twelve keys used to run the footer off the end of
    an 80-column line. Most are `show=False` now.

    Measured against this screen's own `BINDINGS` in isolation, on a bare
    harness with none of the app's own bindings — see
    `test_this_screens_own_contribution_to_the_running_apps_footer` below for
    why that isolation matters and what it does not cover.
    """
    from textual.widgets._footer import FooterKey

    async def body(screen: Any, pilot: Any) -> None:
        footer = screen.query_one(Footer)
        keys = list(footer.query(FooterKey))
        assert keys, "the footer should show at least the handful of constant keys"
        total_width = sum(key.outer_size.width for key in keys)
        assert total_width <= 80, f"footer is {total_width} columns wide, keys={keys}"

    drive(lambda: CutlistScreen(db, written), body)


def test_this_screens_own_contribution_to_the_running_apps_footer(
    db: Database, written: Path
) -> None:
    """The bare-harness measurement above is honest about this screen's own
    `BINDINGS`, but the real app's footer also always carries the F1-F8
    navigation strip and the Ctrl+P command-palette key (`rytp/tui/app.py`,
    `binding_rows()` in `rytp/tui/navigation.py`) — neither file is this
    task's to touch, and measured against the real `RytpApp` that baseline
    alone is already wider than 80 columns, before this screen renders
    anything. BUGS.md entry 29 was scoped to *this* screen's twelve keys; the
    app-level strip is a second, separate overflow that Task 13 cannot fix
    from its own files — the QA batch plan hands it to Task 17 ("the footer
    lesson").

    What this task does control, and what this test pins, is that *this*
    screen adds only a small, constant amount on top of whatever the app
    already carries — not the twelve-plus keys' worth it used to.
    """
    from textual.widgets._footer import FooterKey

    from rytp.tui.app import RytpApp

    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            baseline = sum(
                key.outer_size.width
                for key in app.screen.query_one(Footer).query(FooterKey)
            )
            app.push_screen(CutlistScreen(db, written))
            await pilot.pause()
            with_screen = sum(
                key.outer_size.width
                for key in app.screen.query_one(Footer).query(FooterKey)
            )
        added = with_screen - baseline
        assert 0 < added <= 60, f"this screen added {added} footer columns"

    asyncio.run(main())


def test_undo_is_one_key(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.press("u")
        await pilot.pause()
        slot = screen.view.slot_at(0)
        assert slot is not None and slot.video_id == 3

    drive(lambda: CutlistScreen(db, written), body)


def test_saving_writes_the_file(db: Database, written: Path) -> None:
    async def body(screen: Any, pilot: Any) -> None:
        await pilot.press("s")
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert screen.view.dirty is False
        assert load_cutlist(written).slots[0].video_id == 7

    drive(lambda: CutlistScreen(db, written), body)


def test_the_screen_holds_only_its_view(db: Database, written: Path) -> None:
    """Every decision is `CutlistView`'s; the screen keeps two cursors, and
    Textual keeps those in the `DataTable`s. Measured against a bare `Screen`,
    because Textual's own `__init__` sets dozens of attributes."""
    from textual.screen import Screen

    added = set(vars(CutlistScreen(db, written))) - set(vars(Screen()))
    assert added == {"view", "_db"}
