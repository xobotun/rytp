"""The footer fits an 80-column terminal on every screen (new cleanup item 5).

BUGS.md entry 29 measured the cut-list screen's own eleven keys and found
its footer ran off the end of an 80-column line. That screen's own
contribution was trimmed (`rytp/tui/screens/cutlist.py`), but a second,
separate overflow was missed: `rytp/tui/navigation.py`'s F1-F9 strip plus
Textual's own Ctrl+P command-palette key are bound at *app* level, so they
are added to every screen's footer regardless of what that screen binds
itself. Measured in the running app before this fix: the home screen's
footer alone (no screen pushed) was already 116 columns wide at 80x24, and
the cut-list screen on top of it ran to 167. An isolated measurement of a
single screen's own bindings — which is what an earlier pass checked — never
sees this, because the app-level contribution is invisible from inside one
screen's `BINDINGS` list.

The fix keeps every key bound and reachable (F1 still lists all of them,
and `home_strip()` names them on the home view) but shows only F1 and
Ctrl+Q in the footer itself; screen-level keys stack on top of that, not
replace it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Footer
from textual.widgets._footer import FooterKey

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistParams,
    Slot,
    Substitution,
    cutlist_path,
    write_cutlist,
)
from rytp.db import Database
from rytp.tui.app import RytpApp

CREATED = "2026-09-21T09:00:00+00:00"
PARAMS = CutlistParams(
    consistency=0.25, seed=0, pad_ms=0, speaker="", exclude=(), min_align_score=0.0
)


def _sample_cutlist() -> CutList:
    """One fragment with an alternative, one gap with a substitution.

    Enough shape to push `CutlistScreen` — the screen with the most
    bindings by some margin (BUGS.md entry 29) — and measure its footer for
    real rather than against a lighter screen.
    """
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


def _footer_width(footer: Footer) -> int:
    return sum(key.outer_size.width for key in footer.query(FooterKey))


def test_the_home_screen_footer_fits_80_columns(db: Database) -> None:
    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            width = _footer_width(app.screen.query_one(Footer))
            assert width <= 80, f"home footer is {width} columns wide"

    asyncio.run(main())


def test_the_cutlist_screen_footer_fits_80_columns(db: Database, tmp_path: Path) -> None:
    """The busiest screen (entry 29): its own keys plus the app-level strip."""
    written = write_cutlist(_sample_cutlist(), cutlist_path("demo"))

    async def main() -> None:
        app = RytpApp(db)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            from rytp.tui.screens.cutlist import CutlistScreen

            app.push_screen(CutlistScreen(db, written))
            await pilot.pause()
            width = _footer_width(app.screen.query_one(Footer))
            assert width <= 80, f"cut-list footer is {width} columns wide"

    asyncio.run(main())


def test_only_help_and_quit_are_shown_at_app_level(db: Database) -> None:
    """Every app-level key still works (F1 lists them, `home_strip()` names
    them); only F1 and Ctrl+Q cost a column of the footer itself."""
    from rytp.tui.app import RytpApp as App
    from rytp.tui.navigation import binding_rows

    shown = {b.key for b in App.BINDINGS if b.show}  # type: ignore[union-attr]
    hidden = {b.key for b in App.BINDINGS if not b.show}  # type: ignore[union-attr]
    assert shown == {"f1", "ctrl+q"}
    # Every screen key from the navigation table is still bound, just hidden.
    all_keys = {key for key, _action, _description in binding_rows()}
    assert (all_keys - {"f1", "ctrl+q"}) <= hidden


def test_the_command_palette_overlay_is_disabled(db: Database) -> None:
    """Textual's own Ctrl+P command palette is a second, unused command
    surface — this project has its own (`rytp/tui/palette.py`) — and left
    enabled it adds a Ctrl+P key to every screen's footer for nothing."""
    assert RytpApp.ENABLE_COMMAND_PALETTE is False
