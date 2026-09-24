"""BUGS.md entry 19: the TUI never says what `caption`, `timed` and `aligned`
mean.

F1 is the obvious home for the glossary because it already shows every
binding (`rytp/tui/screens/help.py`). The search screen's cuttable filter
(Ctrl+T) is the other place the vocabulary matters, so it carries a one-line
hint too (`rytp/tui/screens/search.py`).
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from rytp.db import Database
from rytp.tui.app import RytpApp
from rytp.tui.screens.help import GLOSSARY, HelpScreen
from rytp.tui.screens.search import CUTTABLE_HINT, SearchScreen

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELP_MODULE = _REPO_ROOT / "rytp" / "tui" / "screens" / "help.py"


# -- the glossary text itself -------------------------------------------


def test_the_glossary_names_all_three_tiers_and_the_word_cuttable() -> None:
    for tier in ("caption", "timed", "aligned"):
        assert tier in GLOSSARY
    assert "cuttable" in GLOSSARY.lower()


def _tier_line(tier: str) -> str:
    """The glossary's own line for `tier`, so a test can check its wording
    without also matching an unrelated line that happens to share a word."""
    (line,) = (
        line for line in GLOSSARY.split("\n") if line.startswith(f"{tier} —")
    )
    return line


def test_the_glossary_says_caption_is_never_cuttable() -> None:
    assert "never cuttable" in _tier_line("caption")


def test_the_glossary_says_aligned_is_the_only_cuttable_tier() -> None:
    assert "only tier that can be cut" in _tier_line("aligned")


def test_the_glossary_names_the_allow_timed_override() -> None:
    """A user cannot choose `--allow-timed` (Task 6) without understanding
    the tiers it lets them override, so the flag is named right here."""
    assert "--allow-timed" in GLOSSARY


# -- the help screen shows it live ---------------------------------------


def test_f1_shows_the_glossary(db: Database) -> None:
    async def scenario() -> None:
        app = RytpApp(db)
        async with app.run_test():
            screen = HelpScreen()
            await app.push_screen(screen)
            from textual.widgets import Static

            content = str(screen.query_one("#help-glossary", Static).content)
            for tier in ("caption", "timed", "aligned"):
                assert tier in content
            assert "cuttable" in content.lower()

    asyncio.run(scenario())


def test_the_app_binds_a_key_to_help() -> None:
    keys = {binding.key for binding in RytpApp.BINDINGS}
    assert "f1" in keys


# -- the search screen's cuttable filter carries a hint -------------------


def test_the_cuttable_hint_mentions_the_aligned_tier_and_the_override() -> None:
    assert "aligned" in CUTTABLE_HINT
    assert "--allow-timed" in CUTTABLE_HINT


def test_the_cuttable_binding_carries_the_hint_as_a_tooltip() -> None:
    bindings = {binding.key: binding for binding in SearchScreen.BINDINGS}
    assert bindings["ctrl+t"].tooltip == CUTTABLE_HINT


def test_the_status_line_carries_the_hint_once_cuttable_is_on(db: Database) -> None:
    from tests.test_index_search import corpus

    async def scenario() -> None:
        corpus(db, "Добрый вечер")
        app = RytpApp(db)
        async with app.run_test():
            screen = SearchScreen(db)
            await app.push_screen(screen)
            screen.search("добрый вечер")
            screen.action_toggle_cuttable()
            status = str(screen.query_one("#search-status").content)
            assert "cuttable" in status
            assert "aligned" in status
            assert "--allow-timed" in status

    asyncio.run(scenario())


# -- no bare Static.update in help.py (mirrors Task 4's sweep) -----------


def test_help_screen_has_no_bare_static_update() -> None:
    """`help.py` is not in Task 4's `_OWNED_MODULES` list, but the glossary
    added here still must go through `set_text`, so this task checks its own
    file the same way."""
    source = _HELP_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_HELP_MODULE))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "query_one"
        ):
            raise AssertionError("a Static widget's `.update(` is called directly")
