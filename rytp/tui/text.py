"""Markup-safe text for `Static` widgets (BUGS.md entry 20).

`Static.update` parses whatever string it is handed as Textual console
markup. Usage lines spell optional parameters like `[title=…]`, and result
messages can contain anything a handler chose to say, so passing either
straight through crashes the app the moment a bracket appears:

    MarkupError: Expected markup value (found '…] [kind=…] [channel=…]').

Rich's own markup parser tolerates such strings; Textual 8 ships a
stricter one, which is what actually raises. `set_text` is the one place
that renders plain text so no call site has to remember this.

Two fixes were verified by hand (`BUGS.md` entry 20): escaping the string
with `textual.markup.escape` before `update`, or constructing
`textual.content.Content` directly, which never parses markup at all.
`Content` is preferred here: escaping only defeats the parser this widget
happens to use today, whereas `Content(text)` states outright that the
text is plain and is never subject to markup, whatever renders it.
"""

from __future__ import annotations

from collections.abc import Iterable

from rich.text import Text
from textual.content import Content
from textual.widgets import Static

__all__ = ["plain_row", "set_text"]


def set_text(widget: Static, text: str) -> None:
    """Update `widget` with `text` shown verbatim, never as markup."""
    widget.update(Content(text))


def plain_row(cells: Iterable[str]) -> tuple[Text, ...]:
    """Wrap `cells` for `DataTable.add_row` so none of them is markup.

    `DataTable`'s own cell formatter (`textual.widgets._data_table.
    default_cell_formatter`) runs every plain `str` cell through Rich's
    `Text.from_markup` — a *different*, more permissive parser than the one
    `Static` uses, but still not a no-op: `[bracketed] title.mp4` silently
    loses `[bracketed]`, and a closing-tag shape like `clip [/take2].mp4`
    raises `rich.errors.MarkupError` outright. A row or a usage line in a
    cell is exactly the content this project puts there (paths, titles,
    error text), so cells need the same treatment `set_text` gives `Static`.

    Passing a `Text` object instead of a bare `str` skips the formatter's
    markup branch entirely (it only parses `str`), so this is the cheapest
    fix: build the `Text` here, once, instead of taking it on faith at every
    `add_row` call site.
    """
    return tuple(Text(cell, end="") for cell in cells)
