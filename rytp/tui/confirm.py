"""A reusable "are you sure" modal for destructive actions (BUGS.md entry 44).

Nothing in `rytp/tui/` confirmed anything before this. The Videos screen's
`shift+delete` (drop a transcript) is the first caller, but `speakers.remove`,
`assemble.remove` and `render.remove` are all just as destructive and will
want the identical shape, so this lives on its own rather than welded into
one screen.

Usage, from an async screen action::

    confirmed = await self.app.push_screen_wait(
        ConfirmScreen(
            "Drop the transcript of video 42?",
            ('"Interview" [part 1]',
             "",
             "This deletes its words, utterances and speaker labels.",
             "Recovering means re-transcribing the video from scratch."),
            confirm_label="Drop transcript",
        )
    )
    if confirmed:
        ...

`ConfirmScreen` never calls anything itself — it only ever `dismiss`es with
`True` (the destructive button was pressed) or `False` (everything else:
Escape, the Cancel button, or a click outside the dialog). Defaulting to
cancel is deliberate: the Cancel button holds focus on mount, so a stray
Enter — the most likely accidental keypress — resolves to `False`.

Every line of text goes through `rytp.tui.text.set_text`, never a bare
`Static(...)`/`Static.update(...)`: a confirmation that names a video's
title is exactly where a Russian title carrying `[` or `]` would otherwise
crash the app (BUGS.md entry 20).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from rytp.tui.text import set_text

__all__ = ["ConfirmScreen"]


class ConfirmScreen(ModalScreen[bool]):
    """Ask before a destructive action; resolves to a plain `bool`.

    `title` is the one-line question; `lines` are the specifics — what,
    exactly, is about to be destroyed, and what recovering costs. Callers
    are expected to name the thing by its id and title, never "are you
    sure?" alone (BUGS.md entry 44: a generic prompt on the wrong one of
    ~1.6K rows is the misfire this exists to prevent).
    """

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 70%;
        max-width: 100;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #confirm-body {
        margin-bottom: 1;
    }
    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    #confirm-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        title: str,
        lines: Sequence[str],
        *,
        confirm_label: str = "Delete",
        cancel_label: str = "Cancel",
    ) -> None:
        super().__init__()
        self._title = title
        self._lines = tuple(lines)
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button(self._cancel_label, id="confirm-cancel", variant="primary")
                yield Button(self._confirm_label, id="confirm-ok", variant="error")

    def on_mount(self) -> None:
        set_text(
            self.query_one("#confirm-body", Static),
            "\n".join((self._title, "", *self._lines)),
        )
        self.query_one("#confirm-cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "confirm-ok")

    def action_cancel(self) -> None:
        self.dismiss(False)
