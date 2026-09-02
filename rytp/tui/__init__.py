"""rytp TUI — interactive textual app.

The TUI surfaces four screens (DESIGN §3):

* videos  — browse the ``videos`` table.
* queue   — see pending / running / failed downloads.
* transcript — page through a video's ``words`` table.
* speakers  — two-pane mapper (raw diarizer labels ↔ canonical roster).

The textual dependency is **optional**: importing :mod:`rytp.tui.app`
fails clearly if ``textual`` is missing. The headless mapper logic in
:mod:`rytp.speakers` works without textual.
"""