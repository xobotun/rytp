# Settings: deeper surfacing (not built yet)

`rytp/commands/settings.py` (2026-09-24) gives `settings` full CRUD —
`list` / `get` / `set` / `unset` — reachable from the CLI and the palette,
which is enough to unblock pointing `engine.interpreter.<name>` at a
working Python. This is a plan for the surfacing work deliberately left
out of that change. Not a spec; revisit before building any of it.

## 1. `--help` text

`rytp/cli.py:218` gives every group's Typer sub-app
`help=f"Commands in the {cmd.group} group."` — a template, not a
description. The owner has separately asked that group headings say what
the group is *about*, plus list its commands, e.g. for `settings`:

> Read and write operational settings — engine interpreters, default
> transcriber/aligner. Commands: list, get, set, unset.

That needs a per-group description string somewhere both surfaces can
read — `Command` has no group-level field today, only `Command.group` as
a bare tag, so this is a small contracts change (a `GROUP_SUMMARIES: dict[str,
str]` next to `COMMANDS`, or a `group_summary` on the first command
registered in a group) rather than a one-line fix. Worth doing for every
group at once, not just `settings`, since the current text is equally
thin everywhere. Out of scope for the settings CRUD change; raised as a
follow-up because `settings` is a small, low-risk group to pilot it on.

## 2. A TUI settings screen — or not

Against a dedicated screen: `settings` is two-column key/value data with
no navigation, no children, and no per-row action beyond edit/delete —
exactly what the palette already does well (`settings set <key> <value>`
typed as one line). Part 8's `SCREENS` tuple is for the five things
design §10 calls out as needing dedicated navigation (speakers, search,
transcripts, queue, cut lists); settings resembles `doctor` more than any
of those — a flat report/edit surface, not a browsing task.

For a screen: `speakers.map` shows the palette-vs-screen line isn't
purely about data shape — it's about whether picking the *value* is
itself interactive. Editing `engine.interpreter.mfa` by typing a path
blind is exactly the kind of thing a screen could improve (a file
picker, live validation — see §3). If that affordance is built, it
earns the screen `settings` alone wouldn't.

**Recommendation:** don't add a screen for `settings.list` /`get`/`set`/
`unset` as they stand. Revisit only if §3's affordances are built and
make a blind `--value` genuinely worse than a picker.

**F5** is confirmed unused at the app level — `rytp/tui/navigation.py`
skips it in `SCREENS` deliberately, reserved because the speakers
mapper binds it locally (`toggle_suggestions`) and a global and a local
`f5` would collide. It is *not* free for a new top-level screen without
first checking that collision test
(`rytp/tui/navigation.py` docstring, `tests/test_consistency_flags.py`
territory) still passes — likely fine since the mapper's `f5` is scoped
to its own screen, but that's a five-minute check, not an assumption.

## 3. Should `engine.interpreter.*` get its own affordances?

Two things a generic key/value editor cannot do that this key
specifically would benefit from:

- **Path existence.** `settings set` currently accepts any string.
  `interpreter_for` (`rytp/transcribe/registry.py`) falls back silently
  to `sys.executable` only when the *setting is unset*, not when it's
  set to a path that doesn't exist — a typo here fails at engine-launch
  time (`rytp/transcribe/subproc.py:95`, `"engine interpreter not
  found"`), which is late and unclear in context.
- **Import probe.** Checking the path exists is necessary but not
  sufficient — `<path>/bin/python -c "import torch"` (or the specific
  module `engine_remedy` in `rytp/transcribe/health.py` names) is the
  real question, and it's exactly what `doctor`'s engine checks already
  do for the *current* interpreter (`rytp/transcribe/health.py`,
  `rytp/diarize/health.py`).

Both are validation, not storage — they belong as a warning
(`CommandResult.message`, non-fatal) on `settings set` when the key
matches `engine.interpreter.*`, not as a blocking rule: contracts §5's
generic settings need to stay writable even when nothing can currently
be probed (no venv yet, developing on a machine other than the target).
A blocking check would also fight the case this feature exists for —
setting the path *before* the venv is finished being built.

Concretely: `settings_set` special-cases the `engine.interpreter.`
prefix, calls the same probe `doctor` already runs
(`rytp.transcribe.health` / `rytp.diarize.health`), and appends the
result to the success message rather than raising. Small, additive,
doesn't change the return shape, and reuses code instead of duplicating
the interpreter-probing logic that already exists for `doctor`.
