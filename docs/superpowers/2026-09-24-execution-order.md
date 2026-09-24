# Execution order — dependency graph of the eight plans

Derived from each plan's own "Consumes / Prerequisites" section, not guessed.
Task counts are the `### Task N` headings in the repo copies of the plans
(the `~/.cache/rytp-plans` mirror is stale as of 2026-09-23 — do not read it).

| Part | Tasks | Needs |
|---|---|---|
| 1 Foundation | 16 | — |
| 2 Acquisition | 14 | 1 |
| 3 Transcription | 21 | 1, 2 (constants from Task 1, `wav_path` at Task 17, jobs at Task 19) |
| 4 Index & search | 11 | 1, 2, 3 |
| 5 Assembly | 12 | 1 |
| 6 Render | 10 | 1, 2, and 5 at two lazy import sites |
| 7 Speakers | 16 | 1, 2, 3, and 4 for Task 7 only |
| 8 Integration | 13 | 1–7, as code |

```
        ┌──────────────── 5 Assembly ──────────┐
        │                                      │
1 ──────┤                                      ├── 6 Render ──┐
        │                                      │              │
        └── 2 Acquire ──┬── 3 Transcribe ──┬───┘              ├── 8 Integration
                        │                  │                  │
                        └──────────────────┴── 4 Index ── 7 Speakers
```

Critical path: **1 → 2 → 3 → 4 → 7 → 8** = 91 of 113 tasks.
Everything else (5, 6) hangs off it and cannot shorten it.

## Waves

| Wave | Parts in flight | Why they can share a wave |
|---|---|---|
| A | 1 | Everything imports it. Nothing else can start. |
| B | 2, 5 | Both need only Part 1. Disjoint packages (`acquire`/`jobs`/`audio` vs `assemble`). |
| C | 3, 6 | 3 needs 2; 6 needs 2 and 5. Disjoint packages (`transcribe` vs `render`). |
| D | 4 | Needs 3. |
| E | 7 | Needs 3 and 4. |
| F | 8 | Needs all of them on disk. |

Max real concurrency is **two**. A wider fan-out would be agents waiting on
imports that do not exist yet.

## Shared files two parallel agents both touch

The plans were written append-only for exactly this reason:

- `rytp/constants.py` — each part appends one section at the very end, never
  reorders what is above it.
- `rytp/jobs/__init__.py` — each part appends its handler thunk and its
  `register_job_kind(...)` call at the bottom.
- `rytp/commands/__init__.py` — each part adds one line to the sibling-import
  block at the end.
- `rytp/db/schema.py` — migrations append from 12 upward, never renumbered.

Same-wave agents work in separate git worktrees and merge on completion, so an
append collision is a two-line conflict at a known place rather than two agents
writing the same file at the same instant.

## Working location

`~/xobotop_rytp_local` — a local clone of the SMB checkout. All agent work
happens here on local disk; the SMB share is read once and written once.
