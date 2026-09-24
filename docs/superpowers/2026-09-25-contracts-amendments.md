# Contracts amendments — 2026-09-25 QA fix batch

**Source:** `docs/superpowers/plans/2026-09-25-qa-fix-batch.md`, Task 1. This
document is the justification record for the changes Task 1 made to
`docs/superpowers/specs/2026-09-21-rytp-contracts.md`. The contracts file
itself stays terse; the reasoning lives here.

---

## 1. `words.align_scale`

**Forces the change:** `BUGS.md` entry 13. `words.align_score` has held at
least three incommensurable things — a bounded 0–1 energy measure, an
unbounded log-probability, and nothing — with no column saying which. A user
checking "is this good enough to cut?" had no way to read the number
correctly, because the same column meant different things depending on what
had run.

**What breaks if not made:** Task 5 (writing the scale) and Task 6 (reading it
to fix the entries-26/28 assembly bug) have no column to write to or read
from, and every later task that reasons about score comparability has nothing
to check against. The batch's central fix — a per-scale eligibility threshold
in `assemble plan` — is not expressible without this column.

**Alternative rejected:** normalising every scale into one comparable number.
Rejected because there is no principled conversion between a bounded energy
measure and an unbounded log-probability, and inventing one would silently
bake in a conversion nobody could justify — the same failure mode as entry
28's sign-flip hack, dressed differently. Recording the scale and refusing to
convert is the only choice that does not fabricate certainty.

## 2. `jobs.progress`

**Forces the change:** `BUGS.md` entries 3 and 10. `ingest` and `fetch` sit
silently for minutes with nothing indicating they are working rather than
hung, and this is true for any long-running job drained by the worker.

**What breaks if not made:** Task 8a's progress seam has no place to persist
a queued job's progress for `jobs.list` to display; the TUI and CLI would
still show a running job with no signal of its state.

**Alternative rejected:** overloading `jobs.note` for progress as well as the
handler's final finding. Rejected because the two have incompatible
lifetimes — `note` is written once, after the handler finishes, and survives
completion; a progress line is written many times while the handler runs and
is cleared the moment the job leaves `running`. Conflating them would mean a
job's final, load-bearing note gets overwritten by the last progress tick, or
that progress writes have to dodge the handler's return path. Two columns
with two lifetimes is simpler than one column with two meanings.

## 3. `--allow-timed`, an explicit override on `assemble.plan`

**Forces the change:** design question D1, and `BUGS.md` entry 26's
recommendation. Cutting only `aligned` words is correct by default, but a
corpus that has been transcribed and not yet aligned is otherwise completely
unusable for assembly, with no way to say "I know the boundaries are rough,
cut anyway."

**What breaks if not made:** Task 6 has no contractual name or shape for the
override it must implement, and a later task inventing its own spelling is
exactly the "one flag spelled two ways" failure mode `docs/superpowers/specs/2026-09-21-rytp-contracts.md`
§2 already calls out as the recurring defect this project produces.

**Alternative rejected, twice.** First: a numeric threshold admitting `timed`
words above some cutoff. Rejected because §1a's score table shows `timed`
words can carry a score on *either* of two scales depending on whether
`--refine` ran (`energy`) or not (`NULL`), and a `timed`-tier row aligned by a
future run could in principle carry a third. A threshold written against one
scale is meaningless against another, and the scale a given corpus happens to
have is not fixed — the same reason §3's "Score precedence and scale" gives
for never normalising between scales. Second: spelling the flag `--force`.
Rejected because `assemble.plan` already has a `--force` meaning "replace an
existing cut list"; reusing the name for a second, unrelated meaning is
precisely the ambiguity the flag-vocabulary consistency suite exists to catch.

## 4. `GROUP_SUMMARIES`

**Forces the change:** `BUGS.md` entry 6. The command registry has a
`group` field per command but nothing describing what a group is *about*, so
both surfaces either show an undescribed heading or hand-roll a description
that drifts from the actual command list.

**What breaks if not made:** there is no contractual place for a group
description to live, so a fix would either invent a second, informal
mechanism per surface (reintroducing the "two things claiming one
responsibility" failure) or hardcode a description that goes stale the moment
a command is added or removed from the group — which is the literal defect
entry 6 reports.

**Alternative rejected:** a `group_summary` field repeated on every `Command`
in the group. Rejected because it is one fact about the group repeated on N
commands, which is exactly the shape that drifts: two commands in one group
could disagree about what the group is, and nothing would catch it. A single
`dict[str, str]` keyed by group name has one place to be wrong, and the
consistency suite can assert every non-empty group has an entry.

## 5. Engine resolution error type: `UnknownEngineError(RytpError, ValueError)`

**Forces the change:** `BUGS.md` entry 16. `resolve_transcriber`,
`resolve_aligner` and `resolve_diarizer` already raise `ValueError` per
contracts §6, with an already-correct message — but a bare `ValueError` is not
caught by the CLI's `RytpError` funnel (§8), so a mistyped engine name (e.g.
`--aligner wav2vec` instead of `wav2vec2`) surfaces as a ~60-line traceback
instead of the one-line message the funnel exists to produce.

**Verified resolution:** `UnknownEngineError(RytpError, ValueError)` satisfies
both requirements at once through its MRO — it *is* a `ValueError`, so
contracts §6's existing wording stays literally true and any caller catching
`ValueError` still works, and it *is* a `RytpError`, so `rytp/cli.py`'s
existing funnel catches it with no change to the funnel itself.

**Alternative rejected:** widening the CLI funnel to also catch `ValueError`.
Rejected because that would swallow every incidental `ValueError` a handler
happens to raise for an unrelated reason (a bad int conversion, for instance),
printing it as a clean one-liner and hiding a real bug. Narrowing the raised
type instead of widening the catch keeps the funnel's guarantee — "a
`RytpError` is a deliberate, user-facing failure" — intact.

## 6. The engine notes channel: `notes: list[str]` on the three protocols

**Forces the change:** `BUGS.md` entry 36 (by way of Task 3's aligner fix).
`Wav2Vec2Aligner.align` can now detect a word that received no frames at all
and must fall back to an interpolated boundary — a non-fatal finding worth
surfacing, not a failure — but `Aligner.align` returns `list[Span]` with
nowhere to attach it.

**What breaks if not made:** either the finding is dropped silently (the
exact failure mode the whole batch exists to stop reproducing), or `align`'s
return type is widened to carry it, which changes every caller and every fake
across `rytp/transcribe/pipeline.py`, `tests/fake_engines.py`, and both
aligner adapters — for information that is advisory in the overwhelming
majority of calls.

**Alternative rejected:** widening `Aligner.align`'s return type (e.g. to
`tuple[list[Span], list[str]]`, or wrapping it in a result object). Rejected
per the plan's own instruction — Task 5's file list would have to touch
`registry.py`, `subproc.py` and every fake, for a channel that exists purely
for the rare non-fatal case. An instance attribute, drained by the pipeline
with `getattr(engine, "notes", [])` and cleared between calls, gives the same
information with no signature change anywhere. This mirrors exactly the
reasoning that gave job handlers a return note (§5): an operation that can
only raise or stay silent has nowhere to put "this succeeded, and there is
something you should know."
