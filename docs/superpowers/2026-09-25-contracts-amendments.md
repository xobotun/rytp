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

## 7. `words.orig_start_ms` / `words.orig_end_ms`, and `transcribe.unalign`

**Forces the change:** `BUGS.md` entry 44. `words.start_ms`/`end_ms` are
overwritten in place by both `realign_video` and, when `--refine` is on, by
the energy refiner — so the transcriber's own timing is gone the instant
either runs, and the entry's own proposed fix (relabel `aligned` back to
`timed`, keeping whatever timings happen to be on the row) was only ever a
consolation prize. It made `--allow-timed`'s two rows in "Score precedence
and scale" — a `timed` row with a real energy score vs. one with none —
indistinguishable from a `timed` row that is secretly still carrying aligned
boundaries wearing the wrong label. The owner asked for the real fix:
capture the original at write time, so an alignment is reversible for real.

**What breaks if not made:** `transcribe unalign` (below) has no source of
truth to restore from and degrades back into "relabel and hope," which is
the exact non-fix the entry already rejected.

**The columns.** Two new nullable `INTEGER` columns on `words`, gained by
migration 16, no `CHECK` (same reason as `align_scale`: SQLite cannot add one
via `ALTER TABLE`, and there is nothing to check here anyway — both are
either both `NULL` or both set). Written **once, at INSERT time, by whichever
writer creates the row**, and never touched again — not by `--refine`, not by
`realign_video`, not by anything. That last part is the entire point: a
second alignment's input must never become the new "original," or the
ratchet breaks after one round trip.

**What counts as "original."** The transcriber's own per-token timing,
*before* alignment and *before* `--refine` touches anything — i.e. `RawWord`
after `_expand_tokens`' per-token split, read before `refine_boundaries` ever
runs. Not post-refinement. Three reasons, strongest first:

1. Restoring pre-refinement originals and clearing `align_score`/
   `align_scale` lands exactly on the "No aligner, no refine" row of the
   score-precedence table — a state the schema already knows how to mean.
   Restoring *post*-refinement boundaries with a cleared score would match no
   row in that table: a refined boundary with no score is not a state
   anything else in the system produces.
2. In the combined path, `refine_boundaries` runs on the *aligner's* spans,
   not the transcriber's own — so "post-refinement transcriber timing" is not
   even a coherent thing to ask for there. Pre-refinement is the only
   definition that means the same thing in both the combined path and
   `realign_video`.
3. It is the more honest reading of "original": what the transcriber said,
   full stop, with nothing measured or aligned laid on top of it yet.

**Both bounds or neither.** A token whose transcriber timing is a start with
no end (Whisper-class engines routinely omit one) is not restorable — writing
just the start would make `unalign` produce a `timed` row with `end_ms NULL`,
which the schema's own `CHECK` rejects for anything but `caption`. Both
`orig_start_ms` and `orig_end_ms` are set only when the transcriber supplied
both; otherwise both stay `NULL`.

**Captions have no original.** `caption`-tier rows are never produced by
`transcribe_video`/`replace_words` — they come from `rytp/transcribe/captions.py`'s
own `INSERT`, which this change does not touch. There is nothing to restore a
caption row *to*: alignment promotes a video away from `caption` (§3, "Should
captions be promotable" aside), it never demotes back to it, so `unalign`
never needs to reach a caption row. Both columns stay `NULL` for `caption`
words by construction, and that is the whole of the decision.

**Pre-existing rows, and the other way a row can lack an original.** Every
row written before migration 16 has `NULL` in both columns — there is no
guess worth backfilling, the same call migration 13 made for `align_scale`'s
`unknown`. But a second, distinct case produces the same `NULL`: a row
written *after* migration 16 by a transcriber that emitted text only under an
aligner (contracts §4 permits `RawWord.start_ms`/`end_ms` to both be `None`),
so there was never an original to capture regardless of when the row was
written. `transcribe unalign` treats both cases identically — refuse, with a
message naming both possibilities, rather than resolve them differently or
guess. `replace_words` rewrites every row of a video in one transaction, so
in practice a given video's `aligned` rows are uniformly restorable or
uniformly not; the command need not — and does not — support restoring half a
video.

**`transcribe.unalign <video>`.** Sets `source = 'timed'` for the video's
`aligned` rows, restores `start_ms`/`end_ms` from the two new columns, clears
`align_score` and `align_scale`, and resets `engine` to the base transcriber
name (`engine.split("+")[0]`, the same recovery `realign_video` already does
the other direction) — because `words.engine` is documented as recording
"every stage that touched these timings" (contracts §3), and a `timed` row
still tagged `gigaam+wav2vec2+energy` would be exactly that lie, visible in
`videos list --long`'s `engines` column. Text, ordinals, `video_speaker_id`
and confidence are untouched. `utterances` are deleted (they copy timings)
and the index job is re-enqueued, the same as `transcribe.align`. Refuses
outright, for the whole video, when any `aligned` row has no recorded
original — see above. Not a job kind: it is a single `UPDATE` plus a
`DELETE`, cheap enough to run inline, so it is registered `long_running=False`
and does not appear in `tests/test_consistency_jobs.py`'s job-kind roster.

**Alternative rejected:** keep `words.start_ms`/`end_ms` as the only timing
columns and have `unalign` merely flip `source` back to `timed`, per the
entry's own original, more modest proposal. Rejected because it was written
*before* this fix was possible — the entry says so explicitly ("a downgrade
therefore cannot restore \[the old boundaries] — it can only relabel the
current boundaries as `timed`"). Now that the originals survive, doing the
weaker thing on purpose would be strictly worse than what the schema can
support, and would leave `--allow-timed` reasoning about a `timed` tier that
sometimes secretly holds aligned-quality boundaries under a `timed` label
with no way to tell which.
