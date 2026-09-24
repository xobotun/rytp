# Findings from the second independent review — status

**Reviewed:** 2026-09-21. **Resolved:** 2026-09-22. Nothing has been implemented yet, so all fixes are to the plans and contracts.

The reviewer read the design, contracts and all seven plans without being told what had already been reviewed or what was considered settled. Its diagnosis of the pattern is worth keeping: the problems clustered **where the contracts named a responsibility without naming a signature, or named a job kind without naming its payload.**

---

## Resolved

**1.1 — Three incompatible speaker resolvers.** Contracts §5 now pins `SpeakerFilter(video_speaker_ids, description)` and `resolve_speaker_filter(db, *, speaker, video_local_speaker, video_id) -> SpeakerFilter | None`, with four rules spelled out. The substantive half was that Part 1's version stopped at `speakers.id` while every consumer filters `video_speakers.id`, and the contract forbade parts 4 and 5 from writing that join — so nobody was. The expansion now happens inside the resolver. Parts 1, 4 and 5 all adopted it.

**1.2 — `align` jobs with no payload.** `settings.default_aligner` names the aligner; ingest resolves it **once before the loop**, stamps it onto each job, and rejects an unregistered name at enqueue time rather than after five retries. Empty means no `align` job is enqueued at all and words stay `timed`.

**2.1 — Unaligned words marked cuttable.** `words.source` is now `caption | timed | aligned`. `timed` means a transcriber's own word timestamps: searchable, not cuttable. The `align` job upgrades rows in place. Cuttable remains `source = 'aligned'`, which now means what it says.

**3.1 — Stale `VIDEO_JOB_KINDS`.** Derived from `JOB_KINDS[k].target_kind == "video"` instead of hardcoded. `caption_words` is in the test fixtures precisely because the old list would have orphaned it.

**3.2 — Re-transcribe silently discards speaker mapping.** Part 3 turns the counts it already computed into a warning naming the count and the re-diarize command. This exposed a deeper gap, now also fixed: job handlers returned `None`, so a warning reached someone running one video by hand and vanished on the worker — the bulk path. Handlers may now return a note, stored in `jobs.note`, shown by `jobs.list`, counted by `jobs.stats`. Part 2 immediately found a second use: warning when a possibly-translated caption track was taken because the original was unavailable.

**4.1 — Assembler's hot query could not exclude the caption tier.** Partial index `words_alignable ON words(normalized_text) WHERE source = 'aligned'`. Part 5's query-budget fixture is now caption-heavy (nine caption and nine timed videos per aligned one, with a test asserting the ratio) and checks the plan via `EXPLAIN QUERY PLAN` by index name, since statement counting cannot see a rewrite that scans captions inside one query.

**4.2 — Prose drift.** Migration numbering, `target_kind="render"`, the ingest chain description, and `--video-id` → `--video` throughout.

---

## Still open

Updated 2026-09-24, after all eight parts were implemented. The four entries
that used to sit here are closed: `_reject_unknown_aligner` was fixed when
Part 2 was written; Part 2's self-review contradiction and Part 8's stale gap
table were both moot by the time the code existed; and **3.3 TUI scope** is
closed because Part 8 shipped cut-list editing, queue watching and enqueueing
from the TUI. What follows is what implementation actually left open.

**A. `transcribe.run --enqueue` silently does nothing on a video that already
has words.** `transcribe_readiness` returns SATISFIED as soon as any non-caption
word exists, and `queue.enqueue` writes such a job straight in as `done` without
ever calling the handler. Measured: enqueueing `transcribe` on a video with
`timed` words yields a job in state `done`; the same video after
`transcribe remove` yields `pending`.

This matters more than it looks, because the owner chose *"being able to switch
between different transcribers"* over manual boundary marking. Via the worker —
the bulk path — that flow is currently dead, with no error to say so.

The other half of engine-switching is fine: **`align_readiness` is never
SATISFIED** by design ("already aligned" does not mean "aligned by the engine you
just asked for"), so re-enqueueing `align` with a different aligner works, and
the kind is `reopenable=False` so a reconcile will not re-run it behind your
back. Measured: `pending`.

Two ways to close it, both a decision rather than a bug fix:
- `transcribe.run --enqueue` errors when the video already has words, naming
  `transcribe remove` in the message. Cheapest, and keeps `enqueue`'s contract.
- A `--replace` flag that removes the transcript and enqueues in one step.
  Nicer to use; needs the flag threaded through the payload.

**B. `jobs.retry` cannot retry a job by id without also naming its state.** The
query issues one `state = ?` clause with a `"failed"` default and no wildcard, so
retrying a single known job means guessing what state it is in.

**C. TUI startup imports every command module, and therefore every workflow
package.** Part 8's plan asserted that starting the TUI imports none of
`rytp.index`, `.assemble`, `.render`, `.diarize` or `.transcribe`. That is
unattainable as the system is built: contracts §5 makes registration a side
effect of importing a command module, and `COMMANDS` must be complete before
either surface can be generated. Part 8 verified the behaviour predates its own
work and narrowed the test to the constraint that is both true and load-bearing —
no *heavy* optional extra loads eagerly — documenting the reasoning in the test.
The original intent was TUI startup speed. Making it real means lazy command
registration across six parts, which is a design change, not a fix.

## Deliberately not changed

Recorded so they are not re-litigated. Stale cut lists after re-transcribe are fine — timings are baked into the TOML and the media is unchanged. Re-indexing genuinely re-queues, because `enqueue` recomputes state from readiness on conflict. `align` readiness would loop under reconcile but is `reopenable=False`. Caption rows with a null `end_ms` are handled by Part 4's `implied_end_ms`. The cut-list format agrees field for field between parts 5 and 6. Every entity in the deletion table has an owner.

Part 5 kept its `align_score` coalesce after reconsidering it, with a reason: contracts §6 makes `Span.score` optional and MFA reports none, so removing it would refuse to assemble from an MFA corpus.

One thing worth a test that has none: `rytp/models.py` imports `snowballstemmer` lazily inside a function, which is what keeps Part 3's `base.py` stdlib-clean under a foreign interpreter. Load-bearing and currently unpinned.
