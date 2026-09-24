# TODO

Deferred ideas — things worth doing eventually, deliberately not now.
Distinct from `BUGS.md`, which is for defects and requests being batched for
the near term.

Placeholders only: `<url>`, `VIDEO_A`, `https://example.invalid/...`. Never a
real video id, channel id or URL.

---

## Multiple audio tracks per video

**Deferred 2026-09-24.** Today a video may hold at most one `audio` asset —
`SINGLETON_ASSET_ROLES` in `rytp/constants.py:266`, and design §4: "at most
one canonical audio asset, at most one captions asset, and any number of
video renditions". Enforced in `rytp.db.queries.insert_asset`, not by a
unique index.

So a source carrying several language tracks, or a commentary track, cannot
be represented.

**Why it is not a small change.** `cache/wav/{video_id}.wav` is the single
audio timeline every word timestamp refers to. Admitting a second track
means either a second timeline — and then `words.start_ms` needs to say
*which* timeline it measures — or a rule that one track is canonical and the
rest are carried but never transcribed. The first touches the words schema,
the hot path of both indexes, and every cutting decision. The second is
cheap but only solves storage, not use.

**When it would matter:** a Russian-language archive with dubbed or
commentary tracks, or wanting to cut from one track while displaying
another. Neither is a current need.

If taken up, it is a **contracts change** — raise it against
`docs/superpowers/specs/2026-09-21-rytp-contracts.md` rather than varying
from it.


---

## Verify a render by re-transcribing it

**Idea, 2026-09-24.** Feed a rendered output back through the pipeline and
check whether a transcriber still hears the sentence that was assembled. If
Whisper reads the render back as the target phrase, the splice is clean; if
it does not, a seam ate a phoneme or a boundary landed mid-word.

This is an automatic quality signal for assembly, which otherwise has none —
entry 13 shows the alignment scores cannot serve as one, and D1 removed the
quality gate from `--allow-timed` entirely.

Sketch: `render run`, then transcribe the output, then compare the result
against the cut list's target with the same `normalize_text` both sides, and
report the word-level difference. Everything needed already exists; nothing
joins it up.

Caveats to design around:

- A rendered clip is short, and a transcriber given two seconds has little
  context, so a misread may say more about the clip's length than the
  splice. Comparing per word against known boundaries is better than judging
  the sentence as a whole.
- The render must not silently enter the corpus as a cuttable source —
  cutting from previous renders would compound artefacts. Either keep it out
  of `videos`, or mark such rows so assembly skips them by default.

---

## Alignment dies on a chunk too short for its words

**Moved from `BUGS.md` entry 32, 2026-09-24**, because the real fix is a
change to how audio is chunked rather than a patch to the adapter.

`transcribe run 3 --transcriber whisper --aligner wav2vec2` failed with

    RuntimeError: forced_align_impl ... targets length is too long for CTC.
    Found log_probs length: 12, targets length: 17, and number of repeats: 0

CTC alignment requires at least as many audio frames as target tokens. At
wav2vec2's ~20 ms stride, 12 frames is roughly 240 ms of audio being asked to
contain 17 characters. Impossible, so torchaudio refuses rather than
guessing.

The adapter (`rytp/transcribe/align/wav2vec2.py:113`) calls `forced_align`
without checking that the frame count can hold the targets, and has no
fallback when it cannot. Likely arises where a short VAD-derived chunk
receives several words from the bucketing at
`rytp/transcribe/pipeline.py:475`, or where a word's start lands just inside
a chunk that ends immediately after.

**The costly part is what the failure takes with it.** In the combined path,
alignment runs *before* `replace_words`, so the exception propagates before
anything is stored and the whole transcription is discarded — minutes of GPU
work thrown away because one chunk of a few hundred milliseconds could not be
aligned. The video is left with no words at all, not even `timed` ones.

**The combined path is the fragile one — confirmed 2026-09-24.** The video
that failed under `transcribe run --transcriber whisper --aligner wav2vec2`
later succeeded under a standalone `transcribe align --aligner wav2vec2`.
The two paths assign words to chunks differently:

- Combined (`_chunk_spans`, `rytp/transcribe/pipeline.py:141`) aligns *every
  word the transcriber emitted for a chunk* against *that chunk's window*.
  A long phrase returned for a short VAD segment must fit in that segment's
  frames, which is the failure exactly.
- Standalone (`realign_video`) buckets already-stored words by start time, so
  a short chunk receives only words that start inside it.

**The bug is latent, not fixed.** The standalone path can still fail when
many words share a start millisecond inside a short chunk — the transcriber
zero-gap pattern that made the `timed` tier necessary. Do not read one
success as a resolution.

Useful consequence meanwhile: transcribing and aligning as two steps is not
only safer for preserving work (below), it is less likely to hit this at all.

**Owner's requirement (2026-09-24): every video must end up processable, not
most of them.** That rules out the cheapest fix. Skipping an unalignable
chunk would leave those words at `timed` and make the video *partly*
cuttable, which is explicitly not acceptable — the chunk has to be made
alignable, by merging it with a neighbour or re-chunking so that every chunk
holds enough frames for the words assigned to it. Falling back to skipping is
a last resort, and when it happens it must be reported per video rather than
passed over silently.

**Why padding is the wrong instinct, and what the right version is.**
Padding with dummy *tokens* lengthens the side that is already too long.
Padding the frames with *silence* would let CTC succeed and produce
nonsense: 17 characters in 240 ms is ~70 characters per second, against
roughly 15 for Russian speech, so those words were demonstrably not spoken
inside that chunk. Alignment against fabricated silence would place
boundaries confidently inside the padding, and entry 13 shows the score
cannot warn anyone.

The arithmetic points at the bucketing, not the audio. Words are assigned to
a chunk by start time alone (`_chunk_for(chunks, int(row[1]))`), and the
timestamps being bucketed are the transcriber's — the ones measured at 78.7%
zero-length gaps, where many words share a start millisecond. A cluster of
those lands wholesale in one short chunk no matter when the words were
actually said. **This failure is downstream of untrustworthy input
timestamps**, which is the same root cause that made the `timed` tier
necessary.

So the fix is to widen the window with *real surrounding audio* until the
frame budget can hold the targets, align against that, and keep the
boundaries that land in the original region — or to merge the chunk with its
neighbour, which is the same idea expressed in the chunk planner. Either
way, check `T >= L + repeats` before calling `forced_align` rather than
letting torchaudio raise.

Two fixes, and the second matters more than the first:

1. Handle the short chunk: skip it and leave those words at their incoming
   timings, or merge it with a neighbour before aligning, and report which
   words were affected. A note on the job (handlers may return one) is the
   natural channel.
2. **Persist `timed` words before attempting alignment.** The tiers already
   exist precisely so a transcript can be searchable before it is cuttable;
   writing them first makes a failed alignment cost only the alignment.
   `transcribe align` can then upgrade in place, which is the promotion path
   contracts §3 already describes.

Workaround meanwhile: run `transcribe run <id> --transcriber whisper` with no
aligner to bank the `timed` words, then attempt `transcribe align`
separately. The alignment will still fail on this video, but the
transcription survives.
