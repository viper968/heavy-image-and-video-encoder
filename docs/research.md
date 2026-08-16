# What else is there, and what does it actually buy?

Survey of techniques used by the codecs that beat this one, each implemented or
costed against the dev split. **Every number in the "measured" column was run
here**; predictions that did not survive contact with the data are marked as
such, because the interesting result is usually the gap between them.

Sources were read as primary material: libjxl (`d089091`), FLIF, CharLS (the
JPEG-LS reference implementation), Matt Mahoney's `lpaq1.cpp` and *Adaptive
Weighing of Context Models*, the ZPAQ specification, and the FLIF/MANIAC ICIP
2016 paper.

## Measured on this codec

| Technique | Predicted | Measured | Verdict |
|---|---|---|---|
| Logistic mixing on the zero flag (3 experts) | 4-7% | **-1.0%** | kept |
| Secondary estimation (APM/SSE), zero flag | 2-4% | included above | kept |
| Secondary estimation on the magnitude bins | — | **-0.2%** | kept |
| Self-correcting weighted predictor | not isolated in sources | **-0.9%** | kept |
| Chroma bias fix (found while testing predictors) | — | **-0.5%** | kept |
| JPEG-LS bias cancellation (A/B/C/N) | +0.5-2% | **+4.1% worse** | rejected |
| Weighted predictor's error feedback (libjxl p1C/p2C) | part of the design | **+0.5% worse** | disabled |
| 4th expert on weighted-predictor error | libjxl's best tree property | **-0.003%** | rejected, not worth the cost |
| Finer mixer weight-set context (896 sets vs 128) | — | **+0.1% worse** | rejected |
| Splitting the error context into west/north | — | **+0.5% worse** | rejected |
| Cross-channel context on the magnitude bins | — | **+0.2% worse** | rejected |
| Online learned context tree (MANIAC-style) | the big one, see below | **-0.1%** | rejected |
| Match model, hard switch on sustained agreement | the one I'd bet on | **-0.26%** held-out, **8.7x** on repetition | kept |
| Match value averaged into the predictor blend | — | **+1.3% worse** | rejected |
| Match's expected sign as a sign-bit context | — | **-0.26%** | kept |
| Match's expected magnitude as a *split* of the length-bin context | — | **+0.05% worse** | rejected |
| Match's expected magnitude as a *mixed expert* on the length bins | — | **-0.16%** | kept |
| Second APM stage keyed on match state (LPAQ chains two) | 0.6-2% | **-0.09%** | kept |
| Two trend sub-predictors (2W-WW, 2N-NN) in the blend | paq8px calls these strong | **+0.20% worse** | rejected |
| **Learned (NLMS) combiner replacing the error-weighted average** | the one item left | **-1.10%** held-out, -1.95% dev | kept |
| — combiner over 7 mostly-redundant linear inputs | 0.5-1% | **-0.013%** | superseded, see below |
| — combiner over the second ring + GAP (13 inputs) | — | **-1.18%** dev | kept |
| Match value as a combiner input | the thesis test | **+0.15% worse** | rejected |
| Match value with one weight per match state | — | **+0.13% worse** | rejected |
| Combiner confidence as a 5th zero-flag expert | — | **-0.13%** | kept |
| Combiner confidence as a 3rd length-bin expert | — | **-0.18%** | kept |
| Combiner weight sets on 16 activity buckets vs 8 | finer is better | **+0.06% worse** | 8 kept |
| Combiner weight sets split 4 ways on gradient direction | — | **-0.37%** | kept |
| **Half-pel motion vectors** (video) | 5-8% from the proxy | **-3.7%** dev, **-2.3%** held-out | kept |
| Median MV predictor from left/above/above-right (video) | 1-2% | **-0.012%** | kept, but see below |
| Larger motion search, ±12 / ±16 / ±24 (video) | bus clips at ±8 | **+0.01 to +0.04% worse** | rejected |
| Third block mode: spatial prediction of the MC residual (video) | 10%+ from the frame proxy | **-0.2 to -0.8%** per block | rejected |
| Variable block size, 16x16 split into four 8x8 (video) | sharper motion edges | **+1.2% to +22.5% worse** | rejected |
| **Coarse-to-fine motion search** (video) | speed only | **7.4x faster encode at 1080p**, size within 0.01% | kept |
| Pyramid search forced on at CIF | — | **+1.26% worse** on foreman | gated on frame size |
| Skipping the learned combiner on inter blocks | speed only | faster **and -0.21%** | kept |
| Suppressing the match override on inter blocks | 7.9% on Sintel | **-0.12%** | kept |
| Gating the match model on a non-flat neighbourhood | the Sintel loss | **+0.05 to +0.19% worse** | rejected |

### The three that failed are the informative ones

**JPEG-LS bias cancellation** was the confident prediction and the worst result.
The reasoning for it is sound in isolation: correcting a systematically wrong
prediction does not merely make the error cheaper to signal, it makes those
pixels predict *right*, and no amount of probability adaptation can turn a true
magnitude-2 residual into a magnitude-0 one. What that argument misses is that
the correction term is itself adaptive. It moves by ±1 per pixel, so it keeps
shifting the residual distribution that the per-context probability model is
simultaneously trying to learn, and the model spends its life chasing a moving
target. A deadzone variant (apply only when |C| ≥ 2) recovered almost nothing,
which rules out simple ±1 oscillation as the cause. Bias cancellation pays off
against a *fixed* Golomb-Rice code, which is what JPEG-LS actually has — the
technique and the entropy coder are a matched pair, and lifting one out of that
pair is what fails.

**The weighted predictor's error feedback** failed for a related reason. libjxl
corrects each sub-prediction by its neighbours' errors *and* weights the blend
by those same errors. Swept across feedback strengths 0/4/8/16, zero won every
time: with error-weighted blending already in place, correcting the inputs too
double-counts the same signal.

**Extra experts stopped paying.** A fourth expert keyed on the weighted
predictor's max neighbour error — the property libjxl's tree builder ranks
highest — returned 13 bytes on a 464KB image. That context is nearly a
restatement of the neighbour-error context expert 1 already uses. Mixing only
pays when the experts genuinely disagree.

## The learned combiner: the rebuild, and what it did and did not fix

This was the one item the rest of this document pointed at, and it is now built.
The error-weighted average is replaced by an NLMS combiner whose weights are
learned online by gradient descent on the real prediction error. Held out, it is
worth **1.10%**, which closes the JPEG XL gap from 8.17% to **6.99%** and is
roughly three times the size of any other single change in this file.

Three things about it are worth more than the headline number.

**The first attempt returned nothing, and the reason was not the combiner.**
Seven inputs — the four existing sub-predictors plus NE, NW and the planar
`W+N-NW` — gave **-0.013%**, indistinguishable from zero across a full sweep of
step sizes. The combiner is *linear*, and every one of those inputs was already
a linear combination of the same four neighbours. `2W-WW` is worthless beside W
and WW; `W+NE-N` and `W+N-NW` add no span at all. A linear combiner cannot gain
from inputs that are already inside its span, however good each one is on its
own. Replacing them with an actual basis — the second ring (WW, WWW, NN, NNW,
NNE, NWW, NEE) plus two genuinely *nonlinear* predictors (MED and CALIC's GAP) —
took the same machinery from -0.013% to **-1.18%**. The lesson is that "add more
predictors to a mixer" is only half an instruction; what matters is whether they
are reachable from the ones already there.

**Averaging was not the match model's whole problem.** The thesis this rebuild
was built on says an average is dragged by a volatile member while a learned
combiner ignores it, and the match model was the evidence. So the match value
went into the combiner as a test. It measured **0.15% worse**, and with a
separate learned weight per match-length state — which lets the combiner hold a
near-zero weight for an unproven match and a near-unity one for a proven match —
still **0.13% worse**. Both were rejected. A bimodal predictor wants a *switch*,
and the hard override on sustained agreement remains the right combiner for it;
the learned weights improve on the average for predictors that are continuously
somewhat-right, which is a different population. The original finding needed
narrowing, not confirming.

**The confidence it produces is worth almost as much as the prediction.** The
combiner computes two quantities for free: how much its inputs disagreed (the
input energy) and how far it had to move the blend (the correction). Neither is
visible anywhere else in the model, and both bear directly on whether the
residual will be zero. As a fifth expert on the zero flag that is **-0.13%**,
and as a third expert on the length bins another **-0.18%** — together about a
quarter of what the prediction change itself bought. This is also the half of
the rebuild where *logistic* mixing genuinely applies: a logistic mixer combines
probabilities, so it belongs on the binary decisions, while the pixel-value
combiner has to be a linear one. Conflating the two is the easiest way to
mis-scope this work.

Weight-set context followed the usual dilution curve: one set per plane was
-1.13%, eight activity buckets -1.58%, sixteen **worse** again at -1.53%. The
budget was better spent on a different axis — a four-way split on whether the
neighbourhood is smoother vertically or horizontally, which is the property that
decides which predictors deserve weight — and that was worth a further 0.37%.

Dev improved 1.95% against 1.10% held out. The constants (step size, energy
floor, bucket counts) were swept on dev, so some of that spread is fitting; the
held-out number is the one to quote.

## Motion modelling: one of four ideas worked

The README used to name "sub-pixel motion and variable block sizes" as what
video was missing. Half of that was right.

Four things were tried, and the useful part of this section is the order they
were tried in, because three of them were **costed with a numpy proxy before
being built** and two of those were abandoned on the strength of the estimate
alone. The proxy is `_COST`, the same log-ish bit-cost the encoder already uses
to pick block modes, summed over residuals. It is not the real coder — it knows
nothing about context modelling — but it is fast enough to answer "is there
anything here at all" in about a minute per idea.

**Half-pel vectors: -3.7% dev, -2.3% held out.** Kept. The proxy said 5.3% on
bus, 7.9% on mobile, 6.7% on foreman and roughly nothing on the near-static
clips, which is exactly the shape a real sub-pixel effect should have. The
delivered number is smaller than the proxy, as always, because the context model
was already recovering some of it. On foreman this alone was 3.05%, which is
what moved that clip from fourth place to first.

The reference is bilinearly interpolated to four phases and the vector's low bit
picks one, so the inner loop stays a single array index. Search is two-stage —
exhaustive whole-pixel, then the eight neighbouring half-pel positions — which
costs 15% more encode time rather than the 4x a full half-pel search would.
A half-pel refinement also has to *beat* the whole-pixel match by a margin
rather than tie it, because an odd vector component roughly doubles the unary
run that codes it; without that margin the near-static clips paid 0.2% for
precision they had no use for.

**Larger search range: rejected, and the reason is worth keeping.** 19.2% of
bus's temporal blocks sit pinned at the ±8 search limit, which looks like a
clear diagnosis. Widening to ±12, ±16 and ±24 made the file *slightly worse*
every time. The search was never the constraint: it finds those vectors, but a
wider range also means longer vectors to code, and the mode decision charges a
flat penalty that does not price them. A saturating histogram is evidence of a
limit being hit, not evidence that the limit is what costs you.

**Median MV predictor: -0.012%.** Kept, because it is strictly better and free,
but the number is the finding. Predicting each vector from the median of its
left, above and above-right neighbours instead of from the left alone is
textbook H.264 and it bought almost nothing — which says motion vectors are a
negligible share of a *lossless* file. That single measurement is why the
rate-aware search that would normally follow was not built: there is no prize.

**Spatial prediction of the motion-compensated residual: rejected.** This looked
like the big one. Measured per frame, coding `cur - ref_mc` with MED prediction
on top beats coding it flat by 13.7% on bus and 9.9% on foreman. But measured
*per block*, letting each 16x16 block pick the cheaper of the two, the gain
collapses to between 0.2% and 0.8% — because the existing per-block choice
between spatial and temporal already captures nearly all of it, and the third
mode wins outright on only 1-13% of blocks. A frame-level average hid a decision
the codec was already making. Not worth a third block mode and its signalling.

**Variable block size: rejected, decisively.** Splitting each 16x16 into four
8x8 vectors scored worse on every clip — 1.2% on bus, 22.5% on akiyo — with the
extra vectors costing more than the sharper motion boundaries save. This agrees
with the earlier finding that 8x8 blocks alone were slightly worse, and
strengthens it: it is not that 8x8 is the wrong size, it is that subdividing
does not pay at this bitrate. Lossless video spends so much on residuals that
motion side-information is nearly free to *omit* and expensive to add.

## Where video encode time goes, and what a 7x speedup did not fix

At 1080p the encoder took 145s for 16 frames against x264's 0.4s. Profiling put
**85% of that in motion search** — 289 whole-pixel positions, each a full-frame
pass. Three changes took encode to 19.8s, a 7.4x speedup, with output within
0.01% of the same size:

| change | search time | why |
|---|---:|---|
| baseline | 122.9s | 289 full-frame passes per frame |
| cost as a 256-entry byte LUT | — | `(a-b) & 255` indexes a table with the fold baked in, replacing a widen, subtract, abs, minimum and gather. Was 51% of search. |
| padded slicing instead of clipped fancy indexing | — | a strided view rather than a gather; was 25% |
| coarse-to-fine pyramid | **10.9s** | search a quarter-size frame, refine +-1 down two levels |
| one gather instead of four in the half-pel stage | **8.6s** | index the phase as part of the gather rather than fetching all four planes and discarding three |

Two things about the pyramid are worth keeping.

**It has to be gated on frame size.** Forcing it on at CIF cost **1.26% on
foreman**: box-downsampling a 352x288 frame twice destroys the detail that
separates candidate vectors, and refinement cannot recover a match the coarse
level never pointed at. Below 300k pixels an exhaustive search is affordable
anyway, so the codec just does one. Speed work that quietly costs compression on
small inputs is not a speedup, it is a trade, and this one did not need to be.

**Refining against a moving centre was a bug.** The refinement updated the
vector in place, so the second candidate was an offset from wherever the first
had already moved it. Fixing it to evaluate all neighbours against a fixed
centre was both slightly *better* compression and, more importantly, bounded:
the drift is what let a vector reach +-(2*SEARCH+2).

### A latent desync, found by trying to bound that drift

Motion vectors are coded as a differential against the median of three
neighbours, with a unary magnitude that saturates at `MV_MAX`. If a differential
exceeds it, the encoder writes a longer run of ones than the decoder will read
back, and **the stream desyncs from that point with nothing reporting an error**.
Half-pel vectors made this reachable: the refinement could push a vector to
+-(2*SEARCH+1), two neighbours moving hard in opposite directions then give a
differential of 4*SEARCH+2, and `MV_MAX` is 4*SEARCH.

Both foreman and bus produce vectors at +-17 against a limit of 16, so this was
live, not theoretical. Vectors are now clamped to +-2*SEARCH, which bounds any
differential — including one against a median of neighbours — to exactly
`MV_MAX`. Synthetic content did not reproduce it, so the regression test uses
the real clips and asserts the bound directly.

### What was left, and what the C port then did about it

Encode split about evenly at that point: 8.6s of search and 11s of coding. The
prediction recorded here was that closing the rest needed a C implementation and
threads. The C implementation was then written; this is what it actually bought,
and where the prediction was wrong.

**First, a correction to the baseline.** The 19.8s above included numba's
cold-start compilation. Measured warm — the codec encoding its second clip, which
is the honest number for a codec — the numba path does 16 frames of 1080p in
**9.8s**, not 19.8s. Every speedup below is quoted against the warm number.
Anything quoted against 19.8s is inflated by roughly 2x, and the earlier
"~50x slower than x264" was inflated the same way.

One real waste was found and removed: the learned combiner was computing
thirteen multiply-accumulates, a division and thirteen weight updates for every
pixel of an inter block, where `pred` is then overwritten by the reference
sample. Skipping it there was faster **and 0.21% smaller** on the held-out
clips, because the confidence context had been feeding the mixer spatial
information that is misleading on a temporally predicted pixel.

## The C port: where the factor of four came from

`csrc/kernel.c` and `csrc/motion.c`, loaded through `hve/native.py`. Output is
byte-identical to both other paths on every test — that is the whole constraint,
and `tests/test_native.py` compares bytes rather than sizes.

Best of three runs, native and numba interleaved so machine drift hits both:

| workload | stage | numba | native | |
|---|---|---:|---:|---:|
| 4 Kodak photos | encode | 1.63s | 0.95s | 1.7x |
| | decode | 1.03s | 0.85s | 1.2x |
| foreman CIF x16 | search | 1.44s | 0.061s | **23.5x** |
| | encode | 2.11s | 0.42s | 5.0x |
| | decode | 0.47s | 0.37s | 1.3x |
| Sintel 1080p x16 | search | 4.42s | 0.119s | **37.1x** |
| | encode | 9.82s | 2.74s | 3.6x |
| | decode | 4.37s | 2.55s | 1.7x |

Against x264's 0.4s on the same 1080p clip, encode goes from ~25x slower to
**~7x slower**. That is a different category of number, but it is not parity and
this document should not pretend otherwise.

### Why C beat numba, given that numba compiles too

"C is faster" is not an explanation — numba emits LLVM IR and usually lands
within a factor of two of C on a loop like this, which is roughly what the
decode column shows (1.2x-1.7x). The large numbers are all structural:

**The search is threaded, and that is a property of the problem, not the
language.** Every block's search is independent through the whole pyramid, so
the outer loop over block rows hands out to as many cores as there are. That is
30x-40x on a 16-core machine and needs no format change at all, because motion
estimation is an encoder-side decision that the decoder never repeats.

**The search was reordered.** numpy is only fast at whole-frame shapes, so it
iterates search positions outside and blocks inside, materialising a full-frame
cost map per position. In C the loop nest inverts: blocks outside, positions
inside, the running best in a register and the reference window in L1. Most of
the single-threaded gain is this, not the code generation.

**The scratch buffers are narrower.** `fast.py` shares its arrays with the
reference implementation's Python lists and so carries everything as int64. In C
nothing is shared, so the match model's history is `uint8` (it holds pixel
values) and its hash table is `int32` (it holds sample indices). At 1080p that
is 6 MB of random-access working set instead of 24 MB.

**The per-pixel divisions got cheaper.** The self-correcting blend does four
divisions by a small sum of recent errors; those are now a 2048-entry table with
a fallback above it, so a parameter sweep that widens the range loses the speed
rather than the correctness. The two divisions that remain have operands that
provably fit in 32 bits, and a 32-bit `idiv` costs meaningfully less than a
64-bit one. The combiner's weights moved to `int32` for the same reason — they
are clamped to +-2^20 every update — which also lets its 13-tap dot product and
13-tap update vectorise.

### What did not work

**Narrowing `stretch` and `squash` to int16.** Six of the seven table lookups
per pixel hit these two, and at 8 KB each they would fit L1 where 32 KB each do
not. Measured: inside this machine's +-10% run-to-run noise, in both directions
across repeats. Reverted, because a change that cannot be measured should not be
carried. The wider lesson is that this machine's noise floor is about 10% on a
3-second benchmark, so any single-run claim below that is not a claim.

**Threading the pixel loop.** Still not possible, and this part of the earlier
prediction holds exactly. One range coder, one adaptive model bank, and every
pixel depending on all prior ones. The coding loop is now 2.5s of the 2.7s
encode, so it *is* the remaining cost, and getting past it still needs the
slice-independent format change described under "What would actually close the
gap" — at a measured price of roughly 0.3-1% ratio.

### A search bug, found by diffing the port against the original

`_full_search` seeds its running best from the first candidate, which is
`(-radius, -radius)`, but left the vector array at its zero initialisation. A
block whose true best match was exactly that first candidate therefore kept that
candidate's *cost* while reporting the *zero* vector, and was then
motion-compensated from the wrong place. About one block per CIF frame.

This is the kind of bug that a port finds and a test suite does not: nothing was
wrong with the output, the codec stayed lossless, and the only symptom was a
slightly worse ratio. It only surfaced because the C search disagreed with the
numpy one on exactly one block out of 396 and both had to be explained.

Fixing it is worth, across six clips, **+28 bytes** — that is, nothing, and very
slightly the wrong way:

| clip | before | after |
|---|---:|---:|
| bus | 1,048,000 | 1,048,000 |
| mobile | 1,204,587 | 1,204,587 |
| container | 724,219 | 724,245 |
| akiyo | 316,313 | 316,313 |
| foreman | 834,879 | 834,903 |
| Sintel 1080p | 30,966 | **30,944** |

Kept anyway: a search that reports a vector other than the one it scored is a
trap for anyone who later tries to tune the search, and 0.0007% is not a price.

**But the reason it does not pay is the interesting part.** The correct vector
at the corner of the search window is expensive to *send* — the magnitude is
unary, so a large differential is a long run of ones — and the search's cost
proxy does not know that. It scores residuals only; the price of the vector
enters just once, as a flat `mv_penalty=48` in `choose_modes`, and not at all in
the choice between two candidate vectors. Real codecs do rate-distortion
optimisation here and charge each candidate its actual coding cost. That has
never been tried in this codec and is now the most concrete untried motion idea,
ahead of multiple reference frames.

### The cost of a third implementation

There are now three implementations of one loop, and the honest accounting is
that this is a real maintenance tax: `model.py` defines the format, `fast.py`
mirrors it in numba, `csrc/` mirrors it in C, and any model change means three
edits and a byte-identity check. The C is written as a deliberately boring
transcription — same variable names, same order of operations, same comment
anchors — specifically so that diffing it against `fast.py` stays possible.

What makes the tax payable is that the failure mode is loud. `tests/test_native.py`
compares bytes across backends on odd shapes, real photographs and real clips,
and the pure-Python reference still runs under `NUMBA_DISABLE_JIT=1
HVE_NO_NATIVE=1`. Worth noting what happened here as a warning: adding the
native backend silently broke `test_video_fast_path_is_byte_identical`, which
disabled only the numba path to reach the reference and so began comparing the
native path against itself. It passed, testing nothing, until it was read.

### An open finding: the match model costs 7.3% on 1080p Sintel

Disabling the match model entirely makes that clip **28,725 bytes instead of
30,966**, while the same model *saves* 1.25% on the held-out CIF pair. (Both
figures predate the `_full_search` fix, which moved the baseline to 30,944; the
"model off" side has not been re-measured, so treat the loss as ~7.2% and
re-measure both sides before acting on it.) Two
hypotheses were tested and both were wrong: suppressing only its override of the
temporal prediction recovered 0.12% (kept anyway — it is principled and helps
everywhere), and gating it on a non-flat neighbourhood made every set worse
(stills +0.05%, video +0.19%). Whatever is actually happening on near-static
high-resolution content has not been identified, and guessing further without a
diagnosis was not worth more time. It is the largest single known loss in the
codec right now.

## The match model: the one that needed a different combiner

Every other expert reads the same handful of adjacent pixels, so they mostly
agree and mixing them pays little. The match model looks somewhere else: it
hashes the causal neighbourhood (W, N, NW, NE), remembers where that exact
neighbourhood last occurred anywhere earlier in the plane, and offers the pixel
that followed it last time.

Three combiners were tried, and the difference between them is the whole story.

**As a mixer expert only** (predicting just *is the residual zero*): -0.2% on
photographs, but only 2.9% on a tiled noise image — an image built to be
spatially incompressible and perfectly repetitive, where a match model should
win overwhelmingly. It was finding the matches and then throwing the answer
away: knowing the residual is zero is worth about one bit, while knowing *what
the pixel is* is worth eight.

**Averaged into the weighted predictor blend**, alongside the four gradient
sub-predictors: **1.3% worse**. The blend weights each vote by its recent local
accuracy, which is the right combiner for four predictors that are all roughly
right. A match is not like that — it is either exactly right or wildly wrong,
and averaging a wildly wrong value into an otherwise good prediction damages
every pixel it touches. Relaxing the blend's clamp to let confident matches
through changed nothing, confirming the averaging itself was the problem.

**As a hard switch on sustained agreement** — once a match has held for N
consecutive pixels it replaces the prediction outright — the tiled image drops
from 108,658 bytes to 12,554, **8.7x**, while photographs pay 0.009%.

N trades one against the other, swept on the dev split:

| trust threshold | photograph | tiled noise |
|---|---:|---:|
| 2 | +1.8% | 6,075 |
| 4 | +0.24% | 8,248 |
| **8** | **+0.009%** | **12,554** |
| 16 | +0.000% | 21,289 |
| never | baseline | 108,658 |

8, because losing on photographs to win elsewhere is not a trade worth making.
The lesson generalises: a bimodal predictor needs a switch, not an average, and
the reason the first two attempts underperformed was the combiner rather than
the model.

On the held-out 18 the finished model is worth **0.26%**, more than the 0.009%
the single tuning image suggested — Kodak does contain repetitive structure
(brickwork, market awnings, fabric), just not in the image the threshold was
swept on. The gap to JPEG XL closes from 7.20% to 6.92%.

## The learned context tree: built, measured, rejected

This was the lever the research pointed at hardest, so it was built rather than
argued about. A MANIAC-style tree that starts as one context and splits a leaf
when the split is measured to pay for itself, with each leaf tracking a running
mean and two virtual probability slots per candidate property.

The one genuinely novel part: **it transmits nothing.** FLIF and JPEG XL both
send their trees because their encoders make choices the decoder cannot
reproduce. Here every property is causal and the growth rule depends only on
bits both sides have already seen, so encoder and decoder grow byte-identical
trees with zero side information.

It works, it stays lossless, and it is worth **0.1%** — for 35% more encode
time. Loosening the split threshold grew the trees from 2 leaves to 10/44/80 per
plane and bought 0.05%. Swapping it in as the *primary* expert instead of a
supplementary one changed nothing material.

The reason is that FLIF's ~31% is measured against JPEG-LS, whose contexts feed
a **fixed Golomb-Rice code**. A learned tree is how you get adaptivity when your
entropy coder has none. This codec already has an adaptive arithmetic coder,
per-context online probabilities, a mixer over three differing views of the
neighbourhood, and secondary estimation on top. The tree, given the same
properties those experts already see, converges to about the same partition the
hand-designed grid already implements — and the mixer was already covering the
disagreement between partitions. Two routes to the same information; taking both
pays once.

The code was removed rather than left in as dead weight. The design is written
down here, which is the reproducible part.

## What would actually close the remaining gap

The single most relevant published number: **FLIF beats JPEG-LS by ~31% on
Kodak using an identical MED predictor** (ICIP 2016, Fig. 4, normalised
scores). The entire difference is context modelling — specifically MANIAC, a
context tree learned per image.

Both JPEG XL and FLIF partition the context space with a *learned decision
tree* rather than a fixed grid:

- **JPEG XL** builds its MA tree offline, greedily, scoring each candidate
  (property, threshold, predictor) split by the zeroth-order entropy of the
  token histogram on each side, accepting a split only when it beats the
  parent by a fixed bit threshold. Properties include `W+N-NW`, `W-NW`,
  `NW-N`, `N-NE` and the weighted predictor's max error. The tree is
  transmitted.
- **FLIF/MANIAC** grows its tree online during encoding. Each leaf tracks, per
  candidate property, a running average plus two "virtual" contexts (below and
  above that average) with their own cost estimates; when the virtual pair
  beats the actual context by a threshold, the leaf splits on that property.
  The tree is transmitted.

That was built here and did not pay (above). Of the list that followed it,
three have since been done and one has been costed out:

- ~~Condition the sign and magnitude bins on the match~~ — **done**, -0.42%
  combined, and the two halves of it are the clearest illustration in this
  document of mixing beating splitting.
- ~~Mixing on the magnitude bins~~ — **done** as part of the above.
- ~~A second secondary-estimation stage~~ — **done**, -0.09%.
- ~~Hybrid-uint tokens~~ — **not worth building**: the bit budget below shows
  raw bypass bits are 3.3% of the file (4.6% after the learned combiner),
  which is the entire ceiling for any binarisation change, and a redesign would
  capture only part of it.

- ~~A proper mixing network over many more predictors~~ — **done**, and it was
  the largest single change in the project: -1.10% held out, closing the gap
  from 8.17% to 6.99%. See "The learned combiner" above for what it fixed, what
  it did not, and the two ways it was initially mis-built.

That leaves no single large item identified. What the combiner did *not* do is
also informative: it did not rescue the match value (a switch beats a weight for
a bimodal predictor), and it did not help foreman, whose gap is motion
modelling rather than spatial prediction.

The plausible next steps, none of them costed yet:

- **More nonlinear inputs to the combiner.** The linear span is now well
  covered, and the two nonlinear members (MED, GAP) are carrying the change.
  That is a testable prediction: further gains should come from predictors that
  are *not* weighted sums of neighbours — a second GAP variant at a different
  threshold, a median of three predictors, a texture-matched value. Adding more
  linear neighbours should now do close to nothing, and if it does not, the span
  argument above is wrong and worth revisiting.
- **Two combiners at different adaptation rates.** paq8px runs six LMS filters,
  not one. The step-size sweep here has a single flat optimum between mu=1/32
  and 1/64, which is what you would expect if one rate is being asked to serve
  both smooth and busy regions. Two rates would need a combiner of their own,
  and on the evidence in this document that should be a mixer or a switch, not
  an average.
- ~~Sub-pixel motion and variable block sizes for video~~ — **half done**.
  Half-pel vectors are in and worth 2.3% held out; variable block sizes were
  measured and rejected. See "Motion modelling" above. What video still does
  not do is multiple reference frames or bidirectional prediction, and neither
  has been costed.

## Where the bits actually go

Measured on three dev images with counters in the encoder (`Bank.stats`), since
guessing which stage to attack next was clearly not working:

| | |
|---|---:|
| average cost | 3.164 bits per plane sample |
| residuals that are **not** zero | **74.0%** |
| raw unmodelled bypass mantissa bits | **3.3% of all bits** |
| mean bit-length of `abs(d)-1` over nonzero residuals | 1.24 |

Two things follow, and both are discouraging for the obvious next steps.

**The hybrid-uint token idea has a hard ceiling of 3.3%**, and would capture
only a fraction of that. Every bit outside those raw mantissa bits is already
context-modelled, so a better binarisation can only recover what the bypass
path throws away.

**74% of residuals are nonzero.** The "is it zero?" flag is therefore not the
cheap, heavily-skewed decision the design assumes — it costs about 0.83 bits per
sample, roughly a quarter of the whole file, and the sign bit costs most of
another quarter. For an unbiased predictor the sign is genuinely incompressible,
so that quarter is not recoverable by better modelling; it is only recoverable
by making the residuals *smaller*.

Which put the remaining gap in prediction and in the size of the model, not in
the entropy coder or the binarisation — and that is what the learned combiner
then acted on, for 1.10% held out, against the 0.1-0.4% every cheap contextual
trick before it had returned. The diagnosis in this section is the reason that
work was scoped as a combiner rebuild rather than as another context, so the
measurement was worth the trouble of taking.

### Re-measured after the learned combiner

Same three dev images, both columns produced by the same script so they are
comparable to each other. (The absolute figures in the table above do not
reproduce under this script — the original run's setup was not recorded well
enough to reconstruct — so trust the direction here, not the difference against
the older numbers.)

| | before | after |
|---|---:|---:|
| bits per plane sample | 3.492 | **3.415** |
| residuals that are **not** zero | 76.3% | **77.2%** |
| raw unmodelled bypass bits | 5.35% | **4.57%** |
| mean bit-length of `abs(d)-1` over nonzero residuals | 1.55 | **1.41** |

**The combiner produced *more* nonzero residuals, not fewer, and still won.**
That is the opposite of the obvious expectation — better prediction ought to
mean more exact hits — and it says something specific about what changed. The
combiner is not finding more pixels it can nail exactly; it is making the misses
much cheaper, dropping the mean magnitude bit-length from 1.55 to 1.41 and the
raw bypass share from 5.35% to 4.57%. A least-squares fit minimises squared
error, which is not the same objective as maximising exact hits, and here it
trades a few of the latter for a lot of the former. Anyone tempted to tune this
model against a zero-residual *count* should read that row first.

It also means the ceiling on a binarisation change moved the other way, from
3.3% to 4.6% of the file. Still not enough to justify the rebuild, but no longer
quite as dismissable as it was.

## Ruled out on cost

- **GLICBAWLS** — per-pixel weighted least squares over a causal search window;
  thousands of operations per pixel for ~4.5% over JPEG-LS, which is worse than
  MRP achieves far more cheaply.
- **paq8px-scale image modelling** — ~130 hand-written pixel predictors plus 6
  online least-squares filters, ~1000 mixer inputs, 20 concurrent mixer
  contexts. State of the art, three or four orders of magnitude over the budget
  for a readable Python reference implementation.
- **Static-histogram ANS backend** (what JPEG XL actually uses) — architectural
  rather than a modelling gain, and it would trade away the fully online
  adaptation that makes this codec's contexts free.

## Measured here, contradicting the obvious guess

`tools/headroom.py` computes the ideal cost of each predictor's residuals under
this codec's own context model. Two results shaped everything above:

1. **The adaptive coder already emits fewer bytes than the static conditional
   entropy of its own context set.** The entropy coding stage is not where the
   gap lives; local adaptation is already buying more than a static oracle over
   the same contexts would.
2. **Every alternative predictor scored far worse than MED — until the chroma
   wrap bug was fixed, after which they all scored better.** GAP went from
   +2.26% to -0.92%. A measurement that reverses sign when an unrelated bug is
   fixed is a warning about how much of predictor evaluation is really
   evaluating the representation the predictor sees.
