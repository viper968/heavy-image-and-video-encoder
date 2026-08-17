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

**The scratch buffers are narrower.** The numba path shared its arrays with the
reference implementation and so carried everything as int64. In C nothing is
shared, so the match model's history is `uint8` (it holds pixel values) and its
hash table is `int32` (it holds sample indices). At 1080p that is 6 MB of
random-access working set instead of 24 MB.

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
slice-independent format change, whose price is measured under "What is left in
the serial loop" below and is a good deal less flat than this file used to
assume.

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

## The whole program in C, and what that was and was not worth

`csrc/` now builds a standalone `hve` binary: containers, colour transform,
y4m, PNG and CLI, with no Python anywhere. Output is byte-identical to the
Python package on every test — all 24 Kodak images, the CIF clips, 1080p
Sintel — and each side reads the other's files.

**The speed answer is: almost none, and that was predictable.** Measured
before starting, only 15% of a still encode and less of a video encode was
still Python; the rest was already the C kernel. What the binary actually
removes is interpreter startup plus numpy's import, a flat ~0.15 seconds:

| job | binary | python | |
|---|---:|---:|---|
| still encode (768x512) | 0.23s | 0.38s | 1.7x |
| still decode | 0.26s | 0.40s | 1.5x |
| 1080p x16 encode | 2.79s | 2.93s | **1.05x** |
| 1080p x16 decode | 2.73s | 2.87s | 1.05x |

The relative win is entirely a function of how small the job is. Anyone quoting
the 1.7x without saying it is startup is quoting a constant, not a speedup.

**What it is actually worth is distribution.** 1.1 MB statically linked, no
interpreter, no numpy, no libpng, no zlib, no install step. PNG comes from a
vendored lodepng (zlib licence, one file); the alternative was making users
have libpng and cross-building it for Windows. A Windows target exists and is
written to be correct, but has never been run — there is no mingw-w64 on this
machine and no way to install one, and saying "it cross-compiles" without
having done it would be a claim rather than a result.

### The duplication this adds, and why it is a different bargain

Everything outside the pixel loop now exists twice: containers, RCT, y4m. That
is the same shape of problem as the numba path, with a different answer, and
the difference is worth being explicit about.

The numba mirror was 995 lines of adaptive model that had to be re-derived by
hand on every change. The container is a magic string, four varints and a
colour transform — small, totally specified, and it changes almost never. Both
directions are pinned by `tests/test_cli_binary.py`, which encodes the same
input through both and compares bytes.

The genuinely dangerous part is the *constants*: every ladder, tunable and
lookup table the model uses. Retyping those into C is the worst kind of
duplication, because one wrong digit does not fail to compile — it silently
produces a different model, and the only symptom is a slightly worse ratio. So
they are not retyped. `tools/gen_model_constants.py` emits
`csrc/model_constants.h` from the Python definitions, including the 4096-entry
stretch and squash tables and the motion-search cost table, and
`tests/test_generated_constants.py` fails if the checked-in header is stale.
Even `mv_penalty`, which is a default argument rather than a module constant,
is read out of the function signature rather than copied.

The one thing that could not be generated is the colour-transform decision,
which compares two floating-point entropies. A last-ulp difference between C's
`log2` and numpy's could in principle flip it. Measured across the 24 Kodak
images, the narrowest margin between the two candidates is **17%**, so it
cannot; that is a fact about the data, not a proof, and it is written down in
`csrc/transform.c` so the next person knows what would break it.

### A heap corruption that only 1080p could find

Ceiling division. Python spells it `-(-h // 16)` and the C transcription
`-(-h / 16)` looks identical and is wrong: `//` floors, `/` truncates toward
zero, so a 1080-line frame gets 67 block rows instead of 68 and the motion
search writes one row past the end of the array.

It survived every test because **every clip in `testdata/` divides exactly by
16** — 352x288 has no remainder, so the two spellings agree. The first 1920x1080
encode aborted in malloc. Found with ASan in about a minute; the fix is an
`hve_ceil_div` helper and the regression test uses 70x100, 72x96 and 33x47
frames, which fail on all three without it.

This is the third time in this project that C's integer semantics have differed
from Python's in a way that compiles cleanly and produces wrong output rather
than an error, after `/` versus `//` in the kernel and `>>` on negatives. The
list in `docs/HANDOFF.md` is not academic.

## What is left in the serial loop, and what parallelism would cost

Two questions, answered by measurement rather than by estimate.

### Without touching the format: about 20%, and then it is flat

Profiling the standalone binary (which, unlike a dlopened .so, `perf` can
resolve) put 96% of decode time in `hve_code_plane` with no dominant line
inside it. The one structural item that stood out was **ladder lookups**: the
pixel loop ran ten linear scans per sample over sorted arrays of up to fifteen
entries, each an unpredictable branch. Every input is a small non-negative
quantity with a known bound — a sum of three folded byte differences, a
neighbour magnitude — so they tabulate into about 1.5 KB that stays in L1.

Exact, format-preserving, and worth:

| workload | before | after | |
|---|---:|---:|---|
| still encode (kodim05) | 0.245s | 0.189s | **-23%** |
| CIF video encode x16 | 0.434s | 0.353s | **-19%** |
| 1080p encode x16 | 2.840s | 2.786s | -1.9% |
| 1080p decode x16 | 2.625s | 2.543s | -3.1% |

The split is the interesting part. Sintel is near-static, so almost every
residual is zero, the scans exit after an iteration or two, and there was
little to remove. Content with real residuals walks much further down the
ladders, which is where the 20% lives.

Re-profiling afterwards showed a flat distribution: the match model's random
table access at 4.4%, the combiner's dot product and update at ~9% together,
the two APM interpolations at ~3%, the colour-transform decision at ~5% of a
still encode. Nothing else is worth a targeted change, so **this is roughly the
end of what implementation work can do.**

### With a format change: cheap on photographs, expensive where it would help most

The pixel loop is strictly serial — one range coder, one adaptive model, every
pixel depending on all prior ones — so threading it needs independent slices,
each with its own coder and model state and no prediction across its top edge.
Earlier notes in this file estimated that at "0.3-1% ratio". That estimate was
never measured. Measured, by encoding N independent horizontal strips and
summing:

| content | ratio | 2 slices | 4 slices | 8 slices | 16 slices |
|---|---:|---:|---:|---:|---:|
| 18 held-out Kodak | 2.75x | -0.04% | +0.52% | +1.42% | +2.84% |
| foreman CIF x16 | 2.9x | +0.32% | +1.80% | +2.61% | — |
| Sintel 1080p x16 | 1608x | +2.03% | +4.65% | +8.57% | +16.28% |

So the old estimate was right only for stills at two or four slices, and badly
wrong elsewhere.

**The penalty tracks how much of the file is learned model state, not
resolution.** Four slices, same clip length, across the whole video corpus:

| clip | ratio | 4-slice penalty |
|---|---:|---:|
| mobile CIF | 2.0x | +2.13% |
| bus CIF | 2.3x | +2.07% |
| foreman CIF | 2.9x | +1.80% |
| container CIF | 3.4x | +6.50% |
| akiyo CIF | 7.7x | +14.62% |
| Sintel 1080p | 1608x | +4.65% |

Busy content pays about 2%; near-static content pays five to seven times that,
because when the content is nearly free to code, what is left *is* the model
converging, and each slice has to converge again from scratch. Sintel breaks
the monotonic ordering because its slices are 1080p-sized and so have far more
data to amortise that over — both terms matter, and neither alone predicts it.

The practical consequence is awkward and worth stating plainly: **slicing costs
least on the content that is already slowest to encode, and most on the content
that is already fastest.** A 16-thread decode of photographic stills for 2.8%
is a real option; a 16-thread decode of near-static video for 16% is not, and
that is the case where 2.7 seconds actually hurts.

If this is ever built, the shape the measurements support is a *small* number of
slices — two or four — rather than one per core, and making the count a header
field so the encoder can choose it from the content rather than from the
machine.

## Buying speed with ratio: the model-stage ablation

The goal here was explicitly speed — "ballpark of h264", with 10-15% ratio loss
acceptable. So every stage of the model got a switch (`params[P_FEATURES]`, a
bitmask; `hve --features N`) and was measured for what it costs in time and buys
in ratio. Every configuration below was checked to still round-trip losslessly.

kodim13, a busy photograph (baseline 564,475 bytes, 0.247s encode):

| config | bytes | vs base | encode | decode |
|---|---:|---:|---:|---:|
| everything on | 564,475 | — | 0.247s | 0.268s |
| -match | 566,650 | +0.39% | 0.173s | 0.195s |
| -match -lms | 576,124 | +2.06% | 0.135s | 0.181s |
| -match -lms -blend | 585,602 | +3.74% | 0.124s | 0.154s |
| primary context only | 588,712 | +4.29% | 0.097s | 0.115s |

**The entire context-mixing apparatus — mixer, both APM stages, the length-bin
mixer, the learned combiner, the blend and the match model — is worth 4.29% on
this image and costs 2.5x in time.** That is the central trade in this codec,
and until now it had never been priced.

Sintel 1080p x16 (baseline 30,944 bytes, 2.93s) tells a completely different
story, and this is the surprise:

> **This clip is not representative, and the conclusion below is overstated.**
> 30,944 bytes for 16 frames of 1080p should have been the tell: the segment
> starts at frame 0 of the trailer, and frames 0-7 are black. They encode to
> 1,660 bytes and are byte-identical under every preset, so the ablation is
> measuring almost nothing. On the trailer's genuinely busy segments the same
> three stages **cost** 1.4% to 7.9% rather than saving 18.8%. The tables in
> this subsection are left as measured, with the corrected figures in "What the
> full trailer says" below.
> The direction of the finding survives - the spatial stack does not earn its
> cost on 1080p video - but the magnitude does not, and the speed-up is the
> part that was actually worth having.

| config | bytes | vs base | encode |
|---|---:|---:|---:|
| everything on | 30,944 | — | 2.93s |
| -match | 28,643 | **-7.4%** | 2.96s |
| -match -lms | 26,810 | **-13.4%** | 1.95s |
| -match -lms -blend | 25,120 | **-18.8%** | 1.73s |
| primary context only | 38,699 | +25.1% | 1.37s |

Three stages are **actively harmful** on this content: dropping the match model,
the learned combiner and the weighted blend makes the file **18.8% smaller and
1.7x faster at the same time**. There is no trade to make — they are simply
wrong here. The spatial prediction machinery was built and tuned on
photographs, and on near-static high-resolution video where most blocks are
temporally predicted it is modelling noise. Note that this finally explains the
long-standing open item recorded above as "the match model costs 7.3% on 1080p
Sintel": it was not a mystery about the match model, it was the whole spatial
stack being mis-applied.

The mixer and the APM stages, by contrast, are load-bearing everywhere: turning
them off as well (`primary context only`) costs 25% on Sintel.

### What the full trailer says

Six 16-frame segments taken across the whole 3.9GB trailer rather than one at
frame 0, `max` against `fast`, one slice each:

| segment | `max` bytes | `max` | `fast` bytes | `fast` | ratio | speed |
|---|---:|---:|---:|---:|---:|---:|
| @0s (black) | 36,849 | 2.66s | 25,067 | 1.37s | **-31.97%** | 1.94x |
| @10s | 108,766 | 2.67s | 98,981 | 1.40s | -8.99% | 1.91x |
| @45s | 294,267 | 2.93s | 289,779 | 1.59s | -1.52% | 1.85x |
| @40s | 3,793,934 | 3.64s | 3,635,971 | 1.89s | -4.16% | 1.93x |
| @30s | 5,527,928 | 4.30s | 5,966,835 | 2.20s | **+7.93%** | 1.96x |
| @20s | 10,416,338 | 4.54s | 10,567,064 | 2.92s | +1.44% | 1.55x |

The speed-up is the stable part: **1.55x to 1.96x everywhere**, because it comes
from deleting work rather than from the content. The ratio is not stable at all,
and the sign of it flips. On the two genuinely busy segments - the only ones
where the file is big enough for the trade to matter in absolute bytes - the
spatial stack *does* earn its cost, and dropping it costs 1.4% and 7.9%.

Note it is not a clean function of how busy the content is: @40s at 237KB per
frame prefers `fast` by 4.2%, while @30s at 345KB per frame prefers `max` by
7.9%. So there is no cheap density heuristic that would let the encoder choose,
which is a second, independent reason the `auto` preset was dropped.

What survives from the original finding is the useful half: `fast` buys close to
2x for a few percent, which is a good trade and the reason the preset exists. It
is a trade, not the free win that a clip of black frames suggested.

### Combining the preset with slices

Independent slices were measured earlier at +4.64% for four at 1080p. Applying
both, against the current codec's 30,944 bytes at 2.93s, with parallel time
estimated as the slowest slice:

| config | bytes | vs today | wall time | speedup |
|---|---:|---:|---:|---:|
| today (all on, 1 slice) | 30,944 | — | 2.93s | 1.0x |
| fast preset, 1 slice | 25,120 | -18.8% | 1.87s | 1.6x |
| fast preset, 2 slices | 25,565 | -17.4% | 0.94s | 3.1x |
| fast preset, 4 slices | 26,111 | **-15.6%** | 0.50s | **5.9x** |
| fast preset, 8 slices | 27,094 | -12.4% | 0.46s | 6.4x |
| fast preset, 16 slices | 29,006 | **-6.3%** | 0.31s | **9.5x** |

x264 lossless on the same clip is **34,607 bytes in 0.3s**. So the fast preset
at 16 slices reaches x264's encode speed while producing a file **16.2% smaller
than x264 and still 6.3% smaller than this codec produces today**. At four
slices it is 5.9x faster than today, 15.6% smaller than today, and 24.5%
smaller than x264.

The 10-15% ratio budget that was offered turns out not to be needed for video at
all. For stills it is: the fast preset costs +3.74% there, so the feature set
has to be **chosen per file and carried in the header**, not fixed globally.

Two caveats. The wall-clock column assumes perfect thread scaling and ignores
slice setup, so treat it as an upper bound; and this is one 1080p clip, chosen
originally because it was the one that made encode time look bad. A busier 1080p
clip would behave more like foreman, where the spatial stack does earn its cost.

> **Both caveats turned out to be the whole story.** This table is built on the
> same near-black clip, so every ratio column in it is wrong, including "16.2%
> smaller than x264" - a 30,944-byte baseline is 8 black frames and 8 nearly
> black ones. The wall-clock estimates were also optimistic: real 16-slice
> encoding needed a thread-budget fix and a pyramid fix before it came close
> (see "Slices" below). For the measured version of this comparison on content
> that exercises the codec, see the park_joy table in `README.md`: 22,497,649
> bytes in 1.38s against x264 lossless veryslow's 23,108,086 in 2.35s. The
> conclusion held - it just had to be re-earned on real frames.

### What the literature says we should do instead of slices

A survey of how the mainstream codecs get parallelism turned up one design that
directly targets what these measurements say the cost actually is.

Independent slices are expensive here for exactly one reason: **each slice has
to relearn the model from scratch**, and the penalty tracks how much of the file
is learned state. HEVC's Wavefront Parallel Processing does not reset. Each CTU
row starts from a **copy of the entropy-coder state as it stood after the second
CTU of the row above** — a checkpoint, not a reset — so the model stays warm and
only the ramp-up costs anything. (Habermann et al., *Improved Wavefront Parallel
Processing for HEVC Decoding*; the same paper measures the real dependency
distances at under 1 CTU for context, 1.5 for intra prediction and 1.66 for
motion vectors.) Published WPP scaling is 8.7x on 12 cores for 4K, against 9.3x
for fully independent tiles — nearly the same parallelism for a fraction of the
compression cost.

That is the obvious next move for this codec, and it should be cheap precisely
because our measured penalty is a relearning penalty.

Two other findings worth recording:

**Multi-symbol entropy coding.** AV1 replaced VP9's binary bool coder with a
CDF-based multi-symbol coder over alphabets up to 16, and the AOM design paper
claims "more than a factor 2 reduction in throughput cost for typical coding
scenarios over pure binary arithmetic coding", framed as compression-neutral.
This codec is entirely binary — every pixel is a chain of binary decisions — so
the same restructuring is available in principle. No controlled ablation of the
entropy coder alone was ever published, so the "compression-neutral" half is the
design team's claim rather than an independently verified number.

**Bypass bins are the cheap lever.** The single largest throughput win HEVC took
over H.264's CABAC was moving bits out of context-coded mode into bypass
(equiprobable, no context, no table lookup, several per cycle): a 25-31%
BD-cycle reduction (Sze & Budagavi, IEEE SiPS 2013). This codec already bypasses
the low mantissa bits; the sign bit and the upper mantissa bits are the obvious
candidates to examine next.

### Licensing: what could actually be borrowed

The question was whether code could be taken from AV1, VP9, H.264 or H.265. This
project is MIT, and the answer splits cleanly:

| project | licence | borrowable into MIT? |
|---|---|---|
| libaom (AV1), libvpx (VP9), dav1d, rav1e | BSD-2/3-Clause + patent grant | **yes**, carrying the notices and PATENTS file |
| SVT-AV1 | BSD-2-Clause, BSD-3-Clause-Clear from v0.9 | yes, check the version |
| libjxl | BSD-3-Clause | yes (patent posture unverified) |
| FFmpeg / FFV1 | LGPL-2.1-or-later | reference only, not copy-paste |
| **x264, x265** | **GPL-2.0-or-later** | **no** — copyleft would take the whole project |

So the H.264/H.265 implementations are off the table for source reuse, and the
AV1/VP9 family is not. In practice the useful thing to borrow from them is
design rather than code — the entropy-coder structure and the WPP/tile
synchronisation rules are described in papers, and this codec's data structures
look nothing like theirs.

## Slices: why not wavefronts, and what they actually cost

The pixel loop is strictly serial, so the only way onto a second core is to cut
the picture into independent pieces.

**Wavefronts are the better design and are not available here.** HEVC's WPP
starts each CTU row from a *copy* of the entropy state as it stood two CTUs into
the row above - a checkpoint, not a reset - which is why it costs far less than
independent tiles. That works because CABAC's entire context set is a few
hundred bytes. This model is **4.5 MB**: 472 KB of probability banks, mixers and
APMs, 13 KB of combiner weights, and a 4 MB match hash table. Checkpointing that
per CTU row would be 304 MB of copying per 1080p frame. The measurement that
made wavefronts attractive - that our slicing penalty is a *relearning* penalty -
is still right; the remedy is just out of reach at this model size. That is a
consequence of the compression architecture, not an implementation shortcut.

So: independent horizontal slices, each with its own model, coder and motion
search. `csrc/slice.c`, `--slices N`.

### What it costs

Cost is governed by how much data each slice has to amortise its relearning
over, so it falls as the frames get bigger:

| content | 4 slices | 8 | 16 |
|---|---:|---:|---:|
| park_joy 1080p | +0.24% | +0.40% | +0.69% |
| in_to_tree 1080p | +0.22% | +0.29% | +0.46% |
| Sintel 1080p @400 | +0.40% | +0.78% | +1.52% |
| kodim13 still (768x512) | +0.70% | +1.41% | +2.46% |
| foreman CIF | +1.80% | +2.61% | +4.10% |
| Sintel @0 (near-black) | +4.70% | +8.73% | +16.43% |

On real 1080p video sixteen slices costs **under 1%**. The expensive cases are
small frames and pathologically compressible content, which is why the default
is one slice per ~250k pixels capped at the core count - 8 at 1080p, 1 at CIF,
1 for a typical photograph.

### What it buys, and two bugs found getting there

park_joy, 16 frames of 1080p, on a 16-core i5:

| preset | slices | bytes | vs base | encode | decode |
|---|---|---:|---:|---:|---:|
| max | 1 | 22,341,378 | — | 7.89s | 7.07s |
| max | 4 | 22,395,863 | +0.24% | 2.42s | 2.41s |
| **max** | **16** | **22,497,649** | **+0.69%** | **1.38s** | **1.29s** |
| fast | 1 | 23,214,813 | +3.90% | 4.30s | 4.87s |
| **fast** | **16** | **23,366,845** | **+4.58%** | **0.83s** | **0.76s** |

x264 lossless on the same clip, also using all 16 cores: **23,108,086 bytes in
2.35s** at `-preset veryslow`, **23,255,288 in 0.84s** at `-preset medium`.

So the full model at sixteen slices is **2.6% smaller than x264 veryslow and
1.7x faster**, for 0.69% against the unsliced encoder. That is past the goal
without spending any of the 10-15% ratio budget that was on offer, and without
needing the fast preset at all - the preset is now only interesting if you want
to go below x264 `medium`'s time.

Two things had to be fixed before it scaled, and both were mine:

**Thread oversubscription.** The motion search is itself threaded, so every
slice spawned a full core count of search threads: 256 on a 16-core machine at
16 slices. Total CPU work rose 4.5x and wall time at 16 slices was *worse* than
at 4. The slice runner now divides the budget.

**A pyramid cliff between 4 and 8 slices.** The motion search skips its
coarse-to-fine pyramid below `PYRAMID_MIN_PIXELS`, a rule written to stop the
pyramid hurting at CIF. A 1920x135 slice has fewer pixels than CIF but full
horizontal resolution, so the test switched the pyramid off and made the search
exhaustive - 289 positions instead of about 43. Judging the pyramid by the whole
frame rather than the strip took 16 slices from 1.92s to 0.90s. Area was always
the wrong criterion; a thin wide strip is not a small picture.

## Re-tuning after the presets: two inherited constants were wrong

Every constant here was fitted with the full model switched on. The `fast`
preset drops three stages, so `tools/sweep_preset.py` re-sweeps a constant and
reports the dev split **separately for each preset**, rebuilding the binary per
candidate value.

The result was less dramatic than expected and interesting for a different
reason: the constants that turned out to be wrong were wrong for *both* presets,
because they had been inherited from LPAQ and never checked against this model
at all.

| constant | was | swept to | dev split |
|---|---:|---:|---|
| `rc.ADAPT_SHIFT` | 6 | **6** | already optimal, both presets |
| `mix.Mixer` rate | 7 | **24** | -0.057% (max), -0.045% (fast) |
| `mix.APM` rate | 7 | **8** | -0.036% (max), -0.041% (fast) |

Both new values were then checked on the **held-out** split, which is the only
number that counts:

| | before | after | |
|---|---:|---:|---:|
| 18 held-out Kodak, max | 7,711,478 | 7,702,223 | **-0.120%** |
| 18 held-out Kodak, fast | 7,913,577 | 7,906,274 | -0.092% |
| 2 held-out clips, max | 1,151,218 | 1,150,232 | **-0.086%** |

Small, but free - the same operations with a different constant - and it closes
the JPEG XL gap from 6.99% to **6.86%**. The mixer rate sweep is monotone out to
about 24-32 and then turns over, so LPAQ's 7 was not near the optimum for this
model; it was simply never questioned. The APM optimum at 8-9 is shallow.

A per-preset constant would have been possible - both sides know the preset from
the header - but neither of these wanted a different value per preset, so both
stay global.

### A stale cache, caught by the byte-identity tests

Changing a constant broke 27 tests, and the cause was not the model. `native.py`
rebuilds its cached extension when a source file is newer, and its list of
headers to watch contained only `hve.h` - so `model_constants.h` changing did
not invalidate the cache. The Python path went on running the previous model
while the standalone binary ran the new one.

This is precisely the "stale C kernel means every number you measure comes from
the old model" hazard that `docs/HANDOFF.md` warns about, and it took about
fifteen minutes to find because the failure was loud and specific rather than a
quietly worse ratio. The header list is now a glob, so a new header cannot be
forgotten. Worth noting as the argument for keeping those tests expensive
enough to be meaningful.

## AV1's multi-symbol coder: measured, and it cannot pay here

AV1 replaced VP9's binary arithmetic coder with the Daala multi-symbol coder,
which codes one N-ary symbol against a CDF instead of N-1 binary decisions, and
the AOM work reports better than 2x entropy-coding throughput for it. The
obvious question is whether that transfers.

It does not, and the reason is worth writing down, because the headline number
is about a part of the codec that barely exists here.

### The null-coder bound

Rather than argue from a profile, delete the coder. A build with every range
coder operation stubbed out - no renormalisation, no carry propagation, no bytes
written, probability updates retained so the model still walks the same path -
is an upper bound on *any* entropy-coder change, multi-symbol included. It
produces garbage, which is fine, because the question is only how long it takes.

16 frames of 1080p Sintel, one slice, encode:

| | real coder | null coder | coder's share |
|---|---:|---:|---:|
| `fast` preset | 2.01s | 1.87s | **7.0%** |
| `max` preset | 3.51s | 3.76s | *below the noise* |

The `max` row came out negative - the build with less work in it ran slower -
which is the useful part of the measurement: at that point code layout moves the
number more than deleting the entropy coder does. Wall-clock noise on this loop
is a few percent, so the later measurements here use retired cycles instead.

So the ceiling for a perfect multi-symbol coder is seven percent, and a real one
would not reach it. That alone settles it, but the profile says why.

### Where the time actually goes

`perf record` on the `fast` preset, attributed to source lines and bucketed by
phase. `hve_code_plane` is 78% of samples; motion search is larger in CPU terms
but runs on 16 threads, so it is small in wall-clock. Shares below are of the
serial coding loop:

| phase | share |
|---|---:|
| zero flag: 5 experts + mixer + 2 APM stages + update | **35.6%** |
| context indices + ladder lookups | 19.7% |
| prediction: MED, activity, blend, LMS, match | 15.1% |
| LMS update and bookkeeping | 8.4% |
| sign, exponent, mantissa coding | 8.4% |
| inter reference addressing | 6.3% |
| **range coder arithmetic** | **4.6%** |
| ladder bisect and floor-division helpers | 1.5% |

The entropy coder is under five percent. What AV1 actually buys with the
multi-symbol coder is one *modelling* decision per symbol instead of per bit -
and here the expensive decision, the zero flag, is already once per pixel. A
multi-symbol coder cannot make it less than that. The only place the structure
would fit is the exponent's unary chain, which is inside an 8.4% bucket.

There is a deeper reason it does not transfer. The multi-symbol coder needs a
CDF over the alphabet, cheap to look up and cheap to update. This codec's
probabilities come from mixing five experts through a logistic mixer and two APM
stages - a binary quantity by construction. Producing a CDF from a context
mixer means running the mixer per bit anyway, which is the cost the change was
supposed to remove.

### What the ablation says about the ceiling for any model change

The feature gates price the whole model at once. Same clip, one slice:

| features | encode | bytes |
|---|---:|---:|
| `max` (127) | 3.60s | 3,793,934 |
| `fast` (120) | 1.94s | **3,635,971** |
| mixer only (8) | 1.59s | 3,660,905 |
| NBMIX only (64) | 1.50s | 3,779,028 |
| nothing (0) | 1.49s | 3,784,580 |

Switching the entire model off saves 23% against the `fast` preset. That is the
ceiling for *every* modelling change combined, including becoming AV1-shaped,
and the remaining 1.49s is prediction, context formation and memory traffic.
This codec is not entropy-coding-bound and it is not really model-bound either;
it is bound by doing roughly 760 instructions of serial, data-dependent scalar
work per sample.

Two things follow. The first has been acted on: a per-sample integer division
was sitting in that 6.3% addressing bucket, worth about 5% for free.

### The structural one, not yet built or costed

The loop is serial because each pixel's context depends on its neighbours'
*decoded* values. On the encoder that constraint is imaginary: the codec is
lossless, so decoded equals source, and the encoder has the whole source in
hand before it codes anything.

Every context index here is a pure function of source pixels - MED and GAP, the
activity and sign bins, `err_sum`, the luma error map, the match hash, the LMS
taps. Nothing in the context formation needs a coded bit. Only the
probabilities themselves - the expert tables, mixer weights and APM bins - are
order-dependent, and those are genuinely serial.

So the encoder could split in two: a first pass computing predictions and
context indices for a whole plane, which is local per pixel and vectorises, and
a second serial pass doing only the model and the coding. That is roughly the
prediction bucket plus much of the context bucket, so 25-35% of encode, and it
changes no bits at all - the decoder is untouched and the format is untouched.

Two caveats before anyone believes the number. The LMS filter and the match
model adapt in scan order, so they stay sequential even in the first pass; only
the local per-pixel work vectorises. And it helps the encoder only - the
decoder genuinely cannot know a value before decoding it, so decode stays as it
is, and decode is currently the same speed as encode. This is an estimate from
the profile above, not a measurement, and nothing here has been built.

## The cost of a third implementation, and why there are two again

Adding the C made three implementations of one loop: `model.py` defining the
format, `fast.py` mirroring it in numba, `csrc/` mirroring it in C. That was
one too many, and `fast.py` was deleted. The reasoning is worth recording,
because "keep the fallback, it costs nothing" is the intuitive answer and it is
wrong.

**The tax is not occasional, it is total.** Every commit in this repo's history
that touched `model.py` also touched `fast.py` — five out of five:

    5c64cc5  Rewrite the motion search
    f200a4b  Half-pel motion vectors
    d057abc  Replace the averaging blend with a learned combiner
    6ba47bd  Second APM stage
    29bc9af  Let the sign and length bins see the match model

Mirroring is not a thing that happens sometimes when the model changes. It is
what changing the model *is*. Compression experiments are this project's entire
activity, so a third file taxes the one thing it exists to do.

**The tier it bought was narrow and expensive.** Measured on one input, encode:

| path | s/megapixel | a 768x512 Kodak photo |
|---|---:|---:|
| C | 1.1 | 0.4s |
| numba | 11.4 | 4.5s |
| pure Python | 148.7 | 58s |

So `fast.py` served exactly one population: someone with numba installed and no
C compiler, for whom the alternative is a minute per photo. That is real —
Windows without MSVC, slim containers — but it costs **211 MB of dependency**
(llvmlite alone is 180 MB) to be 2-4x slower than a compiler that is usually
already on the machine. numba also lags new Python releases by months, so the
fallback is least likely to work exactly when someone is on a new Python.

**And it had stopped adding coverage.** `fast.py` earned its keep as a second
opinion while it was the only accelerated path. Once the C was pinned directly
against `model.py`, the third implementation was redundancy, not coverage — and
in this very session it actively *hid* a broken test: adding the native backend
silently broke `test_video_fast_path_is_byte_identical`, which disabled only the
numba path to reach the reference and so began comparing native against itself.
It passed, testing nothing, until it was read.

Deleting it changed no output byte on any test clip or image, and made the suite
faster.

### What that costs, stated plainly

Someone without a C compiler now falls from 4.5s to 58s per photo. That is a
real regression for them, accepted deliberately: the fallback is a correctness
guarantee — you can always decode a file — not a usable working mode. The
proper fix is packaging with prebuilt wheels, which this repo does not have and
which would be a better use of the effort than a hand-mirrored kernel.

### What keeps two implementations honest

The C is written as a deliberately boring transcription of `model.code_plane` —
same variable names, same order of operations, same comment anchors — so that
diffing them stays possible when the model changes.

The awkward consequence of losing the middle tier is that byte-identity is now
checked *against a 150s/megapixel reference*. So those tests run on deliberately
tiny inputs (a 40x56 photo crop, 32x48 frames with 16x24 chroma), chosen to
still cover odd dimensions, single rows and columns, the fourth plane kind, and
subsampled chroma with real sub-pixel motion. Real photographs and real clips
are covered by roundtrip and invariant tests, which only need the fast path.
Enlarging the identity cases would make the suite slow enough to stop being run,
which is a worse failure than the one it would be guarding against.

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
