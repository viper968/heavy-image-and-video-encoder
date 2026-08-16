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
  raw bypass bits are 3.3% of the file, which is the entire ceiling for any
  binarisation change, and a redesign would capture only part of it.

What is genuinely left is one item, and it is large:

**A proper mixing network over many more predictors.** paq8px runs ~130 into a
real mixer. This codec runs six into an *error-weighted average*, and four
separate attempts to add predictors to that average have now come back negative
(GAP, least squares, the match value, and the 2W-WW / 2N-NN trend pair). They
fail the same way each time, so the finding is about the combiner and not about
any individual predictor: an average is dragged by a volatile member even when
that member is down-weighted, whereas a logistic mixer learns to ignore it. The
match model itself only started paying when it stopped being a vote and became a
hard switch. Replacing the blend with a mixer is the one change consistent with
every negative result here, and it is a rebuild rather than an increment.

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

Which puts the remaining gap where the first agent's research said it was: in
prediction and in the size of the model, not in the entropy coder or the
binarisation. Every cheap contextual trick here has now returned between 0.1%
and 0.4%, and four separate attempts to improve prediction have come back
negative, because this codec's blend is an *averaging* combiner and averaging is
harmed by volatile predictors even when they are down-weighted. paq8px carries
~130 predictors into a proper mixing network rather than an average, which is a
different architecture, not a bigger version of this one.

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
