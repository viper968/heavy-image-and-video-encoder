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

That was built here and did not pay (above). What is left, in the order I would
try it:

1. **More genuinely different experts.** Every expert added so far keyed on the
   same neighbourhood statistics and the returns collapsed accordingly. The
   experts that would actually disagree are ones seeing different *data*: a
   match model (has this exact neighbourhood pattern occurred before in this
   image, and what followed it?), and a longer-range model reaching several
   rows up. paq8px carries ~130 predictors for exactly this reason.
2. **Mixing on the magnitude bins**, not just secondary estimation. The bins
   currently have one context model each; they are the second largest cost in
   the file.
3. **A second mixing layer**, which every serious context-mixing compressor
   has and which no source I found isolates a number for.
4. **The hybrid-uint token design** — give small residuals their own jointly
   modelled symbol instead of decomposing every one into is-zero / sign /
   unary. Structurally strictly stronger than the current binarisation.

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
