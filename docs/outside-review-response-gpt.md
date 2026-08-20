My read is that HVE is already in the territory where another “smarter predictor” is unlikely to close a 6.86% still-image gap. The stronger opportunity is **changing what the model is conditioned on and how the residual is represented**, plus exploiting spatially local adaptation rather than making the current mixer deeper.

One caveat: I am comparing against the current public libjxl/AV2/VVC designs and literature, not your private implementation details. The current libjxl Modular encoder still has several mechanisms beyond “MA tree + weighted predictor + hybrid uint”: per-group RCT selection, additional predictor families/parameters, previous-channel MA properties, palette/local-palette machinery, patches, and LZ77/RLE choices. ([GitHub][1])

## 1. Stills: where I think the 6.86% is actually living

My best guess is:

**the missing third thing is not another predictor; it is richer *symbol/context factorization*, especially predictor identity + local context clustering/conditioning, followed by local color transform selection.**

There are three distinct layers here that are easy to conflate:

1. **What residual do you produce?**
2. **What context predicts that residual?**
3. **How many statistically distinct residual distributions do you allow?**

You have invested heavily in #1 and #2 through multiple predictors and a fairly sophisticated online mixer. But your description suggests that #3 is comparatively weak.

### The first experiment I'd run: make predictor identity an explicit context

This is conspicuously missing.

Suppose your weighted blend produces

[
e = x-\hat{x}
]

but two pixels with identical activity, gradient, confidence, etc. arrive from very different predictor regimes:

* MED selected because you're sitting on an edge,
* NLMS dominates in a textured patch,
* match model dominates in a repeated region,
* GAP dominates in a smooth directional region.

Those residuals need not have remotely the same conditional distribution.

Your current mixer can *indirectly* learn this, but you are asking the probability model to infer something that is essentially already known at the encoder/decoder: **which predictor regime generated the residual**.

I would give the zero/exponent models explicit low-cardinality state such as:

```text
predictor_class:
  MED
  GAP
  weighted
  NLMS
  match
  blend-dominant
  ...
```

and preferably a coarser representation:

```text
dominant predictor
confidence / winner margin
predictor disagreement
```

The important one is actually **predictor disagreement**.

If five predictors give:

```text
127, 128, 127, 128, 127
```

the residual distribution is fundamentally different from:

```text
90, 128, 162, 141, 103
```

even if local activity is identical.

So I'd test contexts built from:

[
\max_i p_i-\min_i p_i
]

and perhaps the sorted first/second-best predictor disagreement, quantized logarithmically.

That is a much more direct state variable than several of your current proxies.

### Second: explicit residual-shape context, not just residual-energy context

You mention:

* local error energy,
* gradient pairs,
* sign pairs,
* activity.

What I don't see is a compact context describing the **shape of the previous few residuals**.

FFV1 does something conceptually important here: its context is built from several *signed neighboring differences* rather than a generic activity scalar. The published FFV1 design uses five quantized difference terms involving left/top/diagonal/right-neighbor relationships and maps that to a residual coding context. ([IETF Datatracker][2])

I'd test quantized:

```text
e_left
e_top
e_topleft
e_topright
e_left - e_topleft
e_top - e_topleft
```

or, even better, predictor residuals from a small causal stencil.

Your current model seems to know “how busy is this area?” more than “what *kind* of residual process am I in?”

Those are different.

### Third: local RCT selection

This is the most obvious JPEG XL feature you have not described as present.

Current libjxl Modular can choose among multiple reversible color transforms **per modular group**, rather than only choosing one transform for the whole image. Its interface exposes group sizes of 128×128 through 1024×1024 and multiple RCTs; higher effort also explores more RCT choices and properties. ([GitHub][1])

That is particularly interesting because your colour transform is:

> chosen per image by comparing zeroth-order residual entropy

That is a very different adaptation granularity.

For photography, color correlation is not necessarily stationary across the frame. Skin, foliage, sky, artificial lighting, saturated objects, etc. can have very different optimal channel relationships.

A **cheap group-local RCT selector** has a much better chance of producing a real gain than adding another expert to your zero model.

I'd test:

```text
frame RCT
vs
64x64 / 128x128 local RCT
vs
your current global RCT
```

with *no other model changes*.

I would not be surprised by a few tenths of a percent; I also would not assume it closes the whole 6.86%.

### What about JXL's context clustering?

This is the other major thing I'd explicitly test.

Current libjxl does not merely learn an MA tree; it also performs histogram/context clustering, and the encoder has progressively more exhaustive clustering, LZ77, and hybrid-uint selection at higher effort. ([GitHub][3])

Your online logistic mixture is doing something more powerful in one sense, but it is not equivalent.

A useful way to think about it is:

> **Your model combines predictors of a probability. JXL also tries to discover which symbols belong to the same statistical distribution.**

Those aren't interchangeable.

I would therefore run a very diagnostic experiment:

**freeze prediction completely**, dump the sequence of model contexts and actual symbols, and measure:

* empirical entropy per existing context,
* entropy after clustering contexts by histogram similarity,
* gain from clustering with hindsight,
* gain from online adaptation from the same initialization.

If hindsight clustering gives essentially zero, stop chasing it.

If it gives 2–4%, you have found the missing degree of freedom.

### Something else you should explicitly measure: LZ/RLE/repetition

Your rejected list does **not** say you tested LZ77/backreferences.

That's significant because current JXL Modular can use RLE or LZ77 in addition to its entropy coding; its high-effort encoder can perform progressively more exhaustive LZ77 matching/parsing. ([GitHub][4])

I would expect it to matter more on:

* synthetic / screen-like images,
* repeated texture,
* exact duplicated structures,

than on Kodak photography.

So it is probably **not the whole 6.86%**, but it is a genuinely untested orthogonal mechanism.

### My ranking for the still-image gap

| Experiment                                          |                              My prior |
| --------------------------------------------------- | ------------------------------------: |
| Explicit predictor identity + disagreement contexts |                              **High** |
| Local RCT per 64–256 px group                       |                              **High** |
| Context clustering of residual distributions        |                              **High** |
| LZ77/RLE exact-repeat path                          |                                Medium |
| More predictor families                             |                                   Low |
| More APM/SSE stages                                 |                          **Very low** |
| Better range coder                                  | **Near zero** given your measurements |

The APM result you already measured is exactly the kind of evidence I'd trust: you're approaching diminishing returns in the probability-refinement layer.

---

# 2. The 1080p motion puzzle

I think your working theory is **mostly right**, but I would phrase it more carefully:

> At high resolution, the problem is probably not “temporal prediction is inherently useless”; it is that your *current temporal predictor is not predictably better than the already excellent spatial predictor once exact lossless residual statistics and signalling cost are included.*

That distinction matters.

Your three observations are very strong:

* widening search monotonically makes the file larger,
* vectors are not clipping,
* increasing the inter penalty actually favors **fewer** inter blocks.

Those are all evidence against “the motion search radius is the problem.”

And the 0.9% inter-block selection rate on 1080p is not merely small; it says the encoder's own residual model is telling you that the temporal candidate usually isn't competitive.

### Is lack of rate-aware motion search enough to explain it?

**No, not by itself.**

It can certainly cause bad decisions, but your penalty sweep is already close to the important experiment.

If the correct RD objective were roughly

[
J = D_{\text{residual}} + \lambda R_{\text{MV}}
]

then increasing MV penalty should progressively push the encoder toward fewer inter blocks.

You did that.

And the best point moves toward **more intra**, not less.

That's strong evidence that MV signalling cost is not hiding a huge reservoir of gains.

The more interesting possibility is that your motion candidate is intrinsically weak for lossless coding.

## Why lossless temporal prediction behaves differently

For lossy coding, a motion vector that gives a visually close prediction is great.

For lossless coding, that is not sufficient.

Suppose the true pixel is:

```text
100
```

and your motion-compensated predictor gives:

```text
103
```

You must now encode `-3` exactly.

But the spatial predictor might give:

```text
100
```

because the local spatial neighborhood has already learned that structure.

So the comparison isn't:

> motion gets closer to the pixel than intra

It is:

> motion produces a **more compressible conditional residual distribution** than intra.

That is much stricter.

This is one reason lossless video designs often put substantial effort into specialized intra/sample prediction rather than merely increasing motion-search sophistication. VVC lossless uses transform bypass plus specialized lossless prediction machinery, and current AV2 development explicitly contains a dedicated lossless DPCM mode. ([MDPI][5])

### The experiment I would do before touching motion search again

Take your 1080p sequence and generate three residual populations:

```text
A = spatial predictor residual
B = best motion residual
C = spatial predictor applied to motion-compensated reference residual
```

But crucially, evaluate **the actual coded cost under your model**, not SSE.

For every candidate block, calculate something like:

[
C = -\sum_i \log_2 P(r_i \mid \text{context})
+ C_\text{MV}
+ C_\text{mode}
]

using the same model or a cheap proxy.

You will probably find one of two things:

### Case A: motion really loses after coding cost

Then your working theory is correct.

Your spatial model has simply become so good that motion compensation produces a residual with too much temporal estimation noise.

### Case B: motion wins on coded residual cost but the actual codec doesn't exploit it

Then you have a model/decision coupling problem.

That would be much more interesting.

---

## What I would try for temporal prediction

Not a larger search radius.

### 1. Better temporal *predictors*, not better vectors

Your current MC sounds like:

```text
block reference
+ half-pel MV
```

I'd test:

* multiple references,
* bidirectional temporal prediction,
* affine/planar motion for large smooth motion,
* per-pixel or sub-block refinement,
* temporal predictor blended with the spatial predictor.

The last one is particularly important.

Instead of:

```text
inter OR intra
```

use:

```text
prediction = α * spatial + (1-α) * temporal
```

where α is reversible and derived from causal data / a small coded mode.

This is exactly the kind of situation where a temporal predictor can be *partially right* without having to replace the spatial predictor entirely.

### 2. Residual-of-residual / RDPCM

This is the structural idea I would rank highest.

Lossless video standards and research repeatedly exploit a second prediction step on the residual itself. HEVC/VVC lossless work has used residual DPCM, and published lossless intra work specifically examines horizontal/vertical residual re-prediction and cross-residual prediction. ([DOI][6])

For HVE, this could be extremely cheap conceptually:

```text
pixel
  ↓
spatial / temporal prediction
  ↓
residual
  ↓
horizontal or vertical residual prediction
  ↓
second residual
  ↓
your existing coder
```

You don't need to replace your probabilistic model.

You're changing the **source presented to it**.

Given that your entropy engine is already strong, this is much more attractive than adding model depth.

### 3. Piecewise reversible residual mapping

This is the weird one I'd absolutely benchmark.

There is published lossless work showing that reversible piecewise mappings of residual blocks can reduce bitrate substantially in some content, particularly screen content. ([PubMed][7])

The important idea isn't their exact mapping.

It is:

> your exponent/sign representation assumes a certain residual shape; a reversible nonlinear remapping can make that shape easier to code.

That is directly relevant to your design.

Your current zero/exponent/mantissa decomposition is effectively betting that “small magnitude = easy distribution.”

A learned reversible mapping could make that *more true*.

---

# 3. Context sources I think are conspicuously absent

This is where I think you have the most actionable work.

## A. Predictor identity

As above, this is the biggest omission.

I would explicitly condition on:

* predictor winner,
* predictor confidence,
* predictor disagreement,
* possibly predictor class.

Your `NLMS confidence` is not equivalent.

## B. Residual topology

You have activity and energy, but I don't see explicit causal residual relationships such as:

```text
previous residual
above residual
upper-left residual
left-minus-upper-left
top-minus-upper-left
```

This is almost certainly worth testing.

FFV1's context design is a good sanity check here: it derives coding contexts from several quantized local differences rather than relying on one activity statistic. ([IETF Datatracker][2])

## C. Residual magnitude class

You are already modelling exponent, but the **previous residual's magnitude class** is likely a useful predictor of the current one.

For example:

```text
prev exponent
top exponent
prev zero/nonzero
top zero/nonzero
```

is potentially more informative for the *next zero/exponent* than a general activity measure.

This is a cheap context because it can use the state you already have.

## D. Predictor disagreement

This deserves its own category because I suspect it will beat several fancier contexts.

Something like:

```text
range = max(pred_i)-min(pred_i)
```

with logarithmic quantization.

High range means “model uncertainty.”

Low range means “all predictors agree.”

That is almost exactly the latent variable your mixer is trying to discover.

Give it to the mixer instead.

## E. Mode / region state

You need context saying:

```text
boundary / interior
new slice / not
new block / not
motion / spatial
match-hit / no-hit
```

The probability distributions immediately after mode changes are not necessarily stationary.

Your model apparently has `match state`, but I don't see explicit spatial-position state.

## F. Cross-channel local residual state

You have:

> luma-error map for chroma

Good.

But I'd test:

```text
co-located luma residual
neighboring luma residual
previous-channel residual exponent
previous-channel zero/nonzero
cross-channel predictor disagreement
```

JXL explicitly supports extra previous-channel properties in its MA tree. Current libjxl allows multiple previous-channel properties, with the encoder choosing how many to use. ([GitHub][1])

That is very close to a confirmation that cross-channel context is worth more than merely “chroma sees luma activity.”

---

# 4. Structural things worth stealing

Here is how I'd rank the external ideas.

## FFV1

**Worth stealing: context formulation, not the predictor.**

You already have MED.

The interesting FFV1 idea is the compact local-difference context construction and adaptive per-context residual estimation. ([IETF Datatracker][8])

A very worthwhile HVE experiment would be:

> keep your predictor and entropy coder untouched, replace one existing context source with a quantized FFV1-style difference vector.

That's a clean experiment.

## JPEG-LS / LOCO-I

**Worth stealing: run mode / edge classification.**

You already effectively have the MED family, so reproducing LOCO-I's basic predictor isn't the prize.

The interesting concept is that **long runs of zero or repeated residual behavior deserve a different coding regime**.

Your single zero flag is not the same thing as a run mode.

A run of:

```text
0 0 0 0 0 0 0 0 0 ...
```

is fundamentally different from isolated zeros.

I'd benchmark a run-mode side path before adding another model expert.

## CALIC

**Very worth studying.**

CALIC's important idea is not merely “more contexts.” It explicitly uses a large number of contexts to adapt a nonlinear predictor and learn the conditional expectation of residuals. ([McMaster Experts][9])

The HVE translation I'd try is:

> replace some scalar “activity” contexts with a compact learned class describing the local predictor-error regime.

You're already halfway there.

## MRP

**Potentially the most interesting still-image predictor experiment.**

MRP is explicitly designed around selecting/adapting multiple linear predictors to minimize coding rate rather than simply minimizing prediction error. Published MRP implementations use local blocks/classes and optimize coefficients for coding cost. ([TUS Research System][10])

This distinction matters enormously for HVE because you currently have a sophisticated predictor combiner but your encoder decisions sound largely residual-statistics based.

I would not implement full MRP first.

I'd implement a tiny version:

```text
4–8 local linear predictors
partition image into ~32–128 px tiles
choose coefficients by estimated coded residual cost
signal only the winning class/parameters
```

Then measure.

## VVC

For lossless specifically, I would pay attention to:

**RDPCM / residual DPCM, multiple-reference-line ideas, cross-component prediction, and lossless-specific prediction—not the large lossy block machinery.**

VVC explicitly supports lossless coding and specialized residual handling; available encoder implementations expose implicit RDPCM specifically for lossless coding and cross-component tools such as CCLM/JCCR. ([GitHub][11])

The lesson for HVE is:

> don't assume the best lossless codec is just a conventional predictive codec with quantization removed.

Lossless often benefits from its own prediction transform.

## AV2

The new AV2 codebase is especially relevant because the current 1.0 specification was released in May 2026, and the reference software currently contains an explicit lossless DPCM feature. ([AV2][12])

So this is one area where I'd inspect the **actual current AVM lossless path**, not older AV1 literature.

The conspicuous theme is again:

> lossless-specific DPCM rather than merely better generic intra/inter prediction.

---

# 5. What I'd be adversarial about in HVE

There are several things I suspect are dead ends.

## Dead end #1: more experts

You already have five zero experts + mixer + two APM stages, and three exponent experts + mixer + APM.

That is enormous model complexity for a tiny symbol.

Your recent result is telling:

> deleting two APM stages and two constant experts gave **−24% decode for +0.5% bytes**.

That is not a minor cleanup.

That's evidence that **the architecture is over-modelled**.

I would interpret the result as:

> HVE's model contains substantial prediction machinery whose information gain is below its memory/load cost.

So I would not add a sixth or seventh expert until a context analysis proves there is unexplained conditional structure.

## Dead end #2: further mixer sophistication

I would stop touching:

* logistic mixer form,
* weight clipping,
* extra SSE stages,
* more gradient tricks.

Your saturation measurement essentially killed the weight-clamping theory.

The 0.0003% saturation rate is particularly convincing.

## Dead end #3: instruction-count optimization

Your measurements strongly suggest you've found the microarchitectural wall.

The important observation is:

> 10% fewer instructions + 0% wall-time

and

> 6% fewer instructions + ~1% cycles

means you're not ALU-bound.

You're load-port / dependency-chain constrained.

So your optimization unit should be:

> **remove an entire model lookup / state bank / stage**

not:

> make an existing lookup 15% cheaper.

Your latest APM removal is exactly the right style of optimization.

## Dead end #4: ever more complicated prediction blending

You already have:

* MED,
* GAP,
* 13-tap NLMS,
* weighted blend,
* hash matching.

That's a very broad predictor ensemble.

I'd be more suspicious that **the entropy model doesn't condition sharply enough on which one won** than that you're missing predictor #6.

---

# What I would actually build next

If this were my codec, I'd stop adding machinery and run these experiments in this order:

### Experiment 1 — predictor-regime context

Add only:

```text
dominant predictor
predictor disagreement
predictor confidence
```

to zero/exponent contexts.

No other change.

**Expected outcome:** potentially meaningful still-image gain; very cheap decoder cost.

### Experiment 2 — residual-neighbor context

Add FFV1-like quantized causal residual differences.

**Expected outcome:** another potentially meaningful gain with modest state cost. ([IETF Datatracker][2])

### Experiment 3 — local RCT

64/128/256px groups, choose among a small RCT set.

**Expected outcome:** especially useful on Kodak/color photography. Current JXL explicitly does this at Modular group granularity. ([GitHub][1])

### Experiment 4 — residual DPCM

After your normal prediction:

```text
r[x]
→ r[x] - r[x-1]
```

or vertical variant, with a coded mode.

Test whether the second residual has lower **actual modelled cost**, not just lower variance.

This is probably my highest-priority structural experiment for video, and potentially useful for stills too. ([DOI][6])

### Experiment 5 — temporal/spatial blend

Instead of inter/intra binary:

```text
prediction = blend(spatial, temporal)
```

with a small reversible mode set.

This addresses the 1080p problem more directly than increasing MV search.

### Experiment 6 — context clustering ablation

Take the actual context stream and answer one question experimentally:

> How much conditional entropy remains if several contexts are optimally merged?

If the answer is tiny, forget clustering.

If the answer is several percent, that is probably your missing JXL-like mechanism.

Current libjxl's Modular encoder explicitly invests in context clustering and more exhaustive LZ77/HybridUint choices as effort increases. ([GitHub][3])

---

# My answers to your five questions, bluntly

**1. Stills:**
The gap most likely lives in **local/statistical conditioning**, not basic prediction. My top suspects are **predictor-regime contexts, context clustering, and local RCT selection**. Of those, I would test predictor identity/disagreement first because it fits your existing architecture and costs little. JXL's current Modular path also has local RCT selection, previous-channel MA properties, and entropy-stage optimizations that your summary does not indicate HVE has. ([GitHub][1])

**2. 1080p motion:**
Your working theory is probably substantially correct. **Rate-unaware motion search is not sufficient to explain the result**, because your inter-penalty sweep already points toward fewer inter blocks. The more promising change is **lossless-specific temporal prediction**: temporal/spatial blending, multiple references, and especially residual DPCM / residual-of-residual coding.

**3. Missing contexts:**
The biggest omissions I see are **predictor identity, predictor disagreement, causal residual topology, previous residual magnitude/exponent, mode/region state, and richer cross-channel residual state**. These are more compelling than inventing more generic activity statistics.

**4. Structural ideas:**
The strongest candidates are **residual DPCM/RDPCM, MRP-style rate-optimized local predictors, FFV1-style local-difference contexts, LOCO-style run mode, and reversible residual remapping**. VVC and current AV2 are especially useful because both reinforce the value of dedicated lossless prediction machinery rather than simply reusing the lossy architecture. ([IETF Datatracker][2])

**5. Dead ends:**
I would stop pursuing **more experts, more APM stages, more sophisticated mixer mathematics, and instruction-level micro-optimizations** unless an ablation demonstrates real information gain. Your recent −24% decode result is a strong hint that HVE currently has **too much model state relative to its predictive contribution**.

The single highest-value meta-experiment I would add to your benchmark harness is this:

> For every model feature/context, measure **conditional cross-entropy reduction per additional load**.

Not bits/sample.

Not instructions.

Not cache misses.

**Bits saved per extra model-state load**.

That metric matches the architecture you actually have and, based on your hardware observations, should tell you very quickly which parts of HVE deserve to survive.

Which direction would you prioritize next: **still-image gap**, **1080p temporal coding**, or **decode-speed pruning**?

[1]: https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h?utm_source=chatgpt.com "libjxl/lib/include/jxl/encode.h at main · libjxl/libjxl · GitHub"
[2]: https://datatracker.ietf.org/doc/rfc9043/?utm_source=chatgpt.com "RFC 9043 - FFV1 Video Coding Format Versions 0, 1, and 3"
[3]: https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md?utm_source=chatgpt.com "libjxl/doc/encode_effort.md at main · libjxl/libjxl · GitHub"
[4]: https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_ans_params.h?utm_source=chatgpt.com "libjxl/lib/jxl/enc_ans_params.h at main · libjxl/libjxl · GitHub"
[5]: https://www.mdpi.com/2076-3417/14/4/1653?utm_source=chatgpt.com "Sample-Based Gradient Edge and Angular Prediction for VVC Lossless Intra-Coding | MDPI"
[6]: https://doi.org/10.3390/app14041653?utm_source=chatgpt.com "Sample-Based Gradient Edge and Angular Prediction for VVC Lossless Intra-Coding"
[7]: https://pubmed.ncbi.nlm.nih.gov/28113430/?utm_source=chatgpt.com "Piecewise Mapping in HEVC Lossless Intra-Prediction Coding - PubMed"
[8]: https://datatracker.ietf.org/doc/draft-ietf-cellar-ffv1/16/?utm_source=chatgpt.com "draft-ietf-cellar-ffv1-16 - FFV1 Video Coding Format Version 0, 1, and 3"
[9]: https://experts.mcmaster.ca/scholarly-works/1909134?utm_source=chatgpt.com "Context-based, adaptive, lossless image coding"
[10]: https://www.rs.tus.ac.jp/matsuda-lab/matsuda/mrp/index.html?utm_source=chatgpt.com "Lossless Image Coding Using Minimum-Rate Predictors"
[11]: https://github.com/ultravideo/uvg266?utm_source=chatgpt.com "GitHub - ultravideo/uvg266: An open-source VVC encoder based on Kvazaar · GitHub"
[12]: https://av2.aomedia.org/?utm_source=chatgpt.com "AV2 Specification"