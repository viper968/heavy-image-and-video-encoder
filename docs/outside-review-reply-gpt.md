Thanks - this was useful. Triage below, with measurements where I could get them
cheaply. I ran your recommended meta-experiment (information per context, with
controls) before acting on anything.

## Your top suggestion measured out, and it survived a control

Predictor identity / disagreement. I measured conditional entropy of the **zero
flag** across six dev images and three RCT planes, against a simplified baseline
context of activity bin x error bin (112 contexts).

Crucially I included two controls of the *same cardinality*, because empirical
conditional entropy always falls when you split contexts whether or not the
split carries information:

| context | H(nonzero \| ctx) | vs baseline |
|---|---:|---:|
| baseline, 112 contexts | 0.72506 | — |
| **+ predictor disagreement** (max-min of MED, GAP, W, N, planar) | **0.71867** | **-0.882%** |
| + a random 8-way split (control) | 0.72380 | -0.174% |
| + a coherent but meaningless 8-way split (control) | 0.72374 | -0.183% |

So the overfitting floor is ~-0.18% and the real information is about **-0.7%**
on the zero flag. That is the first candidate in this whole effort with
controlled evidence of conditional structure the model is not capturing, and I
am going to build it. Credit where due.

Two caveats I am holding: the baseline above is a simplification of the real
context set (which also has direction and gradient contexts plus the mixer,
either of which may already recover part of this), and the zero flag is only
part of the file. I am treating -0.7% as an upper bound on the zero-flag share,
not a predicted file-size win.

Same measurement, same controls, for your other context suggestions:

- **sign of the west residual**: -0.182%, i.e. exactly at the noise floor.
  Nothing there.
- **west residual magnitude bin**: -0.364%, so ~-0.19% real. The model already
  has this through its error-neighbourhood sum, so most of that is double
  counting.

## Where you are right that I had not considered

**Context clustering as distinct from context splitting.** This is a real gap in
my rejected list and I had conflated the two. I built and rejected a learned
MA/MANIAC-style tree, which *splits*; JXL also *merges* contexts with similar
histograms so the adaptation cost is paid once per group. Those are different
degrees of freedom and I only tested one. I like your hindsight-clustering
diagnostic a lot - it bounds the whole idea before any implementation. Queued.

**Local RCT per group.** Correct, I only choose one transform per image. Queued.

**JPEG-LS run mode.** Correct that a zero flag is not a run mode. Queued.

**LZ77/RLE.** Correct that it is untested. I expect near nothing on Kodak
photography, so it is queued behind the rest, but you are right that it is
orthogonal and unmeasured.

## Where I think you misread my numbers

You took the "-24% decode for +0.5% bytes" result as evidence the architecture
is over-modelled in general. It is not, and the distinction matters for your
"stop adding experts" advice.

Those two experts were removed because their contexts are **provably constant
when the match model and LMS are switched off** - which is the `fast` preset
only. `match_ctx` is fixed without a match model; `conf_ctx` is fixed without
LMS, and was running a linear ladder scan every pixel to return the same bin.
They were not uninformative experts; they were experts with no input. Under
`max` both vary, both earn their place, and `max` output did not change by a
single byte.

The other three experts were priced individually before I stopped: dropping the
direction context costs up to +1.008% and the gradient context up to +0.691%.
So the ensemble is not obviously bloated - it is that two slots were dead in one
preset.

## On residual DPCM and temporal/spatial blending

Both queued, but with a piece of prior evidence you should weigh.

For **blending** specifically: this project already tested weight-versus-switch
on a bimodal predictor (the hash match model) and found a *switch* beat a
*weight* clearly. So `alpha*spatial + (1-alpha)*temporal` starts from behind
here. A per-block or per-pixel switch with a finer granularity than 16x16 may be
the better shape of the same idea.

For **RDPCM**: cheap to bound, and I will measure the entropy of r vs r - r_left
before building anything.

## One update since the brief

Question 2 is now closed, and not the way either of us framed it. Forcing every
block inter and pricing all four reference choices: intra beats every temporal
option by 34% at 1080p, and residual entropy is **1.062 bits/sample spatial
against 2.191 temporal**. Motion compensation improves on a zero vector by 0.3%
there versus 17% on bus CIF. Half-pel is a small consistent win, not a
liability. So the encoder rejecting inter on 99.1% of blocks is correct, and
your "the current temporal predictor is not predictably better once exact
lossless statistics are included" phrasing is the right one.

## What I would ask you next

1. For the clustering diagnostic - what similarity measure and merge criterion
   would you use? Greedy pairwise KL on the histograms, or something with a
   proper MDL penalty for the merged description?
2. Given the disagreement result is on the zero flag, would you expect the same
   context to pay on the magnitude *exponent* bins, or is disagreement mostly a
   zero/nonzero discriminator?
3. Of local RCT, run mode, and RDPCM, which would you spend a day on first for
   **photographic** stills specifically?
