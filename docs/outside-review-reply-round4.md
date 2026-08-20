# Reply to both reviewers - round 4

No held-out coded sizes yet - that needs the build, which is next. But your
three pre-build diagnostics all paid, one of them reversed a verdict, and the
design is now settled by measurement rather than by sweeping. Details below.

## First: I caught a third variant of the same trap, before reporting it

Scoring MDL over all eighteen image-planes **pooled into one context space**
says the texture context is worth **-1.364%**. Scoring it **per plane and
summed** says **+0.069%**.

The second is correct. The codec resets its model per file and separates
contexts by plane kind, so each (image, plane) is a fresh adaptive model.
Pooling amortises description cost over eighteen times more samples than the
coder ever sees. Everything below uses per-plane accounting.

Three different measurements have now failed the same way in three rounds:
finer context partitions fitting their own histograms, per-block colour
transforms fitting their own histograms, and now pooled context statistics
amortising a cost the coder pays separately. The invariant seems to be: **if the
measurement gets to share statistics the codec does not share, it lies.**

## Your coarsening diagnostic worked, and it partly reversed my verdict

Merging the existing 64 texture states rather than inventing a new quantiser -
exactly as specified, so this tests whether the information is *present* rather
than whether a new quantiser happens to fit:

| merged texture states | MDL vs baseline |
|---:|---:|
| 2 | +0.001% |
| **4** | **-0.348%** |
| 8 | -0.306% |
| 16 | -0.201% |
| 32 | -0.047% |
| 64 (unmerged) | +0.069% |

There is a clear minimum at 4 states. **The texture signal is real; six bits of
it was simply too expensive to describe.** So I was wrong to call it dead, and
the diagnostic you specified is what found that - a fresh quantiser sweep would
have found it too but would not have been trustworthy.

It does not survive contact with disagreement, however:

| context | MDL |
|---|---:|
| disagreement only [x5] | **-0.606%** |
| texture, 4 merged states [x4] | -0.348% |
| both [x20] | -0.676% |

0.606 and 0.348 would sum to 0.954 if independent; together they give 0.676.
They are largely the same latent variable - local prediction uncertainty - and
texture contributes 0.07 points for four times the contexts. So disagreement
alone is what gets built, and your "latent description of predictor reliability,
not a missing texture dimension" reading looks right.

## The disagreement design, settled by measurement

**Quantiser.** Geometric spacing was the right instinct, and the first guess was
close, but shorter is better:

| edges | contexts | MDL |
|---|---:|---:|
| 1,2,4,8,16,32,64 (first guess) | x8 | -0.588% |
| **1,2,4,8** | **x5** | **-0.606%** |
| 1,2,8,32 | x5 | -0.486% |
| 1,4,16 | x4 | -0.288% |
| 13-bin fine lattice | x14 | -0.619% |

The upper bins were paying description cost for samples that are almost all in
one state. The 14-bin lattice edges it out by 0.013 points for nearly three
times the contexts, which I am not taking.

**Statistic.** Range (max-min) -0.588%; median absolute deviation -0.392%. To
your question about what that means: the information is specifically in **the
presence of one outlying predictor**, not in general ensemble spread.

**Predictor set** - and this is the part that decides implementation cost:

| set | MDL |
|---|---:|
| **MED, west, north, planar** | **-0.696%** |
| MED, GAP, west, north, planar | -0.606% |
| west, north, nwest, neast (no MED) | -0.540% |
| MED, west, north, nwest, neast, planar | -0.048% |

The best set is also the cheapest, and it needs no GAP - which matters more than
it looks, because GAP only exists when our LMS stage is on, so any set requiring
it could not be used in the `fast` preset at all. All four values are already in
hand where the context is formed; `planar` is one add and one subtract. Adding
nwest and neast destroys the feature entirely, presumably because they are
frequently the outlier themselves and swamp the range.

## Final design

> disagreement = max - min over {MED, west, north, west+north-nwest},
> quantised at 1,2,4,8, as a fifth dimension on the zero-flag context.
> **-0.696% by per-plane MDL.**

Building that next, with the same quantiser fixed across every test image so
there is no per-image fitting, and reporting the held-out coded size across all
18 Kodak images individually rather than only the total - so a gain concentrated
in two images is visible rather than averaged away.

Given that every honest number this exercise has produced came in below its
estimate, I expect less than -0.696% in actual bytes, and I will report it
either way.

## Two things I have not done, for the record

- **Validating the RCT selection proxy** (is argmin zeroth-order entropy the
  same as argmin actual coded bits?). Good catch, queued, and it is independent
  of the local-RCT question.
- **MDL segmentation with arbitrary cut points.** The power-of-two lattice plus
  adjacent merging already gave a clear optimum, so the finer search looks like
  it would only buy the 0.013 points the 14-bin lattice showed.
