Thanks - and the concession on half-pel is appreciated. I measured everything
you raised. Your best idea is currently the best idea anyone has given me; your
adversarial target is wrong by an order of magnitude. Both below.

## Your question, answered directly

> What percentage of decode time is the logistic mixer's weight updates versus
> the 13-tap NLMS?

kodim13, `max` preset, decode pinned to one core:

| stage removed | decode | bytes |
|---|---:|---:|
| full model | 0.2420s | 564,004 |
| the 13-tap NLMS filter and its update | 0.2006s (**-17.1%**) | 573,772 (+1.73%) |
| the mixer dot product and weight update | 0.2382s (**-1.5%**) | 564,936 (+0.17%) |

The mixer is **1.5%** of decode. The NLMS filter you contrasted it with is
**17.1%** - eleven times more. So the adversarial cut does not hold: dropping
online gradient descent for offline weights buys about a percent and a half and
gives up the mechanism the entire model is built on.

The interesting row is NLMS: 17.1% of decode for 1.73% of bytes is the worst
stage trade left in `max`. Our `fast` preset already drops it. Your instinct
that something in the per-pixel adaptive machinery is overpriced was right - you
just named the wrong component.

## Your context suggestions, measured with controls

Conditional entropy of the zero flag, six dev images, three RCT planes, each
against a **control of identical cardinality** carrying no information - because
empirical conditional entropy always falls when contexts are split:

| candidate | contexts | measured | control | real |
|---|---:|---:|---:|---:|
| **CALIC-style texture** (6 causal neighbours vs local mean) | x64 | -2.652% | -1.399% | **~1.25%** |
| predictor disagreement (from the other reviewer) | x8 | -0.882% | -0.174% | ~0.71% |
| **spatial parity (x%2, y%2)** | x4 | -0.351% | -0.078% | ~0.27% |
| west residual magnitude bin | x8 | -0.364% | -0.174% | ~0.19% |

Your CALIC texture context is the strongest single candidate anyone has
proposed. Your spatial-parity suggestion is real but small - and notably these
are demosaiced Kodak scans with no Bayer pattern, so whatever it is finding is
subtler than the mechanism you proposed.

**One important caveat, which cuts against all of these.** Stacking texture with
disagreement is x512 - about 57,000 contexts against a few hundred thousand
samples per plane - and there the random control scored **-8.398% against the
real combination's -4.091%**. The control beat the signal. Past a few hundred
contexts this method is measuring sample sparsity, not information, and it
models no adaptation cost whatsoever. So I read the table as an optimistic
ranking, not as predicted gains. It is also, I suspect, exactly why the learned
MA-style context tree we built earlier looked good right up until it was built.

## Your two structural ideas, bounded

**Hierarchical / Squeeze.** Right in principle - our predictor is strictly
causal and cannot see below or right. To bound it I replaced causal MED with a
*non-causal* predictor no real scheme could achieve: the plain average of all
eight neighbours including S, SE, SW, E. Residual entropy moves **3.714 ->
3.625 bits/sample, about 2.4%**. So the ceiling on the entire idea is 2.4%
against a 6.86% gap, before a real multi-scale scheme pays anything for coding
the pyramid. Worth something, not worth the gap.

**Cross-component prediction (CfL).** You are right that we use luma error as
*context* rather than *predicting* chroma from luma. But we apply a reversible
colour transform first, and I measured what is left: correlation between the
luma residual and each chroma residual **after** the RCT averages **0.0965**,
peak 0.1775. About one percent of variance. The RCT has already taken it. I
think this one is close to dead for our pipeline, though it would matter far
more for a codec without a good decorrelating transform.

## Where I think you were wrong, for the record

- **Half-pel**: conceded, and the data was emphatic - it beats integer-pel on
  every clip we have, by 0.2% to 4.3%.
- **"Rate-unaware search is fatal here"**: our inter-penalty sweep runs 0 to 768
  and its optimum sits *higher* than the current value, i.e. wanting fewer inter
  blocks. Rate-awareness cannot be hiding a reservoir of gains at 1080p. I agree
  with your revised position that it still matters at CIF scale.
- **"The logistic mixer is a dead end for your hardware profile"**: 1.5%.

## What I would ask next

1. The texture context is the strongest candidate measured. CALIC uses the
   local mean as the comparison threshold - would you expect the *predicted*
   value to work better as the threshold, given we have a much stronger
   predictor than CALIC does?
2. Given the control-beats-signal result above, how would you evaluate a
   candidate context *including* its adaptation cost, short of implementing it?
   Is there a standard penalty term you would trust here?
3. Ranking your remaining untested suggestions for photographic stills - run
   mode, local per-group RCT, and residual DPCM - which single one would you
   spend a day on first?
