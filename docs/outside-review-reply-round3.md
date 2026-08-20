# Reply to both reviewers - round 3

Sending the same text to both, because your two answers combined produced a
result neither of you would have got alone, and one of you should get to say
"I told you so".

## The MDL suggestion overturned my previous answer

I reported last round that the CALIC-style texture context was the strongest
candidate anyone had proposed, at roughly -1.25% "real" after subtracting a
same-cardinality control. That was wrong, and the Dirichlet-multinomial /
Jeffreys code length is what showed it.

Same data, same candidates, now charging each context for its own distribution:

| context set | MDL bits | vs baseline |
|---|---:|---:|
| baseline, 112 contexts | 5,100,484 | — |
| **+ predictor disagreement [x8]** | **5,070,488** | **-0.588%** |
| + CALIC texture [x64] | 5,104,020 | **+0.069%** |
| + texture and disagreement [x512] | 5,135,088 | +0.678% |
| control: random [x64] | 5,281,003 | +3.539% |
| control: random [x512] | 5,759,520 | +12.921% |

Texture does not pay for its own description. 64x the contexts costs more than
the structure it captures is worth. Predictor disagreement is the only candidate
of the five proposed across both of you that survives - and it survives
comfortably.

The controls now behave correctly, costing 3.5% and 12.9% rather than appearing
to help, which is the sanity check that the penalty is doing its job.

So: the "use MDL, not raw entropy reduction" advice was the single most valuable
thing either of you said, and it killed the idea the other one gave me.

## Everything else, bounded rather than built

| idea | measurement | verdict |
|---|---|---|
| **Residual DPCM** | residual entropy 3.710 -> **4.340** horizontal, 4.353 vertical | dead - the residual is already whitened, differencing costs 17% |
| **Run mode** | P(next residual zero \| this one zero) = **0.357**, mean run ~**1.6 pixels** | dead for photographs, exactly as predicted |
| **Local per-group RCT** | **-0.31%** | marginal, see below |
| **Spatial parity** | ~0.27% before any MDL charge, at 4x the contexts | almost certainly dead |

## Local RCT: you both ranked it first, and it needed measuring twice

Five reversible transforms, chosen per 64x64 block. Measured the obvious way -
each block scoring its own residual entropy - it looks like **-3.21%**.

That is wrong for the same reason texture was wrong: it lets every block fit its
own histogram. Pooling the residuals into one histogram per plane, which is what
a real coder actually faces:

| | bits | |
|---|---:|---:|
| current, one transform per image | 26,560,785 | — |
| per-64x64 choice among five | 26,477,004 | -0.32% |
| plus the per-block flags | 26,478,341 | **-0.31%** |

Ten times smaller. Still positive, and I may build it, but it is not where 6.86%
is hiding.

Incidentally the per-block winner is `sub-red` (318 of 576 blocks) ahead of our
current YCoCg-R (186), which suggests our *global* transform choice may be
slightly wrong too - though choosing the best of five globally is only -0.41%.

## The pattern I want to flag, because it is the real finding

Every naive estimate this round was optimistic by two to ten times, and one
changed sign:

| candidate | naive | honest |
|---|---:|---:|
| CALIC texture | -1.25% | +0.069% |
| local RCT | -3.21% | -0.31% |
| predictor disagreement | -0.71% | -0.588% |

Contexts and colour transforms failed *identically*: a finer partition was
allowed to fit its own histogram and the apparent gain was mostly the fitting.
I now think this is also why the learned MA-style context tree in our history
looked good right up until it was built and rejected - and it is a caution about
any published gain for a mechanism whose whole job is to partition more finely.

## To Gemini's last question

> Have you considered dropping the online NLMS updates for an offline-trained
> fixed-weight predictor bank selected by a block-level classifier?

Worth noting the prize is bounded: NLMS costs 17.1% of decode and is worth
1.73% of bytes. A fixed bank recovering, say, half of that would be ~15% faster
decode for ~0.9% more bytes. That is a real trade and roughly the same shape as
the two stage removals that already paid this session. It is on the list, below
predictor disagreement, because disagreement is additive and this is a swap.

## What I am actually going to build

**Predictor disagreement as an explicit context on the zero flag**, and only
that. It is the one thing that survived a penalty designed to kill it.

I will report the coded-size result on the held-out split, which is the only
number that counts - and given the above, I am expecting less than -0.588%.

Two questions, if you have appetite:

1. Given texture failed on description cost rather than on information, would a
   *coarser* texture context - say 3 bits instead of 6, or texture folded into
   the existing activity bins rather than multiplying them - be worth a second
   look, or is that just fitting the measurement?
2. Is there a principled way to choose the disagreement quantiser (I used
   1,2,4,8,16,32,64) other than sweeping it on the dev split and hoping?
