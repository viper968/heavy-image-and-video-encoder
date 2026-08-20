This is a very useful update. The important result is not merely that disagreement survived; it is that **your methodology is now successfully rejecting attractive false positives**. The texture/RCT discrepancy is exactly why I would resist adding another context feature before proving that it survives an explicit model-description cost. MDL is designed precisely to trade fit against model complexity rather than reward finer partitions automatically. ([Taylor & Francis Online][1])

### 1. Coarsening the texture context: yes as a diagnostic, but not as a broad second search

I would **not** do a sweep over dozens of 3-bit/4-bit/5-bit texture definitions. Your 64-way texture result already says something stronger:

> whatever conditional information texture contains, it is too weak or too expensive in its current representation to justify six bits of context identity.

A coarser representation could still work because the interesting information might lie in a very low-dimensional variable that your 64-way partition is needlessly fragmenting.

But there is a clean way to determine that without falling back into the fitting trap.

Take the existing 64 texture classes and ask whether they can be **merged into a small number of groups by distributional similarity**, then score the resulting partition with the same MDL calculation. This is exactly the kind of context quantization problem where hierarchical clustering plus an MDL stopping criterion is appropriate. ([ResearchGate][2])

I would try:

```text
64 texture states
      ↓
hierarchical merges
      ↓
32, 16, 8, 4, 2 states
      ↓
MDL score at every level
```

No new feature engineering yet.

If the MDL curve has no minimum below the 112-context baseline, **kill texture completely**.

If, say, 8 or 4 merged classes beat baseline, then you've learned something much more interesting: the texture signal exists, but its useful representation is coarse.

There is a subtle advantage to doing this to the *existing 64 states* rather than inventing a new 3-bit texture quantizer: you are testing whether the information is actually present in the feature, rather than whether some newly tuned quantizer happens to work on the dev set.

So my answer is:

**One coarse/clustering diagnostic is worth doing. A fresh quantizer sweep is not.**

And I would use the same held-out discipline afterward. MDL is particularly appropriate here because the object being selected is simultaneously the partition and its statistical description. ([Springer Nature Link][3])

---

## 2. Choosing the predictor-disagreement quantizer

Here I think there is a better principled approach than “sweep until something wins.”

Your raw variable is

[
D = \max_i p_i-\min_i p_i.
]

The important point is that **the quantizer should be chosen according to the information D carries about the symbol, not according to the numerical scale of D itself**.

Your current

```text
1,2,4,8,16,32,64
```

is actually a reasonable first guess because error/disagreement scales are usually multiplicative rather than additive. But there is a more principled way.

### Use an MDL-optimal 1-D partition of D

Sort the samples by (D).

Now consider splitting that ordered sequence at candidate boundaries. For every proposed partition (Q(D)), calculate:

[
L(Q)+L(Z\mid C,Q(D)).
]

Where:

* (L(Q)) is the description length of the quantizer/boundaries,
* (L(Z\mid C,Q(D))) is the coded zero-flag sequence using the resulting context,
* (C) is your existing baseline context.

Then select the partition with minimum total description length.

This is essentially **MDL histogram/bin selection**, except your objective is conditional-symbol coding rather than density estimation. MDL formulations specifically support choosing both the number and locations of histogram bins rather than fixing them in advance. ([Proceedings of Machine Learning Research][4])

### There is an important implementation detail

I would not allow arbitrary cut points at every observed (D).

That makes the model-description search itself huge and makes the diagnostic unnecessarily optimistic.

Start with candidate thresholds at powers of two:

```text
0
1
2
4
8
16
32
64
128
...
```

and permit **merging adjacent bins only**.

Then your optimization becomes:

```text
fine initial bins
        ↓
merge neighboring bins
        ↓
choose minimum-MDL partition
```

That's especially well matched to disagreement because the variable is ordered.

You could eventually permit thresholds between every integer value, but I would first see whether the power-of-two lattice already gives you essentially all the measurable benefit.

### Even better: use the conditional distribution to establish the quantizer

For each disagreement value/bin, calculate:

[
P(Z=1\mid D,C)
]

and plot it.

If this looks approximately monotonic and smooth, then you have a good reason to use **log-domain scalar quantization**.

For example, something like:

```text
D=0
D=1
D=2–3
D=4–7
D=8–15
D=16–31
...
```

is then not arbitrary; it corresponds to roughly equal resolution in log disagreement.

If instead you see sharp changes around particular values, those empirical transition points are what the MDL segmentation should discover.

---

## One experiment I would add before committing the disagreement context

You have measured:

> disagreement lowers zero-flag entropy by ~0.7% beyond the control floor.

I'd now distinguish **magnitude of disagreement** from **which predictors disagree**.

For example, these can have the same range:

```text
MED GAP W N planar
100 101 100 101 100
```

versus

```text
MED GAP W N planar
100 101 102 103 101
```

and:

```text
100 130 100 130 100
```

with completely different structural interpretations.

Your current feature collapses all three to roughly the same scalar range.

I would therefore do one controlled diagnostic:

[
D_{\mathrm{range}}
]

versus

[
D_{\mathrm{variance}}
]

or perhaps

[
D_{\mathrm{MAD}} = \operatorname{median}|p_i-\operatorname{median}(p)|.
]

Not because I expect one necessarily to win, but because if the information survives **several robust measures of predictor disagreement**, then you've identified a genuine latent variable: *predictor uncertainty*.

If only max-min survives, that's also useful: it says the important thing is specifically the **existence of an outlying predictor**, rather than general ensemble disagreement.

I would stop there rather than create a high-dimensional disagreement feature.

---

## The RCT result also changes my recommendation slightly

Your corrected result:

> **−0.31% after paying for flags**

is exactly the sort of result I'd expect from a legitimate but second-order opportunity.

The striking part is:

> `sub-red` wins 318/576 blocks while global YCoCg-R is currently preferred.

That suggests there may be a simpler bug/opportunity hiding inside the transform-selection criterion.

You said:

> choosing the best of five globally is only −0.41%.

That makes me want to verify **what statistic your global transform selector minimizes** versus what the actual downstream HVE coder costs.

In other words, make sure:

[
\arg\min_T H_0(T)
]

is actually correlated with

[
\arg\min_T L_{\mathrm{HVE}}(T).
]

If the global winner according to zeroth-order entropy is not the global winner according to actual coded bits, you may be leaving some easy improvement on the table even before implementing local RCT.

Your local result is therefore not just “0.31% available.” It also suggests your transform-selection proxy may deserve validation.

---

## And I agree with your interpretation of the DPCM/run results

Those are now nicely bounded.

### RDPCM

3.710 → 4.340/4.353 bits/sample is devastating enough that I would drop it for this photographic residual. The premise was “perhaps the residual remains spatially smooth”; your measurement says the opposite.

### Run mode

(P(0_{n+1}\mid0_n)=0.357), mean run ≈1.6, likewise enough to stop. There's just not enough run structure to pay for a dedicated run mechanism.

That is a good example of why your current methodology is working: both mechanisms sound plausible from codec literature, but the source statistics kill them before implementation.

---

# What I would tell Claude/Gemini back

The emerging story is actually quite clean:

**HVE's missing information is looking increasingly like a latent description of predictor reliability, not a missing generic “texture” dimension.**

You have now seen:

* texture: apparent information, disappears after model cost;
* spatial parity: apparent information, disappears after model cost;
* run structure: absent in the source;
* residual DPCM: residual already whitened;
* local RCT: real but small;
* predictor disagreement: survives explicit model cost.

That puts disagreement in a qualitatively different category.

So I would build that one exactly as proposed, and when you get the actual **byte delta**, the interesting comparison is not against the (-0.588%) MDL number. It is:

[
\frac{\text{actual bits saved}}
{\text{additional model-state / lookup cost}}
]

and, crucially, whether the gain remains across all 18 held-out Kodak images rather than being concentrated in a couple of images.

I would also preserve the **same disagreement quantizer across all test images**, rather than letting each image adapt its bins. That prevents another form of hidden per-image fitting.

And for the quantizer experiment, my preferred sequence is:

**MDL segmentation of the existing ordered disagreement values → held-out test → only then consider alternative disagreement statistics.**

That gives you a principled answer without turning “choose the bins” into another dev-set fishing expedition.

[1]: https://www.tandfonline.com/doi/abs/10.1198/016214501753168398?utm_source=chatgpt.com "Model Selection and the Principle of Minimum Description Length: Journal of the American Statistical Association: Vol 96, No 454"
[2]: https://www.researchgate.net/publication/301672652_Context_Quantization_based_on_Minimum_Description_Length_and_Hierarchical_Clustering?utm_source=chatgpt.com "(PDF) Context Quantization based on Minimum Description Length and Hierarchical Clustering"
[3]: https://link.springer.com/article/10.1007/s10618-022-00846-z?utm_source=chatgpt.com "The minimum description length principle for pattern mining: a survey | Data Mining and Knowledge Discovery | Springer Nature Link"
[4]: https://proceedings.mlr.press/v2/kontkanen07a.html?utm_source=chatgpt.com "MDL Histogram Density Estimation"
