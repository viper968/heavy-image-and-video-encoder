Your new measurements sharpen the picture considerably. The predictor-disagreement result is especially convincing because you did the right control: the raw entropy drop is not being mistaken for information simply because the partition got finer.

## 1. Context clustering: I would use MDL, with a divergence metric only as the search heuristic

I would **not** use greedy pairwise KL as the actual merge criterion.

KL is useful for finding plausible merges, but it has two problems here:

1. It is asymmetric.
2. It measures distributional similarity without accounting for the cost of maintaining separate models.

For a clustering diagnostic, I'd use a **two-stage procedure**:

### Stage A: cheap candidate generation

For two contexts (i,j), use Jensen–Shannon divergence:

[
JS(P_i,P_j)
===========

\frac12D_{KL}(P_i|M)
+
\frac12D_{KL}(P_j|M),
\quad
M=\frac{P_i+P_j}{2}.
]

It is symmetric, bounded, and doesn't blow up when one histogram has a zero in a bin.

For the zero flag this is trivial. For exponent use the full 8-bin distribution.

But don't actually merge simply because JS is small.

### Stage B: accept merges by code-length change

The quantity you really want is:

[
\Delta L =
L_{\text{merged}}
+
L_{\text{model description}}
----------------------------

L_{\text{separate}}.
]

Merge only when (\Delta L<0).

For a **pure hindsight diagnostic**, I'd go one step simpler and use a Bayesian marginal likelihood for each context histogram. A Dirichlet-multinomial code is convenient:

[
L(H)
====

-\log_2
\frac{\Gamma(K\alpha)}
{\Gamma(N+K\alpha)}
\prod_k
\frac{\Gamma(n_k+\alpha)}
{\Gamma(\alpha)}.
]

Then compare:

[
\Delta L =
L(H_i+H_j)-L(H_i)-L(H_j).
]

That automatically charges for having separate distributions instead of pretending that a finer partition is free.

For the binary zero flag, this is particularly clean. I'd probably use Jeffreys' prior, (\alpha=\tfrac12), as a neutral default.

### One important qualification for HVE

There is a trap here:

**histogram clustering is not necessarily equivalent to clustering your adaptive models.**

Suppose contexts A and B have almost identical final histograms. Sharing a model can still be worse because their symbols occur in different temporal patterns and therefore drive the adaptive probability state differently.

So I would make your diagnostic hierarchical:

1. **Histogram MDL:** tells you whether merging is plausible at all.
2. **Prequential/adaptive code length:** tells you whether merging actually helps your HVE learner.

The second test should literally replay the symbol stream through the current adaptation rule, once with separate state and once with merged state.

That distinction is important enough that I would not implement a clustering structure until you have both numbers.

### What I would log

For every candidate merge, record:

```text
histogram JS
histogram MDL gain
adaptive-code-length gain
number of symbols represented
```

Then plot adaptive gain against JS.

You may discover that there is a very sharp region where contexts are statistically similar enough that sharing state is beneficial, which gives you a much better clustering rule than an arbitrary JS threshold.

And there is a useful sanity check:

> If hindsight histogram MDL says clustering should save 2%, but adaptive replay saves only 0.1%, the problem is adaptation dynamics, not missing context partitioning.

That would prevent a lot of wasted engineering.

---

# 2. Should predictor disagreement also predict the exponent?

**Yes, I would measure it. I would not assume it pays.**

Your zero-flag result has a fairly intuitive interpretation:

> disagreement is telling the model how likely the predictors are to be simultaneously correct enough to produce an exact zero residual.

That is especially powerful because zero/nonzero is a sharp regime boundary.

The exponent is different.

Once you're conditioned on

[
Z=1
]

(nonzero), disagreement may predict the **scale** of the error. But the relationship is weaker and less universal.

For example:

```text
predictors: 127,127,128,128,127
actual:     131
```

and

```text
predictors: 90,128,161,140,103
actual:     131
```

both have a nonzero residual, but the second case tells you much more about uncertainty.

So I would expect predictor disagreement to have **some exponent information**, but I would expect it to be less clean than for zero/nonzero.

More importantly, don't test:

[
H(E\mid context)
]

alone, because your exponent is only emitted for nonzero samples.

Test:

[
H(E\mid Z=1,context).
]

And then measure the **joint contribution** to your actual bitstream:

[
H(Z\mid C)
+
P(Z=1),H(E\mid Z=1,C).
]

That decomposition will tell you where the payoff is.

### There is a particularly useful diagnostic here

Split disagreement into bins and calculate:

```text
P(nonzero | disagreement)
E[exponent | nonzero, disagreement]
Var(exponent | nonzero, disagreement)
```

I would expect something like:

```text
low disagreement
    → lots of zeroes
    → when nonzero, mostly exponent 1

high disagreement
    → fewer zeroes
    → broader exponent distribution
```

If you see that monotonic structure, the feature is telling you something genuinely physical/statistical about the predictor ensemble rather than merely correlating accidentally with your existing activity/error bins.

Your controlled result already establishes the first half.

---

# 3. For photographic stills: local RCT first

Of the three, **local RCT is the one I would spend the day on first**.

Not because run mode or RDPCM are bad ideas, but because of the evidence you now have and the nature of Kodak-style photographic material.

### Why local RCT is particularly attractive

Your current transform decision is:

> choose one reversible color transform for the whole image using zeroth-order residual entropy.

The opportunity is to remove the implicit assumption that the best color correlation is stationary over the image.

JPEG XL's current Modular encoder explicitly supports trying RCTs locally per Modular group rather than applying one transform globally by default. ([GitHub][1])

That isn't proof it will give HVE a useful gain, but it is unusually strong evidence that **the optimization axis is real in a production lossless codec**.

And your workload is exactly where I would expect it to have a chance:

* photographic RGB,
* illumination changes,
* different material spectra,
* foliage/sky/skin/artificial objects,
* local saturation differences.

I would start with something extremely conservative:

```text
global RCT
vs
128×128 local RCT
```

with maybe 8–16 candidate transforms, and keep your existing predictor/model completely unchanged.

Do not simultaneously change group size, predictor, context model, etc.

The first experiment should answer one question:

> **Does allowing the color transform to vary spatially reduce the actual downstream coded cost?**

Not residual variance. Not entropy before the model. Actual bytes.

### Why I would put run mode second

Run mode is very real; FFV1 enters run mode when the prediction context is zero and encodes the length until the first nonzero difference, with a separate run-length coding mechanism. ([IETF Datatracker][2])

But photographic material is precisely where I would expect its payoff to be smaller.

You need substantial stretches of exact prediction:

[
e=0,e=0,e=0,\ldots
]

A natural image can have many zero residuals with a good predictor, but exact runs are easily destroyed by:

* texture,
* sensor noise,
* demosaicing,
* subtle gradients,
* your sophisticated predictor changing continuously.

Your zero-flag experiment is encouraging in one respect: apparently you do have substantial predictable zero structure. But it doesn't tell us that those zeroes are **run-length clustered** spatially.

I'd measure that before implementing anything:

[
P(e_{x+1}=0\mid e_x=0)
]

then

[
P(e_{x+2}=0\mid e_x=e_{x+1}=0)
]

and generate the empirical run-length distribution.

If the mean run after entering zero territory is only, say, 1–2 pixels, JPEG-LS/FFV1-style run mode probably isn't going to transform the file.

If you see long tails of runs, it becomes interesting very quickly.

### RDPCM is third for photographic stills

I still like RDPCM structurally, but your target matters.

For photographic stills, after a strong spatial predictor such as MED/GAP/NLMS, I would expect a second finite difference

[
r_x-r_{x-1}
]

to be helpful mainly where the **residual itself is smooth**, which isn't necessarily where your prediction error is concentrated.

So I'd absolutely do the cheap bound you proposed:

```text
H(r)
H(r - r_left)
H(r - r_top)
```

but preferably also:

```text
actual HVE code length(r)
actual HVE code length(r-r_left)
actual HVE code length(r-r_top)
```

because a lower zero-order entropy doesn't guarantee a lower cost through your particular context mixer.

If that experiment shows a convincing gain on Kodak, then move it up immediately.

---

# My concrete one-day order

For **photographic stills**, I'd do:

**1. Local RCT** — 128×128 groups, one isolated ablation.
**2. Run statistics** — no implementation until you know the run distribution.
**3. RDPCM entropy/code-cost bound** — horizontal and vertical.

And independently, I'd continue the **predictor-disagreement context**, because your controlled (-0.7%) estimate is the strongest new evidence you've produced.

One subtle point I would change from my previous advice: I would **not** assume that the disagreement feature needs to be added identically to both zero and exponent models. Your new measurements give you a way to let the data decide. Measure

[
I(D;Z\mid C)
]

and

[
I(D;E\mid Z=1,C)
]

separately, with the same cardinality controls. You may discover that disagreement is an excellent *zero-model* feature but nearly redundant for exponent prediction. That would be a perfectly good result and would keep the decoder cheaper.

The fact that you are now measuring **information per added state/load rather than raw entropy reduction** is also the right methodology for HVE. Your earlier “two constant experts” correction makes that especially important: the relevant question isn't whether a feature exists, but whether its *conditional information survives controls and justifies its model-state cost*.

[1]: https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h?utm_source=chatgpt.com "libjxl/lib/include/jxl/encode.h at main · libjxl/libjxl · GitHub"
[2]: https://datatracker.ietf.org/doc/html/rfc9043?utm_source=chatgpt.com "RFC 9043 - FFV1 Video Coding Format Versions 0, 1, and 3"