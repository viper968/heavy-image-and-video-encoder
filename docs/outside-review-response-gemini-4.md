## Coarser Texture vs. Context Folding

Folding texture into existing activity bins is fundamentally different from creating a high-cardinality Cartesian product ($A \times T$). The Minimum Description Length (MDL) penalty scales linearly with the total context count $K$. Multiplying 64 texture states across 16 activity states creates 1,024 contexts, incurring a parameter penalty of $O(K \log N)$.

* **Avoid Cross-Products:** Incorporating local texture into activity thresholds shifts existing bin boundaries rather than multiplying context count.
* **Coarse Texture (3-Bit):** Reducing to 3 bits (8 states) cuts the description tax by 87.5% while retaining the dominant structural information (e.g., horizontal vs. vertical edge direction).
* **Verdict:** Context folding or coarse discretization is mathematically justified under MDL, provided the average sample density per context satisfies $N / K \gg 1,000$.

---

## Principled Disagreement Quantization

Sweeping thresholds directly against dev byte counts risks fitting context boundaries to dataset-specific noise. Three principled approaches set boundaries without over-fitting:

* **Equal-Frequency Binning (Quantiles):** Position boundaries so that every context bin receives roughly an equal share of samples ($p(c_i) \approx 1/K$). This maximizes context entropy $H(C)$ and guarantees no context suffers from sample sparsity penalties.
* **Information-Theoretic Binning:** Group continuous disagreement values into $K$ discrete bins by maximizing mutual information $I(C; Z)$ with the zero flag $Z$. This can be computed deterministically via dynamic programming on error distributions.
* **Geometric Spacing:** Your manual boundaries $(1, 2, 4, 8, 16, 32, 64)$ are already close to optimal. Because prediction residuals follow heavy-tailed (Laplacian) distributions, logarithmic spacing naturally allocates resolution where sample density is highest.

---

## The Takeaway

Adopting an MDL frame is a decisive improvement in your evaluation framework—it cleanly explains why the learned MA context tree failed previously. Predictor disagreement survives because 8 contexts provide high conditional mutual information at a negligible description cost.

What are the held-out coded sizes on the test split for the zero-flag predictor disagreement context?