Here is a direct, empirical evaluation of your codec’s pipeline based on your constraints and measurements.

### 1. The Stills Gap

The third major element you are missing compared to JPEG XL is **hierarchical/multi-scale prediction** (JXL’s "Squeeze" transform). Raster-scan predictors (LOCO-I, MED, GAP) are fundamentally limited by their immediate local causal neighborhood. JXL resolves lower resolutions first, allowing it to use non-causal, structurally distant pixels (below and to the right in the original resolution) as predictors for the higher-resolution details. This captures long-range redundancies that a 13-tap local NLMS simply cannot see.

### 2. The 1080p Motion Puzzle

Your working theory is directionally correct, but it misses a critical mechanical flaw: **half-pel motion compensation**.

* Sub-pixel interpolation acts as a low-pass filter. In lossless video, film grain and high-frequency sensor noise are preserved. Subtracting a filtered reference block from a sharp original misaligns these high frequencies, yielding a high-entropy residual.
* Your lack of rate-aware motion search is fatal here. Minimizing spatial residuals without weighting the MV cost via a rate-distortion equation like $J = D + \lambda R$ means you are routinely selecting vectors where the unary encoding penalty heavily outweighs the residual savings.

### 3. Context & Structural Shifts

For context modeling and structural alternatives:

* **Texture Contexts:** You lack a quantized local shape/texture context. Look at CALIC’s approach: it forms an 8-bit texture context by comparing neighboring pixels to the local mean, collapsing spatial shapes into discrete states.
* **Temporal Contexts:** For video, incorporate the co-located residual energy from the previous frame.
* **Minimum-Rate Predictors (MRP):** Instead of relying entirely on your load-heavy online NLMS, consider optimizing linear predictor coefficients offline per-block and transmitting them directly in the bitstream.

### 4. The Adversarial Cut

**Drop half-pel motion search immediately.** Stick entirely to integer-pel for lossless video; it preserves exact pixel integrity and eliminates interpolation logic. Furthermore, the **LPAQ-style logistic mixer is a dead end for your hardware profile**. If you are already starved for load-port capacity (IPC 4.40 with spare ALU), deep online gradient descent per pixel will permanently cap your decode speed.

Have you measured the baseline entropy cost of your half-pel interpolated residuals against a strict zero-vector (skip) integer reference on that 1080p clip?