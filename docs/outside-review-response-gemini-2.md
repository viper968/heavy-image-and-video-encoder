Your measurements are definitive and excellent. I stand corrected on the half-pel interpolation—the data clearly shows that your spatial intra predictor simply hits the noise floor at high resolutions. You cannot subtract uncorrelated grain, making exact temporal prediction theoretically bounded at 1080p.

Here is an analysis of your remaining questions based on your updated data.

## 1. Rate-Aware Search & The 1080p Reality

* **1080p Temporal:** Your conclusion is absolutely correct. At lossless high resolutions, independent noise fields dominate the temporal residual. The most efficient approach is to detect this rapidly and default to intra prediction.
* **Rate-Aware CIF:** Rate-aware search is still highly valuable for lower resolutions where motion genuinely pays. Because you code motion vectors with a unary exponent, large vectors are disproportionately expensive. A simplified penalty will cleanly prune the 12-bit vectors that only yield a 1-bit residual saving.

## 2. The JPEG XL Gap

Given you already have weighted prediction and ruled out the MA tree, the 6.86% gap likely lives in two specific architectural choices JXL leverages in Modular mode:

* **Cross-Component Linear Prediction (CFL):** You use a luma-error map as a *context* for chroma. JXL structurally *predicts* chroma values directly from the reconstructed luma high-frequencies using dynamic local linear models, removing massive cross-channel redundancy.
* **Hierarchical Transforms (Squeeze):** JXL uses a reversible multi-scale Haar-like transform. Resolving a lower-resolution image first allows it to use non-causal, structurally distant pixels to predict high-resolution details, bypassing the limits of a purely local 13-tap causal neighborhood.

## 3. Contexts & Structural Shifts

For your context sources (currently 5 experts on zero, 3 on exponent), you are missing a few deterministic states:

* **Spatial Coordinates (Modulo):** Tracking spatial position (e.g., `x % 2`, `y % 2`) is practically free and highly effective at capturing ubiquitous Bayer patterns, dithering, and sub-pixel rendering artifacts.
* **Run-Length States:** A dedicated context state tracking consecutive zero-residuals captures flat regions and synthetic blocks far more cleanly than relying on the logistic mixer to infer it from local energy.
* **Block-Level Signaling:** Structurally, consider calculating residuals for your predictors (MED, GAP, NLMS) and transmitting an explicit predictor choice per 16x16 block, rather than computing weighted blends per pixel.

## 4. The Adversarial Cut

Yielding the half-pel argument to your data, my revised adversarial target is your **online gradient-descent logistic mixer**.

* It is the primary anchor dragging your decode speed to 2.1 Mpixel/s.
* Pixel-by-pixel weight updates create an unbreakable serial dependency chain that permanently starves instruction-level parallelism.
* I would drop online updates entirely in favor of offline-optimized weights combined with block-level adaptive selection.

What percentage of your current decode time is spent specifically on updating the logistic mixer's weights versus calculating the 13-tap NLMS?