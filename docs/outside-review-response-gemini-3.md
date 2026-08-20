I appreciate the rigorous teardown. You brought the receipts, and I yield entirely on the mixer—1.5% is negligible, and missing the NLMS cost was a massive blind spot on my end. Your bounding methodology on the Squeeze transform and CfL is equally definitive. Let's tackle your three questions with that same empirical lens.

## Texture Threshold: Mean vs. Predicted

* Using the **predicted value** is theoretically superior when your predictor drastically outclasses a naive local mean.
* It aligns the texture state directly with the error surface you are about to code, turning the context into a map of expected residual shapes.
* The danger is noise. Because your predictor is highly accurate, neighborhood pixels will cluster tightly around it. You must enforce a quantization step or "dead zone" around the predicted threshold to prevent the model from classifying raw sensor noise as distinct structural textures.

## Approximating Adaptation Cost

* Your control-beats-signal observation is a perfect demonstration of context dilution. To evaluate this without building it, you need a penalty term for model complexity.
* The standard approach is the Bayesian Information Criterion (BIC), which penalizes parameter bloat. You can approximate this penalty using the formula $\text{BIC} = k \ln(n) - 2 \ln(\hat{L})$ where $k$ is the number of context states and $n$ is the sample count.
* For adaptive arithmetic coders and LPAQ-style mixers, a practical rule of thumb is that each newly initialized context state costs roughly $0.5 \log_2(N)$ bits of overhead while the probability model converges.
* If your raw empirical entropy gain minus this convergence tax is negative, the context will fail in practice.

## The One-Day Bet: Local RCT

* I would spend that day entirely on **Local per-group RCT**.
* Run modes are essentially dead weight on photographic film grain; runs of exactly zero residual are too rare in Kodak scans to justify the branching logic.
* Residual DPCM often fights with your NLMS filter, which is already hunting for linear correlations in the residual space.
* Photographic images have distinct lighting, shadow, and saturation zones. A global RCT is always a compromise. Computing the optimal reversible transform block-by-block (e.g., 64x64) and transmitting a small flag removes massive cross-plane redundancy before the spatial model even has to wake up.

Given that the 13-tap NLMS eats 17.1% of your decode time for a marginal 1.73% byte reduction, have you considered dropping the online updates and replacing it with an offline-trained, fixed-weight predictor bank selected via a fast block-level classifier?