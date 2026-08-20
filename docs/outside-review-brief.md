# hve: a brief for an outside reviewer

Paste this whole file into another model (Gemini, ChatGPT, DeepSeek, whatever)
and ask the questions at the end. It is written to be self-contained and to be
*hard to give a useless answer to*: the long "already tried" section exists
because the obvious suggestions have all been measured here, and an answer that
repeats one of them is worth nothing.

Everything below is measured on one machine (i5-12500H, 16 threads, gcc
-O3 -march=native) unless stated. Byte counts are exact; timings are minimum of
7-13 runs, and decode timings are pinned to one P-core because unpinned runs
drift 3-4% from P/E-core migration.

## What it is

A **lossless** image and video codec. Not lossy. It exists to compress harder
than PNG/WebP-lossless/FFV1 by spending decode time.

Pipeline, per plane:

1. **Colour**: reversible RCT for stills, chosen per image by comparing
   zeroth-order residual entropy against no transform. Video is YUV 4:2:0/2:2/4:4:4.
2. **Prediction**: LOCO-I/MED median predictor, then optionally a GAP variant, a
   13-tap NLMS adaptive filter, a 4-predictor self-correcting weighted blend,
   and a hash-based match model. Video adds 16x16 block motion compensation with
   half-pel vectors and a median MV predictor.
3. **Residual coding**: the residual byte is folded to [-128,127] and coded as
   a **zero flag**, then sign, then a unary **exponent** (magnitude bit-length,
   capped at 7), then mantissa bits (2 modelled, rest bypass).
4. **Modelling**: LPAQ-style context mixing. The zero flag is predicted by 5
   experts (each an adaptive 15-bit probability in its own context bank),
   combined by a **logistic mixer** (weights learned online by gradient descent,
   weight set selected by an activity context), then refined by two **APM/SSE**
   stages. The exponent chain has its own 3 experts + mixer + APM.
5. **Entropy coding**: LZMA-style binary adaptive range coder, 15-bit
   probabilities, byte-at-a-time renormalisation.

Parallelism: independent horizontal **slices** (each relearns the model, so it
costs ratio: +0.69% at 16 slices on 1080p, +4.1% on CIF). Motion search is
threaded and effectively free.

There are two implementations that must produce **byte-identical** output: a
readable Python reference (`hve/model.py`, defines the format) and a C kernel
(`csrc/kernel.c`, ~100x faster, what actually runs). Tests enforce identity.

## Where it stands

**Stills**, 18 held-out Kodak images (never used for tuning):

| codec | bytes | decode speed |
|---|---:|---:|
| JPEG XL effort 9 | 7,207,847 | 3.7 Mpixel/s |
| **hve** | **7,702,223** (+6.86%) | 2.1 Mpixel/s |
| WebP lossless | 8,099,860 | 63 Mpixel/s |
| PNG optimised | 11,321,001 | 52 Mpixel/s |

**Video**, single-threaded decode, same clip:

| clip | FFV1 | hve | delta |
|---|---:|---:|---|
| akiyo CIF x16 (static camera) | 746,181 @ 171 fps | 315,981 @ 134 fps | **-57.7%** |
| bus CIF x30 (panning) | 2,294,461 @ 158 fps | 1,964,455 @ 77 fps | -14.4% |
| Sintel 1080p x16 (real motion) | 3,647,022 @ 29 fps | 3,652,863 @ 11.3 fps | -0.3% |

For scale: x264 **lossy** crf23 on that 1080p clip is 142,334 bytes @ 195 fps,
and AV1 crf32 is 34,148 @ 201 fps. We are not competing with those and know it.

## Hard constraints

- **Lossless only.** A lossy mode is out of scope for this question.
- The **format may change** freely (pre-1.0, magic bumped several times).
- Two implementations must stay byte-identical.
- Ratio must be measured on a **held-out split** (6 dev / 18 test Kodak images;
  3 dev / 2 test CIF clips). Dev-set-only wins are not accepted.
- 8-bit only today. No hardware anything.

## Already tried and MEASURED - do not re-propose these

Compression:

- **Learned context tree (JPEG XL MA / FLIF MANIAC style)** - built, rejected.
- **Multi-symbol / rANS entropy coding (AV1 Daala EC)** - a build with the
  entropy coder *entirely deleted* is only 7.0% faster, so that is the ceiling.
  The model, not the coder, is the cost.
- **Hybrid-uint token binarisation** - raw bypass bits are only 3.3-4.6% of the
  file, which caps the whole idea.
- **More linear predictors in the combiner** - the linear span is covered; only
  nonlinear members (MED, GAP) carry the change.
- **Variable block sizes for motion** - measured, rejected.
- **Wavefront parallelism (HEVC WPP)** - infeasible: WPP works because CABAC's
  context set is a few hundred bytes; this model's state is 4.5 MB, so
  checkpointing per row is 304 MB of copying per 1080p frame.
- **A second SSE/APM stage** - it is in, and it is worth +0.003% to +0.09%.
  Both APM stages together buy +0.033% (dev CIF) to +0.21% (dev stills).
- **Clamping the mixer weights** - the weights grow unbounded (92M after 120
  frames) but the mixer saturates on only 0.0003% of decisions, so there is
  nothing to fix.

Video-specific, and this is the biggest open puzzle:

- **Motion compensation is worth 59% on akiyo and 0.70% on 1080p Sintel**, where
  only **0.9% of blocks choose inter mode**. Checked three ways: widening the
  search radius 8->64 makes files monotonically BIGGER (the search scores
  residuals and cannot see that MV magnitudes are coded unary); the vectors are
  not clipping (p90 magnitude stays at 16 half-pels when the cap is lifted); and
  sweeping the inter/intra penalty 0->768 moves 1080p by 0.09% with its optimum
  *higher* than current, i.e. wanting fewer inter blocks. Working theory: at
  high resolution the spatial model already predicts nearly everything, while
  the temporal residual still carries estimation error and grain.

Speed (the kernel is **compute-bound**: IPC 4.40, LLC misses 0.012/sample,
~480 instructions per sample):

- **Prefetching model entries** - nothing to hide at IPC 4.4.
- **Narrowing the banks from int64 to int16/uint16** - halves L1 misses, costs
  +0.55% to +1.94% instructions, moves wall clock zero.
- **Forcing AVX2 gather for the ladder lookups** - removes 10% of instructions
  and 0% of wall time.
- **Replacing ladder lookups with comparison cascades** - worse than the tables.
- **Precomputing the previous-row half of the derivation** - -6.0% instructions,
  -1.0% cycles, wall clock straddles zero.
- The empirical rule that emerged: **count loads, not instructions and not cache
  misses.** There is spare ALU capacity and no spare load-port capacity, so the
  only changes that pay are ones that remove whole stages.

What *did* pay recently: deleting two APM stages and two experts whose context
was provably constant (-24% decode for +0.5% bytes).

## Questions

1. **Stills.** We are 6.86% behind JPEG XL lossless. Where does that gap most
   likely live, given the pipeline above? JXL's lossless path is Modular mode
   with a learned MA tree, self-correcting weighted predictors and hybrid-uint
   tokens with context clustering. We have the weighted predictor and we tried
   the learned tree. What is the *third* thing we are missing?

2. **The 1080p motion puzzle.** Is the working theory right that spatial
   prediction simply dominates at high resolution for lossless, or is there a
   known technique that would make temporal prediction pay there? Note we have
   no rate-aware motion search - is that alone likely to explain it?

3. **Context modelling.** With 5 experts on the zero flag and 3 on the exponent,
   what context *sources* are conspicuously absent? We use: activity (sum of 3
   folded neighbour differences), local error energy, a luma-error map for
   chroma, gradient pairs, sign pairs, match state, NLMS confidence.

4. **Anything structurally different** worth trying for lossless specifically -
   e.g. from FFV1, CharLS/JPEG-LS, LOCO-A, CALIC, MRP (minimum-rate
   predictors), or the lossless modes of AV2/VVC - that is not on the rejected
   list.

5. **Be adversarial**: what in this design would you expect to be a dead end,
   and what would you drop entirely?

Answers that just say "try context mixing" or "use a better entropy coder" are
not useful - read the rejected list first.
