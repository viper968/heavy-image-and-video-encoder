# heavy-image-and-video-encoder

A lossless image and video codec that compresses harder than PNG, lossless WebP,
FFV1, VP9 and AV1 by spending decode time instead of bits.

The output is not viewable in any normal viewer. That is the point: you send the
compressed blob plus this decoder, and the other end gets the **exact original
bytes** back — every pixel identical, verified on every benchmark run below.

```
python -m hve encode photo.png photo.hvi
python -m hve decode photo.hvi restored.png     # byte-identical pixels
python -m hve info   photo.hvi
```

## Results

### Images — 18 held-out Kodak photos (7.08M pixels, 21,233,664 raw bytes)

Every constant in this codec was chosen by measuring compressed size, so the six
images used for tuning cannot also carry the headline number. These 18 are the
held-out split (`tools/corpus.py`); no parameter has ever been fitted to them.

| codec | bytes | bpp | ratio | verified |
|---|---:|---:|---:|---|
| AVIF "lossless" (Pillow) | 4,915,600 | 5.56 | 4.32x | **not actually lossless** (max err 55) |
| JPEG XL, effort 9 | 7,346,399 | 8.30 | 2.89x | lossless |
| JPEG XL, effort 7 | 7,406,679 | 8.37 | 2.87x | lossless |
| **hve** | **7,854,553** | **8.88** | **2.70x** | lossless |
| WebP lossless | 8,099,860 | 9.16 | 2.62x | lossless |
| PNG (optimised) | 11,321,001 | 12.80 | 1.88x | lossless |

30.6% smaller than PNG, 3.0% smaller than lossless WebP, 6.9% larger than
JPEG XL. **JPEG XL is still ahead** — `docs/research.md` records what was tried
against that gap, what each technique was worth when measured, and what is
left.

### Video — 16 frames of `akiyo_cif` (352x288 YUV420, 2,433,024 raw bytes)

| codec | bytes | ratio | verified |
|---|---:|---:|---|
| x264 lossless (veryslow) | 321,064 | 7.58x | lossless |
| x265 lossless (veryslow) | 323,440 | 7.52x | lossless |
| **hve** | **326,472** | **7.45x** | lossless |
| AV1 lossless (libaom) | 340,300 | 7.15x | lossless |
| VP9 lossless | 347,551 | 7.00x | lossless |
| FFV1 level 3 | 745,937 | 3.26x | lossless |
| Ut Video | 1,117,093 | 2.18x | lossless |
| FFVHuff | 1,126,268 | 2.16x | lossless |

Beats AV1 and VP9 in lossless mode, 2.3x smaller than FFV1, within 2% of x264.

On high-motion content the ranking changes. Same 16-frame test on `foreman_cif`,
which pans and has fast head movement:

| codec | bytes | ratio | verified |
|---|---:|---:|---|
| x264 lossless | 846,092 | 2.88x | lossless |
| x265 lossless | 853,010 | 2.85x | lossless |
| AV1 lossless | 880,345 | 2.76x | lossless |
| VP9 lossless | 885,846 | 2.75x | lossless |
| **hve** | **892,588** | **2.73x** | lossless |
| FFV1 level 3 | 971,084 | 2.51x | lossless |

hve drops to last among the block codecs, 5.5% behind x264, while still beating
FFV1 by 8%. That gap is the price of a deliberately simple inter design: one
reference frame, full-pel vectors only, a single 16x16 block size, no
bidirectional prediction. Halving the block size to 8x8 was measured and made it
slightly *worse* (more mode and vector overhead than it saves), so the missing
ingredient is sub-pixel motion and variable block sizes, not finer blocks.

Reproduce with `python tools/bench_image.py testdata/images/*.png` and
`python tools/bench_video.py testdata/video/akiyo_cif.y4m 16`. Every baseline is
decoded and compared against the source before its size is reported — Pillow's
AVIF happily accepts `lossless=True` and returns a lossy image, which is exactly
why nothing here is trusted on its label.

## About the original Huffman idea

The starting plan was to Huffman-code each colour channel so that a value in
0..255 costs fewer than 8 bits. That works, but it is worth far less than it
sounds, and `tools/ladder.py` measures why. On the same 6 photos:

| rung | bytes | bpp | vs previous | what changed |
|---|---:|---:|---:|---|
| raw | 7,077,888 | 24.000 | — | uncompressed RGB |
| huffman-global | 6,680,683 | 22.653 | -5.6% | one Huffman tree over all bytes |
| huffman-perchan | 6,574,377 | 22.293 | -1.6% | **a tree per colour channel — the original idea** |
| huffman-med | 4,679,107 | 15.866 | -28.8% | + MED spatial prediction first |
| huffman-rct-med | 3,319,109 | 11.255 | -29.1% | + reversible colour transform |
| rans-ctx | 3,150,770 | 10.684 | -5.1% | + context-modelled static rANS |
| **hve** | **3,005,344** | **10.191** | -4.6% | + weighted prediction, adaptive contexts, mixing |

Per-channel Huffman on raw pixel values buys **7.1% total**, and splitting the
tree per channel accounts for only 1.6 points of that. The reason is that the
*values* in a photo are close to uniformly distributed — a picture uses most of
the range — so no symbol code can do much with them. What is highly skewed is the
**difference between a pixel and a prediction of it from its neighbours**.
Predict first, and the very same Huffman code pays 28.8%.

Two further limits of Huffman show up after that:

- **It cannot spend less than one bit on a symbol.** After good prediction the
  most common residual (exactly zero) can be 60%+ of all pixels, which deserves
  about 0.7 bits. Huffman is forced to charge 1. Arithmetic/range coding is not.
- **It needs one transmitted table per context.** The whole gain of context
  modelling is that a pixel in smooth sky and a pixel on a hard edge get
  different probability tables — but each static table costs header bytes.
  Measured in `tools/ctx_study.py`, a 365-context JPEG-LS-style model spends
  497KB of header on this set and comes out *worse* overall.

Both are why the final codec uses an **adaptive binary range coder**: it
transmits no tables at all, because encoder and decoder learn identical
probabilities from the pixels they have already processed. Contexts become free,
so the model carries ~10,000 adaptive probability slots and pays nothing for them,
and several of them can be *mixed* per bit rather than one being chosen.

## How it works

**Images** (`hve/image.py`)

1. **Reversible colour transform** — `Y = G + (Cb + Cr)/4`, `Cb = B - G`,
   `Cr = R - G`, all in modular 8-bit arithmetic so nothing widens past a byte
   and nothing is lost. The encoder measures residual entropy with and without
   it and sets a header flag, because on images whose channels do not correlate
   the transform is a loss.
2. **Self-correcting weighted prediction** (`hve/model.py`) — four
   sub-predictors (the JPEG-LS median edge detector, `W + NE - N`, north and
   west) are blended, each weighted by how wrong it has recently been *right
   here*. Where one is reliably right — a vertical edge, a smooth gradient — it
   takes over locally with nothing signalled.
3. **Context modelling** (`hve/model.py`) — the residual is coded as
   *is-it-zero* / *sign* / *bit-length* / *mantissa*, with contexts drawn from
   local gradient activity, how badly neighbouring pixels were predicted, and —
   for chroma — how badly the co-located luma pixel was predicted. Chroma tends
   to go wrong exactly where luma did.
4. **Match model** (`hve/model.py`) — hashes the causal neighbourhood and
   remembers where that exact neighbourhood last occurred anywhere earlier in
   the plane. Once its answer has held for eight consecutive pixels it replaces
   the prediction outright. Gradient predictors cannot see repetition; this
   makes spatially incompressible but repetitive content — tiled textures,
   lettering, screenshots — collapse by an order of magnitude.
5. **Context mixing** (`hve/mix.py`) — the zero flag, which every pixel pays,
   is predicted by four experts that view the neighbourhood differently
   (including the match model) and combined by an LPAQ-style logistic mixer
   whose weights are learned online and selected by context. A secondary
   estimation stage (APM/SSE) then corrects the result's calibration, on the
   zero flag and the magnitude bins alike.
6. **Adaptive binary range coder** (`hve/rc.py`) — LZMA-style, 15-bit
   probabilities, no transmitted tables.

**Video** (`hve/video.py`)

The first frame is coded as a still image. Every later frame is split into 16x16
blocks, each independently choosing between spatial prediction and temporal
prediction from the previous frame at a motion vector found by full search over
±8 pixels. Modes and vectors are coded first, so pixel residuals can still be
coded in plain raster order with full access to their spatial neighbours.
Temporally predicted pixels get their own set of probability banks, since their
residuals look nothing like spatial ones.

Reads and writes `.y4m`, so real test clips work end to end and chroma planes
are coded at their native subsampled size.

## Honest limitations

- **It is slow, and got slower.** This is a readable Python reference
  implementation: ~25s to encode a 768x512 photo, up from 3.2s before the
  mixing layer went in — context mixing and secondary estimation run per coded
  bit. Video is about 1.6s per CIF frame. The algorithms are all O(pixels) — the constant is Python. A C port
  would land in the same class as the codecs it is compared against.
- **JPEG XL still wins on stills** by 6.9% on held-out images. Context mixing,
  secondary estimation, a self-correcting weighted predictor and an online
  learned context tree were all built and measured against that gap; the first
  three are in, the fourth was not worth its cost. See `docs/research.md` for
  every number, including the techniques that made things *worse*.
- **Video uses only the previous frame**, full-pel motion, one block size, and
  no bidirectional prediction. x264's lossless mode remains slightly ahead, and
  on high-motion content the margin is wider (see `results/video_foreman.txt`).
- **Already-compressed input is not the target.** Feeding a JPEG in means
  decoding it to pixels and re-compressing those pixels losslessly, which will
  usually be *larger* than the JPEG. Shrinking an existing JPEG/AVIF/H.265 file
  without quality loss is a different problem (lossless recompression of the
  entropy-coded stream, as Lepton does for JPEG) and is not implemented here.

## Layout

```
hve/rc.py           adaptive binary range coder (the entropy engine)
hve/mix.py          logistic mixing + secondary estimation (stretch/squash, Mixer, APM)
hve/model.py        weighted predictor, match model, context model, residual
                    binarisation — one loop, shared by image and video paths
hve/image.py        .hvi still-image container
hve/video.py        .hvv video container, motion search, block modes
hve/transform.py    RCT, MED predictor, context quantisation
hve/rans.py         static rANS - the earlier design, kept for the ladder comparison
hve/huffman.py      canonical Huffman - the original idea, kept as a working codec
hve/y4m.py          YUV4MPEG2 reader/writer
hve/cli.py          command line interface
tools/ladder.py     measures every rung from Huffman to the final codec
tools/bench_image.py, tools/bench_video.py    benchmarks with losslessness verified
tools/corpus.py     the dev / held-out split, so tuning cannot flatter the results
tools/headroom.py   ideal cost per predictor: is the gap in prediction or coding?
tools/ctx_study.py, tools/pred_study.py, tools/tune.py   the design experiments
docs/research.md    every technique surveyed and what it measured — including the
                    ones that made things worse
tests/              36 tests: roundtrips, edge cases, coder internals
```

## Running it

```bash
pip install numpy pillow                 # imagecodecs and pytest for benchmarks/tests
python -m pytest tests -q
python tools/fetch_testdata.py           # Kodak photos + Xiph clips (not committed)
python tools/ladder.py testdata/images/*.png
python tools/bench_image.py test          # held-out split; "dev" or "all" also work
python tools/bench_video.py testdata/video/akiyo_cif.y4m 16
```
