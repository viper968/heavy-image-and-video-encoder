# heavy-image-and-video-encoder

A lossless image and video codec that compresses harder than PNG, lossless WebP,
FFV1, VP9 and AV1 by spending decode time instead of bits.

The output is not viewable in any normal viewer. That is the point: you send the
compressed blob plus this decoder, and the other end gets the **exact original
bytes** back — every pixel identical, verified on every benchmark run below.

Just want to try it on your own files? `playground/README.md` — two commands
from a fresh clone, and it handles ordinary jpg/mp4 input rather than only the
PNG and .y4m the core CLI takes:

```
./playground/setup.sh
./playground/hve demo
./playground/hve check my_photo.jpg      # round trip, verifies every pixel
```

Picking this up to work on it? Read `docs/HANDOFF.md` first.

## Use it

There is a standalone binary with no Python in it at all — no interpreter, no
numpy, no libpng, no zlib. One C compiler and `make`:

```
make -C csrc                        # -> build/hve   (or `make -C csrc static`)
./build/hve encode photo.png photo.hvi
./build/hve decode photo.hvi restored.png     # byte-identical pixels
./build/hve info   photo.hvi
./build/hve encode clip.y4m clip.hvv --frames 60
```

Statically linked it is 1.1 MB and depends on nothing. `make -C csrc windows`
cross-compiles a `.exe` with mingw-w64 (written but **not yet tested** — see
`csrc/Makefile`).

The Python package does the same thing and is what the research tooling uses:

```
python -m hve encode photo.png photo.hvi
python -m hve decode photo.hvi restored.png
python -m hve info   photo.hvi
```

The two produce **byte-identical files** and read each other's output;
`tests/test_cli_binary.py` checks that on every shape it can think of. Use the
binary to ship, the package to experiment.

## Results

### Images — 18 held-out Kodak photos (7.08M pixels, 21,233,664 raw bytes)

Every constant in this codec was chosen by measuring compressed size, so the six
images used for tuning cannot also carry the headline number. These 18 are the
held-out split (`tools/corpus.py`); no parameter has ever been fitted to them.

| codec | bytes | bpp | ratio | cpu s | verified |
|---|---:|---:|---:|---:|---|
| JPEG XL, effort 9 | 7,207,847 | 8.15 | 2.95x | 60.0 | lossless |
| JPEG XL, effort 7 | 7,305,646 | 8.26 | 2.91x | 14.8 | lossless |
| **hve** | **7,711,460** | **8.72** | **2.75x** | **16.4** | lossless |
| WebP lossless | 8,099,860 | 9.16 | 2.62x | 169.0 | lossless |
| PNG (optimised) | 11,321,001 | 12.80 | 1.88x | 6.1 | lossless |

`cpu s` is encode **and** decode for all 18 images, measured the same way for
every codec.

31.9% smaller than PNG, 4.8% smaller than lossless WebP, 7.0% larger than
JPEG XL. **JPEG XL is still ahead** — `docs/research.md` records what was tried
against that gap, what each technique was worth when measured, and what is
left.

The largest single compression change was replacing the error-weighted averaging
blend with a *learned* combiner over a wider predictor set, which closed the gap
from 8.2% to 7.0%. The bytes have not moved since; the `cpu s` column has,
because the codec's core loop is now compiled C (23.6s to 16.4s for the same
output). Treat that column as approximate: a repeat run of this same benchmark
moved JPEG XL e9 from 63.9s to 77.9s on identical code, so cross-run timing
differences under about 20% are machine load, not signal. The byte counts are
exact and reproducible.

Measured with libjxl 0.12.0, libwebp 1.6.0, Pillow 12.3.0. An earlier run on
libjxl 0.11 put JPEG XL at 7,346,399 and the gap at 6.9%; hve produced byte-
identical output on both machines, so the whole 2.1-point move is the reference
encoder improving, not this one regressing. Baseline versions are printed with
every benchmark run for exactly this reason. (That earlier machine also had a
Pillow with AVIF, whose `lossless=True` returned visibly lossy output — max
error 55 — which is why nothing here is trusted on its label.)

### Video — 16 frames of `akiyo_cif` (352x288 YUV420, 2,433,024 raw bytes)

| codec | bytes | ratio | verified |
|---|---:|---:|---|
| **hve** | **316,313** | **7.69x** | lossless |
| x264 lossless (veryslow) | 321,053 | 7.58x | lossless |
| x265 lossless (veryslow) | 323,443 | 7.52x | lossless |
| AV1 lossless (libaom) | 329,528 | 7.38x | lossless |
| VP9 lossless | 348,041 | 6.99x | lossless |
| FFV1 level 3 | 745,942 | 3.26x | lossless |
| Ut Video | 1,117,097 | 2.18x | lossless |
| FFVHuff | 1,126,272 | 2.16x | lossless |

Ahead of x264, x265, AV1 and VP9 in lossless mode, and 2.4x smaller than FFV1.
Measured with ffmpeg 6.1.6.

High-motion content used to be where this lost. Same 16-frame test on
`foreman_cif`, which pans and has fast head movement:

| codec | bytes | ratio | verified |
|---|---:|---:|---|
| **hve** | **834,903** | **2.91x** | lossless |
| x264 lossless | 846,081 | 2.88x | lossless |
| AV1 lossless | 851,838 | 2.86x | lossless |
| x265 lossless | 852,986 | 2.85x | lossless |
| VP9 lossless | 886,928 | 2.74x | lossless |
| FFV1 level 3 | 971,089 | 2.51x | lossless |

Now first here too, 1.3% ahead of x264. This was 883,078 and fourth two changes
ago. Half of the move came from the learned combiner, which was built for still
images and never tuned on video; the rest came from **half-pel motion vectors**,
which is the one thing this README previously named as the missing ingredient.

Sub-pixel motion matters because real movement does not land on the pixel grid.
Rounding a vector to the nearest whole pixel leaves a residual that no amount of
context modelling recovers, and the effect is confined to exactly the clips that
move: measured before it was built, half-pel was worth 5-8% on the three
high-motion dev clips and essentially nothing on the near-static ones.

The other half of that old prediction was wrong. **Variable block sizes were
measured and rejected**: splitting 16x16 into four 8x8 vectors scored *worse* on
every clip, by 1.2% on bus and 22.5% on akiyo, because the extra vectors cost
more than the sharper motion boundaries save. `docs/research.md` records that
alongside the other rejected idea from this round.

The inter design is still deliberately simple: one reference frame, one block
size, no bidirectional prediction.

### Video — 16 frames of 1080p (`sintel_trailer_2k`, 49,766,400 raw bytes)

CIF clips say nothing about how any of this scales, so here is the same test at
1920x1080. This part of the trailer is nearly static, which is why every ratio
is enormous; treat the ordering as the result and the absolute numbers as a
property of the content.

| codec | bytes | ratio | enc s | verified |
|---|---:|---:|---:|---|
| **hve** | **30,944** | **1608x** | 3.0 | lossless |
| x264 lossless | 34,607 | 1438x | 0.3 | lossless |
| AV1 lossless | 43,372 | 1147x | 1.5 | lossless |
| x265 lossless | 46,281 | 1075x | 1.3 | lossless |
| VP9 lossless | 50,265 | 990x | 0.7 | lossless |
| FFV1 level 3 | 521,316 | 95x | 0.2 | lossless |

10.5% smaller than x264 and 29% smaller than AV1 — and now within **about 10x**
of x264's encode time rather than the 50x this README used to claim, on a clip
where it is also faster than AV1 and x265 were when this table was first
written.

That number has moved twice and the honest version of the story is:

- 145s, when the motion search was 289 full-frame numpy passes per frame;
- 19.8s after rewriting the search — but that figure included numba's cold
  compilation. Measured warm, the numba path does this clip in **9.8s**, so the
  old "50x slower than x264" was inflated by about 2x and should have been ~25x;
- **2.7s warm** now that the core loop is compiled C and the motion search is
  threaded, which is what the 3.0s above measures from cold.

The motion search itself went from 4.4s to 0.12s, because every block's search
is independent and so threads across cores for free. The remaining 2.5s is the
pixel loop, which is strictly serial — every pixel depends on all prior ones —
and cannot be threaded without changing the format. `docs/research.md` records
what that change would cost.

Reproduce with `python tools/bench_image.py test` and
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
| **hve** | **2,925,843** | **9.921** | -7.1% | + weighted prediction, learned combiner, adaptive contexts, mixing |

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
3. **A learned combiner on top of that blend** (`hve/model.py`) — thirteen
   predictors, including the whole second ring of neighbours and CALIC's
   nonlinear GAP, are combined by weights learned online with normalised least
   mean squares. The blend from step 2 is the *origin* of that combination, so
   a zero weight vector reproduces it exactly and the layer can only earn its
   way in. This matters because the weights are learned jointly against the
   real error and are free to go negative, where an error-weighted average
   weights each member in isolation, forces the weights positive, and so is
   dragged by a volatile member however much it is down-weighted. Four earlier
   attempts to add predictors to the average all failed for that reason; under
   the learned combiner the same class of predictor pays 1.1%.
4. **Context modelling** (`hve/model.py`) — the residual is coded as
   *is-it-zero* / *sign* / *bit-length* / *mantissa*, with contexts drawn from
   local gradient activity, how badly neighbouring pixels were predicted, and —
   for chroma — how badly the co-located luma pixel was predicted. Chroma tends
   to go wrong exactly where luma did.
5. **Match model** (`hve/model.py`) — hashes the causal neighbourhood and
   remembers where that exact neighbourhood last occurred anywhere earlier in
   the plane. Once its answer has held for eight consecutive pixels it replaces
   the prediction outright. Gradient predictors cannot see repetition; this
   makes spatially incompressible but repetitive content — tiled textures,
   lettering, screenshots — collapse by an order of magnitude.
6. **Context mixing** (`hve/mix.py`) — the zero flag, which every pixel pays,
   is predicted by five experts that view the neighbourhood differently
   (including the match model, and the learned combiner's own confidence)
   and combined by an LPAQ-style logistic mixer
   whose weights are learned online and selected by context. A secondary
   estimation stage (APM/SSE) then corrects the result's calibration, on the
   zero flag and the magnitude bins alike.
7. **Adaptive binary range coder** (`hve/rc.py`) — LZMA-style, 15-bit
   probabilities, no transmitted tables.

`hve/model.py` is the readable reference implementation and the definition of
the format. `csrc/kernel.c` is the same loop in C — covering both stills and
video, including video's per-block prediction branch — built on first import
and used automatically whenever a compiler is present. Keeping two
implementations of a codec's core loop is exactly how formats get silently
corrupted, so tests require them to emit **byte-identical** bitstreams; drift
fails loudly instead of quietly writing files only one path can read.

There was briefly a third, in numba. It was deleted once the C existed: every
one of the five model changes in this repo's history had to be mirrored into it
by hand, and it stopped adding any coverage the C-versus-reference check does
not already give. That reasoning is in `docs/research.md`, because deciding
*not* to keep an implementation is as much a result as writing one.

**Video** (`hve/video.py`)

The first frame is coded as a still image. Every later frame is split into 16x16
blocks, each independently choosing between spatial prediction and temporal
prediction from the previous frame at a motion vector. Vectors are in **half-pel
units**, found by exhaustive whole-pixel search over ±8 pixels followed by a
refinement over the eight neighbouring half-pel positions; the reference is
bilinearly interpolated to four phases and the vector's low bit selects one.
Chroma divides the luma vector down first, which lands it on quarter-pel
positions of the luma grid for free. Vectors are predicted from the median of
the left, above and above-right neighbours, as in H.264.

Modes and vectors are coded first, so pixel residuals can still be coded in
plain raster order with full access to their spatial neighbours. Temporally
predicted pixels get their own set of probability banks, since their residuals
look nothing like spatial ones.

Reads and writes `.y4m`, so real test clips work end to end and chroma planes
are coded at their native subsampled size.

## Honest limitations

- **The standalone binary is about shipping, not speed.** It removes Python
  startup and numpy's import — a flat ~0.15s — so a small still encodes 1.7x
  faster end to end (0.23s against 0.38s) while 16 frames of 1080p come out
  only 5% faster (2.79s against 2.93s). That is the honest shape of it: the
  Python path was already calling the same C kernel, so there was never much
  interpreter overhead left to remove. What the binary buys is a 1.1 MB
  executable that needs no interpreter, no numpy, no libpng and no zlib.
- **Speed** (12th-gen i5, 16 cores, Python 3.14, core loop compiled from
  `csrc/`): 16 CIF video frames encode in 0.42s and decode in 0.37s; 16 frames
  of 1080p encode in 2.7s and decode in 2.6s. The **motion search is threaded**
  and costs almost nothing now (0.12s at 1080p), so encode time is essentially
  the pixel loop, which is **strictly serial and single-threaded** because every
  pixel depends on all prior ones. That is a property of the format, not of the
  implementation, and it is the remaining reason encode is ~10x slower than
  x264 rather than comparable. The C library is built automatically on first
  import; without a compiler the codec falls back to the pure-Python reference,
  which is roughly 100x slower and produces identical bytes. That fallback is a
  correctness guarantee, not a usable mode — budget about a minute per Kodak
  photo. `tools/bench_image.py --jobs=N` parallelises across images, not within
  one.
- **JPEG XL still wins on stills** by 7.0% on held-out images. Nineteen
  techniques have been built and measured against that gap; eight are in and
  eleven were rejected, several of them ideas the literature rates highly.
  Instrumenting the encoder showed the remaining gap was in prediction rather
  than in the entropy coder or the binarisation — raw unmodelled bits are only
  4.6% of the file — and the learned combiner is what acted on that finding.
  `docs/research.md` has every number and every reason.
- **Video uses only the previous frame**, one block size, and no bidirectional
  prediction. It now leads x264, x265, AV1 and VP9 on both test clips, but that
  is two CIF clips and should not be read as a general claim — the encoder is
  also far slower than any of them, and multiple reference frames and
  bidirectional prediction are the obvious things it still does not do.
- **Already-compressed input is not the target.** Feeding a JPEG in means
  decoding it to pixels and re-compressing those pixels losslessly, which will
  usually be *larger* than the JPEG. Shrinking an existing JPEG/AVIF/H.265 file
  without quality loss is a different problem (lossless recompression of the
  entropy-coded stream, as Lepton does for JPEG) and is not implemented here.

## Layout

```
hve/rc.py           adaptive binary range coder (the entropy engine)
hve/mix.py          logistic mixing + secondary estimation (stretch/squash, Mixer, APM)
hve/model.py        weighted predictor, learned combiner, match model, context
                    model, residual binarisation — the readable reference,
                    shared by image+video
hve/native.py       loads and binds the C kernel; builds it on first import
csrc/kernel.c       the pixel loop in C, for stills and video alike, pinned
                    byte-identical to model.py by tests — this is what runs
csrc/motion.c       threaded motion search (encoder only, no format impact)
csrc/hve.h          shared structs, and the floor-division and shift helpers
                    that keep C's integer semantics matching Python's
csrc/container.c    the .hvi and .hvv containers
csrc/transform.c    RCT and the entropy proxy that decides whether to use it
csrc/y4m.c, imageio.c, util.c, main.c    y4m, PNG, buffers, the CLI
csrc/model_constants.h   GENERATED from model.py by tools/gen_model_constants.py
csrc/third_party/   vendored lodepng (zlib licence), so PNG needs no libpng
build/hve           the standalone binary (gitignored; `make -C csrc`)
hve/image.py        .hvi still-image container
hve/video.py        .hvv video container, motion search, block modes
hve/transform.py    RCT, MED predictor, context quantisation
hve/rans.py         static rANS - the earlier design, kept for the ladder comparison
hve/huffman.py      canonical Huffman - the original idea, kept as a working codec
hve/y4m.py          YUV4MPEG2 reader/writer
hve/cli.py          command line interface
tools/ladder.py     measures every rung from Huffman to the final codec
tools/bench_image.py, tools/bench_video.py    benchmarks with losslessness verified
tools/quick.py      dev-set size reading in ~2s, the A/B harness for model work
tools/corpus.py     the dev / held-out split, so tuning cannot flatter the results
tools/headroom.py   ideal cost per predictor: is the gap in prediction or coding?
tools/ctx_study.py, tools/pred_study.py, tools/tune.py   the design experiments
playground/         setup + a friendly CLI for trying it on ordinary jpg/mp4
                    files; nothing here is part of the codec
docs/HANDOFF.md     start here: state, environment, workflow, what to do next
docs/research.md    every technique surveyed and what it measured — including the
                    eight that were built and rejected
tests/              96 tests: roundtrips, edge cases, coder internals,
                    byte-exactness between the C kernel and the reference, and
                    between the standalone binary and the Python package
```

## Running it

```bash
pip install numpy pillow                 # plus a C compiler (cc/gcc/clang) for
                                         # the fast path; pure Python otherwise
                                         # imagecodecs and pytest for benchmarks/tests
python -m pytest tests -q
python tools/fetch_testdata.py           # Kodak photos + Xiph clips (not committed)
python tools/ladder.py testdata/images/*.png
python tools/bench_image.py --jobs=12 test   # held-out split; "dev"/"all" also work
python tools/bench_video.py testdata/video/akiyo_cif.y4m 16
```
