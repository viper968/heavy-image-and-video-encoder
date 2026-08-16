# Where this is and what to do next

Written so a cold session — or a human — can pick this up without reading the
history. Update it whenever the state below stops being true.

## State

Branch `claude/extreme-compression-experiment-ar5swo`, merged to `main` once
already (PR #1). Working tree should be clean; 36 tests pass via
`.venv/bin/python -m pytest tests -q`.

Held-out results (18 Kodak images, `tools/bench_image.py --jobs=12 test`):
hve 7,854,553 bytes — 30.6% under PNG, 3.0% under lossless WebP, **9.0% over
JPEG XL** (libjxl 0.12.0). Video: beats VP9 and, on akiyo, AV1; see README.

Environment notes for this machine:
- Python is PEP 668-managed. Everything runs from `.venv/bin/python`
  (`python3 -m venv --system-site-packages .venv`, then
  `pip install imagecodecs numba`). `.venv/` is gitignored.
- `testdata/` is gitignored: run `.venv/bin/python tools/fetch_testdata.py`
  (24 Kodak images + 2 Xiph clips, ~100MB).
- Git pushes need `gh auth setup-git` once.

## DONE: JIT the still-image path (kept here for context)

**Why.** The codec spends ~10s encoding and ~9s decoding one 768x512 photo. It
is a pure-Python per-pixel loop, and every remaining compression idea needs A/B
runs across a 24-image corpus, so speed is the gate on all of them.

**Verified feasible.** numba 0.67 works on Python 3.14. A micro-benchmark of a
loop shaped like the codec's hot path (adaptive binary coder + context lookup,
2M iterations) ran 1.01s pure Python vs 0.01s jitted — **188x**, byte-identical
output.

**Design.** `hve/model.py` stays as the readable reference. `hve/fast.py` holds
a numba-jitted equivalent over flat numpy arrays, and `code_plane` dispatches to
it when numba imports, falling back otherwise.

This is deliberately two implementations of the same loop, which is the thing
that bit this project before (video.py once carried a divergent copy of the
pixel loop). The difference is that a test pins them together **bit-exactly** —
both must emit identical bitstreams — so drift fails loudly instead of silently
corrupting the format. That is how real codecs ship a reference decoder
alongside an optimised one.

**Translation checklist** (the mechanical part):
- model dict of Python lists -> individual numpy int arrays passed as arguments
- `mix.Mixer` / `mix.APM` objects -> flat weight/table arrays + inlined logic
- `mix.STRETCH` / `SQUASH` -> module-level numpy arrays
- range coder state -> scalars + a preallocated uint8 output buffer with an index
- `flat` match history and the match hash table -> numpy arrays
- rows / prev / cur -> a 2D uint8 array

**Watch for** (numba int64 vs Python bigint): the mixer's dot product and weight
updates are the only places products get large; everything else is bounded by
construction. `>>` and `//` on negatives floor identically in both, so the
weighted-predictor blend and the video motion-vector scaling are safe. The
byte-exactness test is what actually proves this.

**Progress so far.** `hve/fast.py` exists and works. It holds the whole
still-image path — range coder, mixer, APM, weighted predictor, match model,
all of it — as one numba kernel over flat numpy arrays. Two tests in the suite
(`test_fast_path_is_byte_identical`, `test_fast_path_roundtrips`) confirm it
emits **byte-identical** payloads to the reference on a photo crop, a synthetic
image and pure noise, and that it round-trips. 38 tests green.

**Wired in and measured.** `image._encode_payload` / `_decode_payload` use the
jitted path when numba is importable and fall back to the reference otherwise.
kodim19 encodes to the same 462,749 bytes as before, in **0.29s instead of
9.7s** (33x) and decodes in 0.16s instead of 9.1s (56x). Across the 18 held-out
images hve went from 653.6s CPU to **21.1s** — now faster than both lossless
WebP (192.3s) and JPEG XL effort 9 (65.1s), at identical output sizes. README
updated.

**Still on the reference path: video.** `hve/video.py` calls
`model.code_plane(..., inter=...)`, and the kernel does not cover the per-block
prediction branch. Adding it is the same mirroring exercise, pinned by the same
kind of byte-exactness test. That is the obvious next speed win — video is now
the only slow path at ~1.1s per CIF frame.

## After that, in order

1. **Condition the sign and magnitude bins on the match model.** It currently
   only replaces the prediction when fully confident; when it is *fairly*
   confident its answer still implies an expected residual, which those bins
   ignore. Same mistake the first match-model attempt made, one level down.
2. Mixing (not just secondary estimation) on the magnitude bins.
3. A second mixing layer.
4. Hybrid-uint tokens: give small residuals their own jointly modelled symbol
   instead of decomposing every one into is-zero / sign / unary.

`docs/research.md` has the measured value of everything already tried,
including the four techniques that were built and rejected.
