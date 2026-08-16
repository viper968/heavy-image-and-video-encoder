# Playground

For trying the codec on your own images and videos without reading the rest of
the repo first.

```bash
git clone <this repo> && cd heavy-image-and-video-encoder
./playground/setup.sh
./playground/hve demo
```

`setup.sh` builds `.venv` and installs everything into it. `demo` compresses a
sample, restores it, checks every pixel came back, and shows the size against
PNG, WebP and JPEG XL. No downloads and no arguments.

Then point it at your own files:

```bash
./playground/hve check   photo.jpg        # round trip and verify every pixel
./playground/hve compare photo.png        # size vs PNG, WebP, JPEG XL
./playground/hve compress clip.mp4        # -> clip.hvv
./playground/hve restore  clip.hvv        # -> clip_restored.mkv, plays in VLC
./playground/hve info     clip.hvv        # what is in the file
```

`ffmpeg` is needed for video (`brew install ffmpeg`, `apt install ffmpeg`,
`xbps-install ffmpeg`). Images work without it.

The **first** encode or decode after install takes a few extra seconds while the
C library and the numba kernels are compiled. Both cache to disk, so every run
after that is fast. If you see several seconds for a small image, that was it —
run it again.

Speed depends on what is installed, but the **output bytes never do**: with a C
compiler you get the native path, without one the numba path (2-4x slower),
without either pure Python (~30x slower). All three produce identical files.

## The five commands

| command | what it does |
|---|---|
| `demo` | generated sample, full round trip, no arguments. Start here. |
| `check` | compress → restore → compare every pixel. Prints `LOSSLESS` or fails. |
| `compare` | your image against PNG, lossless WebP and JPEG XL |
| `compress` / `restore` | the actual codec, in and out |
| `info` | dimensions, frame count and size of a compressed file |

Everything takes `--frames N` for video (default 60, `--frames 0` for all).
Encoding runs at roughly half a megapixel per second, so a whole clip is rarely
what you want on the first try — 10 seconds of 1080p is about 20 minutes.

## Four things that will confuse you otherwise

**Your JPEG will be bigger after compression, and that is not a bug.** A JPEG
already threw pixels away permanently. This codec cannot do that, so of course
it produces more bytes. The fair comparison is against another lossless codec —
`compare` shows it. On a real photo hve lands about 32% below PNG and 5% below
lossless WebP, and about 7% above JPEG XL.

**"Lossless" means pixels, not files.** Restoring gives back every pixel value
exactly. It does not give back the original file byte-for-byte, and EXIF,
colour profiles and other metadata are not carried through.

**8 bits per channel only.** 16-bit PNGs, CMYK TIFFs and HDR images are
refused rather than silently converted, because converting them would lose data
*before* the encoder ran and the tool would then report "lossless" about a
pipeline that was not. `--force` overrides this and says so.

**Video gets converted to 8-bit 4:2:0 first.** If your source is 10-bit or
4:4:4, that conversion is lossy and the tool warns you. The codec round trip is
still exact — it is the step *before* the codec that lost something. The
`LOSSLESS` result is about the codec, and the warning tells you when the
pipeline around it was not.

## Is it any good?

On `check`, watch the "smaller than the raw pixels" line, and use `compare` for
anything conclusive. Two real results from this machine:

- **kodim23.png**, 768×512 photo — hve 398,357 B, WebP lossless 422,106 B,
  PNG 556,888 B, JPEG XL e9 372,253 B.
- **8 frames of 736×480 video** — hve 363,213 B against FFV1 at 1,331,985 B,
  which is 3.7× smaller for identical pixels.

It is slower than all of them. That is the trade the codec exists to make.

## Poking at the actual codec

```python
import numpy as np
from PIL import Image
from hve import image

arr = np.array(Image.open("photo.png").convert("RGB"))
blob = image.encode(arr)                       # bytes
assert np.array_equal(image.decode(blob), arr) # always true
```

`hve/model.py` is the readable reference implementation and the definition of
the format; `hve/fast.py` is the same loop compiled with numba and `csrc/kernel.c`
is the same loop again in C. All three must stay byte-identical. If you change
one, change all three and run `pytest tests -q` — a whole test file exists purely
to catch the paths diverging. `docs/HANDOFF.md` explains
the layout, and `docs/research.md` records every technique tried and what it
measured, including the eleven that were rejected.

Benchmarks against the real corpus need the test data:

```bash
.venv/bin/python tools/fetch_testdata.py     # 24 Kodak images + 2 clips, ~100MB
.venv/bin/python tools/bench_image.py --jobs=8 test
```
