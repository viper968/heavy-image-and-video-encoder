# Start here

Written for someone — or some model — picking this up cold. Read this first,
then whichever of the three documents below matches what you are about to do.
Update this file when the state it describes stops being true; a stale handoff
is worse than none.

**Last verified: commit `6ba47bd`, all 39 tests green.**

## What this is

A lossless image and video codec that compresses harder than PNG, lossless
WebP, FFV1, VP9 and AV1 by spending decode time instead of bits. The output is
not viewable in any standard viewer — you ship the blob plus this decoder and
get the original bytes back. It started from a much smaller idea (Huffman-code
each colour channel) and the README tells that story with measurements.

## Where the information lives

| document | what is in it | read it when |
|---|---|---|
| `README.md` | what the codec is, current benchmark tables, how it works, honest limitations | you need the current numbers or the architecture |
| `docs/research.md` | **every technique tried, with its measured result** — including the eight that were rejected and why | before proposing any compression idea; most obvious ones have been tried |
| `docs/HANDOFF.md` | this file: state, environment, workflow, what to do next | first |
| `results/*.txt` | raw benchmark output, with the baseline library versions recorded | you doubt a number in the README |
| `git log` | each commit message states what was measured and what was rejected | you want the reasoning behind a specific change |

`docs/research.md` is the single most valuable file here. It contains primary-
source algorithm details from libjxl, FLIF, CharLS, LPAQ and ZPAQ, and a table
of what each idea was actually worth when implemented. **Eight techniques have
been built and rejected on measurement.** Re-proposing one without reading that
table wastes a session.

## Current standing

Held-out split, 18 Kodak images never used for tuning
(`.venv/bin/python tools/bench_image.py --jobs=12 test`):

| codec | bytes | cpu s |
|---|---:|---:|
| JPEG XL effort 9 | 7,207,847 | 61.5 |
| **hve** | **7,796,932** | **19.0** |
| WebP lossless | 8,099,860 | 171.2 |
| PNG optimised | 11,321,001 | 6.0 |

31.1% under PNG, 3.7% under lossless WebP, **8.2% over JPEG XL**, and faster
than both of them. Video beats VP9 and AV1 on akiyo, loses to x264/x265/AV1 on
foreman; see the README tables.

Baseline sizes move when their libraries move — an earlier machine with libjxl
0.11 put the gap at 6.9% with byte-identical hve output. Every benchmark prints
its library versions for this reason. Do not compare a number across machines
without checking them.

## Environment

Python here is PEP 668-managed, so everything runs from the project venv:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install numba imagecodecs pytest
.venv/bin/python tools/fetch_testdata.py     # 24 Kodak images + 2 Xiph clips, ~100MB
```

- `testdata/` and `.venv/` are gitignored. Benchmarks need both.
- Run everything as `.venv/bin/python`. Bare `python3` lacks numba and
  imagecodecs, and the codec will silently fall back to the slow path.
- Git pushes need `gh auth setup-git` once.
- Branch `claude/extreme-compression-experiment-ar5swo`, already merged to
  `main` once via PR #1. Never force-push.

## The one thing that will bite you

There are **two implementations of the codec's core loop**:

- `hve/model.py` — the readable reference, and the definition of the format.
- `hve/fast.py` — the same loop compiled by numba, ~30x faster, used
  automatically whenever numba imports.

Any model change must be made in **both**, and they must produce
**byte-identical** output. Two tests enforce this
(`test_fast_path_is_byte_identical`, `test_video_fast_path_is_byte_identical`).
They have already caught two real divergences that would otherwise have made
the two paths write mutually unreadable files, and in one case the buggy path
compressed *better*, which is exactly how such a bug hides.

Practical workflow that works well:

1. Make the change in both files.
2. `.venv/bin/python tools/quick.py dev` — dev-set size in ~2s.
3. If it wins, `.venv/bin/python -m pytest tests -q` to confirm the paths agree.
4. If the tests fail, the two implementations have diverged; the measurement is
   meaningless until they agree. Do not record a number from a diverged state.

Gotchas that have already caused wrong measurements:

- numba lets a variable carry over between loop iterations if it is assigned on
  only one path. The reference resets it. Initialise new per-pixel variables at
  the top of the pixel loop in `fast.py`.
- `fast.py` receives tunables through a flat `params` array. **Append** new
  entries; inserting one shifts every index after it and silently scrambles the
  model.
- Context ordering matters: a context computed before the value it depends on
  will read the previous pixel's value. Check where in the loop your input is
  actually assigned.

## What to do next

The remaining gap is **not** in the entropy coder or the binarisation, and this
is measured rather than assumed. Counters in the encoder (`Bank.stats`) show
that 74% of residuals are nonzero and that raw unmodelled bypass bits are only
3.3% of the file. So:

- A better binarisation (hybrid-uint tokens) has a **3.3% ceiling**. Not worth
  the rebuild.
- The zero flag and sign bit are together about half the file, and for an
  unbiased predictor the sign half is incompressible. That half only shrinks if
  the residuals themselves shrink.

Which leaves one real option, and it is a rebuild rather than an increment:

**Replace the error-weighted averaging blend with a logistic mixer over many
more predictors.** Four separate attempts to add predictors to the current
average have come back negative — GAP, least squares, the match value, and the
2W-WW / 2N-NN trend pair that paq8px rates highly. They fail identically, so
the finding is about the combiner: an average is dragged by a volatile member
even when down-weighted, while a mixer learns to ignore it. The match model
only started paying when it stopped being a vote and became a hard switch.
paq8px runs ~130 predictors into a real mixing network; this runs six into an
average.

Expect this to be a substantial piece of work in both `model.py` and
`fast.py`, and expect the byte-exactness tests to be what keeps it honest.

Smaller things that are genuinely still open:

- Re-sweep the tuned constants. Several were fitted on 1-3 images back when a
  full-corpus A/B cost 13 minutes; it now costs ~35s, so `tools/tune.py` can
  run properly on the whole dev set.
- Video encode is bounded by motion search (full search over ±8 in numpy), not
  by coding. Attack the search if video encode speed matters.
- Video's inter design is deliberately simple: one reference frame, full-pel
  vectors, one block size, no bidirectional prediction. 8x8 blocks were
  measured and were slightly worse.

## Honest framing

Every cheap contextual trick has now returned between 0.1% and 0.4%. Six of
them stacked took the gap from 8.97% to 8.17%. Grinding out more of them will
not close 8%, and saying otherwise would misrepresent the trend. Either commit
to the mixer rebuild or bank the current position, which is a codec that is
31% smaller than PNG, 3.7% smaller than lossless WebP, and faster than both of
them and than JPEG XL.
