# Start here

Written for someone — or some model — picking this up cold. Read this first,
then whichever of the three documents below matches what you are about to do.
Update this file when the state it describes stops being true; a stale handoff
is worse than none.

**Last verified: half-pel motion vectors, all 39 tests green, benchmarks in
`results/` regenerated against it.**

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
| `docs/research.md` | **every technique tried, with its measured result** — including the fifteen that were rejected and why | before proposing any compression idea; most obvious ones have been tried |
| `docs/HANDOFF.md` | this file: state, environment, workflow, what to do next | first |
| `results/*.txt` | raw benchmark output, with the baseline library versions recorded | you doubt a number in the README |
| `git log` | each commit message states what was measured and what was rejected | you want the reasoning behind a specific change |

`docs/research.md` is the single most valuable file here. It contains primary-
source algorithm details from libjxl, FLIF, CharLS, LPAQ and ZPAQ, and a table
of what each idea was actually worth when implemented. **Fifteen techniques have
been built or costed and then rejected on measurement.** Re-proposing one without reading that
table wastes a session.

## Current standing

Held-out split, 18 Kodak images never used for tuning
(`.venv/bin/python tools/bench_image.py --jobs=12 test`):

| codec | bytes | cpu s |
|---|---:|---:|
| JPEG XL effort 9 | 7,207,847 | 63.9 |
| **hve** | **7,711,460** | **23.6** |
| WebP lossless | 8,099,860 | 175.9 |
| PNG optimised | 11,321,001 | 6.0 |

31.9% under PNG, 4.8% under lossless WebP, **7.0% over JPEG XL**, and faster
than both of them. Video now leads x264, x265, AV1 and VP9 on **both** test
clips — akiyo 317,074 and foreman 837,709, the latter 1.0% ahead of x264. See
the README tables.

Treat `cpu s` as approximate — a repeat run moved JPEG XL e9 from 63.9s to 77.9s
on identical code. Byte counts are exact.

Baseline sizes move when their libraries move — an earlier machine with libjxl
0.11 put the gap at 6.9% with byte-identical hve output. Every benchmark prints
its library versions for this reason. Do not compare a number across machines
without checking them.

## Environment

Python here is PEP 668-managed, so everything runs from the project venv:

```bash
./playground/setup.sh                        # venv + every dependency, explicitly
.venv/bin/python tools/fetch_testdata.py     # 24 Kodak images + 5 Xiph clips, ~200MB
```

Do **not** create the venv with `--system-site-packages`, which is what this
file used to say. It worked only because the original machine had numpy and
Pillow installed system-wide, and it fails on a clean clone.

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
- Integer division must round identically in both paths. The learned combiner
  divides by input energy, and rather than trust Python and numba to agree on
  how a negative quotient floors, it takes the absolute value, divides, and
  reapplies the sign. Do the same for any new division; `//` on a negative
  numerator is the kind of thing that diverges silently on one platform.
- The combiner reads a second row of history (`prev2`) that nothing else uses.
  If you add state with a lifetime longer than one row, rotate it in both files
  — `model.py` rebinds lists, `fast.py` copies arrays element by element.
- Motion vectors are in **half-pel** units. A vector splits into `d >> 1` whole
  pixels and a phase of `d & 1` indexing four interpolated reference planes,
  and the shift must stay arithmetic so negative vectors floor to the correct
  side. Chroma divides the luma vector by its subsampling factor *first*.
  Anything that reads `mvs` and forgets the units is off by a factor of two.

## What was just done

Two rounds, both of which closed items this file used to name as open.

**The learned combiner** replaced the error-weighted averaging blend: **1.10%
held out** on stills, closing the JPEG XL gap from 8.17% to 6.99%. See "The
learned combiner" in `docs/research.md`.

**Half-pel motion vectors** for video: **2.3% held out**, and 3.05% on foreman
alone, which took that clip from fourth place to first. Together with the
combiner, video now leads x264, x265, AV1 and VP9 on both test clips. Three
other motion ideas were costed and rejected in the same round — see "Motion
modelling" in `docs/research.md`, and note that two of them were killed by a
numpy proxy in about a minute each, without being built.

Two results from the combiner are worth carrying forward, because both
contradict what this document previously assumed:

1. **The combiner is linear, so redundant inputs are worthless.** The first
   build used seven inputs that were all linear combinations of the same four
   neighbours and returned -0.013%. Swapping them for a real basis (the second
   ring) plus two *nonlinear* predictors (MED, GAP) took the identical machinery
   to -1.18%. Before adding a predictor, ask whether it is already inside the
   span of the ones there.
2. **The match model still wants a switch, not a weight.** Feeding the match
   value into the combiner measured 0.15% worse, and 0.13% worse even with one
   learned weight per match-length state. The old "an average is the problem"
   framing was too broad: learned weights beat an average for predictors that
   are *continuously somewhat-right*, and neither combiner suits a bimodal one.

## What to do next

No single large item is identified any more. The honest options, roughly in
order of expected value:

- **More nonlinear inputs to the combiner.** This is a falsifiable prediction
  rather than a hunch: the linear span is now well covered, so further gains
  should come only from predictors that are *not* weighted sums of neighbours
  (a second GAP at a different threshold, a median-of-three, a texture-matched
  value). If adding more linear neighbours does pay, the span argument above is
  wrong and the whole section needs revisiting.
- **Two combiners at different adaptation rates**, as paq8px runs six LMS
  filters rather than one. The step-size sweep here has a single flat optimum,
  which is what you would see if one rate is serving both smooth and busy
  regions. Combining the two should be a mixer or a switch — on the evidence in
  `research.md`, not an average.
- **Re-sweep the tuned constants.** Several predate the combiner entirely, and
  several were fitted on 1-3 images back when a full-corpus A/B cost 13 minutes;
  it now costs ~3s, so `tools/tune.py` can run properly on the whole dev set.
  The combiner's own constants were swept, but only one axis at a time.
- **Multiple reference frames or bidirectional prediction** for video, neither
  of which has been costed. Half-pel vectors are done (-2.3% held out) and
  variable block sizes were measured and rejected, so the two items this section
  used to name are closed. Video encode is still bounded by motion search.

## Honest framing

Before this change, every cheap contextual trick had returned 0.1-0.4% and six
of them stacked moved the gap from 8.97% to 8.17%; the note here was that
grinding out more would not close 8%. That was right, and the fix was
architectural rather than incremental. The same caution now applies one level
up: 7.0% will not close by adding a fourteenth predictor either. The current
position is a codec 31.9% smaller than PNG, 4.8% smaller than lossless WebP,
faster than both and than JPEG XL, and first on low-motion lossless video — and
that is a perfectly good place to stop if the next idea is not a structural one.
