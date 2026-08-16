# Start here

Written for someone — or some model — picking this up cold. Read this first,
then whichever of the three documents below matches what you are about to do.
Update this file when the state it describes stops being true; a stale handoff
is worse than none.

**Last verified: the C backend, all 73 tests green, benchmarks in `results/`
regenerated against it.**

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
| JPEG XL effort 9 | 7,207,847 | 60.0 |
| **hve** | **7,711,460** | **16.4** |
| WebP lossless | 8,099,860 | 169.0 |
| PNG optimised | 11,321,001 | 6.1 |

31.9% under PNG, 4.8% under lossless WebP, **7.0% over JPEG XL**, and faster
than both of them. Video leads x264, x265, AV1 and VP9 on **both** test clips —
akiyo 316,313 and foreman 834,903, the latter 1.3% ahead of x264 — and on 16
frames of 1080p Sintel it is 10.5% under x264 at 30,944 bytes. See the README
tables.

Video encode at 1080p is **2.7s warm against x264's ~0.3s**, roughly 10x. It was
145s, then 9.8s warm after the search rewrite (the 19.8s once published for that
included numba's cold compile), then 2.7s with the C backend. The motion search
is now threaded and effectively free at 0.12s; the remaining time is the pixel
loop, which is serial by construction. See "The C port" in `docs/research.md`.

Treat `cpu s` as approximate — a repeat run moved JPEG XL e9 from 63.9s to 77.9s
on identical code, and this machine's noise floor on a 3-second benchmark is
about 10%. Byte counts are exact.

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

- `testdata/`, `.venv/` and the built `hve/_hve*.so` are gitignored. Benchmarks
  need the first two; the third rebuilds itself.
- A **C compiler** (`cc`, or `$CC`) makes the native backend available. It is
  built on first import and rebuilt automatically whenever anything in `csrc/`
  is newer, so there is no build step to remember. With no compiler the codec
  falls back to numba and everything still works.
- Run everything as `.venv/bin/python`. Bare `python3` lacks numba and
  imagecodecs, and the codec will silently fall back to the slow path.
- Git pushes need `gh auth setup-git` once.
- Branch `claude/extreme-compression-experiment-ar5swo`, already merged to
  `main` once via PR #1. Never force-push.

## The one thing that will bite you

There are **three implementations of the codec's core loop**:

- `hve/model.py` — the readable reference, and the definition of the format.
- `hve/fast.py` — the same loop compiled by numba, ~30x faster than the
  reference, used when numba imports and the C library is unavailable.
- `csrc/kernel.c` via `hve/native.py` — the same loop again in C, another
  1.5-2x on the pixel loop, preferred whenever it builds. Built automatically
  on first import and cached at `hve/_hve*.so`; `HVE_NO_NATIVE=1` disables it.

Any model change must be made in **all three**, and they must produce
**byte-identical** output. `tests/test_native.py` and the two
`*_is_byte_identical` tests in `tests/test_codecs.py` enforce it. They have
caught three real divergences that would otherwise have made the paths write
mutually unreadable files, and in one case the buggy path compressed *better*,
which is exactly how such a bug hides.

Practical workflow that works well:

1. Make the change in all three files.
2. `.venv/bin/python tools/quick.py dev` — dev-set size in ~2s.
3. If it wins, `.venv/bin/python -m pytest tests -q` to confirm the paths agree.
4. If the tests fail, the implementations have diverged; the measurement is
   meaningless until they agree. Do not record a number from a diverged state.

If three is too many to maintain for a change you are exploring, set
`HVE_NO_NATIVE=1` and work in `model.py` + `fast.py` first, then port to C once
the idea has earned its place. What you must **not** do is leave the C stale and
enabled — it is the path that runs by default, so a stale C kernel means every
number you measure comes from the old model.

**Beware of tests that disable one backend to reach the reference.** Adding the
C path silently broke `test_video_fast_path_is_byte_identical`, which switched
off only `fast.available` and so began comparing native against native. It
passed while testing nothing. Any such test must disable *every* accelerated
backend.

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
- **C's integer semantics are not Python's**, and both differences are silent.
  `/` truncates toward zero where `//` floors, so every division whose numerator
  can go negative uses `hve_fdiv()`; `>>` on a negative value is
  implementation-defined before C23, so `csrc/hve.h` static-asserts that it is
  arithmetic; and signed overflow is undefined where numba's int64 wraps, so the
  build passes `-fwrapv`. Adding a division or a shift to the C kernel without
  checking which case it is will produce a stream the other two paths cannot
  read, on some machines only.
- The C kernel narrows several scratch arrays that `fast.py` keeps as int64
  (`flat` is uint8, `match_table` and `lmsw` are int32, `errmap` is uint8). If
  you widen the range of anything stored there — say a combiner weight clamp
  above 2^31 — the C path will wrap where the others do not.

## What was just done

**The C backend** (`csrc/`, `hve/native.py`). Byte-identical output, 3.6x faster
1080p encode and 1.7x faster decode against a *warm* numba baseline; the motion
search alone is 37x faster because it threads across cores and needs no format
change to do so. Stills encode 1.7x faster. Full table in "The C port" in
`docs/research.md`, including the four reasons it beat numba (none of which is
"C is faster") and the one narrowing that was tried and reverted for being
inside the noise.

Two things fell out of writing it that matter more than the speed:

1. **A search bug the tests could not see.** `_full_search` seeded its running
   best from candidate `(-radius, -radius)` while leaving the vector at zero, so
   about one block per CIF frame was compensated from the wrong place. Found
   only because the C search disagreed on 1 block in 396 and both had to be
   explained. Fixed; worth **+28 bytes across six clips**, i.e. nothing.
2. **Why it was worth nothing is the lead.** The correct vector at the search
   boundary is expensive to *send* — magnitudes are unary — and the search's
   cost proxy scores residuals only. Rate-aware search is now the most concrete
   untried motion idea. See below.

Before that, two rounds that closed items this file used to name as open.

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
- **Diagnose the match model's ~7% loss on 1080p Sintel.** It is the largest
  known single loss in the codec: disabling the model entirely made that clip
  28,725 bytes against a baseline that is now 30,944, while the same model saves
  1.25% on the CIF pair. Two hypotheses were tested and both were wrong (see
  `docs/research.md`). Re-measure both sides first — the "model off" figure
  predates the `_full_search` fix — then diagnose before changing anything.
- **Rate-aware motion search.** The search scores residuals only; the cost of
  *sending* a vector enters once as a flat `mv_penalty=48` in `choose_modes` and
  never in the choice between two candidates. Magnitudes are coded unary against
  a median predictor, so a candidate one pixel further out can cost several
  extra bits that the search cannot see. Every real codec charges this. It is
  untried here, it is cheap to try (the differential and its unary length are
  both already computable in `motion_search`), and the `_full_search` fix above
  is direct evidence that the proxy is currently wrong in this specific way.
- **Multiple reference frames or bidirectional prediction** for video, neither
  of which has been costed. Half-pel vectors are done and variable block sizes
  were measured and rejected.
- **Video encode speed beyond the C port** now means the pixel loop and nothing
  else — search is 0.12s of a 2.7s encode. Threading it needs the
  slice-independent format change, at a measured cost of ~0.3-1% ratio. There is
  no further single-threaded trick of the size already taken.

## Honest framing

Before this change, every cheap contextual trick had returned 0.1-0.4% and six
of them stacked moved the gap from 8.97% to 8.17%; the note here was that
grinding out more would not close 8%. That was right, and the fix was
architectural rather than incremental. The same caution now applies one level
up: 7.0% will not close by adding a fourteenth predictor either. The current
position is a codec 31.9% smaller than PNG, 4.8% smaller than lossless WebP,
faster than both and than JPEG XL, and first on low-motion lossless video — and
that is a perfectly good place to stop if the next idea is not a structural one.
