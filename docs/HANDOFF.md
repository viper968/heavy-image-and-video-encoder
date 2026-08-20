# Start here

Written for someone — or some model — picking this up cold. Read this first,
then whichever of the three documents below matches what you are about to do.
Update this file when the state it describes stops being true; a stale handoff
is worse than none.

**Last verified: the standalone C binary, all 96 tests green. Byte counts
unchanged, so `results/` was not regenerated — nothing in it moved.**

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
| JPEG XL effort 9 | 7,207,847 | 65.1 |
| **hve** | **7,702,223** | **17.5** |
| WebP lossless | 8,099,860 | 184.7 |
| PNG optimised | 11,321,001 | 6.2 |

31.9% under PNG, 4.8% under lossless WebP, **6.9% over JPEG XL**, and faster
than both of them. Video leads x264, x265, AV1 and VP9 on **both** test clips —
akiyo 315,981 and foreman 834,251, the latter 1.3% ahead of x264 — and on 16
frames of 1080p Sintel it is 10.5% under x264 at 30,944 bytes. See the README
tables.

Video encode at 1080p is **2.7s against x264's ~0.3s**, roughly 10x. It was
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
  falls back to the pure-Python reference: still correct, roughly 100x slower,
  about a minute per Kodak photo. Treat that as a guarantee, not a usable mode.
- `make -C csrc` builds the **standalone binary** at `build/hve` — no Python,
  no numpy, no libpng. `make -C csrc static` for a 1.1 MB shippable one,
  `windows` to cross-compile (written, never run). It is the thing to hand
  someone; the Python package is the thing to experiment in.
- Run everything as `.venv/bin/python`. Bare `python3` lacks imagecodecs and may
  not find the built library, and the codec will silently use the slow path.
- Git pushes need `gh auth setup-git` once.
- Branch `claude/extreme-compression-experiment-ar5swo`, already merged to
  `main` once via PR #1. Never force-push.

## The one thing that will bite you

There are **two implementations of the codec's core loop**:

- `hve/model.py` — the readable reference, and the definition of the format.
- `csrc/kernel.c` via `hve/native.py` — the same loop in C, ~100x faster, and
  what actually runs. Built automatically on first import and cached at
  `hve/_hve*.so`; `HVE_NO_NATIVE=1` forces the reference instead.

Any model change must be made in **both**, and they must produce
**byte-identical** output. `tests/test_native.py` and the two
`*_is_byte_identical` tests in `tests/test_codecs.py` enforce it. They have
caught three real divergences that would otherwise have made the paths write
mutually unreadable files, and in one case the buggy path compressed *better*,
which is exactly how such a bug hides.

There was a third, `hve/fast.py`, in numba. It was deleted after the C landed —
all five model changes in this repo's history had to be mirrored into it by
hand, and it had stopped adding coverage. Do not add it back without reading
"The cost of a third implementation" in `docs/research.md`.

**Everything *outside* the pixel loop also exists twice**, since the standalone
binary has its own containers, colour transform and y4m. That is a much cheaper
bargain — a magic string, four varints and a colour transform, all rarely
touched and pinned byte-for-byte by `tests/test_cli_binary.py`. The part that
would have been dangerous, the model constants, is **generated**: run
`.venv/bin/python tools/gen_model_constants.py` after changing any constant, or
`tests/test_generated_constants.py` will tell you. Never hand-edit
`csrc/model_constants.h`.

Practical workflow that works well:

1. Make the change in both files.
2. `.venv/bin/python tools/quick.py dev` — dev-set size in ~2s.
3. If it wins, `.venv/bin/python -m pytest tests -q` to confirm the paths agree.
4. If the tests fail, the implementations have diverged; the measurement is
   meaningless until they agree. Do not record a number from a diverged state.

You can explore in `model.py` alone with `HVE_NO_NATIVE=1` and port to C once
the idea has earned its place — but only on small inputs, because the reference
is ~150s per megapixel. What you must **not** do is leave the C stale and
enabled: it is the path that runs by default, so a stale C kernel means every
number you measure comes from the old model.

**Beware of tests that disable one backend to reach the reference.** Adding the
C path silently broke `test_video_fast_path_is_byte_identical`, which switched
off only `fast.available` and so began comparing native against native. It
passed while testing nothing. Any such test must disable *every* accelerated
backend.

Because byte-identity now costs reference-speed, those tests deliberately run on
tiny inputs — a 40x56 photo crop, 32x48 frames. Real photographs and real clips
are covered by roundtrip and invariant tests instead. Do not "improve" the
identity cases by enlarging them; the suite runs in under four seconds and that
is why it gets run.

Gotchas that have already caused wrong measurements:

- Initialise new per-pixel variables at the top of the pixel loop. A value
  assigned on only one branch stays alive into the next pixel in C, and did in
  numba too; the reference resets it, so the two diverge.
- The C kernel receives tunables through the flat array built by
  `model.coder_params()`. **Append** new entries; inserting one shifts every
  index after it and silently scrambles the model. `csrc/hve.h` has the enum.
- Context ordering matters: a context computed before the value it depends on
  will read the previous pixel's value. Check where in the loop your input is
  actually assigned.
- Integer division must round identically in both paths. The learned combiner
  divides by input energy, and rather than trust the two languages to agree on
  how a negative quotient floors, it takes the absolute value, divides, and
  reapplies the sign. Do the same for any new division.
- The combiner reads a second row of history (`prev2`) that nothing else uses.
  If you add state with a lifetime longer than one row, rotate it in both files
  — `model.py` rebinds lists, `kernel.c` memcpys arrays.
- Motion vectors are in **half-pel** units. A vector splits into `d >> 1` whole
  pixels and a phase of `d & 1` indexing four interpolated reference planes,
  and the shift must stay arithmetic so negative vectors floor to the correct
  side. Chroma divides the luma vector by its subsampling factor *first*.
  Anything that reads `mvs` and forgets the units is off by a factor of two.
- **Ceiling division.** Python's `-(-a // b)` transcribes into C as
  `-(-a / b)`, which looks right and is wrong, because `/` truncates toward
  zero where `//` floors. Use `hve_ceil_div`. This is not hypothetical: it gave
  a 1080p frame 67 block rows instead of 68 and corrupted the heap, and it hid
  for a whole session because every clip in `testdata/` divides exactly by 16.
- **C's integer semantics are not Python's**, and both differences are silent.
  `/` truncates toward zero where `//` floors, so every division whose numerator
  can go negative uses `hve_fdiv()`; `>>` on a negative value is
  implementation-defined before C23, so `csrc/hve.h` static-asserts that it is
  arithmetic; and signed overflow is undefined where Python's integers are
  arbitrary precision, so the build passes `-fwrapv`. Adding a division or a shift to the C kernel without
  checking which case it is will produce a stream the other two paths cannot
  read, on some machines only.
- The C kernel narrows several scratch arrays (`flat` is uint8, `match_table`
  and `lmsw` are int32, `errmap` is uint8). If
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

**Then `hve/fast.py` was deleted.** Adding the C made three implementations of
one loop, which was one too many: every model-changing commit in this repo's
history — five of five — had to be mirrored into the numba file by hand, and
once the C was pinned directly against the reference the numba path added
redundancy rather than coverage. It cost 211 MB of dependency (llvmlite alone is
180 MB) to serve only the "numba installed, no C compiler" case. Output did not
change by a byte; the test suite got faster. Same section of `research.md`.

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

**Then the whole program was ported to C** (`csrc/container.c`, `transform.c`,
`y4m.c`, `imageio.c`, `main.c`), producing a standalone `build/hve` with no
Python in it. Byte-identical output on every test, and each side reads the
other's files. Honest accounting: the speed gain is a flat ~0.15s of removed
interpreter startup, which is 1.7x on a small still and **5% at 1080p** — the
Python path was already calling the same C kernel. The value is distribution:
1.1 MB static, no interpreter, no numpy, no libpng, no zlib.

That round found a heap corruption (ceiling division, see above) and one
representation bug: lodepng's encoder auto-picks the smallest colour type, so
a uniform RGB image came back as a 1-bit palette PNG — lossless in pixels, but
not what was handed in. `auto_convert` is off now.

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

**The three-step plan below is done.** Presets travel in the header, slices give
16-way parallelism, and the constants have been re-swept per preset; the codec
now beats x264 lossless on both size and speed at 1080p. What follows are the
remaining open items. See "Slices", "Buying speed with ratio" and "Re-tuning
after the presets" in `docs/research.md`.

Historical context for the plan that produced this: see "Buying speed with
ratio" in `docs/research.md`. In short: every model stage now has a switch
(`params[P_FEATURES]`, `hve --features N`), and pricing them showed that on
1080p video the match model, the learned combiner and the weighted blend do not
earn their cost — dropping all three is **1.55x to 1.96x faster** across the
Sintel trailer, for somewhere between -32% and +7.9% on size depending entirely
on the content.

> The plan below was originally written around a much louder claim — "18.8%
> smaller and 1.7x faster at once" — measured on 16 frames starting at frame 0
> of the trailer, of which the first 8 are **black**. That number is wrong; on
> the trailer's busy segments the three stages cost 1.4-7.9% rather than saving
> 18.8%. The speed-up is real and content-independent, which is why the preset
> still exists. Corrected tables are in "What the full trailer says" in
> `docs/research.md`.

The work that follows from that, in order:

1. **Put the feature bitmask in the container header** and have the encoder
   choose it. It cannot be a global constant: the preset that wins on sparse
   1080p video *costs* 3.74% on a photograph and up to 7.9% on busy video. This
   is a format change and it is the prerequisite for everything else.
2. **Wavefront parallelism rather than independent slices.** Our slicing penalty
   is entirely a model-relearning penalty, and HEVC's WPP avoids exactly that by
   copying the entropy state from after the second block of the row above
   instead of resetting it. Published scaling is 8.7x on 12 cores against 9.3x
   for fully independent tiles — nearly the same parallelism, far less loss.
3. **Then re-tune.** The constants were all fitted with the full model switched
   on; several of them are probably wrong for a preset that drops three stages.
4. `hve/model.py` implements only `FEAT_ALL`. Either teach it the presets or
   accept that the C kernel is the definition for everything else, and say so.

The older options, still open, roughly in order of expected value:

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
- **Rate-aware motion search** — now the best-evidenced item on this list.
  Widening the search radius from 8 to 64 makes 1080p files *monotonically
  bigger* (3,646,266 -> 3,702,994), because the search scores residuals and
  cannot see that magnitudes are coded unary. Any future work on motion has to
  fix this first, or it will measure backwards. Original note follows.
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
- **A y4m frame rate does not survive a round trip.** `hve_y4m_read` parses the
  `F` tag into a local buffer in `csrc/main.c` and then never passes it to the
  encoder, and the container has no field for it, so every clip decodes as
  `F25:1` regardless of what went in. `hve/y4m.py` has the same hole. Pixels are
  lossless — only the header is lost — but the comment at the top of
  `csrc/y4m.c` claims the rate *is* carried through, so the code and the comment
  disagree. Fixing it means a field in the video header, i.e. another format
  bump; worth folding into the next one rather than spending a bump on its own.
- ~~**Precomputing contexts on the encoder side**~~ — **done, and it returned a
  third of what was predicted.** `derive_row` in `csrc/kernel.c` derives a whole
  row of contexts before the serial pass, on the `fast` preset only (blend, LMS
  and match adapt in scan order and cannot be hoisted). Byte-identical; **-8.5%
  wall, -15.3% instructions** on 1080p, ~0 on stills and on `max`. The estimate
  said 25-35%. Two thirds of the shortfall is that the context-index half of the
  work is all ladder lookups, and a gather is the only vector form of a lookup —
  forcing one with `-mtune-ctrl=use_gather` removes 10% of the instructions and
  changes wall time by 0.2%, so there is nothing there. Read "The structural one"
  in `docs/research.md` before trying to push it further; a comparison-cascade
  replacement for the lookups was also built and was *worse*.

  If you touch the derivation, `--batched 0` turns it off, and
  `test_batched_derivation_matches_the_scalar_path` asserts the two produce the
  same bytes. That test is the only thing pinning them: the Python byte-identity
  tests cannot reach this path, because `model.py` implements only the full
  model and the full model switches batching off.
- ~~**AV1's multi-symbol entropy coder**~~ — **measured and ruled out.** A build
  with the range coder stubbed out entirely — the upper bound on any coder
  change — is only 7.0% faster on the `fast` preset and *within noise* on `max`.
  The coder is 4.6% of the serial loop; the expensive decision, the zero flag's
  five-expert mix, is already once per pixel and multi-symbol coding cannot make
  it less. See "AV1's multi-symbol coder" in `docs/research.md` before
  reconsidering.
- **Video encode speed beyond the C port** now means the pixel loop and nothing
  else — search is 0.12s of a 2.7s encode. Implementation work is close to done:
  tabulating the ladder lookups was worth 19-23% on real content and a
  re-profile afterwards is flat. Everything beyond that needs slices, and the
  price is now measured rather than guessed: **+0.52% for 4 slices on stills but
  +4.65% at 1080p and +14.62% on akiyo**, because the penalty tracks how much of
  the file is learned model state. See "What is left in the serial loop" in
  `docs/research.md` before proposing it — the old "0.3-1%" figure in this file
  was never measured and was wrong.

## The serial pass: what is left after this round

Profiled properly for the first time, and **on the decoder** - profiling the
encoder hides the work the decoder cannot avoid, which is how the confidence
hoist stayed invisible. Full tables in "Attacking the serial pass" in
`docs/research.md`.

It is **compute-bound**: IPC 3.81, LLC misses 0.012 per sample. Prefetching and
cache-motivated narrowing are both ruled out; the only lever is fewer
instructions. This round took 679 per sample down to about 480, and 1080p
single-threaded decode from 7.8 to 11.3 fps.

What remains, as a share of decoder instructions:

- **scalar context derivation, ~40%** — the decoder cannot use `derive_row`
  because it needs the just-decoded west pixel. Only part of it truly depends
  on west (task: precompute the previous-row half). The profile says that part
  is worth **~5%**, not the 15-20% a first reading of the bucket suggests, so
  price it before building it.
- **3 experts + mixer, 19%** — all three earn their place, measured.
- **range coder, 14%** — byte-at-a-time renormalisation. A wider renormalisation
  is a core-format change for maybe 5%; not attempted.
- everything else is under 8% each.

Do not re-propose: prefetching, multi-symbol coding, dropping the exponent
chain's mixer, or removing the ladder bounds checks. All four are measured and
written up with the numbers.

## Is this a real format? Read this before pitching it

Measured, not guessed; the tables are in "Could this ship as a real format" in
`docs/research.md`. Short version:

- **Against FFV1**, the lossless video standard: **57.7% smaller on a static
  camera**, 14.4% smaller on a panning CIF clip, and **0.3% smaller and 3.1x
  slower** on moving 1080p. The whole advantage is temporal redundancy.
- **Motion compensation contributes 0.70% at 1080p** and 59% on akiyo. Only
  0.9% of 1080p blocks pick inter. This was checked three ways and is not a
  tuning bug — at high resolution the spatial model already predicts nearly
  everything.
- **Against the web**, which is lossy: AV1 at crf32 is **107x smaller and 22x
  faster to decode** on the same clip. That gap is not closable.
- **On stills** we beat PNG by 32% and WebP lossless by 4.9%, and lose to
  JPEG XL by 6.86% while decoding at half its speed. JPEG XL is smaller *and*
  faster, is ISO/IEC 18181, and Chrome removed it anyway.
- Decoder peak RSS is **63.6 MB** at 1080p (4.5 MB of that is model state).
- 8-bit only; no 10/12-bit, no HDR, no wide gamut, no progressive decode.
- There is **no bitstream spec** — the C is the spec.
- Fuzzing found a reachable heap over-read on the first attempt (fixed,
  `057b48c`). Before this there had been none, ever.

The defensible niche is lossless capture and archival of static-camera content,
where halving FFV1 is a genuine result. Web delivery is not it.

## Honest framing

Before this change, every cheap contextual trick had returned 0.1-0.4% and six
of them stacked moved the gap from 8.97% to 8.17%; the note here was that
grinding out more would not close 8%. That was right, and the fix was
architectural rather than incremental. The same caution now applies one level
up: 7.0% will not close by adding a fourteenth predictor either. The current
position is a codec 31.9% smaller than PNG, 4.8% smaller than lossless WebP,
faster than both and than JPEG XL, and first on low-motion lossless video — and
that is a perfectly good place to stop if the next idea is not a structural one.
