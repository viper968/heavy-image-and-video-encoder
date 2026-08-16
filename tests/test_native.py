"""The native C backend must agree with the reference implementation exactly.

There are two implementations of the codec's core loop — `hve/model.py`, which
is the definition of the format, and `csrc/kernel.c`, which is what actually
runs — and a one-bit divergence between them silently corrupts every file
written by whichever happened to run. These tests are the only thing standing
between that and shipping, so they compare bytes rather than sizes.

The awkward part is that the reference costs about 150 seconds per megapixel,
so byte-identity is checked on inputs small enough to finish while still
covering the branches that matter: odd dimensions, single rows and columns,
the fourth plane kind, and subsampled chroma. Real photographs and real clips
are then covered by roundtrip and invariant tests, which only need the fast
path. Do not "improve" the byte-identity cases by enlarging them — the suite
runs in seconds today and that is why it gets run.

Run with: python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hve import image, model, native, video, y4m          # noqa: E402
from tests.test_codecs import _reference_payload, synthetic   # noqa: E402

CLIPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "testdata", "video")
PHOTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                     "photo_crop.png")

pytestmark = pytest.mark.skipif(not native.available(),
                                reason="native backend unavailable: %s"
                                       % native.load_error())


def _clip(name, limit=6):
    path = os.path.join(CLIPS, name)
    if not os.path.exists(path):
        pytest.skip("run tools/fetch_testdata.py for %s" % name)
    reader = y4m.Y4M(path)
    frames = [[p.copy() for p in f] for f in reader.frames(limit=limit)]
    reader.close()
    return frames


def _encode_both(frames):
    """The same clip through the C kernel and through the pure-Python reference.

    The reference is ~150s/megapixel, so callers must keep the frames tiny.
    """
    real = native.available
    try:
        native.available = lambda: False
        reference = video.encode(frames)
    finally:
        native.available = real
    return video.encode(frames), reference


# --------------------------------------------------------------------------
# stills


@pytest.mark.parametrize("shape", [(1, 1), (1, 40), (40, 1), (7, 9), (33, 51),
                                   (17, 3)])
def test_still_matches_the_reference_on_shapes(shape):
    """Odd sizes are where an off-by-one in the row buffers shows up."""
    h, w = shape
    planes = np.ascontiguousarray(np.random.default_rng(h * 1000 + w).integers(
        0, 256, (3, h, w), dtype=np.uint8))
    assert native.encode_planes(planes) == _reference_payload(planes)


def test_still_matches_the_reference_on_a_photograph():
    """Real photographic data, cropped to what the reference can finish."""
    from PIL import Image as PILImage
    img = np.array(PILImage.open(PHOTO).convert("RGB"))[:40, :56]
    planes, _ = image._planes_from_image(img)
    planes = np.ascontiguousarray(planes, dtype=np.uint8)
    assert native.encode_planes(planes) == _reference_payload(planes)


def test_still_roundtrips_a_whole_photograph():
    """The fast path alone, so this can use the full image."""
    from PIL import Image as PILImage
    img = np.array(PILImage.open(PHOTO).convert("RGB"))
    assert np.array_equal(image.decode(image.encode(img)), img)


@pytest.mark.parametrize("kind", ["flat", "random", "gradient", "extremes"])
def test_still_roundtrips(kind):
    r = np.random.default_rng(3)
    if kind == "flat":
        planes = np.full((3, 32, 40), 200, np.uint8)
    elif kind == "random":
        planes = r.integers(0, 256, (3, 32, 40), dtype=np.uint8)
    elif kind == "gradient":
        planes = (np.add.outer(np.arange(32), np.arange(40)) % 256
                  ).astype(np.uint8)[None].repeat(3, 0)
    else:
        planes = np.zeros((3, 32, 40), np.uint8)
        planes[:, ::2] = 255
    planes = np.ascontiguousarray(planes)
    blob = native.encode_planes(planes)
    back = native.decode_planes(blob, *planes.shape)
    assert np.array_equal(back, planes)


def test_still_alpha_channel_uses_the_fourth_kind():
    """Four planes exercises kind 3, which three-channel content never reaches."""
    planes = np.ascontiguousarray(
        np.random.default_rng(11).integers(0, 256, (4, 20, 23), dtype=np.uint8))
    assert native.encode_planes(planes) == _reference_payload(planes)


# --------------------------------------------------------------------------
# video


def test_video_matches_the_reference():
    """Subsampled chroma and the inter branch, at a size the reference survives."""
    base = [synthetic(32, 48, 1, seed=0), synthetic(16, 24, 1, seed=1),
            synthetic(16, 24, 1, seed=2)]
    frames = [[np.roll(p, i * 3, axis=1) for p in base] for i in range(4)]
    got, want = _encode_both(frames)
    assert got == want


def test_video_matches_the_reference_on_a_real_clip_crop():
    """Real motion rather than a synthetic roll, cropped so the reference finishes.

    A crop of foreman keeps genuine sub-pixel head movement and real chroma,
    which the rolled synthetic frames above do not have — a rolled frame has an
    exact integer motion vector and never exercises the half-pel phases.
    """
    frames = _clip("foreman_cif.y4m", limit=3)
    small = [[f[0][:32, :48].copy(), f[1][:16, :24].copy(), f[2][:16, :24].copy()]
             for f in frames]
    got, want = _encode_both(small)
    assert got == want


@pytest.mark.parametrize("clip", ["foreman_cif.y4m", "bus_cif.y4m"])
def test_video_roundtrips_on_real_clips(clip):
    frames = _clip(clip, limit=5)
    out = video.decode(video.encode(frames))
    for want, got in zip(frames, out):
        for a, b in zip(want, got):
            assert np.array_equal(a, b)


def test_video_odd_dimensions_roundtrip():
    """Odd sizes make the chroma planes and the block grid disagree."""
    base = [synthetic(29, 37, 1, seed=5), synthetic(15, 19, 1, seed=6),
            synthetic(15, 19, 1, seed=7)]
    frames = [[np.roll(p, i, axis=0) for p in base] for i in range(3)]
    out = video.decode(video.encode(frames))
    for want, got in zip(frames, out):
        for a, b in zip(want, got):
            assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# motion estimation


def _search_args():
    return (video.BLOCK, video.SEARCH, video._COST_BYTE, video.HALF_PEL_BIAS,
            video.PYRAMID_MIN_PIXELS, video.PYRAMID_LEVELS,
            video.REFINE_RADIUS)


@pytest.mark.parametrize("clip", ["foreman_cif.y4m", "bus_cif.y4m",
                                  "mobile_cif.y4m"])
def test_native_search_picks_the_same_vectors(clip):
    """Block for block, not merely 'as good'.

    A search that found different but equally cheap vectors would leave every
    future compression measurement comparing two codecs instead of one, and the
    difference would be invisible in the totals.
    """
    frames = _clip(clip, limit=4)
    for i in range(1, len(frames)):
        cur, ref = frames[i][0], frames[i - 1][0]
        nmv, ncost = native.motion_search(cur, ref, *_search_args())
        pmv, pcost = video.motion_search(cur, ref, bs=video.BLOCK)
        assert np.array_equal(nmv, pmv)
        assert np.array_equal(ncost, pcost)


@pytest.mark.parametrize("clip", ["foreman_cif.y4m", "bus_cif.y4m"])
def test_native_spatial_cost_matches(clip):
    frames = _clip(clip, limit=3)
    for f in frames:
        got = native.spatial_cost(f[0], video.BLOCK, video._COST_BYTE)
        want = video.spatial_cost(f[0], bs=video.BLOCK)
        assert np.array_equal(got, want)


def test_native_halfpel_planes_match():
    r = np.random.default_rng(9)
    for shape in [(1, 1), (1, 16), (16, 1), (13, 21), (32, 32)]:
        plane = r.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(native.halfpel_planes(plane),
                              video.halfpel_planes(plane))


def test_full_search_reports_the_vector_it_scored():
    """Regression: the seed candidate is (-radius, -radius), not (0, 0).

    `_full_search` used to record the first position's cost while leaving the
    vector at its zero initialisation, so a block whose best match was exactly
    at the top-left corner of the search window was then predicted from the
    wrong place. Built here as a frame shifted by exactly that amount.
    """
    r = np.random.default_rng(21)
    ref = r.integers(0, 256, (64, 64), dtype=np.uint8)
    # np.roll(ref, +SEARCH)[y] is ref[y - SEARCH], and the search scores
    # ref[y + dy] against cur[y], so the one correct vector is (-SEARCH, -SEARCH)
    # — the very first candidate the loop visits, which is the whole point.
    cur = np.roll(np.roll(ref, video.SEARCH, axis=0), video.SEARCH, axis=1)
    mv, _ = video.motion_search(cur, ref, bs=video.BLOCK)
    interior = mv[1:-1, 1:-1]                       # edges wrap rather than move
    assert (interior[..., 0] == -2 * video.SEARCH).all()
    assert (interior[..., 1] == -2 * video.SEARCH).all()


@pytest.mark.parametrize("clip", ["foreman_cif.y4m", "bus_cif.y4m"])
def test_native_search_keeps_vectors_codeable(clip):
    """The desync bound, checked against the backend that now finds the vectors."""
    frames = _clip(clip, limit=5)
    limit = 2 * video.SEARCH
    for i in range(1, len(frames)):
        _, mvs = video.choose_modes(frames[i][0], frames[i - 1][0])
        assert np.abs(mvs).max() <= limit
        nbx = mvs.shape[1]
        listed = mvs.tolist()
        for by in range(mvs.shape[0]):
            for bx in range(nbx):
                py, px = video.mv_predictor(listed, by, bx, nbx)
                assert abs(int(mvs[by][bx][0]) - py) <= video.MV_MAX
                assert abs(int(mvs[by][bx][1]) - px) <= video.MV_MAX


def test_threading_does_not_change_the_result():
    """Single-threaded and many-threaded searches must agree exactly."""
    frames = _clip("foreman_cif.y4m", limit=3)
    cur, ref = frames[1][0], frames[0][0]
    old = os.environ.get("HVE_THREADS")
    try:
        os.environ["HVE_THREADS"] = "1"
        one, one_cost = native.motion_search(cur, ref, *_search_args())
        os.environ["HVE_THREADS"] = "8"
        many, many_cost = native.motion_search(cur, ref, *_search_args())
    finally:
        if old is None:
            os.environ.pop("HVE_THREADS", None)
        else:
            os.environ["HVE_THREADS"] = old
    assert np.array_equal(one, many)
    assert np.array_equal(one_cost, many_cost)


# --------------------------------------------------------------------------
# the backend contract


def test_container_output_is_backend_independent():
    """A file written by the C backend must decode through the reference too.

    This is the property a user actually depends on: a .hvi written on a machine
    with a compiler has to open on one without.
    """
    img = synthetic(24, 32, 3)
    blob = image.encode(img)
    real = native.available
    try:
        native.available = lambda: False
        assert np.array_equal(image.decode(blob), img)
    finally:
        native.available = real
    assert np.array_equal(image.decode(blob), img)


def test_params_array_matches_the_c_enum():
    """csrc/hve.h indexes `params` by a hand-written enum, so a mismatch in
    length means the C kernel is silently reading the wrong tunables."""
    assert len(model.coder_params()) == 28
