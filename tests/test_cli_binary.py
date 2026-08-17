"""The standalone C binary must agree with the Python package, exactly.

`csrc/` builds an `hve` executable with no Python in it at all: its own
containers, colour transform, y4m and PNG. That is a second implementation of
everything outside the pixel loop, so these tests do the only thing that makes
that safe — encode the same input both ways and compare bytes, then make each
side decode the other's files.

Run with: python -m pytest tests -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hve import image, video, y4m                       # noqa: E402
from tests.test_codecs import synthetic                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "hve")
CLIPS = os.path.join(ROOT, "testdata", "video")


def _build():
    """Build once per session; skip the whole module if there is no compiler."""
    proc = subprocess.run(["make", "-s", "-C", os.path.join(ROOT, "csrc")],
                          capture_output=True)
    if proc.returncode != 0 or not os.path.exists(BINARY):
        return proc.stderr.decode()[-400:] or "make failed"
    return None


_BUILD_ERROR = _build()
pytestmark = pytest.mark.skipif(_BUILD_ERROR is not None,
                                reason="cannot build the binary: %s" % _BUILD_ERROR)


def run(*args):
    proc = subprocess.run([BINARY] + list(args), capture_output=True)
    if proc.returncode != 0:
        raise AssertionError("hve %s failed: %s"
                             % (" ".join(args), proc.stderr.decode().strip()))
    return proc.stdout.decode()


def png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)
    return path


def read_png(path):
    from PIL import Image
    im = Image.open(path)
    return np.array(im.convert("RGB") if im.mode == "P" else im)


# --------------------------------------------------------------------------
# stills


@pytest.mark.parametrize("shape", [(1, 1, 3), (37, 53, 3), (16, 16, 4),
                                   (40, 40), (20, 20, 3)])
def test_binary_matches_python_bytes(tmp_path, shape):
    """Same input, same file — the whole contract in one assertion."""
    r = np.random.default_rng(shape[0] * 100 + shape[1])
    arr = r.integers(0, 256, shape, dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    out = str(tmp_path / "c.hvi")
    run("encode", src, out)
    with open(out, "rb") as fh:
        got = fh.read()
    assert got == image.encode(read_png(src))


def test_binary_matches_python_on_a_photograph(tmp_path):
    from PIL import Image as PILImage
    path = os.path.join(ROOT, "testdata", "images", "kodim05.png")
    if not os.path.exists(path):
        pytest.skip("run tools/fetch_testdata.py")
    out = str(tmp_path / "c.hvi")
    run("encode", path, out)
    with open(out, "rb") as fh:
        got = fh.read()
    assert got == image.encode(np.array(PILImage.open(path)))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "flat"])
def test_still_roundtrips_through_the_binary(tmp_path, mode):
    """Colour types the PNG layer has to get right, including the one that
    lodepng would otherwise re-encode into a smaller representation."""
    from PIL import Image as PILImage
    r = np.random.default_rng(8)
    if mode == "RGB":
        arr = r.integers(0, 256, (23, 31, 3), dtype=np.uint8)
    elif mode == "RGBA":
        arr = r.integers(0, 256, (23, 31, 4), dtype=np.uint8)
    elif mode == "L":
        arr = r.integers(0, 256, (23, 31), dtype=np.uint8)
    elif mode == "flat":
        arr = np.full((20, 20, 3), 128, np.uint8)
    else:
        arr = r.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    src = str(tmp_path / "in.png")
    if mode == "P":
        PILImage.fromarray(arr).convert("P").save(src)
    else:
        PILImage.fromarray(arr).save(src)

    blob, back = str(tmp_path / "a.hvi"), str(tmp_path / "back.png")
    run("encode", src, blob)
    run("decode", blob, back)
    assert np.array_equal(read_png(back), read_png(src))


def test_binary_refuses_16_bit_rather_than_narrowing(tmp_path):
    """Quietly halving every sample and then reporting a lossless round trip
    would be a lie, so this must fail loudly."""
    from PIL import Image as PILImage
    src = str(tmp_path / "deep.png")
    arr = np.random.default_rng(2).integers(0, 65536, (16, 16)).astype(np.uint16)
    PILImage.fromarray(arr).save(src)
    proc = subprocess.run([BINARY, "encode", src, str(tmp_path / "x.hvi")],
                          capture_output=True)
    assert proc.returncode != 0
    assert b"8-bit" in proc.stderr


# --------------------------------------------------------------------------
# video


def _write_y4m(path, frames, w, h):
    y4m.write(path, frames, w, h)
    return path


def _synth_clip(h, w, n=4):
    ch, cw = -(-h // 2), -(-w // 2)
    base = [synthetic(h, w, 1, seed=0), synthetic(ch, cw, 1, seed=1),
            synthetic(ch, cw, 1, seed=2)]
    return [[np.roll(p, i * 2, axis=1) for p in base] for i in range(n)]


def test_video_matches_python_bytes(tmp_path):
    path = os.path.join(CLIPS, "foreman_cif.y4m")
    if not os.path.exists(path):
        pytest.skip("run tools/fetch_testdata.py")
    out = str(tmp_path / "c.hvv")
    run("encode", path, out, "--frames", "6")
    with open(out, "rb") as fh:
        got = fh.read()
    reader = y4m.Y4M(path)
    frames = [[p.copy() for p in f] for f in reader.frames(limit=6)]
    reader.close()
    assert got == video.encode(frames)


@pytest.mark.parametrize("size", [(70, 100), (72, 96), (33, 47)])
def test_video_dimensions_that_are_not_block_multiples(tmp_path, size):
    """Regression: the block grid is a *ceiling* division.

    Python spells that -(-h // 16); transcribed literally into C it becomes
    -(-h / 16), which truncates instead of flooring and gives 67 rows for a
    1080-line frame instead of 68. The motion search then wrote one block row
    past the end of the array. Every clip in testdata/ divides exactly by 16,
    so nothing caught it until a 1080p file corrupted the heap.
    """
    h, w = size
    frames = _synth_clip(h, w)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, w, h)
    blob = str(tmp_path / "a.hvv")
    run("encode", src, blob)
    with open(blob, "rb") as fh:
        assert fh.read() == video.encode(frames)


def test_video_roundtrips_through_the_binary(tmp_path):
    frames = _synth_clip(70, 100)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, 100, 70)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "back.y4m")
    run("encode", src, blob)
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    assert len(got) == len(frames)
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# the two implementations must read each other's files


def test_python_decodes_a_file_the_binary_wrote(tmp_path):
    arr = np.random.default_rng(12).integers(0, 256, (29, 41, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    blob = str(tmp_path / "c.hvi")
    run("encode", src, blob)
    with open(blob, "rb") as fh:
        assert np.array_equal(image.decode(fh.read()), arr)


def test_binary_decodes_a_file_python_wrote(tmp_path):
    arr = np.random.default_rng(13).integers(0, 256, (29, 41, 3), dtype=np.uint8)
    blob = str(tmp_path / "p.hvi")
    with open(blob, "wb") as fh:
        fh.write(image.encode(arr))
    back = str(tmp_path / "back.png")
    run("decode", blob, back)
    assert np.array_equal(read_png(back), arr)


def test_binary_decodes_video_python_wrote(tmp_path):
    frames = _synth_clip(48, 64)
    blob = str(tmp_path / "p.hvv")
    with open(blob, "wb") as fh:
        fh.write(video.encode(frames))
    back = str(tmp_path / "back.y4m")
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# the command line itself


def test_info_reports_the_same_shape_as_python(tmp_path):
    arr = np.random.default_rng(14).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    blob = str(tmp_path / "a.hvi")
    with open(blob, "wb") as fh:
        fh.write(image.encode(arr))
    out = run("info", blob)
    assert "23x17" in out and "3 channels" in out


def test_rejects_a_foreign_container(tmp_path):
    junk = str(tmp_path / "junk.hvi")
    with open(junk, "wb") as fh:
        fh.write(b"XXXX" + b"\0" * 64)
    proc = subprocess.run([BINARY, "decode", junk, str(tmp_path / "o.png")],
                          capture_output=True)
    assert proc.returncode != 0


def test_truncated_file_is_rejected_not_crashed(tmp_path):
    """Container parsing is the only place that touches untrusted bytes."""
    arr = np.random.default_rng(15).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    full = image.encode(arr)
    for cut in (5, 9, len(full) // 2, len(full) - 1):
        path = str(tmp_path / ("cut%d.hvi" % cut))
        with open(path, "wb") as fh:
            fh.write(full[:cut])
        proc = subprocess.run([BINARY, "info", path], capture_output=True)
        assert proc.returncode in (0, 1)          # never a signal


# --------------------------------------------------------------------------
# presets (the feature bitmask in the header)


PRESET_FAST = 127 & ~(4 | 2 | 1)          # -match -lms -blend, as csrc/main.c


@pytest.mark.parametrize("preset,features", [("max", 127), ("fast", PRESET_FAST)])
def test_preset_bytes_match_python(tmp_path, preset, features):
    arr = np.random.default_rng(21).integers(0, 256, (33, 41, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    out = str(tmp_path / "c.hvi")
    run("encode", src, out, "--preset", preset)
    with open(out, "rb") as fh:
        assert fh.read() == image.encode(read_png(src), features=features)


@pytest.mark.parametrize("preset", ["max", "fast"])
def test_preset_is_read_back_from_the_header(tmp_path, preset):
    """Decoding must need no flag: the file says which model made it."""
    arr = np.random.default_rng(22).integers(0, 256, (28, 36, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    blob, back = str(tmp_path / "a.hvi"), str(tmp_path / "b.png")
    run("encode", src, blob, "--preset", preset)
    run("decode", blob, back)                       # no --preset here
    assert np.array_equal(read_png(back), arr)
    assert preset in run("info", blob)


def test_video_preset_roundtrips(tmp_path):
    frames = _synth_clip(48, 64)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, 64, 48)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "b.y4m")
    run("encode", src, blob, "--preset", "fast")
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


def test_pure_python_refuses_a_preset_it_cannot_reproduce(tmp_path):
    """model.py implements only the full model. A reduced-preset file is valid
    but it cannot reproduce it, so it must refuse rather than decode wrongly."""
    from hve import model, native
    arr = np.random.default_rng(23).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    blob = image.encode(arr, features=PRESET_FAST)
    real = native.available
    try:
        native.available = lambda: False
        with pytest.raises(RuntimeError, match="pure-Python"):
            image.decode(blob)
    finally:
        native.available = real
    assert np.array_equal(image.decode(blob), arr)


def test_old_container_magic_is_rejected(tmp_path):
    """The header grew a byte, so the magic was bumped: an old file must fail
    loudly rather than misparse the new field."""
    arr = np.random.default_rng(24).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    blob = bytearray(image.encode(arr))
    blob[:4] = b"HVI2"
    old = str(tmp_path / "old.hvi")
    with open(old, "wb") as fh:
        fh.write(bytes(blob))
    proc = subprocess.run([BINARY, "info", old], capture_output=True)
    assert proc.returncode != 0
    with pytest.raises(ValueError):
        image.decode(bytes(blob))


# --------------------------------------------------------------------------
# slices (thread parallelism)


@pytest.mark.parametrize("nslices", [1, 2, 3, 4, 8])
def test_sliced_stills_roundtrip(tmp_path, nslices):
    """Odd slice counts matter: the boundaries are a derived division, so an
    off-by-one there loses or duplicates a row rather than failing."""
    arr = np.random.default_rng(31).integers(0, 256, (67, 53, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    blob, back = str(tmp_path / "a.hvi"), str(tmp_path / "b.png")
    run("encode", src, blob, "--slices", str(nslices))
    run("decode", blob, back)
    got = read_png(back)
    assert got.shape == arr.shape
    assert np.array_equal(got, arr)


@pytest.mark.parametrize("nslices", [2, 3, 5])
def test_sliced_video_roundtrips(tmp_path, nslices):
    """Slice boundaries have to land on even luma rows or the 4:2:0 chroma
    planes split at a fractional row."""
    frames = _synth_clip(70, 96)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, 96, 70)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "b.y4m")
    run("encode", src, blob, "--slices", str(nslices))
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    assert len(got) == len(frames)
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert a.shape == b.shape
            assert np.array_equal(a, b)


def test_slice_count_is_capped_by_height(tmp_path):
    """Asking for more slices than there are block rows must clamp, not make
    zero-row slices."""
    frames = _synth_clip(32, 64)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, 64, 32)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "b.y4m")
    run("encode", src, blob, "--slices", "32")
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


def test_slices_combine_with_the_fast_preset(tmp_path):
    frames = _synth_clip(64, 96)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, 96, 64)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "b.y4m")
    run("encode", src, blob, "--slices", "4", "--preset", "fast")
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


def test_single_slice_is_byte_identical_to_the_unsliced_encoder(tmp_path):
    """--slices 1 must not wrap the stream, so it stays readable by Python."""
    arr = np.random.default_rng(32).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    one, none = str(tmp_path / "one.hvi"), str(tmp_path / "none.hvi")
    run("encode", src, one, "--slices", "1")
    run("encode", src, none)
    with open(one, "rb") as fh:
        blob = fh.read()
    assert blob == image.encode(arr)
    assert np.array_equal(image.decode(blob), arr)


def test_python_rejects_a_sliced_file_it_cannot_read(tmp_path):
    """Slices are a C-side feature for now; Python must say so rather than
    mistake the wrapper for a corrupt image."""
    arr = np.random.default_rng(33).integers(0, 256, (64, 48, 3), dtype=np.uint8)
    src = png(str(tmp_path / "in.png"), arr)
    blob = str(tmp_path / "a.hvi")
    run("encode", src, blob, "--slices", "4")
    with open(blob, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"HVS1"
    with pytest.raises(ValueError):
        image.decode(data)


# --------------------------------------------------------------------------
# the batched context derivation
#
# The encoder derives a whole row of contexts up front when the model allows
# it (see derive_row in csrc/kernel.c). That is a second implementation of the
# context formation, and the Python byte-identity tests cannot reach it: they
# compare against model.py, which implements only the full model, and the full
# model switches batching off. So it is pinned here instead, both ways round -
# against the scalar derivation directly, and through a decoder that only ever
# uses the scalar one.


@pytest.mark.parametrize("preset", ["fast", "max"])
@pytest.mark.parametrize("h,w", [(64, 96), (29, 37), (17, 33), (1, 1)])
def test_batched_derivation_matches_the_scalar_path(tmp_path, preset, h, w):
    """Deriving a row up front must not change a single byte."""
    frames = _synth_clip(h, w)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, w, h)
    on, off = str(tmp_path / "on.hvv"), str(tmp_path / "off.hvv")
    run("encode", src, on, "--slices", "1", "--preset", preset, "--batched", "1")
    run("encode", src, off, "--slices", "1", "--preset", preset, "--batched", "0")
    with open(on, "rb") as a, open(off, "rb") as b:
        assert a.read() == b.read()


@pytest.mark.parametrize("h,w", [(29, 37), (17, 33), (2, 3)])
def test_batched_encode_decodes_with_the_scalar_decoder(tmp_path, h, w):
    """The decoder has no batched path, so a round trip cross-checks the two.

    Odd sizes are the interesting ones: the derivation handles the first and
    last column outside its main loop, and a chroma plane whose subsampling
    does not divide evenly gets a block grid wider than the plane.
    """
    frames = _synth_clip(h, w)
    src = _write_y4m(str(tmp_path / "in.y4m"), frames, w, h)
    blob, back = str(tmp_path / "a.hvv"), str(tmp_path / "b.y4m")
    run("encode", src, blob, "--slices", "1", "--preset", "fast")
    run("decode", blob, back)
    reader = y4m.Y4M(back)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)
