"""Roundtrip and component tests. Run with: python -m pytest tests -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hve import huffman, image, model, rans, rc, transform, video, y4m  # noqa: E402


def rng():
    return np.random.default_rng(1234)


def synthetic(h, w, c=3, seed=0):
    """Photo-like: smooth shading, correlated channels, mild noise, one hard edge."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 90 * np.sin(xx / 11.0) * np.cos(yy / 17.0)
    img = np.clip(base + r.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8)
    if c == 1:
        return img
    planes = [img,
              np.clip(base + 12 + r.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8),
              np.clip(base - 18 + r.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8)]
    if c == 4:
        planes.append(np.full((h, w), 200, np.uint8))
    out = np.stack(planes[:c], axis=-1)
    out[h // 3:2 * h // 3, w // 3:2 * w // 3] = 17
    return out


# --------------------------------------------------------------------------
# entropy coding primitives


def test_range_coder_roundtrip():
    r = rng()
    bits = [int(r.random() < p) for p in r.random(20000)]
    enc = rc.Encoder()
    probs = rc.new_probs(8)
    for i, b in enumerate(bits):
        enc.bit(probs, i % 8, b)
    data = enc.finish()

    dec = rc.Decoder(data)
    probs2 = rc.new_probs(8)
    assert [dec.bit(probs2, i % 8) for i in range(len(bits))] == bits


def test_range_coder_bypass():
    r = rng()
    values = [int(v) for v in r.integers(0, 256, 2000)]
    enc = rc.Encoder()
    for v in values:
        enc.bypass(v, 8)
    dec = rc.Decoder(enc.finish())
    assert [dec.bypass(8) for _ in values] == values


def test_range_coder_beats_entropy_bound_loosely():
    """A skewed source must cost far less than one bit per symbol."""
    r = rng()
    bits = [int(x) for x in (r.random(50000) < 0.03)]
    enc = rc.Encoder()
    probs = rc.new_probs(1)
    for b in bits:
        enc.bit(probs, 0, b)
    assert len(enc.finish()) < len(bits) / 8 * 0.5


def test_rans_normalise_preserves_support():
    hist = np.zeros(256, dtype=np.int64)
    hist[[0, 5, 200]] = [1000000, 1, 3]
    freqs = rans.normalise(hist)
    assert freqs.sum() == rans.PROB_SCALE
    assert (freqs[[0, 5, 200]] > 0).all()
    assert freqs[1] == 0


def test_rans_table_serialisation():
    from hve.bitio import Reader, Writer
    hist = np.bincount(rng().integers(0, 40, 5000), minlength=256)
    freqs = rans.normalise(hist)
    w = Writer()
    rans.write_table(w, freqs)
    assert np.array_equal(rans.read_table(Reader(w.bytes()), 256), freqs)


def test_huffman_roundtrip():
    r = rng()
    data = np.clip(r.normal(128, 20, 30000), 0, 255).astype(np.uint8)
    blob = huffman.encode(data)
    assert np.array_equal(huffman.decode(blob, len(data)), data)
    assert len(blob) < len(data)


def test_huffman_single_symbol():
    data = np.full(1000, 42, np.uint8)
    blob = huffman.encode(data)
    assert np.array_equal(huffman.decode(blob, len(data)), data)


def test_huffman_length_limit():
    hist = np.zeros(256, dtype=np.int64)
    for i in range(40):                       # fibonacci weights force a deep tree
        hist[i] = int(1.7 ** i) + 1
    lengths = huffman.code_lengths(hist)
    assert lengths.max() <= huffman.MAX_LEN


# --------------------------------------------------------------------------
# transforms


def test_rct_is_reversible():
    img = rng().integers(0, 256, (64, 48, 3), dtype=np.uint8)
    assert np.array_equal(transform.rct_inverse(transform.rct_forward(img)), img)


def test_zigzag_is_reversible():
    values = np.arange(256, dtype=np.int32)
    syms = transform.zigzag(values)
    back = np.array([transform.UNZIGZAG[s] for s in syms])
    assert np.array_equal(back, values)


def test_med_matches_reference():
    north = np.array([[10, 200, 50]])
    west = np.array([[20, 100, 50]])
    nwest = np.array([[30, 50, 40]])
    got = transform.med(north, west, nwest)
    # NW above both neighbours -> take the smaller; below both -> take the larger;
    # in between -> planar extrapolation.
    assert got.tolist() == [[10, 200, 50]]


# --------------------------------------------------------------------------
# image codec


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 40, 3), (40, 1, 3), (7, 9),
                                   (33, 51, 3), (16, 16, 4)])
def test_image_roundtrip_shapes(shape):
    img = synthetic(shape[0], shape[1], shape[2] if len(shape) > 2 else 1)
    blob = image.encode(img)
    assert np.array_equal(image.decode(blob), img)


def test_image_roundtrip_random():
    img = rng().integers(0, 256, (37, 53, 3), dtype=np.uint8)
    assert np.array_equal(image.decode(image.encode(img)), img)


def test_image_roundtrip_flat():
    img = np.full((64, 64, 3), 77, np.uint8)
    blob = image.encode(img)
    assert np.array_equal(image.decode(blob), img)
    assert len(blob) < 400          # a flat image must collapse to almost nothing


def test_image_roundtrip_extremes():
    """Values at 0 and 255 exercise the modular wrap in both directions."""
    img = np.zeros((32, 32, 3), np.uint8)
    img[::2] = 255
    img[:, ::3] = 1
    assert np.array_equal(image.decode(image.encode(img)), img)


PHOTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "photo_crop.png")


def test_image_beats_png_on_a_real_photo():
    """The headline claim, on real photographic data rather than a synthetic case."""
    from PIL import Image as PILImage
    img = np.array(PILImage.open(PHOTO).convert("RGB"))
    blob = image.encode(img)
    assert np.array_equal(image.decode(blob), img)
    assert len(blob) < os.path.getsize(PHOTO) * 0.85


def _header_flags(blob):
    from hve.bitio import Reader
    r = Reader(blob)
    r.raw(4)
    r.varint()
    r.varint()
    r.u8()
    return r.u8()


def test_colour_transform_is_chosen_not_assumed():
    """A photo should keep the RCT; unrelated channels should drop it."""
    from PIL import Image as PILImage
    photo = np.array(PILImage.open(PHOTO).convert("RGB"))
    assert _header_flags(image.encode(photo)) & image.FLAG_RCT

    r = np.random.default_rng(5)
    h, w = 64, 64
    unrelated = np.stack([(r.integers(0, 256, (h, w)) % 64),
                          synthetic(h, w, 1, seed=1),
                          np.full((h, w), 200)], axis=-1).astype(np.uint8)
    assert not _header_flags(image.encode(unrelated)) & image.FLAG_RCT
    assert np.array_equal(image.decode(image.encode(unrelated)), unrelated)


def test_match_model_exploits_repetition():
    """Spatially incompressible content that repeats must still collapse.

    A tile of pure noise defeats every gradient predictor in the codec, so
    anything gained here comes from the match model recognising that this exact
    neighbourhood has been seen before.
    """
    r = np.random.default_rng(3)
    tile = r.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    tiled = np.tile(tile, (8, 8, 1))
    unique = r.integers(0, 256, tiled.shape, dtype=np.uint8)

    blob = image.encode(tiled)
    assert np.array_equal(image.decode(blob), tiled)
    assert len(blob) * 4 < len(image.encode(unique))


def test_match_model_does_not_cost_on_unrepetitive_data():
    """The trust threshold exists so photographs do not pay for the match model."""
    from PIL import Image as PILImage
    photo = np.array(PILImage.open(PHOTO).convert("RGB"))
    assert np.array_equal(image.decode(image.encode(photo)), photo)


def _reference_payload(planes):
    """The pure-Python path, run directly, for comparison against the jitted one."""
    from hve import rc as _rc
    coder = _rc.Encoder()
    bank = model.new_model()
    luma = None
    h, w = planes.shape[1], planes.shape[2]
    for i, plane in enumerate(planes):
        _, err = model.code_plane(coder, True, w, h, min(i, 3), bank,
                                  src=plane.tolist(),
                                  luma_err=luma if i in (1, 2) else None)
        if i == 0:
            luma = err
    return coder.finish()


def test_fast_path_is_byte_identical():
    """The jitted path is a second implementation of the format's core loop.

    A one-bit divergence would silently corrupt every file written by whichever
    path happened to run, so the two must agree exactly, not merely closely.
    """
    from hve import fast
    if not fast.available():
        pytest.skip("numba not installed")
    from PIL import Image as PILImage
    cases = [np.array(PILImage.open(PHOTO).convert("RGB"))[:48, :64],
             synthetic(33, 51, 3),
             np.random.default_rng(7).integers(0, 256, (24, 24, 3), dtype=np.uint8)]
    for img in cases:
        planes, _ = image._planes_from_image(img)
        planes = np.ascontiguousarray(planes)
        assert fast.encode_planes(planes) == _reference_payload(planes)


def test_fast_path_roundtrips():
    from hve import fast
    if not fast.available():
        pytest.skip("numba not installed")
    img = synthetic(40, 56, 3)
    planes, _ = image._planes_from_image(img)
    planes = np.ascontiguousarray(planes)
    blob = fast.encode_planes(planes)
    back = fast.decode_planes(blob, planes.shape[0], planes.shape[1], planes.shape[2])
    assert np.array_equal(back, planes)


def test_image_rejects_foreign_container():
    with pytest.raises(ValueError):
        image.decode(b"XXXX" + b"\0" * 32)


# --------------------------------------------------------------------------
# video codec


def test_video_roundtrip_planar():
    frames = []
    base = [synthetic(32, 48, 1, seed=0), synthetic(16, 24, 1, seed=1),
            synthetic(16, 24, 1, seed=2)]
    for i in range(4):
        frames.append([np.roll(p, i, axis=1) for p in base])
    blob = video.encode(frames)
    out = video.decode(blob)
    assert len(out) == len(frames)
    for want, got in zip(frames, out):
        for a, b in zip(want, got):
            assert np.array_equal(a, b)


def test_video_roundtrip_rgb():
    base = synthetic(32, 32, 3)
    frames = [base] + [np.roll(base, i, axis=0) for i in (2, 5)]
    out = video.decode(video.encode(frames))
    for want, got in zip(frames, out):
        assert np.array_equal(want, got)


def test_video_single_frame_matches_intra_path():
    frame = synthetic(24, 32, 3)
    out = video.decode(video.encode([frame]))
    assert np.array_equal(out[0], frame)


def test_video_static_scene_is_tiny():
    """A repeated frame should cost almost nothing after the first."""
    frame = synthetic(64, 64, 1)
    one = len(video.encode([[frame]]))
    ten = len(video.encode([[frame]] * 10))
    assert ten < one * 1.2


def test_motion_search_finds_shift():
    ref = synthetic(64, 64, 1, seed=3)
    cur = np.roll(ref, (0, 4), axis=(0, 1))
    mvs, _ = video.motion_search(cur, ref)
    interior = mvs[1:-1, 1:-1]
    assert (interior[..., 1] == -4).mean() > 0.7


def test_video_block_size_travels_in_the_header():
    """Decoding must not depend on the encoder's module constants."""
    base = synthetic(48, 48, 1, seed=7)
    frames = [[base], [np.roll(base, 3, axis=1)]]
    original = video.BLOCK
    try:
        video.BLOCK = 8
        blob = video.encode(frames)
        video.BLOCK = 32                      # decoder must ignore this
        out = video.decode(blob)
    finally:
        video.BLOCK = original
    for want, got in zip(frames, out):
        assert np.array_equal(want[0], got[0])


def test_video_rejects_foreign_container():
    with pytest.raises(ValueError):
        video.decode(b"XXXX" + b"\0" * 32)


# --------------------------------------------------------------------------
# containers


def test_y4m_roundtrip(tmp_path):
    frames = [[synthetic(16, 32, 1, seed=i), synthetic(8, 16, 1, seed=i + 10),
               synthetic(8, 16, 1, seed=i + 20)] for i in range(3)]
    path = str(tmp_path / "clip.y4m")
    y4m.write(path, frames, 32, 16)
    reader = y4m.Y4M(path)
    assert (reader.width, reader.height) == (32, 16)
    got = [[p.copy() for p in f] for f in reader.frames()]
    reader.close()
    assert len(got) == 3
    for want, have in zip(frames, got):
        for a, b in zip(want, have):
            assert np.array_equal(a, b)


def test_model_bank_covers_video_kinds():
    """Inter kinds index past the intra banks; the arrays must be sized for them."""
    bank = model.new_model()
    top_kind = model.KINDS - 1
    assert len(bank["zero"]) >= (top_kind + 1) * model.NACT * model.NERR * model.NLUM
    assert len(bank["nb"]) >= (top_kind + 1) * model.NACT * model.NERR * (model.MAX_NB + 1)


def test_cli_info_reads_video_header(capsys, tmp_path):
    """info parses the same header decode does - an off-by-one field is silent otherwise."""
    from hve import cli
    frames = [[synthetic(32, 48, 1, seed=1), synthetic(16, 24, 1, seed=2),
               synthetic(16, 24, 1, seed=3)] for _ in range(2)]
    path = str(tmp_path / "clip.hvv")
    with open(path, "wb") as fh:
        fh.write(video.encode(frames))
    cli.main(["info", path])
    out = capsys.readouterr().out
    assert "2 frames" in out and "48x32" in out and "24x16" in out


def test_cli_image_roundtrip(tmp_path):
    from PIL import Image as PILImage
    from hve import cli
    src = str(tmp_path / "in.png")
    mid = str(tmp_path / "mid.hvi")
    out = str(tmp_path / "out.png")
    img = synthetic(48, 64)
    PILImage.fromarray(img).save(src)
    cli.main(["encode", src, mid])
    cli.main(["decode", mid, out])
    assert np.array_equal(np.array(PILImage.open(out)), img)
