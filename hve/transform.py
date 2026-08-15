"""Decorrelating transforms: reversible colour transform, MED prediction, contexts.

Everything works in modular 8-bit arithmetic. Wrapping around 255 -> 0 costs a
little entropy on the rare pixel that does it, but it keeps every plane exactly
8 bits wide, which is what makes the whole pipeline exactly reversible without
ever widening a sample. PNG's filters use the same trick.
"""

import numpy as np

# Activity ladders. More contexts model the image better but each one costs a
# frequency table in the header, so the encoder tries several and keeps the best.
CTX_LADDERS = {
    4: [1, 4, 12],
    6: [1, 3, 6, 12, 24],
    8: [1, 2, 4, 7, 12, 20, 36],
    12: [1, 2, 3, 4, 6, 9, 13, 20, 30, 46, 72],
    16: [1, 2, 3, 4, 5, 7, 9, 12, 16, 22, 30, 42, 58, 80, 112],
}
CTX_CONFIGS = sorted(CTX_LADDERS)


def rct_forward(rgb):
    """JPEG2000-style reversible colour transform, kept 8-bit by wrapping.

    The chroma planes are stored biased by +128. Unbiased, B-G sits at zero and
    therefore straddles the 255/0 wrap, which corrupts every gradient the model
    computes from it - activity, MED comparisons and neighbour differences all
    see a huge jump across what is really a one-step change. Biasing moves the
    typical value to mid-range where none of that happens.
    """
    a = rgb.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    cb = (b - g) & 255
    cr = (r - g) & 255
    y = (g + ((_centre(cb) + _centre(cr)) >> 2)) & 255
    return np.stack([y, (cb + 128) & 255, (cr + 128) & 255]).astype(np.uint8)


def rct_inverse(planes):
    y, cb_biased, cr_biased = (p.astype(np.int32) for p in planes)
    cb = (cb_biased - 128) & 255
    cr = (cr_biased - 128) & 255
    g = (y - ((_centre(cb) + _centre(cr)) >> 2)) & 255
    b = (g + cb) & 255
    r = (g + cr) & 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _centre(v):
    """Map an unsigned modular byte onto [-128, 127]."""
    return ((v + 128) & 255) - 128


def neighbours(plane):
    """Causal neighbours N, W, NW, NE as int32 planes (zero outside the image)."""
    p = plane.astype(np.int32)
    h, w = p.shape
    west = np.zeros_like(p)
    north = np.zeros_like(p)
    nwest = np.zeros_like(p)
    neast = np.zeros_like(p)
    west[:, 1:] = p[:, :-1]
    north[1:, :] = p[:-1, :]
    nwest[1:, 1:] = p[:-1, :-1]
    neast[1:, :-1] = p[:-1, 1:]
    neast[1:, -1] = p[:-1, -1]
    return north, west, nwest, neast


def med(north, west, nwest):
    """LOCO-I / JPEG-LS median edge detector.

    Picks the smaller neighbour across a detected vertical edge, the larger one
    across a horizontal edge, and the planar extrapolation otherwise.
    """
    hi = np.maximum(north, west)
    lo = np.minimum(north, west)
    planar = np.clip(north + west - nwest, 0, 255)
    return np.where(nwest >= hi, lo, np.where(nwest <= lo, hi, planar))


def predict_plane(plane):
    """Full-plane MED prediction with the standard edge fallbacks."""
    north, west, nwest, _ = neighbours(plane)
    pred = med(north, west, nwest)
    pred[0, :] = west[0, :]
    pred[:, 0] = north[:, 0]
    pred[0, 0] = 128
    return pred


def zigzag(residual):
    """Modular residual -> non-negative symbol, small values near zero."""
    d = _centre(residual)
    return np.where(d >= 0, d << 1, (-d << 1) - 1).astype(np.uint8)


def unzigzag(sym):
    d = -((sym + 1) >> 1) if (sym & 1) else (sym >> 1)
    return d & 255


UNZIGZAG = [(-((s + 1) >> 1) if (s & 1) else (s >> 1)) & 255 for s in range(256)]


def activity(plane):
    """Local texture energy from causal neighbours; the context selector."""
    north, west, nwest, neast = neighbours(plane)
    act = (np.abs(_centre(west - nwest))
           + np.abs(_centre(nwest - north))
           + np.abs(_centre(north - neast)))
    act[0, :] = 0
    act[:, 0] = 0
    return act


def bucketise(act, nctx):
    return np.searchsorted(CTX_LADDERS[nctx], act, side="right").astype(np.int32)


def analyse_plane(plane):
    """Residual symbols and raw activity for one plane."""
    pred = predict_plane(plane)
    resid = (plane.astype(np.int32) - pred) & 255
    return zigzag(resid), activity(plane)
