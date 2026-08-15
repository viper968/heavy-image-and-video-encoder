"""Where does the remaining gap live - prediction, or entropy modelling?

Measures the *ideal* cost of coding each predictor's residuals under the codec's
own context model: sum over contexts of n_c * H(residual | c). That is the floor
a perfect adaptive coder would reach with this model, so differences between
predictors are attributable to prediction alone.

Absolute numbers here are optimistic (a real adaptive coder pays a learning cost
and cannot reach the empirical conditional entropy). Relative numbers between
rows are the point.
"""

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import image as hve_image                       # noqa: E402
from hve.model import ACT_LADDER, ERR_LADDER             # noqa: E402
from hve.transform import _centre, med, rct_forward      # noqa: E402

NEIGHBOURS = ["W", "N", "NW", "NE", "WW", "NN"]


def shift(p, dy, dx):
    """Causal neighbour with edge replication."""
    h, w = p.shape
    out = np.empty_like(p)
    if dy:
        out[dy:, :] = p[:h - dy, :]
        out[:dy, :] = p[0:1, :]
    else:
        out[:, :] = p
    if dx > 0:
        out[:, dx:] = out[:, :w - dx]
        out[:, :dx] = out[:, dx:dx + 1]
    elif dx < 0:
        out[:, :w + dx] = out[:, -dx:]
        out[:, w + dx:] = out[:, w + dx - 1:w + dx]
    return out


def neighbourhood(plane):
    p = plane.astype(np.float64)
    return {
        "W": shift(p, 0, 1), "N": shift(p, 1, 0), "NW": shift(p, 1, 1),
        "NE": shift(p, 1, -1), "WW": shift(p, 0, 2), "NN": shift(p, 2, 0),
        "NNE": shift(p, 2, -1),
    }


def pred_med(nb):
    return med(nb["N"], nb["W"], nb["NW"])


def pred_gap(nb):
    dh = np.abs(nb["W"] - nb["WW"]) + np.abs(nb["N"] - nb["NW"]) + np.abs(nb["N"] - nb["NE"])
    dv = np.abs(nb["W"] - nb["NW"]) + np.abs(nb["N"] - nb["NN"]) + np.abs(nb["NE"] - nb["NNE"])
    d = dv - dh
    base = (nb["W"] + nb["N"]) / 2 + (nb["NE"] - nb["NW"]) / 4
    return np.where(d > 80, nb["W"],
           np.where(d < -80, nb["N"],
           np.where(d > 32, (base + nb["W"]) / 2,
           np.where(d > 8, (3 * base + nb["W"]) / 4,
           np.where(d < -32, (base + nb["N"]) / 2,
           np.where(d < -8, (3 * base + nb["N"]) / 4, base))))))


def fit_lsq(plane, nb, names=NEIGHBOURS):
    """Least-squares optimal linear predictor for this plane (MRP's core idea).

    Weights are fitted on the whole plane and would be sent in the header - six
    numbers, so the transmission cost is nil. Rows 0-1 and columns 0-1 are
    excluded from the fit because their neighbours are replicated, not real.
    """
    h, w = plane.shape
    if h < 4 or w < 4:
        return None, None
    mask = np.zeros((h, w), dtype=bool)
    mask[2:, 2:] = True
    design = np.stack([nb[n][mask] for n in names], axis=1)
    target = plane.astype(np.float64)[mask]
    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    weights = np.round(weights * 64) / 64.0            # quantised for transmission
    pred = sum(weights[i] * nb[n] for i, n in enumerate(names))
    return pred, weights


def conditional_cost(plane, pred):
    """Ideal bytes under the codec's own context model."""
    resid = (plane.astype(np.int64) - np.round(np.clip(pred, 0, 255)).astype(np.int64)) & 255
    d = ((resid + 128) & 255) - 128
    mag = np.abs(d)
    sym = np.where(d >= 0, d * 2, -d * 2 - 1)

    nb = neighbourhood(plane)
    act = (np.abs(_centre((nb["W"] - nb["NW"]).astype(np.int64)))
           + np.abs(_centre((nb["NW"] - nb["N"]).astype(np.int64)))
           + np.abs(_centre((nb["N"] - nb["NE"]).astype(np.int64))))
    err = shift(mag.astype(np.float64), 0, 1) + shift(mag.astype(np.float64), 1, 0)
    ctx = (np.searchsorted(ACT_LADDER, act, side="right") * (len(ERR_LADDER) + 1)
           + np.searchsorted(ERR_LADDER, err, side="right"))

    bits = 0.0
    flat_ctx = ctx.ravel()
    flat_sym = sym.ravel()
    for c in np.unique(flat_ctx):
        chunk = flat_sym[flat_ctx == c]
        hist = np.bincount(chunk, minlength=256).astype(np.float64)
        used = hist > 0
        p = hist[used] / hist.sum()
        bits += float(-(hist[used] * np.log2(p)).sum())
    return bits / 8.0


def main(paths):
    designs = {}
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        for plane in rct_forward(img):
            nb = neighbourhood(plane)
            lsq, _ = fit_lsq(plane, nb)
            candidates = {
                "MED (current)": pred_med(nb),
                "GAP (CALIC)": pred_gap(nb),
                "least-squares fit": lsq,
                "MED+LSQ average": (pred_med(nb) + lsq) / 2 if lsq is not None else None,
            }
            for name, pred in candidates.items():
                if pred is None:
                    continue
                designs[name] = designs.get(name, 0.0) + conditional_cost(plane, pred)
        print("done %s" % path.split("/")[-1], flush=True)

    real = sum(len(hve_image.encode(np.array(Image.open(p).convert("RGB")))) for p in paths)
    print("\n%-22s %12s %9s" % ("predictor", "ideal bytes", "vs MED"))
    ref = designs["MED (current)"]
    for name, total in sorted(designs.items(), key=lambda kv: kv[1]):
        print("%-22s %12d %8.2f%%" % (name, total, 100.0 * (total - ref) / ref))
    print("\n%-22s %12d   <- what the codec actually emits today" % ("hve measured", real))
    print("%-22s %12.1f%%  <- coder overhead above the ideal for MED"
          % ("gap to ideal", 100.0 * (real - ref) / ref))


if __name__ == "__main__":
    main(sys.argv[1:])
