"""Offline study: predictor choice and cross-channel context.

Charges header cost honestly, same as ctx_study, so designs are comparable.
"""

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import rans                                   # noqa: E402
from hve.transform import _centre, med, rct_forward, zigzag  # noqa: E402

ALPHABET = 256
ACT12 = [1, 2, 3, 4, 6, 9, 13, 20, 30, 46, 72]


def shift(p, dy, dx):
    """Causal neighbour fetch with edge replication (zeros above row 0)."""
    out = np.zeros_like(p)
    h, w = p.shape
    ys = slice(dy, h) if dy else slice(0, h)
    src_y = slice(0, h - dy) if dy else slice(0, h)
    if dx > 0:
        out[ys, dx:] = p[src_y, 0:w - dx]
        out[ys, 0:dx] = p[src_y, 0:1]
    elif dx < 0:
        out[ys, 0:w + dx] = p[src_y, -dx:]
        out[ys, w + dx:] = p[src_y, -1:]
    else:
        out[ys, :] = p[src_y, :]
    return out


def predictors(p):
    p = p.astype(np.int32)
    N = shift(p, 1, 0)
    W = shift(p, 0, 1)
    NW = shift(p, 1, 1)
    NE = shift(p, 1, -1)
    NN = shift(p, 2, 0)
    WW = shift(p, 0, 2)
    NNE = shift(p, 2, -1)

    out = {}
    out["MED"] = med(N, W, NW)
    out["avg"] = (N + W) >> 1
    out["planar"] = np.clip(N + W - NW, 0, 255)

    # CALIC's gradient-adjusted predictor
    dh = np.abs(W - WW) + np.abs(N - NW) + np.abs(N - NE)
    dv = np.abs(W - NW) + np.abs(N - NN) + np.abs(NE - NNE)
    d = dv - dh
    base = ((W + N) >> 1) + ((NE - NW) >> 2)
    gap = np.where(d > 80, W,
          np.where(d < -80, N,
          np.where(d > 32, (base + W) >> 1,
          np.where(d > 8, (3 * base + W) >> 2,
          np.where(d < -32, (base + N) >> 1,
          np.where(d < -8, (3 * base + N) >> 2, base))))))
    out["GAP"] = np.clip(gap, 0, 255)
    out["MED+GAP"] = np.clip((out["MED"] + out["GAP"] + 1) >> 1, 0, 255)
    for name in out:
        out[name] = _fix_edges(out[name], N, W)
    return out


def _fix_edges(pred, N, W):
    pred = pred.copy()
    pred[0, :] = W[0, :]
    pred[:, 0] = N[:, 0]
    pred[0, 0] = 128
    return pred


def activity(p):
    p = p.astype(np.int32)
    N, W, NW, NE = shift(p, 1, 0), shift(p, 0, 1), shift(p, 1, 1), shift(p, 1, -1)
    act = (np.abs(_centre(W - NW)) + np.abs(_centre(NW - N)) + np.abs(_centre(N - NE)))
    act[0, :] = 0
    act[:, 0] = 0
    return act


def cost(syms, ctxs, nctx):
    header, bits = 0, 0.0
    flat, sym = ctxs.ravel(), syms.ravel()
    order = np.argsort(flat, kind="stable")
    flat_s, sym_s = flat[order], sym[order]
    edges = np.searchsorted(flat_s, np.arange(nctx + 1))
    for c in range(nctx):
        chunk = sym_s[edges[c]:edges[c + 1]]
        if len(chunk) == 0:
            header += 1
            continue
        hist = np.bincount(chunk, minlength=ALPHABET)
        freqs = rans.normalise(hist)
        header += rans.table_size(freqs)
        bits += rans.estimate_bits(hist, freqs)
    return header + bits / 8.0


def main(paths):
    names = ["MED", "avg", "planar", "GAP", "MED+GAP"]
    totals = {n: 0.0 for n in names}
    totals["MED+xchan"] = 0.0
    totals["GAP+xchan"] = 0.0

    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        planes = rct_forward(img)
        acts = [np.searchsorted(ACT12, activity(p), side="right") for p in planes]
        preds = [predictors(p) for p in planes]

        for name in names:
            tot = 0.0
            for p, a, pr in zip(planes, acts, preds):
                syms = zigzag((p.astype(np.int32) - pr[name]) & 255)
                tot += cost(syms, a.astype(np.int64), 12)
            totals[name] += tot

        # cross-channel: chroma context conditioned on the co-located luma residual
        for base in ("MED", "GAP"):
            luma = zigzag((planes[0].astype(np.int32) - preds[0][base]) & 255)
            lbucket = np.searchsorted([1, 3, 6, 12, 24], luma, side="right")
            tot = cost(zigzag((planes[0].astype(np.int32) - preds[0][base]) & 255),
                       acts[0].astype(np.int64), 12)
            for i in (1, 2):
                syms = zigzag((planes[i].astype(np.int32) - preds[i][base]) & 255)
                ctx = acts[i].astype(np.int64) * 6 + lbucket
                tot += cost(syms, ctx, 72)
            totals[base + "+xchan"] += tot

    print("%-12s %12s %8s" % ("design", "total B", "vs MED"))
    ref = totals["MED"]
    for name, tot in sorted(totals.items(), key=lambda kv: kv[1]):
        print("%-12s %12d %7.2f%%" % (name, tot, 100.0 * (tot - ref) / ref))


if __name__ == "__main__":
    main(sys.argv[1:])
