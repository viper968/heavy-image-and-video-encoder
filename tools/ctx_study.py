"""Offline study: how much does each context design actually buy?

Reports header + payload in bytes for a candidate context function, so the
cost of transmitting more frequency tables is charged honestly against the
modelling gain they produce.
"""

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import rans                                        # noqa: E402
from hve.transform import (_centre, neighbours, predict_plane,  # noqa: E402
                           rct_forward, zigzag)

ALPHABET = 256


def cost(syms, ctxs, nctx):
    """Total bytes to send `syms` under a context assignment."""
    header = 0
    bits = 0.0
    flat = ctxs.ravel()
    sym = syms.ravel()
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
    return header + bits / 8.0, header


def q_ladder(act, thresholds):
    return np.searchsorted(thresholds, act, side="right")


def jls_quant(d):
    """JPEG-LS gradient quantiser: 9 signed levels."""
    out = np.zeros_like(d)
    for t, v in ((2, 1), (7, 2), (21, 3)):
        out = np.where(d > t, v + 1, out)
    out = np.where(d > 21, 4, out)
    out = np.where((d > 7) & (d <= 21), 3, out)
    out = np.where((d > 2) & (d <= 7), 2, out)
    out = np.where((d > 0) & (d <= 2), 1, out)
    out = np.where((d < 0) & (d >= -2), -1, out)
    out = np.where((d < -2) & (d >= -7), -2, out)
    out = np.where((d < -7) & (d >= -21), -3, out)
    out = np.where(d < -21, -4, out)
    return out


def designs(plane):
    north, west, nwest, neast = neighbours(plane)
    d1 = _centre(west - nwest)
    d2 = _centre(nwest - north)
    d3 = _centre(north - neast)
    act = np.abs(d1) + np.abs(d2) + np.abs(d3)
    act[0, :] = 0
    act[:, 0] = 0

    out = {}
    for k, thr in [
        (4, [1, 4, 12]),
        (8, [1, 2, 4, 7, 12, 20, 36]),
        (12, [1, 2, 3, 4, 6, 9, 13, 20, 30, 46, 72]),
        (16, [1, 2, 3, 4, 5, 7, 9, 12, 16, 22, 30, 42, 58, 80, 112]),
        (24, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 29, 35, 42, 51,
              62, 76, 94, 118, 150]),
        (32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 28, 32,
              37, 43, 50, 58, 68, 80, 94, 110, 130, 154, 182, 215, 255]),
    ]:
        out["act%d" % k] = (q_ladder(act, thr), k)

    # activity x texture sign pattern
    a12 = q_ladder(act, [1, 2, 3, 4, 6, 9, 13, 20, 30, 46, 72])
    sign = (np.sign(d1) + 1) * 3 + (np.sign(d2) + 1)
    sign[0, :] = 4
    sign[:, 0] = 4
    out["act12xsign9"] = (a12 * 9 + sign, 12 * 9)

    a8 = q_ladder(act, [1, 2, 4, 7, 12, 20, 36])
    out["act8xsign9"] = (a8 * 9 + sign, 8 * 9)

    # JPEG-LS style quantised gradient triple, sign-folded to 365
    q1, q2, q3 = jls_quant(d1), jls_quant(d2), jls_quant(d3)
    idx = (q1 * 81 + q2 * 9 + q3)
    folded = np.abs(idx)
    folded[0, :] = 0
    folded[:, 0] = 0
    out["jpegls365"] = (folded, 365)
    return out


def main(paths):
    totals = {}
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        planes = rct_forward(img)
        per_design = {}
        for p in planes:
            syms = zigzag((p.astype(np.int32) - predict_plane(p)) & 255)
            for name, (ctxs, nctx) in designs(p).items():
                total, header = cost(syms, ctxs.astype(np.int64), nctx)
                cur = per_design.setdefault(name, [0.0, 0])
                cur[0] += total
                cur[1] += header
        for name, (total, header) in per_design.items():
            t = totals.setdefault(name, [0.0, 0])
            t[0] += total
            t[1] += header
        best = min(per_design.items(), key=lambda kv: kv[1][0])
        print("%-28s best=%-14s %8d B" % (path.split("/")[-1], best[0], best[1][0]))

    print("\n%-14s %12s %10s" % ("design", "total B", "header B"))
    for name, (total, header) in sorted(totals.items(), key=lambda kv: kv[1][0]):
        print("%-14s %12d %10d" % (name, total, header))


if __name__ == "__main__":
    main(sys.argv[1:])
