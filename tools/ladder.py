"""The compression ladder: what each idea is actually worth, in bytes.

Starts from the original plan - Huffman-code each colour channel so 0..255 costs
under 8 bits - and adds one idea at a time, measuring every rung on real photos.
"""

import io
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import huffman, image, rans                    # noqa: E402
from hve.transform import (CTX_LADDERS, activity, bucketise,  # noqa: E402
                           predict_plane, rct_forward, zigzag)

RUNGS = [
    ("raw", "uncompressed RGB bytes"),
    ("huffman-global", "one Huffman tree over all bytes"),
    ("huffman-perchan", "a Huffman tree per colour channel (the original idea)"),
    ("huffman-med", "+ MED spatial prediction first"),
    ("huffman-rct-med", "+ reversible colour transform"),
    ("rans-ctx", "+ context-modelled static rANS (computed)"),
    ("hve", "+ adaptive contexts, cross-channel, sign modelling (measured)"),
]


def static_rans_size(planes):
    """Exact size of the static-table design: real header plus modelled payload."""
    total = 0
    for plane in planes:
        syms = zigzag((plane.astype(np.int32) - predict_plane(plane)) & 255)
        best = None
        for nctx in CTX_LADDERS:
            buckets = bucketise(activity(plane), nctx)
            header, bits = 0, 0.0
            for ctx in range(nctx):
                hist = np.bincount(syms[buckets == ctx], minlength=256)
                freqs = rans.normalise(hist)
                header += rans.table_size(freqs)
                if freqs is not None:
                    bits += rans.estimate_bits(hist, freqs)
            size = header + bits / 8.0
            best = size if best is None else min(best, size)
        total += best
    return total


def measure(path):
    img = np.array(Image.open(path).convert("RGB"))
    h, w, _ = img.shape
    out = {"raw": img.size}

    out["huffman-global"] = len(huffman.encode(img.ravel()))
    out["huffman-perchan"] = sum(len(huffman.encode(img[..., c])) for c in range(3))

    med = 0
    for c in range(3):
        plane = img[..., c]
        med += len(huffman.encode(zigzag((plane.astype(np.int32) - predict_plane(plane)) & 255)))
    out["huffman-med"] = med

    planes = rct_forward(img)
    rct_med = 0
    for plane in planes:
        rct_med += len(huffman.encode(zigzag((plane.astype(np.int32) - predict_plane(plane)) & 255)))
    out["huffman-rct-med"] = rct_med

    out["rans-ctx"] = int(round(static_rans_size(planes)))

    blob = image.encode(img)
    assert np.array_equal(img, image.decode(blob)), "hve roundtrip failed on " + path
    out["hve"] = len(blob)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    out["_png"] = len(buf.getvalue())
    out["_pixels"] = h * w
    return out


def main(paths):
    totals = {}
    for path in paths:
        res = measure(path)
        for k, v in res.items():
            totals[k] = totals.get(k, 0) + v
        print("measured %s" % path.split("/")[-1], flush=True)

    raw = totals["raw"]
    pixels = totals["_pixels"]
    print("\n%-18s %12s %9s %9s %10s  %s"
          % ("rung", "bytes", "bpp", "vs raw", "vs prev", "what changed"))
    prev = None
    for key, blurb in RUNGS:
        size = totals[key]
        bpp = size * 8.0 / pixels
        gain = "" if prev is None else "%+.1f%%" % (100.0 * (size - prev) / prev)
        print("%-18s %12d %9.3f %8.2fx %10s  %s"
              % (key, size, bpp, raw / size, gain, blurb))
        prev = size
    print("\n%-18s %12d %9.3f %8.2fx" % ("(PNG for scale)", totals["_png"],
                                         totals["_png"] * 8.0 / pixels,
                                         raw / totals["_png"]))


if __name__ == "__main__":
    main(sys.argv[1:])
