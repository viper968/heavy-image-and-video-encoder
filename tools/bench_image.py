"""Image benchmark against every codec available, with losslessness verified.

Every baseline is decoded and compared against the source before its size is
reported. Pillow will happily accept `lossless=True` for AVIF and hand back an
image that is not lossless, so an unverified number is worth nothing here.
"""

import io
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import image as hve_image                    # noqa: E402


def pillow_codec(fmt, **kwargs):
    def run(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt, **kwargs)
        blob = buf.getvalue()
        back = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
        return blob, back
    return run


def jxl_codec(effort):
    from imagecodecs import jpegxl_decode, jpegxl_encode

    def run(arr):
        blob = jpegxl_encode(arr, lossless=True, effort=effort)
        return blob, jpegxl_decode(blob)
    return run


def webp_lossless_codec():
    from imagecodecs import webp_decode, webp_encode

    def run(arr):
        blob = webp_encode(arr, level=-1, method=6)
        return blob, webp_decode(blob)[..., :3]
    return run


def hve_codec(arr):
    blob = hve_image.encode(arr)
    return blob, hve_image.decode(blob)


def build_codecs():
    codecs = [
        ("hve", hve_codec),
        ("PNG (optimised)", pillow_codec("PNG", optimize=True)),
        ("WebP lossless", pillow_codec("WEBP", lossless=True, quality=100, method=6)),
        ("AVIF 'lossless'", pillow_codec("AVIF", lossless=True, quality=100)),
    ]
    try:
        codecs.append(("JPEG XL e7", jxl_codec(7)))
        codecs.append(("JPEG XL e9", jxl_codec(9)))
    except ImportError:
        pass
    return codecs


def main(paths):
    codecs = build_codecs()
    totals = {name: 0 for name, _ in codecs}
    times = {name: 0.0 for name, _ in codecs}
    status = {name: "lossless" for name, _ in codecs}
    pixels = 0
    raw = 0

    for path in paths:
        arr = np.array(Image.open(path).convert("RGB"))
        pixels += arr.shape[0] * arr.shape[1]
        raw += arr.size
        for name, run in codecs:
            start = time.time()
            blob, back = run(arr)
            times[name] += time.time() - start
            totals[name] += len(blob)
            if not np.array_equal(arr, back):
                err = np.abs(arr.astype(int) - back.astype(int))
                status[name] = "LOSSY (max err %d)" % err.max()
        print("done %s" % path.split("/")[-1], flush=True)

    print("\n%d images, %d pixels, %d raw bytes\n" % (len(paths), pixels, raw))
    print("%-18s %12s %8s %8s %9s  %s"
          % ("codec", "bytes", "bpp", "ratio", "enc s", "verified"))
    rows = sorted(totals.items(), key=lambda kv: kv[1])
    for name, size in rows:
        print("%-18s %12d %8.3f %7.2fx %9.1f  %s"
              % (name, size, size * 8.0 / pixels, raw / size, times[name], status[name]))

    lossless = [(n, s) for n, s in rows if status[n] == "lossless"]
    if lossless and lossless[0][0] != "hve":
        best = lossless[0]
        mine = totals["hve"]
        print("\nhve is %+.2f%% vs best verified lossless (%s)"
              % (100.0 * (mine - best[1]) / best[1], best[0]))


if __name__ == "__main__":
    main(sys.argv[1:])
