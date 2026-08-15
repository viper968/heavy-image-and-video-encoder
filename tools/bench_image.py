"""Image benchmark against every codec available, with losslessness verified.

Every baseline is decoded and compared against the source before its size is
reported. Pillow will happily accept `lossless=True` for AVIF and hand back an
image that is not lossless, so an unverified number is worth nothing here.
"""

import io
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    """Only codecs this machine can actually run.

    Pillow builds differ in which formats they carry, and imagecodecs may be
    absent entirely. A benchmark that dies because one baseline is missing is
    useless on a machine that has the others, so each candidate is probed once
    on a tiny image and dropped if it cannot run.
    """
    candidates = [
        ("hve", hve_codec),
        ("PNG (optimised)", pillow_codec("PNG", optimize=True)),
        ("WebP lossless", pillow_codec("WEBP", lossless=True, quality=100, method=6)),
        ("AVIF 'lossless'", pillow_codec("AVIF", lossless=True, quality=100)),
    ]
    try:
        candidates.append(("JPEG XL e7", jxl_codec(7)))
        candidates.append(("JPEG XL e9", jxl_codec(9)))
    except ImportError:
        pass

    probe = np.zeros((8, 8, 3), dtype=np.uint8)
    probe[::2] = 200
    codecs = []
    for name, run in candidates:
        try:
            run(probe)
        except Exception as exc:                      # missing format, missing lib
            print("skip %s: %s" % (name, type(exc).__name__), file=sys.stderr)
            continue
        codecs.append((name, run))
    return codecs


def _baseline_versions():
    """Baseline sizes move when their libraries move; record which ones ran."""
    import PIL
    parts = ["Pillow %s" % PIL.__version__]
    try:
        from imagecodecs import __version__ as ic, jpegxl_version, webp_version
        parts += ["imagecodecs %s" % ic, jpegxl_version(), webp_version()]
    except ImportError:
        parts.append("imagecodecs absent")
    return ", ".join(parts)


_CODECS = None


def _worker_init():
    global _CODECS
    _CODECS = build_codecs()


def _measure(path):
    """Run every codec on one image. Sizes are deterministic, so splitting the
    corpus across processes changes only how long the run takes.

    Timing uses process CPU time, not wall clock: with more workers than
    physical cores, wall time per task inflates from contention and would make
    a parallel run look slower than a serial one for the same work.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    sizes, times, status = {}, {}, {}
    for name, run in _CODECS:
        start = time.process_time()
        blob, back = run(arr)
        times[name] = time.process_time() - start
        sizes[name] = len(blob)
        if np.array_equal(arr, back):
            status[name] = "lossless"
        else:
            err = np.abs(arr.astype(int) - back.astype(int))
            status[name] = "LOSSY (max err %d)" % err.max()
    return path, arr.shape[0] * arr.shape[1], arr.size, sizes, times, status


def main(paths, jobs=1):
    wall = time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_worker_init) as pool:
            results = []
            for res in pool.imap_unordered(_measure, paths):
                results.append(res)
                print("done %s" % res[0].split("/")[-1], flush=True)
    else:
        _worker_init()
        results = []
        for path in paths:
            results.append(_measure(path))
            print("done %s" % path.split("/")[-1], flush=True)
    wall = time.time() - wall

    names = [name for name, _ in build_codecs()]
    totals = {n: 0 for n in names}
    times = {n: 0.0 for n in names}
    status = {n: "lossless" for n in names}
    pixels = raw = 0
    for _, px, rawbytes, sizes, ts, st in results:
        pixels += px
        raw += rawbytes
        for n in names:
            totals[n] += sizes[n]
            times[n] += ts[n]
            if st[n] != "lossless":
                status[n] = st[n]

    print("\n%d images, %d pixels, %d raw bytes" % (len(paths), pixels, raw))
    print("%.1fs wall clock across %d process%s" % (wall, jobs, "" if jobs == 1 else "es"))
    print("baselines: %s\n" % _baseline_versions())
    print("%-18s %12s %8s %8s %9s  %s"
          % ("codec", "bytes", "bpp", "ratio", "cpu s", "verified"))
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
    argv = sys.argv[1:]
    jobs = 1
    for i, a in enumerate(argv):
        if a.startswith("--jobs"):
            jobs = int(a.split("=", 1)[1]) if "=" in a else int(argv[i + 1])
            argv = [v for j, v in enumerate(argv)
                    if j != i and not (("=" not in a) and j == i + 1)]
            break
    if jobs <= 0:
        import os
        jobs = os.cpu_count() or 1

    if not argv or argv[0] in ("dev", "test", "all"):
        import corpus
        which = argv[0] if argv else "test"
        paths = {"dev": corpus.dev, "test": corpus.test,
                 "all": lambda: corpus.dev() + corpus.test()}[which]()
        print("split: %s (%s)\n" % (which, corpus.describe()))
        main(paths, jobs)
    else:
        main(argv, jobs)
