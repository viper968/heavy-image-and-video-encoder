"""Video benchmark against ffmpeg's lossless encoders, with losslessness verified.

Each baseline is decoded back to raw YUV and compared sample-for-sample against
the source frames, so "lossless" in the table means checked, not claimed.
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, ".")
from hve import video as hve_video, y4m               # noqa: E402


def find_ffmpeg():
    for candidate in ("ffmpeg", os.environ.get("FFMPEG")):
        if candidate and subprocess.run(["which", candidate],
                                        capture_output=True).returncode == 0:
            return candidate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


FFMPEG = find_ffmpeg()

BASELINES = [
    ("FFV1 (level 3)", ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1"], "mkv"),
    ("x264 lossless", ["-c:v", "libx264", "-qp", "0", "-preset", "veryslow"], "mkv"),
    ("x265 lossless", ["-c:v", "libx265", "-preset", "veryslow",
                       "-x265-params", "lossless=1:log-level=error"], "mkv"),
    ("VP9 lossless", ["-c:v", "libvpx-vp9", "-lossless", "1"], "mkv"),
    ("AV1 lossless", ["-c:v", "libaom-av1", "-cpu-used", "5",
                      "-aom-params", "lossless=1"], "mkv"),
    ("FFVHuff", ["-c:v", "ffvhuff"], "mkv"),
    ("Ut Video", ["-c:v", "utvideo"], "mkv"),
]


def run_ffmpeg(args):
    proc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y"] + args,
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[-500:])


def bench_baseline(src_y4m, args, ext, frames):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out." + ext)
        back = os.path.join(tmp, "back.y4m")
        start = time.time()
        run_ffmpeg(["-i", src_y4m] + args + [out])
        elapsed = time.time() - start
        size = os.path.getsize(out)
        run_ffmpeg(["-i", out, "-pix_fmt", "yuv420p", back])
        reader = y4m.Y4M(back)
        decoded = [[p.copy() for p in f] for f in reader.frames()]
        reader.close()
    ok = len(decoded) == len(frames) and all(
        np.array_equal(a, b) for fa, fb in zip(frames, decoded) for a, b in zip(fa, fb))
    return size, elapsed, ok


def main(path, nframes=16):
    reader = y4m.Y4M(path)
    frames = [[p.copy() for p in f] for f in reader.frames(limit=nframes)]
    width, height = reader.width, reader.height
    reader.close()
    raw = sum(p.size for f in frames for p in f)
    print("%s: %d frames of %dx%d, %d raw bytes\n" % (path.split("/")[-1],
                                                      len(frames), width, height, raw))

    results = []
    start = time.time()
    blob = hve_video.encode(frames)
    enc_time = time.time() - start
    start = time.time()
    decoded = hve_video.decode(blob)
    dec_time = time.time() - start
    ok = all(np.array_equal(a, b) for fa, fb in zip(frames, decoded) for a, b in zip(fa, fb))
    results.append(("hve", len(blob), enc_time, dec_time, ok))
    print("hve done (%.1fs enc, %.1fs dec)" % (enc_time, dec_time), flush=True)

    if FFMPEG:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.y4m")
            y4m.write(src, frames, width, height)
            for name, args, ext in BASELINES:
                try:
                    size, elapsed, baseline_ok = bench_baseline(src, args, ext, frames)
                except Exception as exc:                      # encoder missing or failed
                    print("skip %s: %s" % (name, str(exc).splitlines()[-1:]), flush=True)
                    continue
                results.append((name, size, elapsed, None, baseline_ok))
                print("%s done" % name, flush=True)
    else:
        print("ffmpeg not found - baselines skipped")

    print("\n%-18s %12s %10s %9s %9s  %s"
          % ("codec", "bytes", "ratio", "enc s", "dec s", "verified"))
    for name, size, enc, dec, ok in sorted(results, key=lambda r: r[1]):
        print("%-18s %12d %9.2fx %9.1f %9s  %s"
              % (name, size, raw / size, enc,
                 "%.1f" % dec if dec is not None else "-",
                 "lossless" if ok else "LOSSY"))
    note = "\nNote: container overhead (mkv) is included in the ffmpeg numbers."
    print(note)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], int(args[1]) if len(args) > 1 else 16)
