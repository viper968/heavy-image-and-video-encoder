"""A friendly front end for experimenting with the codec on your own files.

`python -m hve` is the real interface and it is deliberately literal: it takes
a PNG or a .y4m and nothing else. This wrapper exists so that someone who just
cloned the repo can point it at an ordinary jpg or mp4 and get an answer,
without first learning what a .y4m is.

It adds three things the core CLI does not have, all of them about honesty
rather than convenience:

  * It converts video via ffmpeg, and *tells you* when that conversion is
    itself lossy. The codec is lossless; a pipeline that quietly resamples
    10-bit 4:4:4 video down to 8-bit 4:2:0 before the codec ever sees it is
    not, and reporting "lossless" for that would be a lie.
  * It refuses image modes the codec cannot represent (16-bit, CMYK) instead
    of letting Pillow silently downconvert them.
  * `check` does a full round trip and compares every pixel, so the losslessness
    claim is something you verify rather than something you read.

Run `./playground/hve` with no arguments for the command list.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402

from hve import image as hve_image                       # noqa: E402
from hve import video as hve_video                       # noqa: E402
from hve import fast, native, y4m                        # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm",
             ".webp", ".gif", ".tga"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpg", ".mpeg",
             ".wmv", ".flv", ".y4m", ".yuv"}

# Pillow modes the codec stores exactly. Everything else either has more than 8
# bits per sample or is in a colour space that cannot survive the trip to RGB.
SAFE_MODES = {"L", "RGB", "RGBA"}
# These convert to a safe mode without changing any pixel value.
CONVERTIBLE = {"P": "RGB", "PA": "RGBA", "1": "L"}

# Rough throughput on the machine this was written on, only ever used to warn
# before something takes twenty minutes. Encode is the slower direction.
MP_PER_S_ENCODE = 0.5
DEFAULT_FRAMES = 60


# --------------------------------------------------------------------------
# small helpers


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0


def need(tool):
    path = shutil.which(tool)
    if not path:
        die("%s is required for video and was not found on PATH.\n"
            "Install ffmpeg (which provides both ffmpeg and ffprobe)." % tool)
    return path


def die(msg):
    sys.stderr.write("error: %s\n" % msg)
    raise SystemExit(1)


def warn(msg):
    sys.stderr.write("warning: %s\n" % msg)


# Formats that already threw pixels away. Comparing a lossless codec's output
# against one of these is not a comparison, and someone running this on a phone
# photo for the first time will otherwise conclude the codec is terrible.
LOSSY_EXT = {".jpg", ".jpeg", ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v",
             ".mpg", ".mpeg", ".wmv", ".flv"}


def report_sizes(path, raw, on_disk, nbytes, timing=None, kind="image"):
    print("  raw pixels     %10s" % human(raw))
    print("  source file    %10s" % human(on_disk))
    print("  hve            %10s   %.2fx smaller than the raw pixels"
          % (human(nbytes), float(raw) / nbytes))
    if os.path.splitext(path)[1].lower() not in LOSSY_EXT:
        print("  vs the source file           %.2fx smaller"
              % (float(on_disk) / nbytes))
    if timing:
        print("  " + timing)
    if os.path.splitext(path)[1].lower() in LOSSY_EXT:
        print()
        print("  Note: %s is a lossy file. It is smaller than this because it"
              % os.path.basename(path))
        print("  discarded image data permanently, which a lossless codec by "
              "definition cannot")
        print("  do. The fair comparison is against another *lossless* codec:")
        if kind == "video":
            print("    `restore` writes an FFV1 .mkv - compare that file's "
                  "size with the .hvv.")
        else:
            print("    run `compare` to see it against PNG, WebP and JPEG XL.")


def kind_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext == ".hvi":
        return "hvi"
    if ext == ".hvv":
        return "hvv"
    return None


# --------------------------------------------------------------------------
# loading


def load_image(path, force=False):
    """Pillow array plus a note about anything the codec cannot keep.

    The core CLI converts any unusual mode straight to RGB. For a 16-bit or
    CMYK source that throws information away before the encoder runs, and the
    round trip would then be reported as lossless while not being lossless at
    all. Refuse instead, unless explicitly overridden.
    """
    img = Image.open(path)
    mode = img.mode
    if mode in CONVERTIBLE:
        img = img.convert(CONVERTIBLE[mode])
    elif mode not in SAFE_MODES:
        msg = ("image mode %r has more than 8 bits per sample or a non-RGB "
               "colour space; converting it would lose data before the codec "
               "sees it. This codec is 8-bit only." % mode)
        if not force:
            die(msg + "\nPass --force to convert anyway (the result will NOT "
                      "be lossless with respect to the original).")
        warn(msg + " Converting anyway because --force was given.")
        img = img.convert("RGBA" if "A" in mode else "RGB")
    return np.array(img), mode


def probe(path):
    """Width, height, pixel format and frame count of a video's first stream."""
    need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,pix_fmt,nb_frames,r_frame_rate",
         path], capture_output=True, text=True)
    if out.returncode != 0:
        die("ffprobe could not read %s\n%s" % (path, out.stderr.strip()))
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        die("no video stream found in %s" % path)
    return streams[0]


def to_y4m(path, dest, frames=None):
    """Transcode any container to the 8-bit 4:2:0 y4m the codec reads.

    Returns a note describing what the conversion cost, or None if it was
    exact. This is the step people forget: the codec being lossless says
    nothing about a pipeline that resampled the pixels on the way in.
    """
    need("ffmpeg")
    info = probe(path)
    pix = info.get("pix_fmt", "?")
    lossy = None
    if pix != "yuv420p":
        lossy = ("source pixel format is %s; it is being converted to yuv420p "
                 "for the codec. That conversion is lossy, so the restored "
                 "video will match the converted source exactly but not the "
                 "original file." % pix)

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", path]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", dest]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        die("ffmpeg failed converting %s\n%s" % (path, res.stderr.strip()))
    return lossy


def read_y4m(path, limit=None):
    reader = y4m.Y4M(path)
    frames = [[p.copy() for p in f] for f in reader.frames(limit=limit)]
    reader.close()
    if not frames:
        die("no frames read from %s" % path)
    return frames, reader


def estimate(pixels_total):
    secs = pixels_total / 1e6 / MP_PER_S_ENCODE
    if secs > 60:
        warn("this looks like about %.0f minutes of encoding. Use --frames to "
             "do less of it." % (secs / 60.0))
    return secs


# --------------------------------------------------------------------------
# commands


def cmd_compress(args):
    kind = kind_of(args.input)
    if kind in ("hvi", "hvv"):
        die("%s is already compressed. Use `restore` or `check`." % args.input)
    if kind is None:
        die("do not recognise %s as an image or video." % args.input)

    on_disk = os.path.getsize(args.input)
    if kind == "image":
        arr, mode = load_image(args.input, args.force)
        out = args.output or os.path.splitext(args.input)[0] + ".hvi"
        estimate(arr.shape[0] * arr.shape[1])
        t = time.time()
        blob = hve_image.encode(arr)
        dt = time.time() - t
        raw = arr.size
        print("%s  %dx%d %s" % (os.path.basename(args.input),
                                arr.shape[1], arr.shape[0], mode))
    else:
        out = args.output or os.path.splitext(args.input)[0] + ".hvv"
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.y4m")
            note = (None if args.input.lower().endswith(".y4m")
                    else to_y4m(args.input, src, args.frames or None))
            if args.input.lower().endswith(".y4m"):
                src = args.input
            if note:
                warn(note)
            frames, reader = read_y4m(src, args.frames or None)
            raw = sum(p.size for f in frames for p in f)
            estimate(raw)
            print("%s  %dx%d  %d frames" % (os.path.basename(args.input),
                                            reader.width, reader.height,
                                            len(frames)))
            t = time.time()
            blob = hve_video.encode(frames)
            dt = time.time() - t

    with open(out, "wb") as fh:
        fh.write(blob)
    report_sizes(args.input, raw, on_disk, len(blob),
                 "encoded in %.1fs -> %s" % (dt, out), kind)


def cmd_restore(args):
    with open(args.input, "rb") as fh:
        blob = fh.read()
    t = time.time()
    if blob[:4] == hve_image.MAGIC:
        arr = hve_image.decode(blob)
        out = args.output or os.path.splitext(args.input)[0] + "_restored.png"
        Image.fromarray(arr).save(out)
    elif blob[:4] == hve_video.MAGIC:
        frames = hve_video.decode(blob)
        out = args.output or os.path.splitext(args.input)[0] + "_restored.mkv"
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "out.y4m")
            h, w = frames[0][0].shape
            y4m.write(raw, frames, w, h)
            if out.lower().endswith(".y4m"):
                shutil.copy(raw, out)
            else:
                need("ffmpeg")
                # FFV1 so the file you can actually open in VLC is still
                # mathematically identical to what the codec produced.
                res = subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", raw,
                     "-c:v", "ffv1", "-level", "3", out],
                    capture_output=True, text=True)
                if res.returncode != 0:
                    die("ffmpeg failed writing %s\n%s" % (out, res.stderr.strip()))
    else:
        die("%s is not a .hvi or .hvv file." % args.input)
    print("restored in %.1fs -> %s" % (time.time() - t, out))
    print("note: pixels are restored exactly; metadata such as EXIF is not "
          "carried through the codec.")


def cmd_check(args):
    """Compress, restore, and compare every pixel. The claim, verified."""
    kind = kind_of(args.input)
    if kind is None:
        die("do not recognise %s as an image or video." % args.input)
    on_disk = os.path.getsize(args.input)

    if kind == "image":
        arr, mode = load_image(args.input, args.force)
        raw = arr.size
        print("%s  %dx%d %s" % (os.path.basename(args.input),
                                arr.shape[1], arr.shape[0], mode))
        estimate(arr.shape[0] * arr.shape[1])
        t = time.time(); blob = hve_image.encode(arr); enc = time.time() - t
        t = time.time(); back = hve_image.decode(blob); dec = time.time() - t
        identical = np.array_equal(arr, back)
        worst = int(np.abs(arr.astype(int) - back.astype(int)).max())
    elif kind in ("video",):
        with tempfile.TemporaryDirectory() as tmp:
            src = args.input
            note = None
            if not args.input.lower().endswith(".y4m"):
                src = os.path.join(tmp, "in.y4m")
                note = to_y4m(args.input, src, args.frames or None)
            if note:
                warn(note)
            frames, reader = read_y4m(src, args.frames or None)
            raw = sum(p.size for f in frames for p in f)
            print("%s  %dx%d  %d frames" % (os.path.basename(args.input),
                                            reader.width, reader.height,
                                            len(frames)))
            estimate(raw)
            t = time.time(); blob = hve_video.encode(frames); enc = time.time() - t
            t = time.time(); back = hve_video.decode(blob); dec = time.time() - t
            identical = (len(frames) == len(back) and
                         all(np.array_equal(a, b)
                             for fa, fb in zip(frames, back)
                             for a, b in zip(fa, fb)))
            worst = 0 if identical else 255
    else:
        die("check works on ordinary images and videos, not on %s." % args.input)

    report_sizes(args.input, raw, on_disk, len(blob),
                 "encode %.1fs   decode %.1fs" % (enc, dec), kind)
    print()
    if identical:
        print("  LOSSLESS: every pixel of the round trip is identical.")
    else:
        print("  NOT LOSSLESS: max absolute difference %d. This is a bug - "
              "please report it with the input file." % worst)
        raise SystemExit(2)


def cmd_compare(args):
    """hve against the codecs it is trying to beat, on your file."""
    kind = kind_of(args.input)
    if kind != "image":
        die("compare currently only handles still images.")
    arr, _ = load_image(args.input, args.force)
    if arr.ndim == 3 and arr.shape[2] == 4:
        die("compare needs an image without an alpha channel.")
    rgb = arr if arr.ndim == 2 else arr[:, :, :3]

    rows = []
    t = time.time()
    blob = hve_image.encode(arr)
    rows.append(("hve", len(blob), time.time() - t))

    import io
    for name, fmt, kw in [("PNG (optimised)", "PNG", dict(optimize=True)),
                          ("WebP lossless", "WEBP", dict(lossless=True,
                                                         quality=100,
                                                         method=6))]:
        try:
            buf = io.BytesIO()
            t = time.time()
            Image.fromarray(rgb).save(buf, format=fmt, **kw)
            rows.append((name, len(buf.getvalue()), time.time() - t))
        except Exception as exc:                       # noqa: BLE001
            warn("%s skipped: %s" % (name, exc))

    try:
        import imagecodecs
        for name, effort in [("JPEG XL e7", 7), ("JPEG XL e9", 9)]:
            t = time.time()
            enc = imagecodecs.jpegxl_encode(rgb, lossless=True, effort=effort)
            rows.append((name, len(enc), time.time() - t))
    except ImportError:
        warn("imagecodecs not installed, skipping JPEG XL "
             "(pip install imagecodecs)")
    except Exception as exc:                           # noqa: BLE001
        warn("JPEG XL skipped: %s" % exc)

    raw = rgb.size
    print("\n%s  %dx%d" % (os.path.basename(args.input),
                           rgb.shape[1], rgb.shape[0]))
    print("%-18s %12s %8s %9s" % ("codec", "bytes", "ratio", "enc s"))
    for name, size, secs in sorted(rows, key=lambda r: r[1]):
        mark = " <-" if name == "hve" else ""
        print("%-18s %12d %7.2fx %9.1f%s" % (name, size, raw / size, secs, mark))


def cmd_demo(args):
    """Round-trip a generated sample, so the tool works before any download."""
    out = os.path.join(ROOT, "playground", "demo_output")
    os.makedirs(out, exist_ok=True)
    src_png, pil, is_real = _demo_source(out)
    img = np.array(pil)

    print("\n--- image ---")
    args.input, args.force, args.frames = src_png, False, 0
    cmd_check(args)
    cmd_compare(args)
    if not is_real:
        print("\n  Take that table as a smoke test, not a result. Synthetic "
              "images are easy to")
        print("  tilt either way - an earlier version of this sample put hve "
              "*behind* PNG for")
        print("  equally arbitrary reasons. For numbers worth quoting:")
        print("      .venv/bin/python tools/fetch_testdata.py")
        print("      .venv/bin/python tools/bench_image.py --jobs=8 test")

    if shutil.which("ffmpeg"):
        print("\n--- video (the same image panned for 24 frames) ---")
        src_y4m = os.path.join(out, "sample.y4m")
        _sample_video(img, src_y4m, frames=24)
        args.input, args.frames = src_y4m, 0
        cmd_check(args)
    else:
        print("\n(skipping the video demo: ffmpeg is not installed)")

    print("\nSample files are in %s" % out)


def _sample_image(width=384, height=256):
    """Something photo-like, with no download required.

    Getting this wrong is easy and misleading. The first version of this
    function drew mathematical gradients and a block of uniform random noise,
    and hve came out *behind* PNG on it — not because the codec is bad but
    because neither regime resembles a photograph. Exact linear ramps are the
    one thing PNG's filters handle perfectly, and uniform noise is
    incompressible for everyone while still charging a context model for
    trying. A sample that flatters or maligns the codec is equally useless.

    So this builds something with a photograph's statistics instead: smooth
    low-frequency colour fields (correlated, not linear), a few hard edges, and
    fine grain on top. The small tiled patch stays, because repetition is the
    one place the match model does something no gradient predictor can.
    """
    rng = np.random.default_rng(7)

    # Low-frequency colour field: random coarse grid, smoothed. Photographs are
    # mostly this - locally smooth, globally unpredictable.
    coarse = rng.integers(40, 216, size=(6, 8, 3)).astype(np.float64)
    field = np.repeat(np.repeat(coarse, height // 6 + 1, 0), width // 8 + 1, 1)
    field = field[:height, :width]
    for _ in range(3):                       # cheap box blur, numpy only
        pad = np.pad(field, ((8, 8), (8, 8), (0, 0)), mode="edge")
        field = sum(pad[i:i + height, j:j + width]
                    for i in (0, 8, 16) for j in (0, 8, 16)) / 9.0
    img = field

    # A few edges, since prediction behaves completely differently across them.
    yy, xx = np.mgrid[0:height, 0:width]
    disc = ((xx - width * 0.62) ** 2 + (yy - height * 0.4) ** 2) < (height * 0.22) ** 2
    img[disc] = img[disc] * 0.55 + 90
    img[int(height * 0.72):int(height * 0.78), :] *= 0.7

    # Grain. Real sensors put a little uncorrelated noise on everything, and it
    # dominates what a lossless codec actually spends its bits on.
    img += rng.normal(0.0, 3.0, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # One repetitive patch, small enough not to distort the overall result.
    tile = rng.integers(0, 256, size=(16, 16, 3)).astype(np.uint8)
    th, tw = height // 5, width // 5
    reps = np.tile(tile, (th // 16 + 1, tw // 16 + 1, 1))
    img[-th:, -tw:] = reps[:th, :tw]
    return img


def _demo_source(out):
    """Prefer a real photograph if the test corpus was fetched.

    A generated sample is honest but synthetic, and the interesting question is
    always "what does it do on a real photo". If testdata/ is present, use it.
    """
    kodak = os.path.join(ROOT, "testdata", "images", "kodim23.png")
    if os.path.exists(kodak):
        print("Using a real photograph from testdata/ (kodim23.png).")
        return kodak, Image.open(kodak).convert("RGB"), True
    print("Building a photo-like sample image "
          "(no testdata/ found - run tools/fetch_testdata.py for real photos).")
    img = _sample_image()
    path = os.path.join(out, "sample.png")
    Image.fromarray(img).save(path, optimize=True)
    return path, Image.fromarray(img), False


def _sample_video(img, dest, frames=24):
    """Pan the sample image, which gives the motion search something real."""
    h, w = img.shape[:2]
    ch, cw = h - 32, w - 32
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(frames):
            dx = (i * 2) % 32
            dy = (i * 1) % 32
            crop = img[dy:dy + ch, dx:dx + cw]
            Image.fromarray(crop).save(os.path.join(tmp, "f%03d.png" % i))
        res = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", "25",
             "-i", os.path.join(tmp, "f%03d.png"),
             "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", dest],
            capture_output=True, text=True)
        if res.returncode != 0:
            die("ffmpeg failed building the sample video\n%s" % res.stderr.strip())


def cmd_info(args):
    from hve.bitio import Reader
    with open(args.input, "rb") as fh:
        blob = fh.read()
    r = Reader(blob)
    magic = r.raw(4)
    if magic == hve_image.MAGIC:
        width, height = r.varint(), r.varint()
        channels, flags = r.u8(), r.u8()
        print("hve image  %dx%d  %d channels  rct=%d  %s  %.3f bpp"
              % (width, height, channels, flags & hve_image.FLAG_RCT,
                 human(len(blob)), len(blob) * 8.0 / (width * height)))
    elif magic == hve_video.MAGIC:
        nframes, nplanes, flags, block = r.varint(), r.u8(), r.u8(), r.u8()
        shapes = ["%dx%d" % (r.varint(), r.varint()) for _ in range(nplanes)]
        print("hve video  %d frames  %d planes (%s)  block=%d  rct=%d  %s"
              % (nframes, nplanes, ", ".join(shapes), block,
                 flags & hve_video.FLAG_RCT, human(len(blob))))
    else:
        die("%s is not a .hvi or .hvv file." % args.input)


# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="./playground/hve",
        description="Experiment with the hve codec on ordinary image and "
                    "video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  ./playground/hve demo                       generated sample, no downloads
  ./playground/hve check photo.jpg            round trip and verify every pixel
  ./playground/hve compare photo.jpg          hve vs PNG, WebP and JPEG XL
  ./playground/hve compress clip.mp4          -> clip.hvv (60 frames by default)
  ./playground/hve restore clip.hvv           -> clip_restored.mkv, plays in VLC
""")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, frames=False, force=True):
        s = sub.add_parser(name, help=help_)
        s.add_argument("input", nargs="?" if name == "demo" else None)
        if name in ("compress", "restore"):
            s.add_argument("-o", "--output")
        if frames:
            s.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                           help="video frames to use (0 = all). Default %d, "
                                "because encoding is slow enough that a whole "
                                "clip is rarely what you wanted first."
                                % DEFAULT_FRAMES)
        if force:
            s.add_argument("--force", action="store_true",
                           help="convert image modes the codec cannot store "
                                "exactly (the result will not be lossless)")
        s.set_defaults(func=fn)
        return s

    add("demo", cmd_demo, "round-trip a generated sample (start here)",
        frames=False, force=False)
    add("check", cmd_check, "compress, restore and verify every pixel",
        frames=True)
    add("compare", cmd_compare, "size against PNG, WebP and JPEG XL")
    add("compress", cmd_compress, "an image or video -> .hvi / .hvv", frames=True)
    add("restore", cmd_restore, ".hvi / .hvv -> .png / .mkv", force=False)
    add("info", cmd_info, "describe a .hvi or .hvv file", force=False)

    args = p.parse_args(argv)
    if not native.available() and not fast.available():
        warn("no C compiler and no numba, so this will run roughly 30x slower. "
             "Output is byte-identical either way. Run ./playground/setup.sh")
    elif not native.available():
        warn("no C compiler found, so the numba path is running instead "
             "(2-4x slower, byte-identical output). Reason: %s"
             % native.load_error())
    args.func(args)


if __name__ == "__main__":
    main()
