"""Command line interface: python -m hve encode|decode|info IN OUT."""

import argparse
import os
import sys
import time

import numpy as np

from . import image, model, video, y4m

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".webp", ".gif"}


def _load_image(path):
    from PIL import Image
    img = Image.open(path)
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
    return np.array(img)


def _save_image(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


def cmd_encode(args):
    src_ext = os.path.splitext(args.input)[1].lower()
    start = time.time()
    features = model.FEAT_ALL if args.preset == "max" else PRESET_FAST
    if src_ext == ".y4m":
        reader = y4m.Y4M(args.input)
        frames = [[p.copy() for p in f] for f in reader.frames(limit=args.frames)]
        reader.close()
        blob = video.encode(frames, features=features,
                            progress=_progress(len(frames)) if args.verbose else None)
        original = sum(p.size for f in frames for p in f)
    elif src_ext in IMAGE_EXT:
        arr = _load_image(args.input)
        blob = image.encode(arr, features=features)
        original = arr.size
    else:
        raise SystemExit("unsupported input type: %s" % src_ext)

    with open(args.output, "wb") as fh:
        fh.write(blob)
    elapsed = time.time() - start
    print("%s -> %s  %d -> %d bytes (%.2fx, %.1fs)"
          % (args.input, args.output, original, len(blob), original / len(blob), elapsed))
    if os.path.exists(args.input) and src_ext != ".y4m":
        on_disk = os.path.getsize(args.input)
        print("  source file on disk was %d bytes (%.2fx smaller now)"
              % (on_disk, on_disk / len(blob)))


def cmd_decode(args):
    with open(args.input, "rb") as fh:
        blob = fh.read()
    start = time.time()
    if blob[:4] == video.MAGIC:
        frames = video.decode(blob)
        planes = frames[0]
        if isinstance(planes, np.ndarray):
            raise SystemExit("RGB video output is not supported by the .y4m writer")
        y4m.write(args.output, frames, planes[0].shape[1], planes[0].shape[0])
    elif blob[:4] == image.MAGIC:
        _save_image(image.decode(blob), args.output)
    else:
        raise SystemExit("unrecognised container")
    print("%s -> %s (%.1fs)" % (args.input, args.output, time.time() - start))


def cmd_info(args):
    with open(args.input, "rb") as fh:
        blob = fh.read()
    from .bitio import Reader
    r = Reader(blob)
    magic = r.raw(4)
    if magic == image.MAGIC:
        width, height = r.varint(), r.varint()
        channels, flags, features = r.u8(), r.u8(), r.u8()
        print("hve image  %dx%d  %d channels  rct=%d  preset %s  %d bytes  %.3f bpp"
              % (width, height, channels, flags & image.FLAG_RCT,
                 _preset_name(features), len(blob),
                 len(blob) * 8.0 / (width * height)))
    elif magic == video.MAGIC:
        nframes, nplanes, flags, block = r.varint(), r.u8(), r.u8(), r.u8()
        features = r.u8()
        shapes = ["%dx%d" % (r.varint(), r.varint()) for _ in range(nplanes)]
        print("hve video  %d frames  %d planes (%s)  block=%d  rct=%d  "
              "preset %s  %d bytes"
              % (nframes, nplanes, ", ".join(shapes), block,
                 flags & video.FLAG_RCT, _preset_name(features), len(blob)))
    else:
        raise SystemExit("unrecognised container")


# Keep this in step with csrc/main.c's PRESET_FAST, which carries the reasoning.
# tests/test_cli_binary.py imports this and checks the binary agrees, so the two
# cannot drift silently.
PRESET_FAST = model.FEAT_ALL & ~(model.FEAT_MATCH | model.FEAT_LMS
                                 | model.FEAT_BLEND | model.FEAT_APM1
                                 | model.FEAT_APM2)


def _preset_name(features):
    if features == model.FEAT_ALL:
        return "max"
    if features == PRESET_FAST:
        return "fast"
    return "custom(0x%02x)" % features


def _progress(total):
    def report(i, _n):
        sys.stderr.write("\rframe %d/%d" % (i + 1, total))
        sys.stderr.flush()
        if i + 1 == total:
            sys.stderr.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hve", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="compress an image or .y4m video")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--frames", type=int, default=None, help="limit video frames")
    enc.add_argument("-v", "--verbose", action="store_true")
    enc.add_argument("--preset", choices=("max", "fast"), default="max",
                     help="model stages to run; recorded in the file")
    enc.set_defaults(func=cmd_encode)

    dec = sub.add_parser("decode", help="restore the original bytes")
    dec.add_argument("input")
    dec.add_argument("output")
    dec.set_defaults(func=cmd_decode)

    info = sub.add_parser("info", help="describe a compressed file")
    info.add_argument("input")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
