"""Fast video A/B harness, the counterpart to tools/quick.py.

Encodes a fixed prefix of each clip and prints the total. Encode only - decode
adds nothing to a size comparison, and the byte-exactness tests already cover
whether the two implementations agree.

Defaults to the dev clips. Pass `test` for the held-out pair, but only for a
final number: tuning against them would defeat the point of holding them out.
"""

import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus                                          # noqa: E402
from hve import video, y4m                             # noqa: E402

FRAMES = 16


def load(path, frames):
    reader = y4m.Y4M(path)
    out = [[p.copy() for p in f] for f in reader.frames(limit=frames)]
    reader.close()
    return out


def measure(paths, frames=FRAMES, verbose=True):
    total = 0
    start = time.time()
    for path in paths:
        clip = load(path, frames)
        n = len(video.encode(clip))
        total += n
        if verbose:
            print("  %-20s %9d" % (os.path.basename(path), n), flush=True)
    if verbose:
        print("total %d   (%.1fs)" % (total, time.time() - start))
    return total


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else FRAMES
    paths = {"dev": corpus.video_dev, "test": corpus.video_test,
             "all": lambda: corpus.video_dev() + corpus.video_test()}[which]()
    if not paths:
        raise SystemExit("no clips found - run tools/fetch_testdata.py")
    print("split: %s, %d clips, %d frames each" % (which, len(paths), frames))
    measure(paths, frames)
