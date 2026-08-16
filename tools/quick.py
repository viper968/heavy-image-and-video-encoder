"""Fast dev-set size measurement, for A/B-ing model changes.

Encodes only — decode adds nothing to a size comparison — so a full dev-set
reading takes a couple of seconds with the jitted path. Pass `test` to score the
held-out split instead, but only for a final number: tuning against it would
defeat the point of holding it out.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus                                          # noqa: E402
from hve import image                                  # noqa: E402


def measure(paths, verbose=True):
    total = 0
    rows = []
    start = time.time()
    for path in paths:
        arr = np.array(Image.open(path).convert("RGB"))
        n = len(image.encode(arr))
        rows.append((os.path.basename(path), n))
        total += n
    if verbose:
        for name, n in rows:
            print("  %-16s %9d" % (name, n))
        print("total %d   (%.1fs)" % (total, time.time() - start))
    return total


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    paths = {"dev": corpus.dev, "test": corpus.test,
             "all": lambda: corpus.dev() + corpus.test()}[which]()
    print("split: %s, %d images" % (which, len(paths)))
    measure(paths)
