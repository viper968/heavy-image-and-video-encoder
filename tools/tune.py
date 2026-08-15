"""Parameter sweep for the adaptive model, measured as real encoded bytes."""

import itertools
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from hve import image, model, rc                     # noqa: E402

SWEEP = {
    "adapt": [4, 5, 6],
    "act": ["a16", "a24", "a12"],
    "err": ["e4", "e6"],
}

ACT_SETS = {
    "a12": [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 60],
    "a16": [1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64, 88, 120],
    "a24": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40, 49, 60, 73,
            89, 109, 133, 163, 200],
}
ERR_SETS = {"e4": [1, 4, 12], "e6": [1, 2, 4, 8, 16]}


def apply(adapt, act, err):
    rc.ADAPT_SHIFT = adapt
    model.ACT_LADDER = ACT_SETS[act]
    model.ERR_LADDER = ERR_SETS[err]
    model.NACT = len(model.ACT_LADDER) + 1
    model.NERR = len(model.ERR_LADDER) + 1


def main(paths):
    imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
    results = []
    for adapt, act, err in itertools.product(*SWEEP.values()):
        apply(adapt, act, err)
        total = sum(len(image.encode(a)) for a in imgs)
        results.append((total, adapt, act, err))
        print("adapt=%d %s %s -> %d" % (adapt, act, err, total), flush=True)
    print("\nbest:")
    for total, adapt, act, err in sorted(results)[:5]:
        print("  %d  adapt=%d %s %s" % (total, adapt, act, err))


if __name__ == "__main__":
    main(sys.argv[1:])
