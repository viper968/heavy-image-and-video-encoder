"""The tuning / held-out split for the Kodak corpus.

Every constant in this codec (adaptation rate, context ladders, mantissa
modelling) was chosen by measuring compressed size, so those measurements are
fitted to whatever images they were run on. Reporting the headline result on the
same images would overstate it. Tuning uses DEV; headline numbers come from
TEST, which no parameter has ever been chosen against.
"""

import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "testdata", "images")

DEV = ["01", "05", "08", "13", "19", "23"]


def _path(num):
    return os.path.join(ROOT, "kodim%s.png" % num)


def dev():
    return [_path(n) for n in DEV if os.path.exists(_path(n))]


def test():
    everything = sorted(glob.glob(os.path.join(ROOT, "kodim*.png")))
    held = set(dev())
    return [p for p in everything if p not in held]


def describe():
    return "dev=%d images, held-out test=%d images" % (len(dev()), len(test()))


if __name__ == "__main__":
    print(describe())
    print("dev :", " ".join(os.path.basename(p) for p in dev()))
    print("test:", " ".join(os.path.basename(p) for p in test()))
