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


VIDEO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "testdata", "video")

# Same split, same reason. bus/mobile/container are for tuning the motion model;
# akiyo and foreman carry the published numbers and nothing is fitted to them.
VIDEO_DEV = ["bus_cif.y4m", "mobile_cif.y4m", "container_cif.y4m"]
VIDEO_TEST = ["akiyo_cif.y4m", "foreman_cif.y4m"]


def _video_paths(names):
    out = [os.path.join(VIDEO_ROOT, n) for n in names]
    return [p for p in out if os.path.exists(p)]


def video_dev():
    return _video_paths(VIDEO_DEV)


def video_test():
    return _video_paths(VIDEO_TEST)


def describe():
    return ("dev=%d images, held-out test=%d images; "
            "dev=%d clips, held-out test=%d clips"
            % (len(dev()), len(test()), len(video_dev()), len(video_test())))


if __name__ == "__main__":
    print(describe())
    print("dev :", " ".join(os.path.basename(p) for p in dev()))
    print("test:", " ".join(os.path.basename(p) for p in test()))
