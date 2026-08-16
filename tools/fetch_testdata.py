"""Download the test corpus: Kodak photos and Xiph video clips.

The data is not committed - it is several tens of megabytes of other people's
material. This pulls the exact files the published benchmark numbers used.
"""

import os
import sys
import urllib.request

# The whole Kodak set: tools/corpus.py splits it into six dev images and
# eighteen held-out ones, and the headline numbers come from the held-out half.
KODAK = ["%02d" % i for i in range(1, 25)]
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim%s.png"
# The two clips the published video numbers are measured on. They are the
# held-out set: nothing is tuned against them.
CLIPS = {
    "akiyo_cif.y4m": "https://media.xiph.org/video/derf/y4m/akiyo_cif.y4m",
    "foreman_cif.y4m": "https://media.xiph.org/video/derf/y4m/foreman_cif.y4m",
}

# Development clips, for tuning the motion model against. Chosen to span the
# motion regimes the two held-out clips do not both cover, because a motion
# model fitted to one kind of movement is easy to build by accident:
#   bus       - fast horizontal pan plus independent object motion
#   mobile    - slow pan over dense texture, the classic hard case for a
#               residual coder
#   container - near-static with one slow-moving object
# Without these there would be two video clips in the whole project and both
# carry the headline numbers, so every measurement would be fitted to the thing
# it reports. tools/corpus.py keeps the split honest for images; this is the
# same idea for video.
DEV_CLIPS = {
    "bus_cif.y4m": "https://media.xiph.org/video/derf/y4m/bus_cif.y4m",
    "mobile_cif.y4m": "https://media.xiph.org/video/derf/y4m/mobile_cif.y4m",
    "container_cif.y4m": "https://media.xiph.org/video/derf/y4m/container_cif.y4m",
}

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata")


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print("have %s" % os.path.basename(path))
        return
    print("get  %s" % os.path.basename(path), flush=True)
    urllib.request.urlretrieve(url, path)


def main():
    images = os.path.join(ROOT, "images")
    video = os.path.join(ROOT, "video")
    os.makedirs(images, exist_ok=True)
    os.makedirs(video, exist_ok=True)
    for num in KODAK:
        fetch(KODAK_URL % num, os.path.join(images, "kodim%s.png" % num))
    for name, url in CLIPS.items():
        fetch(url, os.path.join(video, name))
    if "--no-dev-clips" not in sys.argv:
        for name, url in DEV_CLIPS.items():
            fetch(url, os.path.join(video, name))
    print("\ntestdata ready in %s" % ROOT)


if __name__ == "__main__":
    sys.exit(main())
