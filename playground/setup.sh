#!/bin/sh
# One-command setup for the playground. Safe to re-run.
#
# Creates .venv at the repo root (the same one tools/ and the tests expect) and
# installs everything explicitly into it. Deliberately does NOT use
# --system-site-packages: inheriting the host's numpy and Pillow works on the
# machine that happens to have them and breaks everywhere else.
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

PY=${PYTHON:-python3}
echo "Using $PY ($($PY --version 2>&1))"

if [ ! -d .venv ]; then
    echo "Creating .venv ..."
    "$PY" -m venv .venv
else
    echo ".venv already exists, reusing it"
fi

echo "Installing dependencies ..."
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r playground/requirements.txt

echo
echo "Checking the install ..."
./.venv/bin/python - <<'PY'
import sys
ok = True

import numpy, PIL
print("  numpy   ", numpy.__version__)
print("  Pillow  ", PIL.__version__)

try:
    import numba
    print("  numba   ", numba.__version__)
except ImportError:
    ok = False
    print("  numba    MISSING - the codec will still work but ~30x slower")

sys.path.insert(0, ".")
from hve import native
if native.available():
    print("  C backend built (fastest path enabled)")
else:
    print("  C backend UNAVAILABLE - falling back to numba.")
    print("           reason:", native.load_error())

try:
    import imagecodecs
    print("  imagecodecs", imagecodecs.__version__)
except ImportError:
    print("  imagecodecs MISSING - `hve compare` will skip JPEG XL/WebP")

import shutil
for tool in ("ffmpeg", "ffprobe"):
    path = shutil.which(tool)
    print("  %-8s %s" % (tool, path or "MISSING - video support needs this"))

sys.exit(0 if ok else 0)
PY

echo
echo "Done. Try:"
echo "    ./playground/hve demo"
echo "    ./playground/hve check /path/to/your/photo.jpg"
