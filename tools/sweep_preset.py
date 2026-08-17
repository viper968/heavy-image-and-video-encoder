"""Re-sweep a model constant, separately for each preset.

Every constant in this codec was fitted with the full model switched on and one
slice. The `fast` preset drops three stages, which changes what the remaining
ones see, so a constant that was optimal before need not be now. This rebuilds
the C binary for each candidate value and measures the dev split.

    .venv/bin/python tools/sweep_preset.py rc ADAPT_SHIFT 4 5 6 7 8

Uses the dev corpus only: the held-out images and clips must not be tuned on.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import corpus  # noqa: E402

BINARY = os.path.join(ROOT, "build", "hve")
SCRATCH = os.environ.get("HVE_SCRATCH", "/tmp")


def set_constant(module, name, value):
    path = os.path.join(ROOT, "hve", module + ".py")
    with open(path) as fh:
        text = fh.read()
    pattern = re.compile(r"^(%s\s*=\s*)(-?\d+)" % re.escape(name), re.M)
    if not pattern.search(text):
        raise SystemExit("no constant %s in hve/%s.py" % (name, module))
    with open(path, "w") as fh:
        fh.write(pattern.sub(lambda m: m.group(1) + str(value), text, count=1))


def read_constant(module, name):
    path = os.path.join(ROOT, "hve", module + ".py")
    with open(path) as fh:
        m = re.search(r"^%s\s*=\s*(-?\d+)" % re.escape(name), fh.read(), re.M)
    return int(m.group(1))


def rebuild():
    subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                 "gen_model_constants.py")],
                   capture_output=True, check=True)
    r = subprocess.run(["make", "-s", "-C", os.path.join(ROOT, "csrc")],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode()[-500:])


def measure(preset):
    """Total dev-split bytes at one preset. Single slice, so this measures the
    model and not the slice count."""
    total = 0
    for path in corpus.dev():
        out = os.path.join(SCRATCH, "sweep.hvi")
        subprocess.run([BINARY, "encode", path, out, "--preset", preset,
                        "--slices", "1"], capture_output=True, check=True)
        total += os.path.getsize(out)
    for path in corpus.video_dev():
        out = os.path.join(SCRATCH, "sweep.hvv")
        subprocess.run([BINARY, "encode", path, out, "--frames", "8",
                        "--preset", preset, "--slices", "1"],
                       capture_output=True, check=True)
        total += os.path.getsize(out)
    return total


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    module, name, values = argv[0], argv[1], [int(v) for v in argv[2:]]
    original = read_constant(module, name)
    print("sweeping %s.%s (currently %d)\n" % (module, name, original))
    print("%8s %14s %14s" % ("value", "max", "fast"))
    results = {}
    try:
        for v in values:
            set_constant(module, name, v)
            rebuild()
            row = {p: measure(p) for p in ("max", "fast")}
            results[v] = row
            print("%8d %14d %14d" % (v, row["max"], row["fast"]), flush=True)
    finally:
        set_constant(module, name, original)
        rebuild()
    for preset in ("max", "fast"):
        best = min(results, key=lambda v: results[v][preset])
        base = results.get(original, {}).get(preset)
        note = ""
        if base:
            note = "  (%+.3f%% vs current)" % (100.0 * (results[best][preset] - base) / base)
        print("\nbest for %-4s: %s = %d%s" % (preset, name, best, note))


if __name__ == "__main__":
    main(sys.argv[1:])
