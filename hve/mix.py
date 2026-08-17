"""Logistic mixing and secondary estimation - the context-mixing layer.

A single context per binary decision forces one model to be right about
everything. Mixing lets several models each predict the bit and combines them in
the logistic domain, where combination is a weighted sum:

    p_mix = squash( sum_i w_i * stretch(p_i) )

The weights are learned online by gradient descent on coding loss and are
selected by their own small context, so the mixer can learn "in flat areas trust
the cross-channel model, on edges trust the gradient model".

Everything here is integer arithmetic over already-coded data only, so the
decoder reproduces it exactly.

Conventions follow LPAQ so its published constants apply directly: probabilities
are 12-bit (0..4095) and are the probability the bit is **1**; stretched values
are 256*ln(p/(1-p)) clamped to +-2047.
"""

import math

PROB_BITS = 12
PROB_ONE = 1 << PROB_BITS          # 4096
STRETCH_LIMIT = 2047

# squash(x) = 4096 / (1 + e^(-x/256)) over the whole stretched range. LPAQ
# interpolates 33 anchor points; tabulating every value is both exact and
# faster in Python, where the table lookup costs the same either way.
_SQUASH = [0] * (2 * STRETCH_LIMIT + 1)
for _i in range(-STRETCH_LIMIT, STRETCH_LIMIT + 1):
    _v = int(round(PROB_ONE / (1.0 + math.exp(-_i / 256.0))))
    _SQUASH[_i + STRETCH_LIMIT] = min(PROB_ONE - 1, max(1, _v))

# stretch(p), the exact inverse, built by inverting squash the way LPAQ does.
_STRETCH = [0] * PROB_ONE
for _p in range(PROB_ONE):
    _q = min(PROB_ONE - 1, max(1, _p))
    _v = int(round(256.0 * math.log(_q / float(PROB_ONE - _q))))
    _STRETCH[_p] = min(STRETCH_LIMIT, max(-STRETCH_LIMIT, _v))

SQUASH = _SQUASH
STRETCH = _STRETCH


def squash(x):
    if x <= -STRETCH_LIMIT:
        return 1
    if x >= STRETCH_LIMIT:
        return PROB_ONE - 1
    return _SQUASH[x + STRETCH_LIMIT]


def stretch(p):
    return _STRETCH[p]


WEIGHT_ONE = 1 << 16               # 16.16 fixed point; this weight passes an input through


class Mixer:
    """Gradient-descent logistic mixer with context-selected weight sets.

    The learning rate is 24, not LPAQ's 7. That 7 was inherited along with the
    rest of the design and never checked against this model; swept on the dev
    split it is monotonically better out to about 24-32 for both presets, and
    the win transferred to the held-out split (-0.086% on stills, -0.043% on
    video) rather than being a dev-set artefact.

    `n` inputs, `nctx` independent weight vectors, weights in 16.16 fixed point.

    LPAQ starts every weight at zero and lets gradient descent find them. Here
    the first expert is the established context model and the others are
    additions, so expert 0 starts at unity and the rest at zero: the mixer opens
    as an exact copy of the model it is replacing and can only improve from
    there, instead of spending the first few thousand pixels relearning it.
    """

    def __init__(self, n, nctx, rate=24):
        self.n = n
        self.rate = rate
        weights = [0] * (n * nctx)
        for c in range(nctx):
            weights[c * n] = WEIGHT_ONE
        self.weights = weights
        self.inputs = [0] * n
        self.base = 0
        self.pr = PROB_ONE >> 1

    def mix(self, inputs, ctx):
        """`inputs` are stretched probabilities; returns a 12-bit P(bit=1)."""
        base = ctx * self.n
        w = self.weights
        total = 0
        for i in range(self.n):
            total += inputs[i] * w[base + i]
        self.inputs = inputs
        self.base = base
        pr = squash(total >> 16)
        self.pr = pr
        return pr

    def update(self, bit):
        """LPAQ's rule: err = ((y<<12) - pr) * rate, w_i += (t_i*err + 0x8000) >> 16."""
        err = ((bit << PROB_BITS) - self.pr) * self.rate
        w = self.weights
        base = self.base
        inputs = self.inputs
        for i in range(self.n):
            w[base + i] += (inputs[i] * err + 0x8000) >> 16


class APM:
    """Adaptive probability map, a.k.a. secondary symbol estimation.

    Rate 8 rather than LPAQ's 7, for the same reason as Mixer above: swept on
    the dev split it is a shallow optimum at 8-9 for both presets, and the win
    held up on the held-out split (-0.034% on stills, -0.043% on video).

    Takes a probability the model already believes and refines it by what
    actually happened the last times this (context, probability bucket) pair came
    up. Buckets are spaced evenly on the stretch scale and adjacent ones are
    interpolated, so the correction stays smooth; only the nearer of the two is
    updated.
    """

    BUCKETS = 33                   # 32 intervals of 128 stretch units, plus an end point

    def __init__(self, nctx, rate=8):
        self.rate = rate
        table = []
        for _ in range(nctx):
            for j in range(self.BUCKETS):
                table.append(squash((j - 16) * 128) << 4)     # identity map, 16-bit
        self.table = table
        self.index = 0

    def refine(self, pr, ctx):
        s = _STRETCH[pr] + 2048                               # 1..4095
        w = s & 127
        idx = ctx * self.BUCKETS + (s >> 7)
        self.index = idx + (1 if w >= 64 else 0)
        table = self.table
        return (table[idx] * (128 - w) + table[idx + 1] * w) >> 11

    def update(self, bit):
        target = 65535 if bit else 0
        idx = self.index
        table = self.table
        table[idx] += (target - table[idx]) >> self.rate
