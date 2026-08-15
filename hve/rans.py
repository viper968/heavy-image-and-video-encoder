"""Static rANS entropy coder with per-context frequency tables.

rANS (range Asymmetric Numeral Systems) reaches the entropy of the model it is
given, unlike Huffman which is stuck at whole numbers of bits per symbol. That
difference is the whole ballgame for prediction residuals, where the most likely
symbol often deserves well under one bit.

Layout: 12-bit probabilities, 32-bit state, byte-at-a-time renormalisation.
The encoder runs backwards over the symbol stream and the decoder runs forwards,
which is what lets a *static* table be transmitted once in the header.
"""

import numpy as np

from .bitio import Reader, Writer

PROB_BITS = 12
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23          # lower bound of the normalised interval
RENORM_SHIFT = 19         # (RANS_L >> PROB_BITS) << 8  ==  1 << 19


def normalise(hist, scale=PROB_SCALE):
    """Scale a histogram to sum to exactly `scale`, keeping every used symbol.

    Any symbol that actually occurs must keep a non-zero frequency or it becomes
    impossible to code, so counts are floored at 1 and the rounding error is
    repaid by the most frequent symbols.
    """
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if total == 0:
        return None
    used = hist > 0
    freqs = np.zeros(len(hist), dtype=np.int64)
    freqs[used] = np.maximum(1, np.round(hist[used] * scale / total).astype(np.int64))

    diff = scale - int(freqs.sum())
    if diff > 0:
        freqs[int(np.argmax(freqs))] += diff
    elif diff < 0:
        # Take back from the largest frequencies, never dropping one to zero.
        for idx in np.argsort(-freqs):
            if diff == 0:
                break
            if freqs[idx] > 1:
                take = min(int(freqs[idx]) - 1, -diff)
                freqs[idx] -= take
                diff += take
        if diff != 0:  # only reachable if the alphabet exceeds `scale`
            raise ValueError("cannot normalise histogram to %d" % scale)
    return freqs.astype(np.int32)


def cumulative(freqs):
    cum = np.zeros(len(freqs) + 1, dtype=np.int32)
    np.cumsum(freqs, out=cum[1:])
    return cum[:-1]


def slot_table(freqs):
    """slot -> symbol lookup of size PROB_SCALE, for the decoder."""
    return bytes(np.repeat(np.arange(len(freqs), dtype=np.uint8), freqs))


def estimate_bits(hist, freqs):
    """Exact rANS cost (to well under a byte) of coding `hist` with `freqs`."""
    hist = np.asarray(hist, dtype=np.float64)
    used = hist > 0
    p = freqs[used] / PROB_SCALE
    return float(-(hist[used] * np.log2(p)).sum())


# --------------------------------------------------------------------------
# table serialisation


def write_table(w: Writer, freqs):
    """Sparse encoding: only symbols that occur cost anything."""
    if freqs is None:
        w.varint(0)
        return
    nz = np.nonzero(freqs)[0]
    w.varint(len(nz))
    prev = 0
    for i in nz:
        i = int(i)
        w.varint(i - prev)
        w.varint(int(freqs[i]) - 1)
        prev = i + 1


def table_size(freqs):
    """Byte cost of write_table without building it."""
    if freqs is None:
        return 1
    w = Writer()
    write_table(w, freqs)
    return len(w.buf)


def read_table(r: Reader, alphabet):
    n = r.varint()
    if n == 0:
        return None
    freqs = np.zeros(alphabet, dtype=np.int32)
    prev = 0
    for _ in range(n):
        i = prev + r.varint()
        freqs[i] = r.varint() + 1
        prev = i + 1
    return freqs


# --------------------------------------------------------------------------
# coder


class Encoder:
    """Symbols must be pushed in *reverse* stream order."""

    def __init__(self):
        self.out = bytearray()
        self.x = RANS_L

    def encode_flat(self, indices, freq_flat, cum_flat):
        """Encode a whole stream.

        `indices` is a python list of ctx * alphabet + symbol, in forward order;
        `freq_flat` / `cum_flat` are flat python lists indexed the same way.
        Flattening the lookup keeps the hot loop to plain list indexing.
        """
        out = self.out
        append = out.append
        x = self.x
        for i in range(len(indices) - 1, -1, -1):
            idx = indices[i]
            f = freq_flat[idx]
            x_max = f << RENORM_SHIFT
            while x >= x_max:
                append(x & 0xFF)
                x >>= 8
            x = ((x // f) << PROB_BITS) + (x % f) + cum_flat[idx]
        self.x = x

    def encode_one(self, ctx_base, sym, freq_flat, cum_flat):
        idx = ctx_base + sym
        f = freq_flat[idx]
        x = self.x
        x_max = f << RENORM_SHIFT
        while x >= x_max:
            self.out.append(x & 0xFF)
            x >>= 8
        self.x = ((x // f) << PROB_BITS) + (x % f) + cum_flat[idx]

    def finish(self):
        x = self.x
        for _ in range(4):
            self.out.append(x & 0xFF)
            x >>= 8
        return bytes(bytearray(reversed(self.out)))


class Decoder:
    def __init__(self, data, pos=0):
        self.data = data
        self.x = int.from_bytes(data[pos:pos + 4], "big")
        self.pos = pos + 4

    def decode(self, freqs, cums, slots):
        x = self.x
        slot = x & (PROB_SCALE - 1)
        s = slots[slot]
        x = freqs[s] * (x >> PROB_BITS) + slot - cums[s]
        data = self.data
        pos = self.pos
        while x < RANS_L:
            x = (x << 8) | data[pos]
            pos += 1
        self.x = x
        self.pos = pos
        return s
