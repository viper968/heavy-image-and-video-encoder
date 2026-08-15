"""Canonical Huffman coding - the starting point of the experiment.

Kept as a real working codec, not just an estimate, so the compression ladder in
tools/ladder.py can report measured bytes for the "Huffman the colour channels"
idea and every step that improves on it.
"""

import heapq

import numpy as np

MAX_LEN = 15


def code_lengths(hist, max_len=MAX_LEN):
    """Huffman code lengths for a histogram, capped at `max_len` bits.

    If the natural tree is too deep the histogram is flattened and rebuilt,
    which always terminates and costs a fraction of a percent in practice.
    """
    hist = np.asarray(hist, dtype=np.int64).copy()
    while True:
        lengths = _lengths_once(hist)
        if lengths is None or lengths.max() <= max_len:
            return lengths
        used = hist > 0
        hist[used] = (hist[used] + 1) // 2


def _lengths_once(hist):
    symbols = np.nonzero(hist)[0]
    if len(symbols) == 0:
        return None
    lengths = np.zeros(len(hist), dtype=np.int32)
    if len(symbols) == 1:
        lengths[symbols[0]] = 1
        return lengths

    heap = [(int(hist[s]), [int(s)]) for s in symbols]
    heapq.heapify(heap)
    while len(heap) > 1:
        w1, g1 = heapq.heappop(heap)
        w2, g2 = heapq.heappop(heap)
        for s in g1:
            lengths[s] += 1
        for s in g2:
            lengths[s] += 1
        heapq.heappush(heap, (w1 + w2, g1 + g2))
    return lengths


def canonical_codes(lengths):
    """Canonical code assignment: only the lengths need transmitting."""
    codes = np.zeros(len(lengths), dtype=np.uint32)
    code = 0
    for length in range(1, int(lengths.max()) + 1):
        for sym in np.nonzero(lengths == length)[0]:
            codes[sym] = code
            code += 1
        code <<= 1
    return codes


def encoded_bits(hist, lengths):
    return int((np.asarray(hist, dtype=np.int64) * lengths).sum())


def table_bytes(lengths):
    """Cost of sending the code lengths: one nibble per symbol."""
    return (len(lengths) + 1) // 2


def encode(symbols, hist=None):
    """Encode a uint8 array. Returns (table_bytes + payload) as bytes."""
    symbols = np.asarray(symbols, dtype=np.uint8).ravel()
    if hist is None:
        hist = np.bincount(symbols, minlength=256)
    lengths = code_lengths(hist)
    codes = canonical_codes(lengths)

    out = bytearray()
    for i in range(0, 256, 2):
        out.append((int(lengths[i]) << 4) | int(lengths[i + 1]))

    sym_codes = codes[symbols].astype(np.uint32)
    sym_lens = lengths[symbols].astype(np.int64)
    total = int(sym_lens.sum())
    starts = np.zeros(len(sym_lens) + 1, dtype=np.int64)
    np.cumsum(sym_lens, out=starts[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(starts[:-1], sym_lens)
    shift = np.repeat(sym_lens, sym_lens) - 1 - within
    bits = ((np.repeat(sym_codes, sym_lens) >> shift.astype(np.uint32)) & 1).astype(np.uint8)
    out += np.packbits(bits).tobytes()
    return bytes(out)


def decode(data, count):
    lengths = np.zeros(256, dtype=np.int32)
    for i in range(128):
        lengths[2 * i] = data[i] >> 4
        lengths[2 * i + 1] = data[i] & 0xF
    codes = canonical_codes(lengths)
    lookup = {}
    for sym in np.nonzero(lengths)[0]:
        lookup[(int(lengths[sym]), int(codes[sym]))] = int(sym)

    out = np.zeros(count, dtype=np.uint8)
    bitpos = 128 * 8
    for i in range(count):
        code = 0
        length = 0
        while True:
            code = (code << 1) | ((data[bitpos >> 3] >> (7 - (bitpos & 7))) & 1)
            bitpos += 1
            length += 1
            sym = lookup.get((length, code))
            if sym is not None:
                out[i] = sym
                break
    return out
