"""Adaptive binary range coder (LZMA-style) plus a bank of learned bit models.

Static rANS tables hit a wall: every extra context costs a frequency table in
the header, so the model can never be richer than a few dozen contexts. An
adaptive coder transmits nothing - encoder and decoder learn the same
probabilities from the bits they have already seen - so contexts become free
and the model can be as detailed as we like.
"""

PROB_BITS = 15
PROB_ONE = 1 << PROB_BITS
PROB_INIT = PROB_ONE >> 1
ADAPT_SHIFT = 6  # swept against the Kodak set, see tools/tune.py
TOP = 1 << 24
MASK32 = (1 << 32) - 1


class Encoder:
    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        low = self.low
        if low < 0xFF000000 or low > MASK32:
            carry = low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            if self.cache_size > 1:
                filler = (0xFF + carry) & 0xFF
                self.out.extend(bytes([filler]) * (self.cache_size - 1))
            self.cache = (low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (low << 8) & MASK32

    def bit(self, probs, ctx, value):
        p = probs[ctx]
        bound = (self.range >> PROB_BITS) * p
        if value:
            self.low += bound
            self.range -= bound
            probs[ctx] = p - (p >> ADAPT_SHIFT)
        else:
            self.range = bound
            probs[ctx] = p + ((PROB_ONE - p) >> ADAPT_SHIFT)
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()

    def bit_p(self, p, value):
        """Code a bit against an externally supplied P(bit=0), with no update.

        Used when the probability comes from a mixer or an APM rather than from
        a single stored counter — those stages own their own adaptation.
        """
        bound = (self.range >> PROB_BITS) * p
        if value:
            self.low += bound
            self.range -= bound
        else:
            self.range = bound
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()

    def bypass(self, value, nbits):
        """Uniform bits, no model - for mantissas that carry no useful structure."""
        for i in range(nbits - 1, -1, -1):
            self.range >>= 1
            if (value >> i) & 1:
                self.low += self.range
            while self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self._shift_low()

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class Decoder:
    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos + 5
        self.range = MASK32
        self.code = int.from_bytes(data[pos + 1:pos + 5], "big")

    def bit(self, probs, ctx):
        p = probs[ctx]
        bound = (self.range >> PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            probs[ctx] = p + ((PROB_ONE - p) >> ADAPT_SHIFT)
            value = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[ctx] = p - (p >> ADAPT_SHIFT)
            value = 1
        while self.range < TOP:
            self.range <<= 8
            self.code = ((self.code << 8) | self.data[self.pos]) & MASK32
            self.pos += 1
        return value

    def bit_p(self, p):
        """Decode a bit against an externally supplied P(bit=0), with no update."""
        bound = (self.range >> PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            value = 0
        else:
            self.code -= bound
            self.range -= bound
            value = 1
        while self.range < TOP:
            self.range <<= 8
            self.code = ((self.code << 8) | self.data[self.pos]) & MASK32
            self.pos += 1
        return value

    def bypass(self, nbits):
        value = 0
        for _ in range(nbits):
            self.range >>= 1
            if self.code >= self.range:
                self.code -= self.range
                value = (value << 1) | 1
            else:
                value <<= 1
            while self.range < TOP:
                self.range <<= 8
                self.code = ((self.code << 8) | self.data[self.pos]) & MASK32
                self.pos += 1
        return value


def new_probs(n):
    return [PROB_INIT] * n
