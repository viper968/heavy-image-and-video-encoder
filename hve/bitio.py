"""Byte-oriented container primitives (LEB128 varints)."""


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def u8(self, v):
        self.buf.append(v & 0xFF)

    def varint(self, v):
        if v < 0:
            raise ValueError("varint is unsigned, got %d" % v)
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                self.buf.append(b | 0x80)
            else:
                self.buf.append(b)
                return

    def svarint(self, v):
        """Zig-zag then varint, for values that may be negative."""
        self.varint((v << 1) ^ (v >> 63) if v < 0 else v << 1)

    def raw(self, data):
        self.buf += data

    def bytes(self):
        return bytes(self.buf)


class Reader:
    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    def u8(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def varint(self):
        shift = 0
        result = 0
        while True:
            b = self.data[self.pos]
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def svarint(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def raw(self, n):
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out
