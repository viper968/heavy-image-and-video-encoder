"""Minimal YUV4MPEG2 reader/writer, so video tests use real raw source frames."""

import numpy as np

SUBSAMPLING = {"420jpeg": (2, 2), "420mpeg2": (2, 2), "420paldv": (2, 2),
               "420": (2, 2), "422": (2, 1), "444": (1, 1)}


class Y4M:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        header = self._read_line()
        if not header.startswith(b"YUV4MPEG2"):
            raise ValueError("not a y4m file")
        self.width = self.height = None
        self.colourspace = "420jpeg"
        self.rate = "25:1"
        self.interlace = "p"
        self.aspect = "1:1"
        for tag in header.split()[1:]:
            key, val = tag[:1], tag[1:].decode()
            if key == b"W":
                self.width = int(val)
            elif key == b"H":
                self.height = int(val)
            elif key == b"C":
                self.colourspace = val
            elif key == b"F":
                self.rate = val
            elif key == b"I":
                self.interlace = val
            elif key == b"A":
                self.aspect = val
        self.sx, self.sy = SUBSAMPLING[self.colourspace.split("p")[0]
                                       if self.colourspace.startswith("420p")
                                       else self.colourspace]
        self.cw = -(-self.width // self.sx)
        self.ch = -(-self.height // self.sy)
        self.frame_bytes = self.width * self.height + 2 * self.cw * self.ch

    def _read_line(self):
        out = bytearray()
        while True:
            c = self.fh.read(1)
            if not c or c == b"\n":
                return bytes(out)
            out += c

    def frames(self, limit=None):
        """Yield frames as [Y, U, V] uint8 planes."""
        count = 0
        while limit is None or count < limit:
            marker = self._read_line()
            if not marker:
                return
            if not marker.startswith(b"FRAME"):
                raise ValueError("bad frame marker %r" % marker[:16])
            raw = self.fh.read(self.frame_bytes)
            if len(raw) < self.frame_bytes:
                return
            buf = np.frombuffer(raw, dtype=np.uint8)
            n = self.width * self.height
            m = self.cw * self.ch
            yield [buf[:n].reshape(self.height, self.width),
                   buf[n:n + m].reshape(self.ch, self.cw),
                   buf[n + m:n + 2 * m].reshape(self.ch, self.cw)]
            count += 1

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write(path, frames, width, height, colourspace="420jpeg", rate="25:1"):
    with open(path, "wb") as fh:
        fh.write(b"YUV4MPEG2 W%d H%d F%s Ip A1:1 C%s\n"
                 % (width, height, rate.encode(), colourspace.encode()))
        for planes in frames:
            fh.write(b"FRAME\n")
            for p in planes:
                fh.write(np.ascontiguousarray(p, dtype=np.uint8).tobytes())
