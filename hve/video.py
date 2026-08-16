"""The .hvv video codec: intra first frame, then per-block motion-compensated frames.

Each later frame is split into blocks that independently choose between spatial
prediction (the still-image path) and temporal prediction from the previous
frame at a searched motion vector. Modes and vectors are coded first, so pixel
residuals can still be coded in plain raster order and keep full access to their
spatial neighbours.
"""

import numpy as np

from . import fast, model, rc
from .bitio import Reader, Writer
from .transform import predict_plane, rct_forward, rct_inverse

MAGIC = b"HVV1"
FLAG_RCT = 1
BLOCK = 16
SEARCH = 8

MODE_SPATIAL = 0
MODE_TEMPORAL = model.MODE_TEMPORAL

MV_MAX = 2 * SEARCH

_COST = np.round(np.log2(1.0 + np.arange(129)) * 8).astype(np.int32)


def new_video_model():
    bank = model.new_model()
    bank["mode"] = rc.new_probs(4)
    bank["mv_zero"] = rc.new_probs(2)
    bank["mv_sign"] = rc.new_probs(2)
    bank["mv_mag"] = rc.new_probs(2 * (MV_MAX + 1))
    return bank


# --------------------------------------------------------------------------
# motion estimation (encoder only - the decoder just applies what it is told)


def _shifted(ref, dy, dx):
    h, w = ref.shape
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return ref[ys[:, None], xs[None, :]]


def _block_sums(cost, nby, nbx, bs):
    h, w = cost.shape
    pad = np.zeros((nby * bs, nbx * bs), dtype=cost.dtype)
    pad[:h, :w] = cost
    return pad.reshape(nby, bs, nbx, bs).sum(axis=(1, 3))


def _residual_cost(a, b):
    """Modular residual magnitude, mapped through a log-ish bit-cost proxy."""
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return _COST[np.minimum(diff, 256 - diff)]


def motion_search(cur, ref, bs=None, search=None):
    """Full search on the luma plane; returns per-block best vector and cost."""
    bs = BLOCK if bs is None else bs
    search = SEARCH if search is None else search
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    best = None
    bmv = np.zeros((nby, nbx, 2), dtype=np.int32)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sums = _block_sums(_residual_cost(cur, _shifted(ref, dy, dx)), nby, nbx, bs)
            if best is None:
                best = sums
                continue
            better = sums < best
            best = np.where(better, sums, best)
            bmv[..., 0] = np.where(better, dy, bmv[..., 0])
            bmv[..., 1] = np.where(better, dx, bmv[..., 1])
    return bmv, best


def spatial_cost(cur, bs=None):
    """What the same blocks would cost under plain spatial prediction."""
    bs = BLOCK if bs is None else bs
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    resid = _residual_cost(cur, predict_plane(cur).astype(np.uint8))
    return _block_sums(resid, nby, nbx, bs)


def choose_modes(cur_y, ref_y, mv_penalty=48, bs=None):
    """Pick spatial vs temporal per block, charging a flat price for sending a vector."""
    bs = BLOCK if bs is None else bs
    bmv, tcost = motion_search(cur_y, ref_y, bs=bs)
    scost = spatial_cost(cur_y, bs=bs)
    moving = (bmv != 0).any(axis=-1)
    tcost = tcost + np.where(moving, mv_penalty, 0)
    modes = (tcost < scost).astype(np.int32)
    bmv[modes == MODE_SPATIAL] = 0
    return modes, bmv


# --------------------------------------------------------------------------
# mode / vector coding


def _code_mv_component(coder, encode, bank, axis, value=None):
    probs_zero = bank["mv_zero"]
    probs_sign = bank["mv_sign"]
    probs_mag = bank["mv_mag"]
    base = axis * (MV_MAX + 1)
    if encode:
        mag = abs(value)
        coder.bit(probs_zero, axis, 1 if mag else 0)
        if mag:
            coder.bit(probs_sign, axis, 1 if value < 0 else 0)
            for i in range(mag - 1):
                coder.bit(probs_mag, base + i, 1)
            if mag - 1 < MV_MAX:
                coder.bit(probs_mag, base + mag - 1, 0)
        return value
    if not coder.bit(probs_zero, axis):
        return 0
    neg = coder.bit(probs_sign, axis)
    mag = 1
    while mag - 1 < MV_MAX and coder.bit(probs_mag, base + mag - 1):
        mag += 1
    return -mag if neg else mag


def code_block_info(coder, encode, bank, nby, nbx, modes=None, mvs=None):
    """Modes and motion vectors for one frame, differentially against the left block."""
    if not encode:
        modes = np.zeros((nby, nbx), dtype=np.int32)
        mvs = np.zeros((nby, nbx, 2), dtype=np.int32)
    for by in range(nby):
        pred_mv = (0, 0)
        for bx in range(nbx):
            left = modes[by][bx - 1] if bx else 0
            up = modes[by - 1][bx] if by else 0
            ctx = left * 2 + up
            if encode:
                coder.bit(bank["mode"], ctx, int(modes[by][bx]))
            else:
                modes[by][bx] = coder.bit(bank["mode"], ctx)
            if modes[by][bx] == MODE_TEMPORAL:
                if encode:
                    dy = int(mvs[by][bx][0]) - pred_mv[0]
                    dx = int(mvs[by][bx][1]) - pred_mv[1]
                    _code_mv_component(coder, True, bank, 0, dy)
                    _code_mv_component(coder, True, bank, 1, dx)
                else:
                    dy = _code_mv_component(coder, False, bank, 0)
                    dx = _code_mv_component(coder, False, bank, 1)
                    mvs[by][bx][0] = dy + pred_mv[0]
                    mvs[by][bx][1] = dx + pred_mv[1]
                pred_mv = (int(mvs[by][bx][0]), int(mvs[by][bx][1]))
            else:
                pred_mv = (0, 0)
    return modes, mvs


# --------------------------------------------------------------------------
# jitted path


def _inter_tuple(modes_i, mvs_i, block, sy, sx, ref):
    return (1, modes_i, mvs_i, max(1, block // sy), max(1, block // sx), sy, sx, ref)


def _encode_payload_fast(frames, block, luma_h, luma_w, progress):
    """Same frame loop as the reference below, with the jitted primitives.

    Motion search stays in numpy and block decisions stay here; only the
    per-pixel coding and the block-info bits cross into the kernel.
    """
    samples = sum(p.size for p in _to_planes(frames[0])[0]) * len(frames)
    coder = fast.Coder(True, capacity=samples * 2 + 65536)
    bank = fast.Bank(video=True)
    errmap = np.zeros((luma_h, luma_w), dtype=np.int64)
    flat = np.zeros(luma_h * luma_w, dtype=np.int64)
    prev = None

    for fi, frame in enumerate(frames):
        planes, _ = _to_planes(frame)
        cur = [np.ascontiguousarray(p, dtype=np.int64) for p in planes]
        if prev is None:
            for i, pl in enumerate(cur):
                bank.code(coder, True, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0, errmap, flat)
        else:
            modes, mvs = choose_modes(planes[0], prev[0].astype(np.uint8), bs=block)
            modes_i = np.ascontiguousarray(modes, dtype=np.int64)
            mvs_i = np.ascontiguousarray(mvs, dtype=np.int64)
            fast._code_block_info(True, coder.st, coder.data, coder.out, bank.mode_p,
                                  bank.mv_zero, bank.mv_sign, bank.mv_mag,
                                  modes_i, mvs_i, MV_MAX)
            for i, pl in enumerate(cur):
                ph, pw = pl.shape
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                bank.code(coder, True, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0, errmap, flat,
                          inter=_inter_tuple(modes_i, mvs_i, block, sy, sx, prev[i]))
        prev = cur
        if progress:
            progress(fi, len(frames))
    return coder.finish()


def _decode_payload_fast(payload, shapes, nframes, block, progress):
    coder = fast.Coder(False, payload=payload)
    bank = fast.Bank(video=True)
    luma_h, luma_w = shapes[0]
    errmap = np.zeros((luma_h, luma_w), dtype=np.int64)
    flat = np.zeros(luma_h * luma_w, dtype=np.int64)
    nby, nbx = -(-luma_h // block), -(-luma_w // block)
    prev = None
    out = []

    for fi in range(nframes):
        cur = [np.zeros((h, wd), dtype=np.int64) for (h, wd) in shapes]
        if prev is None:
            for i, pl in enumerate(cur):
                bank.code(coder, False, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0, errmap, flat)
        else:
            modes_i = np.zeros((nby, nbx), dtype=np.int64)
            mvs_i = np.zeros((nby, nbx, 2), dtype=np.int64)
            fast._code_block_info(False, coder.st, coder.data, coder.out, bank.mode_p,
                                  bank.mv_zero, bank.mv_sign, bank.mv_mag,
                                  modes_i, mvs_i, MV_MAX)
            for i, pl in enumerate(cur):
                ph, pw = pl.shape
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                bank.code(coder, False, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0, errmap, flat,
                          inter=_inter_tuple(modes_i, mvs_i, block, sy, sx, prev[i]))
        prev = cur
        out.append([p.astype(np.uint8) for p in cur])
        if progress:
            progress(fi, nframes)
    return out


# --------------------------------------------------------------------------
# container


def _to_planes(frame):
    """Accept either an RGB frame or a list of planes (e.g. Y/U/V)."""
    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
        return list(rct_forward(frame)), FLAG_RCT
    if isinstance(frame, np.ndarray) and frame.ndim == 2:
        return [frame], 0
    return [np.asarray(p, dtype=np.uint8) for p in frame], 0


def encode(frames, progress=None):
    """Compress a sequence of frames to .hvv bytes."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    first, flags = _to_planes(frames[0])
    nplanes = len(first)

    w = Writer()
    w.raw(MAGIC)
    w.varint(len(frames))
    w.u8(nplanes)
    w.u8(flags)
    w.u8(BLOCK)          # in the header, so the decoder never depends on a constant
    for p in first:
        w.varint(p.shape[1])
        w.varint(p.shape[0])

    block = BLOCK
    luma_h, luma_w = first[0].shape

    if fast.available():
        payload = _encode_payload_fast(frames, block, luma_h, luma_w, progress)
        w.varint(len(payload))
        w.raw(payload)
        return w.bytes()

    coder = rc.Encoder()
    bank = new_video_model()
    prev_rows = None

    for fi, frame in enumerate(frames):
        planes, _ = _to_planes(frame)
        rows_all = []
        luma_err = None
        if prev_rows is None:
            for i, plane in enumerate(planes):
                _, err = model.code_plane(coder, True, plane.shape[1], plane.shape[0],
                                          min(i, 3), bank, src=plane.tolist(),
                                          luma_err=luma_err if i in (1, 2) else None)
                rows_all.append(plane.tolist())
                if i == 0:
                    luma_err = err
        else:
            nby, nbx = -(-luma_h // block), -(-luma_w // block)
            modes, mvs = choose_modes(planes[0], np.array(prev_rows[0], dtype=np.uint8),
                                      bs=block)
            code_block_info(coder, True, bank, nby, nbx, modes.tolist(),
                            mvs.tolist())
            for i, plane in enumerate(planes):
                ph, pw = plane.shape
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                inter = model.InterInfo(modes.tolist(), mvs.tolist(),
                                        max(1, block // sy), max(1, block // sx),
                                        sy, sx, prev_rows[i])
                _, err = model.code_plane(coder, True, pw, ph, min(i, 3), bank,
                                          src=plane.tolist(), inter=inter,
                                          luma_err=luma_err if i in (1, 2) else None)
                rows_all.append(plane.tolist())
                if i == 0:
                    luma_err = err
        prev_rows = rows_all
        if progress:
            progress(fi, len(frames))

    payload = coder.finish()
    w.varint(len(payload))
    w.raw(payload)
    return w.bytes()


def decode(data, progress=None):
    """Decompress .hvv bytes back to a list of frames."""
    r = Reader(data)
    if r.raw(4) != MAGIC:
        raise ValueError("not an .hvv stream")
    nframes = r.varint()
    nplanes = r.u8()
    flags = r.u8()
    block = r.u8()
    shapes = []
    for _ in range(nplanes):
        pw = r.varint()
        ph = r.varint()
        shapes.append((ph, pw))
    payload_len = r.varint()
    payload = r.raw(payload_len)

    if fast.available():
        out = []
        for planes in _decode_payload_fast(payload, shapes, nframes, block, progress):
            out.append(rct_inverse(planes) if flags & FLAG_RCT else planes)
        return out

    coder = rc.Decoder(payload)
    bank = new_video_model()
    luma_h, luma_w = shapes[0]
    prev_rows = None
    out = []

    for fi in range(nframes):
        rows_all = []
        luma_err = None
        if prev_rows is None:
            for i, (ph, pw) in enumerate(shapes):
                rows, err = model.code_plane(coder, False, pw, ph, min(i, 3), bank,
                                             luma_err=luma_err if i in (1, 2) else None)
                rows_all.append(rows)
                if i == 0:
                    luma_err = err
        else:
            nby, nbx = -(-luma_h // block), -(-luma_w // block)
            modes, mvs = code_block_info(coder, False, bank, nby, nbx)
            modes, mvs = modes.tolist(), mvs.tolist()
            for i, (ph, pw) in enumerate(shapes):
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                inter = model.InterInfo(modes, mvs, max(1, block // sy),
                                        max(1, block // sx), sy, sx, prev_rows[i])
                rows, err = model.code_plane(coder, False, pw, ph, min(i, 3), bank,
                                             inter=inter,
                                             luma_err=luma_err if i in (1, 2) else None)
                rows_all.append(rows)
                if i == 0:
                    luma_err = err
        prev_rows = rows_all
        planes = [np.array(rows, dtype=np.uint8) for rows in rows_all]
        out.append(rct_inverse(planes) if flags & FLAG_RCT else planes)
        if progress:
            progress(fi, nframes)
    return out
