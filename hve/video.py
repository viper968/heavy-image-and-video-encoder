"""The .hvv video codec: intra first frame, then per-block motion-compensated frames.

Each later frame is split into blocks that independently choose between spatial
prediction (the still-image path) and temporal prediction from the previous
frame at a searched motion vector. Modes and vectors are coded first, so pixel
residuals can still be coded in plain raster order and keep full access to their
spatial neighbours.
"""

import bisect

import numpy as np

from . import model, rc
from .bitio import Reader, Writer
from .transform import predict_plane, rct_forward, rct_inverse

MAGIC = b"HVV1"
FLAG_RCT = 1
BLOCK = 16
SEARCH = 8

MODE_SPATIAL = 0
MODE_TEMPORAL = 1

# Model kinds 0-3 are the intra banks from model.py; 4-7 mirror them for
# temporally predicted pixels, whose residuals have quite different statistics.
INTER_KIND_OFFSET = 4
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
# pixel coding


def code_inter_plane(coder, encode, width, height, kind, bank, modes, mvs,
                     bs_y, bs_x, mv_sy, mv_sx, ref_rows, src=None, luma_err=None):
    """Raster-order pixel coding where each block's prediction source varies.

    `bs_y`/`bs_x` are this plane's block size and `mv_sy`/`mv_sx` the motion
    vector divisors, so a subsampled chroma plane reuses the luma decisions.
    """
    zero_p = bank["zero"]
    sign_p = bank["sign"]
    nb_p = bank["nb"]
    mant_p = bank["mant"]
    bit = coder.bit
    bypass = coder.bypass
    bisect_right = bisect.bisect_right
    act_ladder, err_ladder, lum_ladder = model.ACT_LADDER, model.ERR_LADDER, model.LUMA_LADDER
    nerr, nlum, max_nb = model.NERR, model.NLUM, model.MAX_NB
    nact = model.NACT

    def kind_offsets(k):
        return (k * nact * nerr * nlum, k * nact * nerr * (max_nb + 1),
                k * 9, k * (max_nb + 1))

    off_intra = kind_offsets(kind)
    off_inter = kind_offsets(kind + INTER_KIND_OFFSET)

    rows = []
    err_rows = []
    prev = [0] * width
    prev_err = [0] * width
    nbx = len(modes[0])

    for y in range(height):
        cur = [0] * width
        cur_err = [0] * width
        srow = src[y] if encode else None
        lrow = luma_err[y] if luma_err is not None else None
        by = min(y // bs_y, len(modes) - 1)
        mode_row = modes[by]
        mv_row = mvs[by]

        # Per-row lookup tables: which reference row and column each pixel reads.
        mode_x = [0] * width
        ref_row_of_x = [None] * width
        ref_col_of_x = [0] * width
        for x in range(width):
            bx = min(x // bs_x, nbx - 1)
            if mode_row[bx] == MODE_TEMPORAL:
                mode_x[x] = 1
                ry = y + int(mv_row[bx][0]) // mv_sy
                rx = x + int(mv_row[bx][1]) // mv_sx
                ref_row_of_x[x] = ref_rows[0 if ry < 0 else (height - 1 if ry >= height else ry)]
                ref_col_of_x[x] = 0 if rx < 0 else (width - 1 if rx >= width else rx)

        west = 0
        west_err = 0
        first_row = y == 0

        for x in range(width):
            if first_row:
                spatial_pred = 128 if x == 0 else west
                act = 0
                sgn = 4
            else:
                north = prev[x]
                if x == 0:
                    spatial_pred = north
                    act = 0
                    sgn = 4
                else:
                    nwest = prev[x - 1]
                    neast = prev[x + 1] if x + 1 < width else north
                    if nwest >= north:
                        if nwest >= west:
                            spatial_pred = north if north < west else west
                        else:
                            spatial_pred = north + west - nwest
                    elif nwest <= west:
                        spatial_pred = north if north > west else west
                    else:
                        spatial_pred = north + west - nwest
                    if spatial_pred > 255:
                        spatial_pred = 255
                    elif spatial_pred < 0:
                        spatial_pred = 0
                    d1 = ((west - nwest + 128) & 255) - 128
                    d2 = ((nwest - north + 128) & 255) - 128
                    d3 = ((north - neast + 128) & 255) - 128
                    act = ((d1 if d1 >= 0 else -d1) + (d2 if d2 >= 0 else -d2)
                           + (d3 if d3 >= 0 else -d3))
                    sgn = (0 if d1 < 0 else (1 if d1 == 0 else 2)) * 3 \
                        + (0 if d2 < 0 else (1 if d2 == 0 else 2))

            if mode_x[x]:
                pred = ref_row_of_x[x][ref_col_of_x[x]]
                k_zero, k_nb, k_sign, k_mant = off_inter
            else:
                pred = spatial_pred
                k_zero, k_nb, k_sign, k_mant = off_intra

            if first_row:
                err_sum = west_err
            else:
                err_sum = west_err + prev_err[x] + (prev_err[x + 1] if x + 1 < width else 0)
            sub = bisect_right(act_ladder, act) * nerr + bisect_right(err_ladder, err_sum)
            lum = bisect_right(lum_ladder, lrow[x]) if lrow is not None else 0
            zctx = k_zero + sub * nlum + lum
            nbbase = k_nb + sub * (max_nb + 1)

            if encode:
                d = ((srow[x] - pred + 128) & 255) - 128
                mag = -d if d < 0 else d
                bit(zero_p, zctx, 1 if mag else 0)
                if mag:
                    bit(sign_p, k_sign + sgn, 1 if d < 0 else 0)
                    v = mag - 1
                    nb = v.bit_length()
                    for i in range(nb):
                        bit(nb_p, nbbase + i, 1)
                    if nb < max_nb:
                        bit(nb_p, nbbase + nb, 0)
                    if nb >= 2:
                        bit(mant_p, k_mant + nb, (v >> (nb - 2)) & 1)
                        if nb > 2:
                            bypass(v & ((1 << (nb - 2)) - 1), nb - 2)
                value = (pred + d) & 255
            else:
                if bit(zero_p, zctx):
                    neg = bit(sign_p, k_sign + sgn)
                    nb = 0
                    while nb < max_nb and bit(nb_p, nbbase + nb):
                        nb += 1
                    if nb < 2:
                        v = nb
                    else:
                        top = bit(mant_p, k_mant + nb)
                        rest = bypass(nb - 2) if nb > 2 else 0
                        v = (1 << (nb - 1)) | (top << (nb - 2)) | rest
                    mag = v + 1
                    value = (pred - mag if neg else pred + mag) & 255
                else:
                    mag = 0
                    value = pred & 255

            cur[x] = value
            cur_err[x] = mag
            west = value
            west_err = mag

        rows.append(cur)
        err_rows.append(cur_err)
        prev = cur
        prev_err = cur_err

    return rows, err_rows


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

    coder = rc.Encoder()
    bank = new_video_model()
    prev_rows = None
    block = BLOCK
    luma_h, luma_w = first[0].shape

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
                _, err = code_inter_plane(coder, True, pw, ph, min(i, 3), bank,
                                          modes.tolist(), mvs.tolist(),
                                          max(1, block // sy), max(1, block // sx),
                                          sy, sx, prev_rows[i], src=plane.tolist(),
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
                rows, err = code_inter_plane(coder, False, pw, ph, min(i, 3), bank,
                                             modes, mvs, max(1, block // sy),
                                             max(1, block // sx), sy, sx,
                                             prev_rows[i],
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
