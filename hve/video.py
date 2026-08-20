"""The .hvv video codec: intra first frame, then per-block motion-compensated frames.

Each later frame is split into blocks that independently choose between spatial
prediction (the still-image path) and temporal prediction from the previous
frame at a searched motion vector. Modes and vectors are coded first, so pixel
residuals can still be coded in plain raster order and keep full access to their
spatial neighbours.
"""

import numpy as np

from . import model, native, rc
from .bitio import Reader, Writer
from .transform import predict_plane, rct_forward, rct_inverse

MAGIC = b"HVV4"
FLAG_RCT = 1
BLOCK = 16
SEARCH = 8              # full-pel search radius, in whole pixels

MODE_SPATIAL = 0
MODE_TEMPORAL = model.MODE_TEMPORAL

# Motion vectors are in HALF-pixel units, so the usable displacement is still
# +-SEARCH whole pixels and a differential can reach twice that in half-pels.
#
# Real motion does not land on the pixel grid, and rounding it there leaves a
# residual that no amount of context modelling recovers. Measured against the
# codec's own cost proxy before building it, half-pel is worth -5.3% on bus,
# -7.9% on mobile and -6.7% on foreman, and roughly nothing on the near-static
# clips - which is the right shape, because those have no sub-pixel motion to
# capture. The two rejected alternatives are recorded in docs/research.md.
MV_MAX = 4 * SEARCH
HALF_PEL_BIAS = 64      # cost units a half-pel refinement must beat, swept on dev

_COST = np.round(np.log2(1.0 + np.arange(129)) * 8).astype(np.int32)

# The same cost, indexed directly by the *wrapped byte difference* rather than
# by the folded magnitude. `(a - b) & 255` on uint8 is a single subtract with no
# widening, and the fold is baked into the table, so scoring a search position
# becomes one subtract and one table lookup instead of a widen, a subtract, an
# abs, a minimum and a gather. Measured at 51% of search time before this.
_COST_BYTE = _COST[np.minimum(np.arange(256), 256 - np.arange(256)).clip(0, 128)]

# Coarse-to-fine search. The full search was 289 positions at full resolution
# for every frame; at 1080p that is 10 seconds per frame and 85% of encode time.
# Searching a small pyramid first and refining down costs about a tenth of that
# and finds nearly the same vectors, because a motion vector that is right at
# full resolution is almost always right to within a pixel at half.
#
# "Nearly" is the catch, and it is why this is gated on frame size. Forcing the
# pyramid on at CIF cost 1.26% on foreman - box-downsampling a 352x288 frame
# twice throws away the detail that distinguishes one candidate vector from
# another, and the refinement cannot recover a match the coarse level never
# pointed at. At CIF an exhaustive search is affordable anyway, so below this
# many pixels the codec simply does one and gives up nothing.
PYRAMID_MIN_PIXELS = 300_000
PYRAMID_LEVELS = 2
REFINE_RADIUS = 1


def new_video_model():
    bank = model.new_model()
    bank["mode"] = rc.new_probs(4)
    bank["mv_zero"] = rc.new_probs(2)
    bank["mv_sign"] = rc.new_probs(2)
    bank["mv_mag"] = rc.new_probs(2 * (MV_MAX + 1))
    return bank


# --------------------------------------------------------------------------
# motion estimation (encoder only - the decoder just applies what it is told)


def halfpel_planes(ref):
    """The four half-pel phases of a reference plane, as (4, h, w) uint8.

    Plane index is (dy & 1) * 2 + (dx & 1), so a half-pel vector splits into a
    whole-pixel offset of `d >> 1` and a phase of `d & 1`. That works for
    negative vectors too, because an arithmetic shift floors: -3 becomes offset
    -2 and phase 1, which is -1.5 pixels.

    Bilinear rather than a longer filter. A 6-tap would be sharper, but every
    tap has to be reproduced bit-exactly by both implementations, and the
    measurement said most of the win is in having *any* sub-pixel position at
    all rather than in the quality of the interpolant.

    Edges replicate, matching the clamping the pixel loop already does.
    """
    r = ref.astype(np.int32)
    right = np.concatenate([r[:, 1:], r[:, -1:]], axis=1)
    down = np.concatenate([r[1:, :], r[-1:, :]], axis=0)
    diag = np.concatenate([right[1:, :], right[-1:, :]], axis=0)
    return np.stack([r,
                     (r + right + 1) >> 1,
                     (r + down + 1) >> 1,
                     (r + right + down + diag + 2) >> 2]).astype(np.uint8)


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
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        return _COST_BYTE[np.subtract(a, b, dtype=np.uint8)]
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return _COST[np.minimum(diff, 256 - diff)]


def _gather_blocks(plane, mv, bs, h, w):
    """Sample `plane` for every block at its own whole-pixel offset.

    The half-pel refinement gives each block a different centre, so this is a
    per-block gather rather than one global shift. Doing it with fancy indexing
    keeps it a single numpy operation over the whole frame instead of a Python
    loop over several hundred blocks.
    """
    nby, nbx = mv.shape[0], mv.shape[1]
    ys = (np.arange(nby)[:, None] * bs + np.arange(bs)[None, :])[:, None, :] \
        + mv[..., 0][:, :, None]
    xs = (np.arange(nbx)[:, None] * bs + np.arange(bs)[None, :])[None, :, :] \
        + mv[..., 1][:, :, None]
    np.clip(ys, 0, h - 1, out=ys)
    np.clip(xs, 0, w - 1, out=xs)
    return plane[ys[:, :, :, None], xs[:, :, None, :]]


def _block_cost_at(cur, planes, mv, bs):
    """Cost of every block predicted from its own half-pel vector in `mv`.

    The phase index joins the gather rather than driving a loop over all four
    planes. Selecting per block afterwards meant fetching the whole frame four
    times and discarding three of them, which at 1080p was the single most
    expensive step left in the search.
    """
    h, w = cur.shape
    nby, nbx = mv.shape[0], mv.shape[1]
    phase = ((mv[..., 0] & 1) * 2 + (mv[..., 1] & 1)).astype(np.intp)
    ys = (np.arange(nby)[:, None] * bs + np.arange(bs)[None, :])[:, None, :] \
        + (mv[..., 0] >> 1)[:, :, None]
    xs = (np.arange(nbx)[:, None] * bs + np.arange(bs)[None, :])[None, :, :] \
        + (mv[..., 1] >> 1)[:, :, None]
    np.clip(ys, 0, h - 1, out=ys)
    np.clip(xs, 0, w - 1, out=xs)
    got = planes[phase[:, :, None, None], ys[:, :, :, None], xs[:, :, None, :]]

    pad = np.zeros((nby * bs, nbx * bs), dtype=cur.dtype)
    pad[:h, :w] = cur
    blocks = pad.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    return _residual_cost(blocks, got).sum(axis=(2, 3)).astype(np.int64)


def _halve(p):
    """Box-average down by two, replicating an odd last row or column."""
    if p.shape[0] & 1:
        p = np.concatenate([p, p[-1:]], axis=0)
    if p.shape[1] & 1:
        p = np.concatenate([p, p[:, -1:]], axis=1)
    a = p.astype(np.uint16)
    return (((a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2] + 2)
             >> 2).astype(np.uint8))


def _full_search(cur, ref, bs, radius, nby, nbx):
    """Exhaustive whole-pixel search, scored by one strided view per position.

    Padding the reference once and slicing it beats re-deriving clipped index
    arrays for every position, and a slice is a view rather than a copy.
    """
    pad = np.pad(ref, radius, mode="edge")
    h, w = cur.shape
    best = None
    mv = np.zeros((nby, nbx, 2), dtype=np.int32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            view = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            sums = _block_sums(_residual_cost(cur, view), nby, nbx, bs)
            if best is None:
                # The seed is (-radius, -radius), not (0, 0). Leaving `mv` at
                # its zero initialisation here meant that a block whose true
                # best vector was the very first candidate kept that
                # candidate's *cost* while reporting the *zero* vector, and
                # then got motion-compensated from the wrong place. Found by
                # diffing the C port against this function.
                best = sums
                mv[..., 0] = dy
                mv[..., 1] = dx
                continue
            better = sums < best
            best = np.where(better, sums, best)
            mv[..., 0] = np.where(better, dy, mv[..., 0])
            mv[..., 1] = np.where(better, dx, mv[..., 1])
    return mv, best


def _refine_whole(cur, ref, mv, bs, radius=REFINE_RADIUS):
    """Try the whole-pixel neighbourhood of each block's own current vector."""
    h, w = cur.shape
    best = _block_cost_whole(cur, ref, mv, bs)
    centre = mv.copy()          # fixed, so candidates cannot walk off the centre
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            cand = centre + np.array([dy, dx], dtype=np.int32)
            cost = _block_cost_whole(cur, ref, cand, bs)
            better = cost < best
            best = np.where(better, cost, best)
            mv[..., 0] = np.where(better, cand[..., 0], mv[..., 0])
            mv[..., 1] = np.where(better, cand[..., 1], mv[..., 1])
    return mv, best


def _block_cost_whole(cur, ref, mv, bs):
    h, w = cur.shape
    nby, nbx = mv.shape[0], mv.shape[1]
    pad = np.zeros((nby * bs, nbx * bs), dtype=cur.dtype)
    pad[:h, :w] = cur
    blocks = pad.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)
    got = _gather_blocks(ref, mv, bs, h, w)
    return _residual_cost(blocks, got).sum(axis=(2, 3)).astype(np.int64)


def motion_search(cur, ref, bs=None, search=None):
    """Per-block best vector and cost, in half-pel units.

    Coarse to fine. An exhaustive whole-pixel search at full resolution costs
    (2*search+1)^2 passes over the frame, which at 1080p was 85% of encode
    time. Instead the search runs on a quarter-size pyramid, where the same
    displacement range needs a quarter of the radius and a sixteenth of the
    pixels, and each finer level only has to correct the doubling by +-1.

    The final stage tries the eight half-pel neighbours. Sub-pixel motion is a
    refinement of the right whole-pixel match rather than a different match
    somewhere else, so a full half-pel search would cost four times as much to
    find nearly the same vectors.
    """
    bs = BLOCK if bs is None else bs
    search = SEARCH if search is None else search
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    planes = halfpel_planes(ref)

    # Halve until an exhaustive search is affordable, but only as deep as the
    # block size and search radius survive.
    levels = 0
    while (levels < PYRAMID_LEVELS
           and (h * w) >> (2 * levels) > PYRAMID_MIN_PIXELS
           and (bs >> (levels + 1)) >= 4
           and (search >> (levels + 1)) >= 1
           and min(h, w) >> (levels + 1) >= bs):
        levels += 1

    curs, refs = [cur], [ref]
    for _ in range(levels):
        curs.append(_halve(curs[-1]))
        refs.append(_halve(refs[-1]))

    radius = -(-search >> levels)               # ceil, so coverage never shrinks
    bmv, best = _full_search(curs[levels], refs[levels], bs >> levels,
                             radius, nby, nbx)
    for level in range(levels - 1, -1, -1):
        bmv = bmv * 2
        bmv, best = _refine_whole(curs[level], refs[level], bmv, bs >> level)

    np.clip(bmv, -search, search, out=bmv)
    bmv *= 2                                    # whole pixels -> half-pel units
    best = _block_cost_at(cur, planes, bmv, bs)
    best = best.astype(np.int64)
    centre = bmv.copy()         # fixed, as above
    # A half-pel vector costs more to send than the whole-pel one it refines:
    # the magnitudes are coded in half-pel units, so an odd component roughly
    # doubles the unary run. Requiring the refinement to beat the whole-pel
    # match by a margin rather than merely tie is what stops a near-static clip
    # paying for sub-pixel precision it has no use for. Swept on the dev split.
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            if sy == 0 and sx == 0:
                continue
            # Clamped to +-search whole pixels, expressed in half-pels. Every
            # vector and every median of vectors then lives in that range, so a
            # differential cannot exceed 4*search = MV_MAX. Without this the
            # half-pel step can reach +-(2*search+1), a differential can reach
            # 4*search+2, and the encoder writes a longer unary run than the
            # decoder reads back - a desync that silently corrupts the stream
            # rather than failing. Rare enough that the test suite never hit it,
            # which is exactly why it is pinned by a test now.
            cand = np.clip(centre + np.array([sy, sx], dtype=np.int32),
                           -2 * search, 2 * search)
            cost = _block_cost_at(cur, planes, cand, bs)
            better = cost + HALF_PEL_BIAS < best
            best = np.where(better, cost, best)
            bmv[..., 0] = np.where(better, cand[..., 0], bmv[..., 0])
            bmv[..., 1] = np.where(better, cand[..., 1], bmv[..., 1])
    return bmv, best


def spatial_cost(cur, bs=None):
    """What the same blocks would cost under plain spatial prediction."""
    bs = BLOCK if bs is None else bs
    h, w = cur.shape
    nby, nbx = -(-h // bs), -(-w // bs)
    resid = _residual_cost(cur, predict_plane(cur).astype(np.uint8))
    return _block_sums(resid, nby, nbx, bs)


def choose_modes(cur_y, ref_y, mv_penalty=48, bs=None):
    """Pick spatial vs temporal per block, charging a flat price for sending a vector.

    The native search is threaded over block rows and picks the same vectors as
    the numpy one — `tests/test_native.py` compares them block for block on real
    clips, because a search that merely finds *good* vectors rather than the
    *same* vectors would quietly turn every future compression measurement into
    a comparison of two codecs.
    """
    bs = BLOCK if bs is None else bs
    if native.available():
        bmv, tcost = native.motion_search(
            cur_y, ref_y, bs, SEARCH, _COST_BYTE, HALF_PEL_BIAS,
            PYRAMID_MIN_PIXELS, PYRAMID_LEVELS, REFINE_RADIUS)
        scost = native.spatial_cost(cur_y, bs, _COST_BYTE)
    else:
        bmv, tcost = motion_search(cur_y, ref_y, bs=bs)
        scost = spatial_cost(cur_y, bs=bs)
    moving = (bmv != 0).any(axis=-1)
    tcost = tcost + np.where(moving, mv_penalty, 0)
    modes = (tcost < scost).astype(np.int32)
    bmv[modes == MODE_SPATIAL] = 0
    return modes, bmv


# --------------------------------------------------------------------------
# mode / vector coding


def _median3(a, b, c):
    return a + b + c - min(a, b, c) - max(a, b, c)


def mv_predictor(mvs, by, bx, nbx):
    """Predict this block's vector from its already-coded neighbours.

    The componentwise median of left, above and above-right, which is what
    H.264 uses. The previous version here predicted from the left block alone
    and reset to zero after every spatially-coded block, so a single intra
    block in the middle of a smooth pan made the next vector cost full price.

    A median is specifically better than an average for this: one neighbour
    sitting on a moving object while the other two track the background gets
    outvoted rather than dragging the prediction, which is the same reason the
    match model wants a switch rather than a vote.

    Blocks coded spatially count as zero vectors. Both sides walk the blocks in
    raster order, so left, above and above-right are always already decided.
    """
    if by == 0:
        if bx == 0:
            return 0, 0
        return int(mvs[0][bx - 1][0]), int(mvs[0][bx - 1][1])
    ly, lx = (int(mvs[by][bx - 1][0]), int(mvs[by][bx - 1][1])) if bx else (0, 0)
    uy, ux = int(mvs[by - 1][bx][0]), int(mvs[by - 1][bx][1])
    if bx + 1 < nbx:
        ry, rx = int(mvs[by - 1][bx + 1][0]), int(mvs[by - 1][bx + 1][1])
    elif bx:
        ry, rx = int(mvs[by - 1][bx - 1][0]), int(mvs[by - 1][bx - 1][1])
    else:
        ry, rx = 0, 0
    return _median3(ly, uy, ry), _median3(lx, ux, rx)


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
        for bx in range(nbx):
            left = modes[by][bx - 1] if bx else 0
            up = modes[by - 1][bx] if by else 0
            ctx = left * 2 + up
            if encode:
                coder.bit(bank["mode"], ctx, int(modes[by][bx]))
            else:
                modes[by][bx] = coder.bit(bank["mode"], ctx)
            if modes[by][bx] == MODE_TEMPORAL:
                py, px = mv_predictor(mvs, by, bx, nbx)
                if encode:
                    _code_mv_component(coder, True, bank, 0,
                                       int(mvs[by][bx][0]) - py)
                    _code_mv_component(coder, True, bank, 1,
                                       int(mvs[by][bx][1]) - px)
                else:
                    dy = _code_mv_component(coder, False, bank, 0)
                    dx = _code_mv_component(coder, False, bank, 1)
                    mvs[by][bx][0] = dy + py
                    mvs[by][bx][1] = dx + px
    return modes, mvs


# --------------------------------------------------------------------------
# native path


def _phase_rows(rows):
    """halfpel_planes for the pure-Python path, which works in lists of ints."""
    return [p.tolist() for p in halfpel_planes(np.array(rows, dtype=np.uint8))]


def _native_inter(modes_i, mvs_i, block, sy, sx, prev_plane):
    return (1, modes_i, mvs_i, max(1, block // sy), max(1, block // sx), sy, sx,
            native.halfpel_planes(prev_plane))


def _encode_payload_native(frames, block, luma_h, luma_w, progress, features):
    """The frame loop, with the per-pixel coding done by the C kernel.

    Planes stay uint8 the whole way, which matters at 1080p: the four half-pel
    phases of a luma plane are 8 MB as bytes and 66 MB widened to int64, and the
    kernel reads them at random offsets where cache behaviour is the whole cost.
    """
    samples = sum(p.size for p in _to_planes(frames[0])[0]) * len(frames)
    coder = native.Coder(True, capacity=samples * 2 + 65536)
    bank = native.Bank(luma_h, luma_w, video=True, features=features)
    prev = None

    for fi, frame in enumerate(frames):
        planes, _ = _to_planes(frame)
        cur = [np.ascontiguousarray(p, dtype=np.uint8) for p in planes]
        if prev is None:
            for i, pl in enumerate(cur):
                bank.code(coder, True, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0)
        else:
            modes, mvs = choose_modes(cur[0], prev[0], bs=block)
            modes_i = np.ascontiguousarray(modes, dtype=np.int64)
            mvs_i = np.ascontiguousarray(mvs, dtype=np.int64)
            bank.code_block_info(coder, True, modes_i, mvs_i, MV_MAX)
            for i, pl in enumerate(cur):
                ph, pw = pl.shape
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                bank.code(coder, True, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0,
                          inter=_native_inter(modes_i, mvs_i, block, sy, sx,
                                              prev[i]))
        prev = cur
        if progress:
            progress(fi, len(frames))
    return coder.finish()


def _decode_payload_native(payload, shapes, nframes, block, progress, features):
    coder = native.Coder(False, payload=payload)
    luma_h, luma_w = shapes[0]
    bank = native.Bank(luma_h, luma_w, video=True, features=features)
    nby, nbx = -(-luma_h // block), -(-luma_w // block)
    prev = None
    out = []

    for fi in range(nframes):
        cur = [np.zeros((h, wd), dtype=np.uint8) for (h, wd) in shapes]
        if prev is None:
            for i, pl in enumerate(cur):
                bank.code(coder, False, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0)
        else:
            modes_i = np.zeros((nby, nbx), dtype=np.int64)
            mvs_i = np.zeros((nby, nbx, 2), dtype=np.int64)
            bank.code_block_info(coder, False, modes_i, mvs_i, MV_MAX)
            for i, pl in enumerate(cur):
                ph, pw = pl.shape
                sy = max(1, luma_h // ph)
                sx = max(1, luma_w // pw)
                bank.code(coder, False, pl, min(i, 3), 1 if i in (1, 2) else 0,
                          1 if i == 0 else 0,
                          inter=_native_inter(modes_i, mvs_i, block, sy, sx,
                                              prev[i]))
        prev = cur
        out.append([p.copy() for p in cur])
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


def encode(frames, progress=None, features=None):
    """Compress a sequence of frames to .hvv bytes.

    `features` selects which model stages run and is written into the header,
    so decoding needs no matching argument. See model.FEAT_*.
    """
    features = model.FEATURES if features is None else features
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
    w.u8(features)
    for p in first:
        w.varint(p.shape[1])
        w.varint(p.shape[0])

    block = BLOCK
    luma_h, luma_w = first[0].shape

    from .image import _check_features
    _check_features(features)
    if native.available():
        payload = _encode_payload_native(frames, block, luma_h, luma_w, progress,
                                         features)
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
                                        sy, sx,
                                        _phase_rows(prev_rows[i]))
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
    features = r.u8()
    shapes = []
    for _ in range(nplanes):
        pw = r.varint()
        ph = r.varint()
        shapes.append((ph, pw))
    payload_len = r.varint()
    payload = r.raw(payload_len)

    from .image import _check_features
    _check_features(features)
    if native.available():
        out = []
        for planes in _decode_payload_native(payload, shapes, nframes, block,
                                             progress, features):
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
                                        max(1, block // sx), sy, sx,
                                        _phase_rows(prev_rows[i]))
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
