"""Context model and residual binarisation, shared by the encoder and decoder.

Encoder and decoder must derive *identical* contexts from identical history, so
both directions run through this one function with an `encode` flag rather than
being written twice. The still-image and video paths share it too, via the
optional `inter` argument - a second copy of this loop would drift out of sync
the first time the model changed.

A residual is coded as:
    is-it-zero?            (context: plane, activity, neighbour error, luma error)
    sign                   (context: plane, gradient signs)
    bit-length of |d|-1    (unary, context: plane, activity, neighbour error)
    mantissa               (top two bits modelled, remainder as raw bypass bits)
Most pixels in a photo predict perfectly, so the common case costs a single
well-modelled binary decision.
"""

import bisect

from . import rc

KINDS = 8    # 0-3 spatially predicted (luma, Cb, Cr, alpha); 4-7 the
             # motion-compensated mirror used by video.py
INTER_KIND_OFFSET = 4
ACT_LADDER = [1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64, 88, 120]   # 16
ERR_LADDER = [1, 2, 4, 7, 12, 20]                                       # 7
LUMA_LADDER = [2, 10]                                                   # 3
NACT = len(ACT_LADDER) + 1
NERR = len(ERR_LADDER) + 1
NLUM = len(LUMA_LADDER) + 1
MAX_NB = 7
MODE_TEMPORAL = 1


def new_model():
    """A fresh probability bank. One bank is shared by every plane and frame."""
    return {
        "zero": rc.new_probs(KINDS * NACT * NERR * NLUM),
        "sign": rc.new_probs(KINDS * 9),
        "nb": rc.new_probs(KINDS * NACT * NERR * (MAX_NB + 1)),
        "mant": rc.new_probs(KINDS * (MAX_NB + 1) * 2),
    }


def kind_offsets(kind):
    return (kind * NACT * NERR * NLUM,
            kind * NACT * NERR * (MAX_NB + 1),
            kind * 9,
            kind * (MAX_NB + 1) * 2)


class InterInfo:
    """Per-block prediction sources for a video frame.

    `bs_y`/`bs_x` are this plane's block size and `mv_sy`/`mv_sx` the motion
    vector divisors, so a subsampled chroma plane can reuse the luma decisions.
    """

    __slots__ = ("modes", "mvs", "bs_y", "bs_x", "mv_sy", "mv_sx", "ref_rows")

    def __init__(self, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref_rows):
        self.modes = modes
        self.mvs = mvs
        self.bs_y = bs_y
        self.bs_x = bs_x
        self.mv_sy = mv_sy
        self.mv_sx = mv_sx
        self.ref_rows = ref_rows


def code_plane(coder, encode, width, height, kind, model, src=None, luma_err=None,
               inter=None):
    """Code one plane. Returns (rows, err_rows) as lists of python ints.

    `src` is required when encoding (a list of row lists). `luma_err` supplies
    the co-located luma error magnitudes, which sharpen the chroma contexts a
    lot - chroma tends to go wrong exactly where luma did. `inter` switches on
    the video path, where each block picks its own prediction source.
    """
    zero_p = model["zero"]
    sign_p = model["sign"]
    nb_p = model["nb"]
    mant_p = model["mant"]
    bit = coder.bit
    bypass = coder.bypass
    bisect_right = bisect.bisect_right
    act_ladder, err_ladder, lum_ladder = ACT_LADDER, ERR_LADDER, LUMA_LADDER

    k_zero, k_nb, k_sign, k_mant = kind_offsets(kind)
    if inter is not None:
        i_zero, i_nb, i_sign, i_mant = kind_offsets(kind + INTER_KIND_OFFSET)
        nbx = len(inter.modes[0])
        ref_rows = inter.ref_rows

    rows = []
    err_rows = []
    prev = [0] * width
    prev_err = [0] * width

    for y in range(height):
        cur = [0] * width
        cur_err = [0] * width
        srow = src[y] if encode else None
        lrow = luma_err[y] if luma_err is not None else None

        if inter is not None:
            # Per-row lookup tables: which reference row and column each pixel
            # reads, resolved once per row instead of once per pixel.
            by = min(y // inter.bs_y, len(inter.modes) - 1)
            mode_row = inter.modes[by]
            mv_row = inter.mvs[by]
            mode_x = [0] * width
            ref_row_of_x = [None] * width
            ref_col_of_x = [0] * width
            for x in range(width):
                bx = min(x // inter.bs_x, nbx - 1)
                if mode_row[bx] == MODE_TEMPORAL:
                    mode_x[x] = 1
                    ry = y + int(mv_row[bx][0]) // inter.mv_sy
                    rx = x + int(mv_row[bx][1]) // inter.mv_sx
                    ref_row_of_x[x] = ref_rows[
                        0 if ry < 0 else (height - 1 if ry >= height else ry)]
                    ref_col_of_x[x] = 0 if rx < 0 else (width - 1 if rx >= width else rx)

        west = 0
        west_err = 0
        first_row = y == 0

        for x in range(width):
            if first_row:
                pred = 128 if x == 0 else west
                act = 0
                sgn = 4
            else:
                north = prev[x]
                if x == 0:
                    pred = north
                    act = 0
                    sgn = 4
                else:
                    nwest = prev[x - 1]
                    neast = prev[x + 1] if x + 1 < width else north
                    if nwest >= north:
                        if nwest >= west:
                            pred = north if north < west else west
                        else:
                            pred = north + west - nwest
                    elif nwest <= west:
                        pred = north if north > west else west
                    else:
                        pred = north + west - nwest
                    if pred > 255:
                        pred = 255
                    elif pred < 0:
                        pred = 0
                    d1 = ((west - nwest + 128) & 255) - 128
                    d2 = ((nwest - north + 128) & 255) - 128
                    d3 = ((north - neast + 128) & 255) - 128
                    act = ((d1 if d1 >= 0 else -d1) + (d2 if d2 >= 0 else -d2)
                           + (d3 if d3 >= 0 else -d3))
                    sgn = (0 if d1 < 0 else (1 if d1 == 0 else 2)) * 3 \
                        + (0 if d2 < 0 else (1 if d2 == 0 else 2))

            if inter is not None and mode_x[x]:
                pred = ref_row_of_x[x][ref_col_of_x[x]]
                b_zero, b_nb, b_sign, b_mant = i_zero, i_nb, i_sign, i_mant
            else:
                b_zero, b_nb, b_sign, b_mant = k_zero, k_nb, k_sign, k_mant

            if first_row:
                err_sum = west_err
            else:
                err_sum = west_err + prev_err[x] + (prev_err[x + 1] if x + 1 < width else 0)
            sub = bisect_right(act_ladder, act) * NERR + bisect_right(err_ladder, err_sum)
            lum = bisect_right(lum_ladder, lrow[x]) if lrow is not None else 0
            zctx = b_zero + sub * NLUM + lum
            nbbase = b_nb + sub * (MAX_NB + 1)

            if encode:
                d = ((srow[x] - pred + 128) & 255) - 128
                mag = -d if d < 0 else d
                bit(zero_p, zctx, 1 if mag else 0)
                if mag:
                    bit(sign_p, b_sign + sgn, 1 if d < 0 else 0)
                    v = mag - 1
                    nb = v.bit_length()
                    for i in range(nb):
                        bit(nb_p, nbbase + i, 1)
                    if nb < MAX_NB:
                        bit(nb_p, nbbase + nb, 0)
                    if nb >= 2:
                        bit(mant_p, b_mant + nb * 2, (v >> (nb - 2)) & 1)
                        if nb >= 3:
                            bit(mant_p, b_mant + nb * 2 + 1, (v >> (nb - 3)) & 1)
                            if nb > 3:
                                bypass(v & ((1 << (nb - 3)) - 1), nb - 3)
                value = (pred + d) & 255
            else:
                if bit(zero_p, zctx):
                    neg = bit(sign_p, b_sign + sgn)
                    nb = 0
                    while nb < MAX_NB and bit(nb_p, nbbase + nb):
                        nb += 1
                    if nb < 2:
                        v = nb
                    else:
                        v = (1 << (nb - 1)) | (bit(mant_p, b_mant + nb * 2) << (nb - 2))
                        if nb >= 3:
                            v |= bit(mant_p, b_mant + nb * 2 + 1) << (nb - 3)
                            if nb > 3:
                                v |= bypass(nb - 3)
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
