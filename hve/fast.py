"""numba-jitted still-image path. Must stay byte-identical to hve/model.py.

`hve/model.py` is the reference: readable, pure Python, and the definition of
the format. This file is the same loop expressed over flat numpy arrays so numba
can compile it, roughly a hundred times faster.

Two implementations of one loop is exactly the mistake that bit this project
before, when video.py carried a divergent copy of the pixel loop. The difference
here is `test_fast_path_is_byte_identical`, which encodes the same images both
ways and requires the bitstreams to match exactly. A compression format cannot
tolerate a one-bit difference — it would silently corrupt every file written by
whichever path happened to run — so drift has to fail loudly, and it does.

Video still uses the reference path; only stills are jitted so far.
"""

import numpy as np

from . import mix, model, rc

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                   # pure-Python fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


def _jit(fn):
    return njit(cache=True)(fn)


# --------------------------------------------------------------------------
# primitives, mirroring hve/rc.py and hve/mix.py


def _bisect_right(ladder, v):
    """bisect.bisect_right over a small sorted array."""
    i = 0
    n = ladder.shape[0]
    while i < n and ladder[i] <= v:
        i += 1
    return i


def _squash(sq, x):
    if x <= -2047:
        return 1
    if x >= 2047:
        return 4095
    return sq[x + 2047]


def _shift_low(st, out):
    low = st[0]
    if low < 0xFF000000 or low > 0xFFFFFFFF:
        carry = low >> 32
        out[st[4]] = (st[2] + carry) & 0xFF
        st[4] += 1
        if st[3] > 1:
            filler = (0xFF + carry) & 0xFF
            for _ in range(st[3] - 1):
                out[st[4]] = filler
                st[4] += 1
        st[2] = (low >> 24) & 0xFF
        st[3] = 0
    st[3] += 1
    st[0] = (low << 8) & 0xFFFFFFFF


def _enc_bit_p(st, out, p, value):
    bound = (st[1] >> 15) * p
    if value:
        st[0] += bound
        st[1] -= bound
    else:
        st[1] = bound
    while st[1] < 16777216:
        st[1] = (st[1] << 8) & 0xFFFFFFFF
        _shift_low(st, out)


def _enc_bit(st, out, probs, ctx, value):
    p = probs[ctx]
    bound = (st[1] >> 15) * p
    if value:
        st[0] += bound
        st[1] -= bound
        probs[ctx] = p - (p >> 6)
    else:
        st[1] = bound
        probs[ctx] = p + ((32768 - p) >> 6)
    while st[1] < 16777216:
        st[1] = (st[1] << 8) & 0xFFFFFFFF
        _shift_low(st, out)


def _enc_bypass(st, out, value, nbits):
    for i in range(nbits - 1, -1, -1):
        st[1] = st[1] >> 1
        if (value >> i) & 1:
            st[0] += st[1]
        while st[1] < 16777216:
            st[1] = (st[1] << 8) & 0xFFFFFFFF
            _shift_low(st, out)


def _dec_bit_p(st, data, p):
    bound = (st[1] >> 15) * p
    if st[0] < bound:
        st[1] = bound
        v = 0
    else:
        st[0] -= bound
        st[1] -= bound
        v = 1
    while st[1] < 16777216:
        st[1] = st[1] << 8
        st[0] = ((st[0] << 8) | data[st[2]]) & 0xFFFFFFFF
        st[2] += 1
    return v


def _dec_bit(st, data, probs, ctx):
    p = probs[ctx]
    bound = (st[1] >> 15) * p
    if st[0] < bound:
        st[1] = bound
        probs[ctx] = p + ((32768 - p) >> 6)
        v = 0
    else:
        st[0] -= bound
        st[1] -= bound
        probs[ctx] = p - (p >> 6)
        v = 1
    while st[1] < 16777216:
        st[1] = st[1] << 8
        st[0] = ((st[0] << 8) | data[st[2]]) & 0xFFFFFFFF
        st[2] += 1
    return v


def _dec_bypass(st, data, nbits):
    value = 0
    for _ in range(nbits):
        st[1] = st[1] >> 1
        if st[0] >= st[1]:
            st[0] -= st[1]
            value = (value << 1) | 1
        else:
            value = value << 1
        while st[1] < 16777216:
            st[1] = st[1] << 8
            st[0] = ((st[0] << 8) | data[st[2]]) & 0xFFFFFFFF
            st[2] += 1
    return value


_bisect_right = _jit(_bisect_right)
_squash = _jit(_squash)
_shift_low = _jit(_shift_low)
_enc_bit_p = _jit(_enc_bit_p)
_enc_bit = _jit(_enc_bit)
_enc_bypass = _jit(_enc_bypass)
_dec_bit_p = _jit(_dec_bit_p)
_dec_bit = _jit(_dec_bit)
_dec_bypass = _jit(_dec_bypass)

# --------------------------------------------------------------------------
# the plane loop, mirroring model.code_plane

# Indices into the params array. Tunables are passed rather than closed over so
# the parameter sweeps in tools/ can still move them at runtime.
(P_NACT, P_NERR, P_NLUM, P_NSIDE, P_NDIFF, P_NMATCH, P_MAXNB, P_MATCHMAX,
 P_MATCHTRUST, P_WP1, P_WP2, P_WPSHIFT, P_W0, P_W1, P_W2, P_W3, P_HASHMASK,
 P_ADAPT, P_MVMAX) = range(19)


def _finish_encode(st, out):
    for _ in range(5):
        _shift_low(st, out)
    return st[4]


def _code_mv(encode, st, data, out, mv_zero, mv_sign, mv_mag, axis, mv_max, value):
    base = axis * (mv_max + 1)
    if encode:
        mag = -value if value < 0 else value
        _enc_bit(st, out, mv_zero, axis, 1 if mag else 0)
        if mag:
            _enc_bit(st, out, mv_sign, axis, 1 if value < 0 else 0)
            for i in range(mag - 1):
                _enc_bit(st, out, mv_mag, base + i, 1)
            if mag - 1 < mv_max:
                _enc_bit(st, out, mv_mag, base + mag - 1, 0)
        return value
    if not _dec_bit(st, data, mv_zero, axis):
        return 0
    neg = _dec_bit(st, data, mv_sign, axis)
    mag = 1
    while mag - 1 < mv_max and _dec_bit(st, data, mv_mag, base + mag - 1):
        mag += 1
    return -mag if neg else mag


def _code_block_info(encode, st, data, out, mode_p, mv_zero, mv_sign, mv_mag,
                     modes, mvs, mv_max):
    """Per-block modes and motion vectors, differential against the left block."""
    nby = modes.shape[0]
    nbx = modes.shape[1]
    for by in range(nby):
        pred_y = 0
        pred_x = 0
        for bx in range(nbx):
            left = modes[by, bx - 1] if bx else 0
            up = modes[by - 1, bx] if by else 0
            ctx = left * 2 + up
            if encode:
                _enc_bit(st, out, mode_p, ctx, modes[by, bx])
            else:
                modes[by, bx] = _dec_bit(st, data, mode_p, ctx)
            if modes[by, bx] == 1:
                if encode:
                    _code_mv(True, st, data, out, mv_zero, mv_sign, mv_mag, 0,
                             mv_max, mvs[by, bx, 0] - pred_y)
                    _code_mv(True, st, data, out, mv_zero, mv_sign, mv_mag, 1,
                             mv_max, mvs[by, bx, 1] - pred_x)
                else:
                    dy = _code_mv(False, st, data, out, mv_zero, mv_sign, mv_mag, 0,
                                  mv_max, 0)
                    dx = _code_mv(False, st, data, out, mv_zero, mv_sign, mv_mag, 1,
                                  mv_max, 0)
                    mvs[by, bx, 0] = dy + pred_y
                    mvs[by, bx, 1] = dx + pred_x
                pred_y = mvs[by, bx, 0]
                pred_x = mvs[by, bx, 1]
            else:
                pred_y = 0
                pred_x = 0


def _code_plane1(encode, plane, data, out, st, params,
                 zero_p, dir_p, diff_p, match_p, sign_p, nb_p, nbm_p, mant_p,
                 mixw, nbmixw, apm0, apm1, apm2, stretch, sq,
                 act_l, err_l, lum_l, side_l, diff_l, mexp_l,
                 match_table, flat, errmap, stats,
                 kind, use_luma, write_errmap,
                 inter_on, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref_plane):
    height = plane.shape[0]
    width = plane.shape[1]

    nact = params[P_NACT]
    nerr = params[P_NERR]
    nlum = params[P_NLUM]
    nside = params[P_NSIDE]
    ndiff = params[P_NDIFF]
    nmatch = params[P_NMATCH]
    max_nb = params[P_MAXNB]
    match_max = params[P_MATCHMAX]
    match_trust = params[P_MATCHTRUST]
    wp_p1 = params[P_WP1]
    wp_p2 = params[P_WP2]
    wp_shift = params[P_WPSHIFT]
    wp_w0 = params[P_W0]
    wp_w1 = params[P_W1]
    wp_w2 = params[P_W2]
    wp_w3 = params[P_W3]
    hash_mask = params[P_HASHMASK]
    adapt = params[P_ADAPT]

    k_zero = kind * nact * nerr * nlum
    k_nb = kind * nact * nerr * (max_nb + 1)
    k_sign = kind * 9 * 4
    k_mant = kind * (max_nb + 1) * 2
    ikind = kind + 4
    i_zero = ikind * nact * nerr * nlum
    i_nb = ikind * nact * nerr * (max_nb + 1)
    i_sign = ikind * 9 * 4
    i_mant = ikind * (max_nb + 1) * 2
    kind_dir = kind * 9 * nside * nside
    kind_diff = kind * ndiff * ndiff
    kind_match = kind * nmatch
    kind_mix = kind * nact
    kind_nbapm = kind * (max_nb + 1) * nact

    prev = np.zeros(width, dtype=np.int64)
    cur = np.zeros(width, dtype=np.int64)
    prev_err = np.zeros(width, dtype=np.int64)
    cur_err = np.zeros(width, dtype=np.int64)
    terr_prev = np.zeros(width + 2, dtype=np.int64)
    terr_cur = np.zeros(width + 2, dtype=np.int64)
    werr_prev = np.zeros((4, width + 2), dtype=np.int64)
    werr_cur = np.zeros((4, width + 2), dtype=np.int64)
    ex = np.zeros(4, dtype=np.int64)
    mode_x = np.zeros(width, dtype=np.int64)
    ref_y = np.zeros(width, dtype=np.int64)
    ref_x = np.zeros(width, dtype=np.int64)

    match_table[:] = 0
    flat_n = 0
    match_pos = 0
    match_len = 0
    nbx = modes.shape[1]
    nby = modes.shape[0]

    for y in range(height):
        cur[:] = 0
        cur_err[:] = 0
        west = 0
        west_err = 0
        first_row = y == 0

        if inter_on:
            by = y // bs_y
            if by >= nby:
                by = nby - 1
            for x in range(width):
                bx = x // bs_x
                if bx >= nbx:
                    bx = nbx - 1
                if modes[by, bx] == 1:
                    mode_x[x] = 1
                    ry = y + mvs[by, bx, 0] // mv_sy
                    rx = x + mvs[by, bx, 1] // mv_sx
                    ref_y[x] = 0 if ry < 0 else (height - 1 if ry >= height else ry)
                    ref_x[x] = 0 if rx < 0 else (width - 1 if rx >= width else rx)
                else:
                    mode_x[x] = 0

        for x in range(width):
            mval = -1
            msign = 0
            mexp_b = 0
            north = 0
            nwest = 0
            neast = 0
            q0 = 0
            q1 = 0
            q2 = 0
            q3 = 0
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
                    sgn = ((0 if d1 < 0 else (1 if d1 == 0 else 2)) * 3
                           + (0 if d2 < 0 else (1 if d2 == 0 else 2)))

                    te_w = terr_cur[x]
                    te_n = terr_prev[x + 1]
                    te_nw = terr_prev[x]
                    te_ne = terr_prev[x + 2]
                    sum_wn = te_n + te_w
                    q0 = west + neast - north
                    q1 = north - (((sum_wn + te_ne) * wp_p1) >> 5)
                    q2 = west - (((sum_wn + te_nw) * wp_p2) >> 5)
                    q3 = pred
                    a0 = (werr_cur[0, x] + werr_prev[0, x]
                          + werr_prev[0, x + 1] + werr_prev[0, x + 2])
                    a1 = (werr_cur[1, x] + werr_prev[1, x]
                          + werr_prev[1, x + 1] + werr_prev[1, x + 2])
                    a2 = (werr_cur[2, x] + werr_prev[2, x]
                          + werr_prev[2, x + 1] + werr_prev[2, x + 2])
                    a3 = (werr_cur[3, x] + werr_prev[3, x]
                          + werr_prev[3, x + 1] + werr_prev[3, x + 2])
                    m0 = (wp_w0 << wp_shift) // (a0 + 4) + 4
                    m1 = (wp_w1 << wp_shift) // (a1 + 4) + 4
                    m2 = (wp_w2 << wp_shift) // (a2 + 4) + 4
                    m3 = (wp_w3 << wp_shift) // (a3 + 4) + 4
                    total = m0 + m1 + m2 + m3
                    blend = (q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3
                             + (total >> 1)) // total
                    agree_n = te_n >= 0
                    agree_w = te_w >= 0
                    agree_nw = te_nw >= 0
                    if not ((agree_n == agree_w) and (agree_w == agree_nw)):
                        lo = north if north < west else west
                        hi = north if north > west else west
                        if neast < lo:
                            lo = neast
                        elif neast > hi:
                            hi = neast
                        if blend < lo:
                            blend = lo
                        elif blend > hi:
                            blend = hi
                    pred = 255 if blend > 255 else (0 if blend < 0 else blend)

            if (not first_row) and x:
                mhash = ((west * 0x2F0FD693 + north * 0x9E3779B1
                          + nwest * 0x85EBCA77 + neast * 0xC2B2AE3D)
                         >> 8) & hash_mask
                if match_len == 0:
                    match_pos = match_table[mhash]
                if 0 < match_pos < flat_n:
                    mval = flat[match_pos]
                match_table[mhash] = flat_n

            if inter_on and mode_x[x]:
                pred = ref_plane[ref_y[x], ref_x[x]]
                b_zero = i_zero
                b_nb = i_nb
                b_sign = i_sign
                b_mant = i_mant
                b_kind = ikind
            else:
                b_zero = k_zero
                b_nb = k_nb
                b_sign = k_sign
                b_mant = k_mant
                b_kind = kind

            north_err = 0 if first_row else prev_err[x]
            if first_row:
                err_sum = west_err
            else:
                err_sum = west_err + north_err
                if x + 1 < width:
                    err_sum += prev_err[x + 1]
            act_b = _bisect_right(act_l, act)
            sub = act_b * nerr + _bisect_right(err_l, err_sum)
            lum = _bisect_right(lum_l, errmap[y, x]) if use_luma else 0
            zctx = b_zero + sub * nlum + lum
            nbbase = b_nb + sub * (max_nb + 1)

            dir_ctx = (kind_dir
                       + (sgn * nside + _bisect_right(side_l, west_err)) * nside
                       + _bisect_right(side_l, north_err))
            if first_row or x == 0:
                diff_ctx = kind_diff
            else:
                dwn = west - north
                dne = nwest - neast
                diff_ctx = (kind_diff
                            + _bisect_right(diff_l, dwn if dwn >= 0 else -dwn) * ndiff
                            + _bisect_right(diff_l, dne if dne >= 0 else -dne))
            if mval < 0:
                match_ctx = kind_match
                msign = 0
            else:
                if mval == pred:
                    agree = 0
                elif -3 < mval - pred < 3:
                    agree = 1
                else:
                    agree = 2
                hit = match_len if match_len < match_max else match_max
                match_ctx = kind_match + 1 + hit * 3 + agree
                if match_len >= match_trust:
                    pred = mval
                mexp = mval - pred
                msign = 1 if mexp < 0 else (2 if mexp == 0 else 3)
                ae = mexp if mexp >= 0 else -mexp
                mexp_b = 1 + _bisect_right(mexp_l, ae)

            ex[0] = stretch[4095 - (zero_p[zctx] >> 3)]
            ex[1] = stretch[4095 - (dir_p[dir_ctx] >> 3)]
            ex[2] = stretch[4095 - (diff_p[diff_ctx] >> 3)]
            ex[3] = stretch[4095 - (match_p[match_ctx] >> 3)]
            mix_ctx = kind_mix + act_b
            mbase = mix_ctx * 4
            dot = (ex[0] * mixw[mbase] + ex[1] * mixw[mbase + 1]
                   + ex[2] * mixw[mbase + 2] + ex[3] * mixw[mbase + 3])
            pr_mix = _squash(sq, dot >> 16)

            s = stretch[pr_mix] + 2048
            aw = s & 127
            aidx = mix_ctx * 33 + (s >> 7)
            refined = (apm0[aidx] * (128 - aw) + apm0[aidx + 1] * aw) >> 11
            aupd = aidx + (1 if aw >= 64 else 0)
            pr1 = (pr_mix + 3 * refined) >> 2
            s3 = stretch[pr1] + 2048
            aw3 = s3 & 127
            aidx3 = ((b_kind * 7 + mexp_b) * 4 + msign) * 33 + (s3 >> 7)
            ref3 = (apm2[aidx3] * (128 - aw3) + apm2[aidx3 + 1] * aw3) >> 11
            aupd3 = aidx3 + (1 if aw3 >= 64 else 0)
            pr1 = (pr1 + 3 * ref3) >> 2
            if pr1 < 1:
                pr1 = 1
            elif pr1 > 4095:
                pr1 = 4095
            p_zero = (4096 - pr1) << 3

            if encode:
                d = ((plane[y, x] - pred + 128) & 255) - 128
                mag = -d if d < 0 else d
                nonzero = 1 if mag else 0
                stats[0] += 1
                stats[1] += nonzero
                _enc_bit_p(st, out, p_zero, nonzero)
            else:
                nonzero = _dec_bit_p(st, data, p_zero)
                d = 0
                mag = 0

            p = zero_p[zctx]
            zero_p[zctx] = (p - (p >> adapt)) if nonzero else (p + ((32768 - p) >> adapt))
            p = dir_p[dir_ctx]
            dir_p[dir_ctx] = (p - (p >> adapt)) if nonzero else (p + ((32768 - p) >> adapt))
            p = diff_p[diff_ctx]
            diff_p[diff_ctx] = (p - (p >> adapt)) if nonzero else (p + ((32768 - p) >> adapt))
            p = match_p[match_ctx]
            match_p[match_ctx] = (p - (p >> adapt)) if nonzero else (p + ((32768 - p) >> adapt))
            err = ((nonzero << 12) - pr_mix) * 7
            mixw[mbase] += (ex[0] * err + 0x8000) >> 16
            mixw[mbase + 1] += (ex[1] * err + 0x8000) >> 16
            mixw[mbase + 2] += (ex[2] * err + 0x8000) >> 16
            mixw[mbase + 3] += (ex[3] * err + 0x8000) >> 16
            target = 65535 if nonzero else 0
            apm0[aupd] += (target - apm0[aupd]) >> 7
            apm2[aupd3] += (target - apm2[aupd3]) >> 7

            if encode:
                if mag:
                    _enc_bit(st, out, sign_p, b_sign + sgn * 4 + msign, 1 if d < 0 else 0)
                    v = mag - 1
                    nb = 0
                    t = v
                    while t:
                        nb += 1
                        t >>= 1
                    stats[3] += nb
                    limit = nb if nb < max_nb else nb - 1
                    for i in range(limit + 1):
                        more = 1 if i < nb else 0
                        ctx = nbbase + i
                        mixc = b_kind * (max_nb + 1) + i
                        mctx = mixc * 7 + mexp_b
                        nb0 = stretch[4095 - (nb_p[ctx] >> 3)]
                        nb1 = stretch[4095 - (nbm_p[mctx] >> 3)]
                        nmb = mixc * 2
                        ndot = nb0 * nbmixw[nmb] + nb1 * nbmixw[nmb + 1]
                        pr = _squash(sq, ndot >> 16)
                        pr_nbmix = pr
                        actx = kind_nbapm + i * nact + act_b
                        s2 = stretch[pr] + 2048
                        w2 = s2 & 127
                        i2 = actx * 33 + (s2 >> 7)
                        ref2 = (apm1[i2] * (128 - w2) + apm1[i2 + 1] * w2) >> 11
                        u2 = i2 + (1 if w2 >= 64 else 0)
                        pr = (pr + 3 * ref2) >> 2
                        if pr < 1:
                            pr = 1
                        elif pr > 4095:
                            pr = 4095
                        _enc_bit_p(st, out, (4096 - pr) << 3, more)
                        p = nb_p[ctx]
                        nb_p[ctx] = ((p - (p >> adapt)) if more
                                     else (p + ((32768 - p) >> adapt)))
                        p = nbm_p[mctx]
                        nbm_p[mctx] = ((p - (p >> adapt)) if more
                                       else (p + ((32768 - p) >> adapt)))
                        nerr2 = ((more << 12) - pr_nbmix) * 7
                        nbmixw[nmb] += (nb0 * nerr2 + 0x8000) >> 16
                        nbmixw[nmb + 1] += (nb1 * nerr2 + 0x8000) >> 16
                        t2 = 65535 if more else 0
                        apm1[u2] += (t2 - apm1[u2]) >> 7
                    if nb >= 2:
                        _enc_bit(st, out, mant_p, b_mant + nb * 2, (v >> (nb - 2)) & 1)
                        if nb >= 3:
                            _enc_bit(st, out, mant_p, b_mant + nb * 2 + 1,
                                     (v >> (nb - 3)) & 1)
                            if nb > 3:
                                _enc_bypass(st, out, v & ((1 << (nb - 3)) - 1), nb - 3)
                                stats[2] += nb - 3
                value = (pred + d) & 255
            else:
                if nonzero:
                    neg = _dec_bit(st, data, sign_p, b_sign + sgn * 4 + msign)
                    nb = 0
                    while nb < max_nb:
                        ctx = nbbase + nb
                        mixc = b_kind * (max_nb + 1) + nb
                        mctx = mixc * 7 + mexp_b
                        nb0 = stretch[4095 - (nb_p[ctx] >> 3)]
                        nb1 = stretch[4095 - (nbm_p[mctx] >> 3)]
                        nmb = mixc * 2
                        ndot = nb0 * nbmixw[nmb] + nb1 * nbmixw[nmb + 1]
                        pr = _squash(sq, ndot >> 16)
                        pr_nbmix = pr
                        actx = kind_nbapm + nb * nact + act_b
                        s2 = stretch[pr] + 2048
                        w2 = s2 & 127
                        i2 = actx * 33 + (s2 >> 7)
                        ref2 = (apm1[i2] * (128 - w2) + apm1[i2 + 1] * w2) >> 11
                        u2 = i2 + (1 if w2 >= 64 else 0)
                        pr = (pr + 3 * ref2) >> 2
                        if pr < 1:
                            pr = 1
                        elif pr > 4095:
                            pr = 4095
                        more = _dec_bit_p(st, data, (4096 - pr) << 3)
                        p = nb_p[ctx]
                        nb_p[ctx] = ((p - (p >> adapt)) if more
                                     else (p + ((32768 - p) >> adapt)))
                        p = nbm_p[mctx]
                        nbm_p[mctx] = ((p - (p >> adapt)) if more
                                       else (p + ((32768 - p) >> adapt)))
                        nerr2 = ((more << 12) - pr_nbmix) * 7
                        nbmixw[nmb] += (nb0 * nerr2 + 0x8000) >> 16
                        nbmixw[nmb + 1] += (nb1 * nerr2 + 0x8000) >> 16
                        t2 = 65535 if more else 0
                        apm1[u2] += (t2 - apm1[u2]) >> 7
                        if not more:
                            break
                        nb += 1
                    if nb < 2:
                        v = nb
                    else:
                        v = (1 << (nb - 1)) | (
                            _dec_bit(st, data, mant_p, b_mant + nb * 2) << (nb - 2))
                        if nb >= 3:
                            v |= _dec_bit(st, data, mant_p,
                                          b_mant + nb * 2 + 1) << (nb - 3)
                            if nb > 3:
                                v |= _dec_bypass(st, data, nb - 3)
                    mag = v + 1
                    value = (pred - mag if neg else pred + mag) & 255
                else:
                    mag = 0
                    value = pred & 255

            if (not first_row) and x:
                terr_cur[x + 1] = pred - value
                e0 = q0 - value
                e1 = q1 - value
                e2 = q2 - value
                e3 = q3 - value
                werr_cur[0, x + 1] = e0 if e0 >= 0 else -e0
                werr_cur[1, x + 1] = e1 if e1 >= 0 else -e1
                werr_cur[2, x + 1] = e2 if e2 >= 0 else -e2
                werr_cur[3, x + 1] = e3 if e3 >= 0 else -e3

            if mval == value:
                match_pos += 1
                match_len += 1
            else:
                match_len = 0
            flat[flat_n] = value
            flat_n += 1

            cur[x] = value
            cur_err[x] = mag
            west = value
            west_err = mag
            if not encode:
                plane[y, x] = value

        if write_errmap:
            for xx in range(width):
                errmap[y, xx] = cur_err[xx]
        for xx in range(width):
            prev[xx] = cur[xx]
            prev_err[xx] = cur_err[xx]
        for xx in range(width + 2):
            terr_prev[xx] = terr_cur[xx]
            terr_cur[xx] = 0
        for k in range(4):
            for xx in range(width + 2):
                werr_prev[k, xx] = werr_cur[k, xx]
                werr_cur[k, xx] = 0


_finish_encode = _jit(_finish_encode)
_code_mv = _jit(_code_mv)
_code_block_info = _jit(_code_block_info)
_code_plane1 = _jit(_code_plane1)


# --------------------------------------------------------------------------
# wrappers


def available():
    return HAVE_NUMBA


def _params():
    """Tunables read at call time, so tools/ sweeps still take effect."""
    from . import video as _video
    return np.array([
        model.NACT, model.NERR, model.NLUM, model.NSIDE, model.NDIFF,
        model.NMATCH, model.MAX_NB, model.MATCH_MAX_LEN, model.MATCH_TRUST,
        model.WP_P1, model.WP_P2, model.WP_SHIFT,
        model.WP_MAXW[0], model.WP_MAXW[1], model.WP_MAXW[2], model.WP_MAXW[3],
        model.MATCH_HASH_MASK, rc.ADAPT_SHIFT, _video.MV_MAX,
    ], dtype=np.int64)


_BANK_KEYS = ("zero", "zero_dir", "zero_diff", "zero_match", "sign", "nb",
              "nb_match", "mant")


class Bank:
    """The adaptive state, as numpy arrays the kernel can mutate in place.

    For stills this lives for one image; for video it lives for the whole clip,
    exactly as the Python `model.new_model()` dict does.
    """

    def __init__(self, video=False):
        bank = model.new_model()
        self.arrs = [np.array(bank[k], dtype=np.int64) for k in _BANK_KEYS]
        self.mixw = np.array(bank["zero_mix"].weights, dtype=np.int64)
        self.nbmixw = np.array(bank["nb_mix"].weights, dtype=np.int64)
        self.apm0 = np.array(bank["zero_apm"].table, dtype=np.int64)
        self.apm1 = np.array(bank["nb_apm"].table, dtype=np.int64)
        self.apm2 = np.array(bank["zero_apm2"].table, dtype=np.int64)
        self.ladders = [np.array(x, dtype=np.int64) for x in
                        (model.ACT_LADDER, model.ERR_LADDER, model.LUMA_LADDER,
                         model.SIDE_LADDER, model.DIFF_LADDER,
                         model.MEXP_LADDER)]
        self.tables = [np.array(mix.STRETCH, dtype=np.int64),
                       np.array(mix.SQUASH, dtype=np.int64)]
        self.match_table = np.zeros(model.MATCH_HASH_MASK + 1, dtype=np.int64)
        # [pixels, nonzero, bypass bits, sum of nb, zero-flag cost*1024]
        self.stats = np.zeros(8, dtype=np.int64)
        self.params = _params()
        if video:
            from . import video as _video
            vb = _video.new_video_model()
            self.mode_p = np.array(vb["mode"], dtype=np.int64)
            self.mv_zero = np.array(vb["mv_zero"], dtype=np.int64)
            self.mv_sign = np.array(vb["mv_sign"], dtype=np.int64)
            self.mv_mag = np.array(vb["mv_mag"], dtype=np.int64)

    def code(self, coder, encode, plane, kind, use_luma, write_errmap, errmap,
             flat, inter=None):
        if inter is None:
            on, modes, mvs = 0, _DUMMY_MODES, _DUMMY_MVS
            bs_y = bs_x = mv_sy = mv_sx = 1
            ref = _DUMMY_PLANE
        else:
            on, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref = inter
        _code_plane1(encode, plane, coder.data, coder.out, coder.st, self.params,
                     *self.arrs, self.mixw, self.nbmixw, self.apm0, self.apm1, self.apm2, *self.tables,
                     *self.ladders, self.match_table, flat, errmap, self.stats,
                     kind, use_luma, write_errmap,
                     on, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref)


_DUMMY_MODES = np.zeros((1, 1), dtype=np.int64)
_DUMMY_MVS = np.zeros((1, 1, 2), dtype=np.int64)
_DUMMY_PLANE = np.zeros((1, 1), dtype=np.int64)


class Coder:
    """Range-coder state shared across planes and frames of one stream."""

    def __init__(self, encode, capacity=0, payload=None):
        self.encode = encode
        self.st = np.zeros(5, dtype=np.int64)
        self.st[1] = 0xFFFFFFFF
        if encode:
            self.st[3] = 1
            self.out = np.zeros(capacity, dtype=np.uint8)
            self.data = np.zeros(1, dtype=np.uint8)
        else:
            self.data = np.frombuffer(payload, dtype=np.uint8).copy()
            self.out = np.zeros(1, dtype=np.uint8)
            self.st[0] = int.from_bytes(bytes(self.data[1:5]), "big")
            self.st[2] = 5

    def finish(self):
        return bytes(self.out[:int(_finish_encode(self.st, self.out))])


def encode_planes(planes_u8):
    """Range-coder payload for (C, H, W) uint8 planes, byte-identical to model.py."""
    planes = np.ascontiguousarray(planes_u8, dtype=np.int64)
    channels, height, width = planes.shape
    bank = Bank()
    coder = Coder(True, capacity=planes.size * 2 + 65536)
    errmap = np.zeros((height, width), dtype=np.int64)
    flat = np.zeros(height * width, dtype=np.int64)
    for i in range(channels):
        bank.code(coder, True, planes[i], min(i, 3), 1 if i in (1, 2) else 0,
                  1 if i == 0 else 0, errmap, flat)
    return coder.finish()


def decode_planes(payload, channels, height, width):
    """Inverse of encode_planes."""
    planes = np.zeros((channels, height, width), dtype=np.int64)
    bank = Bank()
    coder = Coder(False, payload=payload)
    errmap = np.zeros((height, width), dtype=np.int64)
    flat = np.zeros(height * width, dtype=np.int64)
    for i in range(channels):
        bank.code(coder, False, planes[i], min(i, 3), 1 if i in (1, 2) else 0,
                  1 if i == 0 else 0, errmap, flat)
    return planes.astype(np.uint8)
