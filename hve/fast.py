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
 P_ADAPT) = range(18)


def _code_planes(encode, planes, data, out, params,
                 zero_p, dir_p, diff_p, match_p, sign_p, nb_p, mant_p,
                 mixw, apm0, apm1, stretch, sq,
                 act_l, err_l, lum_l, side_l, diff_l,
                 match_table, flat, errmap):
    nplanes = planes.shape[0]
    height = planes.shape[1]
    width = planes.shape[2]

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

    # coder state: encoder [low, range, cache, cache_size, outpos]
    #              decoder [code, range, inpos, 0, 0]
    st = np.zeros(5, dtype=np.int64)
    if encode:
        st[1] = 0xFFFFFFFF
        st[3] = 1
    else:
        st[1] = 0xFFFFFFFF
        st[0] = ((np.int64(data[1]) << 24) | (np.int64(data[2]) << 16)
                 | (np.int64(data[3]) << 8) | np.int64(data[4]))
        st[2] = 5

    prev = np.zeros(width, dtype=np.int64)
    cur = np.zeros(width, dtype=np.int64)
    prev_err = np.zeros(width, dtype=np.int64)
    cur_err = np.zeros(width, dtype=np.int64)
    terr_prev = np.zeros(width + 2, dtype=np.int64)
    terr_cur = np.zeros(width + 2, dtype=np.int64)
    werr_prev = np.zeros((4, width + 2), dtype=np.int64)
    werr_cur = np.zeros((4, width + 2), dtype=np.int64)
    ex = np.zeros(4, dtype=np.int64)

    for pi in range(nplanes):
        kind = pi if pi < 3 else 3
        use_luma = (pi == 1) or (pi == 2)

        k_zero = kind * nact * nerr * nlum
        k_nb = kind * nact * nerr * (max_nb + 1)
        k_sign = kind * 9
        k_mant = kind * (max_nb + 1) * 2
        kind_dir = kind * 9 * nside * nside
        kind_diff = kind * ndiff * ndiff
        kind_match = kind * nmatch
        kind_mix = kind * nact
        kind_nbapm = kind * (max_nb + 1) * nact

        match_table[:] = 0
        flat_n = 0
        match_pos = 0
        match_len = 0
        prev[:] = 0
        prev_err[:] = 0
        terr_prev[:] = 0
        terr_cur[:] = 0
        werr_prev[:, :] = 0
        werr_cur[:, :] = 0

        for y in range(height):
            cur[:] = 0
            cur_err[:] = 0
            west = 0
            west_err = 0
            first_row = y == 0

            for x in range(width):
                mval = -1
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
                zctx = k_zero + sub * nlum + lum
                nbbase = k_nb + sub * (max_nb + 1)

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
                if pr1 < 1:
                    pr1 = 1
                elif pr1 > 4095:
                    pr1 = 4095
                p_zero = (4096 - pr1) << 3

                if encode:
                    d = ((np.int64(planes[pi, y, x]) - pred + 128) & 255) - 128
                    mag = -d if d < 0 else d
                    nonzero = 1 if mag else 0
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

                if encode:
                    if mag:
                        _enc_bit(st, out, sign_p, k_sign + sgn, 1 if d < 0 else 0)
                        v = mag - 1
                        nb = 0
                        t = v
                        while t:
                            nb += 1
                            t >>= 1
                        limit = nb if nb < max_nb else nb - 1
                        for i in range(limit + 1):
                            more = 1 if i < nb else 0
                            ctx = nbbase + i
                            pr = 4095 - (nb_p[ctx] >> 3)
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
                            t2 = 65535 if more else 0
                            apm1[u2] += (t2 - apm1[u2]) >> 7
                        if nb >= 2:
                            _enc_bit(st, out, mant_p, k_mant + nb * 2, (v >> (nb - 2)) & 1)
                            if nb >= 3:
                                _enc_bit(st, out, mant_p, k_mant + nb * 2 + 1,
                                         (v >> (nb - 3)) & 1)
                                if nb > 3:
                                    _enc_bypass(st, out, v & ((1 << (nb - 3)) - 1), nb - 3)
                    value = (pred + d) & 255
                else:
                    if nonzero:
                        neg = _dec_bit(st, data, sign_p, k_sign + sgn)
                        nb = 0
                        while nb < max_nb:
                            ctx = nbbase + nb
                            pr = 4095 - (nb_p[ctx] >> 3)
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
                            t2 = 65535 if more else 0
                            apm1[u2] += (t2 - apm1[u2]) >> 7
                            if not more:
                                break
                            nb += 1
                        if nb < 2:
                            v = nb
                        else:
                            v = (1 << (nb - 1)) | (
                                _dec_bit(st, data, mant_p, k_mant + nb * 2) << (nb - 2))
                            if nb >= 3:
                                v |= _dec_bit(st, data, mant_p,
                                              k_mant + nb * 2 + 1) << (nb - 3)
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
                    planes[pi, y, x] = value

            if pi == 0:
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

    if encode:
        for _ in range(5):
            _shift_low(st, out)
        return st[4]
    return 0


_code_planes = _jit(_code_planes)


# --------------------------------------------------------------------------
# wrappers


def available():
    return HAVE_NUMBA


def _params():
    """Tunables read at call time, so tools/ sweeps still take effect."""
    return np.array([
        model.NACT, model.NERR, model.NLUM, model.NSIDE, model.NDIFF,
        model.NMATCH, model.MAX_NB, model.MATCH_MAX_LEN, model.MATCH_TRUST,
        model.WP_P1, model.WP_P2, model.WP_SHIFT,
        model.WP_MAXW[0], model.WP_MAXW[1], model.WP_MAXW[2], model.WP_MAXW[3],
        model.MATCH_HASH_MASK, rc.ADAPT_SHIFT,
    ], dtype=np.int64)


_BANK_KEYS = ("zero", "zero_dir", "zero_diff", "zero_match", "sign", "nb", "mant")


def _arrays(height, width):
    bank = model.new_model()
    arrs = [np.array(bank[k], dtype=np.int64) for k in _BANK_KEYS]
    arrs.append(np.array(bank["zero_mix"].weights, dtype=np.int64))
    arrs.append(np.array(bank["zero_apm"].table, dtype=np.int64))
    arrs.append(np.array(bank["nb_apm"].table, dtype=np.int64))
    ladders = [np.array(x, dtype=np.int64) for x in
               (model.ACT_LADDER, model.ERR_LADDER, model.LUMA_LADDER,
                model.SIDE_LADDER, model.DIFF_LADDER)]
    tables = [np.array(mix.STRETCH, dtype=np.int64),
              np.array(mix.SQUASH, dtype=np.int64)]
    scratch = [np.zeros(model.MATCH_HASH_MASK + 1, dtype=np.int64),
               np.zeros(height * width, dtype=np.int64),
               np.zeros((height, width), dtype=np.int64)]
    return arrs, tables, ladders, scratch


def encode_planes(planes_u8):
    """Range-coder payload for (C, H, W) uint8 planes, byte-identical to model.py."""
    planes = np.ascontiguousarray(planes_u8, dtype=np.int64)
    _, height, width = planes.shape
    arrs, tables, ladders, scratch = _arrays(height, width)
    out = np.zeros(planes.size * 2 + 65536, dtype=np.uint8)
    n = _code_planes(True, planes, np.zeros(1, dtype=np.uint8), out, _params(),
                     *arrs, *tables, *ladders, *scratch)
    return bytes(out[:int(n)])


def decode_planes(payload, channels, height, width):
    """Inverse of encode_planes."""
    planes = np.zeros((channels, height, width), dtype=np.int64)
    data = np.frombuffer(payload, dtype=np.uint8)
    arrs, tables, ladders, scratch = _arrays(height, width)
    _code_planes(False, planes, data, np.zeros(1, dtype=np.uint8), _params(),
                 *arrs, *tables, *ladders, *scratch)
    return planes.astype(np.uint8)
