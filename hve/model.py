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
from array import array

from . import mix, rc

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


# Secondary contexts for the zero flag's extra experts. The point of mixing is
# that the experts disagree, so these deliberately look at the neighbourhood
# differently from the primary activity-and-error context.
DIFF_LADDER = [1, 2, 4, 8, 16, 32, 64]         # 8 buckets
NDIFF = len(DIFF_LADDER) + 1
SIDE_LADDER = [1, 2, 4, 8]                     # 5 buckets per axis
NSIDE = len(SIDE_LADDER) + 1
ZERO_EXPERTS = 5
# Expert 5 reads the learned combiner's own confidence, which nothing else in
# the model can see. Two numbers fall out of the combiner for free: how much its
# inputs disagreed (the input energy) and how far it had to move the blend to
# satisfy them (the correction). Wide disagreement or a large correction both
# say "this pixel is unlikely to predict exactly", which is precisely the
# question the zero flag asks. This is the half of the rebuild where logistic
# mixing genuinely applies - it combines probabilities, so it belongs on the
# binary decisions rather than on the pixel value.
CONF_LADDER = [64, 128, 256, 1024, 4096, 16384, 65536]    # 8 buckets
NCONF = len(CONF_LADDER) + 1
ADJ_LADDER = [0, 1, 2, 4, 8]                              # 6 buckets
NADJ = len(ADJ_LADDER) + 1
# When a match exists but has not yet earned the right to replace the
# prediction, it still has an opinion about the residual: mval - pred. The sign
# bit used to ignore that entirely.
NMSIGN = 4
# The match also has an opinion about the residual's magnitude. Splitting the
# length-bin contexts on it was measured and lost to dilution, so it enters as
# a second expert to be mixed instead - mixing combines predictions where
# splitting divides the data.
MEXP_LADDER = [0, 1, 3, 7, 15]
NMEXP = len(MEXP_LADDER) + 2      # 0 means 'no match at all'

# Match model. Every other expert reads the same handful of adjacent pixels, so
# they mostly agree and mixing them pays little. This one looks somewhere else
# entirely: it hashes the causal neighbourhood, remembers where that exact
# neighbourhood last occurred anywhere earlier in the plane, and offers the
# pixel that followed it last time. On repeated structure - fabric, brickwork,
# lettering, anything periodic - it knows things no gradient can.
MATCH_HASH_BITS = 20
MATCH_HASH_MASK = (1 << MATCH_HASH_BITS) - 1
MATCH_MAX_LEN = 7        # match lengths saturate here; longer is not more certain
# Context is (how long the match has held) x (does it agree with the predictor).
NMATCH = 1 + (MATCH_MAX_LEN + 1) * 3
# Once a match has held this many pixels it replaces the prediction outright.
#
# Averaging it into the weighted blend instead was measured and was much worse
# (+1.3%): a match is either exactly right or wildly wrong, and averaging a
# wildly wrong value into an otherwise good prediction damages every pixel it
# touches. A hard switch on sustained agreement is the right combiner.
#
# The threshold trades photographs against repetitive content. Swept on the dev
# split: at 2 a photo pays 1.8% and tiled noise costs 6.1KB; at 8 the photo pays
# 0.009% and tiled noise costs 12.6KB, still 8.7x better than without the model.
# Losing on photographs to win elsewhere is not a trade worth making, so 8.
MATCH_TRUST = 8

# Self-correcting weighted predictor, in the style of JPEG XL's modular mode.
# Four sub-predictors each keep a running record of how wrong they have been in
# this neighbourhood; the blend weight is inversely proportional to that. Where
# one sub-predictor is reliably right - a vertical edge, a smooth gradient - it
# takes over locally without anything having to be signalled.
#
# The structure follows libjxl's predictor #6; the constants are tuned here
# against the dev split rather than copied, since the surrounding model differs.
# libjxl feeds each sub-predictor's neighbours' errors back into the
# sub-predictions themselves (its p1C/p2C constants). Swept here across
# 0/4/8/16 that consistently *hurt* - the blend weights already carry the error
# information, and correcting the inputs as well double-counts it - so the
# feedback strength is zero and only the error-weighted blending is kept.
WP_P1 = 0
WP_P2 = 0
WP_MAXW = (13, 12, 12, 11)
WP_SHIFT = 12           # dynamic range of the weights against the +4 floor

# Learned linear combiner over the sub-predictors.
#
# The error-weighted blend above sets each weight from that sub-predictor's own
# recent error, in isolation, and always positive and summing to one. That is
# what makes it fragile: a volatile member still drags the result even when it
# is down-weighted, and it can never express "this predictor is reliably wrong
# in the same direction, so subtract it". Four separate attempts to add
# predictors to that average came back negative (GAP, least squares, the match
# value, the 2W-WW / 2N-NN trend pair), failing the same way each time, which
# points at the combiner rather than at any of the predictors.
#
# This layer learns the weights jointly by gradient descent on the actual
# prediction error, normalised by input energy (NLMS), so a useless input
# converges to zero weight and a systematically wrong one to a negative weight.
#
# The blend is the origin rather than a competitor: every weight starts at zero,
# so the combiner opens byte-identical to the model it replaces and has to earn
# anything it gets. That is the same trick mix.Mixer uses on its first expert,
# and the reason a failure here shows up as "no change" rather than as a
# regression. Concretely the inputs are an affine combination,
# pred + sum_i w_i * (p_i - pred), which also keeps every input small and so
# conditions the normalised step far better than raw 0..255 values would.
#
# Choosing what goes in is not "add every predictor you have". The combiner is
# *linear*, so a predictor that is already a linear combination of the others
# adds no span and cannot pay: 2W-WW is worthless next to W and WW, and so are
# W+NE-N and the planar W+N-NW. That was measured the slow way first - seven
# mostly-redundant inputs returned -0.013%. What earns a slot is either a
# neighbour the others cannot reach (the second ring: WW, NN, NNW, NNE) or a
# genuinely nonlinear predictor (MED, GAP).
LMS_NPRED = 13
LMS_WSHIFT = 16         # weights are 16.16 fixed point, as in mix.Mixer
LMS_RATE = 5            # NLMS step size, mu = 1/32; swept 3..8 on the dev split
LMS_EPS = 64            # keeps the energy divisor away from zero on flat runs
LMS_STEP_CLAMP = 1 << 16
LMS_WCLAMP = 1 << 20    # 16x unity; well inside int64 for the fast path
# Weight sets are (plane, activity, gradient direction). Finer is not
# automatically better - every extra set divides the training data, and 13
# weights need a few thousand samples to converge. The full 16-bucket activity
# ladder measured worse than 8 for exactly that reason (-1.53% against -1.58%),
# so the ladder is halved and the freed budget spent on a split that carries
# more information per set: how much smoother the neighbourhood is vertically
# than horizontally, which is the property that decides which predictors deserve
# weight in the first place. Two-way -1.86%, four-way -1.95%.
LMS_ACT_SHIFT = 1
LMS_NDIR = 4


def lms_nctx():
    """Weight sets per plane. A function because tools/ sweeps move NACT."""
    return ((NACT + (1 << LMS_ACT_SHIFT) - 1) >> LMS_ACT_SHIFT) * LMS_NDIR


def new_model():
    """A fresh probability bank. One bank is shared by every plane and frame."""
    return {
        "zero": rc.new_probs(KINDS * NACT * NERR * NLUM),
        # Expert 2: directional error, which the summed error context throws away.
        "zero_dir": rc.new_probs(KINDS * 9 * NSIDE * NSIDE),
        # Expert 3: raw neighbour differences rather than aggregated activity.
        "zero_diff": rc.new_probs(KINDS * NDIFF * NDIFF),
        # Expert 4: the match model.
        "zero_match": rc.new_probs(KINDS * NMATCH),
        # Expert 5: the learned combiner's disagreement and correction size.
        "zero_conf": rc.new_probs(KINDS * NCONF * NADJ),
        "sign": rc.new_probs(KINDS * 9 * NMSIGN),
        "nb": rc.new_probs(KINDS * NACT * NERR * (MAX_NB + 1)),
        "nb_match": rc.new_probs(KINDS * (MAX_NB + 1) * NMEXP),
        # The length bins get the combiner's confidence as well. How far apart
        # the predictors were says something about how big the residual will be,
        # not just about whether it is zero.
        "nb_conf": rc.new_probs(KINDS * (MAX_NB + 1) * NCONF),
        "nb_mix": mix.Mixer(3, KINDS * (MAX_NB + 1)),
        "mant": rc.new_probs(KINDS * (MAX_NB + 1) * 2),
        "zero_mix": mix.Mixer(ZERO_EXPERTS, KINDS * NACT),
        "zero_apm": mix.APM(KINDS * NACT),
        # LPAQ chains two secondary-estimation stages with different contexts.
        # This one sees the match state, which the activity-keyed first stage
        # cannot; both are blended 1:3 with their input the way LPAQ does.
        "zero_apm2": mix.APM(KINDS * NMEXP * NMSIGN),
        # The magnitude bins get secondary estimation too. They are the second
        # biggest cost after the zero flag, and a full mixer there was measured
        # to cost more time than it returned.
        "nb_apm": mix.APM(KINDS * (MAX_NB + 1) * NACT),
        # Weight sets for the learned combiner: (plane, activity, direction).
        "lms": [0] * (KINDS * lms_nctx() * LMS_NPRED),
    }


def _adapt(probs, ctx, bit):
    """The same exponential update rc.bit applies, for models the coder no
    longer owns: once a bit is coded from a mixed probability, each expert has
    to be nudged by hand."""
    p = probs[ctx]
    if bit:
        probs[ctx] = p - (p >> rc.ADAPT_SHIFT)
    else:
        probs[ctx] = p + ((rc.PROB_ONE - p) >> rc.ADAPT_SHIFT)


def kind_offsets(kind):
    return (kind * NACT * NERR * NLUM,
            kind * NACT * NERR * (MAX_NB + 1),
            kind * 9 * NMSIGN,
            kind * (MAX_NB + 1) * 2)


class InterInfo:
    """Per-block prediction sources for a video frame.

    `bs_y`/`bs_x` are this plane's block size and `mv_sy`/`mv_sx` the motion
    vector divisors, so a subsampled chroma plane can reuse the luma decisions.

    `ref_phases` is the reference plane at its four half-pel phases, indexed
    (dy & 1) * 2 + (dx & 1). Chroma divides the luma vector down first, which
    lands it on quarter-pel positions of the luma grid for free.
    """

    __slots__ = ("modes", "mvs", "bs_y", "bs_x", "mv_sy", "mv_sx", "ref_phases")

    def __init__(self, modes, mvs, bs_y, bs_x, mv_sy, mv_sx, ref_phases):
        self.modes = modes
        self.mvs = mvs
        self.bs_y = bs_y
        self.bs_x = bs_x
        self.mv_sy = mv_sy
        self.mv_sx = mv_sx
        self.ref_phases = ref_phases


def code_plane(coder, encode, width, height, kind, model, src=None, luma_err=None,
               inter=None):
    """Code one plane. Returns (rows, err_rows) as lists of python ints.

    `src` is required when encoding (a list of row lists). `luma_err` supplies
    the co-located luma error magnitudes, which sharpen the chroma contexts a
    lot - chroma tends to go wrong exactly where luma did. `inter` switches on
    the video path, where each block picks its own prediction source.
    """
    zero_p = model["zero"]
    dir_p = model["zero_dir"]
    diff_p = model["zero_diff"]
    match_p = model["zero_match"]
    conf_p = model["zero_conf"]
    sign_p = model["sign"]
    nb_p = model["nb"]
    nbm_p = model["nb_match"]
    nbc_p = model["nb_conf"]
    nb_mix = model["nb_mix"]
    mant_p = model["mant"]
    zero_mix = model["zero_mix"]
    zero_apm = model["zero_apm"]
    zero_apm2 = model["zero_apm2"]
    nb_apm = model["nb_apm"]
    lms_w = model["lms"]
    bit = coder.bit
    bypass = coder.bypass
    bisect_right = bisect.bisect_right
    act_ladder, err_ladder, lum_ladder = ACT_LADDER, ERR_LADDER, LUMA_LADDER
    side_ladder, diff_ladder = SIDE_LADDER, DIFF_LADDER
    stretch_t = mix.STRETCH

    k_zero, k_nb, k_sign, k_mant = kind_offsets(kind)
    kind_dir = kind * 9 * NSIDE * NSIDE
    kind_diff = kind * NDIFF * NDIFF
    kind_match = kind * NMATCH
    kind_mix = kind * NACT
    kind_nbapm = kind * (MAX_NB + 1) * NACT
    kind_lms = kind * lms_nctx() * LMS_NPRED
    kind_conf = kind * NCONF * NADJ
    lms_x = [0] * LMS_NPRED
    conf_ladder, adj_ladder = CONF_LADDER, ADJ_LADDER
    if inter is not None:
        i_zero, i_nb, i_sign, i_mant = kind_offsets(kind + INTER_KIND_OFFSET)
        nbx = len(inter.modes[0])
        ref_phases = inter.ref_phases

    rows = []
    err_rows = []
    prev = [0] * width
    # The second ring. Only the learned combiner reads it; the contexts do not.
    prev2 = [0] * width
    prev_err = [0] * width

    # Match-model state. `flat` is every pixel decoded so far in raster order;
    # `match_table` maps a neighbourhood hash to the position that followed that
    # neighbourhood last time. Both sides build these from decoded values only.
    match_table = array("i", [0]) * (MATCH_HASH_MASK + 1)
    flat = []
    flat_append = flat.append
    match_pos = 0
    match_len = 0
    hash_mask = MATCH_HASH_MASK

    # Weighted-predictor history. Index x is stored at x+1 so the neighbours at
    # x-1 and x+1 are always in range without a bounds test.
    wp_w0, wp_w1, wp_w2, wp_w3 = WP_MAXW
    terr_prev = [0] * (width + 2)
    terr_cur = [0] * (width + 2)
    werr_prev = [[0] * (width + 2) for _ in range(4)]
    werr_cur = [[0] * (width + 2) for _ in range(4)]

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
                    # Scale the luma vector to this plane, then split it into a
                    # whole-pixel offset and a half-pel phase. The shift floors,
                    # so a negative half-pel vector lands on the correct side.
                    hy = int(mv_row[bx][0]) // inter.mv_sy
                    hx = int(mv_row[bx][1]) // inter.mv_sx
                    ry = y + (hy >> 1)
                    rx = x + (hx >> 1)
                    phase = ref_phases[(hy & 1) * 2 + (hx & 1)]
                    ref_row_of_x[x] = phase[
                        0 if ry < 0 else (height - 1 if ry >= height else ry)]
                    ref_col_of_x[x] = 0 if rx < 0 else (width - 1 if rx >= width else rx)

        west = 0
        west_err = 0
        first_row = y == 0

        for x in range(width):
            mval = -1
            # Reset per-pixel, not carried over: csrc/kernel.c mirrors this loop
            # and a value left alive across iterations there — as numba also did
            # assigns it.
            lms_on = False
            lms_base_w = 0
            lms_pred = 0
            lms_adj = 0
            energy = 0
            if first_row:
                pred = 128 if x == 0 else west
                act = 0
                act_b = 0
                sgn = 4
            else:
                north = prev[x]
                if x == 0:
                    pred = north
                    act = 0
                    act_b = 0
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
                    act_b = bisect_right(act_ladder, act)
                    sgn = (0 if d1 < 0 else (1 if d1 == 0 else 2)) * 3 \
                        + (0 if d2 < 0 else (1 if d2 == 0 else 2))

                    # Blend four sub-predictors, weighting each by how well it
                    # has been doing right here.
                    te_w = terr_cur[x]
                    te_n = terr_prev[x + 1]
                    te_nw = terr_prev[x]
                    te_ne = terr_prev[x + 2]
                    sum_wn = te_n + te_w
                    q0 = west + neast - north
                    q1 = north - (((sum_wn + te_ne) * WP_P1) >> 5)
                    q2 = west - (((sum_wn + te_nw) * WP_P2) >> 5)
                    q3 = pred
                    e = werr_prev[0]
                    a0 = werr_cur[0][x] + e[x] + e[x + 1] + e[x + 2]
                    e = werr_prev[1]
                    a1 = werr_cur[1][x] + e[x] + e[x + 1] + e[x + 2]
                    e = werr_prev[2]
                    a2 = werr_cur[2][x] + e[x] + e[x + 1] + e[x + 2]
                    e = werr_prev[3]
                    a3 = werr_cur[3][x] + e[x] + e[x + 1] + e[x + 2]
                    m0 = (wp_w0 << WP_SHIFT) // (a0 + 4) + 4
                    m1 = (wp_w1 << WP_SHIFT) // (a1 + 4) + 4
                    m2 = (wp_w2 << WP_SHIFT) // (a2 + 4) + 4
                    m3 = (wp_w3 << WP_SHIFT) // (a3 + 4) + 4
                    total = m0 + m1 + m2 + m3
                    blend = (q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3 + (total >> 1)) // total
                    # Only trust an extrapolation outside the neighbours when all
                    # three recent errors agree in sign; otherwise rein it in.
                    if not ((te_n >= 0) == (te_w >= 0) == (te_nw >= 0)):
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

                    # Skip the learned combiner where its answer is thrown
                    # away. A temporally predicted pixel overwrites `pred`
                    # with the reference sample a few lines below, so on an
                    # inter block all of this - thirteen multiply-accumulates,
                    # a division and thirteen weight updates - is computed and
                    # then discarded. Most pixels in a video are inter, and at
                    # 1080p this was a third of the coding time.
                    #
                    # `lms_on`, `energy` and `lms_adj` keep their per-pixel
                    # defaults when this is skipped, so the confidence context
                    # sees a consistent 'no combiner ran here' state on both
                    # sides rather than a stale one.
                    if inter is None or not mode_x[x]:
                        # The second ring, clamped back onto the first at the edges
                        # of the plane so both paths agree on what is unavailable.
                        # Names extend the existing convention, so `nnwest` is two
                        # rows up and one column left while `nwwest` is one row up
                        # and two columns left - different pixels, similar names.
                        wwest = cur[x - 2] if x >= 2 else west
                        wwwest = cur[x - 3] if x >= 3 else wwest
                        nwwest = prev[x - 2] if x >= 2 else nwest
                        neeast = prev[x + 2] if x + 2 < width else neast
                        if y >= 2:
                            nnorth = prev2[x]
                            nnwest = prev2[x - 1]
                            nneast = prev2[x + 1] if x + 1 < width else nnorth
                        else:
                            nnorth = north
                            nnwest = nwest
                            nneast = neast

                        # GAP, CALIC's gradient-adjusted prediction. It earns a slot
                        # because it is *nonlinear* - it switches on the ratio of
                        # horizontal to vertical gradient - so unlike another
                        # weighted sum of neighbours it is not already inside the
                        # combiner's span.
                        dh = (abs(west - wwest) + abs(north - nwest)
                              + abs(north - neast))
                        dv = (abs(west - nwest) + abs(north - nnorth)
                              + abs(neast - nneast))
                        dd = dv - dh
                        if dd > 80:
                            gap = west
                        elif dd < -80:
                            gap = north
                        else:
                            gap = (west + north) // 2 + (neast - nwest) // 4
                            if dd > 32:
                                gap = (gap + west) // 2
                            elif dd > 8:
                                gap = (3 * gap + west) // 4
                            elif dd < -32:
                                gap = (gap + north) // 2
                            elif dd < -8:
                                gap = (3 * gap + north) // 4

                        # The second ring first, then the two nonlinear predictors.
                        lms_x[0] = west - pred
                        lms_x[1] = north - pred
                        lms_x[2] = nwest - pred
                        lms_x[3] = neast - pred
                        lms_x[4] = wwest - pred
                        lms_x[5] = nnorth - pred
                        lms_x[6] = nnwest - pred
                        lms_x[7] = nneast - pred
                        lms_x[8] = q3 - pred
                        lms_x[9] = gap - pred
                        lms_x[10] = wwwest - pred
                        lms_x[11] = nwwest - pred
                        lms_x[12] = neeast - pred
                        lms_base_w = kind_lms + (
                            (act_b >> LMS_ACT_SHIFT) * LMS_NDIR
                            + (0 if dd < -32 else (1 if dd <= 0 else
                                (2 if dd <= 32 else 3)))) * LMS_NPRED
                        acc = 0
                        energy = LMS_EPS
                        for i in range(LMS_NPRED):
                            xi = lms_x[i]
                            acc += lms_w[lms_base_w + i] * xi
                            energy += xi * xi
                        lms_adj = acc >> LMS_WSHIFT
                        adj = pred + lms_adj
                        pred = 255 if adj > 255 else (0 if adj < 0 else adj)
                        lms_pred = pred
                        lms_on = True

            if not first_row and x:
                # Match model: where did this exact neighbourhood last occur?
                mhash = ((west * 0x2F0FD693 + north * 0x9E3779B1
                          + nwest * 0x85EBCA77 + neast * 0xC2B2AE3D) >> 8) & hash_mask
                if not match_len:
                    match_pos = match_table[mhash]
                if 0 < match_pos < len(flat):
                    mval = flat[match_pos]
                match_table[mhash] = len(flat)

            if inter is not None and mode_x[x]:
                pred = ref_row_of_x[x][ref_col_of_x[x]]
                b_zero, b_nb, b_sign, b_mant = i_zero, i_nb, i_sign, i_mant
                b_kind = kind + INTER_KIND_OFFSET
            else:
                b_zero, b_nb, b_sign, b_mant = k_zero, k_nb, k_sign, k_mant
                b_kind = kind

            north_err = 0 if first_row else prev_err[x]
            if first_row:
                err_sum = west_err
            else:
                err_sum = west_err + north_err + (prev_err[x + 1] if x + 1 < width else 0)
            sub = act_b * NERR + bisect_right(err_ladder, err_sum)
            lum = bisect_right(lum_ladder, lrow[x]) if lrow is not None else 0
            zctx = b_zero + sub * NLUM + lum
            nbbase = b_nb + sub * (MAX_NB + 1)

            # Three views of the same neighbourhood, mixed. They are chosen to
            # disagree: aggregated activity, directional error, raw differences.
            dir_ctx = (kind_dir + (sgn * NSIDE + bisect_right(side_ladder, west_err)) * NSIDE
                       + bisect_right(side_ladder, north_err))
            if first_row or x == 0:
                diff_ctx = kind_diff
            else:
                dwn = west - north
                dne = nwest - neast
                diff_ctx = (kind_diff
                            + bisect_right(diff_ladder, dwn if dwn >= 0 else -dwn) * NDIFF
                            + bisect_right(diff_ladder, dne if dne >= 0 else -dne))
            if mval < 0:
                match_ctx = kind_match
                msign = 0
                mexp_b = 0
            else:
                agree = 0 if mval == pred else (1 if -3 < mval - pred < 3 else 2)
                hit = match_len if match_len < MATCH_MAX_LEN else MATCH_MAX_LEN
                match_ctx = kind_match + 1 + hit * 3 + agree
                # ...but never over a temporally predicted block. The match
                # model detects *spatial* repetition; on an inter block it was
                # overwriting a motion-compensated prediction that is usually
                # already exact. In flat or near-static content its hash matches
                # everywhere, so it trivially earns sustained agreement and then
                # replaces a perfect prediction with a worse one - measured at
                # 7.9% on 16 frames of 1080p Sintel. Suppressing only the
                # override keeps its contexts and its sign and magnitude
                # experts, which is where its value on intra blocks comes from.
                if match_len >= MATCH_TRUST and not (inter is not None
                                                     and mode_x[x]):
                    pred = mval
                mexp = mval - pred
                msign = 1 if mexp < 0 else (2 if mexp == 0 else 3)
                mexp_b = 1 + bisect_right(MEXP_LADDER, mexp if mexp >= 0 else -mexp)

            conf_b = bisect_right(conf_ladder, energy)
            conf_ctx = (kind_conf + conf_b * NADJ
                        + bisect_right(adj_ladder,
                                       lms_adj if lms_adj >= 0 else -lms_adj))
            experts = [stretch_t[4095 - (zero_p[zctx] >> 3)],
                       stretch_t[4095 - (dir_p[dir_ctx] >> 3)],
                       stretch_t[4095 - (diff_p[diff_ctx] >> 3)],
                       stretch_t[4095 - (match_p[match_ctx] >> 3)],
                       stretch_t[4095 - (conf_p[conf_ctx] >> 3)]]
            mix_ctx = kind_mix + act_b
            pr1 = zero_mix.mix(experts, mix_ctx)
            pr1 = (pr1 + 3 * zero_apm.refine(pr1, mix_ctx)) >> 2
            actx2 = (b_kind * NMEXP + mexp_b) * NMSIGN + msign
            pr1 = (pr1 + 3 * zero_apm2.refine(pr1, actx2)) >> 2
            if pr1 < 1:
                pr1 = 1
            elif pr1 > 4095:
                pr1 = 4095
            p_zero = (4096 - pr1) << 3

            if encode:
                d = ((srow[x] - pred + 128) & 255) - 128
                mag = -d if d < 0 else d
                nonzero = 1 if mag else 0
                coder.bit_p(p_zero, nonzero)
                _adapt(zero_p, zctx, nonzero)
                _adapt(dir_p, dir_ctx, nonzero)
                _adapt(diff_p, diff_ctx, nonzero)
                _adapt(match_p, match_ctx, nonzero)
                _adapt(conf_p, conf_ctx, nonzero)
                zero_mix.update(nonzero)
                zero_apm.update(nonzero)
                zero_apm2.update(nonzero)
                if mag:
                    bit(sign_p, b_sign + sgn * NMSIGN + msign, 1 if d < 0 else 0)
                    v = mag - 1
                    nb = v.bit_length()
                    limit = nb if nb < MAX_NB else nb - 1
                    for i in range(limit + 1):
                        more = 1 if i < nb else 0
                        ctx = nbbase + i
                        mixc = b_kind * (MAX_NB + 1) + i
                        mctx = mixc * NMEXP + mexp_b
                        cctx = mixc * NCONF + conf_b
                        pr = nb_mix.mix([stretch_t[4095 - (nb_p[ctx] >> 3)],
                                         stretch_t[4095 - (nbm_p[mctx] >> 3)],
                                         stretch_t[4095 - (nbc_p[cctx] >> 3)]], mixc)
                        actx = kind_nbapm + i * NACT + act_b
                        pr = (pr + 3 * nb_apm.refine(pr, actx)) >> 2
                        if pr < 1:
                            pr = 1
                        elif pr > 4095:
                            pr = 4095
                        coder.bit_p((4096 - pr) << 3, more)
                        _adapt(nb_p, ctx, more)
                        _adapt(nbm_p, mctx, more)
                        _adapt(nbc_p, cctx, more)
                        nb_mix.update(more)
                        nb_apm.update(more)
                    if nb >= 2:
                        bit(mant_p, b_mant + nb * 2, (v >> (nb - 2)) & 1)
                        if nb >= 3:
                            bit(mant_p, b_mant + nb * 2 + 1, (v >> (nb - 3)) & 1)
                            if nb > 3:
                                bypass(v & ((1 << (nb - 3)) - 1), nb - 3)
                value = (pred + d) & 255
            else:
                nonzero = coder.bit_p(p_zero)
                _adapt(zero_p, zctx, nonzero)
                _adapt(dir_p, dir_ctx, nonzero)
                _adapt(diff_p, diff_ctx, nonzero)
                _adapt(match_p, match_ctx, nonzero)
                _adapt(conf_p, conf_ctx, nonzero)
                zero_mix.update(nonzero)
                zero_apm.update(nonzero)
                zero_apm2.update(nonzero)
                if nonzero:
                    neg = bit(sign_p, b_sign + sgn * NMSIGN + msign)
                    nb = 0
                    while nb < MAX_NB:
                        ctx = nbbase + nb
                        mixc = b_kind * (MAX_NB + 1) + nb
                        mctx = mixc * NMEXP + mexp_b
                        cctx = mixc * NCONF + conf_b
                        pr = nb_mix.mix([stretch_t[4095 - (nb_p[ctx] >> 3)],
                                         stretch_t[4095 - (nbm_p[mctx] >> 3)],
                                         stretch_t[4095 - (nbc_p[cctx] >> 3)]], mixc)
                        actx = kind_nbapm + nb * NACT + act_b
                        pr = (pr + 3 * nb_apm.refine(pr, actx)) >> 2
                        if pr < 1:
                            pr = 1
                        elif pr > 4095:
                            pr = 4095
                        more = coder.bit_p((4096 - pr) << 3)
                        _adapt(nb_p, ctx, more)
                        _adapt(nbm_p, mctx, more)
                        _adapt(nbc_p, cctx, more)
                        nb_mix.update(more)
                        nb_apm.update(more)
                        if not more:
                            break
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

            if lms_on:
                # NLMS: w += mu * err * x / |x|^2. The division is done once and
                # the sign carried separately, so the reference and the C
                # path cannot disagree about how a negative quotient rounds.
                #
                # This learns from the combiner's own prediction even on pixels
                # where the match model or an inter block overrode it, which is
                # what keeps the weights meaningful through repetitive regions
                # instead of freezing wherever the switch fires.
                lerr = value - lms_pred
                step = ((-lerr if lerr < 0 else lerr) << LMS_WSHIFT) // energy
                if step > LMS_STEP_CLAMP:
                    step = LMS_STEP_CLAMP
                if lerr < 0:
                    step = -step
                for i in range(LMS_NPRED):
                    wi = lms_w[lms_base_w + i] + ((step * lms_x[i]) >> LMS_RATE)
                    if wi > LMS_WCLAMP:
                        wi = LMS_WCLAMP
                    elif wi < -LMS_WCLAMP:
                        wi = -LMS_WCLAMP
                    lms_w[lms_base_w + i] = wi

            if not first_row and x:
                terr_cur[x + 1] = pred - value
                d0 = q0 - value
                d1_ = q1 - value
                d2_ = q2 - value
                d3_ = q3 - value
                werr_cur[0][x + 1] = d0 if d0 >= 0 else -d0
                werr_cur[1][x + 1] = d1_ if d1_ >= 0 else -d1_
                werr_cur[2][x + 1] = d2_ if d2_ >= 0 else -d2_
                werr_cur[3][x + 1] = d3_ if d3_ >= 0 else -d3_

            if mval == value:
                match_pos += 1
                match_len += 1
            else:
                match_len = 0
            flat_append(value)

            cur[x] = value
            cur_err[x] = mag
            west = value
            west_err = mag

        rows.append(cur)
        err_rows.append(cur_err)
        prev2 = prev
        prev = cur
        prev_err = cur_err
        terr_prev, terr_cur = terr_cur, [0] * (width + 2)
        werr_prev, werr_cur = werr_cur, [[0] * (width + 2) for _ in range(4)]

    return rows, err_rows


# Feature bits: which stages of the model are switched on.
#
# Every one of these costs time and buys ratio, and the trade is very different
# per stage - see "Buying speed with ratio" in docs/research.md for the measured
# table. They exist so a speed preset can drop the expensive stages, the way
# x264 has --preset. The bitmask travels in the container header, so a decoder
# always knows which model the encoder used.
#
# NOTE: hve/model.py's own code_plane implements only FEAT_ALL. It remains the
# reference for the default preset; the C kernel is the definition for the rest.
FEAT_BLEND = 1      # the four-way error-weighted sub-predictor blend
FEAT_LMS = 2        # the 13-tap learned (NLMS) combiner
FEAT_MATCH = 4      # the match model
FEAT_MIX = 8        # the 5-expert logistic mixer on the zero flag
FEAT_APM1 = 16      # first secondary-estimation stage
FEAT_APM2 = 32      # second secondary-estimation stage, keyed on match state
FEAT_NBMIX = 64     # mixer + APM on the magnitude bins
FEAT_ALL = 127
FEATURES = FEAT_ALL
# Position of FEATURES in coder_params(); the C side has the same enum.
P_FEATURES = 28


def coder_params():
    """The tunables the compiled backends read, as a flat array.

    `hve/native.py` hands this straight to the C kernel, which indexes it by
    position, so the order is part of the contract: **append only**. Inserting an
    entry shifts every index after it and silently scrambles the model rather
    than failing. `csrc/hve.h` carries the matching enum.

    Read at call time rather than baked in, so the sweeps in `tools/` still take
    effect. It lives here, next to the constants it reads.
    """
    import numpy as np

    from . import rc, video as _video
    return np.array([
        NACT, NERR, NLUM, NSIDE, NDIFF, NMATCH, MAX_NB, MATCH_MAX_LEN,
        MATCH_TRUST, WP_P1, WP_P2, WP_SHIFT,
        WP_MAXW[0], WP_MAXW[1], WP_MAXW[2], WP_MAXW[3],
        MATCH_HASH_MASK, rc.ADAPT_SHIFT, _video.MV_MAX,
        LMS_NPRED, LMS_WSHIFT, LMS_RATE, LMS_EPS,
        LMS_STEP_CLAMP, LMS_WCLAMP, NADJ, LMS_ACT_SHIFT, LMS_NDIR,
        FEATURES,
    ], dtype=np.int64)
