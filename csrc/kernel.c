/* Native pixel loop and range coder. Byte-identical to hve/model.py.
 *
 * This is a deliberately boring transcription of model.code_plane: same variable
 * names, same order of operations, same comment anchors. A clever rewrite would
 * be faster to read and impossible to diff against the reference when the model
 * changes, and the model changes often.
 *
 * What is *not* a transcription, and why:
 *   - the scratch arrays are narrowed (see hve.h);
 *   - the four self-correcting blend weights come from a table instead of four
 *     divisions per pixel;
 *   - the remaining divisions are 32-bit.
 * All three are speed-only and provably exact. Everything else matches.
 */

#include <stdlib.h>
#include <string.h>

#include "hve.h"
#include "model_constants.h"

/* --------------------------------------------------------------------------
 * primitives, mirroring hve/rc.py and hve/mix.py
 */

static inline int64_t sq_of(const int64_t *sq, int64_t x)
{
    if (x <= -2047)
        return 1;
    if (x >= 2047)
        return 4095;
    return sq[x + 2047];
}

static inline void shift_low(hve_rc *r, uint8_t *out)
{
    int64_t low = r->s[0];
    if (low < 0xFF000000 || low > 0xFFFFFFFF) {
        int64_t carry = low >> 32;
        out[r->s[4]++] = (uint8_t)((r->s[2] + carry) & 0xFF);
        if (r->s[3] > 1) {
            uint8_t filler = (uint8_t)((0xFF + carry) & 0xFF);
            for (int64_t i = 0; i < r->s[3] - 1; i++)
                out[r->s[4]++] = filler;
        }
        r->s[2] = (low >> 24) & 0xFF;
        r->s[3] = 0;
    }
    r->s[3]++;
    r->s[0] = (low << 8) & 0xFFFFFFFF;
}

static inline void enc_bit_p(hve_rc *r, uint8_t *out, int64_t p, int64_t value)
{
    int64_t bound = (r->s[1] >> 15) * p;
    if (value) {
        r->s[0] += bound;
        r->s[1] -= bound;
    } else {
        r->s[1] = bound;
    }
    while (r->s[1] < 16777216) {
        r->s[1] = (r->s[1] << 8) & 0xFFFFFFFF;
        shift_low(r, out);
    }
}

static inline void enc_bit(hve_rc *r, uint8_t *out, int64_t *probs, int64_t ctx,
                           int64_t value)
{
    int64_t p = probs[ctx];
    int64_t bound = (r->s[1] >> 15) * p;
    if (value) {
        r->s[0] += bound;
        r->s[1] -= bound;
        probs[ctx] = p - (p >> 6);
    } else {
        r->s[1] = bound;
        probs[ctx] = p + ((32768 - p) >> 6);
    }
    while (r->s[1] < 16777216) {
        r->s[1] = (r->s[1] << 8) & 0xFFFFFFFF;
        shift_low(r, out);
    }
}

static inline void enc_bypass(hve_rc *r, uint8_t *out, int64_t value,
                              int64_t nbits)
{
    for (int64_t i = nbits - 1; i >= 0; i--) {
        r->s[1] >>= 1;
        if ((value >> i) & 1)
            r->s[0] += r->s[1];
        while (r->s[1] < 16777216) {
            r->s[1] = (r->s[1] << 8) & 0xFFFFFFFF;
            shift_low(r, out);
        }
    }
}

/* One payload byte, or zero once the payload is exhausted. Past the end the
 * decoder is producing nonsense either way; the point is that it does so
 * inside its own buffer, and that s[2] keeps counting so the overrun can be
 * reported afterwards. The branch predicts perfectly on valid streams. */
static inline int64_t rc_byte(hve_rc *r, const uint8_t *data)
{
    int64_t i = r->s[2]++;
    return i < r->s[3] ? (int64_t)data[i] : 0;
}

static inline int64_t dec_bit_p(hve_rc *r, const uint8_t *data, int64_t p)
{
    int64_t bound = (r->s[1] >> 15) * p, v;
    if (r->s[0] < bound) {
        r->s[1] = bound;
        v = 0;
    } else {
        r->s[0] -= bound;
        r->s[1] -= bound;
        v = 1;
    }
    while (r->s[1] < 16777216) {
        r->s[1] <<= 8;
        r->s[0] = ((r->s[0] << 8) | rc_byte(r, data)) & 0xFFFFFFFF;
    }
    return v;
}

static inline int64_t dec_bit(hve_rc *r, const uint8_t *data, int64_t *probs,
                              int64_t ctx)
{
    int64_t p = probs[ctx];
    int64_t bound = (r->s[1] >> 15) * p, v;
    if (r->s[0] < bound) {
        r->s[1] = bound;
        probs[ctx] = p + ((32768 - p) >> 6);
        v = 0;
    } else {
        r->s[0] -= bound;
        r->s[1] -= bound;
        probs[ctx] = p - (p >> 6);
        v = 1;
    }
    while (r->s[1] < 16777216) {
        r->s[1] <<= 8;
        r->s[0] = ((r->s[0] << 8) | rc_byte(r, data)) & 0xFFFFFFFF;
    }
    return v;
}

static inline int64_t dec_bypass(hve_rc *r, const uint8_t *data, int64_t nbits)
{
    int64_t value = 0;
    for (int64_t i = 0; i < nbits; i++) {
        r->s[1] >>= 1;
        if (r->s[0] >= r->s[1]) {
            r->s[0] -= r->s[1];
            value = (value << 1) | 1;
        } else {
            value <<= 1;
        }
        while (r->s[1] < 16777216) {
            r->s[1] <<= 8;
            r->s[0] = ((r->s[0] << 8) | rc_byte(r, data)) & 0xFFFFFFFF;
        }
    }
    return value;
}

static int g_batched = 1;

void hve_set_batched(int on)
{
    g_batched = on ? 1 : 0;
}

int hve_batched_enabled(void)
{
    return g_batched;
}

int64_t hve_finish_encode(hve_rc *r, uint8_t *out)
{
    for (int i = 0; i < 5; i++)
        shift_low(r, out);
    return r->s[4];
}

/* --------------------------------------------------------------------------
 * modes and motion vectors
 */

static int64_t code_mv(int encode, hve_rc *r, const uint8_t *data, uint8_t *out,
                       int64_t *mv_zero, int64_t *mv_sign, int64_t *mv_mag,
                       int64_t axis, int64_t mv_max, int64_t value)
{
    int64_t base = axis * (mv_max + 1);
    if (encode) {
        int64_t mag = value < 0 ? -value : value;
        enc_bit(r, out, mv_zero, axis, mag ? 1 : 0);
        if (mag) {
            enc_bit(r, out, mv_sign, axis, value < 0 ? 1 : 0);
            for (int64_t i = 0; i < mag - 1; i++)
                enc_bit(r, out, mv_mag, base + i, 1);
            if (mag - 1 < mv_max)
                enc_bit(r, out, mv_mag, base + mag - 1, 0);
        }
        return value;
    }
    if (!dec_bit(r, data, mv_zero, axis))
        return 0;
    {
        int64_t neg = dec_bit(r, data, mv_sign, axis);
        int64_t mag = 1;
        while (mag - 1 < mv_max && dec_bit(r, data, mv_mag, base + mag - 1))
            mag++;
        return neg ? -mag : mag;
    }
}

static inline int64_t median3(int64_t a, int64_t b, int64_t c)
{
    int64_t lo = a < b ? a : b;
    int64_t hi = a > b ? a : b;
    lo = lo < c ? lo : c;
    hi = hi > c ? hi : c;
    return a + b + c - lo - hi;
}

/* Mirror of video.mv_predictor - median of left, above, above-right. */
static void mv_predictor(const int64_t *mvs, int64_t by, int64_t bx,
                         int64_t nbx, int64_t *py, int64_t *px)
{
    int64_t ly, lx, uy, ux, ry, rx;
    if (by == 0) {
        if (bx == 0) {
            *py = 0;
            *px = 0;
            return;
        }
        *py = mvs[(bx - 1) * 2];
        *px = mvs[(bx - 1) * 2 + 1];
        return;
    }
    if (bx > 0) {
        ly = mvs[(by * nbx + bx - 1) * 2];
        lx = mvs[(by * nbx + bx - 1) * 2 + 1];
    } else {
        ly = 0;
        lx = 0;
    }
    uy = mvs[((by - 1) * nbx + bx) * 2];
    ux = mvs[((by - 1) * nbx + bx) * 2 + 1];
    if (bx + 1 < nbx) {
        ry = mvs[((by - 1) * nbx + bx + 1) * 2];
        rx = mvs[((by - 1) * nbx + bx + 1) * 2 + 1];
    } else if (bx > 0) {
        ry = mvs[((by - 1) * nbx + bx - 1) * 2];
        rx = mvs[((by - 1) * nbx + bx - 1) * 2 + 1];
    } else {
        ry = 0;
        rx = 0;
    }
    *py = median3(ly, uy, ry);
    *px = median3(lx, ux, rx);
}

void hve_code_block_info(int encode, hve_rc *r, const uint8_t *data,
                         uint8_t *out, int64_t *mode_p, int64_t *mv_zero,
                         int64_t *mv_sign, int64_t *mv_mag, int64_t *modes,
                         int64_t *mvs, int64_t nby, int64_t nbx, int64_t mv_max)
{
    for (int64_t by = 0; by < nby; by++) {
        for (int64_t bx = 0; bx < nbx; bx++) {
            int64_t i = by * nbx + bx;
            int64_t left = bx ? modes[i - 1] : 0;
            int64_t up = by ? modes[i - nbx] : 0;
            int64_t ctx = left * 2 + up;
            if (encode)
                enc_bit(r, out, mode_p, ctx, modes[i]);
            else
                modes[i] = dec_bit(r, data, mode_p, ctx);
            if (modes[i] == 1) {
                int64_t py, px;
                mv_predictor(mvs, by, bx, nbx, &py, &px);
                if (encode) {
                    code_mv(1, r, data, out, mv_zero, mv_sign, mv_mag, 0,
                            mv_max, mvs[i * 2] - py);
                    code_mv(1, r, data, out, mv_zero, mv_sign, mv_mag, 1,
                            mv_max, mvs[i * 2 + 1] - px);
                } else {
                    int64_t dy = code_mv(0, r, data, out, mv_zero, mv_sign,
                                         mv_mag, 0, mv_max, 0);
                    int64_t dx = code_mv(0, r, data, out, mv_zero, mv_sign,
                                         mv_mag, 1, mv_max, 0);
                    mvs[i * 2] = dy + py;
                    mvs[i * 2 + 1] = dx + px;
                }
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * the plane loop, mirroring fast._code_plane1 and model.code_plane
 */

/* Blend-weight table. m_k = (wp_w[k] << wp_shift) / (a + 4) + 4, and `a` is a
 * sum of four recent absolute errors, so it is small and bounded in practice.
 * Tabulating the common range replaces four integer divisions per pixel with
 * four L1 hits; anything above the table still divides, so a parameter sweep
 * that widens the error range stays correct rather than silently wrong. */
#define WTAB 2048

typedef struct {
    int32_t t[4][WTAB];
    int64_t num[4];
} wtable;

static void wtable_init(wtable *w, const int64_t *params)
{
    const int wp[4] = {P_W0, P_W1, P_W2, P_W3};
    for (int k = 0; k < 4; k++) {
        w->num[k] = params[wp[k]] << params[P_WPSHIFT];
        for (int a = 0; a < WTAB; a++)
            w->t[k][a] = (int32_t)(w->num[k] / (a + 4) + 4);
    }
}

static inline int64_t wt(const wtable *w, int k, int64_t a)
{
    if ((uint64_t)a < WTAB)
        return w->t[k][a];
    return w->num[k] / (a + 4) + 4;
}

/* Ladder lookups, tabulated.
 *
 * hve_bisect is a linear scan, and the pixel loop does ten of them per sample
 * over ladders of up to fifteen entries, every one an unpredictable branch.
 * Every input is a small non-negative quantity with a known bound - a sum of
 * three folded byte differences, a neighbour error, a residual magnitude - so
 * the whole answer fits in a byte table small enough to stay in L1.
 *
 * Built by calling hve_bisect itself, and anything past the table falls back to
 * it, so this stays exact no matter what the ladders are changed to.
 */
#define LUT_ACT  385        /* three folded byte differences */
#define LUT_ERR  385        /* three neighbour magnitudes */
#define LUT_SIDE 129        /* one neighbour magnitude */
#define LUT_DIFF 256        /* |west - north| */
#define LUT_LUM  129        /* one luma residual magnitude */
#define LUT_MEXP 256        /* |match value - prediction| */
#define LUT_ADJ  16         /* the combiner correction, which saturates early */

typedef struct {
    uint8_t act[LUT_ACT], err[LUT_ERR], side[LUT_SIDE], diff[LUT_DIFF];
    uint8_t lum[LUT_LUM], mexp[LUT_MEXP], adj[LUT_ADJ];
    /* int32 mirrors of the four tables the batched derivation indexes. A
     * gather that widens bytes is not something the vectoriser will emit, and
     * these are a few kilobytes between them, so the scalar path keeps the
     * compact uint8 tables and the batched path reads these. */
    int32_t act32[LUT_ACT], err32[LUT_ERR], side32[LUT_SIDE], diff32[LUT_DIFF];
    int32_t lum32[LUT_LUM];
} ladder_luts;

static void luts_init(ladder_luts *L, const hve_model *m)
{
    for (int i = 0; i < LUT_ACT; i++)  L->act[i]  = (uint8_t)hve_bisect(m->act_l, i);
    for (int i = 0; i < LUT_ERR; i++)  L->err[i]  = (uint8_t)hve_bisect(m->err_l, i);
    for (int i = 0; i < LUT_SIDE; i++) L->side[i] = (uint8_t)hve_bisect(m->side_l, i);
    for (int i = 0; i < LUT_DIFF; i++) L->diff[i] = (uint8_t)hve_bisect(m->diff_l, i);
    for (int i = 0; i < LUT_LUM; i++)  L->lum[i]  = (uint8_t)hve_bisect(m->lum_l, i);
    for (int i = 0; i < LUT_MEXP; i++) L->mexp[i] = (uint8_t)hve_bisect(m->mexp_l, i);
    for (int i = 0; i < LUT_ADJ; i++)  L->adj[i]  = (uint8_t)hve_bisect(m->adj_l, i);
    for (int i = 0; i < LUT_ACT; i++)  L->act32[i]  = L->act[i];
    for (int i = 0; i < LUT_ERR; i++)  L->err32[i]  = L->err[i];
    for (int i = 0; i < LUT_SIDE; i++) L->side32[i] = L->side[i];
    for (int i = 0; i < LUT_DIFF; i++) L->diff32[i] = L->diff[i];
    for (int i = 0; i < LUT_LUM; i++)  L->lum32[i]  = L->lum[i];
}

#define LOOKUP(tbl, size, ladder, v) \
    ((uint64_t)(v) < (size) ? (int64_t)(tbl)[(v)] : hve_bisect((ladder), (v)))

/* --------------------------------------------------------------------------
 * Batched context derivation (encoder only)
 *
 * The pixel loop is serial because each pixel's context is built from its
 * neighbours' *decoded* values. On the encoder that constraint is imaginary:
 * this codec is lossless, so decoded equals source, and the encoder holds the
 * whole plane before it codes anything. Every context index below is a pure
 * function of the source - nothing here reads a coded bit - so a whole row of
 * them can be derived up front in flat loops a compiler can vectorise, leaving
 * the serial pass to do only the model and the arithmetic coder.
 *
 * This runs only when the blend, LMS and match stages are all off, which is
 * the `fast` preset. Those three adapt in scan order and genuinely cannot be
 * hoisted. The decoder can never use this path: it does not know a value until
 * it has decoded it.
 *
 * It must agree with the scalar derivation in hve_code_plane exactly, since
 * both write the same bitstream. tests/test_batched.py pins that.
 */
typedef struct {
    int64_t width, use_luma, inter_on;
    int64_t nerr, nlum, nside, ndiff, max_nb;
    int64_t k_zero, k_nb, i_zero, i_nb, kind_dir, kind_diff;
} derive_cfg;

typedef struct {
    int32_t *pred, *d, *actb, *sgn, *zctx, *dirc, *diffc, *nbb, *isel;
} derive_row_buf;

static void derive_row(const derive_cfg *c, const ladder_luts *restrict L,
                       const uint8_t *restrict src,
                       const int32_t *restrict prev,
                       const int32_t *restrict prev_err,
                       const uint8_t *restrict errmap,
                       const uint8_t *restrict mode_x,
                       const uint8_t *restrict ref_p,
                       const int32_t *restrict ref_y,
                       const int32_t *restrict ref_x,
                       const uint8_t *restrict ref, int64_t height,
                       int first_row, const derive_row_buf *b)
{
    const int64_t width = c->width;
    int32_t *restrict pred = b->pred;
    int32_t *restrict d = b->d;
    int32_t *restrict actb = b->actb;
    int32_t *restrict sgn = b->sgn;
    int64_t x;

    /* 1. MED prediction. The scalar form is the LOCO-I median predictor
     * written as a nest of comparisons; this is the same function with the
     * branches turned into min/max so it vectorises. */
    if (first_row) {
        pred[0] = 128;
        for (x = 1; x < width; x++)
            pred[x] = src[x - 1];
    } else {
        pred[0] = prev[0];
        for (x = 1; x < width; x++) {
            int32_t west = src[x - 1], north = prev[x], nwest = prev[x - 1];
            int32_t lo = north < west ? north : west;
            int32_t hi = north < west ? west : north;
            int32_t p = nwest >= hi ? lo
                      : (nwest <= lo ? hi : north + west - nwest);
            pred[x] = p > 255 ? 255 : (p < 0 ? 0 : p);
        }
    }

    /* 2. Temporally predicted pixels take the reference instead. */
    if (c->inter_on) {
        for (x = 0; x < width; x++) {
            b->isel[x] = mode_x[x];
            if (mode_x[x])
                pred[x] = ref[((int64_t)ref_p[x] * height + ref_y[x]) * width
                              + ref_x[x]];
        }
    } else {
        memset(b->isel, 0, (size_t)width * sizeof(int32_t));
    }

    /* 3. Residual and its magnitude. cur_err doubles as the magnitude row,
     * which the serial pass would otherwise fill one pixel at a time. */
    for (x = 0; x < width; x++) {
        int32_t v = ((src[x] - pred[x] + 128) & 255) - 128;
        d[x] = v;
    }

    /* 4. Activity and the sign pair. */
    actb[0] = 0;
    sgn[0] = 4;
    if (first_row) {
        for (x = 1; x < width; x++) {
            actb[x] = 0;
            sgn[x] = 4;
        }
    } else {
        for (x = 1; x < width; x++) {
            int32_t west = src[x - 1], north = prev[x], nwest = prev[x - 1];
            int32_t neast = (x + 1 < width) ? prev[x + 1] : north;
            int32_t d1 = ((west - nwest + 128) & 255) - 128;
            int32_t d2 = ((nwest - north + 128) & 255) - 128;
            int32_t d3 = ((north - neast + 128) & 255) - 128;
            int32_t act = (d1 < 0 ? -d1 : d1) + (d2 < 0 ? -d2 : d2)
                        + (d3 < 0 ? -d3 : d3);
            actb[x] = L->act32[act];
            sgn[x] = ((d1 > 0) - (d1 < 0) + 1) * 3 + ((d2 > 0) - (d2 < 0) + 1);
        }
    }

    /* 5. The error-neighbourhood contexts. west_err is the previous pixel's
     * magnitude, which step 3 already produced for the whole row.
     *
     * Everything here is deliberately int32. The obvious int64 spelling costs
     * the vectoriser outright ("unsupported data-type long int"), and every
     * context index in this model fits in far less than 32 bits - the largest
     * table has a few thousand entries - so the width buys nothing.
     *
     * The luma term is multiplied by a 0/1 flag rather than branched on, so
     * the loop body has no condition in it; when the plane has no luma map
     * errmap points at a zero row. */
    {
        const int32_t nerr32 = (int32_t)c->nerr, nlum32 = (int32_t)c->nlum;
        const int32_t nside32 = (int32_t)c->nside;
        const int32_t nb_span = (int32_t)(c->max_nb + 1);
        const int32_t kdir = (int32_t)c->kind_dir;
        const int32_t zi = (int32_t)c->i_zero, zk = (int32_t)c->k_zero;
        const int32_t ni = (int32_t)c->i_nb, nk = (int32_t)c->k_nb;
        const int32_t *restrict Ler = L->err32;
        const int32_t *restrict Lsd = L->side32;
        const int32_t *restrict Llm = L->lum32;
        const int32_t lum_on = (int32_t)c->use_luma;
        int32_t *restrict zc = b->zctx;
        int32_t *restrict nb = b->nbb;
        int32_t *restrict dc = b->dirc;
        const int32_t *restrict isel = b->isel;
        /* The interior runs without any boundary tests in it, so the two ends
         * are done by hand: at x == 0 there is no west, and at the last column
         * there is no north-east. */
#define DERIVE_CTX(x_, we_, ne_, ex_)                                        \
        do {                                                                 \
            int32_t sub = actb[x_] * nerr32 + Ler[(we_) + (ne_) + (ex_)];    \
            int32_t lum = lum_on * Llm[errmap[x_]];                          \
            zc[x_] = (isel[x_] ? zi : zk) + sub * nlum32 + lum;              \
            nb[x_] = (isel[x_] ? ni : nk) + sub * nb_span;                   \
            dc[x_] = kdir + (sgn[x_] * nside32 + Lsd[we_]) * nside32         \
                   + Lsd[ne_];                                               \
        } while (0)
        if (first_row) {
            DERIVE_CTX(0, 0, 0, 0);
            for (x = 1; x < width; x++) {
                int32_t we = d[x - 1] < 0 ? -d[x - 1] : d[x - 1];
                DERIVE_CTX(x, we, 0, 0);
            }
        } else {
            DERIVE_CTX(0, 0, prev_err[0], width > 1 ? prev_err[1] : 0);
            for (x = 1; x + 1 < width; x++) {
                int32_t we = d[x - 1] < 0 ? -d[x - 1] : d[x - 1];
                DERIVE_CTX(x, we, prev_err[x], prev_err[x + 1]);
            }
            if (width > 1) {
                int32_t we = d[width - 2] < 0 ? -d[width - 2] : d[width - 2];
                DERIVE_CTX(width - 1, we, prev_err[width - 1], 0);
            }
        }
#undef DERIVE_CTX
    }

    /* 6. The gradient-pair context. */
    b->diffc[0] = (int32_t)c->kind_diff;
    if (first_row) {
        for (x = 1; x < width; x++)
            b->diffc[x] = (int32_t)c->kind_diff;
    } else {
        const int32_t kdiff = (int32_t)c->kind_diff;
        const int32_t ndiff32 = (int32_t)c->ndiff;
        const int32_t *restrict Ldf = L->diff32;
        int32_t *restrict fc = b->diffc;
        for (x = 1; x + 1 < width; x++) {
            int32_t west = src[x - 1], north = prev[x], nwest = prev[x - 1];
            int32_t dwn = west - north, dne = nwest - prev[x + 1];
            fc[x] = kdiff + Ldf[dwn >= 0 ? dwn : -dwn] * ndiff32
                  + Ldf[dne >= 0 ? dne : -dne];
        }
        if (width > 1) {
            int32_t west = src[width - 2], north = prev[width - 1];
            int32_t dwn = west - north, dne = prev[width - 2] - north;
            fc[width - 1] = kdiff + Ldf[dwn >= 0 ? dwn : -dwn] * ndiff32
                          + Ldf[dne >= 0 ? dne : -dne];
        }
    }
}

int hve_code_plane(int encode, uint8_t *plane, int64_t height, int64_t width,
                   const uint8_t *data, uint8_t *out, hve_rc *r, hve_model *m,
                   int64_t kind, int64_t use_luma, int64_t write_errmap,
                   const hve_inter *inter)
{
    const int64_t *params = m->params;
    const int64_t nact = params[P_NACT];
    const int64_t nerr = params[P_NERR];
    const int64_t nlum = params[P_NLUM];
    const int64_t nside = params[P_NSIDE];
    const int64_t ndiff = params[P_NDIFF];
    const int64_t nmatch = params[P_NMATCH];
    const int64_t max_nb = params[P_MAXNB];
    const int64_t match_max = params[P_MATCHMAX];
    const int64_t match_trust = params[P_MATCHTRUST];
    const int64_t wp_p1 = params[P_WP1];
    const int64_t wp_p2 = params[P_WP2];
    const int64_t hash_mask = params[P_HASHMASK];
    const int64_t adapt = params[P_ADAPT];
    const int64_t lms_n = params[P_LMSN];
    const int64_t lms_wshift = params[P_LMSWSHIFT];
    const int64_t lms_rate = params[P_LMSRATE];
    const int64_t lms_eps = params[P_LMSEPS];
    const int64_t lms_step_clamp = params[P_LMSSTEP];
    const int64_t lms_wclamp = params[P_LMSWCLAMP];
    const int64_t nadj = params[P_NADJ];
    const int64_t lms_act_shift = params[P_LMSACTSHIFT];
    const int64_t lms_ndir = params[P_LMSNDIR];
    const int64_t nconf = m->conf_l.n + 1;
    const int64_t feat = params[P_FEATURES];
    const int f_blend = (feat & HVE_FEAT_BLEND) != 0;
    const int f_lms   = (feat & HVE_FEAT_LMS) != 0;
    const int f_match = (feat & HVE_FEAT_MATCH) != 0;
    const int f_mix   = (feat & HVE_FEAT_MIX) != 0;
    const int f_apm1  = (feat & HVE_FEAT_APM1) != 0;
    const int f_apm2  = (feat & HVE_FEAT_APM2) != 0;
    const int f_nbmix = (feat & HVE_FEAT_NBMIX) != 0;

    const int64_t k_zero = kind * nact * nerr * nlum;
    const int64_t k_nb = kind * nact * nerr * (max_nb + 1);
    const int64_t k_sign = kind * 9 * 4;
    const int64_t k_mant = kind * (max_nb + 1) * 2;
    const int64_t ikind = kind + 4;
    const int64_t i_zero = ikind * nact * nerr * nlum;
    const int64_t i_nb = ikind * nact * nerr * (max_nb + 1);
    const int64_t i_sign = ikind * 9 * 4;
    const int64_t i_mant = ikind * (max_nb + 1) * 2;
    const int64_t kind_dir = kind * 9 * nside * nside;
    const int64_t kind_diff = kind * ndiff * ndiff;
    const int64_t kind_match = kind * nmatch;
    const int64_t kind_mix = kind * nact;
    const int64_t kind_nbapm = kind * (max_nb + 1) * nact;
    const int64_t kind_lms = kind * (((nact + (1 << lms_act_shift) - 1)
                                      >> lms_act_shift) * lms_ndir * lms_n);
    const int64_t kind_conf = kind * nconf * nadj;

    const int64_t *stretch = m->stretch;
    const int64_t *sq = m->squash;
    int32_t *lmsw = m->lmsw;
    int64_t *mixw = m->mixw;
    int64_t *nbmixw = m->nbmixw;
    int64_t *apm0 = m->apm0, *apm1 = m->apm1, *apm2 = m->apm2;

    const int64_t inter_on = inter && inter->on;
    const int64_t nbx = inter_on ? inter->nbx : 1;
    const int64_t nby = inter_on ? inter->nby : 1;

    wtable wtab;
    wtable_init(&wtab, params);
    ladder_luts luts;
    luts_init(&luts, m);

    /* The batched derivation above needs the blend, LMS and match stages off,
     * because those three adapt in scan order, and it needs the source, so it
     * is an encoder path only. */
    const int batched = encode && !f_blend && !f_lms && !f_match
                        && hve_batched_enabled();

    /* One allocation for every row buffer, so a plane costs one malloc. */
    const int64_t w2 = width + 2;
    int32_t *mem = (int32_t *)calloc(5 * (size_t)width + 10 * w2 + 2 * (size_t)width
                                     + (batched ? 9 * (size_t)width : 0),
                                     sizeof(int32_t));
    uint8_t *bytes = (uint8_t *)calloc(3 * (size_t)width, 1);
    if (!mem || !bytes) {
        free(mem);
        free(bytes);
        return -1;
    }
    int32_t *prev = mem;
    int32_t *prev2 = prev + width;
    int32_t *cur = prev2 + width;
    int32_t *prev_err = cur + width;
    int32_t *cur_err = prev_err + width;
    int32_t *terr_prev = cur_err + width;
    int32_t *terr_cur = terr_prev + w2;
    int32_t *werr_prev = terr_cur + w2;          /* 4 x w2 */
    int32_t *werr_cur = werr_prev + 4 * w2;      /* 4 x w2 */
    int32_t *ref_y = werr_cur + 4 * w2;
    int32_t *ref_x = ref_y + width;
    uint8_t *mode_x = bytes;
    uint8_t *ref_p = bytes + width;
    /* Stays zero: the batched derivation multiplies the luma term out rather
     * than branching, so it always has a row to read. */
    const uint8_t *zero_row = bytes + 2 * width;

    derive_row_buf drow;
    derive_cfg dcfg;
    {
        int32_t *p = ref_x + width;
        drow.pred = p;      drow.d = p + width;      drow.actb = p + 2 * width;
        drow.sgn = p + 3 * width;  drow.zctx = p + 4 * width;
        drow.dirc = p + 5 * width; drow.diffc = p + 6 * width;
        drow.nbb = p + 7 * width;  drow.isel = p + 8 * width;
        dcfg.width = width;       dcfg.use_luma = use_luma;
        dcfg.inter_on = inter_on; dcfg.nerr = nerr;
        dcfg.nlum = nlum;         dcfg.nside = nside;
        dcfg.ndiff = ndiff;       dcfg.max_nb = max_nb;
        dcfg.k_zero = k_zero;     dcfg.k_nb = k_nb;
        dcfg.i_zero = i_zero;     dcfg.i_nb = i_nb;
        dcfg.kind_dir = kind_dir; dcfg.kind_diff = kind_diff;
    }

    int64_t ex[5];
    int32_t lms_x[HVE_LMS_MAX];

    if (f_match)
        memset(m->match_table, 0,
               ((size_t)hash_mask + 1) * sizeof(int32_t));
    int64_t flat_n = 0, match_pos = 0, match_len = 0;

    /* With the match and LMS stages off these two contexts never vary, so the
     * batched path reads them from here instead of re-deriving per pixel -
     * conf_ctx in particular was a linear ladder scan for a constant. */
    const int64_t flat_match_ctx = kind_match;
    const int64_t flat_conf_b = hve_bisect(m->conf_l, 0);
    const int64_t flat_conf_ctx = kind_conf + flat_conf_b * nadj
                                + LOOKUP(luts.adj, LUT_ADJ, m->adj_l, 0);

    for (int64_t y = 0; y < height; y++) {
        memset(cur, 0, (size_t)width * sizeof(int32_t));
        memset(cur_err, 0, (size_t)width * sizeof(int32_t));
        int64_t west = 0, west_err = 0;
        const int first_row = (y == 0);

        if (inter_on) {
            /* Walk blocks, not pixels. The per-pixel form needed an integer
             * division by a runtime block size for every sample, which the
             * compiler cannot strength-reduce; it was two of the hottest lines
             * in the kernel. Everything but ref_x is constant across a block. */
            int64_t by = y / inter->bs_y;
            if (by >= nby)
                by = nby - 1;
            for (int64_t bx = 0; bx < nbx; bx++) {
                int64_t x0 = bx * inter->bs_x, x1;
                /* The grid is sized from luma, and a plane whose subsampling
                 * does not divide evenly gets a grid wider than it is: odd
                 * sizes make luma_w / chroma_w truncate to 1, so bs_x stays
                 * 16 on a half-width plane. Blocks past the edge have no
                 * pixels, and the last block with any keeps them all -- which
                 * is what clamping bx to nbx - 1 used to do per pixel. */
                if (x0 >= width)
                    break;
                x1 = (bx == nbx - 1) ? width : x0 + inter->bs_x;
                if (x1 > width)
                    x1 = width;
                if (inter->modes[by * nbx + bx] != 1) {
                    memset(mode_x + x0, 0, (size_t)(x1 - x0));
                    continue;
                }
                int64_t hy = hve_fdiv(inter->mvs[(by * nbx + bx) * 2],
                                      inter->mv_sy);
                int64_t hx = hve_fdiv(inter->mvs[(by * nbx + bx) * 2 + 1],
                                      inter->mv_sx);
                int64_t ry = y + (hy >> 1), dx = hx >> 1;
                uint8_t phase = (uint8_t)((hy & 1) * 2 + (hx & 1));
                int32_t ryc = (int32_t)(ry < 0 ? 0
                                        : (ry >= height ? height - 1 : ry));
                memset(mode_x + x0, 1, (size_t)(x1 - x0));
                for (int64_t x = x0; x < x1; x++) {
                    int64_t rx = x + dx;
                    ref_p[x] = phase;
                    ref_y[x] = ryc;
                    ref_x[x] = (int32_t)(rx < 0 ? 0
                                         : (rx >= width ? width - 1 : rx));
                }
            }
        }

        if (batched)
            derive_row(&dcfg, &luts, plane + y * width, prev, prev_err,
                       use_luma ? m->errmap + y * m->errmap_stride : zero_row,
                       mode_x, ref_p, ref_y, ref_x,
                       inter_on ? inter->ref : NULL, height, first_row, &drow);

        for (int64_t x = 0; x < width; x++) {
            int64_t mval = -1, msign = 0, mexp_b = 0;
            int64_t north = 0, nwest = 0, neast = 0;
            int64_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            int lms_on = 0;
            int64_t lms_base_w = 0, lms_pred = 0, lms_adj = 0, energy = 0;
            int64_t pred = 0, act = 0, act_b = 0, sgn = 0;
            int64_t b_zero, b_nb, b_sign, b_mant, b_kind;
            int64_t north_err, err_sum, zctx, nbbase, dir_ctx, diff_ctx;
            int64_t match_ctx, conf_ctx, conf_b;

            if (batched) {
                /* Derived for the whole row already. With the match and LMS
                 * stages off, the match and confidence contexts are the same
                 * every pixel, so they are hoisted out of the plane entirely
                 * rather than re-bisected here. */
                pred = drow.pred[x];
                act_b = drow.actb[x];
                sgn = drow.sgn[x];
                zctx = drow.zctx[x];
                nbbase = drow.nbb[x];
                dir_ctx = drow.dirc[x];
                diff_ctx = drow.diffc[x];
                match_ctx = flat_match_ctx;
                conf_ctx = flat_conf_ctx;
                conf_b = flat_conf_b;
                if (drow.isel[x]) {
                    b_sign = i_sign;
                    b_mant = i_mant;
                    b_kind = ikind;
                } else {
                    b_sign = k_sign;
                    b_mant = k_mant;
                    b_kind = kind;
                }
                goto have_context;
            }

            if (first_row) {
                pred = (x == 0) ? 128 : west;
                act = 0;
                act_b = 0;
                sgn = 4;
            } else {
                north = prev[x];
                if (x == 0) {
                    pred = north;
                    act = 0;
                    act_b = 0;
                    sgn = 4;
                } else {
                    int64_t d1, d2, d3;
                    nwest = prev[x - 1];
                    neast = (x + 1 < width) ? prev[x + 1] : north;
                    if (nwest >= north) {
                        if (nwest >= west)
                            pred = north < west ? north : west;
                        else
                            pred = north + west - nwest;
                    } else if (nwest <= west) {
                        pred = north > west ? north : west;
                    } else {
                        pred = north + west - nwest;
                    }
                    if (pred > 255)
                        pred = 255;
                    else if (pred < 0)
                        pred = 0;
                    d1 = ((west - nwest + 128) & 255) - 128;
                    d2 = ((nwest - north + 128) & 255) - 128;
                    d3 = ((north - neast + 128) & 255) - 128;
                    act = (d1 < 0 ? -d1 : d1) + (d2 < 0 ? -d2 : d2)
                        + (d3 < 0 ? -d3 : d3);
                    act_b = LOOKUP(luts.act, LUT_ACT, m->act_l, act);
                    sgn = (d1 < 0 ? 0 : (d1 == 0 ? 1 : 2)) * 3
                        + (d2 < 0 ? 0 : (d2 == 0 ? 1 : 2));

                    if (f_blend) {
                        int64_t te_w = terr_cur[x];
                        int64_t te_n = terr_prev[x + 1];
                        int64_t te_nw = terr_prev[x];
                        int64_t te_ne = terr_prev[x + 2];
                        int64_t sum_wn = te_n + te_w;
                        int64_t a0, a1, a2, a3, m0, m1, m2, m3, total, blend;
                        q0 = west + neast - north;
                        q1 = north - (((sum_wn + te_ne) * wp_p1) >> 5);
                        q2 = west - (((sum_wn + te_nw) * wp_p2) >> 5);
                        q3 = pred;
                        a0 = (int64_t)werr_cur[0 * w2 + x] + werr_prev[0 * w2 + x]
                           + werr_prev[0 * w2 + x + 1] + werr_prev[0 * w2 + x + 2];
                        a1 = (int64_t)werr_cur[1 * w2 + x] + werr_prev[1 * w2 + x]
                           + werr_prev[1 * w2 + x + 1] + werr_prev[1 * w2 + x + 2];
                        a2 = (int64_t)werr_cur[2 * w2 + x] + werr_prev[2 * w2 + x]
                           + werr_prev[2 * w2 + x + 1] + werr_prev[2 * w2 + x + 2];
                        a3 = (int64_t)werr_cur[3 * w2 + x] + werr_prev[3 * w2 + x]
                           + werr_prev[3 * w2 + x + 1] + werr_prev[3 * w2 + x + 2];
                        m0 = wt(&wtab, 0, a0);
                        m1 = wt(&wtab, 1, a1);
                        m2 = wt(&wtab, 2, a2);
                        m3 = wt(&wtab, 3, a3);
                        total = m0 + m1 + m2 + m3;
                        {
                            int64_t num = q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3
                                        + (total >> 1);
                            if (num >= -0x7FFFFFFF && num <= 0x7FFFFFFF
                                && total <= 0x7FFFFFFF)
                                blend = hve_fdiv32((int32_t)num, (int32_t)total);
                            else
                                blend = hve_fdiv(num, total);
                        }
                        if (!(((te_n >= 0) == (te_w >= 0))
                              && ((te_w >= 0) == (te_nw >= 0)))) {
                            int64_t lo = north < west ? north : west;
                            int64_t hi = north > west ? north : west;
                            if (neast < lo)
                                lo = neast;
                            else if (neast > hi)
                                hi = neast;
                            if (blend < lo)
                                blend = lo;
                            else if (blend > hi)
                                blend = hi;
                        }
                        pred = blend > 255 ? 255 : (blend < 0 ? 0 : blend);
                    }

                    if (f_lms && !(inter_on && mode_x[x])) {
                        int64_t wwest = x >= 2 ? cur[x - 2] : west;
                        int64_t wwwest = x >= 3 ? cur[x - 3] : wwest;
                        int64_t nwwest = x >= 2 ? prev[x - 2] : nwest;
                        int64_t neeast = (x + 2 < width) ? prev[x + 2] : neast;
                        int64_t nnorth, nnwest, nneast, dh, dv, dd, gap, acc;
                        if (y >= 2) {
                            nnorth = prev2[x];
                            nnwest = prev2[x - 1];
                            nneast = (x + 1 < width) ? prev2[x + 1] : nnorth;
                        } else {
                            nnorth = north;
                            nnwest = nwest;
                            nneast = neast;
                        }
#define ABS64(v) ((v) < 0 ? -(v) : (v))
                        dh = ABS64(west - wwest) + ABS64(north - nwest)
                           + ABS64(north - neast);
                        dv = ABS64(west - nwest) + ABS64(north - nnorth)
                           + ABS64(neast - nneast);
#undef ABS64
                        dd = dv - dh;
                        if (dd > 80) {
                            gap = west;
                        } else if (dd < -80) {
                            gap = north;
                        } else {
                            gap = (west + north) / 2 + hve_fdiv(neast - nwest, 4);
                            if (dd > 32)
                                gap = hve_fdiv(gap + west, 2);
                            else if (dd > 8)
                                gap = hve_fdiv(3 * gap + west, 4);
                            else if (dd < -32)
                                gap = hve_fdiv(gap + north, 2);
                            else if (dd < -8)
                                gap = hve_fdiv(3 * gap + north, 4);
                        }

                        lms_x[0] = west - pred;
                        lms_x[1] = north - pred;
                        lms_x[2] = nwest - pred;
                        lms_x[3] = neast - pred;
                        lms_x[4] = wwest - pred;
                        lms_x[5] = nnorth - pred;
                        lms_x[6] = nnwest - pred;
                        lms_x[7] = nneast - pred;
                        lms_x[8] = q3 - pred;
                        lms_x[9] = gap - pred;
                        lms_x[10] = wwwest - pred;
                        lms_x[11] = nwwest - pred;
                        lms_x[12] = neeast - pred;
                        lms_base_w = kind_lms
                            + ((act_b >> lms_act_shift) * lms_ndir
                               + (dd < -32 ? 0 : (dd <= 0 ? 1 : (dd <= 32 ? 2 : 3))))
                              * lms_n;
                        acc = 0;
                        energy = lms_eps;
                        {
                            const int32_t *wv = lmsw + lms_base_w;
                            int32_t en = 0;
                            for (int64_t i = 0; i < lms_n; i++) {
                                int32_t xi = lms_x[i];
                                acc += (int64_t)wv[i] * xi;
                                en += xi * xi;
                            }
                            energy += en;
                        }
                        lms_adj = acc >> lms_wshift;
                        {
                            int64_t adj = pred + lms_adj;
                            pred = adj > 255 ? 255 : (adj < 0 ? 0 : adj);
                        }
                        lms_pred = pred;
                        lms_on = 1;
                    }
                }
            }

            if (f_match && !first_row && x) {
                int64_t mhash = (((int64_t)west * 0x2F0FD693
                                  + (int64_t)north * 0x9E3779B1
                                  + (int64_t)nwest * 0x85EBCA77
                                  + (int64_t)neast * 0xC2B2AE3D) >> 8) & hash_mask;
                if (match_len == 0)
                    match_pos = m->match_table[mhash];
                if (match_pos > 0 && match_pos < flat_n)
                    mval = m->flat[match_pos];
                m->match_table[mhash] = (int32_t)flat_n;
            }

            if (inter_on && mode_x[x]) {
                pred = inter->ref[((int64_t)ref_p[x] * height + ref_y[x]) * width
                                  + ref_x[x]];
                b_zero = i_zero;
                b_nb = i_nb;
                b_sign = i_sign;
                b_mant = i_mant;
                b_kind = ikind;
            } else {
                b_zero = k_zero;
                b_nb = k_nb;
                b_sign = k_sign;
                b_mant = k_mant;
                b_kind = kind;
            }

            north_err = first_row ? 0 : prev_err[x];
            if (first_row) {
                err_sum = west_err;
            } else {
                err_sum = west_err + north_err;
                if (x + 1 < width)
                    err_sum += prev_err[x + 1];
            }
            int64_t sub = act_b * nerr
                        + LOOKUP(luts.err, LUT_ERR, m->err_l, err_sum);
            int64_t lum = use_luma
                ? LOOKUP(luts.lum, LUT_LUM, m->lum_l,
                         m->errmap[y * m->errmap_stride + x]) : 0;
            zctx = b_zero + sub * nlum + lum;
            nbbase = b_nb + sub * (max_nb + 1);

            dir_ctx = kind_dir
                + (sgn * nside
                   + LOOKUP(luts.side, LUT_SIDE, m->side_l, west_err)) * nside
                + LOOKUP(luts.side, LUT_SIDE, m->side_l, north_err);
            if (first_row || x == 0) {
                diff_ctx = kind_diff;
            } else {
                int64_t dwn = west - north;
                int64_t dne = nwest - neast;
                diff_ctx = kind_diff
                    + LOOKUP(luts.diff, LUT_DIFF, m->diff_l,
                             dwn >= 0 ? dwn : -dwn) * ndiff
                    + LOOKUP(luts.diff, LUT_DIFF, m->diff_l,
                             dne >= 0 ? dne : -dne);
            }
            if (mval < 0) {
                match_ctx = kind_match;
                msign = 0;
            } else {
                int64_t agree, hit, mexp, ae;
                if (mval == pred)
                    agree = 0;
                else if (mval - pred > -3 && mval - pred < 3)
                    agree = 1;
                else
                    agree = 2;
                hit = match_len < match_max ? match_len : match_max;
                match_ctx = kind_match + 1 + hit * 3 + agree;
                if (match_len >= match_trust && !(inter_on && mode_x[x]))
                    pred = mval;
                mexp = mval - pred;
                msign = mexp < 0 ? 1 : (mexp == 0 ? 2 : 3);
                ae = mexp >= 0 ? mexp : -mexp;
                mexp_b = 1 + LOOKUP(luts.mexp, LUT_MEXP, m->mexp_l, ae);
            }

            conf_b = hve_bisect(m->conf_l, energy);
            conf_ctx = kind_conf + conf_b * nadj
                + LOOKUP(luts.adj, LUT_ADJ, m->adj_l,
                         lms_adj >= 0 ? lms_adj : -lms_adj);

        have_context:
            ex[0] = stretch[4095 - (m->zero_p[zctx] >> 3)];
            ex[1] = stretch[4095 - (m->dir_p[dir_ctx] >> 3)];
            ex[2] = stretch[4095 - (m->diff_p[diff_ctx] >> 3)];
            ex[3] = stretch[4095 - (m->match_p[match_ctx] >> 3)];
            ex[4] = stretch[4095 - (m->conf_p[conf_ctx] >> 3)];
            int64_t mix_ctx = kind_mix + act_b;
            int64_t mbase = mix_ctx * 5;
            int64_t pr_mix;
            if (f_mix) {
                int64_t dot = ex[0] * mixw[mbase] + ex[1] * mixw[mbase + 1]
                            + ex[2] * mixw[mbase + 2] + ex[3] * mixw[mbase + 3]
                            + ex[4] * mixw[mbase + 4];
                pr_mix = sq_of(sq, dot >> 16);
            } else {
                /* the primary context model alone, which is what the mixer's
                 * expert 0 starts out as a copy of */
                pr_mix = 4095 - (m->zero_p[zctx] >> 3);
                if (pr_mix < 1)
                    pr_mix = 1;
            }

            int64_t pr1 = pr_mix, aupd = 0, aupd3 = 0;
            if (f_apm1) {
                int64_t s = stretch[pr_mix] + 2048;
                int64_t aw = s & 127;
                int64_t aidx = mix_ctx * 33 + (s >> 7);
                int64_t refined = (apm0[aidx] * (128 - aw)
                                   + apm0[aidx + 1] * aw) >> 11;
                aupd = aidx + (aw >= 64 ? 1 : 0);
                pr1 = (pr_mix + 3 * refined) >> 2;
            }
            if (f_apm2) {
                int64_t s3 = stretch[pr1] + 2048;
                int64_t aw3 = s3 & 127;
                int64_t aidx3 = ((b_kind * 7 + mexp_b) * 4 + msign) * 33
                              + (s3 >> 7);
                int64_t ref3 = (apm2[aidx3] * (128 - aw3)
                                + apm2[aidx3 + 1] * aw3) >> 11;
                aupd3 = aidx3 + (aw3 >= 64 ? 1 : 0);
                pr1 = (pr1 + 3 * ref3) >> 2;
            }
            if (pr1 < 1)
                pr1 = 1;
            else if (pr1 > 4095)
                pr1 = 4095;
            int64_t p_zero = (4096 - pr1) << 3;

            int64_t d = 0, mag = 0, nonzero, value;
            if (encode) {
                d = ((plane[y * width + x] - pred + 128) & 255) - 128;
                mag = d < 0 ? -d : d;
                nonzero = mag ? 1 : 0;
                m->stats[0]++;
                m->stats[1] += nonzero;
                enc_bit_p(r, out, p_zero, nonzero);
            } else {
                nonzero = dec_bit_p(r, data, p_zero);
            }

            {
                int64_t p;
                p = m->zero_p[zctx];
                m->zero_p[zctx] = nonzero ? p - (p >> adapt)
                                          : p + ((32768 - p) >> adapt);
                p = m->dir_p[dir_ctx];
                m->dir_p[dir_ctx] = nonzero ? p - (p >> adapt)
                                            : p + ((32768 - p) >> adapt);
                p = m->diff_p[diff_ctx];
                m->diff_p[diff_ctx] = nonzero ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
                p = m->match_p[match_ctx];
                m->match_p[match_ctx] = nonzero ? p - (p >> adapt)
                                                : p + ((32768 - p) >> adapt);
                p = m->conf_p[conf_ctx];
                m->conf_p[conf_ctx] = nonzero ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
            }
            {
                int64_t target = nonzero ? 65535 : 0;
                if (f_mix) {
                    int64_t err = ((nonzero << 12) - pr_mix) * HVE_MIX_RATE;
                    mixw[mbase] += (ex[0] * err + 0x8000) >> 16;
                    mixw[mbase + 1] += (ex[1] * err + 0x8000) >> 16;
                    mixw[mbase + 2] += (ex[2] * err + 0x8000) >> 16;
                    mixw[mbase + 3] += (ex[3] * err + 0x8000) >> 16;
                    mixw[mbase + 4] += (ex[4] * err + 0x8000) >> 16;
                }
                if (f_apm1)
                    apm0[aupd] += (target - apm0[aupd]) >> HVE_APM_RATE;
                if (f_apm2)
                    apm2[aupd3] += (target - apm2[aupd3]) >> HVE_APM_RATE;
            }

            if (encode) {
                if (mag) {
                    int64_t v, nb, t, limit;
                    enc_bit(r, out, m->sign_p, b_sign + sgn * 4 + msign,
                            d < 0 ? 1 : 0);
                    v = mag - 1;
                    nb = 0;
                    t = v;
                    while (t) {
                        nb++;
                        t >>= 1;
                    }
                    m->stats[3] += nb;
                    limit = nb < max_nb ? nb : nb - 1;
                    for (int64_t i = 0; i <= limit; i++) {
                        int64_t more = i < nb ? 1 : 0;
                        int64_t ctx = nbbase + i;
                        int64_t mixc = b_kind * (max_nb + 1) + i;
                        int64_t mctx = mixc * 7 + mexp_b;
                        int64_t cctx = mixc * nconf + conf_b;
                        int64_t nb0 = stretch[4095 - (m->nb_p[ctx] >> 3)];
                        int64_t nb1 = stretch[4095 - (m->nbm_p[mctx] >> 3)];
                        int64_t nb2 = stretch[4095 - (m->nbc_p[cctx] >> 3)];
                        int64_t nmb = mixc * 3;
                        int64_t pr;
                        if (f_nbmix) {
                            int64_t ndot = nb0 * nbmixw[nmb]
                                         + nb1 * nbmixw[nmb + 1]
                                         + nb2 * nbmixw[nmb + 2];
                            pr = sq_of(sq, ndot >> 16);
                        } else {
                            pr = 4095 - (m->nb_p[ctx] >> 3);
                            if (pr < 1)
                                pr = 1;
                        }
                        int64_t pr_nbmix = pr;
                        int64_t actx = kind_nbapm + i * nact + act_b;
                        int64_t u2 = 0;
                        if (f_nbmix) {
                            int64_t s2 = stretch[pr] + 2048;
                            int64_t w2b = s2 & 127;
                            int64_t i2 = actx * 33 + (s2 >> 7);
                            int64_t ref2 = (apm1[i2] * (128 - w2b)
                                            + apm1[i2 + 1] * w2b) >> 11;
                            u2 = i2 + (w2b >= 64 ? 1 : 0);
                            pr = (pr + 3 * ref2) >> 2;
                        }
                        int64_t p, nerr2, t2;
                        if (pr < 1)
                            pr = 1;
                        else if (pr > 4095)
                            pr = 4095;
                        enc_bit_p(r, out, (4096 - pr) << 3, more);
                        p = m->nb_p[ctx];
                        m->nb_p[ctx] = more ? p - (p >> adapt)
                                            : p + ((32768 - p) >> adapt);
                        p = m->nbm_p[mctx];
                        m->nbm_p[mctx] = more ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
                        p = m->nbc_p[cctx];
                        m->nbc_p[cctx] = more ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
                        if (f_nbmix) {
                            nerr2 = ((more << 12) - pr_nbmix) * HVE_MIX_RATE;
                            nbmixw[nmb] += (nb0 * nerr2 + 0x8000) >> 16;
                            nbmixw[nmb + 1] += (nb1 * nerr2 + 0x8000) >> 16;
                            nbmixw[nmb + 2] += (nb2 * nerr2 + 0x8000) >> 16;
                            t2 = more ? 65535 : 0;
                            apm1[u2] += (t2 - apm1[u2]) >> HVE_APM_RATE;
                        }
                    }
                    if (nb >= 2) {
                        enc_bit(r, out, m->mant_p, b_mant + nb * 2,
                                (v >> (nb - 2)) & 1);
                        if (nb >= 3) {
                            enc_bit(r, out, m->mant_p, b_mant + nb * 2 + 1,
                                    (v >> (nb - 3)) & 1);
                            if (nb > 3) {
                                enc_bypass(r, out, v & ((1 << (nb - 3)) - 1),
                                           nb - 3);
                                m->stats[2] += nb - 3;
                            }
                        }
                    }
                }
                value = (pred + d) & 255;
            } else {
                if (nonzero) {
                    int64_t neg = dec_bit(r, data, m->sign_p,
                                          b_sign + sgn * 4 + msign);
                    int64_t nb = 0, v;
                    while (nb < max_nb) {
                        int64_t ctx = nbbase + nb;
                        int64_t mixc = b_kind * (max_nb + 1) + nb;
                        int64_t mctx = mixc * 7 + mexp_b;
                        int64_t cctx = mixc * nconf + conf_b;
                        int64_t nb0 = stretch[4095 - (m->nb_p[ctx] >> 3)];
                        int64_t nb1 = stretch[4095 - (m->nbm_p[mctx] >> 3)];
                        int64_t nb2 = stretch[4095 - (m->nbc_p[cctx] >> 3)];
                        int64_t nmb = mixc * 3;
                        int64_t pr;
                        if (f_nbmix) {
                            int64_t ndot = nb0 * nbmixw[nmb]
                                         + nb1 * nbmixw[nmb + 1]
                                         + nb2 * nbmixw[nmb + 2];
                            pr = sq_of(sq, ndot >> 16);
                        } else {
                            pr = 4095 - (m->nb_p[ctx] >> 3);
                            if (pr < 1)
                                pr = 1;
                        }
                        int64_t pr_nbmix = pr;
                        int64_t actx = kind_nbapm + nb * nact + act_b;
                        int64_t u2 = 0;
                        if (f_nbmix) {
                            int64_t s2 = stretch[pr] + 2048;
                            int64_t w2b = s2 & 127;
                            int64_t i2 = actx * 33 + (s2 >> 7);
                            int64_t ref2 = (apm1[i2] * (128 - w2b)
                                            + apm1[i2 + 1] * w2b) >> 11;
                            u2 = i2 + (w2b >= 64 ? 1 : 0);
                            pr = (pr + 3 * ref2) >> 2;
                        }
                        int64_t more, p, nerr2, t2;
                        if (pr < 1)
                            pr = 1;
                        else if (pr > 4095)
                            pr = 4095;
                        more = dec_bit_p(r, data, (4096 - pr) << 3);
                        p = m->nb_p[ctx];
                        m->nb_p[ctx] = more ? p - (p >> adapt)
                                            : p + ((32768 - p) >> adapt);
                        p = m->nbm_p[mctx];
                        m->nbm_p[mctx] = more ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
                        p = m->nbc_p[cctx];
                        m->nbc_p[cctx] = more ? p - (p >> adapt)
                                              : p + ((32768 - p) >> adapt);
                        if (f_nbmix) {
                            nerr2 = ((more << 12) - pr_nbmix) * HVE_MIX_RATE;
                            nbmixw[nmb] += (nb0 * nerr2 + 0x8000) >> 16;
                            nbmixw[nmb + 1] += (nb1 * nerr2 + 0x8000) >> 16;
                            nbmixw[nmb + 2] += (nb2 * nerr2 + 0x8000) >> 16;
                            t2 = more ? 65535 : 0;
                            apm1[u2] += (t2 - apm1[u2]) >> HVE_APM_RATE;
                        }
                        if (!more)
                            break;
                        nb++;
                    }
                    if (nb < 2) {
                        v = nb;
                    } else {
                        v = ((int64_t)1 << (nb - 1))
                          | (dec_bit(r, data, m->mant_p, b_mant + nb * 2)
                             << (nb - 2));
                        if (nb >= 3) {
                            v |= dec_bit(r, data, m->mant_p, b_mant + nb * 2 + 1)
                                 << (nb - 3);
                            if (nb > 3)
                                v |= dec_bypass(r, data, nb - 3);
                        }
                    }
                    mag = v + 1;
                    value = (neg ? pred - mag : pred + mag) & 255;
                } else {
                    mag = 0;
                    value = pred & 255;
                }
            }

            if (lms_on) {
                int64_t lerr = value - lms_pred;
                int64_t step = (((lerr < 0 ? -lerr : lerr) << lms_wshift)
                                / energy);
                if (step > lms_step_clamp)
                    step = lms_step_clamp;
                if (lerr < 0)
                    step = -step;
                if (step >= -0x7FFFFFFF && step <= 0x7FFFFFFF
                    && lms_wclamp <= 0x7FFFFFFF) {
                    int32_t st = (int32_t)step;
                    int32_t lo = (int32_t)-lms_wclamp, hi = (int32_t)lms_wclamp;
                    int32_t *wv = lmsw + lms_base_w;
                    int32_t rate = (int32_t)lms_rate;
                    for (int64_t i = 0; i < lms_n; i++) {
                        int32_t wi = wv[i] + ((st * lms_x[i]) >> rate);
                        wv[i] = wi > hi ? hi : (wi < lo ? lo : wi);
                    }
                } else {
                    for (int64_t i = 0; i < lms_n; i++) {
                        int64_t wi = lmsw[lms_base_w + i]
                                   + ((step * lms_x[i]) >> lms_rate);
                        if (wi > lms_wclamp)
                            wi = lms_wclamp;
                        else if (wi < -lms_wclamp)
                            wi = -lms_wclamp;
                        lmsw[lms_base_w + i] = (int32_t)wi;
                    }
                }
            }

            /* These rows feed the weighted blend and nothing else, so they are
             * dead work whenever it is off - and q0..q3 are only set inside
             * the blend, so filling them without it would store nonsense. */
            if (f_blend && !first_row && x) {
                int64_t e0 = q0 - value, e1 = q1 - value;
                int64_t e2 = q2 - value, e3 = q3 - value;
                terr_cur[x + 1] = (int32_t)(pred - value);
                werr_cur[0 * w2 + x + 1] = (int32_t)(e0 >= 0 ? e0 : -e0);
                werr_cur[1 * w2 + x + 1] = (int32_t)(e1 >= 0 ? e1 : -e1);
                werr_cur[2 * w2 + x + 1] = (int32_t)(e2 >= 0 ? e2 : -e2);
                werr_cur[3 * w2 + x + 1] = (int32_t)(e3 >= 0 ? e3 : -e3);
            }

            /* The flat history and the run counter exist for the match model. */
            if (f_match) {
                if (mval == value) {
                    match_pos++;
                    match_len++;
                } else {
                    match_len = 0;
                }
                m->flat[flat_n++] = (uint8_t)value;
            }

            cur[x] = (int32_t)value;
            cur_err[x] = (int32_t)mag;
            west = value;
            west_err = mag;
            if (!encode)
                plane[y * width + x] = (uint8_t)value;
        }

        if (write_errmap) {
            for (int64_t xx = 0; xx < width; xx++)
                m->errmap[y * m->errmap_stride + xx] = (uint8_t)cur_err[xx];
        }
        memcpy(prev2, prev, (size_t)width * sizeof(int32_t));
        memcpy(prev, cur, (size_t)width * sizeof(int32_t));
        memcpy(prev_err, cur_err, (size_t)width * sizeof(int32_t));
        memcpy(terr_prev, terr_cur, w2 * sizeof(int32_t));
        memset(terr_cur, 0, w2 * sizeof(int32_t));
        memcpy(werr_prev, werr_cur, 4 * w2 * sizeof(int32_t));
        memset(werr_cur, 0, 4 * w2 * sizeof(int32_t));
    }

    free(mem);
    free(bytes);
    return 0;
}
