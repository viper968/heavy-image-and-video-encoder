/* Shared declarations for the native codec core.
 *
 * This is the second implementation of the pixel loop. hve/model.py is the
 * first, and it is the definition of the format; this one is what actually
 * runs. The two must produce byte-identical streams; tests/test_native.py and
 * tests/test_codecs.py enforce it. Read docs/HANDOFF.md before changing
 * anything here.
 *
 * (There was briefly a third, hve/fast.py, in numba. It was removed once this
 * one existed: it cost a mirrored edit on every model change while adding no
 * coverage this file does not already provide. See docs/research.md.)
 *
 * Two language-level hazards separate C from Python, and both are handled
 * explicitly rather than assumed:
 *
 *   - `/` truncates toward zero in C and floors in Python. Every division whose
 *     numerator can be negative goes through hve_fdiv().
 *   - `>>` on a negative signed value is implementation-defined in C before
 *     C23. Every real compiler emits an arithmetic shift; the static assertion
 *     below fails the build on one that does not, rather than silently writing
 *     a stream the other two paths cannot read.
 *
 * Signed overflow is undefined in C but is well defined on Python's
 * arbitrary-precision integers. The build passes -fwrapv so the two agree.
 */

#ifndef HVE_H
#define HVE_H

#include <stdint.h>

enum { HVE_ASR_IS_ARITHMETIC = 1 / (int)(((int64_t)-8 >> 1) == (int64_t)-4) };

/* Floor division, matching Python's `//`. The divisor is always positive here,
 * but the sign test covers both to keep the helper honest. */
static inline int64_t hve_fdiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
        q--;
    return q;
}

/* The same, in 32-bit, for the two per-pixel divisions whose operands are
 * provably small. A 64-bit idiv costs roughly half again as much as a 32-bit
 * one, and these sit on the dependency chain that produces `pred`. The caller
 * checks the bound and falls back, so a parameter sweep that widens the range
 * loses the speed rather than the correctness. */
static inline int32_t hve_fdiv32(int32_t a, int32_t b)
{
    int32_t q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
        q--;
    return q;
}

/* A ladder plus its length, so the bisect below needs no separate argument. */
typedef struct {
    const int64_t *v;
    int64_t n;
} hve_ladder;

/* bisect.bisect_right over a short sorted array. */
static inline int64_t hve_bisect(hve_ladder l, int64_t v)
{
    int64_t i = 0;
    while (i < l.n && l.v[i] <= v)
        i++;
    return i;
}

/* Indices into the params array built by model.coder_params(). Append only. */
enum {
    P_NACT, P_NERR, P_NLUM, P_NSIDE, P_NDIFF, P_NMATCH, P_MAXNB, P_MATCHMAX,
    P_MATCHTRUST, P_WP1, P_WP2, P_WPSHIFT, P_W0, P_W1, P_W2, P_W3, P_HASHMASK,
    P_ADAPT, P_MVMAX,
    P_LMSN, P_LMSWSHIFT, P_LMSRATE, P_LMSEPS, P_LMSSTEP, P_LMSWCLAMP,
    P_NADJ, P_LMSACTSHIFT, P_LMSNDIR, P_FEATURES,
    P_COUNT
};

/* Feature bits in params[P_FEATURES]; see hve/model.py. */
enum {
    HVE_FEAT_BLEND = 1, HVE_FEAT_LMS = 2, HVE_FEAT_MATCH = 4,
    HVE_FEAT_MIX = 8, HVE_FEAT_APM1 = 16, HVE_FEAT_APM2 = 32,
    HVE_FEAT_NBMIX = 64, HVE_FEAT_ALL = 127
};

#define HVE_LMS_MAX 32          /* upper bound on LMS_NPRED, for stack arrays */

/* Range-coder state. Encoding uses low/range/cache/cache_size/pos; decoding
 * reuses the same five slots as code/range/pos, so one five-element array
 * serves both directions exactly as hve/rc.py's two classes do. */
typedef struct {
    int64_t s[5];
} hve_rc;

/* Every array the pixel loop touches. Python owns the memory; this is a view.
 *
 * The probability banks stay int64 because Python builds them from
 * model.new_model() and the parameter sweeps in tools/ read them back. The
 * scratch arrays are narrowed: `flat` holds pixel values and `errmap` holds
 * residual magnitudes, both of which fit in a byte, and `match_table` holds a
 * sample index. At 1080p that takes the match model's two random-access
 * working sets from 24 MB to 6 MB, which is the single largest speedup in this
 * file. The reference implementation cannot do the same, because its arrays are
 * Python lists whose element type is not something it gets to choose.
 */
typedef struct {
    int64_t *zero_p, *dir_p, *diff_p, *match_p, *sign_p, *nb_p, *nbm_p,
            *mant_p, *conf_p, *nbc_p;
    int64_t *mixw, *nbmixw;
    /* The combiner's weights are clamped to +-LMS_WCLAMP every update, so they
     * fit in 32 bits with three bits to spare. Keeping them there lets the
     * 13-tap dot product and its 13-tap update vectorise. */
    int32_t *lmsw;
    int64_t *apm0, *apm1, *apm2;
    /* Narrowing these two to int16 was tried and is not worth it: it saves
     * 48 KB of table but costs a sign extension on every one of the six
     * lookups per pixel, and the result sat inside this machine's +-10%
     * run-to-run noise. Left wide. */
    const int64_t *stretch, *squash;
    hve_ladder act_l, err_l, lum_l, side_l, diff_l, mexp_l, conf_l, adj_l;
    int32_t *match_table;
    uint8_t *flat;
    /* The luma error map is shared with the chroma planes, which are smaller,
     * so it carries its own stride rather than reusing the plane width. */
    uint8_t *errmap;
    int64_t errmap_stride;
    int64_t *stats;
    const int64_t *params;
} hve_model;

/* Per-plane inter-prediction inputs. `on` is 0 for stills. */
typedef struct {
    int64_t on;
    const int64_t *modes;       /* (nby, nbx) */
    const int64_t *mvs;         /* (nby, nbx, 2), half-pel */
    int64_t nby, nbx;
    int64_t bs_y, bs_x, mv_sy, mv_sx;
    const uint8_t *ref;         /* (4, height, width) half-pel phases */
} hve_inter;

int hve_code_plane(int encode, uint8_t *plane, int64_t height, int64_t width,
                   const uint8_t *data, uint8_t *out, hve_rc *rc,
                   hve_model *m, int64_t kind, int64_t use_luma,
                   int64_t write_errmap, const hve_inter *inter);

void hve_code_block_info(int encode, hve_rc *rc, const uint8_t *data,
                         uint8_t *out, int64_t *mode_p, int64_t *mv_zero,
                         int64_t *mv_sign, int64_t *mv_mag, int64_t *modes,
                         int64_t *mvs, int64_t nby, int64_t nbx,
                         int64_t mv_max);

int64_t hve_finish_encode(hve_rc *rc, uint8_t *out);

/* --- motion search (encoder only; see csrc/motion.c) --- */

void hve_halfpel_planes(const uint8_t *ref, int64_t h, int64_t w, uint8_t *out);

/* `cost_tbl` is video._COST_BYTE, passed in rather than recomputed so the
 * rounding of the log-cost proxy cannot drift from numpy's. */
int hve_motion_search(const uint8_t *cur, const uint8_t *ref,
                      int64_t h, int64_t w, int64_t bs, int64_t search,
                      int64_t halfpel_bias, int64_t pyramid_min_pixels,
                      int64_t pyramid_levels, int64_t refine_radius,
                      const int32_t *cost_tbl,
                      int32_t *mv_out, int64_t *cost_out, int nthreads);

int hve_spatial_cost(const uint8_t *cur, int64_t h, int64_t w, int64_t bs,
                     const int32_t *cost_tbl, int64_t *cost_out, int nthreads);

int hve_threads_default(void);

/* Override the core count the search uses; 0 restores the default. */
void hve_set_threads(int n);

#endif /* HVE_H */
