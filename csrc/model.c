/* Builds the adaptive model bank the pixel loop mutates.
 *
 * The Python side does this in hve/native.py's Bank, from model.new_model().
 * Here the sizes and initial values come from csrc/model_constants.h, which is
 * generated from those same Python definitions — so there is still exactly one
 * place where the model's shape is decided.
 */

#include <stdlib.h>
#include <string.h>

#include "hve.h"
#include "hvefmt.h"
#include "model_constants.h"
#include "hvemodel.h"

static int64_t g_features = -1;

void hve_set_features(int64_t features)
{
    g_features = features;
}

int64_t hve_default_features(void)
{
    return g_features >= 0 ? g_features : HVE_PARAMS[P_FEATURES];
}

static int64_t *fill(size_t n, int64_t v)
{
    int64_t *p = (int64_t *)malloc(n * sizeof(int64_t));
    if (!p)
        return NULL;
    for (size_t i = 0; i < n; i++)
        p[i] = v;
    return p;
}

static int64_t *widen(const int16_t *src, size_t n)
{
    int64_t *p = (int64_t *)malloc(n * sizeof(int64_t));
    if (!p)
        return NULL;
    for (size_t i = 0; i < n; i++)
        p[i] = src[i];
    return p;
}

/* Mixer weights: expert 0 starts at unity and the rest at zero, so the mixer
 * opens as an exact copy of the model it replaces and can only improve. */
static int64_t *mixer_weights(size_t nctx, size_t n)
{
    int64_t *p = (int64_t *)calloc(nctx * n, sizeof(int64_t));
    if (!p)
        return NULL;
    for (size_t c = 0; c < nctx; c++)
        p[c * n] = HVE_WEIGHT_ONE;
    return p;
}

/* Every APM context starts from the same identity map. */
static int64_t *apm_table(size_t nctx)
{
    int64_t *p = (int64_t *)malloc(nctx * HVE_APM_BUCKETS * sizeof(int64_t));
    if (!p)
        return NULL;
    for (size_t c = 0; c < nctx; c++)
        for (int j = 0; j < HVE_APM_BUCKETS; j++)
            p[c * HVE_APM_BUCKETS + j] = HVE_APM_INIT[j];
    return p;
}

#define LADDER(field, arr) \
    do { \
        m->m.field.v = (arr); \
        m->m.field.n = (int64_t)(sizeof(arr) / sizeof((arr)[0])); \
    } while (0)

void hve_bank_free(hve_bank *m)
{
    if (!m)
        return;
    for (size_t i = 0; i < sizeof(m->owned) / sizeof(m->owned[0]); i++)
        free(m->owned[i]);
    free(m->match_table);
    free(m->flat);
    free(m->errmap);
    free(m->mode_p);
    free(m->mv_zero);
    free(m->mv_sign);
    free(m->mv_mag);
    memset(m, 0, sizeof(*m));
}

int hve_bank_init(hve_bank *m, int64_t luma_h, int64_t luma_w,
                  int64_t features)
{
    memset(m, 0, sizeof(*m));
    int n = 0;

#define OWN(expr) (m->owned[n++] = (void *)(expr))
    if (!(m->m.zero_p = OWN(fill(HVE_N_ZERO, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.dir_p = OWN(fill(HVE_N_DIR, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.diff_p = OWN(fill(HVE_N_DIFF, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.match_p = OWN(fill(HVE_N_MATCH, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.sign_p = OWN(fill(HVE_N_SIGN, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.nb_p = OWN(fill(HVE_N_NB, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.nbm_p = OWN(fill(HVE_N_NBM, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.mant_p = OWN(fill(HVE_N_MANT, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.conf_p = OWN(fill(HVE_N_CONF, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.nbc_p = OWN(fill(HVE_N_NBC, HVE_PROB_INIT)))) goto oom;
    if (!(m->m.mixw = OWN(mixer_weights(HVE_N_MIXW / HVE_ZERO_EXPERTS,
                                        HVE_ZERO_EXPERTS)))) goto oom;
    if (!(m->m.nbmixw = OWN(mixer_weights(HVE_N_NBMIXW / HVE_NB_EXPERTS,
                                          HVE_NB_EXPERTS)))) goto oom;
    if (!(m->m.apm0 = OWN(apm_table(HVE_N_APM0 / HVE_APM_BUCKETS)))) goto oom;
    if (!(m->m.apm1 = OWN(apm_table(HVE_N_APM1 / HVE_APM_BUCKETS)))) goto oom;
    if (!(m->m.apm2 = OWN(apm_table(HVE_N_APM2 / HVE_APM_BUCKETS)))) goto oom;
    if (!(m->m.stretch = OWN(widen(HVE_STRETCH,
                                   sizeof(HVE_STRETCH) / sizeof(int16_t))))) goto oom;
    if (!(m->m.squash = OWN(widen(HVE_SQUASH,
                                  sizeof(HVE_SQUASH) / sizeof(int16_t))))) goto oom;
    if (!(m->m.stats = OWN(fill(8, 0)))) goto oom;
#undef OWN

    m->m.lmsw = (int32_t *)calloc(HVE_N_LMSW, sizeof(int32_t));
    if (!m->m.lmsw) goto oom;
    m->owned[n++] = m->m.lmsw;

    LADDER(act_l, HVE_LADDER_ACT);
    LADDER(err_l, HVE_LADDER_ERR);
    LADDER(lum_l, HVE_LADDER_LUM);
    LADDER(side_l, HVE_LADDER_SIDE);
    LADDER(diff_l, HVE_LADDER_DIFF);
    LADDER(mexp_l, HVE_LADDER_MEXP);
    LADDER(conf_l, HVE_LADDER_CONF);
    LADDER(adj_l, HVE_LADDER_ADJ);

    for (int i = 0; i < P_COUNT; i++)
        m->params[i] = HVE_PARAMS[i];
    m->params[P_FEATURES] = features;
    m->m.params = m->params;
    const int want_match = (features & HVE_FEAT_MATCH) != 0;
    /* 4 MB of hash table, and the pixel loop clears it once per plane. With
     * the match model switched off none of that is read, so allocating and
     * clearing it is pure waste - and with slices it is waste multiplied by
     * the slice count, which was enough to flatten the threading speedup. */
    if (want_match) {
        m->match_table = (int32_t *)calloc((size_t)HVE_MATCH_HASH_MASK + 1,
                                           sizeof(int32_t));
        if (!m->match_table)
            goto oom;
    }
    m->flat = (uint8_t *)calloc((size_t)luma_h * luma_w, 1);
    m->errmap = (uint8_t *)calloc((size_t)luma_h * luma_w, 1);
    if (!m->flat || !m->errmap)
        goto oom;
    m->m.match_table = m->match_table;
    m->m.flat = m->flat;
    m->m.errmap = m->errmap;
    m->m.errmap_stride = luma_w;
    return 0;

oom:
    hve_set_error("out of memory building the model bank");
    hve_bank_free(m);
    return -1;
}

int hve_bank_video(hve_bank *m)
{
    m->mode_p = fill(4, HVE_PROB_INIT);
    m->mv_zero = fill(2, HVE_PROB_INIT);
    m->mv_sign = fill(2, HVE_PROB_INIT);
    m->mv_mag = fill(2 * (HVE_MV_MAX + 1), HVE_PROB_INIT);
    if (!m->mode_p || !m->mv_zero || !m->mv_sign || !m->mv_mag) {
        hve_set_error("out of memory building the video model bank");
        return -1;
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * range-coder wrapper, mirroring hve/native.py's Coder
 */

int hve_coder_encode_init(hve_coder *c, size_t capacity)
{
    memset(c, 0, sizeof(*c));
    c->rc.s[1] = 0xFFFFFFFF;
    c->rc.s[3] = 1;
    c->out = (uint8_t *)calloc(capacity ? capacity : 1, 1);
    if (!c->out) {
        hve_set_error("out of memory reserving %zu bytes of output", capacity);
        return -1;
    }
    c->cap = capacity;
    return 0;
}

void hve_coder_decode_init(hve_coder *c, const uint8_t *payload, size_t n)
{
    memset(c, 0, sizeof(*c));
    c->rc.s[1] = 0xFFFFFFFF;
    c->data = payload;
    /* The first byte is the coder's leading zero; the next four seed `code`. */
    int64_t code = 0;
    for (size_t i = 1; i < 5 && i < n; i++)
        code = (code << 8) | payload[i];
    c->rc.s[0] = code;
    c->rc.s[2] = 5;
}

size_t hve_coder_finish(hve_coder *c)
{
    return (size_t)hve_finish_encode(&c->rc, c->out);
}

void hve_coder_free(hve_coder *c)
{
    free(c->out);
    c->out = NULL;
}
