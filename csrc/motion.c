/* Motion estimation, threaded over blocks. Encoder only.
 *
 * This mirrors hve/video.motion_search exactly, including its tie-breaking:
 * every stage takes the first candidate that is *strictly* better, in the same
 * order numpy visits them, so the vectors chosen here are the vectors the numpy
 * path chooses and the two produce identical bitstreams. tests/test_native.py
 * checks that on real clips rather than trusting the argument.
 *
 * The reordering that makes it fast is that numpy iterates positions in the
 * outer loop and blocks in the inner one, because a whole-frame shift is the
 * only shape numpy is quick at. Every block's search is independent through the
 * entire pyramid, so here it is blocks outside and positions inside: the running
 * best stays in a register, the reference window stays in L1, and block rows are
 * handed out to threads.
 *
 * Two quirks of the numpy version are load-bearing and reproduced literally:
 *   - the coarse full search pads the *cost map* with zeros, so a block hanging
 *     off the right or bottom edge is scored only on its real pixels;
 *   - the refinement stages pad the *current frame* with zeros instead and
 *     still sample a clamped reference pixel, so those same positions do carry
 *     a cost.
 * They disagree with each other. Making them agree would change the vectors and
 * so the output, which is a compression experiment rather than a port.
 */

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hve.h"

#define MAXLEV 8

static inline int64_t clamp64(int64_t v, int64_t lo, int64_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* --------------------------------------------------------------------------
 * pixel-level helpers
 */

void hve_halfpel_planes(const uint8_t *ref, int64_t h, int64_t w, uint8_t *out)
{
    const int64_t n = h * w;
    for (int64_t y = 0; y < h; y++) {
        const uint8_t *row = ref + y * w;
        const uint8_t *below = ref + (y + 1 < h ? y + 1 : h - 1) * w;
        uint8_t *o0 = out + y * w;
        uint8_t *o1 = o0 + n;
        uint8_t *o2 = o1 + n;
        uint8_t *o3 = o2 + n;
        for (int64_t x = 0; x < w; x++) {
            int64_t xr = x + 1 < w ? x + 1 : w - 1;
            int r = row[x], rr = row[xr], d = below[x], dd = below[xr];
            o0[x] = (uint8_t)r;
            o1[x] = (uint8_t)((r + rr + 1) >> 1);
            o2[x] = (uint8_t)((r + d + 1) >> 1);
            o3[x] = (uint8_t)((r + rr + d + dd + 2) >> 2);
        }
    }
}

/* Box-average down by two, replicating an odd last row or column. */
static void halve(const uint8_t *p, int64_t h, int64_t w, uint8_t *out,
                  int64_t oh, int64_t ow)
{
    for (int64_t y = 0; y < oh; y++) {
        int64_t y0 = 2 * y, y1 = 2 * y + 1;
        if (y0 >= h) y0 = h - 1;
        if (y1 >= h) y1 = h - 1;
        for (int64_t x = 0; x < ow; x++) {
            int64_t x0 = 2 * x, x1 = 2 * x + 1;
            if (x0 >= w) x0 = w - 1;
            if (x1 >= w) x1 = w - 1;
            out[y * ow + x] = (uint8_t)((p[y0 * w + x0] + p[y0 * w + x1]
                                         + p[y1 * w + x0] + p[y1 * w + x1] + 2) >> 2);
        }
    }
}

/* Coarse stage: out-of-frame pixels contribute nothing (zero-padded cost map). */
static int64_t cost_search(const uint8_t *cur, const uint8_t *ref,
                           int64_t h, int64_t w, const int32_t *tbl,
                           int64_t by, int64_t bx, int64_t bs,
                           int64_t dy, int64_t dx)
{
    int64_t total = 0;
    int64_t y0 = by * bs, x0 = bx * bs;
    int64_t ylim = y0 + bs < h ? y0 + bs : h;
    int64_t xlim = x0 + bs < w ? x0 + bs : w;
    for (int64_t y = y0; y < ylim; y++) {
        int64_t ry = clamp64(y + dy, 0, h - 1);
        const uint8_t *crow = cur + y * w;
        const uint8_t *rrow = ref + ry * w;
        for (int64_t x = x0; x < xlim; x++) {
            int64_t rx = clamp64(x + dx, 0, w - 1);
            total += tbl[(uint8_t)(crow[x] - rrow[rx])];
        }
    }
    return total;
}

/* Refinement stages: the current frame is zero-padded and the reference is
 * still sampled, so pixels past the edge do carry a cost. */
static int64_t cost_whole(const uint8_t *cur, const uint8_t *ref,
                          int64_t h, int64_t w, const int32_t *tbl,
                          int64_t by, int64_t bx, int64_t bs,
                          int64_t dy, int64_t dx)
{
    int64_t total = 0;
    int64_t y0 = by * bs, x0 = bx * bs;
    for (int64_t i = 0; i < bs; i++) {
        int64_t y = y0 + i;
        int64_t ry = clamp64(y + dy, 0, h - 1);
        const uint8_t *rrow = ref + ry * w;
        const uint8_t *crow = (y < h) ? cur + y * w : NULL;
        if (crow && x0 + bs <= w && x0 + dx >= 0 && x0 + bs + dx <= w) {
            /* interior fast path: no clamping and no padding in this row */
            const uint8_t *r = rrow + x0 + dx;
            const uint8_t *c = crow + x0;
            for (int64_t j = 0; j < bs; j++)
                total += tbl[(uint8_t)(c[j] - r[j])];
            continue;
        }
        for (int64_t j = 0; j < bs; j++) {
            int64_t x = x0 + j;
            int64_t rx = clamp64(x + dx, 0, w - 1);
            int cv = (crow && x < w) ? crow[x] : 0;
            total += tbl[(uint8_t)(cv - rrow[rx])];
        }
    }
    return total;
}

/* Half-pel stage: the vector is in half-pel units and its low bit selects one
 * of the four interpolated phases. An arithmetic shift floors, so a negative
 * vector splits to the correct side. */
static int64_t cost_at(const uint8_t *cur, const uint8_t *planes,
                       int64_t h, int64_t w, const int32_t *tbl,
                       int64_t by, int64_t bx, int64_t bs,
                       int64_t hy, int64_t hx)
{
    const uint8_t *ref = planes + ((hy & 1) * 2 + (hx & 1)) * h * w;
    return cost_whole(cur, ref, h, w, tbl, by, bx, bs, hy >> 1, hx >> 1);
}

/* --------------------------------------------------------------------------
 * the per-block pipeline
 */

typedef struct {
    const uint8_t *curs[MAXLEV + 1];
    const uint8_t *refs[MAXLEV + 1];
    int64_t hs[MAXLEV + 1], ws[MAXLEV + 1];
    const uint8_t *planes;
    const int32_t *tbl;
    int64_t h, w, bs, search, levels, radius, refine_radius, bias;
    int64_t nby, nbx;
    int32_t *mv_out;
    int64_t *cost_out;
    int64_t next_row;
    pthread_mutex_t lock;
} search_job;

static void search_block(const search_job *j, int64_t by, int64_t bx)
{
    const int64_t L = j->levels;
    int64_t my = 0, mx = 0, best = 0, cy, cx;
    int first = 1;

    for (int64_t dy = -j->radius; dy <= j->radius; dy++) {
        for (int64_t dx = -j->radius; dx <= j->radius; dx++) {
            int64_t c = cost_search(j->curs[L], j->refs[L], j->hs[L], j->ws[L],
                                    j->tbl, by, bx, j->bs >> L, dy, dx);
            if (first || c < best) {
                first = 0;
                best = c;
                my = dy;
                mx = dx;
            }
        }
    }

    for (int64_t l = L - 1; l >= 0; l--) {
        int64_t bsl = j->bs >> l;
        my *= 2;
        mx *= 2;
        best = cost_whole(j->curs[l], j->refs[l], j->hs[l], j->ws[l], j->tbl,
                          by, bx, bsl, my, mx);
        cy = my;
        cx = mx;
        for (int64_t dy = -j->refine_radius; dy <= j->refine_radius; dy++) {
            for (int64_t dx = -j->refine_radius; dx <= j->refine_radius; dx++) {
                if (dy == 0 && dx == 0)
                    continue;
                int64_t c = cost_whole(j->curs[l], j->refs[l], j->hs[l],
                                       j->ws[l], j->tbl, by, bx, bsl,
                                       cy + dy, cx + dx);
                if (c < best) {
                    best = c;
                    my = cy + dy;
                    mx = cx + dx;
                }
            }
        }
    }

    my = clamp64(my, -j->search, j->search) * 2;
    mx = clamp64(mx, -j->search, j->search) * 2;
    best = cost_at(j->curs[0], j->planes, j->h, j->w, j->tbl, by, bx, j->bs,
                   my, mx);
    cy = my;
    cx = mx;
    for (int64_t sy = -1; sy <= 1; sy++) {
        for (int64_t sx = -1; sx <= 1; sx++) {
            if (sy == 0 && sx == 0)
                continue;
            /* Clamped to +-search whole pixels expressed in half-pels, so no
             * vector and no median of vectors can produce a differential above
             * MV_MAX. Without it the encoder can write a longer unary run than
             * the decoder reads back and silently corrupt the stream. */
            int64_t ky = clamp64(cy + sy, -2 * j->search, 2 * j->search);
            int64_t kx = clamp64(cx + sx, -2 * j->search, 2 * j->search);
            int64_t c = cost_at(j->curs[0], j->planes, j->h, j->w, j->tbl,
                                by, bx, j->bs, ky, kx);
            if (c + j->bias < best) {
                best = c;
                my = ky;
                mx = kx;
            }
        }
    }

    j->mv_out[(by * j->nbx + bx) * 2] = (int32_t)my;
    j->mv_out[(by * j->nbx + bx) * 2 + 1] = (int32_t)mx;
    j->cost_out[by * j->nbx + bx] = best;
}

static void *search_worker(void *arg)
{
    search_job *j = (search_job *)arg;
    for (;;) {
        int64_t by;
        pthread_mutex_lock(&j->lock);
        by = j->next_row++;
        pthread_mutex_unlock(&j->lock);
        if (by >= j->nby)
            return NULL;
        for (int64_t bx = 0; bx < j->nbx; bx++)
            search_block(j, by, bx);
    }
}

static void run_threaded(search_job *j, int nthreads)
{
    pthread_t tid[64];
    int made = 0;
    j->next_row = 0;
    pthread_mutex_init(&j->lock, NULL);
    if (nthreads > 64)
        nthreads = 64;
    if (nthreads < 1)
        nthreads = 1;
    for (int i = 0; i < nthreads - 1; i++)
        if (pthread_create(&tid[made], NULL, search_worker, j) == 0)
            made++;
    search_worker(j);
    for (int i = 0; i < made; i++)
        pthread_join(tid[i], NULL);
    pthread_mutex_destroy(&j->lock);
}

int hve_motion_search(const uint8_t *cur, const uint8_t *ref,
                      int64_t h, int64_t w, int64_t bs, int64_t search,
                      int64_t halfpel_bias, int64_t pyramid_min_pixels,
                      int64_t pyramid_levels, int64_t refine_radius,
                      const int32_t *cost_tbl,
                      int32_t *mv_out, int64_t *cost_out, int nthreads)
{
    search_job j;
    uint8_t *owned[2 * MAXLEV + 1];
    int nowned = 0;
    int64_t levels = 0;

    memset(&j, 0, sizeof(j));
    if (pyramid_levels > MAXLEV)
        pyramid_levels = MAXLEV;

    /* Halve until an exhaustive search is affordable, but only as deep as the
     * block size and search radius survive. Transcribed from video.py. */
    while (levels < pyramid_levels
           && ((h * w) >> (2 * levels)) > pyramid_min_pixels
           && (bs >> (levels + 1)) >= 4
           && (search >> (levels + 1)) >= 1
           && ((h < w ? h : w) >> (levels + 1)) >= bs)
        levels++;

    j.curs[0] = cur;
    j.refs[0] = ref;
    j.hs[0] = h;
    j.ws[0] = w;
    for (int64_t l = 1; l <= levels; l++) {
        int64_t ph = j.hs[l - 1], pw = j.ws[l - 1];
        int64_t oh = (ph + 1) / 2, ow = (pw + 1) / 2;
        uint8_t *c = (uint8_t *)malloc((size_t)oh * ow);
        uint8_t *r = (uint8_t *)malloc((size_t)oh * ow);
        if (!c || !r) {
            free(c);
            free(r);
            goto oom;
        }
        owned[nowned++] = c;
        owned[nowned++] = r;
        halve(j.curs[l - 1], ph, pw, c, oh, ow);
        halve(j.refs[l - 1], ph, pw, r, oh, ow);
        j.curs[l] = c;
        j.refs[l] = r;
        j.hs[l] = oh;
        j.ws[l] = ow;
    }

    {
        uint8_t *planes = (uint8_t *)malloc((size_t)h * w * 4);
        if (!planes)
            goto oom;
        owned[nowned++] = planes;
        hve_halfpel_planes(ref, h, w, planes);
        j.planes = planes;
    }

    j.tbl = cost_tbl;
    j.h = h;
    j.w = w;
    j.bs = bs;
    j.search = search;
    j.levels = levels;
    j.radius = -((-search) >> levels);          /* ceil, so coverage never shrinks */
    j.refine_radius = refine_radius;
    j.bias = halfpel_bias;
    j.nby = (h + bs - 1) / bs;
    j.nbx = (w + bs - 1) / bs;
    j.mv_out = mv_out;
    j.cost_out = cost_out;

    run_threaded(&j, nthreads);

    for (int i = 0; i < nowned; i++)
        free(owned[i]);
    return 0;

oom:
    for (int i = 0; i < nowned; i++)
        free(owned[i]);
    return -1;
}

/* --------------------------------------------------------------------------
 * spatial cost, for the mode decision
 */

typedef struct {
    const uint8_t *cur;
    const int32_t *tbl;
    int64_t h, w, bs, nby, nbx;
    int64_t *cost_out;
    int64_t next_row;
    pthread_mutex_t lock;
} spatial_job;

/* MED prediction with the same edge fallbacks transform.predict_plane uses:
 * outside neighbours read as zero, so row 0 predicts from the west neighbour,
 * column 0 from the north one, and the very first pixel from 128. */
static inline int med_pred(const uint8_t *cur, int64_t w, int64_t y, int64_t x)
{
    int north, west, nwest, hi, lo, planar;
    if (y == 0)
        return x == 0 ? 128 : cur[x - 1];
    if (x == 0)
        return cur[(y - 1) * w];
    north = cur[(y - 1) * w + x];
    west = cur[y * w + x - 1];
    nwest = cur[(y - 1) * w + x - 1];
    hi = north > west ? north : west;
    lo = north < west ? north : west;
    if (nwest >= hi)
        return lo;
    if (nwest <= lo)
        return hi;
    planar = north + west - nwest;
    return planar < 0 ? 0 : (planar > 255 ? 255 : planar);
}

static void *spatial_worker(void *arg)
{
    spatial_job *j = (spatial_job *)arg;
    for (;;) {
        int64_t by;
        pthread_mutex_lock(&j->lock);
        by = j->next_row++;
        pthread_mutex_unlock(&j->lock);
        if (by >= j->nby)
            return NULL;
        for (int64_t bx = 0; bx < j->nbx; bx++) {
            int64_t total = 0;
            int64_t y0 = by * j->bs, x0 = bx * j->bs;
            int64_t ylim = y0 + j->bs < j->h ? y0 + j->bs : j->h;
            int64_t xlim = x0 + j->bs < j->w ? x0 + j->bs : j->w;
            for (int64_t y = y0; y < ylim; y++)
                for (int64_t x = x0; x < xlim; x++)
                    total += j->tbl[(uint8_t)(j->cur[y * j->w + x]
                                              - med_pred(j->cur, j->w, y, x))];
            j->cost_out[by * j->nbx + bx] = total;
        }
    }
}

int hve_spatial_cost(const uint8_t *cur, int64_t h, int64_t w, int64_t bs,
                     const int32_t *cost_tbl, int64_t *cost_out, int nthreads)
{
    spatial_job j;
    pthread_t tid[64];
    int made = 0;

    memset(&j, 0, sizeof(j));
    j.cur = cur;
    j.tbl = cost_tbl;
    j.h = h;
    j.w = w;
    j.bs = bs;
    j.nby = (h + bs - 1) / bs;
    j.nbx = (w + bs - 1) / bs;
    j.cost_out = cost_out;
    pthread_mutex_init(&j.lock, NULL);
    if (nthreads > 64)
        nthreads = 64;
    if (nthreads < 1)
        nthreads = 1;
    for (int i = 0; i < nthreads - 1; i++)
        if (pthread_create(&tid[made], NULL, spatial_worker, &j) == 0)
            made++;
    spatial_worker(&j);
    for (int i = 0; i < made; i++)
        pthread_join(tid[i], NULL);
    pthread_mutex_destroy(&j.lock);
    return 0;
}

int hve_threads_default(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1)
        n = 1;
    if (n > 64)
        n = 64;
    return (int)n;
}
