/* Horizontal slices, so the pixel loop can use more than one core.
 *
 * The pixel loop is strictly serial - one range coder, one adaptive model,
 * every pixel depending on all prior ones - so the only way to use a second
 * core is to cut the picture into pieces that do not depend on each other.
 *
 * WHY NOT WAVEFRONTS. HEVC gets thread parallelism far more cheaply than this,
 * by starting each CTU row from a *copy* of the entropy state as it stood two
 * CTUs into the row above rather than from a reset. That works because CABAC's
 * whole context set is a few hundred bytes. This model is 4.5 MB - 472 KB of
 * probability banks, mixers and APMs, plus a 4 MB match hash table - so
 * checkpointing it per row would cost 304 MB of copying per 1080p frame.
 * Wavefronts are the better design and they are simply not available at this
 * model size; that is a consequence of the compression architecture, not an
 * implementation shortcut.
 *
 * So each slice is fully independent: its own model, its own range coder, its
 * own motion search, no prediction across the boundary. The price is that every
 * slice relearns the model from scratch, which is exactly what the measurements
 * in docs/research.md show dominates the cost - so slice counts should stay
 * small, and the count is a per-file choice rather than "one per core".
 *
 * The container is deliberately dumb: a slice file is a list of complete,
 * ordinary .hvi/.hvv sub-streams. That costs about 30 bytes of repeated header
 * per slice and buys a decoder that is a loop around the one that already
 * existed, with no new bitstream surface to get wrong.
 */

#include <stdlib.h>
#include <string.h>

#include "hve.h"
#include "hvefmt.h"
#include "model_constants.h"
#include "thread.h"

#define HVS_MAGIC "HVS1"
#define HVE_MAX_SLICES 64

/* Slice k of a picture `h` rows tall, with boundaries aligned to `sy` so a
 * subsampled chroma plane splits at an integer row too. Both sides derive this
 * the same way, so the bounds are never transmitted. */
static void slice_bounds(int64_t h, int64_t sy, int nslices, int k,
                         int64_t *y0, int64_t *y1)
{
    int64_t a = (k * h / nslices) / sy * sy;
    int64_t b = ((k + 1) * h / nslices) / sy * sy;
    if (k == nslices - 1)
        b = h;
    *y0 = a;
    *y1 = b < a ? a : b;
}

/* Pixels a slice should have before it is worth cutting another.
 *
 * The cost of slicing is a model-relearning cost, so it is governed by how much
 * data each slice has to amortise that over. Measured at 16 slices: 0.69% on
 * 1080p park_joy, 0.46% on in_to_tree, 1.52% on a bright Sintel segment - but
 * 4.10% on CIF foreman and 16.43% on a near-black clip. Roughly a quarter of a
 * megapixel per slice keeps the real cases under half a percent and stops small
 * or highly-compressible frames from being cut at all. */
#define HVE_PIXELS_PER_SLICE 250000

int hve_slice_auto(int64_t h, int64_t w, int64_t sy, int cores)
{
    int64_t want = (h * w) / HVE_PIXELS_PER_SLICE;
    if (want < 1)
        want = 1;
    if (cores > 0 && want > cores)
        want = cores;
    return hve_slice_count(h, sy, (int)want);
}

int hve_slice_count(int64_t h, int64_t sy, int wanted)
{
    if (wanted < 1)
        wanted = 1;
    if (wanted > HVE_MAX_SLICES)
        wanted = HVE_MAX_SLICES;
    /* A slice thinner than one block row cannot hold a motion vector, and a
     * zero-row slice cannot be coded at all. */
    int64_t min_rows = HVE_BLOCK > sy ? HVE_BLOCK : sy;
    while (wanted > 1 && h / wanted < min_rows)
        wanted--;
    return wanted;
}

/* --------------------------------------------------------------------------
 * shared worker plumbing
 */

typedef struct {
    int nslices;
    hve_buf *parts;
    int *status;
    /* stills */
    const uint8_t *img;
    int64_t h, w;
    int channels;
    int64_t features;
    /* video */
    const hve_frame *frames;
    int nframes;
    hve_frame *bands;           /* nslices * nframes, encoder-side */
    /* decode */
    const uint8_t **subs;
    size_t *sublen;
    uint8_t **out_img;
    int64_t *out_h;
    hve_frame **out_frames;
    int *out_nframes;
    int64_t next;
    hve_mutex lock;
    int (*run)(void *job, int k);
} slice_job;

static HVE_WORKER(slice_worker, arg)
{
    slice_job *j = (slice_job *)arg;
    for (;;) {
        int64_t k;
        hve_mutex_lock(&j->lock);
        k = j->next++;
        hve_mutex_unlock(&j->lock);
        if (k >= j->nslices)
            HVE_WORKER_RETURN;
        j->status[k] = j->run(j, (int)k);
    }
}

static int run_slices(slice_job *j, int nthreads)
{
    hve_thread tid[HVE_MAX_SLICES];
    int made = 0;
    /* The motion search inside each slice is itself threaded. Left alone that
     * gives nslices * ncores threads - 256 on a 16-core machine at 16 slices -
     * and the contention was bad enough that total CPU work rose 4.5x and wall
     * time got *worse* than at 4 slices. Divide the budget instead. */
    int inner = hve_threads_default() / (j->nslices > 0 ? j->nslices : 1);
    hve_set_threads(inner < 1 ? 1 : inner);
    j->next = 0;
    hve_mutex_init(&j->lock);
    if (nthreads > j->nslices)
        nthreads = j->nslices;
    if (nthreads < 1)
        nthreads = 1;
    for (int i = 0; i < nthreads - 1; i++)
        if (hve_thread_start(&tid[made], slice_worker, j) == 0)
            made++;
    slice_worker(j);
    for (int i = 0; i < made; i++)
        hve_thread_join(tid[i]);
    hve_mutex_free(&j->lock);
    hve_set_threads(0);                 /* restore the default budget */
    for (int k = 0; k < j->nslices; k++)
        if (j->status[k] != 0)
            return -1;
    return 0;
}

/* Wrapper: magic, slice count, one length per slice, then the sub-streams. */
static int write_wrapper(hve_buf *out, const hve_buf *parts, int nslices)
{
    int e = 0;
    e |= hve_buf_put(out, HVS_MAGIC, 4);
    e |= hve_buf_varint(out, (uint64_t)nslices);
    for (int k = 0; k < nslices; k++)
        e |= hve_buf_varint(out, parts[k].len);
    for (int k = 0; k < nslices; k++)
        e |= hve_buf_put(out, parts[k].data, parts[k].len);
    return e;
}

static int read_wrapper(const uint8_t *blob, size_t n, int *nslices,
                        const uint8_t **subs, size_t *sublen)
{
    hve_rd r = {blob, n, 0, 0};
    const uint8_t *magic = hve_rd_raw(&r, 4);
    if (!magic || memcmp(magic, HVS_MAGIC, 4) != 0) {
        hve_set_error("not a sliced hve stream");
        return -1;
    }
    int ns = (int)hve_rd_varint(&r);
    if (r.error || ns < 1 || ns > HVE_MAX_SLICES) {
        hve_set_error("corrupt slice header");
        return -1;
    }
    for (int k = 0; k < ns; k++)
        sublen[k] = (size_t)hve_rd_varint(&r);
    for (int k = 0; k < ns; k++) {
        subs[k] = hve_rd_raw(&r, sublen[k]);
        if (!subs[k]) {
            hve_set_error("truncated slice %d", k);
            return -1;
        }
    }
    *nslices = ns;
    return 0;
}

int hve_is_sliced(const uint8_t *blob, size_t n)
{
    return n >= 4 && !memcmp(blob, HVS_MAGIC, 4);
}

/* --------------------------------------------------------------------------
 * stills
 */

static int still_encode_one(void *arg, int k)
{
    slice_job *j = (slice_job *)arg;
    int64_t y0, y1;
    slice_bounds(j->h, 1, j->nslices, k, &y0, &y1);
    return hve_image_encode(j->img + y0 * j->w * j->channels, y1 - y0, j->w,
                            j->channels, j->features, &j->parts[k]);
}

static int still_decode_one(void *arg, int k)
{
    slice_job *j = (slice_job *)arg;
    int64_t h, w;
    int ch;
    uint8_t *part = NULL;
    if (hve_image_decode(j->subs[k], j->sublen[k], &part, &h, &w, &ch) != 0)
        return -1;
    j->out_img[k] = part;
    j->out_h[k] = h;
    if (k == 0) {
        j->w = w;
        j->channels = ch;
    } else if (w != j->w || ch != j->channels) {
        hve_set_error("slice %d disagrees about width or channel count", k);
        free(part);
        j->out_img[k] = NULL;
        return -1;
    }
    return 0;
}

int hve_slice_image_encode(const uint8_t *img, int64_t h, int64_t w,
                           int channels, int64_t features, int nslices,
                           hve_buf *out)
{
    slice_job j;
    memset(&j, 0, sizeof(j));
    nslices = hve_slice_count(h, 1, nslices);
    hve_buf parts[HVE_MAX_SLICES];
    int status[HVE_MAX_SLICES];
    memset(parts, 0, sizeof(parts));
    j.nslices = nslices;
    j.parts = parts;
    j.status = status;
    j.img = img;
    j.h = h;
    j.w = w;
    j.channels = channels;
    j.features = features;
    j.run = still_encode_one;

    int rc = run_slices(&j, hve_threads_default());
    if (rc == 0)
        rc = write_wrapper(out, parts, nslices);
    for (int k = 0; k < nslices; k++)
        hve_buf_free(&parts[k]);
    return rc;
}

int hve_slice_image_decode(const uint8_t *blob, size_t n, uint8_t **img,
                           int64_t *h_out, int64_t *w_out, int *channels_out)
{
    slice_job j;
    memset(&j, 0, sizeof(j));
    const uint8_t *subs[HVE_MAX_SLICES];
    size_t sublen[HVE_MAX_SLICES];
    uint8_t *parts[HVE_MAX_SLICES] = {NULL};
    int64_t hs[HVE_MAX_SLICES] = {0};
    int status[HVE_MAX_SLICES];
    int nslices = 0;

    if (read_wrapper(blob, n, &nslices, subs, sublen) != 0)
        return -1;
    j.nslices = nslices;
    j.subs = subs;
    j.sublen = sublen;
    j.out_img = parts;
    j.out_h = hs;
    j.status = status;
    j.run = still_decode_one;
    /* slice 0 publishes the width and channel count; run it first so the
     * others have something to disagree with */
    if (still_decode_one(&j, 0) != 0)
        return -1;
    status[0] = 0;
    j.next = 1;
    int rc = 0;
    if (nslices > 1) {
        hve_mutex_init(&j.lock);
        j.next = 1;
        hve_thread tid[HVE_MAX_SLICES];
        int made = 0, nthreads = hve_threads_default();
        if (nthreads > nslices)
            nthreads = nslices;
        for (int i = 0; i < nthreads - 1; i++)
            if (hve_thread_start(&tid[made], slice_worker, &j) == 0)
                made++;
        slice_worker(&j);
        for (int i = 0; i < made; i++)
            hve_thread_join(tid[i]);
        hve_mutex_free(&j.lock);
        for (int k = 1; k < nslices; k++)
            if (status[k] != 0)
                rc = -1;
    }

    int64_t total = 0;
    for (int k = 0; k < nslices && rc == 0; k++)
        total += hs[k];
    uint8_t *full = rc == 0
        ? (uint8_t *)malloc((size_t)total * j.w * j.channels) : NULL;
    if (rc == 0 && !full) {
        hve_set_error("out of memory joining slices");
        rc = -1;
    }
    if (rc == 0) {
        int64_t at = 0;
        for (int k = 0; k < nslices; k++) {
            memcpy(full + at * j.w * j.channels, parts[k],
                   (size_t)hs[k] * j.w * j.channels);
            at += hs[k];
        }
        *img = full;
        *h_out = total;
        *w_out = j.w;
        *channels_out = j.channels;
    }
    for (int k = 0; k < nslices; k++)
        free(parts[k]);
    return rc;
}

/* --------------------------------------------------------------------------
 * video
 */

static int video_encode_one(void *arg, int k)
{
    slice_job *j = (slice_job *)arg;
    return hve_video_encode(j->bands + (size_t)k * j->nframes, j->nframes,
                            j->features, &j->parts[k], 0);
}

static int video_decode_one(void *arg, int k)
{
    slice_job *j = (slice_job *)arg;
    hve_frame *fs = NULL;
    int nf = 0;
    if (hve_video_decode(j->subs[k], j->sublen[k], &fs, &nf, 0) != 0)
        return -1;
    j->out_frames[k] = fs;
    j->out_nframes[k] = nf;
    return 0;
}

int hve_slice_video_encode(const hve_frame *frames, int nframes,
                           int64_t features, int nslices, hve_buf *out,
                           int verbose)
{
    if (nframes < 1) {
        hve_set_error("no frames");
        return -1;
    }
    const int nplanes = frames[0].nplanes;
    const int64_t luma_h = frames[0].p[0].h;
    int64_t sy = luma_h / frames[0].p[nplanes > 1 ? 1 : 0].h;
    if (sy < 1)
        sy = 1;
    nslices = hve_slice_count(luma_h, sy, nslices);

    hve_buf parts[HVE_MAX_SLICES];
    int status[HVE_MAX_SLICES];
    memset(parts, 0, sizeof(parts));
    hve_frame *bands = (hve_frame *)calloc((size_t)nslices * nframes,
                                           sizeof(hve_frame));
    if (!bands) {
        hve_set_error("out of memory splitting into slices");
        return -1;
    }

    int rc = 0;
    for (int k = 0; k < nslices && rc == 0; k++) {
        for (int f = 0; f < nframes && rc == 0; f++) {
            hve_frame *b = &bands[(size_t)k * nframes + f];
            int64_t hs[HVE_MAX_PLANES], ws[HVE_MAX_PLANES];
            for (int i = 0; i < nplanes; i++) {
                int64_t s = luma_h / frames[f].p[i].h;
                if (s < 1)
                    s = 1;
                int64_t y0, y1;
                slice_bounds(luma_h, sy, nslices, k, &y0, &y1);
                hs[i] = y1 / s - y0 / s;
                ws[i] = frames[f].p[i].w;
            }
            if (hve_frame_alloc(b, nplanes, hs, ws) != 0) {
                rc = -1;
                break;
            }
            for (int i = 0; i < nplanes; i++) {
                int64_t s = luma_h / frames[f].p[i].h;
                if (s < 1)
                    s = 1;
                int64_t y0, y1;
                slice_bounds(luma_h, sy, nslices, k, &y0, &y1);
                memcpy(b->p[i].data,
                       frames[f].p[i].data + (y0 / s) * frames[f].p[i].w,
                       (size_t)b->p[i].h * b->p[i].w);
            }
        }
    }

    /* Judge the pyramid by the whole frame, not by the strip. A 1920x135 slice
     * has fewer pixels than the CIF threshold but full horizontal resolution,
     * so leaving the area test alone switched the pyramid off and made the
     * search exhaustive - 289 positions instead of ~43. That was a cliff
     * between 4 and 8 slices where total CPU work tripled. */
    hve_set_pyramid_min((int64_t)luma_h * frames[0].p[0].w
                        > HVE_PYRAMID_MIN_PIXELS ? 0 : (int64_t)1 << 62);

    if (rc == 0) {
        slice_job j;
        memset(&j, 0, sizeof(j));
        j.nslices = nslices;
        j.parts = parts;
        j.status = status;
        j.bands = bands;
        j.nframes = nframes;
        j.features = features;
        j.run = video_encode_one;
        rc = run_slices(&j, hve_threads_default());
        if (rc == 0)
            rc = write_wrapper(out, parts, nslices);
    }
    hve_set_pyramid_min(-1);
    if (verbose)
        fprintf(stderr, "%d slices\n", nslices);

    for (size_t i = 0; i < (size_t)nslices * nframes; i++)
        hve_frame_free(&bands[i]);
    free(bands);
    for (int k = 0; k < nslices; k++)
        hve_buf_free(&parts[k]);
    return rc;
}

int hve_slice_video_decode(const uint8_t *blob, size_t n, hve_frame **frames_out,
                           int *nframes_out, int verbose)
{
    slice_job j;
    memset(&j, 0, sizeof(j));
    const uint8_t *subs[HVE_MAX_SLICES];
    size_t sublen[HVE_MAX_SLICES];
    hve_frame *parts[HVE_MAX_SLICES] = {NULL};
    int counts[HVE_MAX_SLICES] = {0};
    int status[HVE_MAX_SLICES];
    int nslices = 0;

    if (read_wrapper(blob, n, &nslices, subs, sublen) != 0)
        return -1;
    j.nslices = nslices;
    j.subs = subs;
    j.sublen = sublen;
    j.out_frames = parts;
    j.out_nframes = counts;
    j.status = status;
    j.run = video_decode_one;

    int rc = run_slices(&j, hve_threads_default());
    hve_frame *joined = NULL;
    int nframes = rc == 0 ? counts[0] : 0;
    for (int k = 1; k < nslices && rc == 0; k++)
        if (counts[k] != nframes) {
            hve_set_error("slice %d has %d frames, slice 0 has %d", k,
                          counts[k], nframes);
            rc = -1;
        }

    if (rc == 0) {
        const int nplanes = parts[0][0].nplanes;
        joined = (hve_frame *)calloc((size_t)nframes, sizeof(hve_frame));
        if (!joined) {
            hve_set_error("out of memory joining slices");
            rc = -1;
        }
        for (int f = 0; f < nframes && rc == 0; f++) {
            int64_t hs[HVE_MAX_PLANES] = {0}, ws[HVE_MAX_PLANES];
            for (int i = 0; i < nplanes; i++) {
                ws[i] = parts[0][f].p[i].w;
                for (int k = 0; k < nslices; k++)
                    hs[i] += parts[k][f].p[i].h;
            }
            if (hve_frame_alloc(&joined[f], nplanes, hs, ws) != 0) {
                rc = -1;
                break;
            }
            for (int i = 0; i < nplanes; i++) {
                int64_t at = 0;
                for (int k = 0; k < nslices; k++) {
                    memcpy(joined[f].p[i].data + at * ws[i],
                           parts[k][f].p[i].data,
                           (size_t)parts[k][f].p[i].h * ws[i]);
                    at += parts[k][f].p[i].h;
                }
            }
        }
    }
    if (verbose)
        fprintf(stderr, "%d slices\n", nslices);

    for (int k = 0; k < nslices; k++) {
        if (parts[k]) {
            for (int f = 0; f < counts[k]; f++)
                hve_frame_free(&parts[k][f]);
            free(parts[k]);
        }
    }
    if (rc != 0) {
        if (joined) {
            for (int f = 0; f < nframes; f++)
                hve_frame_free(&joined[f]);
            free(joined);
        }
        return -1;
    }
    *frames_out = joined;
    *nframes_out = nframes;
    return 0;
}
