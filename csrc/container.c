/* The .hvi and .hvv containers, mirroring hve/image.py and hve/video.py.
 *
 * Both are a short header of magic plus LEB128 varints, then one range-coder
 * payload for the whole file. The interesting decisions are which colour
 * layout to use (stills) and which prediction mode each block gets (video);
 * everything else is bookkeeping.
 */

#include <stdlib.h>
#include <string.h>

#include "hve.h"
#include "hvefmt.h"
#include "hvemodel.h"
#include "model_constants.h"

static int64_t plane_kind(int i)
{
    return i < 3 ? i : 3;
}

/* --------------------------------------------------------------------------
 * stills
 */

/* Pick the colour transform that actually helps this image.
 *
 * The RCT is a large win on photographs, where the channels track each other,
 * but a large loss on synthetic images whose channels are unrelated. One
 * entropy estimate each is cheap, so measure instead of assuming.
 */
static int choose_planes(const uint8_t *img, int64_t h, int64_t w, int channels,
                         uint8_t **planes_out, int *nplanes_out, unsigned *flags)
{
    const int64_t n = h * w;
    *flags = 0;
    if (channels == 1) {
        uint8_t *p = (uint8_t *)malloc((size_t)n);
        if (!p)
            return -1;
        memcpy(p, img, (size_t)n);
        *planes_out = p;
        *nplanes_out = 1;
        return 0;
    }
    if (channels != 3 && channels != 4) {
        uint8_t *p = (uint8_t *)malloc((size_t)n * channels);
        if (!p)
            return -1;
        for (int c = 0; c < channels; c++)
            for (int64_t i = 0; i < n; i++)
                p[(int64_t)c * n + i] = img[i * channels + c];
        *planes_out = p;
        *nplanes_out = channels;
        return 0;
    }

    uint8_t *rgb = (uint8_t *)malloc((size_t)n * 3);
    uint8_t *plain = (uint8_t *)malloc((size_t)n * 3);
    uint8_t *trans = (uint8_t *)malloc((size_t)n * 3);
    if (!rgb || !plain || !trans) {
        free(rgb);
        free(plain);
        free(trans);
        return -1;
    }
    for (int64_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++) {
            rgb[i * 3 + c] = img[i * channels + c];
            plain[(int64_t)c * n + i] = img[i * channels + c];
        }
    hve_rct_forward(rgb, h, w, trans);
    free(rgb);

    int use_rct = hve_residual_entropy(trans, 3, h, w)
                  <= hve_residual_entropy(plain, 3, h, w);
    uint8_t *chosen = use_rct ? trans : plain;
    free(use_rct ? plain : trans);
    if (use_rct)
        *flags = HVE_FLAG_RCT;

    if (channels == 4) {
        uint8_t *all = (uint8_t *)realloc(chosen, (size_t)n * 4);
        if (!all) {
            free(chosen);
            return -1;
        }
        for (int64_t i = 0; i < n; i++)
            all[3 * n + i] = img[i * 4 + 3];
        chosen = all;
    }
    *planes_out = chosen;
    *nplanes_out = channels;
    return 0;
}

/* Code every plane of a still through the kernel. Shared by both directions. */
static int code_still(hve_coder *c, hve_bank *bank, int encode, uint8_t *planes,
                      int nplanes, int64_t h, int64_t w)
{
    for (int i = 0; i < nplanes; i++) {
        if (hve_code_plane(encode, planes + (int64_t)i * h * w, h, w,
                           c->data, c->out, &c->rc, &bank->m, plane_kind(i),
                           (i == 1 || i == 2) ? 1 : 0, i == 0 ? 1 : 0,
                           NULL) != 0) {
            hve_set_error("the pixel loop ran out of memory");
            return -1;
        }
    }
    return 0;
}

int hve_image_encode(const uint8_t *img, int64_t h, int64_t w, int channels,
                     hve_buf *out)
{
    uint8_t *planes = NULL;
    int nplanes = 0;
    unsigned flags = 0;
    hve_bank bank;
    hve_coder coder;

    if (channels < 1 || channels > HVE_MAX_PLANES) {
        hve_set_error("unsupported channel count %d", channels);
        return -1;
    }
    if (choose_planes(img, h, w, channels, &planes, &nplanes, &flags) != 0) {
        hve_set_error("out of memory preparing the colour planes");
        return -1;
    }
    if (hve_bank_init(&bank, h, w) != 0) {
        free(planes);
        return -1;
    }
    size_t capacity = (size_t)h * w * nplanes * 2 + 65536;
    if (hve_coder_encode_init(&coder, capacity) != 0) {
        free(planes);
        hve_bank_free(&bank);
        return -1;
    }

    int rc = code_still(&coder, &bank, 1, planes, nplanes, h, w);
    size_t payload = rc == 0 ? hve_coder_finish(&coder) : 0;

    if (rc == 0) {
        rc |= hve_buf_put(out, HVI_MAGIC, 4);
        rc |= hve_buf_varint(out, (uint64_t)w);
        rc |= hve_buf_varint(out, (uint64_t)h);
        rc |= hve_buf_u8(out, (unsigned)channels);
        rc |= hve_buf_u8(out, flags);
        rc |= hve_buf_varint(out, payload);
        rc |= hve_buf_put(out, coder.out, payload);
    }

    free(planes);
    hve_bank_free(&bank);
    hve_coder_free(&coder);
    return rc;
}

int hve_image_decode(const uint8_t *blob, size_t n, uint8_t **img,
                     int64_t *h_out, int64_t *w_out, int *channels_out)
{
    hve_rd r = {blob, n, 0, 0};
    const uint8_t *magic = hve_rd_raw(&r, 4);
    if (!magic || memcmp(magic, HVI_MAGIC, 4) != 0) {
        hve_set_error("not an .hvi stream");
        return -1;
    }
    int64_t w = (int64_t)hve_rd_varint(&r);
    int64_t h = (int64_t)hve_rd_varint(&r);
    int channels = (int)hve_rd_u8(&r);
    unsigned flags = hve_rd_u8(&r);
    size_t payload_len = (size_t)hve_rd_varint(&r);
    const uint8_t *payload = hve_rd_raw(&r, payload_len);
    if (r.error || !payload || h <= 0 || w <= 0 || channels < 1
        || channels > HVE_MAX_PLANES) {
        hve_set_error("corrupt .hvi header");
        return -1;
    }

    hve_bank bank;
    hve_coder coder;
    uint8_t *planes = (uint8_t *)calloc((size_t)h * w * channels, 1);
    if (!planes) {
        hve_set_error("out of memory for a %lldx%lld image",
                      (long long)w, (long long)h);
        return -1;
    }
    if (hve_bank_init(&bank, h, w) != 0) {
        free(planes);
        return -1;
    }
    hve_coder_decode_init(&coder, payload, payload_len);
    int rc = code_still(&coder, &bank, 0, planes, channels, h, w);
    hve_bank_free(&bank);
    if (rc != 0) {
        free(planes);
        return -1;
    }

    const int64_t np = h * w;
    uint8_t *outimg = (uint8_t *)malloc((size_t)np * channels);
    if (!outimg) {
        free(planes);
        hve_set_error("out of memory assembling the image");
        return -1;
    }
    if (channels == 1) {
        memcpy(outimg, planes, (size_t)np);
    } else if ((flags & HVE_FLAG_RCT) && channels >= 3) {
        uint8_t *rgb = (uint8_t *)malloc((size_t)np * 3);
        if (!rgb) {
            free(planes);
            free(outimg);
            hve_set_error("out of memory inverting the colour transform");
            return -1;
        }
        hve_rct_inverse(planes, h, w, rgb);
        for (int64_t i = 0; i < np; i++) {
            for (int c = 0; c < 3; c++)
                outimg[i * channels + c] = rgb[i * 3 + c];
            if (channels == 4)
                outimg[i * 4 + 3] = planes[3 * np + i];
        }
        free(rgb);
    } else {
        for (int64_t i = 0; i < np; i++)
            for (int c = 0; c < channels; c++)
                outimg[i * channels + c] = planes[(int64_t)c * np + i];
    }
    free(planes);
    *img = outimg;
    *h_out = h;
    *w_out = w;
    *channels_out = channels;
    return 0;
}

/* --------------------------------------------------------------------------
 * video
 */

/* Pick spatial vs temporal per block, charging a flat price for a vector. */
static int choose_modes(const uint8_t *cur, const uint8_t *ref, int64_t h,
                        int64_t w, int64_t nby, int64_t nbx, int64_t *modes,
                        int32_t *mvs, int nthreads)
{
    const int64_t nb = nby * nbx;
    int64_t *tcost = (int64_t *)malloc((size_t)nb * sizeof(int64_t));
    int64_t *scost = (int64_t *)malloc((size_t)nb * sizeof(int64_t));
    if (!tcost || !scost) {
        free(tcost);
        free(scost);
        hve_set_error("out of memory scoring block modes");
        return -1;
    }
    if (hve_motion_search(cur, ref, h, w, HVE_BLOCK, HVE_SEARCH,
                          HVE_HALF_PEL_BIAS, HVE_PYRAMID_MIN_PIXELS,
                          HVE_PYRAMID_LEVELS, HVE_REFINE_RADIUS, HVE_COST_BYTE,
                          mvs, tcost, nthreads) != 0) {
        free(tcost);
        free(scost);
        return -1;
    }
    hve_spatial_cost(cur, h, w, HVE_BLOCK, HVE_COST_BYTE, scost, nthreads);
    for (int64_t i = 0; i < nb; i++) {
        int moving = mvs[i * 2] != 0 || mvs[i * 2 + 1] != 0;
        int64_t t = tcost[i] + (moving ? HVE_MV_PENALTY : 0);
        modes[i] = t < scost[i] ? 1 : 0;
        if (!modes[i]) {
            mvs[i * 2] = 0;
            mvs[i * 2 + 1] = 0;
        }
    }
    free(tcost);
    free(scost);
    return 0;
}

/* One frame's planes, coded either intra or against `prev`. */
static int code_frame(hve_coder *c, hve_bank *bank, int encode, hve_frame *cur,
                      const hve_frame *prev, int64_t *modes, int64_t *mvs64,
                      uint8_t *phases, int64_t luma_h, int64_t luma_w)
{
    for (int i = 0; i < cur->nplanes; i++) {
        hve_plane *pl = &cur->p[i];
        hve_inter inter, *ip = NULL;
        if (prev) {
            int64_t sy = luma_h / pl->h;
            int64_t sx = luma_w / pl->w;
            if (sy < 1) sy = 1;
            if (sx < 1) sx = 1;
            hve_halfpel_planes(prev->p[i].data, pl->h, pl->w, phases);
            inter.on = 1;
            inter.modes = modes;
            inter.mvs = mvs64;
            inter.nby = hve_ceil_div(luma_h, HVE_BLOCK);
            inter.nbx = hve_ceil_div(luma_w, HVE_BLOCK);
            inter.bs_y = HVE_BLOCK / sy > 0 ? HVE_BLOCK / sy : 1;
            inter.bs_x = HVE_BLOCK / sx > 0 ? HVE_BLOCK / sx : 1;
            inter.mv_sy = sy;
            inter.mv_sx = sx;
            inter.ref = phases;
            ip = &inter;
        }
        if (hve_code_plane(encode, pl->data, pl->h, pl->w, c->data, c->out,
                           &c->rc, &bank->m, plane_kind(i),
                           (i == 1 || i == 2) ? 1 : 0, i == 0 ? 1 : 0, ip) != 0) {
            hve_set_error("the pixel loop ran out of memory");
            return -1;
        }
    }
    return 0;
}

static size_t frame_samples(const hve_frame *f)
{
    size_t n = 0;
    for (int i = 0; i < f->nplanes; i++)
        n += (size_t)f->p[i].h * f->p[i].w;
    return n;
}

int hve_video_encode(const hve_frame *frames, int nframes, hve_buf *out,
                     int verbose)
{
    if (nframes < 1) {
        hve_set_error("no frames");
        return -1;
    }
    const int nplanes = frames[0].nplanes;
    const int64_t luma_h = frames[0].p[0].h, luma_w = frames[0].p[0].w;
    const int64_t nby = hve_ceil_div(luma_h, HVE_BLOCK);
    const int64_t nbx = hve_ceil_div(luma_w, HVE_BLOCK);
    const int nthreads = hve_threads_default();
    int rc = -1;

    hve_bank bank;
    hve_coder coder;
    memset(&coder, 0, sizeof(coder));
    if (hve_bank_init(&bank, luma_h, luma_w) != 0)
        return -1;
    if (hve_bank_video(&bank) != 0) {
        hve_bank_free(&bank);
        return -1;
    }
    size_t capacity = frame_samples(&frames[0]) * (size_t)nframes * 2 + 65536;
    if (hve_coder_encode_init(&coder, capacity) != 0) {
        hve_bank_free(&bank);
        return -1;
    }

    int64_t *modes = (int64_t *)calloc((size_t)nby * nbx, sizeof(int64_t));
    int64_t *mvs64 = (int64_t *)calloc((size_t)nby * nbx * 2, sizeof(int64_t));
    int32_t *mvs32 = (int32_t *)calloc((size_t)nby * nbx * 2, sizeof(int32_t));
    uint8_t *phases = (uint8_t *)malloc((size_t)luma_h * luma_w * 4);
    /* One scratch frame, so the encoder can hand the kernel a writable copy and
     * still keep the original for the next frame's reference. */
    hve_frame work;
    memset(&work, 0, sizeof(work));
    if (!modes || !mvs64 || !mvs32 || !phases) {
        hve_set_error("out of memory setting up the video encoder");
        goto done;
    }
    {
        int64_t hs[HVE_MAX_PLANES], ws[HVE_MAX_PLANES];
        for (int i = 0; i < nplanes; i++) {
            hs[i] = frames[0].p[i].h;
            ws[i] = frames[0].p[i].w;
        }
        if (hve_frame_alloc(&work, nplanes, hs, ws) != 0)
            goto done;
    }

    for (int fi = 0; fi < nframes; fi++) {
        const hve_frame *f = &frames[fi];
        if (f->nplanes != nplanes) {
            hve_set_error("frame %d has %d planes, expected %d", fi,
                          f->nplanes, nplanes);
            goto done;
        }
        for (int i = 0; i < nplanes; i++) {
            if (f->p[i].h != work.p[i].h || f->p[i].w != work.p[i].w) {
                hve_set_error("frame %d plane %d changed size", fi, i);
                goto done;
            }
            memcpy(work.p[i].data, f->p[i].data,
                   (size_t)f->p[i].h * f->p[i].w);
        }
        const hve_frame *prev = fi ? &frames[fi - 1] : NULL;
        if (prev) {
            if (choose_modes(f->p[0].data, prev->p[0].data, luma_h, luma_w,
                             nby, nbx, modes, mvs32, nthreads) != 0)
                goto done;
            for (int64_t i = 0; i < nby * nbx * 2; i++)
                mvs64[i] = mvs32[i];
            hve_code_block_info(1, &coder.rc, coder.data, coder.out,
                                bank.mode_p, bank.mv_zero, bank.mv_sign,
                                bank.mv_mag, modes, mvs64, nby, nbx,
                                HVE_MV_MAX);
        }
        if (code_frame(&coder, &bank, 1, &work, prev, modes, mvs64, phases,
                       luma_h, luma_w) != 0)
            goto done;
        if (verbose)
            fprintf(stderr, "\rframe %d/%d", fi + 1, nframes);
    }
    if (verbose)
        fprintf(stderr, "\n");

    {
        size_t payload = hve_coder_finish(&coder);
        int e = 0;
        e |= hve_buf_put(out, HVV_MAGIC, 4);
        e |= hve_buf_varint(out, (uint64_t)nframes);
        e |= hve_buf_u8(out, (unsigned)nplanes);
        e |= hve_buf_u8(out, 0);                    /* flags: planar input */
        e |= hve_buf_u8(out, HVE_BLOCK);
        for (int i = 0; i < nplanes; i++) {
            e |= hve_buf_varint(out, (uint64_t)frames[0].p[i].w);
            e |= hve_buf_varint(out, (uint64_t)frames[0].p[i].h);
        }
        e |= hve_buf_varint(out, payload);
        e |= hve_buf_put(out, coder.out, payload);
        rc = e;
    }

done:
    hve_frame_free(&work);
    free(modes);
    free(mvs64);
    free(mvs32);
    free(phases);
    hve_bank_free(&bank);
    hve_coder_free(&coder);
    return rc;
}

int hve_video_decode(const uint8_t *blob, size_t n, hve_frame **frames_out,
                     int *nframes_out, int verbose)
{
    hve_rd r = {blob, n, 0, 0};
    const uint8_t *magic = hve_rd_raw(&r, 4);
    if (!magic || memcmp(magic, HVV_MAGIC, 4) != 0) {
        hve_set_error("not an .hvv stream");
        return -1;
    }
    int nframes = (int)hve_rd_varint(&r);
    int nplanes = (int)hve_rd_u8(&r);
    (void)hve_rd_u8(&r);                            /* flags */
    int block = (int)hve_rd_u8(&r);
    int64_t hs[HVE_MAX_PLANES], ws[HVE_MAX_PLANES];
    if (r.error || nframes < 1 || nplanes < 1 || nplanes > HVE_MAX_PLANES) {
        hve_set_error("corrupt .hvv header");
        return -1;
    }
    for (int i = 0; i < nplanes; i++) {
        ws[i] = (int64_t)hve_rd_varint(&r);
        hs[i] = (int64_t)hve_rd_varint(&r);
    }
    size_t payload_len = (size_t)hve_rd_varint(&r);
    const uint8_t *payload = hve_rd_raw(&r, payload_len);
    if (r.error || !payload || block != HVE_BLOCK) {
        hve_set_error(block != HVE_BLOCK && !r.error
                      ? "stream uses block size %d, this build only does %d"
                      : "corrupt .hvv header", block, HVE_BLOCK);
        return -1;
    }

    const int64_t luma_h = hs[0], luma_w = ws[0];
    const int64_t nby = hve_ceil_div(luma_h, HVE_BLOCK);
    const int64_t nbx = hve_ceil_div(luma_w, HVE_BLOCK);
    hve_frame *frames = (hve_frame *)calloc((size_t)nframes, sizeof(hve_frame));
    int64_t *modes = (int64_t *)calloc((size_t)nby * nbx, sizeof(int64_t));
    int64_t *mvs64 = (int64_t *)calloc((size_t)nby * nbx * 2, sizeof(int64_t));
    uint8_t *phases = (uint8_t *)malloc((size_t)luma_h * luma_w * 4);
    hve_bank bank;
    hve_coder coder;
    int rc = -1;
    int made = 0;

    if (hve_bank_init(&bank, luma_h, luma_w) != 0)
        goto done_nobank;
    if (hve_bank_video(&bank) != 0)
        goto done;
    if (!frames || !modes || !mvs64 || !phases) {
        hve_set_error("out of memory setting up the video decoder");
        goto done;
    }
    hve_coder_decode_init(&coder, payload, payload_len);

    for (int fi = 0; fi < nframes; fi++) {
        if (hve_frame_alloc(&frames[fi], nplanes, hs, ws) != 0)
            goto done;
        made = fi + 1;
        const hve_frame *prev = fi ? &frames[fi - 1] : NULL;
        if (prev) {
            memset(modes, 0, (size_t)nby * nbx * sizeof(int64_t));
            memset(mvs64, 0, (size_t)nby * nbx * 2 * sizeof(int64_t));
            hve_code_block_info(0, &coder.rc, coder.data, coder.out,
                                bank.mode_p, bank.mv_zero, bank.mv_sign,
                                bank.mv_mag, modes, mvs64, nby, nbx,
                                HVE_MV_MAX);
        }
        if (code_frame(&coder, &bank, 0, &frames[fi], prev, modes, mvs64,
                       phases, luma_h, luma_w) != 0)
            goto done;
        if (verbose)
            fprintf(stderr, "\rframe %d/%d", fi + 1, nframes);
    }
    if (verbose)
        fprintf(stderr, "\n");
    *frames_out = frames;
    *nframes_out = nframes;
    frames = NULL;
    rc = 0;

done:
    hve_bank_free(&bank);
done_nobank:
    if (frames) {
        for (int i = 0; i < made; i++)
            hve_frame_free(&frames[i]);
        free(frames);
    }
    free(modes);
    free(mvs64);
    free(phases);
    return rc;
}
