/* The `hve` command line tool: encode, decode, info.
 *
 * Same surface as `python -m hve`, and the files are interchangeable in both
 * directions — tests/test_cli_binary.py checks that by round-tripping each
 * through the other.
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hve.h"
#include "hvefmt.h"
#include "hvemodel.h"
#include "model_constants.h"

#ifndef HVE_VERSION
#define HVE_VERSION "0.1.0"
#endif

static double now(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}

static const char *extension(const char *path)
{
    const char *dot = strrchr(path, '.');
    return dot ? dot : "";
}

static int ieq(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        int ca = *a >= 'A' && *a <= 'Z' ? *a + 32 : *a;
        int cb = *b >= 'A' && *b <= 'Z' ? *b + 32 : *b;
        if (ca != cb)
            return 0;
    }
    return *a == *b;
}

static void usage(FILE *fh)
{
    fprintf(fh,
        "hve " HVE_VERSION " - lossless image and video compression\n"
        "\n"
        "usage:\n"
        "  hve encode <in.png|in.y4m> <out.hvi|out.hvv> [--frames N] [-v]\n"
        "  hve decode <in.hvi|in.hvv> <out.png|out.y4m> [-v]\n"
        "  hve info   <in.hvi|in.hvv>\n"
        "\n"
        "The output is not viewable in any normal viewer: ship the blob plus\n"
        "this program and the other end gets the exact original bytes back.\n"
        "\n"
        "options:\n"
        "  --frames N   stop after N video frames\n"
        "  --threads N  motion-search threads (default: one per core)\n"
        "  --preset P   max (default, best ratio) or fast (1.8-2.5x quicker\n"
        "               on 1080p video for 1-4%% more bytes). Recorded in the\n"
        "               file, so decoding needs no flag.\n"
        "  --features N explicit model-stage bitmask, for experiments\n"
        "  -v           per-frame progress on stderr\n");
}

static int fail(void)
{
    fprintf(stderr, "error: %s\n", hve_last_error());
    return 1;
}

/* --------------------------------------------------------------------------
 * commands
 */

/* Presets. "max" is the full model and the best ratio on photographs; "fast"
 * drops the three stages that a 1080p video ablation showed are not merely
 * expensive there but actively harmful. Which one wins is content-dependent,
 * which is exactly why the choice travels in the header. */
#define PRESET_MAX  HVE_FEAT_ALL
#define PRESET_FAST (HVE_FEAT_ALL & ~(HVE_FEAT_MATCH | HVE_FEAT_LMS \
                                      | HVE_FEAT_BLEND))

static const char *preset_name(int64_t f)
{
    if (f == PRESET_MAX)
        return "max";
    if (f == PRESET_FAST)
        return "fast";
    return "custom";
}

static int cmd_encode(const char *in, const char *out, int frames, int verbose,
                      int64_t features)
{
    hve_buf blob = {0};
    double t0 = now();
    size_t original;

    if (ieq(extension(in), ".y4m")) {
        hve_frame *fs = NULL;
        int n = 0;
        char rate[64] = "25:1";
        if (hve_y4m_read(in, &fs, &n, frames, rate, sizeof(rate)) != 0)
            return fail();
        original = 0;
        for (int i = 0; i < n; i++)
            for (int p = 0; p < fs[i].nplanes; p++)
                original += (size_t)fs[i].p[p].h * fs[i].p[p].w;
        int rc = hve_video_encode(fs, n, features, &blob, verbose);
        for (int i = 0; i < n; i++)
            hve_frame_free(&fs[i]);
        free(fs);
        if (rc != 0) {
            hve_buf_free(&blob);
            return fail();
        }
    } else {
        uint8_t *img = NULL;
        int64_t h = 0, w = 0;
        int channels = 0;
        if (hve_png_read(in, &img, &h, &w, &channels) != 0)
            return fail();
        original = (size_t)h * w * channels;
        int rc = hve_image_encode(img, h, w, channels, features, &blob);
        free(img);
        if (rc != 0) {
            hve_buf_free(&blob);
            return fail();
        }
    }

    if (hve_write_file(out, blob.data, blob.len) != 0) {
        hve_buf_free(&blob);
        return fail();
    }
    printf("%s -> %s  %zu -> %zu bytes (%.2fx, %.1fs, preset %s)\n", in, out,
           original, blob.len, (double)original / (double)blob.len, now() - t0,
           preset_name(features));
    hve_buf_free(&blob);
    return 0;
}

static int cmd_decode(const char *in, const char *out, int verbose)
{
    uint8_t *blob = NULL;
    size_t n = 0;
    double t0 = now();
    if (hve_read_file(in, &blob, &n) != 0)
        return fail();
    if (n < 4) {
        free(blob);
        hve_set_error("%s is too short to be an hve file", in);
        return fail();
    }

    int rc;
    if (!memcmp(blob, HVV_MAGIC, 4)) {
        hve_frame *frames = NULL;
        int nframes = 0;
        rc = hve_video_decode(blob, n, &frames, &nframes, verbose);
        if (rc == 0) {
            rc = hve_y4m_write(out, frames, nframes, "25:1");
            for (int i = 0; i < nframes; i++)
                hve_frame_free(&frames[i]);
            free(frames);
        }
    } else if (!memcmp(blob, HVI_MAGIC, 4)) {
        uint8_t *img = NULL;
        int64_t h = 0, w = 0;
        int channels = 0;
        rc = hve_image_decode(blob, n, &img, &h, &w, &channels);
        if (rc == 0) {
            rc = hve_png_write(out, img, h, w, channels);
            free(img);
        }
    } else {
        hve_set_error("unrecognised container in %s", in);
        rc = -1;
    }
    free(blob);
    if (rc != 0)
        return fail();
    printf("%s -> %s (%.1fs)\n", in, out, now() - t0);
    return 0;
}

static int cmd_info(const char *in)
{
    uint8_t *blob = NULL;
    size_t n = 0;
    if (hve_read_file(in, &blob, &n) != 0)
        return fail();
    hve_rd r = {blob, n, 0, 0};
    const uint8_t *magic = hve_rd_raw(&r, 4);
    if (!magic) {
        free(blob);
        hve_set_error("%s is too short", in);
        return fail();
    }
    if (!memcmp(magic, HVI_MAGIC, 4)) {
        int64_t w = (int64_t)hve_rd_varint(&r);
        int64_t h = (int64_t)hve_rd_varint(&r);
        unsigned channels = hve_rd_u8(&r), flags = hve_rd_u8(&r);
        int64_t feat = (int64_t)hve_rd_u8(&r);
        if (r.error || !h || !w) {
            free(blob);
            hve_set_error("corrupt .hvi header");
            return fail();
        }
        printf("hve image  %lldx%lld  %u channels  rct=%u  preset %s  %zu bytes"
               "  %.3f bpp\n", (long long)w, (long long)h, channels,
               flags & HVE_FLAG_RCT, preset_name(feat), n,
               (double)n * 8.0 / (double)(w * h));
    } else if (!memcmp(magic, HVV_MAGIC, 4)) {
        uint64_t nframes = hve_rd_varint(&r);
        unsigned nplanes = hve_rd_u8(&r), flags = hve_rd_u8(&r);
        unsigned block = hve_rd_u8(&r);
        int64_t feat = (int64_t)hve_rd_u8(&r);
        printf("hve video  %llu frames  %u planes (", (unsigned long long)nframes,
               nplanes);
        for (unsigned i = 0; i < nplanes && !r.error; i++) {
            uint64_t w = hve_rd_varint(&r), h = hve_rd_varint(&r);
            printf("%s%llux%llu", i ? ", " : "", (unsigned long long)w,
                   (unsigned long long)h);
        }
        printf(")  block=%u  rct=%u  preset %s  %zu bytes\n", block,
               flags & HVE_FLAG_RCT, preset_name(feat), n);
        if (r.error) {
            free(blob);
            hve_set_error("corrupt .hvv header");
            return fail();
        }
    } else {
        free(blob);
        hve_set_error("unrecognised container in %s", in);
        return fail();
    }
    free(blob);
    return 0;
}

/* --------------------------------------------------------------------------
 * argument handling
 */

int main(int argc, char **argv)
{
    const char *positional[2] = {NULL, NULL};
    int npos = 0, frames = 0, verbose = 0;
    /* Default to the full model: it is what every previous release produced,
     * and it wins on stills and on busy video alike. `fast` is the deliberate
     * trade. There is no `auto`: the decision is worth 1-4% on real content and
     * a probe encode costs far more than that is worth, and a centre-crop probe
     * was measured picking the wrong preset on the very clip that motivated the
     * feature. x264 makes the caller choose a preset; so does this. */
    int64_t features = PRESET_MAX;

    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")
        || !strcmp(argv[1], "help")) {
        usage(stdout);
        return 0;
    }
    if (!strcmp(argv[1], "--version")) {
        printf("hve %s\n", HVE_VERSION);
        return 0;
    }

    const char *cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
            verbose = 1;
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            hve_set_threads(atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--preset") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "max"))
                features = PRESET_MAX;
            else if (!strcmp(v, "fast"))
                features = PRESET_FAST;
            else {
                fprintf(stderr, "error: unknown preset %s (max, fast)\n", v);
                return 2;
            }
        } else if (!strcmp(argv[i], "--features") && i + 1 < argc) {
            /* Research knob: an explicit stage bitmask. The decoder reads it
             * back out of the header, so only encoding needs it. */
            features = strtoll(argv[++i], NULL, 0);
        } else if (argv[i][0] == '-' && argv[i][1]) {
            fprintf(stderr, "error: unknown option %s\n", argv[i]);
            return 2;
        } else if (npos < 2) {
            positional[npos++] = argv[i];
        } else {
            fprintf(stderr, "error: too many arguments\n");
            return 2;
        }
    }

    if (!strcmp(cmd, "encode")) {
        if (npos != 2) {
            usage(stderr);
            return 2;
        }
        return cmd_encode(positional[0], positional[1], frames, verbose,
                          features);
    }
    if (!strcmp(cmd, "decode")) {
        if (npos != 2) {
            usage(stderr);
            return 2;
        }
        return cmd_decode(positional[0], positional[1], verbose);
    }
    if (!strcmp(cmd, "info")) {
        if (npos != 1) {
            usage(stderr);
            return 2;
        }
        return cmd_info(positional[0]);
    }
    fprintf(stderr, "error: unknown command %s\n\n", cmd);
    usage(stderr);
    return 2;
}
