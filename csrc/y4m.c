/* Minimal YUV4MPEG2 reader and writer, mirroring hve/y4m.py.
 *
 * Only the tags that change how the bytes are laid out are honoured: width,
 * height and colour space.
 *
 * The frame rate is parsed and can be handed back to the writer, but nothing
 * currently carries it between the two: the container has no field for it, so
 * `hve encode` reads the `F` tag into a local and drops it, and every decode
 * writes F25:1. Pixels round-trip exactly; the playback rate does not. Fixing
 * that needs a header field and therefore a format bump - see the note in
 * docs/HANDOFF.md.
 */

#include <stdlib.h>
#include <string.h>

#include "hvefmt.h"

static int subsampling(const char *cs, int *sx, int *sy)
{
    /* "420p10" and friends are 10-bit and unsupported; check the base tag. */
    if (!strncmp(cs, "420", 3) && (cs[3] == '\0' || !strncmp(cs + 3, "jpeg", 4)
                                   || !strncmp(cs + 3, "mpeg2", 5)
                                   || !strncmp(cs + 3, "paldv", 5))) {
        *sx = 2;
        *sy = 2;
        return 0;
    }
    if (!strcmp(cs, "422")) {
        *sx = 2;
        *sy = 1;
        return 0;
    }
    if (!strcmp(cs, "444")) {
        *sx = 1;
        *sy = 1;
        return 0;
    }
    hve_set_error("unsupported y4m colour space %s (this codec is 8-bit "
                  "420/422/444 only)", cs);
    return -1;
}

static int read_line(FILE *fh, char *buf, size_t n)
{
    size_t i = 0;
    for (;;) {
        int c = fgetc(fh);
        if (c == EOF)
            return i ? (int)i : -1;
        if (c == '\n') {
            buf[i < n ? i : n - 1] = '\0';
            return (int)i;
        }
        if (i + 1 < n)
            buf[i] = (char)c;
        i++;
    }
}

int hve_y4m_read(const char *path, hve_frame **frames_out, int *nframes_out,
                 int limit, char *rate, size_t rate_n)
{
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        hve_set_error("cannot open %s", path);
        return -1;
    }
    char header[1024];
    if (read_line(fh, header, sizeof(header)) < 0
        || strncmp(header, "YUV4MPEG2", 9) != 0) {
        hve_set_error("%s is not a y4m file", path);
        fclose(fh);
        return -1;
    }

    int width = 0, height = 0, sx = 2, sy = 2;
    char cs[64] = "420jpeg";
    if (rate && rate_n)
        snprintf(rate, rate_n, "25:1");
    for (char *tok = strtok(header + 9, " \t"); tok; tok = strtok(NULL, " \t")) {
        if (tok[0] == 'W')
            width = atoi(tok + 1);
        else if (tok[0] == 'H')
            height = atoi(tok + 1);
        else if (tok[0] == 'C')
            snprintf(cs, sizeof(cs), "%s", tok + 1);
        else if (tok[0] == 'F' && rate && rate_n)
            snprintf(rate, rate_n, "%s", tok + 1);
    }
    if (width <= 0 || height <= 0 || subsampling(cs, &sx, &sy) != 0) {
        if (width <= 0 || height <= 0)
            hve_set_error("y4m header has no usable size");
        fclose(fh);
        return -1;
    }

    const int64_t cw = hve_ceil_div(width, sx), ch = hve_ceil_div(height, sy);
    const int64_t hs[3] = {height, ch, ch};
    const int64_t ws[3] = {width, cw, cw};

    hve_frame *frames = NULL;
    int count = 0, cap = 0;
    for (;;) {
        if (limit > 0 && count >= limit)
            break;
        char marker[64];
        if (read_line(fh, marker, sizeof(marker)) < 0)
            break;
        if (strncmp(marker, "FRAME", 5) != 0) {
            hve_set_error("bad y4m frame marker at frame %d", count);
            goto fail;
        }
        if (count == cap) {
            int newcap = cap ? cap * 2 : 16;
            hve_frame *p = (hve_frame *)realloc(frames,
                                                (size_t)newcap * sizeof(*p));
            if (!p) {
                hve_set_error("out of memory holding %d frames", newcap);
                goto fail;
            }
            frames = p;
            cap = newcap;
        }
        if (hve_frame_alloc(&frames[count], 3, hs, ws) != 0)
            goto fail;
        int short_read = 0;
        for (int i = 0; i < 3 && !short_read; i++) {
            size_t need = (size_t)hs[i] * ws[i];
            if (fread(frames[count].p[i].data, 1, need, fh) != need)
                short_read = 1;
        }
        if (short_read) {                   /* truncated file: stop cleanly */
            hve_frame_free(&frames[count]);
            break;
        }
        count++;
    }
    fclose(fh);
    if (count == 0) {
        hve_set_error("%s contains no complete frames", path);
        free(frames);
        return -1;
    }
    *frames_out = frames;
    *nframes_out = count;
    return 0;

fail:
    for (int i = 0; i < count; i++)
        hve_frame_free(&frames[i]);
    free(frames);
    fclose(fh);
    return -1;
}

int hve_y4m_write(const char *path, const hve_frame *frames, int nframes,
                  const char *rate)
{
    if (nframes < 1 || frames[0].nplanes != 3) {
        hve_set_error("y4m output needs three planes per frame");
        return -1;
    }
    const int64_t w = frames[0].p[0].w, h = frames[0].p[0].h;
    const int64_t cw = frames[0].p[1].w, chh = frames[0].p[1].h;
    const char *cs = "420jpeg";
    if (cw == w && chh == h)
        cs = "444";
    else if (cw == hve_ceil_div(w, 2) && chh == h)
        cs = "422";
    else if (cw != hve_ceil_div(w, 2) || chh != hve_ceil_div(h, 2)) {
        hve_set_error("chroma planes are %lldx%lld against a %lldx%lld luma, "
                      "which is not a y4m subsampling",
                      (long long)cw, (long long)chh, (long long)w, (long long)h);
        return -1;
    }

    FILE *fh = fopen(path, "wb");
    if (!fh) {
        hve_set_error("cannot create %s", path);
        return -1;
    }
    fprintf(fh, "YUV4MPEG2 W%lld H%lld F%s Ip A1:1 C%s\n", (long long)w,
            (long long)h, rate && *rate ? rate : "25:1", cs);
    for (int f = 0; f < nframes; f++) {
        fputs("FRAME\n", fh);
        for (int i = 0; i < 3; i++) {
            size_t n = (size_t)frames[f].p[i].h * frames[f].p[i].w;
            if (fwrite(frames[f].p[i].data, 1, n, fh) != n) {
                hve_set_error("short write on %s", path);
                fclose(fh);
                return -1;
            }
        }
    }
    if (fclose(fh) != 0) {
        hve_set_error("error closing %s", path);
        return -1;
    }
    return 0;
}
