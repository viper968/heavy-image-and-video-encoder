/* Byte buffers, varints, error reporting and whole-file I/O. */

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "hvefmt.h"

static char g_error[512] = "";

const char *hve_last_error(void)
{
    return g_error[0] ? g_error : "unknown error";
}

void hve_set_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

/* --------------------------------------------------------------------------
 * buffer
 */

void hve_buf_free(hve_buf *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static int buf_reserve(hve_buf *b, size_t extra)
{
    size_t need = b->len + extra;
    if (need <= b->cap)
        return 0;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < need)
        cap += cap / 2 + 1;
    uint8_t *p = (uint8_t *)realloc(b->data, cap);
    if (!p) {
        hve_set_error("out of memory growing a buffer to %zu bytes", cap);
        return -1;
    }
    b->data = p;
    b->cap = cap;
    return 0;
}

int hve_buf_put(hve_buf *b, const void *src, size_t n)
{
    if (buf_reserve(b, n) != 0)
        return -1;
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

int hve_buf_u8(hve_buf *b, unsigned v)
{
    uint8_t byte = (uint8_t)(v & 0xFF);
    return hve_buf_put(b, &byte, 1);
}

int hve_buf_varint(hve_buf *b, uint64_t v)
{
    uint8_t tmp[10];
    size_t n = 0;
    for (;;) {
        uint8_t byte = (uint8_t)(v & 0x7F);
        v >>= 7;
        tmp[n++] = v ? (uint8_t)(byte | 0x80) : byte;
        if (!v)
            break;
    }
    return hve_buf_put(b, tmp, n);
}

/* --------------------------------------------------------------------------
 * reader. Every accessor is bounds-checked and sets `error` rather than
 * reading past the end: this parses untrusted files.
 */

unsigned hve_rd_u8(hve_rd *r)
{
    if (r->pos >= r->len) {
        r->error = 1;
        return 0;
    }
    return r->data[r->pos++];
}

uint64_t hve_rd_varint(hve_rd *r)
{
    uint64_t result = 0;
    int shift = 0;
    for (;;) {
        if (r->pos >= r->len || shift > 63) {
            r->error = 1;
            return 0;
        }
        uint8_t b = r->data[r->pos++];
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80))
            return result;
        shift += 7;
    }
}

const uint8_t *hve_rd_raw(hve_rd *r, size_t n)
{
    if (n > r->len - r->pos || r->pos > r->len) {
        r->error = 1;
        return NULL;
    }
    const uint8_t *p = r->data + r->pos;
    r->pos += n;
    return p;
}

/* --------------------------------------------------------------------------
 * frames
 */

void hve_frame_free(hve_frame *f)
{
    for (int i = 0; i < f->nplanes; i++) {
        free(f->p[i].data);
        f->p[i].data = NULL;
    }
    f->nplanes = 0;
}

int hve_frame_alloc(hve_frame *f, int nplanes, const int64_t *hs,
                    const int64_t *ws)
{
    memset(f, 0, sizeof(*f));
    if (nplanes < 1 || nplanes > HVE_MAX_PLANES) {
        hve_set_error("plane count %d out of range", nplanes);
        return -1;
    }
    f->nplanes = nplanes;
    for (int i = 0; i < nplanes; i++) {
        if (hs[i] <= 0 || ws[i] <= 0 || hs[i] > (1 << 20) || ws[i] > (1 << 20)) {
            hve_set_error("implausible plane size %lldx%lld",
                          (long long)ws[i], (long long)hs[i]);
            hve_frame_free(f);
            return -1;
        }
        f->p[i].h = hs[i];
        f->p[i].w = ws[i];
        f->p[i].data = (uint8_t *)calloc((size_t)hs[i] * ws[i], 1);
        if (!f->p[i].data) {
            hve_set_error("out of memory allocating a %lldx%lld plane",
                          (long long)ws[i], (long long)hs[i]);
            hve_frame_free(f);
            return -1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------
 * files
 */

int hve_read_file(const char *path, uint8_t **data, size_t *len)
{
    FILE *fh = fopen(path, "rb");
    if (!fh) {
        hve_set_error("cannot open %s", path);
        return -1;
    }
    if (fseek(fh, 0, SEEK_END) != 0) {
        hve_set_error("cannot seek %s", path);
        fclose(fh);
        return -1;
    }
    long n = ftell(fh);
    if (n < 0) {
        hve_set_error("cannot size %s", path);
        fclose(fh);
        return -1;
    }
    rewind(fh);
    uint8_t *buf = (uint8_t *)malloc((size_t)n + 1);
    if (!buf) {
        hve_set_error("out of memory reading %s (%ld bytes)", path, n);
        fclose(fh);
        return -1;
    }
    if (n > 0 && fread(buf, 1, (size_t)n, fh) != (size_t)n) {
        hve_set_error("short read on %s", path);
        free(buf);
        fclose(fh);
        return -1;
    }
    fclose(fh);
    *data = buf;
    *len = (size_t)n;
    return 0;
}

int hve_write_file(const char *path, const uint8_t *data, size_t len)
{
    FILE *fh = fopen(path, "wb");
    if (!fh) {
        hve_set_error("cannot create %s", path);
        return -1;
    }
    if (len && fwrite(data, 1, len, fh) != len) {
        hve_set_error("short write on %s", path);
        fclose(fh);
        return -1;
    }
    if (fclose(fh) != 0) {
        hve_set_error("error closing %s", path);
        return -1;
    }
    return 0;
}
