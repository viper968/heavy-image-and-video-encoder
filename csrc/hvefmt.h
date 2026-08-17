/* Containers, transforms and file I/O for the standalone `hve` binary.
 *
 * csrc/hve.h covers the pixel loop and the motion search — the parts that must
 * be byte-identical to hve/model.py. This header covers everything wrapped
 * around them: the .hvi and .hvv containers, the reversible colour transform,
 * y4m and PNG.
 *
 * These pieces exist in Python too (hve/image.py, hve/video.py, hve/transform.py,
 * hve/y4m.py). That duplication is deliberate and much cheaper than the kernel's
 * was: a varint header and a colour transform are small and totally specified,
 * where the pixel loop is a 600-line adaptive model. Both directions are pinned
 * by tests/test_cli_binary.py, which makes the binary and the Python package
 * read each other's files.
 */

#ifndef HVE_FMT_H
#define HVE_FMT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * a growable byte buffer, and LEB128 varints (mirrors hve/bitio.py)
 */

/* Ceiling division for positive operands.
 *
 * Python spells this -(-a // b), and that idiom transcribes into C looking
 * correct and being wrong: `//` floors, `/` truncates toward zero, so
 * -(-1080 / 16) is 67 in C and 68 in Python. It only misbehaves when the
 * division is inexact, which is why a 352x288 clip never showed it and a
 * 1920x1080 one corrupted the heap. Use this instead.
 */
static inline int64_t hve_ceil_div(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

typedef struct {
    uint8_t *data;
    size_t len, cap;
} hve_buf;

void hve_buf_free(hve_buf *b);
int hve_buf_put(hve_buf *b, const void *src, size_t n);
int hve_buf_u8(hve_buf *b, unsigned v);
int hve_buf_varint(hve_buf *b, uint64_t v);

typedef struct {
    const uint8_t *data;
    size_t len, pos;
    int error;
} hve_rd;

unsigned hve_rd_u8(hve_rd *r);
uint64_t hve_rd_varint(hve_rd *r);
const uint8_t *hve_rd_raw(hve_rd *r, size_t n);

/* --------------------------------------------------------------------------
 * transforms (mirrors hve/transform.py)
 */

/* rgb is (h, w, 3) interleaved; out is 3 planes of h*w. */
void hve_rct_forward(const uint8_t *rgb, int64_t h, int64_t w, uint8_t *out);
void hve_rct_inverse(const uint8_t *planes, int64_t h, int64_t w, uint8_t *rgb);

/* Zeroth-order entropy of the MED residuals, in bytes — the cheap proxy that
 * decides whether the colour transform is worth applying. */
double hve_residual_entropy(const uint8_t *planes, int nplanes,
                            int64_t h, int64_t w);

/* --------------------------------------------------------------------------
 * containers
 */

#define HVE_MAX_PLANES 4

typedef struct {
    int64_t h, w;
    uint8_t *data;              /* h*w bytes, owned */
} hve_plane;

typedef struct {
    int nplanes;
    hve_plane p[HVE_MAX_PLANES];
} hve_frame;

void hve_frame_free(hve_frame *f);
int hve_frame_alloc(hve_frame *f, int nplanes, const int64_t *hs,
                    const int64_t *ws);

/* Stills. `img` is (h, w, channels) interleaved, channels in 1..4. */
int hve_image_encode(const uint8_t *img, int64_t h, int64_t w, int channels,
                     hve_buf *out);
int hve_image_decode(const uint8_t *blob, size_t n, uint8_t **img,
                     int64_t *h, int64_t *w, int *channels);

/* Video. Frames are planar (Y, U, V) at their native subsampled sizes. */
int hve_video_encode(const hve_frame *frames, int nframes, hve_buf *out,
                     int verbose);
int hve_video_decode(const uint8_t *blob, size_t n, hve_frame **frames,
                     int *nframes, int verbose);

/* --------------------------------------------------------------------------
 * file I/O
 */

int hve_png_read(const char *path, uint8_t **img, int64_t *h, int64_t *w,
                 int *channels);
int hve_png_write(const char *path, const uint8_t *img, int64_t h, int64_t w,
                  int channels);

int hve_y4m_read(const char *path, hve_frame **frames, int *nframes,
                 int limit, char *rate, size_t rate_n);
int hve_y4m_write(const char *path, const hve_frame *frames, int nframes,
                  const char *rate);

int hve_read_file(const char *path, uint8_t **data, size_t *len);
int hve_write_file(const char *path, const uint8_t *data, size_t len);

const char *hve_last_error(void);
void hve_set_error(const char *fmt, ...);

#endif /* HVE_FMT_H */
