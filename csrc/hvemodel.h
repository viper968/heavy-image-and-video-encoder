/* The model bank plus the buffers it owns, and the range coder wrapper.
 *
 * hve_model (csrc/hve.h) is a plain view of pointers, because Python owns that
 * memory when the kernel is called through hve/native.py. The binary has no
 * Python, so this is the owning version.
 */

#ifndef HVE_MODEL_H
#define HVE_MODEL_H

#include "hve.h"
#include "hvefmt.h"

typedef struct {
    hve_model m;
    void *owned[20];            /* everything malloc'd for `m`, for one free */
    int32_t *match_table;
    uint8_t *flat;
    uint8_t *errmap;
    /* video only */
    int64_t *mode_p, *mv_zero, *mv_sign, *mv_mag;
} hve_bank;

int hve_bank_init(hve_bank *m, int64_t luma_h, int64_t luma_w);
int hve_bank_video(hve_bank *m);
void hve_bank_free(hve_bank *m);

/* Range-coder state plus its output buffer, mirroring hve/native.py's Coder. */
typedef struct {
    hve_rc rc;
    uint8_t *out;               /* encoding: owned output buffer */
    size_t cap;
    const uint8_t *data;        /* decoding: borrowed payload */
} hve_coder;

int hve_coder_encode_init(hve_coder *c, size_t capacity);
void hve_coder_decode_init(hve_coder *c, const uint8_t *payload, size_t n);
size_t hve_coder_finish(hve_coder *c);
void hve_coder_free(hve_coder *c);

#endif /* HVE_MODEL_H */
