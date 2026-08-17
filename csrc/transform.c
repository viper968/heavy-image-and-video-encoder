/* Reversible colour transform and the entropy proxy that decides whether to
 * use it. Mirrors hve/transform.py and image._residual_entropy.
 *
 * Everything is modular 8-bit: wrapping 255 -> 0 costs a little entropy on the
 * rare pixel that does it, but keeps every plane exactly 8 bits wide, which is
 * what makes the pipeline reversible without ever widening a sample.
 */

#include <math.h>

#include "hvefmt.h"
#include "med.h"

/* Map an unsigned modular byte onto [-128, 127]. */
static inline int centre(int v)
{
    return ((v + 128) & 255) - 128;
}

void hve_rct_forward(const uint8_t *rgb, int64_t h, int64_t w, uint8_t *out)
{
    const int64_t n = h * w;
    uint8_t *y = out, *cbp = out + n, *crp = out + 2 * n;
    for (int64_t i = 0; i < n; i++) {
        int r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
        int cb = (b - g) & 255;
        int cr = (r - g) & 255;
        /* Arithmetic shift: the sum can be negative and must floor, exactly as
         * numpy's >> does. csrc/hve.h static-asserts that this holds. */
        y[i] = (uint8_t)((g + ((centre(cb) + centre(cr)) >> 2)) & 255);
        /* Chroma is stored biased by +128: unbiased, B-G sits at zero and so
         * straddles the 255/0 wrap, which corrupts every gradient the model
         * computes from it. */
        cbp[i] = (uint8_t)((cb + 128) & 255);
        crp[i] = (uint8_t)((cr + 128) & 255);
    }
}

void hve_rct_inverse(const uint8_t *planes, int64_t h, int64_t w, uint8_t *rgb)
{
    const int64_t n = h * w;
    const uint8_t *y = planes, *cbp = planes + n, *crp = planes + 2 * n;
    for (int64_t i = 0; i < n; i++) {
        int cb = (cbp[i] - 128) & 255;
        int cr = (crp[i] - 128) & 255;
        int g = (y[i] - ((centre(cb) + centre(cr)) >> 2)) & 255;
        rgb[i * 3] = (uint8_t)((g + cr) & 255);
        rgb[i * 3 + 1] = (uint8_t)g;
        rgb[i * 3 + 2] = (uint8_t)((g + cb) & 255);
    }
}

/* Zeroth-order entropy of the MED residuals, in bytes.
 *
 * Only ever used to compare two candidate colour layouts, so absolute accuracy
 * does not matter and the ordering does. The two candidates are never close:
 * across the 24 Kodak images the narrowest margin is 17%, so a last-ulp
 * difference between this and numpy's version cannot flip the decision.
 */
double hve_residual_entropy(const uint8_t *planes, int nplanes,
                            int64_t h, int64_t w)
{
    const int64_t n = h * w;
    double total = 0.0;
    for (int k = 0; k < nplanes; k++) {
        const uint8_t *p = planes + (int64_t)k * n;
        double hist[256] = {0.0};
        for (int64_t yy = 0; yy < h; yy++) {
            for (int64_t xx = 0; xx < w; xx++) {
                int resid = (p[yy * w + xx] - hve_med_pred(p, w, yy, xx)) & 255;
                int d = centre(resid);
                int sym = d >= 0 ? (d << 1) : ((-d << 1) - 1);
                hist[sym & 255] += 1.0;
            }
        }
        double count = (double)n;
        for (int s = 0; s < 256; s++)
            if (hist[s] > 0.0)
                total += -(hist[s] * log2(hist[s] / count));
    }
    return total / 8.0;
}
