/* The MED / LOCO-I predictor, shared by the motion search and the colour
 * transform decision so there is one definition of it in C.
 *
 * Edge behaviour matches transform.predict_plane: neighbours outside the image
 * read as zero, so row 0 predicts from the west neighbour, column 0 from the
 * north one, and the very first pixel from 128.
 */

#ifndef HVE_MED_H
#define HVE_MED_H

#include <stdint.h>

static inline int hve_med_pred(const uint8_t *plane, int64_t w, int64_t y,
                               int64_t x)
{
    int north, west, nwest, hi, lo, planar;
    if (y == 0)
        return x == 0 ? 128 : plane[x - 1];
    if (x == 0)
        return plane[(y - 1) * w];
    north = plane[(y - 1) * w + x];
    west = plane[y * w + x - 1];
    nwest = plane[(y - 1) * w + x - 1];
    hi = north > west ? north : west;
    lo = north < west ? north : west;
    if (nwest >= hi)
        return lo;
    if (nwest <= lo)
        return hi;
    planar = north + west - nwest;
    return planar < 0 ? 0 : (planar > 255 ? 255 : planar);
}

#endif /* HVE_MED_H */
