/* PNG in and out, via the vendored lodepng.
 *
 * The Python CLI leans on Pillow and converts anything unusual to RGB or RGBA.
 * This does the same, with one difference that matters: it *refuses* 16-bit
 * input rather than silently narrowing it. The codec is 8-bit, and quietly
 * throwing away half of every sample before encoding, then reporting the round
 * trip as lossless, would be a lie. The playground already refuses these for
 * the same reason; the binary should not be laxer than the wrapper.
 */

#include <stdlib.h>

#include "hvefmt.h"
#include "third_party/lodepng.h"

int hve_png_read(const char *path, uint8_t **img, int64_t *h_out,
                 int64_t *w_out, int *channels_out)
{
    uint8_t *file = NULL;
    size_t filelen = 0;
    if (hve_read_file(path, &file, &filelen) != 0)
        return -1;

    LodePNGState state;
    lodepng_state_init(&state);
    unsigned w = 0, hh = 0;
    unsigned err = lodepng_inspect(&w, &hh, &state, file, filelen);
    if (err) {
        hve_set_error("cannot read %s: %s", path, lodepng_error_text(err));
        lodepng_state_cleanup(&state);
        free(file);
        return -1;
    }
    if (state.info_png.color.bitdepth > 8) {
        hve_set_error("%s is %u bits per sample; this codec is 8-bit only, and "
                      "converting would lose data before the encoder saw it",
                      path, state.info_png.color.bitdepth);
        lodepng_state_cleanup(&state);
        free(file);
        return -1;
    }
    /* Grey and palette both come back expanded; alpha decides 3 vs 4 channels,
     * matching what Pillow's mode would have given the Python CLI. */
    int has_alpha = state.info_png.color.colortype == LCT_RGBA
                    || state.info_png.color.colortype == LCT_GREY_ALPHA
                    || (state.info_png.color.colortype == LCT_PALETTE
                        && lodepng_has_palette_alpha(&state.info_png.color));
    int grey = state.info_png.color.colortype == LCT_GREY && !has_alpha;
    lodepng_state_cleanup(&state);

    int channels = grey ? 1 : (has_alpha ? 4 : 3);
    LodePNGColorType want = grey ? LCT_GREY : (has_alpha ? LCT_RGBA : LCT_RGB);
    uint8_t *pixels = NULL;
    err = lodepng_decode_memory(&pixels, &w, &hh, file, filelen, want, 8);
    free(file);
    if (err) {
        hve_set_error("cannot decode %s: %s", path, lodepng_error_text(err));
        return -1;
    }
    *img = pixels;
    *w_out = (int64_t)w;
    *h_out = (int64_t)hh;
    *channels_out = channels;
    return 0;
}

int hve_png_write(const char *path, const uint8_t *img, int64_t h, int64_t w,
                  int channels)
{
    LodePNGColorType type;
    switch (channels) {
    case 1: type = LCT_GREY; break;
    case 2: type = LCT_GREY_ALPHA; break;
    case 3: type = LCT_RGB; break;
    case 4: type = LCT_RGBA; break;
    default:
        hve_set_error("cannot write a %d-channel PNG", channels);
        return -1;
    }
    /* auto_convert off, deliberately.
     *
     * By default lodepng picks the smallest lossless representation, which
     * turned a uniform RGB image into a 1-bit palette PNG. No pixel changed,
     * but `hve decode` handing back a different colour type than it was given
     * is a surprise: it breaks the obvious `open(original) == open(restored)`
     * check, and it does not match what the Python CLI writes. Channels in,
     * same channels out.
     */
    LodePNGState state;
    lodepng_state_init(&state);
    state.info_raw.colortype = type;
    state.info_raw.bitdepth = 8;
    state.info_png.color.colortype = type;
    state.info_png.color.bitdepth = 8;
    state.encoder.auto_convert = 0;

    uint8_t *out = NULL;
    size_t n = 0;
    unsigned err = lodepng_encode(&out, &n, img, (unsigned)w, (unsigned)h,
                                  &state);
    lodepng_state_cleanup(&state);
    if (err) {
        hve_set_error("cannot encode %s: %s", path, lodepng_error_text(err));
        free(out);
        return -1;
    }
    int rc = hve_write_file(path, out, n);
    free(out);
    return rc;
}
