#pragma once

#include <cstdint>

namespace postproc {

/**
 * @brief Post-processing parameters.
 *
 * All values follow the same conventions as the original my-img implementation:
 *   - clarity:            0.0 .. 1.0
 *   - sharpen_amount:     0.0 .. 3.0
 *   - sharpen_radius:     1 .. 10
 *   - sharpen_threshold:  0 .. 255 (only sharpen pixels whose diff exceeds this)
 *   - smart_sharpen_strength: 0.0 .. 3.0
 *   - smart_sharpen_radius:   1 .. 10
 *   - edge_sharpen_amount:    0.0 .. 3.0
 *   - edge_sharpen_radius:    1 .. 10
 *   - edge_sharpen_threshold: 0.0 .. 1.0
 */
struct Params {
    float clarity = 0.0f;

    float sharpen_amount = 0.0f;
    int   sharpen_radius = 1;
    float sharpen_threshold = 0.0f;

    float smart_sharpen_strength = 0.0f;
    int   smart_sharpen_radius = 2;

    float edge_sharpen_amount = 0.0f;
    int   edge_sharpen_radius = 2;
    float edge_sharpen_threshold = 0.3f;
};

/**
 * @brief Apply clarity / sharpen / smart-sharpen / edge-sharpen to a raw image.
 *
 * Operates in place on a contiguous RGB or RGBA uint8 buffer.
 * Alpha channel (if present) is left untouched.
 *
 * @param data   pointer to pixel data, row-major, width * height * channels bytes
 * @param width  image width in pixels
 * @param height image height in pixels
 * @param channels 3 (RGB) or 4 (RGBA); other values are rejected
 * @param params processing parameters
 * @return true on success, false on invalid input
 */
bool apply(uint8_t* data, int width, int height, int channels, const Params& params);

} // namespace postproc
