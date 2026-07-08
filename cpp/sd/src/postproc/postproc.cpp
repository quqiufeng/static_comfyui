#include "postproc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace postproc {

namespace {

inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline uint8_t f2u8(float v) {
    return static_cast<uint8_t>(clamp01(v) * 255.0f + 0.5f);
}

inline float u8f(uint8_t v) {
    return v / 255.0f;
}

// ---------------------------------------------------------------------------
// Image conversion helpers
// ---------------------------------------------------------------------------
struct FloatImage {
    int width = 0;
    int height = 0;
    int channels = 0; // 3 for RGB, 4 for RGBA
    std::vector<float> data; // interleaved R G B [A]

    bool init(uint8_t* src, int w, int h, int c) {
        if (!src || w <= 0 || h <= 0 || (c != 3 && c != 4)) return false;
        width = w;
        height = h;
        channels = c;
        data.resize(static_cast<size_t>(w) * h * c);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = u8f(src[i]);
        }
        return true;
    }

    void to_uint8(uint8_t* dst) const {
        for (size_t i = 0; i < data.size(); ++i) {
            dst[i] = f2u8(data[i]);
        }
    }

    float* pixel(int x, int y) {
        return &data[static_cast<size_t>(y) * width * channels + x * channels];
    }

    const float* pixel(int x, int y) const {
        return &data[static_cast<size_t>(y) * width * channels + x * channels];
    }
};

// ---------------------------------------------------------------------------
// Gaussian kernel
// ---------------------------------------------------------------------------
static std::vector<float> make_gaussian_kernel(int radius, float sigma) {
    const int size = 2 * radius + 1;
    std::vector<float> kernel(size);
    float sum = 0.0f;
    const float two_sigma2 = 2.0f * sigma * sigma;
    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i - radius);
        float v = std::exp(-(x * x) / two_sigma2);
        kernel[i] = v;
        sum += v;
    }
    for (float& v : kernel) v /= sum;
    return kernel;
}

// ---------------------------------------------------------------------------
// Separable gaussian blur per RGB channel (alpha untouched if c==4)
// ---------------------------------------------------------------------------
static void gaussian_blur(FloatImage& dst, const FloatImage& src, int radius) {
    radius = std::max(1, radius);
    const float sigma = radius / 2.0f;
    const std::vector<float> kernel = make_gaussian_kernel(radius, sigma);
    const int w = src.width;
    const int h = src.height;
    const int c = src.channels;

    dst = src; // copy dimensions and alpha if present

    std::vector<float> tmp(src.data.size());

    // Horizontal pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < 3; ++ch) { // only RGB
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int sx = std::clamp(x + k, 0, w - 1);
                    sum += src.pixel(sx, y)[ch] * kernel[k + radius];
                }
                tmp[static_cast<size_t>(y) * w * c + x * c + ch] = sum;
            }
            if (c == 4) {
                tmp[static_cast<size_t>(y) * w * c + x * c + 3] = src.pixel(x, y)[3];
            }
        }
    }

    // Vertical pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < 3; ++ch) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int sy = std::clamp(y + k, 0, h - 1);
                    sum += tmp[static_cast<size_t>(sy) * w * c + x * c + ch] * kernel[k + radius];
                }
                dst.pixel(x, y)[ch] = sum;
            }
            if (c == 4) {
                dst.pixel(x, y)[3] = src.pixel(x, y)[3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grayscale from RGB (luminance)
// ---------------------------------------------------------------------------
static std::vector<float> to_grayscale(const FloatImage& img) {
    std::vector<float> gray(static_cast<size_t>(img.width) * img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            const float* p = img.pixel(x, y);
            gray[static_cast<size_t>(y) * img.width + x] = p[0] * 0.299f + p[1] * 0.587f + p[2] * 0.114f;
        }
    }
    return gray;
}

// ---------------------------------------------------------------------------
// Sobel edge magnitude, returns a [0,1] normalized image-sized buffer
// ---------------------------------------------------------------------------
static std::vector<float> sobel_edges(const FloatImage& img, float* out_max = nullptr) {
    const int w = img.width;
    const int h = img.height;
    std::vector<float> gray = to_grayscale(img);
    std::vector<float> edge(static_cast<size_t>(w) * h, 0.0f);
    float max_val = 0.0f;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const float* row_m = &gray[static_cast<size_t>(y - 1) * w + x - 1];
            const float* row_c = &gray[static_cast<size_t>(y) * w + x - 1];
            const float* row_p = &gray[static_cast<size_t>(y + 1) * w + x - 1];

            float gx = (-1.0f * row_m[0] + 1.0f * row_m[2]) +
                       (-2.0f * row_c[0] + 2.0f * row_c[2]) +
                       (-1.0f * row_p[0] + 1.0f * row_p[2]);
            float gy = (-1.0f * row_m[0] - 2.0f * row_m[1] - 1.0f * row_m[2]) +
                       ( 1.0f * row_p[0] + 2.0f * row_p[1] + 1.0f * row_p[2]);
            float mag = std::sqrt(gx * gx + gy * gy);
            edge[static_cast<size_t>(y) * w + x] = mag;
            if (mag > max_val) max_val = mag;
        }
    }

    if (out_max) *out_max = max_val;
    return edge;
}

// ---------------------------------------------------------------------------
// Dilation with a square kernel of half-size radius
// ---------------------------------------------------------------------------
static void dilate(std::vector<float>& mask, int w, int h, int radius) {
    radius = std::max(1, radius);
    std::vector<float> src = mask;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float vmax = 0.0f;
            for (int ky = -radius; ky <= radius; ++ky) {
                int sy = std::clamp(y + ky, 0, h - 1);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int sx = std::clamp(x + kx, 0, w - 1);
                    vmax = std::max(vmax, src[static_cast<size_t>(sy) * w + sx]);
                }
            }
            mask[static_cast<size_t>(y) * w + x] = vmax;
        }
    }
}

// ---------------------------------------------------------------------------
// Clarity: large-radius unsharp mask
// ---------------------------------------------------------------------------
static void apply_clarity(FloatImage& img, float amount) {
    if (amount <= 0.0f) return;
    int radius = std::max(3, static_cast<int>(amount * 50.0f));
    if (radius % 2 == 0) ++radius;

    FloatImage blurred;
    gaussian_blur(blurred, img, radius);

    for (size_t i = 0; i < img.data.size(); ++i) {
        float detail = img.data[i] - blurred.data[i];
        img.data[i] = clamp01(img.data[i] + detail * amount * 0.5f);
    }
}

// ---------------------------------------------------------------------------
// USM sharpen
// ---------------------------------------------------------------------------
static void apply_sharpen(FloatImage& img, float amount, int radius, float threshold) {
    if (amount <= 0.0f) return;
    radius = std::max(1, std::min(radius, 10));
    if (radius % 2 == 0) ++radius;

    FloatImage blurred;
    gaussian_blur(blurred, img, radius);

    const float thr = threshold / 255.0f;
    std::vector<float> gray_diff;
    if (threshold > 0.0f) {
        gray_diff = to_grayscale(img);
        std::vector<float> gray_blur = to_grayscale(blurred);
        for (size_t i = 0; i < gray_diff.size(); ++i) {
            gray_diff[i] = std::abs(gray_diff[i] - gray_blur[i]);
        }
    }

    const int w = img.width;
    const int h = img.height;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t base = static_cast<size_t>(y) * w * img.channels + x * img.channels;
            bool apply = true;
            if (threshold > 0.0f) {
                apply = gray_diff[static_cast<size_t>(y) * w + x] > thr;
            }
            if (!apply) continue;
            for (int ch = 0; ch < 3; ++ch) {
                float detail = img.data[base + ch] - blurred.data[base + ch];
                img.data[base + ch] = clamp01(img.data[base + ch] + detail * amount);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Smart sharpen: edge-aware detail enhancement using Sobel weights
// ---------------------------------------------------------------------------
static void apply_smart_sharpen(FloatImage& img, float strength, int radius) {
    if (strength <= 0.0f) return;
    radius = std::max(1, std::min(radius, 10));

    FloatImage blurred;
    gaussian_blur(blurred, img, radius);

    float edge_max = 0.0f;
    std::vector<float> edges = sobel_edges(img, &edge_max);
    if (edge_max > 0.0f) {
        for (float& v : edges) v /= edge_max;
    }

    const int w = img.width;
    const int h = img.height;
    const int c = img.channels;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t base = static_cast<size_t>(y) * w * c + x * c;
            float weight = edges[static_cast<size_t>(y) * w + x] * strength;
            for (int ch = 0; ch < 3; ++ch) {
                float detail = img.data[base + ch] - blurred.data[base + ch];
                img.data[base + ch] = clamp01(img.data[base + ch] + detail * weight);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge sharpen: detect edges, dilate mask, apply USM only on edge regions
// ---------------------------------------------------------------------------
static void apply_edge_sharpen(FloatImage& img, float amount, int radius, float threshold) {
    if (amount <= 0.0f) return;
    radius = std::max(1, std::min(radius, 10));
    if (radius % 2 == 0) ++radius;

    float edge_max = 0.0f;
    std::vector<float> edges = sobel_edges(img, &edge_max);
    const float thresh = edge_max * threshold + 1e-6f;

    const int w = img.width;
    const int h = img.height;
    std::vector<float> mask(static_cast<size_t>(w) * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            mask[static_cast<size_t>(y) * w + x] = edges[static_cast<size_t>(y) * w + x] > thresh ? 1.0f : 0.0f;
        }
    }
    dilate(mask, w, h, radius);

    FloatImage blurred;
    gaussian_blur(blurred, img, radius);

    const int c = img.channels;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float m = mask[static_cast<size_t>(y) * w + x];
            if (m <= 0.0f) continue;
            size_t base = static_cast<size_t>(y) * w * c + x * c;
            for (int ch = 0; ch < 3; ++ch) {
                float blurred_pix = blurred.data[base + ch];
                float sharpened = img.data[base + ch] * (1.0f + amount) - blurred_pix * amount;
                img.data[base + ch] = clamp01(sharpened);
            }
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
bool apply(uint8_t* data, int width, int height, int channels, const Params& params) {
    if (!data || width <= 0 || height <= 0 || (channels != 3 && channels != 4)) {
        return false;
    }

    bool has_work = (params.clarity > 0.0f ||
                     params.sharpen_amount > 0.0f ||
                     params.smart_sharpen_strength > 0.0f ||
                     params.edge_sharpen_amount > 0.0f);
    if (!has_work) return true;

    FloatImage img;
    if (!img.init(data, width, height, channels)) return false;

    if (params.clarity > 0.0f) {
        apply_clarity(img, params.clarity);
    }
    if (params.sharpen_amount > 0.0f) {
        apply_sharpen(img, params.sharpen_amount, params.sharpen_radius, params.sharpen_threshold);
    }
    if (params.smart_sharpen_strength > 0.0f) {
        apply_smart_sharpen(img, params.smart_sharpen_strength, params.smart_sharpen_radius);
    }
    if (params.edge_sharpen_amount > 0.0f) {
        apply_edge_sharpen(img, params.edge_sharpen_amount, params.edge_sharpen_radius, params.edge_sharpen_threshold);
    }

    img.to_uint8(data);
    return true;
}

} // namespace postproc
