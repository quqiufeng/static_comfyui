// FreeU_V2 backbone 缩放：ggml 实现 vs 朴素参考（复刻 ComfyUI FreeU_V2 output_block_patch）
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "ggml.h"
#include "ggml-cpu.h"
#include "core/ggml_fourier.h"

static std::vector<float> reference(const std::vector<float>& x, int W, int H, int C, int N, float b) {
    std::vector<float> out = x;
    const int half = C / 2;
    for (int n = 0; n < N; ++n) {
        std::vector<float> m((size_t)H * W);
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float s = 0.0f;
                for (int c = 0; c < C; ++c) {
                    s += x[((size_t)n * C + c) * H * W + (size_t)i * W + j];
                }
                m[(size_t)i * W + j] = s / (float)C;
            }
        }
        float mn = m[0], mx = m[0];
        for (float v : m) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const float span = mx - mn;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                const float f = (b - 1.0f) * ((m[(size_t)i * W + j] - mn) / span) + 1.0f;
                for (int c = 0; c < half; ++c) {
                    out[((size_t)n * C + c) * H * W + (size_t)i * W + j] *= f;
                }
            }
        }
    }
    return out;
}

static bool run_case(int W, int H, int C, int N, float b, enum ggml_type type) {
    std::vector<float> x((size_t)W * H * C * N);
    unsigned int       seed = 424242u;
    for (auto& v : x) {
        seed = seed * 1664525u + 1013904223u;
        v    = ((float)(seed >> 8) / (float)0xFFFFFF) * 2.0f - 1.0f;
    }
    if (type == GGML_TYPE_F16) {
        for (auto& v : x) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
    }

    std::vector<float> expect = reference(x, W, H, C, N, b);

    ggml_init_params params = {256ull * 1024 * 1024, nullptr, false};
    ggml_context*    ctx    = ggml_init(params);
    ggml_tensor*     t      = ggml_new_tensor_4d(ctx, type, W, H, C, N);
    if (type == GGML_TYPE_F32) {
        memcpy(ggml_get_data(t), x.data(), x.size() * sizeof(float));
    } else {
        ggml_fp16_t* hp = (ggml_fp16_t*)ggml_get_data(t);
        for (size_t i = 0; i < x.size(); ++i) hp[i] = ggml_fp32_to_fp16(x[i]);
    }

    ggml_tensor* y = ggml_ext_freeu_v2_backbone(ctx, t, b);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> got(x.size());
    if (y->type == GGML_TYPE_F32) {
        memcpy(got.data(), ggml_get_data(y), x.size() * sizeof(float));
    } else if (y->type == GGML_TYPE_F16) {
        const ggml_fp16_t* hp = (const ggml_fp16_t*)ggml_get_data(y);
        for (size_t i = 0; i < x.size(); ++i) got[i] = ggml_fp16_to_fp32(hp[i]);
    } else {
        printf("unexpected type %d\n", (int)y->type);
        ggml_free(ctx);
        return false;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) max_abs = std::max(max_abs, std::fabs(got[i] - expect[i]));
    const float tol = (type == GGML_TYPE_F32) ? 1e-5f : 5e-3f;
    const bool  ok  = max_abs < tol;
    printf("%-4s W=%2d H=%2d C=%d N=%d b=%.2f  max_abs=%.3e  %s\n",
           type == GGML_TYPE_F32 ? "f32" : "f16", W, H, C, N, b, max_abs, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_case(8, 8, 4, 1, 1.3f, GGML_TYPE_F32);
    ok &= run_case(16, 8, 8, 2, 1.4f, GGML_TYPE_F32);  // batch=2（CFG 打包）
    ok &= run_case(7, 9, 6, 1, 1.3f, GGML_TYPE_F32);   // 奇数尺寸
    ok &= run_case(16, 16, 4, 2, 1.1f, GGML_TYPE_F16); // f16 输入输出
    ok &= run_case(8, 8, 4, 1, 1.0f, GGML_TYPE_F32);   // b == 1 短路
    printf(ok ? "ALL OK\n" : "SOME CASES FAILED\n");
    return ok ? 0 : 1;
}
