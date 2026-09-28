// FreeU Fourier_filter 单元验证：ggml 闭式投影 vs. 朴素 DFT 参考实现（严格复刻 ComfyUI）
#include <cstring>
#include <cmath>
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "core/ggml_fourier.h"

using cplx = std::complex<double>;

// 参考实现：fftn -> fftshift -> 中心 2x2 罩 scale -> ifftshift -> ifftn -> real
// 布局与输入一致：idx = ((c)*H + i)*W + j，ne0=W 最快。
static std::vector<float> reference(const std::vector<float>& x,
                                    int                        W,
                                    int                        H,
                                    int                        C,
                                    float                      scale) {
    std::vector<float> out(x.size());
    const int          crow = H / 2;
    const int          ccol = W / 2;
    for (int c = 0; c < C; ++c) {
        const float* xc = x.data() + (size_t)c * H * W;
        float*       oc = out.data() + (size_t)c * H * W;

        // 2D DFT
        std::vector<cplx> X((size_t)H * W);
        for (int p = 0; p < H; ++p) {
            for (int q = 0; q < W; ++q) {
                cplx sum = 0.0;
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        double ang = -2.0 * M_PI * ((double)p * i / H + (double)q * j / W);
                        sum += (double)xc[(size_t)i * W + j] * cplx(cos(ang), sin(ang));
                    }
                }
                X[(size_t)p * W + q] = sum;
            }
        }
        // fftshift: out[idx] = a[(idx - n/2) mod n]
        auto roll = [](const std::vector<cplx>& a, int n, int shift) {
            std::vector<cplx> r(n);
            for (int i = 0; i < n; ++i) {
                int s = ((i - shift) % n + n) % n;
                r[i]  = a[s];
            }
            return r;
        };
        std::vector<cplx> Y((size_t)H * W);
        for (int i = 0; i < H; ++i) {
            std::vector<cplx> row(W), rrow;
            for (int q = 0; q < W; ++q) row[q] = X[(size_t)i * W + q];
            rrow = roll(row, W, W / 2);
            for (int q = 0; q < W; ++q) Y[(size_t)i * W + q] = rrow[q];
        }
        std::vector<cplx> Z((size_t)H * W);
        for (int q = 0; q < W; ++q) {
            std::vector<cplx> col(H), rcol;
            for (int i = 0; i < H; ++i) col[i] = Y[(size_t)i * W + q];
            rcol = roll(col, H, H / 2);
            for (int i = 0; i < H; ++i) Z[(size_t)i * W + q] = rcol[i];
        }
        // mask：中心 2x2
        for (int i = crow - 1; i <= crow; ++i) {
            for (int j = ccol - 1; j <= ccol; ++j) {
                Z[(size_t)i * W + j] *= (double)scale;
            }
        }
        // ifftshift
        std::vector<cplx> M((size_t)H * W);
        for (int i = 0; i < H; ++i) {
            std::vector<cplx> row(W), rrow;
            for (int q = 0; q < W; ++q) row[q] = Z[(size_t)i * W + q];
            rrow = roll(row, W, -W / 2);
            for (int q = 0; q < W; ++q) M[(size_t)i * W + q] = rrow[q];
        }
        std::vector<cplx> U((size_t)H * W);
        for (int q = 0; q < W; ++q) {
            std::vector<cplx> col(H), rcol;
            for (int i = 0; i < H; ++i) col[i] = M[(size_t)i * W + q];
            rcol = roll(col, H, -H / 2);
            for (int i = 0; i < H; ++i) U[(size_t)i * W + q] = rcol[i];
        }
        // IDFT + real
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                cplx sum = 0.0;
                for (int p = 0; p < H; ++p) {
                    for (int q = 0; q < W; ++q) {
                        double ang = 2.0 * M_PI * ((double)p * i / H + (double)q * j / W);
                        sum += U[(size_t)p * W + q] * cplx(cos(ang), sin(ang));
                    }
                }
                oc[(size_t)i * W + j] = (float)(sum.real() / (double)(H * W));
            }
        }
    }
    return out;
}

static bool run_case(int W, int H, int C, float scale, enum ggml_type type) {
    std::vector<float> x((size_t)W * H * C);
    unsigned int       seed = 12345u;
    for (auto& v : x) {
        seed       = seed * 1664525u + 1013904223u;
        v          = ((float)(seed >> 8) / (float)0xFFFFFF) * 2.0f - 1.0f;
    }

    if (type == GGML_TYPE_F16) {
        for (auto& v : x) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));  // 与实际喂给算子的数值一致
    }

    std::vector<float> expect = reference(x, W, H, C, scale);

    ggml_init_params params = {256ull * 1024 * 1024, nullptr, false};
    ggml_context*    ctx    = ggml_init(params);
    ggml_tensor*     t      = ggml_new_tensor_4d(ctx, type, W, H, C, 1);
    if (type == GGML_TYPE_F32) {
        memcpy(ggml_get_data(t), x.data(), x.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h(x.size());
        for (size_t i = 0; i < x.size(); ++i) h[i] = ggml_fp32_to_fp16(x[i]);
        memcpy(ggml_get_data(t), h.data(), h.size() * sizeof(ggml_fp16_t));
    }

    ggml_tensor* y = ggml_ext_fourier_filter_lowfreq(ctx, t, scale);
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
        printf("unexpected output type %d\n", (int)y->type);
        ggml_free(ctx);
        return false;
    }
    const float* yd = got.data();
    float        max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float d     = std::fabs(yd[i] - expect[i]);
        max_abs     = std::max(max_abs, d);
        max_rel     = std::max(max_rel, d / std::max(1e-3f, std::fabs(expect[i])));
    }
    printf("  expect:"); for (int i=0;i<std::min<int>(8,x.size());++i) printf(" %.6f", expect[i]); printf("\n  got   :"); for (int i=0;i<std::min<int>(8,x.size());++i) printf(" %.6f", yd[i]); printf("\n"); const float tol = (type == GGML_TYPE_F32) ? 1e-5f : 5e-3f;
    bool        ok  = max_abs < tol;
    printf("%-4s W=%2d H=%2d C=%d scale=%.2f  max_abs=%.3e max_rel=%.3e  %s\n",
           type == GGML_TYPE_F32 ? "f32" : "f16", W, H, C, scale, max_abs, max_rel, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_case(8, 8, 3, 0.2f, GGML_TYPE_F32);
    ok &= run_case(8, 8, 3, 0.9f, GGML_TYPE_F32);
    ok &= run_case(16, 16, 1, 0.2f, GGML_TYPE_F32);
    ok &= run_case(16, 8, 4, 1.3f, GGML_TYPE_F32);
    ok &= run_case(7, 9, 2, 0.2f, GGML_TYPE_F32);   // 奇数尺寸
    ok &= run_case(16, 16, 2, 0.2f, GGML_TYPE_F16); // f16 输入/输出
    printf(ok ? "ALL OK\n" : "SOME CASES FAILED\n");
    return ok ? 0 : 1;
}
