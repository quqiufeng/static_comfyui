// FreSca 频带滤波单元验证：freq_band_filter vs 朴素 DFT 参考（严格复刻 ComfyUI）
// 对应 src/runtime/freq_filter.h（patch patches/sdcpp-freeu-sag-v2.patch 的 FreSca 部分）
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "runtime/freq_filter.h"

using cplx = std::complex<double>;

// 参考：fftn -> fftshift -> 盒 [cc-f_c, cc+f_c) 置 scale_low（其余 scale_high）
//        -> ifftshift -> ifftn -> real
// 布局与输入一致：idx = ((c)*H + i)*W + j，W 最快。
static std::vector<float> reference(const std::vector<float>& x,
                                    int                        W,
                                    int                        H,
                                    int                        C,
                                    float                      scale_low,
                                    float                      scale_high,
                                    int                        freq_cutoff) {
    std::vector<float> out(x.size());
    const int          ch = H / 2;
    const int          cw = W / 2;
    const int          fh = std::min(freq_cutoff, H / 2);
    const int          fw = std::min(freq_cutoff, W / 2);

    auto roll = [](const std::vector<cplx>& a, int n, int shift) {
        std::vector<cplx> r(n);
        for (int i = 0; i < n; ++i) {
            int s = ((i - shift) % n + n) % n;
            r[i]  = a[s];
        }
        return r;
    };

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

        // fftshift（两轴，roll(a, n, n/2)）
        std::vector<cplx> Y((size_t)H * W);
        for (int i = 0; i < H; ++i) {
            std::vector<cplx> row(W);
            for (int q = 0; q < W; ++q) row[q] = X[(size_t)i * W + q];
            row = roll(row, W, W / 2);
            for (int q = 0; q < W; ++q) Y[(size_t)i * W + q] = row[q];
        }
        std::vector<cplx> Z((size_t)H * W);
        for (int q = 0; q < W; ++q) {
            std::vector<cplx> col(H);
            for (int i = 0; i < H; ++i) col[i] = Y[(size_t)i * W + q];
            col = roll(col, H, H / 2);
            for (int i = 0; i < H; ++i) Z[(size_t)i * W + q] = col[i];
        }

        // mask：全 scale_high，盒 [ch-fh, ch+fh) × [cw-fw, cw+fw) 置 scale_low
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                double v = (double)scale_high;
                if (fh > 0 && fw > 0 && i >= ch - fh && i < ch + fh && j >= cw - fw && j < cw + fw) {
                    v = (double)scale_low;
                }
                Z[(size_t)i * W + j] *= v;
            }
        }

        // ifftshift（两轴，roll(a, n, -n/2)）
        std::vector<cplx> M((size_t)H * W);
        for (int i = 0; i < H; ++i) {
            std::vector<cplx> row(W);
            for (int q = 0; q < W; ++q) row[q] = Z[(size_t)i * W + q];
            row = roll(row, W, -W / 2);
            for (int q = 0; q < W; ++q) M[(size_t)i * W + q] = row[q];
        }
        std::vector<cplx> U((size_t)H * W);
        for (int q = 0; q < W; ++q) {
            std::vector<cplx> col(H);
            for (int i = 0; i < H; ++i) col[i] = M[(size_t)i * W + q];
            col = roll(col, H, -H / 2);
            for (int i = 0; i < H; ++i) U[(size_t)i * W + q] = col[i];
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

static bool run_case(int         W,
                     int         H,
                     int         C,
                     float       scale_low,
                     float       scale_high,
                     int         freq_cutoff) {
    std::vector<float> x((size_t)W * H * C);
    unsigned int       seed = 12345u;
    for (auto& v : x) {
        seed = seed * 1664525u + 1013904223u;
        v    = ((float)(seed >> 8) / (float)0xFFFFFF) * 2.0f - 1.0f;
    }

    std::vector<float> expect = reference(x, W, H, C, scale_low, scale_high, freq_cutoff);

    std::vector<float> got = x;
    sd::guidance::freq_band_filter(got.data(), C, H, W, scale_low, scale_high, freq_cutoff);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float d = std::fabs(got[i] - expect[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1e-3f, std::fabs(expect[i])));
    }
    const float tol = 1e-5f;
    bool        ok  = max_abs < tol;
    printf("W=%2d H=%2d C=%d low=%.2f high=%.2f cut=%2d  max_abs=%.3e max_rel=%.3e  %s\n",
           W, H, C, scale_low, scale_high, freq_cutoff, max_abs, max_rel, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= run_case(8, 8, 3, 1.0f, 1.25f, 3);   // 常规
    ok &= run_case(16, 16, 4, 1.0f, 1.5f, 20); // cutoff 超过轴长 → 截断（偶数全谱）
    ok &= run_case(7, 9, 2, 1.0f, 1.5f, 20);   // 奇数尺寸（盒缺 +cc 一个 bin）
    ok &= run_case(6, 8, 2, 1.0f, 1.25f, 0);   // cutoff=0 → 盒空 → high*x
    ok &= run_case(8, 8, 1, 1.3f, 1.3f, 5);    // low==high → 均匀缩放
    ok &= run_case(5, 12, 3, 0.9f, 1.4f, 2);
    ok &= run_case(8, 1, 2, 1.0f, 1.25f, 4);   // H=1 → 轴 cc=0 → 盒空
    ok &= run_case(1, 8, 2, 1.0f, 1.25f, 4);   // W=1
    printf(ok ? "ALL OK\n" : "SOME CASES FAILED\n");
    return ok ? 0 : 1;
}
