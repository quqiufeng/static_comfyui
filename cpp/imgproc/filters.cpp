// filters.cpp — ImageFilter（对齐 Pillow kernel/公式）+ ImageStat
#include "imgproc.h"
#include "imgproc_internal.h"

namespace {
Image* as_rgba(Image* im) { return im->frames.empty() || im->frames[0].bpp == 4 ? im : to_rgba(im); }

// 通用 NxN 卷积。严格对齐 Pillow ImagingFilter.c：
//   - 边界像素复制原值（不参与卷积）
//   - 浮点累加：ss = (offset+0.5) + Σ kernel*px/scale，截断到 [0,255]
//   - 3x3 的行方向对 kernel 做垂直翻转（Pillow 的既有行为）；5x5 不翻
Image* convolve(Image* rg, const int* k, int kw, int kh, int scale, int offset) {
    Image* o = new_rgba(rg->w, rg->h, rg->n); o->loop = rg->loop;
    int W = rg->w, H = rg->h, rx = kw / 2, ry = kh / 2;
    const float off = offset + 0.5f;
    for (int f = 0; f < rg->n; ++f) {
        uint8_t* s = rg->frames[f].data.data(); uint8_t* d = o->frames[f].data.data();
        // 先整帧复制，再只算内部区域（等价于 Pillow 的边界复制）
        memcpy(d, s, (size_t)W * H * 4);
        for (int y = ry; y < H - ry; ++y) for (int x = rx; x < W - rx; ++x) {
            for (int c = 0; c < 3; ++c) {
                float ss = off;
                for (int j = 0; j < kh; ++j) {
                    int yy = (kh == 3) ? (y + ry - j) : (y + j - ry);
                    for (int i = 0; i < kw; ++i) {
                        int xx = x + i - rx;
                        ss += (float)k[j * kw + i] * (float)s[((size_t)yy * W + xx) * 4 + c] / (float)scale;
                    }
                }
                d[((size_t)y * W + x) * 4 + c] = (uint8_t)(ss < 0 ? 0 : (ss > 255 ? 255 : ss));
            }
        }
    }
    return o;
}

// 单次盒模糊（radius 半径，窗口 2r+1），边缘 clamp
Image* box_blur_radius(Image* rg, int radius) {
    if (radius < 1) return dup_img(rg);
    int W = rg->w, H = rg->h, k = 2 * radius + 1;
    Image* o = new_rgba(W, H, rg->n); o->loop = rg->loop;
    std::vector<int> tmp((size_t)W * H * 4);
    for (int f = 0; f < rg->n; ++f) {
        uint8_t* s = rg->frames[f].data.data();
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            int sr = 0, sg = 0, sb = 0;
            for (int i = -radius; i <= radius; ++i) { uint8_t* q = s + ((size_t)y * W + clampi(x + i, 0, W - 1)) * 4; sr += q[0]; sg += q[1]; sb += q[2]; }
            size_t oo = ((size_t)y * W + x) * 4;
            tmp[oo] = sr / k; tmp[oo + 1] = sg / k; tmp[oo + 2] = sb / k; tmp[oo + 3] = s[oo + 3];
        }
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            int sr = 0, sg = 0, sb = 0;
            for (int j = -radius; j <= radius; ++j) { size_t q = ((size_t)clampi(y + j, 0, H - 1) * W + x) * 4; sr += tmp[q]; sg += tmp[q + 1]; sb += tmp[q + 2]; }
            uint8_t* d = o->frames[f].data.data() + ((size_t)y * W + x) * 4;
            d[0] = sr / k; d[1] = sg / k; d[2] = sb / k; d[3] = s[((size_t)y * W + x) * 4 + 3];
        }
    }
    return o;
}

// Pillow 风格高斯：3 次盒模糊近似（sigma=radius）
Image* gaussian_blur_r(Image* rg, float radius) {
    if (radius <= 0) return dup_img(rg);
    const int n = 3; float sigma = radius;
    float wIdeal = sqrt((12 * sigma * sigma / n) + 1);
    int wl = (int)floor(wIdeal); if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    float mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4.0f * wl - 4);
    int m = (int)round(mIdeal);
    Image* cur = dup_img(rg);
    for (int i = 0; i < n; ++i) { int sz = (i < m ? wl : wu) / 2; Image* nx = box_blur_radius(cur, sz); delete cur; cur = nx; }
    return cur;
}

Image* rank_filter(Image* rg, int size, int mode) {
    int W = rg->w, H = rg->h, r = size / 2;
    Image* o = new_rgba(W, H, rg->n); o->loop = rg->loop;
    int rank = size * size / 2;
    std::vector<int> vals;
    for (int f = 0; f < rg->n; ++f) {
        uint8_t* s = rg->frames[f].data.data(); uint8_t* d = o->frames[f].data.data();
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            for (int c = 0; c < 3; ++c) {
                vals.clear();
                for (int j = -r; j <= r; ++j) for (int i = -r; i <= r; ++i) { int xx = clampi(x + i, 0, W - 1), yy = clampi(y + j, 0, H - 1); vals.push_back(s[((size_t)yy * W + xx) * 4 + c]); }
                std::sort(vals.begin(), vals.end());
                int v;
                if (mode == 0) v = vals.front();
                else if (mode == 1) v = vals.back();
                else if (mode == 4) v = vals[std::min(rank, (int)vals.size() - 1)];
                else if (mode == 3) { int best = vals[0], bc = 0, cur = vals[0], cc = 0; for (int t : vals) { if (t == cur) cc++; else { if (cc > bc) { bc = cc; best = cur; } cur = t; cc = 1; } } if (cc > bc) best = cur; v = best; }
                else v = vals[vals.size() / 2];
                d[((size_t)y * W + x) * 4 + c] = (uint8_t)v;
            }
            d[((size_t)y * W + x) * 4 + 3] = s[((size_t)y * W + x) * 4 + 3];
        }
    }
    return o;
}
}  // namespace

extern "C" {

void* img_filter(void* p, const char* name, float r1, float r2, int i1) {
    Image* im = (Image*)p; if (!im || !name) return nullptr;
    std::string n = name;
    Image* rg = as_rgba(im);
    Image* out = nullptr;
    static const int BLUR[25] = {1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1};
    static const int SMOOTH[9] = {1,1,1, 1,5,1, 1,1,1};
    static const int SMOOTH_MORE[25] = {1,1,1,1,1, 1,5,5,5,1, 1,5,44,5,1, 1,5,5,5,1, 1,1,1,1,1};
    static const int DETAIL[9] = {0,-1,0, -1,10,-1, 0,-1,0};
    static const int SHARPEN[9] = {-2,-2,-2, -2,32,-2, -2,-2,-2};
    static const int EE[9] = {-1,-1,-1, -1,10,-1, -1,-1,-1};
    static const int EEM[9] = {-1,-1,-1, -1,9,-1, -1,-1,-1};
    static const int FE[9] = {-1,-1,-1, -1,8,-1, -1,-1,-1};
    static const int EMB[9] = {-1,0,0, 0,1,0, 0,0,0};
    if (n == "BLUR") out = convolve(rg, BLUR, 5, 5, 16, 0);
    else if (n == "SMOOTH") out = convolve(rg, SMOOTH, 3, 3, 13, 0);
    else if (n == "SMOOTH_MORE") out = convolve(rg, SMOOTH_MORE, 5, 5, 100, 0);
    else if (n == "DETAIL") out = convolve(rg, DETAIL, 3, 3, 6, 0);
    else if (n == "SHARPEN") out = convolve(rg, SHARPEN, 3, 3, 16, 0);
    else if (n == "EDGE_ENHANCE") out = convolve(rg, EE, 3, 3, 2, 0);
    else if (n == "EDGE_ENHANCE_MORE") out = convolve(rg, EEM, 3, 3, 1, 0);
    else if (n == "FIND_EDGES") out = convolve(rg, FE, 3, 3, 1, 0);
    else if (n == "CONTOUR") out = convolve(rg, FE, 3, 3, 1, 255);
    else if (n == "EMBOSS") out = convolve(rg, EMB, 3, 3, 1, 128);
    else if (n == "BoxBlur") out = box_blur_radius(rg, std::max(1, (int)(r1 > 0 ? r1 : 1)));
    else if (n == "GaussianBlur") out = gaussian_blur_r(rg, r1 > 0 ? r1 : 2);
    else if (n == "UnsharpMask") {
        float radius = r1 > 0 ? r1 : 2, percent = r2 > 0 ? r2 : 150; int threshold = i1;
        Image* bl = gaussian_blur_r(rg, radius);
        out = dup_img(rg);
        for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); uint8_t* b = bl->frames[f].data.data(); for (size_t i = 0; i < npx(rg); ++i) for (int c = 0; c < 3; ++c) { int diff = s[i * 4 + c] - b[i * 4 + c]; if (abs(diff) * 100 / 255 < threshold) continue; s[i * 4 + c] = clamp8(s[i * 4 + c] + diff * percent / 100.0f); } }
        delete bl;
    }
    else if (n == "MedianFilter") out = rank_filter(rg, i1 > 0 ? i1 : 3, 2);
    else if (n == "MinFilter") out = rank_filter(rg, i1 > 0 ? i1 : 3, 0);
    else if (n == "MaxFilter") out = rank_filter(rg, i1 > 0 ? i1 : 3, 1);
    else if (n == "ModeFilter") out = rank_filter(rg, i1 > 0 ? i1 : 3, 3);
    else if (n == "RankFilter") out = rank_filter(rg, i1 > 0 ? i1 : 3, 4);
    else out = convolve(rg, SMOOTH, 3, 3, 13, 0);
    if (rg != im) delete rg;
    return out;
}

int img_stat(void* p, const char* what, char* out, int outlen) {
    Image* im = (Image*)p; if (!im || !what || !out) return -1;
    std::string w = what;
    Image* rg = as_rgba(im);
    std::string s;
    for (int c = 0; c < 4; ++c) {
        std::vector<int> vals;
        for (size_t i = 0; i < npx(rg); ++i) vals.push_back(rg->frames[0].data[i * 4 + c]);
        double res = 0;
        if (w == "mean") { long t = 0; for (int v : vals) t += v; res = (double)t / vals.size(); }
        else if (w == "sum") { long t = 0; for (int v : vals) t += v; res = (double)t; }
        else if (w == "median") { std::sort(vals.begin(), vals.end()); res = vals[vals.size() / 2]; }
        else if (w == "extrema") { auto mm = std::minmax_element(vals.begin(), vals.end()); res = *mm.first; }
        else if (w == "stddev") { double m = 0; for (int v : vals) m += v; m /= vals.size(); double ss = 0; for (int v : vals) ss += (v - m) * (v - m); res = sqrt(ss / vals.size()); }
        char buf[64];
        if (w == "mean" || w == "stddev") snprintf(buf, 64, "%.6f", res); else snprintf(buf, 64, "%d", (int)(res + 0.5));
        if (c) s += ","; s += buf;
    }
    if (rg != im) delete rg;
    if ((int)s.size() >= outlen) s.resize(outlen - 1);
    strcpy(out, s.c_str()); return 0;
}

int img_getextrema(void* p, char* out, int outlen) {
    Image* im = (Image*)p; if (!im || !out) return -1;
    Image* rg = as_rgba(im);
    int mn[4] = {255, 255, 255, 255}, mx[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < npx(rg); ++i) for (int c = 0; c < 4; ++c) { int v = rg->frames[0].data[i * 4 + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
    if (rg != im) delete rg;
    snprintf(out, outlen, "%d,%d,%d,%d,%d,%d,%d,%d", mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], mn[3], mx[3]);
    return 0;
}

}  // extern "C"
