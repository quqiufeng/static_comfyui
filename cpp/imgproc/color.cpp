// color.cpp — 模式转换 / 拆分合并 / point(LUT) / 调色板 / 量化（median cut）
#include "imgproc.h"
#include "imgproc_internal.h"

namespace {
inline uint8_t lum(int r, int g, int b) { return clamp8(0.299f * r + 0.587f * g + 0.114f * b); }

Image* as_rgba(Image* im) { return im->frames.empty() || im->frames[0].bpp == 4 ? im : to_rgba(im); }

// 简单 median-cut 量化（源：RGBA）
void median_cut(const std::vector<uint8_t>& px4, int colors, std::vector<RGB>& out) {
    struct Box { std::vector<int> idx; };
    std::vector<Box> boxes;
    Box b0; b0.idx.resize(px4.size() / 4); for (size_t i = 0; i < b0.idx.size(); ++i) b0.idx[i] = (int)i;
    boxes.push_back(b0);
    auto axis_range = [&](Box& bx, int ch, int& mn, int& mx) {
        mn = 255; mx = 0;
        for (int pi : bx.idx) { int v = px4[(size_t)pi * 4 + ch]; if (v < mn) mn = v; if (v > mx) mx = v; }
    };
    while ((int)boxes.size() < colors) {
        int best = -1, best_span = -1, best_ch = 0;
        for (size_t bi = 0; bi < boxes.size(); ++bi) {
            for (int ch = 0; ch < 3; ++ch) { int mn, mx; axis_range(boxes[bi], ch, mn, mx); if (mx - mn > best_span) { best_span = mx - mn; best = (int)bi; best_ch = ch; } }
        }
        if (best < 0 || best_span <= 0) break;
        Box& bx = boxes[best];
        std::sort(bx.idx.begin(), bx.idx.end(), [&](int a, int b) { return px4[(size_t)a * 4 + best_ch] < px4[(size_t)b * 4 + best_ch]; });
        size_t mid = bx.idx.size() / 2;
        Box b1, b2;
        b1.idx.assign(bx.idx.begin(), bx.idx.begin() + mid);
        b2.idx.assign(bx.idx.begin() + mid, bx.idx.end());
        if (b1.idx.empty() || b2.idx.empty()) break;
        boxes[best] = b1; boxes.push_back(b2);
    }
    out.clear();
    for (auto& bx : boxes) {
        if (bx.idx.empty()) continue;
        long sR = 0, sG = 0, sB = 0;
        for (int pi : bx.idx) { sR += px4[(size_t)pi * 4]; sG += px4[(size_t)pi * 4 + 1]; sB += px4[(size_t)pi * 4 + 2]; }
        int k = (int)bx.idx.size();
        out.push_back({(uint8_t)(sR / k), (uint8_t)(sG / k), (uint8_t)(sB / k)});
    }
    if (out.empty()) out.push_back({0, 0, 0});
}
inline int nearest_pal(const std::vector<RGB>& pal, int r, int g, int b) {
    int best = 0; long bd = 1L << 30;
    for (size_t i = 0; i < pal.size(); ++i) {
        long dr = r - pal[i].r, dg = g - pal[i].g, db = b - pal[i].b;
        long d = dr * dr + dg * dg + db * db;
        if (d < bd) { bd = d; best = (int)i; }
    }
    return best;
}
Image* quantize_to_P(Image* im, int colors, const std::vector<RGB>* forced) {
    Image* rg = as_rgba(im);
    std::vector<uint8_t> px4;
    px4.reserve(npx(rg) * 4);
    for (size_t i = 0; i < rg->frames[0].data.size(); i += 4) if (rg->frames[0].data[i + 3] >= 128) for (int c = 0; c < 3; ++c) px4.push_back(rg->frames[0].data[i + c]);
    std::vector<RGB> pal;
    if (forced && !forced->empty()) pal = *forced;
    else { std::vector<uint8_t> all; all.reserve(npx(rg) * 4); for (size_t i = 0; i < rg->frames[0].data.size(); i += 4) for (int c = 0; c < 4; ++c) all.push_back(rg->frames[0].data[i + c]); median_cut(all, colors, pal); }
    Image* o = new Image(); o->w = rg->w; o->h = rg->h; o->n = rg->n; o->loop = rg->loop; o->mode = "P";
    o->frames.resize(rg->n);
    for (int f = 0; f < rg->n; ++f) {
        Frame& d = o->frames[f]; d.bpp = 1; d.hasPalette = true; d.palette = pal; d.delay = rg->frames[f].delay;
        d.data.assign(npx(rg), 0);
        uint8_t* sp = rg->frames[f].data.data();
        for (int i = 0; i < (int)npx(rg); ++i) d.data[i] = (uint8_t)nearest_pal(pal, sp[i * 4], sp[i * 4 + 1], sp[i * 4 + 2]);
    }
    if (rg != im) delete rg;
    return o;
}
}  // namespace

extern "C" {

void* img_convert(void* p, const char* mode) {
    Image* im = (Image*)p; if (!im || !mode) return nullptr;
    std::string m = mode;
    Image* rg = as_rgba(im);
    if (m == "RGBA") { if (rg == im) return dup_img(im); return rg; }
    if (m == "RGB") {
        Image* o = new Image(); o->w = rg->w; o->h = rg->h; o->n = rg->n; o->loop = rg->loop; o->mode = "RGB"; o->frames.resize(rg->n);
        for (int f = 0; f < rg->n; ++f) { Frame& d = o->frames[f]; d.bpp = 4; d.delay = rg->frames[f].delay; d.data.resize(npx(rg) * 4); uint8_t* s = rg->frames[f].data.data(); for (size_t i = 0; i < npx(rg); ++i) { d.data[i * 4] = s[i * 4]; d.data[i * 4 + 1] = s[i * 4 + 1]; d.data[i * 4 + 2] = s[i * 4 + 2]; d.data[i * 4 + 3] = 255; } }
        if (rg != im) delete rg; return o;
    }
    if (m == "L" || m == "LA") {
        Image* o = new Image(); o->w = rg->w; o->h = rg->h; o->n = rg->n; o->loop = rg->loop; o->mode = m; o->frames.resize(rg->n);
        for (int f = 0; f < rg->n; ++f) { Frame& d = o->frames[f]; d.bpp = 4; d.delay = rg->frames[f].delay; d.data.resize(npx(rg) * 4); uint8_t* s = rg->frames[f].data.data(); for (size_t i = 0; i < npx(rg); ++i) { uint8_t l = lum(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]); d.data[i * 4] = d.data[i * 4 + 1] = d.data[i * 4 + 2] = l; d.data[i * 4 + 3] = (m == "LA") ? s[i * 4 + 3] : 255; } }
        if (rg != im) delete rg; return o;
    }
    if (m == "P") { Image* o = quantize_to_P(rg, 256, nullptr); if (rg != im) delete rg; return o; }
    if (rg == im) return dup_img(im);
    return rg;
}

void* img_split(void* p, int channel) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* rg = as_rgba(im);
    Image* o = new Image(); o->w = rg->w; o->h = rg->h; o->n = rg->n; o->loop = rg->loop; o->mode = "L"; o->frames.resize(rg->n);
    for (int f = 0; f < rg->n; ++f) { Frame& d = o->frames[f]; d.bpp = 4; d.delay = rg->frames[f].delay; d.data.resize(npx(rg) * 4); uint8_t* s = rg->frames[f].data.data(); for (size_t i = 0; i < npx(rg); ++i) { uint8_t v = s[i * 4 + clampi(channel, 0, 3)]; d.data[i * 4] = d.data[i * 4 + 1] = d.data[i * 4 + 2] = v; d.data[i * 4 + 3] = 255; } }
    if (rg != im) delete rg; return o;
}

void* img_merge_rgb(void* rp, void* gp, void* bp) {
    Image* r = (Image*)rp; Image* g = (Image*)gp; Image* b = (Image*)bp;
    if (!r || !g || !b || r->w != g->w || r->w != b->w) return nullptr;
    Image* rr = as_rgba(r); Image* gg = as_rgba(g); Image* bb = as_rgba(b);
    Image* o = new_rgba(r->w, r->h, r->n); o->mode = "RGB";
    for (int f = 0; f < r->n; ++f) { uint8_t* pr = rr->frames[f].data.data(); uint8_t* pg = gg->frames[f].data.data(); uint8_t* pb = bb->frames[f].data.data(); uint8_t* d = o->frames[f].data.data(); for (size_t i = 0; i < npx(r); ++i) { d[i * 4] = pr[i * 4]; d[i * 4 + 1] = pg[i * 4]; d[i * 4 + 2] = pb[i * 4]; d[i * 4 + 3] = 255; } }
    if (rr != r) delete rr; if (gg != g) delete gg; if (bb != b) delete bb;
    return o;
}

void* img_point(void* p, int channel, const char* lut_csv) {
    Image* im = (Image*)p; if (!im || !lut_csv) return nullptr;
    std::vector<int> lu = parse_csv_ints(lut_csv);
    if (lu.size() != 256) return nullptr;
    Image* o = dup_img(im);
    Image* rg = as_rgba(o);
    bool all = (channel < 0);
    for (int f = 0; f < rg->n; ++f) { uint8_t* s = rg->frames[f].data.data(); for (size_t i = 0; i < npx(rg); ++i) { if (all) for (int c = 0; c < 3; ++c) s[i * 4 + c] = (uint8_t)clampi(lu[s[i * 4 + c]], 0, 255); else if (channel < 4) s[i * 4 + channel] = (uint8_t)clampi(lu[s[i * 4 + channel]], 0, 255); } }
    if (rg != o) { *o = *rg; delete rg; }
    return o;
}

void* img_quantize(void* p, int colors, int method) {
    (void)method;
    Image* im = (Image*)p; if (!im || colors < 1) return nullptr;
    return quantize_to_P(im, clampi(colors, 1, 256), nullptr);
}

int img_putpalette(void* p, const char* rgb_csv) {
    Image* im = (Image*)p; if (!im || !rgb_csv) return -1;
    std::vector<int> v = parse_csv_ints(rgb_csv);
    std::vector<RGB> pal; for (size_t i = 0; i + 2 < v.size(); i += 3) pal.push_back({(uint8_t)v[i], (uint8_t)v[i + 1], (uint8_t)v[i + 2]});
    if (pal.empty()) return -1;
    Image* o = quantize_to_P(im, (int)pal.size(), &pal);
    *im = *o; delete o; return 0;
}

int img_getpalette(void* p, char* out, int outlen) {
    Image* im = (Image*)p; if (!im || !out) return -1;
    std::string s;
    for (auto& f : im->frames) if (f.bpp == 1 && f.hasPalette) {
        for (size_t i = 0; i < f.palette.size(); ++i) { char buf[16]; snprintf(buf, 16, "%d,%d,%d,", f.palette[i].r, f.palette[i].g, f.palette[i].b); s += buf; }
        break;
    }
    if ((int)s.size() >= outlen) s.resize(outlen - 1);
    strcpy(out, s.c_str()); return 0;
}

}  // extern "C"
