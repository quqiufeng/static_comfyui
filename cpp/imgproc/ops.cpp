// ops.cpp — ImageOps / ImageChops / ImageEnhance（对齐 Pillow 公式）
#include "imgproc.h"
#include "imgproc_internal.h"

namespace {
Image* as_rgba(Image* im) { return im->frames.empty() || im->frames[0].bpp == 4 ? im : to_rgba(im); }
inline uint8_t lum(int r, int g, int b) { return clamp8(0.299f * r + 0.587f * g + 0.114f * b); }
void ensure_rgba_inplace(Image* im) { if (im->frames[0].bpp != 4) { Image* t = to_rgba(im); *im = *t; delete t; } }
inline uint8_t mix(uint8_t a, uint8_t b, float f) { return clamp8(a + (b - a) * f); }  // Pillow blend
}

extern "C" {

void* img_invert(void* p) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* o = as_rgba(im); if (o == im) o = dup_img(im);
    for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) { s[i * 4] = 255 - s[i * 4]; s[i * 4 + 1] = 255 - s[i * 4 + 1]; s[i * 4 + 2] = 255 - s[i * 4 + 2]; } }
    return o;
}

void* img_grayscale(void* p) { return img_convert(p, "L"); }

void* img_posterize(void* p, int bits) {
    Image* im = (Image*)p; if (!im) return nullptr;
    bits = clampi(bits, 1, 8);
    int mask = ~((1 << (8 - bits)) - 1);
    Image* o = dup_img(im); ensure_rgba_inplace(o);
    for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) for (int c = 0; c < 3; ++c) s[i * 4 + c] = (uint8_t)(s[i * 4 + c] & mask); }
    return o;
}

void* img_solarize(void* p, int threshold) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* o = dup_img(im); ensure_rgba_inplace(o);
    for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) for (int c = 0; c < 3; ++c) if (s[i * 4 + c] >= threshold) s[i * 4 + c] = 255 - s[i * 4 + c]; }
    return o;
}

void* img_autocontrast(void* p, int cutoff) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* o = dup_img(im); ensure_rgba_inplace(o);
    int total = (int)(npx(o));
    for (int c = 0; c < 3; ++c) {
        long hist[256] = {0};
        for (size_t i = 0; i < npx(o); ++i) hist[o->frames[0].data[i * 4 + c]]++;
        int cut = total * clampi(cutoff, 0, 49) / 100;
        int lo = 0, hi = 255; long acc = 0;
        for (int i = 0; i < 256; ++i) { acc += hist[i]; if (acc > cut) { lo = i; break; } }
        acc = 0; for (int i = 255; i >= 0; --i) { acc += hist[i]; if (acc > cut) { hi = i; break; } }
        if (hi <= lo) continue;
        uint8_t lut[256];
        for (int i = 0; i < 256; ++i) lut[i] = clamp8((float)(i - lo) * 255.0f / (hi - lo));
        for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) s[i * 4 + c] = lut[s[i * 4 + c]]; }
    }
    return o;
}

void* img_equalize(void* p) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* o = dup_img(im); ensure_rgba_inplace(o);
    for (int c = 0; c < 3; ++c) {
        long h[256] = {0};
        for (size_t i = 0; i < npx(o); ++i) h[o->frames[0].data[i * 4 + c]]++;
        long nzsum = 0; int nonzero = 0; long last = 0;
        for (int i = 0; i < 256; ++i) if (h[i]) { nzsum += h[i]; nonzero++; last = h[i]; }
        uint8_t lut[256];
        if (nonzero <= 1) { for (int i = 0; i < 256; ++i) lut[i] = (uint8_t)i; }
        else {
            long step = (nzsum - last) / 255;
            if (step == 0) { for (int i = 0; i < 256; ++i) lut[i] = (uint8_t)i; }
            else { long n = step / 2; for (int i = 0; i < 256; ++i) { lut[i] = (uint8_t)clampi((int)(n / step), 0, 255); n += h[i]; } }
        }
        for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) s[i * 4 + c] = lut[s[i * 4 + c]]; }
    }
    return o;
}

void* img_colorize(void* p, int black, int white, const char* mid_csv) {
    Image* im = (Image*)p; if (!im) return nullptr;
    std::vector<int> mid = parse_csv_ints(mid_csv);
    Image* rg = as_rgba(im);
    Image* o = dup_img(rg); if (rg != im) delete rg;
    int bR = CH_R(black), bG = CH_G(black), bB = CH_B(black);
    int wR = CH_R(white), wG = CH_G(white), wB = CH_B(white);
    bool hasm = mid.size() >= 3;
    int mR = hasm ? mid[0] : 0, mG = hasm ? mid[1] : 0, mB = hasm ? mid[2] : 0;
    int midpt = hasm && (int)mid.size() >= 4 ? mid[3] : 128;
    for (int f = 0; f < o->n; ++f) { uint8_t* s = o->frames[f].data.data(); for (size_t i = 0; i < npx(o); ++i) { int l = s[i * 4]; int r0, g0, b0, r1, g1, b1; float tt; if (hasm && l < midpt) { r0 = bR; g0 = bG; b0 = bB; r1 = mR; g1 = mG; b1 = mB; tt = midpt ? (float)l / midpt : 0; } else if (hasm) { r0 = mR; g0 = mG; b0 = mB; r1 = wR; g1 = wG; b1 = wB; tt = (255 - midpt) ? (float)(l - midpt) / (255 - midpt) : 0; } else { r0 = bR; g0 = bG; b0 = bB; r1 = wR; g1 = wG; b1 = wB; tt = l / 255.0f; } s[i * 4] = clamp8(r0 + (r1 - r0) * tt); s[i * 4 + 1] = clamp8(g0 + (g1 - g0) * tt); s[i * 4 + 2] = clamp8(b0 + (b1 - b0) * tt); } }
    return o;
}

void* img_expand(void* p, int border, int rgba) { return img_pad(p, border, border, border, border, rgba); }

// ---------------- ImageChops ----------------
void* img_chop(void* ap, void* bp, const char* op) {
    Image* a = (Image*)ap; Image* b = (Image*)bp; if (!a || !b || !op || a->w != b->w || a->h != b->h) return nullptr;
    std::string o = op;
    Image* ra = to_rgba(a); Image* rb = to_rgba(b);
    Image* out = dup_img(ra);
    bool logical = (o == "logical_and" || o == "logical_or" || o == "logical_xor");
    for (int f = 0; f < out->n; ++f) {
        uint8_t* pa = out->frames[f].data.data(); uint8_t* pb = rb->frames[f].data.data(); uint8_t* oa = ra->frames[f].data.data();
        for (size_t i = 0; i < npx(a); ++i) for (int c = 0; c < 3; ++c) {
            int x = oa[i * 4 + c], y = pb[i * 4 + c], v;
            if (o == "add") v = x + y;
            else if (o == "subtract") v = x - y;
            else if (o == "difference") v = abs(x - y);
            else if (o == "multiply") v = x * y / 255;
            else if (o == "screen") v = 255 - (255 - x) * (255 - y) / 255;
            else if (o == "darker") v = std::min(x, y);
            else if (o == "lighter") v = std::max(x, y);
            else if (o == "blend") v = (x + y) / 2;
            else if (o == "soft_light") v = ((255 - x) * (x * y)) / 65536 + (x * (255 - ((255 - x) * (255 - y) / 255))) / 255;
            else if (o == "hard_light") v = (y < 128) ? (x * y / 127) : (255 - ((255 - y) * (255 - x)) / 127);
            else if (o == "overlay") v = (x < 128) ? (x * y / 127) : (255 - ((255 - x) * (255 - y)) / 127);
            else if (o == "add_modulo") { pa[i * 4 + c] = (uint8_t)((x + y) & 0xFF); continue; }
            else if (o == "subtract_modulo") { pa[i * 4 + c] = (uint8_t)((x - y) & 0xFF); continue; }
            else if (o == "logical_and") { pa[i * 4 + c] = (x && y) ? 255 : 0; continue; }
            else if (o == "logical_or") { pa[i * 4 + c] = (x || y) ? 255 : 0; continue; }
            else if (o == "logical_xor") { pa[i * 4 + c] = ((x != 0) ^ (y != 0)) ? 255 : 0; continue; }
            else v = x;
            pa[i * 4 + c] = (uint8_t)clampi(v, 0, 255);
        }
    }
    (void)logical;
    delete ra; delete rb; return out;
}

void* img_chop_const(void* p, const char* op, int val) {
    Image* im = (Image*)p; if (!im || !op) return nullptr;
    std::string o = op;
    Image* out = dup_img(im); ensure_rgba_inplace(out);
    for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); for (size_t i = 0; i < npx(out); ++i) for (int c = 0; c < 3; ++c) { int x = s[i * 4 + c], v; if (o == "add") v = x + val; else if (o == "subtract") v = x - val; else if (o == "darker") v = std::min(x, val); else if (o == "lighter") v = std::max(x, val); else if (o == "multiply") v = x * val / 255; else v = x; s[i * 4 + c] = (uint8_t)clampi(v, 0, 255); } }
    return out;
}

void* img_chop_offset(void* p, int dx, int dy) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* rg = as_rgba(im);
    Image* o = new_rgba(rg->w, rg->h, rg->n); o->loop = rg->loop;
    for (int f = 0; f < rg->n; ++f) {
        uint8_t* s = rg->frames[f].data.data(); uint8_t* d = o->frames[f].data.data();
        for (int y = 0; y < rg->h; ++y) for (int x = 0; x < rg->w; ++x) {
            int sx = ((x - dx) % rg->w + rg->w) % rg->w, sy = ((y - dy) % rg->h + rg->h) % rg->h;
            memcpy(d + ((size_t)y * rg->w + x) * 4, s + ((size_t)sy * rg->w + sx) * 4, 4);
        }
    }
    if (rg != im) delete rg; return o;
}

// ---------------- ImageEnhance（= blend(degenerate, image, factor)）----------------
void* img_enhance(void* p, const char* kind, float factor) {
    Image* im = (Image*)p; if (!im || !kind) return nullptr;
    std::string k = kind;
    Image* rg = as_rgba(im);
    Image* out = dup_img(rg); if (rg != im) delete rg;
    if (k == "brightness") {
        for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); for (size_t i = 0; i < npx(out); ++i) for (int c = 0; c < 3; ++c) s[i * 4 + c] = mix(0, s[i * 4 + c], factor); }
    } else if (k == "color") {
        for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); for (size_t i = 0; i < npx(out); ++i) { uint8_t l = lum(s[i * 4], s[i * 4 + 1], s[i * 4 + 2]); for (int c = 0; c < 3; ++c) s[i * 4 + c] = mix(l, s[i * 4 + c], factor); } }
    } else if (k == "contrast") {
        long m = 0; for (size_t i = 0; i < npx(out); ++i) m += lum(out->frames[0].data[i * 4], out->frames[0].data[i * 4 + 1], out->frames[0].data[i * 4 + 2]);
        uint8_t gray = (uint8_t)(int)((double)m / npx(out) + 0.5);
        for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); for (size_t i = 0; i < npx(out); ++i) for (int c = 0; c < 3; ++c) s[i * 4 + c] = mix(gray, s[i * 4 + c], factor); }
    } else if (k == "sharpness") {
        Image* sm = (Image*)img_filter(out, "SMOOTH", 0, 0, 0);
        for (int f = 0; f < out->n; ++f) { uint8_t* s = out->frames[f].data.data(); uint8_t* b = sm->frames[f].data.data(); for (size_t i = 0; i < npx(out) * 4; ++i) s[i] = clamp8(b[i] + (s[i] - b[i]) * factor); }
        delete sm;
    }
    return out;
}

// ImageChops.composite(a,b,mask)：out = a + (b-a)*mask/255（逐通道，含 A）
void* img_chops_composite(void* ap, void* bp, void* mp) {
    Image* a = (Image*)ap; Image* b = (Image*)bp; Image* m = (Image*)mp;
    if (!a || !b || !m || a->w != b->w || a->h != b->h) return nullptr;
    Image* ra = to_rgba(a); Image* rb = to_rgba(b); Image* rm = to_rgba(m);
    Image* out = dup_img(ra);
    for (int f = 0; f < out->n; ++f) {
        uint8_t* pa = out->frames[f].data.data(); uint8_t* pb = rb->frames[f].data.data(); uint8_t* pm = rm->frames[f].data.data();
        for (size_t i = 0; i < npx(a); ++i) { int mm = pm[i * 4]; for (int c = 0; c < 4; ++c) pa[i * 4 + c] = (uint8_t)((pa[i * 4 + c] * mm + pb[i * 4 + c] * (255 - mm) + 127) / 255); }
    }
    delete ra; delete rb; delete rm; return out;
}

// ImageOps.pad：等比缩放到放进 (w,h)，再按 centering 贴到 (w,h) 的纯色底上
void* img_ops_pad(void* p, int w, int h, const char* resample, int color_rgba, float cx, float cy) {
    Image* im = (Image*)p; if (!im || w <= 0 || h <= 0) return nullptr;
    float scale = std::min((float)w / im->w, (float)h / im->h);
    int nw = std::max(1, (int)round(im->w * scale)), nh = std::max(1, (int)round(im->h * scale));
    Image* r = (Image*)img_resize(im, nw, nh, resample);
    Image* o = new_rgba(w, h, im->n); o->loop = im->loop;
    img_fill(o, color_rgba);
    int ox = (int)((w - nw) * cx), oy = (int)((h - nh) * cy);
    if (o->frames[0].bpp != 4) { Image* t = to_rgba(o); *o = *t; delete t; }
    Image* rr = (r->frames[0].bpp == 4) ? r : to_rgba(r);
    for (int f = 0; f < o->n; ++f) { uint8_t* dp = o->frames[f].data.data(); uint8_t* sp = rr->frames[f].data.data();
        for (int y = 0; y < nh; ++y) for (int x = 0; x < nw; ++x) { int bx = ox + x, by = oy + y; if (bx < 0 || bx >= w || by < 0 || by >= h) continue; memcpy(dp + ((size_t)by * w + bx) * 4, sp + ((size_t)y * nw + x) * 4, 4); } }
    if (rr != r) delete rr; delete r; return o;
}

}  // extern "C"
