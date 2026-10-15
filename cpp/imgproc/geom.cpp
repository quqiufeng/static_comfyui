// geom.cpp — 几何：crop/resize/thumbnail/rotate/transpose/pad/scale/paste/composite/blend/getbbox
#include "imgproc.h"
#include "imgproc_internal.h"

namespace {
Image* as_rgba(Image* im) { return im->frames.empty() || im->frames[0].bpp == 4 ? im : to_rgba(im); }

inline float cubic(float v) {
    v = (float)fabs(v);
    if (v < 1) return 1.5f * v * v * v - 2.5f * v * v + 1;
    if (v < 2) return -0.5f * v * v * v + 2.5f * v * v - 4 * v + 2;
    return 0;
}
inline float lanczos(float x) {
    if (x == 0) return 1; float a = 3; x = (float)fabs(x);
    if (x >= a) return 0;
    return (float)(a * sin(M_PI * x) * sin(M_PI * x / a) / (M_PI * M_PI * x * x));
}
}  // namespace

extern "C" {

void* img_crop(void* p, int x, int y, int w, int h) {
    Image* im = (Image*)p; if (!im || w <= 0 || h <= 0) return nullptr;
    Image* o = new Image(); o->w = w; o->h = h; o->n = im->n; o->loop = im->loop; o->mode = im->mode;
    o->frames.resize(im->n);
    for (int f = 0; f < im->n; ++f) {
        const Frame& s = im->frames[f]; Frame& d = o->frames[f];
        d.bpp = s.bpp; d.delay = s.delay; d.transparent = s.transparent; d.palette = s.palette; d.hasPalette = s.hasPalette;
        int b = s.bpp; d.data.assign((size_t)w * h * b, 0);
        for (int yy = 0; yy < h; ++yy) for (int xx = 0; xx < w; ++xx) {
            int sx = x + xx, sy = y + yy;
            if (sx < 0 || sx >= im->w || sy < 0 || sy >= im->h) continue;
            for (int c = 0; c < b; ++c) d.data[((size_t)yy * w + xx) * b + c] = s.data[((size_t)sy * im->w + sx) * b + c];
        }
    }
    return o;
}

void* img_resize(void* p, int nw, int nh, const char* resample) {
    Image* im = (Image*)p; if (!im || nw <= 0 || nh <= 0) return nullptr;
    int method = 1;
    if (resample) { std::string s = resample; if (s == "nearest") method = 0; else if (s == "bicubic") method = 2; else if (s == "lanczos") method = 3; else method = 1; }
    Image* rg = as_rgba(im);
    Image* o = new_rgba(nw, nh, rg->n); o->loop = rg->loop;
    for (int f = 0; f < rg->n; ++f) {
        o->frames[f].delay = rg->frames[f].delay;
        uint8_t* sp = rg->frames[f].data.data(); uint8_t* dp = o->frames[f].data.data();
        float sx = (float)rg->w / nw, sy = (float)rg->h / nh;
        for (int y = 0; y < nh; ++y) for (int x = 0; x < nw; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f, fy = (y + 0.5f) * sy - 0.5f;
            uint8_t* d = dp + ((size_t)y * nw + x) * 4;
            if (method == 0) {
                int ix = clampi((int)floor(fx + 0.5f), 0, rg->w - 1), iy = clampi((int)floor(fy + 0.5f), 0, rg->h - 1);
                memcpy(d, sp + ((size_t)iy * rg->w + ix) * 4, 4);
            } else if (method == 1) {
                int x0 = (int)floor(fx), y0 = (int)floor(fy); float tx = fx - x0, ty = fy - y0;
                for (int c = 0; c < 4; ++c) {
                    float v = 0;
                    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
                        int xx = clampi(x0 + i, 0, rg->w - 1), yy = clampi(y0 + j, 0, rg->h - 1);
                        v += (i ? tx : 1 - tx) * (j ? ty : 1 - ty) * sp[((size_t)yy * rg->w + xx) * 4 + c];
                    }
                    d[c] = clamp8(v);
                }
            } else {
                float (*kern)(float) = (method == 2) ? cubic : lanczos;
                int rad = (method == 2) ? 2 : 3, x0 = (int)floor(fx), y0 = (int)floor(fy);
                for (int c = 0; c < 4; ++c) {
                    float sum = 0, wsum = 0;
                    for (int j = -rad + 1; j <= rad; ++j) for (int i = -rad + 1; i <= rad; ++i) {
                        int xx = clampi(x0 + i, 0, rg->w - 1), yy = clampi(y0 + j, 0, rg->h - 1);
                        float wt = kern(fx - (x0 + i)) * kern(fy - (y0 + j));
                        sum += wt * sp[((size_t)yy * rg->w + xx) * 4 + c]; wsum += wt;
                    }
                    d[c] = clamp8(wsum != 0 ? sum / wsum : sum);
                }
            }
        }
    }
    if (rg != im) delete rg;
    return o;
}

void* img_thumbnail(void* p, int w, int h) {
    Image* im = (Image*)p; if (!im) return nullptr;
    float s = std::min((float)w / im->w, (float)h / im->h); if (s > 1) s = 1;
    int nw = std::max(1, (int)(im->w * s)), nh = std::max(1, (int)(im->h * s));
    Image* r = (Image*)img_resize(im, nw, nh, "lanczos");
    *im = *r; delete r; return im;
}

void* img_scale(void* p, float fx, float fy, const char* resample) {
    Image* im = (Image*)p; if (!im) return nullptr;
    return img_resize(im, std::max(1, (int)(im->w * fx)), std::max(1, (int)(im->h * fy)), resample);
}

void* img_transpose(void* p, int m) {
    Image* im = (Image*)p; if (!im) return nullptr;
    bool swap = (m == 2 || m == 4 || m == 5 || m == 6);
    int nw = swap ? im->h : im->w, nh = swap ? im->w : im->h;
    Image* o = new Image(); o->w = nw; o->h = nh; o->n = im->n; o->loop = im->loop; o->mode = im->mode;
    o->frames.resize(im->n);
    for (int f = 0; f < im->n; ++f) {
        const Frame& s = im->frames[f]; Frame& d = o->frames[f];
        d.bpp = s.bpp; d.delay = s.delay; d.transparent = s.transparent; d.palette = s.palette; d.hasPalette = s.hasPalette;
        int b = s.bpp; d.data.assign((size_t)nw * nh * b, 0);
        for (int y = 0; y < im->h; ++y) for (int x = 0; x < im->w; ++x) {
            int nx, ny;
            switch (m) {
                case 0: nx = im->w - 1 - x; ny = y; break;
                case 1: nx = x; ny = im->h - 1 - y; break;
                case 2: nx = y; ny = im->w - 1 - x; break;
                case 3: nx = im->w - 1 - x; ny = im->h - 1 - y; break;
                case 4: nx = im->h - 1 - y; ny = x; break;
                case 5: nx = y; ny = x; break;
                default: nx = im->h - 1 - y; ny = im->w - 1 - x; break;
            }
            for (int c = 0; c < b; ++c) d.data[((size_t)ny * nw + nx) * b + c] = s.data[((size_t)y * im->w + x) * b + c];
        }
    }
    return o;
}

void* img_mirror(void* p) { return img_transpose(p, 0); }

void* img_pad(void* p, int l, int t, int r, int b, int rgba) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* rg = as_rgba(im);
    Image* o = new_rgba(rg->w + l + r, rg->h + t + b, rg->n); o->loop = rg->loop;
    img_fill(o, rgba);
    for (int f = 0; f < rg->n; ++f) {
        o->frames[f].delay = rg->frames[f].delay;
        uint8_t* sp = rg->frames[f].data.data(); uint8_t* dp = o->frames[f].data.data();
        for (int y = 0; y < rg->h; ++y) for (int x = 0; x < rg->w; ++x)
            memcpy(dp + ((size_t)(y + t) * o->w + (x + l)) * 4, sp + ((size_t)y * rg->w + x) * 4, 4);
    }
    if (rg != im) delete rg;
    return o;
}

void* img_rotate(void* p, float deg, int expand, int fill_rgba) {
    Image* im = (Image*)p; if (!im) return nullptr;
    Image* rg = as_rgba(im);
    double rad = deg * M_PI / 180.0, cs = cos(rad), sn = sin(rad);
    int W = rg->w, H = rg->h, nw = W, nh = H;
    if (expand) { nw = (int)ceil(fabs(W * cs) + fabs(H * sn)); nh = (int)ceil(fabs(W * sn) + fabs(H * cs)); }
    Image* o = new_rgba(nw, nh, rg->n); img_fill(o, fill_rgba);
    double cx = W / 2.0, cy = H / 2.0, ncx = nw / 2.0, ncy = nh / 2.0;
    for (int f = 0; f < rg->n; ++f) {
        o->frames[f].delay = rg->frames[f].delay;
        uint8_t* sp = rg->frames[f].data.data(); uint8_t* dp = o->frames[f].data.data();
        for (int y = 0; y < nh; ++y) for (int x = 0; x < nw; ++x) {
            double dx = x - ncx, dy = y - ncy;
            double sx = cs * dx + sn * dy + cx, sy = -sn * dx + cs * dy + cy;
            if (sx < -1 || sy < -1 || sx > W || sy > H) continue;
            int ix = (int)floor(sx), iy = (int)floor(sy); double tx = sx - ix, ty = sy - iy;
            uint8_t* d = dp + ((size_t)y * nw + x) * 4;
            for (int c = 0; c < 4; ++c) {
                float v = 0;
                for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
                    int xx = clampi(ix + i, 0, W - 1), yy = clampi(iy + j, 0, H - 1);
                    v += (i ? tx : 1 - tx) * (j ? ty : 1 - ty) * sp[((size_t)yy * W + xx) * 4 + c];
                }
                d[c] = clamp8(v);
            }
        }
    }
    if (rg != im) delete rg;
    return o;
}

static void paste_into(Image* base, Image* src, int x, int y, Image* mask) {
    if (base->frames[0].bpp != 4) { Image* t = to_rgba(base); *base = *t; delete t; }
    Image* s = (src->frames[0].bpp == 4) ? src : to_rgba(src);
    for (int f = 0; f < base->n && f < s->n; ++f) {
        uint8_t* dp = base->frames[f].data.data(); uint8_t* sp2 = s->frames[f].data.data();
        uint8_t* mp = mask ? mask->frames[0].data.data() : nullptr;
        for (int sy = 0; sy < s->h; ++sy) for (int sx = 0; sx < s->w; ++sx) {
            int bx = x + sx, by = y + sy;
            if (bx < 0 || bx >= base->w || by < 0 || by >= base->h) continue;
            uint8_t* d = dp + ((size_t)by * base->w + bx) * 4;
            uint8_t* spr = sp2 + ((size_t)sy * s->w + sx) * 4;
            if (mp && mask->w == s->w && mask->h == s->h) {
                float a = mp[((size_t)sy * s->w + sx) * 4] / 255.0f;
                for (int c = 0; c < 4; ++c) d[c] = clamp8(d[c] * (1 - a) + spr[c] * a);
            } else memcpy(d, spr, 4);
        }
    }
    if (s != src) delete s;
}

void* img_paste(void* bp, void* sp, int x, int y) {
    Image* base = (Image*)bp; Image* src = (Image*)sp;
    if (!base || !src) return nullptr; paste_into(base, src, x, y, nullptr); return base;
}
void* img_paste_mask(void* bp, void* sp, int x, int y, void* mp) {
    Image* base = (Image*)bp; Image* src = (Image*)sp; Image* mask = (Image*)mp;
    if (!base || !src) return nullptr; paste_into(base, src, x, y, mask); return base;
}

void* img_composite(void* bp, void* op) {
    Image* base = (Image*)bp; Image* over = (Image*)op;
    if (!base || !over || base->w != over->w || base->h != over->h) return nullptr;
    if (base->frames[0].bpp != 4) { Image* t = to_rgba(base); *base = *t; delete t; }
    Image* o = to_rgba(over);
    for (int f = 0; f < base->n; ++f) {
        uint8_t* a = base->frames[f].data.data(); uint8_t* b = o->frames[f].data.data();
        for (int i = 0; i < base->w * base->h; ++i) {
            float sa = b[i * 4 + 3] / 255.0f, da = a[i * 4 + 3] / 255.0f;
            float oa = sa + da * (1 - sa);
            if (oa <= 0) { a[i * 4 + 3] = 0; continue; }
            for (int c = 0; c < 3; ++c) a[i * 4 + c] = clamp8((b[i * 4 + c] * sa + a[i * 4 + c] * da * (1 - sa)) / oa);
            a[i * 4 + 3] = clamp8(oa * 255);
        }
    }
    delete o; return base;
}

void* img_blend(void* ap, void* bp, float alpha) {
    Image* a = (Image*)ap; Image* b = (Image*)bp;
    if (!a || !b || a->w != b->w || a->h != b->h) return nullptr;
    Image* ra = to_rgba(a); Image* rb = to_rgba(b);
    for (int f = 0; f < ra->n; ++f) {
        uint8_t* pa = ra->frames[f].data.data(); uint8_t* pb = rb->frames[f].data.data();
        for (size_t i = 0; i < ra->frames[f].data.size(); ++i) pa[i] = clamp8(pa[i] * (1 - alpha) + pb[i] * alpha);
    }
    delete rb; return ra;
}

int img_getbbox4(void* p, int rgba, char* out, int outlen) {
    Image* im = (Image*)p; if (!im) return -1;
    Image* rg = as_rgba(im);
    int x0 = im->w, y0 = im->h, x1 = -1, y1 = -1;
    for (int y = 0; y < rg->h; ++y) for (int x = 0; x < rg->w; ++x) {
        uint8_t* q = rg->frames[0].data.data() + ((size_t)y * rg->w + x) * 4;
        int v = pack_rgba(q[0], q[1], q[2], q[3]);
        if (v != rgba) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
    }
    if (rg != im) delete rg;
    if (x1 < 0) { snprintf(out, outlen, "None"); return 0; }
    snprintf(out, outlen, "%d,%d,%d,%d", x0, y0, x1 + 1, y1 + 1);
    return 0;
}
void* img_getbbox(void* p, int rgba) { (void)p; (void)rgba; return nullptr; }

}  // extern "C"
