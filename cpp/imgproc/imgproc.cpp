// imgproc.cpp — 独立图像处理底层库实现
//
// 只依赖 libgif.so.7（运行时动态解析）+ 标准库。
//
// 帧模型：每个帧要么是「调色板索引图」(bpp=1) 要么是「RGBA」(bpp=4)。
// GIF 解码保留原始调色板 + 索引（逐帧可有本地色表），因此镜像等
// 逐像素操作可无损进行（不重新量化，保持与源一致的观感）。

#include "imgproc.h"
#include "gif_lib.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

namespace {

struct Frame {
    int bpp = 1;                                   // 1=索引, 4=RGBA
    int delay = 100;                               // 毫秒
    int transparent = NO_TRANSPARENT_COLOR;        // 索引（仅 bpp==1）
    std::vector<uint8_t> data;                     // w*h*bpp
    std::vector<GifColorType> palette;             // bpp==1 时的色表
    bool hasPalette = false;
};

struct Image {
    int w = 0, h = 0, n = 0;
    std::vector<Frame> frames;
};

static int pow2ceil(int n) {
    int c = 2;
    while (c < n) c <<= 1;
    return (c > 256) ? 256 : c;
}

// ---- GIF 解码（保留调色板 + 索引）----
Image* decode_gif(const char* path) {
    int err = 0;
    GifFileType* g = DGifOpenFileName(path, &err);
    if (!g) return nullptr;
    if (DGifSlurp(g) != GIF_OK) { DGifCloseFile(g, &err); return nullptr; }

    const int W = g->SWidth, H = g->SHeight, N = g->ImageCount;
    if (W <= 0 || H <= 0 || N <= 0) { DGifCloseFile(g, &err); return nullptr; }

    Image* im = new Image();
    im->w = W; im->h = H; im->n = N;
    im->frames.resize(N);

    for (int i = 0; i < N; ++i) {
        SavedImage* si = &g->SavedImages[i];
        GraphicsControlBlock gcb;
        gcb.DisposalMode = DISPOSAL_UNSPECIFIED;
        gcb.UserInputFlag = false;
        gcb.DelayTime = 0;
        gcb.TransparentColor = NO_TRANSPARENT_COLOR;
        DGifSavedExtensionToGCB(g, i, &gcb);

        Frame& fr = im->frames[i];
        fr.delay = gcb.DelayTime * 10;

        int fw = si->ImageDesc.Width  > 0 ? si->ImageDesc.Width  : W;
        int fh = si->ImageDesc.Height > 0 ? si->ImageDesc.Height : H;
        int L = si->ImageDesc.Left, T = si->ImageDesc.Top;
        ColorMapObject* cm = si->ImageDesc.ColorMap ? si->ImageDesc.ColorMap : g->SColorMap;

        bool full = (L == 0 && T == 0 && fw == W && fh == H);
        if (full && cm) {
            fr.bpp = 1;
            fr.transparent = gcb.TransparentColor;
            fr.hasPalette = true;
            int cc = cm->ColorCount;
            fr.palette.assign(cm->Colors, cm->Colors + cc);
            fr.data.assign((size_t)W * H, 0);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    fr.data[(size_t)y * W + x] = (uint8_t)si->RasterBits[y * fw + x];
        } else {
            // 非全屏帧：转到 RGBA（黑底合成），bpp=4
            fr.bpp = 4;
            fr.data.assign((size_t)W * H * 4, 0);
            const int tIdx = gcb.TransparentColor;
            for (int y = 0; y < fh; ++y) {
                for (int x = 0; x < fw; ++x) {
                    int idx = si->RasterBits[y * fw + x];
                    int px = T + y, py = L + x;
                    if (idx == tIdx || !cm || idx >= cm->ColorCount) continue;
                    if (px < 0 || px >= H || py < 0 || py >= W) continue;
                    GifColorType c = cm->Colors[idx];
                    size_t o = ((size_t)px * W + py) * 4;
                    fr.data[o] = c.Red; fr.data[o + 1] = c.Green;
                    fr.data[o + 2] = c.Blue; fr.data[o + 3] = 255;
                }
            }
        }
    }

    DGifCloseFile(g, &err);
    return im;
}

// ---- GIF 编码（bpp==1 用本地色表；bpp==4 用全局 332 量化兜底）----
int encode_gif(Image* im, const char* path) {
    const int W = im->w, H = im->h, N = im->n;
    int err = 0;
    GifFileType* g = EGifOpenFileName(path, false, &err);
    if (!g) return -1;
    EGifSetGifVersion(g, true);

    // 屏幕描述：用第一帧色表（若有）作全局色表；否则 332 灰阶占位
    std::vector<GifColorType> gpal;
    bool anyPal = false;
    for (auto& f : im->frames) if (f.bpp == 1 && f.hasPalette) { gpal = f.palette; anyPal = true; break; }
    if (!anyPal) {
        for (int r = 0; r < 8; ++r) for (int gg = 0; gg < 8; ++gg) for (int b = 0; b < 4; ++b)
            gpal.push_back({(uint8_t)(r * 36), (uint8_t)(gg * 36), (uint8_t)(b * 85)});
    }
    std::vector<GifColorType> gp = gpal;
    gp.resize(pow2ceil((int)gp.size()), {0, 0, 0});
    ColorMapObject* gcmo = GifMakeMapObject((int)gp.size(), gp.data());
    if (!gcmo) { EGifCloseFile(g, &err); return -2; }
    if (EGifPutScreenDesc(g, W, H, 8, 0, gcmo) == GIF_ERROR) {
        GifFreeMapObject(gcmo); EGifCloseFile(g, &err); return -3;
    }

    std::vector<uint8_t> row(W);
    for (int i = 0; i < N; ++i) {
        Frame& fr = im->frames[i];
        int tIdx = (fr.bpp == 1) ? fr.transparent : NO_TRANSPARENT_COLOR;
        bool hasT = (tIdx != NO_TRANSPARENT_COLOR);

        GraphicsControlBlock gcb;
        gcb.DisposalMode = DISPOSE_DO_NOT;
        gcb.UserInputFlag = false;
        gcb.DelayTime = fr.delay / 10;
        gcb.TransparentColor = hasT ? tIdx : NO_TRANSPARENT_COLOR;
        unsigned char ext[4];
        ext[0] = (unsigned char)((gcb.DisposalMode << 2) | (hasT ? 0x01 : 0));
        ext[1] = (unsigned char)(gcb.DelayTime & 0xFF);
        ext[2] = (unsigned char)((gcb.DelayTime >> 8) & 0xFF);
        ext[3] = (unsigned char)(hasT ? tIdx : 0);
        EGifPutExtension(g, GRAPHICS_EXT_FUNC_CODE, 4, ext);

        ColorMapObject* local = nullptr;
        std::vector<GifColorType> lpad;
        if (fr.bpp == 1 && fr.hasPalette && !fr.palette.empty()) {
            lpad = fr.palette;
            lpad.resize(pow2ceil((int)lpad.size()), {0, 0, 0});
            local = GifMakeMapObject((int)lpad.size(), lpad.data());
        }
        if (EGifPutImageDesc(g, 0, 0, W, H, false, local) == GIF_ERROR) {
            if (local) GifFreeMapObject(local);
            GifFreeMapObject(gcmo); EGifCloseFile(g, &err); return -4;
        }

        if (fr.bpp == 1) {
            for (int y = 0; y < H; ++y) {
                const uint8_t* src = fr.data.data() + (size_t)y * W;
                EGifPutLine(g, (GifPixelType*)src, W);
            }
        } else {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    size_t o = ((size_t)y * W + x) * 4;
                    row[x] = (uint8_t)(((fr.data[o] >> 5) << 5) |
                                       ((fr.data[o + 1] >> 5) << 2) |
                                       (fr.data[o + 2] >> 6));
                }
                EGifPutLine(g, row.data(), W);
            }
        }
        if (local) GifFreeMapObject(local);
    }

    GifFreeMapObject(gcmo);
    if (EGifCloseFile(g, &err) != GIF_OK) return -5;
    return 0;
}

}  // namespace

extern "C" {

void* img_open(const char* path) {
    if (!path) return nullptr;
    FILE* fp = fopen(path, "rb");
    if (!fp) return nullptr;
    unsigned char sig[6] = {0};
    size_t rd = fread(sig, 1, 6, fp);
    fclose(fp);
    if (rd < 6) return nullptr;
    if (memcmp(sig, "GIF87a", 6) == 0 || memcmp(sig, "GIF89a", 6) == 0)
        return (void*)decode_gif(path);
    return nullptr;
}

// 复制结构（尺寸/帧数/每帧 bpp·调色板·延时·透明），像素清零。
void* img_clone_blank(void* srcp) {
    Image* src = (Image*)srcp;
    if (!src) return nullptr;
    Image* im = new Image();
    im->w = src->w; im->h = src->h; im->n = src->n;
    im->frames.resize(src->n);
    for (int i = 0; i < src->n; ++i) {
        const Frame& s = src->frames[i];
        Frame& d = im->frames[i];
        d.bpp = s.bpp; d.delay = s.delay; d.transparent = s.transparent;
        d.palette = s.palette; d.hasPalette = s.hasPalette;
        d.data.assign((size_t)src->w * src->h * s.bpp, 0);
    }
    return (void*)im;
}

void* img_new(int w, int h, int n) {
    if (w <= 0 || h <= 0 || n <= 0) return nullptr;
    Image* im = new Image();
    im->w = w; im->h = h; im->n = n;
    im->frames.resize(n);
    for (auto& f : im->frames) { f.bpp = 1; f.data.assign((size_t)w * h, 0); }
    return (void*)im;
}

int img_width(void* hp)  { Image* im = (Image*)hp; return im ? im->w : -1; }
int img_height(void* hp) { Image* im = (Image*)hp; return im ? im->h : -1; }
int img_count(void* hp)  { Image* im = (Image*)hp; return im ? im->n : -1; }
int img_frame_bpp(void* hp, int i) {
    Image* im = (Image*)hp;
    if (!im || i < 0 || i >= im->n) return 0;
    return im->frames[i].bpp;
}

// 返回第 i 帧像素缓冲首地址（bpp 由 img_frame_bpp 给出；每像素 bpp 字节）
unsigned char* img_frame_ptr(void* hp, int i) {
    Image* im = (Image*)hp;
    if (!im || i < 0 || i >= im->n) return nullptr;
    return im->frames[i].data.data();
}
// 兼容旧名
unsigned char* img_pixels(void* hp, int i) { return img_frame_ptr(hp, i); }

int img_delay(void* hp, int i) {
    Image* im = (Image*)hp;
    if (!im || i < 0 || i >= im->n) return 0;
    return im->frames[i].delay;
}
int img_set_delay(void* hp, int i, int ms) {
    Image* im = (Image*)hp;
    if (!im || i < 0 || i >= im->n) return -1;
    im->frames[i].delay = ms;
    return 0;
}

int img_write(void* hp, const char* path) {
    Image* im = (Image*)hp;
    if (!im || !path) return -1;
    return encode_gif(im, path);
}

int img_free(void* hp) {
    Image* im = (Image*)hp;
    if (im) delete im;
    return 0;
}

unsigned int img_u32_get(void* p, long off) {
    unsigned int v; memcpy(&v, (const char*)p + off, 4); return v;
}
void img_u32_set(void* p, long off, unsigned int v) {
    memcpy((char*)p + off, &v, 4);
}
unsigned char img_u8_get(void* p, long off) {
    return *(const unsigned char*)((const char*)p + off);
}
int img_u8_set(void* p, long off, unsigned char v) {
    *((char*)p + off) = (char)v;
    return 0;
}

}  // extern "C"
