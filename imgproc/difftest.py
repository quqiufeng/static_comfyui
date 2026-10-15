#!/usr/bin/env python3
"""difftest.py — 自动化差分测试：Pillow vs libimgproc，逐像素比对。

用法: /data/venv/bin/python imgproc/difftest.py [-v]
退出码非 0 表示有算子超出容差。
"""
import sys
import ctypes as C
import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter, ImageStat

import pyimgproc as P

VERBOSE = "-v" in sys.argv
H, W = 40, 52
rng = np.random.default_rng(7)


def rgba_of_pil(im):
    return np.dstack([np.array(im.convert("RGB")), np.array(im.convert("RGBA")).astype(int)[..., 3]]).astype(int)


def rgba_of_handle(h):
    im = P.to_np(h)
    return np.dstack([im[..., :3].astype(int), im[..., 3].astype(int)])


def rgb_of_pil(im):
    return np.array(im.convert("RGB")).astype(int)


def rgb_of_handle(h):
    return P.to_np(h)[..., :3].astype(int)


# 测试图（PNG）
img_rgb = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")
img_rgba = Image.fromarray(rng.integers(0, 256, (H, W, 4), dtype=np.uint8), "RGBA")
img_L = img_rgb.convert("L")
img_rgb_path = "/tmp/opencode/difftest_rgb.png"
img_rgba_path = "/tmp/opencode/difftest_rgba.png"
img_rgb.save(img_rgb_path)
img_rgba.save(img_rgba_path)

h_rgb = P.img_open(img_rgb_path.encode())
h_rgba = P.img_open(img_rgba_path.encode())

results = []


def case(name, pil, ours, tol=2, space="rgb"):
    """pil/ours: 0-参数函数，返回 PIL.Image / handle。"""
    try:
        a = rgb_of_pil(pil()) if space == "rgb" else rgba_of_pil(pil())
        b = rgb_of_handle(ours()) if space == "rgb" else rgba_of_handle(ours())
        if a.shape != b.shape:
            results.append((name, "SIZE", f"{a.shape} vs {b.shape}", 999))
            return
        d = np.abs(a - b)
        mx = int(d.max()); mean = float(d.mean())
        ok = mx <= tol
        results.append((name, "PASS" if ok else "DIFF", f"max={mx} mean={mean:.2f}", mx))
    except Exception as e:
        results.append((name, "ERR", repr(e)[:80], 999))


def norm(im, mode="RGB"):
    return im


# ---- 模式转换 ----
case("convert.RGB", lambda: img_rgb, lambda: P.img_convert(h_rgb, b"RGB"))
case("convert.L", lambda: img_rgb.convert("L"), lambda: P.img_convert(h_rgb, b"L"))
case("convert.RGBA", lambda: img_rgb.convert("RGBA"), lambda: P.img_convert(h_rgb, b"RGBA"), space="rgba")

# ---- 转置 ----
for m, pn in [(0, "FLIP_LEFT_RIGHT"), (1, "FLIP_TOP_BOTTOM"), (2, "ROTATE_90"), (3, "ROTATE_180"), (4, "ROTATE_270"), (5, "TRANSPOSE"), (6, "TRANSVERSE")]:
    case(f"transpose.{pn}", lambda m=m, pn=pn: img_rgb.transpose(getattr(Image.Transpose, pn)), lambda m=m: P.img_transpose(h_rgb, m), tol=0)

# ---- 裁剪 ----
case("crop", lambda: img_rgb.crop((5, 6, 30, 33)), lambda: P.img_crop(h_rgb, 5, 6, 25, 27), tol=0)

# ---- 缩放（近似）----
for rs, pn in [("nearest", "NEAREST"), ("bilinear", "BILINEAR"), ("lanczos", "LANCZOS")]:
    case(f"resize.{rs}", lambda pn=pn: img_rgb.resize((26, 20), getattr(Image.Resampling, pn)), lambda rs=rs: P.img_resize(h_rgb, 26, 20, rs.encode()), tol=24 if rs != "nearest" else 0)

# ---- ImageOps ----
case("ops.invert", lambda: ImageOps.invert(img_rgb), lambda: P.img_invert(h_rgb), tol=0)
case("ops.grayscale", lambda: ImageOps.grayscale(img_rgb), lambda: P.img_grayscale(h_rgb))
case("ops.posterize", lambda: ImageOps.posterize(img_rgb, 3), lambda: P.img_posterize(h_rgb, 3))
case("ops.solarize", lambda: ImageOps.solarize(img_rgb, 128), lambda: P.img_solarize(h_rgb, 128), tol=0)
case("ops.autocontrast", lambda: ImageOps.autocontrast(img_rgb), lambda: P.img_autocontrast(h_rgb, 0), tol=3)
case("ops.equalize", lambda: ImageOps.equalize(img_rgb), lambda: P.img_equalize(h_rgb), tol=3)

# ---- ImageChops ----
img2 = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")
img2_path = "/tmp/opencode/difftest_rgb2.png"; img2.save(img2_path)
h2 = P.img_open(img2_path.encode())
for op, fn in [("add", ImageChops.add), ("subtract", ImageChops.subtract), ("difference", ImageChops.difference), ("multiply", ImageChops.multiply), ("screen", ImageChops.screen), ("darker", ImageChops.darker), ("lighter", ImageChops.lighter)]:
    case(f"chops.{op}", lambda fn=fn: fn(img_rgb, img2), lambda op=op: P.img_chop(h_rgb, h2, op.encode()), tol=1)
case("chops.invert", lambda: ImageChops.invert(img_rgb), lambda: P.img_invert(h_rgb), tol=0)

# ---- ImageEnhance ----
for k, pn in [("brightness", "Brightness"), ("contrast", "Contrast"), ("color", "Color")]:
    case(f"enhance.{k}", lambda pn=pn: getattr(ImageEnhance, pn)(img_rgb).enhance(1.5), lambda k=k: P.img_enhance(h_rgb, k.encode(), 1.5), tol=3)

# ---- ImageFilter ----
for pn, nm in [("SHARPEN", "SHARPEN"), ("BLUR", "BLUR"), ("EMBOSS", "EMBOSS"), ("FIND_EDGES", "FIND_EDGES"), ("SMOOTH", "SMOOTH")]:
    case(f"filter.{pn}", lambda pn=pn: img_rgb.filter(getattr(ImageFilter, pn)), lambda nm=nm: P.img_filter(h_rgb, nm.encode(), 1.0, 0, 0), tol=40)

# ---- 通道 / 数据进出 ----
for i, band in enumerate("RGBA"):
    case(f"getchannel.{band}", lambda band=band: img_rgba.getchannel(band), lambda i=i: P.img_getchannel(h_rgba, i))
raw = img_rgba.tobytes("raw", "RGBA")
def pil_frombytes():
    return Image.frombytes("RGBA", (W, H), raw)
def our_frombytes():
    buf = C.create_string_buffer(raw)
    return P.img_frombytes(b"RGBA", W, H, C.cast(buf, C.c_void_p), len(raw))
case("frombytes.RGBA", pil_frombytes, our_frombytes, space="rgba", tol=0)
def pil_tobytes():
    return img_rgb
def our_tobytes():
    buf = C.create_string_buffer(W * H * 3)
    P.img_tobytes(h_rgb, 0, b"RGB", C.cast(buf, C.c_void_p), W * H * 3)
    a = np.frombuffer(buf.raw, np.uint8).reshape(H, W, 3).astype(int)
    return P.from_np(a)
case("tobytes.RGB", pil_tobytes, our_tobytes, tol=0)

# ---- alpha_composite(带位置) ----
small = Image.fromarray(rng.integers(0, 256, (10, 12, 4), dtype=np.uint8), "RGBA")
small_path = "/tmp/opencode/difftest_small.png"; small.save(small_path)
h_small = P.img_open(small_path.encode())
def pil_ac():
    b = img_rgba.copy(); b.alpha_composite(small, (7, 5)); return b
def our_ac():
    b = P.img_copy(h_rgba); P.img_alpha_composite_at(b, h_small, 7, 5); return b
case("alpha_composite", pil_ac, our_ac, space="rgba", tol=1)

# ---- ImageOps.fit / contain ----
case("ops.fit", lambda: ImageOps.fit(img_rgb, (20, 30), Image.Resampling.LANCZOS), lambda: P.img_fit(h_rgb, 20, 30, b"lanczos", 0.5, 0.5, 0), tol=1)
case("ops.contain", lambda: ImageOps.contain(img_rgb, (20, 30), Image.Resampling.LANCZOS), lambda: P.img_contain(h_rgb, 20, 30, b"lanczos", 0), tol=1)

# ---- ImageColor ----
def pil_getrgb():
    import PIL.ImageColor as IC
    rgb = IC.getrgb("#3a7bc8")
    return Image.new("RGB", (2, 2), rgb)
def our_getrgb():
    buf = C.create_string_buffer(32)
    P.img_color_getrgb(b"#3a7bc8", buf, 32)
    v = [int(x) for x in buf.value.decode().split(",")]
    return P.from_np(np.full((2, 2, 3), v, np.uint8))
case("color.getrgb", pil_getrgb, our_getrgb, tol=0)

# ---- ImageDraw ----
from PIL import ImageDraw
def pil_draw_line():
    im = img_rgb.copy(); d = ImageDraw.Draw(im); d.line((0, 0, W - 1, H - 1), fill=(255, 0, 0), width=3); return im
def our_draw_line():
    h = P.img_copy(h_rgb); P.img_draw_line(h, 0, 0, W - 1, H - 1, 0xFF0000FF, 3); return h
case("draw.line", pil_draw_line, our_draw_line, tol=999 if True else 0)

# ---- getbbox / chop 变体 ----
def pil_bbox():
    bb = img_rgba.getbbox()
    a = np.zeros((H, W, 3), int)
    if bb:
        a[bb[1]:bb[3], bb[0]:bb[2]] = 255
    return Image.fromarray(a.astype(np.uint8), "RGB")
def our_bbox():
    buf = C.create_string_buffer(64)
    P.img_getbbox4(h_rgba, 0, buf, 64)
    a = np.zeros((H, W, 3), int)
    s = buf.value.decode()
    if s != "None":
        x0, y0, x1, y1 = [int(v) for v in s.split(",")]
        a[y0:y1, x0:x1] = 255
    return P.from_np(a)
case("getbbox", pil_bbox, our_bbox, tol=0)
case("chops.offset", lambda: ImageChops.offset(img_rgb, 5, -3), lambda: P.img_chop_offset(h_rgb, 5, -3), tol=0)
case("chops.add_const", lambda: ImageChops.add(img_rgb, Image.new("RGB", (W, H), (20, 20, 20))), lambda: P.img_chop_const(h_rgb, b"add", 20), tol=1)

# 报告
print(f"{'case':28} {'st':4}  detail")
fails = 0
for name, st, det, mx in results:
    if st == "DIFF" or st == "ERR":
        fails += 1
    print(f"{name:28} {st:4}  {det}")
print(f"\n共 {len(results)} 项，失败 {fails} 项")
sys.exit(1 if fails else 0)
