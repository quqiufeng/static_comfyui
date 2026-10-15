# mirror_gif.static.py — 把 GIF 逐帧水平镜像（PIL-free）
#
# 用法: mirror_gif IN.gif OUT.gif
#
# 图像编解码交给运行时加载的 C++ 底层库 libimgproc.so（内部动态链接 libgif.so.7）；
# 「镜像」这一图像处理逻辑本身由本 StaticPy 程序在原始像素缓冲上完成——
# 即此前 mirror_gif.py（PIL）的等价重写。

extern fn img_open(path: str) -> ptr from "imgproc"
extern fn img_clone_blank(src: ptr) -> ptr from "imgproc"
extern fn img_frame_ptr(h: ptr, i: int) -> ptr from "imgproc"
extern fn img_frame_bpp(h: ptr, i: int) -> int from "imgproc"
extern fn img_width(h: ptr) -> int from "imgproc"
extern fn img_height(h: ptr) -> int from "imgproc"
extern fn img_count(h: ptr) -> int from "imgproc"
extern fn img_write(h: ptr, path: str) -> int from "imgproc"
extern fn img_u8_get(p: ptr, off: int) -> int from "imgproc"
extern fn img_u8_set(p: ptr, off: int, v: int) -> int from "imgproc"
extern fn img_free(h: ptr) -> int from "imgproc"


def main():
    args = list_to_py_list(argv())
    if py_list_length(args) < 3:
        print("usage: mirror_gif IN.gif OUT.gif")
        exit_program(1)
    fin: str = py_list_ref(args, 1)
    fout: str = py_list_ref(args, 2)

    src = img_open(fin)
    if src is None:
        print("imgproc: cannot open " + fin)
        exit_program(1)

    w: int = img_width(src)
    h: int = img_height(src)
    n: int = img_count(src)
    dst = img_clone_blank(src)
    if dst is None:
        print("imgproc: clone failed")
        exit_program(1)

    i: int = 0
    while i < n:
        bpp: int = img_frame_bpp(src, i)
        s = img_frame_ptr(src, i)
        d = img_frame_ptr(dst, i)
        if bpp == 1:
            y: int = 0
            while y < h:
                base: int = y * w
                x: int = 0
                while x < w:
                    img_u8_set(d, base + (w - 1 - x), img_u8_get(s, base + x))
                    x = x + 1
                y = y + 1
        else:
            y4: int = 0
            while y4 < h:
                base4: int = y4 * w * 4
                x4: int = 0
                while x4 < w:
                    so: int = base4 + x4 * 4
                    do: int = base4 + (w - 1 - x4) * 4
                    img_u8_set(d, do, img_u8_get(s, so))
                    img_u8_set(d, do + 1, img_u8_get(s, so + 1))
                    img_u8_set(d, do + 2, img_u8_get(s, so + 2))
                    if bpp == 4:
                        img_u8_set(d, do + 3, img_u8_get(s, so + 3))
                    x4 = x4 + 1
                y4 = y4 + 1
        i = i + 1

    rc: int = img_write(dst, fout)
    img_free(src)
    img_free(dst)
    if rc != 0:
        print("imgproc: write failed rc=" + string_of_int(rc))
        exit_program(1)
    print("mirrored " + fin + " -> " + fout)
