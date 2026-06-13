# test_conv.static.py — 验证 conv2d_torch / group_norm_torch / linear_torch 能跑
extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"
extern fn exit_program(code: int) -> void from "stdlib"

def load_weights(path: str, n: int) -> list[float]:
    data: list[float] = make_float_array(n)
    fp = fopen(path, "rb")
    if fp == 0:
        print("failed to open weights")
        exit_program(1)
    fread(data, 4, n, fp)
    fclose(fp)
    return data

def w_slice(data: list[float], offset: int, n: int) -> list[float]:
    r: list[float] = make_float_array(n)
    i: int = 0
    while i < n:
        float_array_set(r, i, float_array_ref(data, offset + i))
        i = i + 1
    return r

def main():
    print("test conv start")
    data: list[float] = load_weights("/tmp/sd_weights/sdxl_merged_f32/weights.bin", 2567463684)
    print("weights loaded")

    n: int = 1; hh: int = 64; ww: int = 64
    latent: list[float] = make_float_array(n * 4 * hh * ww)
    i: int = 0
    while i < n * 4 * hh * ww:
        float_array_set(latent, i, 0.1)
        i = i + 1

    print("input conv")
    h_cur: list[float] = conv2d_torch(latent, w_slice(data,320,11520), w_slice(data,0,320), n, 4, 320, hh, ww, 3, 1, 1)
    print("input conv done sum="); print(arr_sum(h_cur, n*320*hh*ww))

    print("group norm")
    group_norm_torch(h_cur, w_slice(data,422080,320), w_slice(data,421760,320), 32, 320, hh, ww)
    print("gn done sum="); print(arr_sum(h_cur, n*320*hh*ww))

    print("silu")
    silu_torch(h_cur, n*320*hh*ww)
    print("silu done sum="); print(arr_sum(h_cur, n*320*hh*ww))

    print("second conv")
    h_cur = conv2d_torch(h_cur, w_slice(data,422720,921600), w_slice(data,422400,320), n, 320, 320, hh, ww, 3, 1, 1)
    print("second conv done sum="); print(arr_sum(h_cur, n*320*hh*ww))

    print("linear")
    emb: list[float] = make_float_array(n*1280)
    arr_fill(emb, 0.5, n*1280)
    y: list[float] = linear_torch(emb, w_slice(data,12160,409600), w_slice(data,11840,320), n, 1280, 320*2)
    print("y[0]="); print(float_array_ref(y,0))
    print("y[1]="); print(float_array_ref(y,1))
    print("y[100]="); print(float_array_ref(y,100))
    print("y[200]="); print(float_array_ref(y,200))
    print("y[400]="); print(float_array_ref(y,400))
    print("y[600]="); print(float_array_ref(y,600))
    print("linear done sum="); print(arr_sum(y, n*640))

    print("test conv done")
    exit_program(0)

main()
