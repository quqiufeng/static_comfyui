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
    print("linear only test")
    data: list[float] = load_weights("/tmp/sd_weights/sdxl_merged_f32/weights.bin", 2567463684)
    print("weights loaded")
    n: int = 1
    emb: list[float] = make_float_array(n*1280)
    arr_fill(emb, 0.5, n*1280)
    print("emb filled")
    w1: list[float] = w_slice(data,12160,409600)
    print("w1 sliced")
    b1: list[float] = w_slice(data,11840,320)
    print("b1 sliced")
    y: list[float] = linear_torch(emb, w1, b1, n, 1280, 640)
    print("linear returned")
    print("linear returned")
    print("y[0]="); print(float_array_ref(y,0))

main()
