extern fn fopen(path: str, mode: str) -> ptr from "libc"
extern fn fread(buf: ptr, size: int, count: int, fp: ptr) -> int from "libc"
extern fn fclose(fp: ptr) -> int from "libc"
extern fn exit_program(code: int) -> void from "stdlib"

def main():
    print("load big weights")
    data: list[float] = make_float_array(2567463684)
    fp = fopen("/tmp/sd_weights/sdxl_merged_f32/weights.bin", "rb")
    fread(data, 4, 2567463684, fp)
    fclose(fp)
    print("loaded")
    a: list[float] = make_float_array(1280)
    arr_fill(a, 0.5, 1280)
    print("small array filled")
    t: ptr = st_from_blob_2d(a, 1, 1280)
    print("from_blob ok")
    st_tensor_free(t)
    print("done")
    exit_program(0)
main()
