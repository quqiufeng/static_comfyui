# torch_ops.static.py — staticpyTorch: PyTorch C API for StaticPy
# 运行时 dlopen libtorch.so，提供 PyTorch 级别的算子
# 所有函数以 st_ 为前缀（static_torch）

extern fn st_cuda_available() -> int from "staticpy_torch"
extern fn st_conv2d(input: ptr, weight: ptr, bias: ptr,
    stride_h: int, stride_w: int, pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int, groups: int) -> ptr from "staticpy_torch"
extern fn st_linear(input: ptr, weight: ptr, bias: ptr) -> ptr from "staticpy_torch"
extern fn st_group_norm(input: ptr, num_groups: int, weight: ptr, bias: ptr, eps: float) -> ptr from "staticpy_torch"
extern fn st_layer_norm(input: ptr, weight: ptr, bias: ptr, eps: float) -> ptr from "staticpy_torch"
extern fn st_mm(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_silu(input: ptr) -> ptr from "staticpy_torch"
extern fn st_relu(input: ptr) -> ptr from "staticpy_torch"
extern fn st_gelu(input: ptr) -> ptr from "staticpy_torch"
extern fn st_sigmoid(input: ptr) -> ptr from "staticpy_torch"
extern fn st_tanh(input: ptr) -> ptr from "staticpy_torch"
extern fn st_softmax(input: ptr, dim: int) -> ptr from "staticpy_torch"
extern fn st_upsample_nearest(input: ptr, scale_h: int, scale_w: int) -> ptr from "staticpy_torch"
extern fn st_sum(t: ptr) -> float from "staticpy_torch"
extern fn st_mean(t: ptr) -> float from "staticpy_torch"
extern fn st_tensor_numel(t: ptr) -> int from "staticpy_torch"
extern fn st_tensor_data(t: ptr) -> ptr from "staticpy_torch"
extern fn st_tensor_free(t: ptr) -> void from "staticpy_torch"
