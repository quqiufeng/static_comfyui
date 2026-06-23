# sd_runtime/ops.static.py — 张量基元层
# 1:1 对齐 comfyui_ref/comfy/ops.py
# 所有 extern fn 映射到 /opt/ReScheme/libtorch_std_helper.h 的 extern "C" API
#
# 使用方式:
#   from ops import torch_std_linear, torch_std_conv2d, ...

# ==============================================================================
# Tensor 创建
# ==============================================================================

extern fn torch_std_tensor_from_blob(data: ptr, shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_zeros(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_ones(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_empty(shape: ptr, ndim: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_full(shape: ptr, ndim: int, value: float, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_clone(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_to_dtype(t: ptr, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_delete_tensor(t: ptr) -> void from "torch_std_helper"
extern fn torch_std_arange(start: int, end: int, step: int, dtype: int) -> ptr from "torch_std_helper"
 
# 数据类型常量
DTYPE_FLOAT32: int = 0
DTYPE_FLOAT64: int = 1
DTYPE_INT32:   int = 2
DTYPE_INT64:   int = 3

# ==============================================================================
# Tensor 元数据
# ==============================================================================

extern fn torch_std_numel(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_ndim(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_shape(t: ptr, out: ptr) -> void from "torch_std_helper"

# ==============================================================================
# Tensor 设备
# ==============================================================================

extern fn torch_std_cuda_is_available() -> int from "torch_std_helper"
extern fn torch_std_to_cuda(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_to_cpu(t: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_is_cuda(t: ptr) -> int from "torch_std_helper"
extern fn torch_std_manual_seed(seed: int) -> void from "torch_std_helper"

# ==============================================================================
# 数学运算 （逐元素）
# ==============================================================================

extern fn torch_std_add(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sub(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_mul(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_div(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_pow(a: ptr, exp: float) -> ptr from "torch_std_helper"
extern fn torch_std_exp(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_log(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sqrt(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_neg(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_abs(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_cos(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sin(a: ptr) -> ptr from "torch_std_helper"
 
# ==============================================================================
# 激活函数
# ==============================================================================

extern fn torch_std_relu(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_leaky_relu(a: ptr, negative_slope: float) -> ptr from "torch_std_helper"
extern fn torch_std_sigmoid(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_tanh(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_softmax(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_log_softmax(a: ptr, dim: int) -> ptr from "torch_std_helper"

# SiLU / GELU 需要手动实现或通过已有 ops 组合
# libtorch_std_helper 没有直接提供 silu/gelu
# 用 sigmoid 组合实现: silu(x) = x * sigmoid(x)

def torch_std_silu(a: ptr) -> ptr:
    """SiLU / Swish: x * sigmoid(x)"""
    s: ptr = torch_std_sigmoid(a)
    return torch_std_mul(a, s)

def torch_std_gelu(a: ptr) -> ptr:
    """GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    sqrt_2_over_pi: float = 0.7978845608028654  # sqrt(2/pi)
    a_cubed: ptr = torch_std_pow(a, 3.0)
    inner: ptr = torch_std_add(a, torch_std_mul_scalar(a_cubed, 0.044715))
    inner = torch_std_mul_scalar(inner, sqrt_2_over_pi)
    tanh_inner: ptr = torch_std_tanh(inner)
    one_plus_tanh: ptr = torch_std_add_scalar(tanh_inner, 1.0)
    result: ptr = torch_std_mul(torch_std_mul_scalar(a, 0.5), one_plus_tanh)
    return result

# ==============================================================================
# 归约
# ==============================================================================

extern fn torch_std_sum(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sum_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_mean(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_mean_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_max(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_max_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"
extern fn torch_std_min(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_min_dim(a: ptr, dim: int, keepdim: int) -> ptr from "torch_std_helper"

# ==============================================================================
# 索引与采样
# ==============================================================================

extern fn torch_std_argmax(a: ptr) -> int from "torch_std_helper"
extern fn torch_std_argmax_dim1(a: ptr, dim: int) -> int from "torch_std_helper"
extern fn torch_std_multinomial(probs: ptr, num_samples: int, replacement: int) -> ptr from "torch_std_helper"
extern fn torch_std_gather(input: ptr, dim: int, index: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_index_select(input: ptr, dim: int, index: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_index_tensor(input: ptr, index: ptr) -> ptr from "torch_std_helper"

# ==============================================================================
# 形状操作
# ==============================================================================

extern fn torch_std_reshape(a: ptr, shape: ptr, ndim: int) -> ptr from "torch_std_helper"
extern fn torch_std_transpose(a: ptr, dim0: int, dim1: int) -> ptr from "torch_std_helper"
extern fn torch_std_permute(a: ptr, dims: ptr, ndim: int) -> ptr from "torch_std_helper"
extern fn torch_std_squeeze(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_unsqueeze(a: ptr, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_cat(tensors: ptr, n: int, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_stack(tensors: ptr, n: int, dim: int) -> ptr from "torch_std_helper"
extern fn torch_std_slice(a: ptr, dim: int, start: int, end: int, step: int) -> ptr from "torch_std_helper"

# ==============================================================================
# 神经网络层
# ==============================================================================

extern fn torch_std_matmul(a: ptr, b: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_linear(input: ptr, weight: ptr, bias: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_attention(q: ptr, k: ptr, v: ptr, heads: int, mask: ptr, skip_reshape: int) -> ptr from "torch_std_helper"
extern fn torch_std_conv1d(input: ptr, weight: ptr, bias: ptr,
    stride: int, padding: int, dilation: int, groups: int) -> ptr from "torch_std_helper"
extern fn torch_std_conv2d(input: ptr, weight: ptr, bias: ptr,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int,
    groups: int) -> ptr from "torch_std_helper"
extern fn torch_std_max_pool2d(input: ptr, kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    dilation_h: int, dilation_w: int) -> ptr from "torch_std_helper"
extern fn torch_std_avg_pool2d(input: ptr, kernel_h: int, kernel_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int) -> ptr from "torch_std_helper"
extern fn torch_std_batch_norm2d(input: ptr, weight: ptr, bias: ptr,
    running_mean: ptr, running_var: ptr,
    training: int, momentum: float, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_layer_norm(input: ptr, weight: ptr, bias: ptr, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_rms_norm(input: ptr, weight: ptr, eps: float) -> ptr from "torch_std_helper"
extern fn torch_std_group_norm(input: ptr, weight: ptr, bias: ptr, num_groups: int, eps: float) -> ptr from "torch_std_helper"
 
# ==============================================================================
# 标量运算辅助
# ==============================================================================

extern fn torch_std_mul_scalar(t: ptr, s: float) -> ptr from "torch_std_helper"
extern fn torch_std_add_scalar(t: ptr, s: float) -> ptr from "torch_std_helper"
