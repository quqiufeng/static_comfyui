# torch.static.py — staticpyTorch 库
# 用户调 torch.conv2d(...)，不碰 C 函数
# 底层 auto 管理 tensor 生命周期 + GPU/CPU 切换

# ===== Tensor 创建 =====

def torch_tensor(data: list[float], dims: list[int]) -> ptr:
    """从 float 数组创建 tensor"""
    n: int = 1
    i: int = 0
    while i < len(dims):
        n = n * dims[i]
        i = i + 1
    # 通过 st_* 函数创建
    return t_new(data, dims)

def torch_zeros(shape: list[int]) -> ptr:
    """全零 tensor"""
    n: int = 1
    i: int = 0
    while i < len(shape):
        n = n * shape[i]
        i = i + 1
    return st_new(shape)

def torch_ones(shape: list[int]) -> ptr:
    """全一 tensor"""
    return st_ones(shape)

# ===== Tensor 属性 =====

def torch_numel(t: ptr) -> int:
    return st_tensor_numel(t)

def torch_data(t: ptr) -> list[float]:
    return st_tensor_data(t)

def torch_free(t: ptr):
    st_tensor_free(t)

# ===== GPU 管理 =====

def torch_cuda_available() -> int:
    return st_cuda_available()

def torch_to_cuda(t: ptr):
    st_tensor_to_cuda(t)

def torch_to_cpu(t: ptr):
    st_tensor_to_cpu(t)

# ===== 神经网络算子 =====

def torch_conv2d(input: ptr, weight: ptr, bias: ptr,
                  stride: int, padding: int, dilation: int, groups: int) -> ptr:
    return st_conv2d(input, weight, bias, stride, stride, padding, padding, dilation, dilation, groups)

def torch_conv2d_simple(input: ptr, weight: ptr, bias: ptr) -> ptr:
    """Conv2d 默认 k=3, s=1, p=1"""
    return st_conv2d(input, weight, bias, 1, 1, 1, 1, 1, 1, 1)

def torch_linear(input: ptr, weight: ptr, bias: ptr) -> ptr:
    return st_linear(input, weight, bias)

def torch_group_norm(input: ptr, num_groups: int, weight: ptr, bias: ptr) -> ptr:
    return st_group_norm(input, num_groups, weight, bias, 1e-6)

def torch_layer_norm(input: ptr, weight: ptr, bias: ptr) -> ptr:
    return st_layer_norm(input, weight, bias, 1e-5)

def torch_mm(a: ptr, b: ptr) -> ptr:
    return st_mm(a, b)

def torch_matmul(a: ptr, b: ptr) -> ptr:
    return st_matmul(a, b)

# ===== 激活函数 =====

def torch_silu(x: ptr) -> ptr: return st_silu(x)
def torch_relu(x: ptr) -> ptr: return st_relu(x)
def torch_gelu(x: ptr) -> ptr: return st_gelu(x)
def torch_sigmoid(x: ptr) -> ptr: return st_sigmoid(x)
def torch_tanh(x: ptr) -> ptr: return st_tanh(x)
def torch_softmax(x: ptr, dim: int) -> ptr: return st_softmax(x, dim)

# ===== 采样 =====

def torch_upsample_nearest(x: ptr, scale_h: int, scale_w: int) -> ptr:
    return st_upsample_nearest(x, scale_h, scale_w)

# ===== 逐元素 =====

def torch_add(a: ptr, b: ptr) -> ptr: return st_add_tensor(a, b)
def torch_sub(a: ptr, b: ptr) -> ptr: return st_sub_tensor(a, b)
def torch_mul(a: ptr, b: ptr) -> ptr: return st_mul_tensor(a, b)

# ===== 归约 =====

def torch_sum(t: ptr) -> float: return st_sum(t)
def torch_mean(t: ptr) -> float: return st_mean(t)

# ===== 高阶 UNet 构建块 =====

def torch_resblock(x: ptr, emb: ptr,
                    w_in1: ptr, b_in1: ptr, w_c1: ptr, b_c1: ptr,
                    w_emb: ptr, b_emb: ptr,
                    w_in2: ptr, b_in2: ptr, w_c2: ptr, b_c2: ptr,
                    w_skip: ptr, b_skip: ptr, c_out: int) -> ptr:
    """ResBlock: GN → SiLU → Conv → +temb → GN → SiLU → Conv → +skip"""
    x = torch_group_norm(x, 32, w_in1, b_in1)
    x = torch_silu(x)
    x = torch_conv2d_simple(x, w_c1, b_c1)
    
    # time embedding injection
    _scale_shift: ptr = torch_linear(emb, w_emb, b_emb)
    _c: int = c_out * 2
    _scale: ptr = st_slice(_scale_shift, 0, _c // 2)
    _shift: ptr = st_slice(_scale_shift, _c // 2, _c)
    x = torch_add(torch_mul(x, _scale), _shift)
    
    x = torch_group_norm(x, 32, w_in2, b_in2)
    x = torch_silu(x)
    x = torch_conv2d_simple(x, w_c2, b_c2)
    # skip connection handled externally
    return x
