# sd_runtime/nn.static.py — 神经网络函数式工具
# 1:1 对齐 comfyui_ref/comfy/ldm/modules/diffusionmodules/util.py
#
# 依赖: libtorch_std_helper 需新增 torch_std_arange / torch_std_cos / torch_std_sin

from ops import *

import math

# ==============================================================================
# 额外 extern fn（需添加到 libtorch_std_helper）
# ==============================================================================

extern fn torch_std_arange(start: int, end: int, step: int, dtype: int) -> ptr from "torch_std_helper"
extern fn torch_std_cos(a: ptr) -> ptr from "torch_std_helper"
extern fn torch_std_sin(a: ptr) -> ptr from "torch_std_helper"

# ==============================================================================
# Timestep Embedding (util.py:229)
# ==============================================================================

def timestep_embedding(timesteps: ptr, dim: int, max_period: float) -> ptr:
    """Create sinusoidal timestep embeddings.
    
    对应 comfy/ldm/modules/diffusionmodules/util.py timestep_embedding()
    
    :param timesteps: 1-D float32 ptr [N], one per batch element
    :param dim: dimension of output
    :param max_period: controls minimum frequency (default 10000)
    :return: ptr [N x dim]
    
    PyTorch reference:
        half = dim // 2
        freqs = exp(-log(max_period) * arange(0, half, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = cat([cos(args), sin(args)], dim=-1)
    """
    half: int = dim // 2
    
    # arange(0, half, 1, float32) -> (half,)
    idx: ptr = torch_std_arange(0, half, 1, DTYPE_FLOAT32)
    
    # Move to same device as timesteps
    if torch_std_is_cuda(timesteps):
        idx = torch_std_to_cuda(idx)
    
    # freqs = exp(-log(max_period) * idx / half)
    neg_log: float = -math.log(max_period)
    freqs: ptr = torch_std_mul_scalar(idx, neg_log / half)
    freqs = torch_std_exp(freqs)
    
    # timesteps[:, None] -> (N, 1)
    # freqs[None] -> (1, half)
    ts_2d: ptr = torch_std_unsqueeze(timesteps, 1)   # (N, 1)
    freqs_2d: ptr = torch_std_unsqueeze(freqs, 0)     # (1, half)
    
    # args = ts[:, None] * freqs[None]  (broadcast)
    args: ptr = torch_std_mul(ts_2d, freqs_2d)         # (N, half)
    
    # embedding = cat([cos(args), sin(args)], dim=-1)
    cos_args: ptr = torch_std_cos(args)
    sin_args: ptr = torch_std_sin(args)
    
    tensors: ptr = make_ptr_array(2)
    ptr_array_set(tensors, 0, cos_args)
    ptr_array_set(tensors, 1, sin_args)
    embedding: ptr = torch_std_cat(tensors, 2, -1)     # (N, half*2)
    
    # If odd dim, pad one zero column
    if dim % 2 == 1:
        # slice first column, multiply by 0 to get zeros of same shape
        first_col: ptr = torch_std_slice(embedding, 1, 0, 1, 1)  # (N, 1)
        zeros: ptr = torch_std_mul_scalar(first_col, 0.0)
        tensors2: ptr = make_ptr_array(2)
        ptr_array_set(tensors2, 0, embedding)
        ptr_array_set(tensors2, 1, zeros)
        embedding = torch_std_cat(tensors2, 2, -1)
    
    return embedding


# ==============================================================================
# mean_flat (util.py:270)
# ==============================================================================

def mean_flat(tensor: ptr) -> ptr:
    """Mean over all non-batch dimensions.
    Input: (N, C, H, W)  -> Output: (N,)
    PyTorch equivalent: tensor.mean(dim=list(range(1, len(tensor.shape))))
    """
    n: int = torch_std_numel(tensor)
    nd: int = torch_std_ndim(tensor)
    if nd <= 1:
        return tensor
    
    # Get batch size from shape
    shape_buf: ptr = make_int_array(nd)
    torch_std_shape(tensor, shape_buf)
    batch: int = int_array_ref(shape_buf, 0)
    
    # Reshape to (batch, -1) then mean over dim=1
    new_shape: ptr = make_int_array(2)
    int_array_set(new_shape, 0, batch)
    int_array_set(new_shape, 1, n // batch)
    
    flat: ptr = torch_std_reshape(tensor, new_shape, 2)
    result: ptr = torch_std_mean_dim(flat, 1, 0)
    return result


# ==============================================================================
# extract_into_tensor (util.py:171)
# ==============================================================================

def extract_into_tensor(a: ptr, t: ptr, x_shape: ptr) -> ptr:
    """Gather from 'a' at indices 't' and reshape for broadcasting.
    
    PyTorch: out = a.gather(-1, t)
             return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    a: (steps,) — e.g. alphas_cumprod
    t: (batch,) — integer indices
    x_shape: shape of target tensor (used for ndim only)
    """
    # index_select along dim 0: a[t[i]] for each i in t
    result: ptr = torch_std_index_select(a, 0, t)  # (batch,)
    
    # Unsqueeze to match spatial dims of x
    # x_shape ndim tells us how many trailing dims to add
    # For UNet/VAE (4D): (batch,) -> (batch, 1, 1, 1)
    result = torch_std_unsqueeze(result, 1)  # (batch, 1)
    # Add more dims: typical usage needs (batch, 1, 1, 1)
    # We'll chain unsqueeze for standard 4D case
    result = torch_std_unsqueeze(result, 1)  # (batch, 1, 1) — actually unsqueeze at 1 again shifts
    # Simpler approach: reshape to (batch, 1, 1, 1)
    batch: int = torch_std_numel(result)
    shape_arr: ptr = make_int_array(4)
    int_array_set(shape_arr, 0, batch)
    int_array_set(shape_arr, 1, 1)
    int_array_set(shape_arr, 2, 1)
    int_array_set(shape_arr, 3, 1)
    result = torch_std_reshape(result, shape_arr, 4)
    return result


# ==============================================================================
# Sinusoidal position embedding (用于位置编码)
# ==============================================================================

def sinusoidal_embedding(dim: int, max_period: float, length: int) -> ptr:
    """Precompute sinusoidal position embeddings.
    Returns (length, dim) tensor.
    """
    # positions = arange(0, length)
    positions: ptr = torch_std_arange(0, length, 1, DTYPE_FLOAT32)  # (length,)
    positions = torch_std_unsqueeze(positions, 1)  # (length, 1)
    
    half: int = dim // 2
    idx: ptr = torch_std_arange(0, half, 1, DTYPE_FLOAT32)  # (half,)
    freqs: ptr = torch_std_mul_scalar(idx, -math.log(max_period) / half)
    freqs = torch_std_exp(freqs)
    # freqs: (half,)
    freqs = torch_std_unsqueeze(freqs, 0)  # (1, half)
    
    args: ptr = torch_std_mul(positions, freqs)  # (length, half)
    
    cos_args: ptr = torch_std_cos(args)
    sin_args: ptr = torch_std_sin(args)
    
    tensors: ptr = make_ptr_array(2)
    ptr_array_set(tensors, 0, cos_args)
    ptr_array_set(tensors, 1, sin_args)
    emb: ptr = torch_std_cat(tensors, 2, -1)  # (length, dim)
    return emb


# ==============================================================================
# make_beta_schedule (util.py:89)
# ==============================================================================

def make_beta_schedule_linear(n_timestep: int) -> ptr:
    """Linear beta schedule from linear_start to linear_end.
    Returns (n_timestep,) float32 tensor.
    
    PyTorch: betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep)**2
    """
    linear_start: float = 1e-4
    linear_end: float = 2e-2
    start_sqrt: float = math.sqrt(linear_start)
    end_sqrt: float = math.sqrt(linear_end)
    
    # linspace via arange + scale
    ar: ptr = torch_std_arange(0, n_timestep, 1, DTYPE_FLOAT32)  # (n_timestep,)
    step: float = (end_sqrt - start_sqrt) / max(n_timestep - 1, 1)
    
    betas: ptr = torch_std_mul_scalar(ar, step)
    betas = torch_std_add_scalar(betas, start_sqrt)
    betas = torch_std_pow(betas, 2.0)
    return betas
