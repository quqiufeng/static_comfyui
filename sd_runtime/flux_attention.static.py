# sd_runtime/flux_attention.static.py — FLUX 位置编码 + 注意力辅助
# 1:1 对齐 comfyui_ref/comfy/ldm/flux/layers.py + math.py
#
# C++ torch_std_flux_forward 已包含完整模型前向；
# 本模块仅提供调用前需要的 RoPE 位置编码。
#
# 用法:
#   from flux_attention import embed_nd

from ops import *
from nn import timestep_embedding
import math


# ==============================================================================
# FLUX EmbedND (flux/layers.py:12) — C++ 实现
# ==============================================================================

extern fn torch_std_flux_embed_nd(ids: ptr, dim: int, theta: float,
                                   axes_dim: ptr, n_axes: int) -> ptr from "torch_std_helper"

def embed_nd(dim: int, theta: float, axes_dim: ptr, n_axes: int, ids: ptr) -> ptr:
    """Multi-axis RoPE position embedding (C++).
    
    对应 comfy/ldm/flux/layers.py EmbedND.forward()
    
    ids: (N, n_axes) float32 positions (e.g. (H*W, 3) for (t, h, w))
    dim: head dimension (typically 128)
    theta: base frequency (typically 10000)
    axes_dim: int array of per-axis dims (e.g. [64, 64, 64])
    n_axes: number of axes
    
    Returns: (1, N, sum(axes_dim[i]//2), 2, 2) position embedding
    """
    return torch_std_flux_embed_nd(ids, dim, theta, axes_dim, n_axes)


# ==============================================================================
# FLUX timestep_embedding (flux/layers.py:29)
# ==============================================================================
# 已由 nn.static.py 的 timestep_embedding 提供
# 区别：FLUX 版本多了 time_factor=1000.0
#       本函数提供 FLUX 专用版本

def flux_timestep_embedding(t: ptr, dim: int, max_period: float) -> ptr:
    """FLUX sinusoidal timestep embedding with time_factor=1000.
    
    对应 comfy/ldm/flux/layers.py timestep_embedding()
    
    t: (N,) float timesteps
    dim: output dimension
    max_period: typically 10000
    """
    # t = 1000.0 * t  (time_factor)
    t_scaled: ptr = torch_std_mul_scalar(t, 1000.0)
    return timestep_embedding(t_scaled, dim, max_period)
