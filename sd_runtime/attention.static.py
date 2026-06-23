# sd_runtime/attention.static.py — Attention 核心函数
# 1:1 对齐 comfyui_ref/comfy/ldm/modules/attention.py
#
# 注意：C++ 运行时 (libtorch_std_helper) 已包含 SD UNet/FLUX 完整 forward，
#       本模块仅提供轻量函数式接口用于管线编排和自定义 attention。
#
# 所有权重张量由调用者传入 (函数式，无状态)。
#
# 用法:
#   from attention import optimized_attention, cross_attention

from ops import *
import math

# ==============================================================================
# 注意力核心计算
# ==============================================================================

def optimized_attention(q: ptr, k: ptr, v: ptr, heads: int,
                         mask: ptr, skip_reshape: int) -> ptr:
    """Multi-head scaled dot-product attention.
    
    Standard (skip_reshape=0):
        q, k, v: (B, seq, D) float32
        heads: number of heads (D必须被heads整除)
        dim_head = D // heads
        Returns: (B, seq, D)
    
    Pre-reshaped FLUX (skip_reshape=1):
        q, k, v: (B, heads, seq, dim_head)
        Returns: (B, heads, seq, dim_head)
    
    mask: attention bias or null
    """
    return torch_std_attention(q, k, v, heads, mask, skip_reshape)


# ==============================================================================
# CrossAttention (attention.py:794)
# ==============================================================================

def cross_attention(q_w: ptr, k_w: ptr, v_w: ptr,
                     o_w: ptr, o_b: ptr,
                     x: ptr, context: ptr,
                     heads: int, mask: ptr) -> ptr:
    """CrossAttention forward.
    
    q_w/k_w/v_w: Linear weight (no bias in SD)
    o_w/o_b: output Linear (bias optional)
    x: (B, seq_q, query_dim)
    context: (B, seq_k, context_dim) or x for self-attn
    heads: number of heads
    mask: attention bias or null
    """
    q: ptr = torch_std_linear(x, q_w, null)
    k: ptr = torch_std_linear(context, k_w, null)
    v: ptr = torch_std_linear(context, v_w, null)
    attn_out: ptr = torch_std_attention(q, k, v, heads, mask, 0)
    return torch_std_linear(attn_out, o_w, o_b)


# ==============================================================================
# GEGLU (attention.py:94)
# ==============================================================================

def geglu_forward(proj_w: ptr, proj_b: ptr, x: ptr) -> ptr:
    """GEGLU: proj(x) -> chunk(2, dim=-1) -> gelu(gate) * x"""
    proj: ptr = torch_std_linear(x, proj_w, proj_b)
    ndim: int = torch_std_ndim(proj)
    shape_buf: ptr = make_int_array(ndim)
    torch_std_shape(proj, shape_buf)
    last_dim: int = int_array_ref(shape_buf, ndim - 1)
    half: int = last_dim // 2
    x_part: ptr = torch_std_slice(proj, ndim - 1, 0, half, 1)
    gate: ptr = torch_std_slice(proj, ndim - 1, half, last_dim, 1)
    return torch_std_mul(x_part, torch_std_gelu(gate))


# ==============================================================================
# FeedForward (attention.py:104)
# ==============================================================================

def feed_forward(in_w: ptr, in_b: ptr, out_w: ptr, out_b: ptr,
                  x: ptr, glu: int, glu_w: ptr, glu_b: ptr) -> ptr:
    """FeedForward: in -> activation -> out
    
    glu=0: Linear -> GELU -> Linear
    glu=1: GEGLU (proj -> chunk -> gelu(gate)*x) -> Linear
    """
    if glu:
        hidden: ptr = geglu_forward(glu_w, glu_b, x)
    else:
        hidden: ptr = torch_std_gelu(torch_std_linear(x, in_w, in_b))
    return torch_std_linear(hidden, out_w, out_b)
