# sd_runtime/sd_gligen.static.py — GLIGEN 布局条件注入 (Phase 15)
# 对位 comfyui_ref/comfy/gligen.py (299 行)
#
# GLIGEN: gated cross-attention 注入 UNet transformer blocks.
# 使用 JIT TorchScript 模型 (含 GLIGEN 权重 + 修改后的 UNet) 进行前向。
# 备选: 通过 position_net + gated_attention 在 UNet 外部处理。

from ops import *


# ==============================================================================
# GLIGEN 全局状态
# ==============================================================================

_gligen_jit: ptr
_gligen_has_boxes: int  # 0/1 flag


# ==============================================================================
# 位置编码 (FourierEmbedder)
# ==============================================================================

def gligen_fourier_embed(boxes: ptr, num_freqs: int) -> ptr:
    """Fourier embedding of bounding boxes.
    
    boxes: (B, N, 4) in [x1, y1, x2, y2] normalized coordinates
    Returns: (B, N, num_freqs*2*4) fourier features
    """
    # For each freq: sin(freq*x), cos(freq*x) for x1,y1,x2,y2
    # Simplified: use JIT module for position encoding
    return torch_std_jit_forward(_gligen_jit, boxes)


# ==============================================================================
# GLIGEN 模型管理
# ==============================================================================

def gligen_init(jit_model_path: str) -> void:
    """Initialize GLIGEN model (JIT module with gated attention baked in).
    
    The JIT module should include the base UNet + GLIGEN gated attention modules.
    Alternatively, loads a standalone GLIGEN position_net + attention weights.
    """
    global _gligen_jit, _gligen_has_boxes
    _gligen_jit = torch_std_jit_load(jit_model_path)
    _gligen_has_boxes = 0


def gligen_set_boxes(boxes_data: ptr, n_boxes: int, 
                      latent_h: int, latent_w: int) -> void:
    """Set GLIGEN bounding box conditioning.
    
    boxes_data: float array of [x1, y1, x2, y2, ...] normalized coords
    n_boxes: number of boxes
    latent_h, latent_w: latent spatial dimensions
    """
    global _gligen_has_boxes
    _gligen_has_boxes = 1 if n_boxes > 0 else 0
    # Store boxes in JIT module's internal state via a call
    if n_boxes > 0:
        shape = make_int_array(2)
        int_array_set(shape, 0, 1)
        int_array_set(shape, 1, n_boxes * 4)
        boxes = torch_std_tensor_from_blob(boxes_data, shape, 2, 0)
        torch_std_jit_forward(_gligen_jit, boxes)


def gligen_apply(unet_latent: ptr, timestep: ptr,
                  text_emb: ptr, guidance: float) -> ptr:
    """Run UNet + GLIGEN forward (JIT model with baked-in gated attention).
    
    unet_latent: (1, 4, H, W) latent
    timestep: (1,) timestep scalar
    text_emb: (1, 77, 768) text embeddings
    guidance: CFG scale
    
    Returns: noise prediction
    """
    global _gligen_jit, _gligen_has_boxes
    if _gligen_has_boxes:
        return torch_std_jit_forward(_gligen_jit, unet_latent, timestep, text_emb)
    # Fallback: use base UNet without GLIGEN
    return torch_std_sd_unet_forward(None, unet_latent, timestep, text_emb,
                                      None, null, 0)


def gligen_free() -> void:
    global _gligen_jit
    torch_std_jit_module_delete(_gligen_jit)
