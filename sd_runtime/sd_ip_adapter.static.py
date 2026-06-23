# sd_runtime/sd_ip_adapter.static.py — IP-Adapter 图像提示注入 (Phase 15)
#
# IP-Adapter: 通过 cross-attention 将图像特征注入 UNet.
# 使用 JIT TorchScript 模型 (含 IP-Adapter 权重) 进行前向。
# 备选: 在外部处理 image→clip_image_emb→注入 UNet.

from ops import *


# ==============================================================================
# IP-Adapter 全局状态
# ==============================================================================

_ip_adapter_jit: ptr
_ip_adapter_image_emb: ptr
_ip_adapter_scale: float


# ==============================================================================
# IP-Adapter 模型管理
# ==============================================================================

def ip_adapter_init(jit_model_path: str) -> void:
    """加载 IP-Adapter JIT TorchScript 模型.
    
    JIT 模型应包含 CLIP image encoder + IP-Adapter cross-attn 注入.
    """
    global _ip_adapter_jit
    _ip_adapter_jit = torch_std_jit_load(jit_model_path)
    _ip_adapter_image_emb = null
    _ip_adapter_scale = 1.0


def ip_adapter_load_image(image: ptr) -> ptr:
    """Encode image with IP-Adapter CLIP image encoder.
    
    image: (1, 3, H, W) float32
    Returns: image embeddings for cross-attention injection
    """
    global _ip_adapter_jit
    img_emb = torch_std_jit_forward(_ip_adapter_jit, image)
    return img_emb


def ip_adapter_set_image(image_emb: ptr, scale: float) -> void:
    """Set image embedding for cross-attention injection."""
    global _ip_adapter_image_emb, _ip_adapter_scale
    _ip_adapter_image_emb = image_emb
    _ip_adapter_scale = scale


def ip_adapter_apply(unet_latent: ptr, timestep: ptr,
                      text_emb: ptr, guidance: float) -> ptr:
    """UNet forward with IP-Adapter cross-attention injection.
    
    Uses JIT model (UNet + IP-Adapter baked in) or pure UNet fallback.
    """
    global _ip_adapter_jit, _ip_adapter_image_emb, _ip_adapter_scale
    if _ip_adapter_image_emb != null and _ip_adapter_scale > 0.0:
        return torch_std_jit_forward(_ip_adapter_jit, unet_latent,
                                      timestep, text_emb,
                                      _ip_adapter_image_emb,
                                      _ip_adapter_scale)
    # Fallback: base UNet without IP-Adapter
    return torch_std_sd_unet_forward(None, unet_latent, timestep,
                                      text_emb, None, null, 0)


def ip_adapter_free() -> void:
    global _ip_adapter_jit
    torch_std_jit_module_delete(_ip_adapter_jit)
