# sd_runtime/sd_ip_adapter.static.py — IP-Adapter 图像提示注入 (libTorch C++)
#
# IP-Adapter: 通过 torch_std_sd_unet_forward_v2 的 ip_adapt 参数注入.
# 使用纯 libTorch, 无需 JIT.

from ops import *


_ip_adapter_img_emb: ptr
_ip_adapter_scale: float


def ip_adapter_init() -> void:
    """Initialize IP-Adapter (no model load — uses UNet v2 hook)."""
    global _ip_adapter_img_emb, _ip_adapter_scale
    _ip_adapter_img_emb = null
    _ip_adapter_scale = 1.0


def ip_adapter_set_image(image_emb: ptr, scale: float) -> void:
    global _ip_adapter_img_emb, _ip_adapter_scale
    _ip_adapter_img_emb = image_emb
    _ip_adapter_scale = scale


def ip_adapter_apply(unet_latent: ptr, timestep: ptr,
                      text_emb: ptr, weight_ptrs: ptr, n_weights: int) -> ptr:
    """UNet forward with IP-Adapter cross-attn injection via C++ v2."""
    global _ip_adapter_img_emb, _ip_adapter_scale
    return torch_std_sd_unet_forward_v2(
        weight_ptrs, n_weights,
        unet_latent, timestep, text_emb,
        null, 0, null, null, 0, 0.0,
        null, null, null, 0,
        _ip_adapter_img_emb, _ip_adapter_scale)


def ip_adapter_free() -> void:
    pass
