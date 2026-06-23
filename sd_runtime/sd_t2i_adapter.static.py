# sd_runtime/sd_t2i_adapter.static.py — T2I-Adapter (Phase 9 gap)
# 对位 comfyui_ref/comfy/t2i_adapter/adapter.py (~200行)
#
# T2I-Adapter: lightweight conditioning network that processes hint images
# into feature maps that are injected into UNet via feature addition.
# Simpler than ControlNet — no trainable copy of UNet.
#
# 策略: 使用 JIT TorchScript 模型 (adapter.pt) 进行 hint → feature 映射

from ops import *


# 全局状态
_adapter_jit: ptr


def t2i_adapter_init(jit_model_path: str) -> void:
    """Load T2I-Adapter TorchScript model."""
    global _adapter_jit
    _adapter_jit = torch_std_jit_load(jit_model_path)


def t2i_adapter_preprocess(hint: ptr, target_h: int, target_w: int) -> ptr:
    """Preprocess hint image for adapter.
    
    hint: (1, C, H, W) float32
    target_h, target_w: UNet latent size * 8 (image space)
    
    Returns: (1, 3, target_h, target_w) processed hint
    """
    c = int(torch_std_size(hint, 1))
    if c > 3:
        hint = torch_std_narrow(hint, 1, 0, 3)
    elif c == 1:
        # Grayscale → RGB: repeat channel 3 times via narrow
        tensors = make_ptr_array(3)
        ptr_array_set(tensors, 0, hint)
        ptr_array_set(tensors, 1, hint)
        ptr_array_set(tensors, 2, hint)
        hint = torch_std_cat(tensors, 3, 1)
    
    resized = torch_std_image_resize(hint, target_h, target_w, "bilinear")
    return resized


def t2i_adapter_forward(hint: ptr) -> ptr:
    """Run T2I-Adapter forward → feature conditioning.
    
    hint: (1, 3, H, W) preprocessed hint image
    Returns: list of feature maps (ptr array)
    """
    global _adapter_jit
    return torch_std_jit_forward(_adapter_jit, hint)


def t2i_adapter_apply(unet_features: ptr, adapter_features: ptr,
                       strength: float, start_percent: float,
                       end_percent: float) -> ptr:
    """Apply T2I-Adapter features to UNet.
    
    unet_features: UNet's mid/output features
    adapter_features: T2I-Adapter output features
    strength: blending factor
    start_percent, end_percent: which part of sampling to apply
    
    Returns: modified features (same shape as unet_features)
    """
    if strength <= 0.0:
        return unet_features
    # For SD, adapter features are added to UNet encoder features
    # Simplified: scale and add
    scaled = torch_std_mul_scalar(adapter_features, strength)
    return torch_std_add(unet_features, scaled)


def t2i_adapter_free() -> void:
    """Free T2I-Adapter resources."""
    global _adapter_jit
    torch_std_jit_module_delete(_adapter_jit)
