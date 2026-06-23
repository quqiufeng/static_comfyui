# sd_runtime/sd_controlnet.static.py — ControlNet 编排 (Phase 9)
#
# Flat functions. 全局状态: controlnet weights ptr.

from ops import *


_cn_weights: ptr
_cn_n_weights: int


def controlnet_load(safetensors_path: str) -> void:
    """Load ControlNet safetensors → weight ptrs array."""
    global _cn_weights, _cn_n_weights
    # Load safetensors and extract weight ptrs
    dict_ptr = torch_std_safetensors_load(safetensors_path)
    _cn_n_weights = torch_std_safetensors_count(dict_ptr)
    _cn_weights = make_ptr_array(_cn_n_weights)
    for i in range(_cn_n_weights):
        ptr_array_set(_cn_weights, i,
                      torch_std_safetensors_tensor(dict_ptr, i))
    # We keep dict_ptr alive since tensors reference its memory
    # (simplification: in practice, tensors are cloned)


def controlnet_forward(x: ptr, timestep: ptr, text_emb: ptr,
                        hint: ptr, num_hint_channels: int) -> ptr:
    """Run ControlNet forward → control features."""
    global _cn_weights, _cn_n_weights
    return torch_std_controlnet_forward(
        _cn_weights, _cn_n_weights,
        x, timestep, text_emb, hint, num_hint_channels)


def controlnet_apply(unet_output: ptr, control_features: ptr,
                      strength: float) -> ptr:
    """Apply control to UNet output."""
    return torch_std_controlnet_apply(unet_output, control_features, strength)
