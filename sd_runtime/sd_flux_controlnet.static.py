# sd_runtime/sd_flux_controlnet.static.py — FLUX ControlNet (Phase 11 gap)
# 对位 comfyui_ref/comfy/ldm/flux/controlnet.py (208 行)
#
# FLUX ControlNet: 双流 control (img + txt 均受控)
# 使用 C++ torch_std_flux_forward 作为基础,
# control 特征通过特征加法注入 img 和 txt 表示。

from ops import *


# 全局状态
_fc_weights: ptr
_fc_n_weights: int
_fc_n_blocks: int
_fc_n_heads_img: int
_fc_n_heads_txt: int
_fc_head_dim: int


def flux_controlnet_init(safetensors_path: str) -> void:
    """Load FLUX ControlNet weights."""
    global _fc_weights, _fc_n_weights
    dict_ptr = torch_std_safetensors_load(safetensors_path)
    _fc_n_weights = torch_std_safetensors_count(dict_ptr)
    _fc_weights = make_ptr_array(_fc_n_weights)
    for i in range(_fc_n_weights):
        ptr_array_set(_fc_weights, i,
                      torch_std_safetensors_tensor(dict_ptr, i))
    _fc_n_blocks = 24
    _fc_n_heads_img = 24
    _fc_n_heads_txt = 16
    _fc_head_dim = 128


def flux_controlnet_forward(img: ptr, txt: ptr, timestep: ptr,
                             hint: ptr, guidance: float) -> ptr:
    """FLUX ControlNet forward.
    
    Processes hint through control network, returns control features
    that will be injected into the main FLUX MMDiT forward.
    
    img: (1, 16*H, W) latent patches
    txt: (1, L, 4096) T5 text embeddings
    timestep: (1,) scalar timestep
    hint: (1, 3, H*8, W*8) control image
    guidance: CFG scale
    
    Returns: control features ptr array
    """
    global _fc_weights, _fc_n_weights
    # Simplified: use C++ controlnet forward on FLUX features
    # In full implementation: separate control UNet that produces
    # per-block control features for both img and txt streams
    return torch_std_controlnet_forward(
        _fc_weights, _fc_n_weights,
        img, timestep, txt, hint, 3)


def flux_controlnet_inject(flux_features: ptr, control_features: ptr,
                            strength: float) -> ptr:
    """Inject control features into FLUX MMDiT features.
    
    flux_features: dict or tensor of FLUX MMDiT features
    control_features: ControlNet output features
    strength: blending factor
    
    Returns: modified features
    """
    scaled = torch_std_mul_scalar(control_features, strength)
    return torch_std_add(flux_features, scaled)


def flux_controlnet_free() -> void:
    pass
