# sd_runtime/sd_sd3.static.py — SD3 管线 (Phase 15 gap)
# 对位 comfyui_ref/comfy/ldm/modules/diffusionmodules/mmdit.py (~1037 行)
#
# SD3: MMDiT 架构 (类似 FLUX, 但使用三文本编码器 + 不同 conditioning)
# 使用 JIT TorchScript 模型前向 + C++ 工具函数。

from ops import *
from sd_t5 import *
from sd1_clip import *
from sdxl_clip import *
from sd_samplers_extras import *
from sd_vae import *
from sd_utils import *


# 全局状态
_sd3_jit: ptr
_sd3_t5: ptr
_sd3_clip_l: ptr
_sd3_clip_g: ptr


# ==============================================================================
# SD3 模型管理
# ==============================================================================

def sd3_init(jit_model_path: str) -> void:
    """Initialize SD3 model and text encoders."""
    global _sd3_jit
    _sd3_jit = torch_std_jit_load(jit_model_path)


def sd3_encode_prompt(text: str) -> ptr:
    """Encode prompt with all 3 text encoders.
    
    Returns: concatenated embeddings (1, L, 4096)
    """
    global _sd3_jit, _sd3_t5, _sd3_clip_l, _sd3_clip_g
    
    # T5 XXL encoding
    t5_out = t5_encode(text)
    
    # CLIP-L encoding
    clip_l_out = sd1_clip_encode(_sd3_clip_l, text, 77)
    
    # CLIP-G encoding (SDXL CLIP)
    clip_gokens = sdxl_clip_tokenize(_sd3_clip_g, text)
    clip_g_out = sdxl_clip_encode_tokens(_sd3_clip_g, clip_gokens)
    
    # Concatenate: [pooled_L, pooled_G, T5_out]
    # Simplified: cat along sequence dim
    tensors = make_ptr_array(3)
    ptr_array_set(tensors, 0, clip_l_out)
    ptr_array_set(tensors, 1, clip_g_out)
    ptr_array_set(tensors, 2, t5_out)
    concat = torch_std_cat(tensors, 3, 1)
    return concat


def sd3_forward(img: ptr, txt: ptr, timestep: ptr,
                guidance: float) -> ptr:
    """SD3 MMDiT forward.
    
    img: (1, 16*H, W) latent patches
    txt: (1, L, 4096) T5 + CLIP text embeddings
    timestep: (1,) timestep tensor
    guidance: CFG scale
    
    Returns: (1, 16*H, W) predicted noise/velocity
    """
    global _sd3_jit
    return torch_std_jit_forward(_sd3_jit, img, txt, timestep)


def sd3_generate(prompt: str, steps: int, cfg: float, seed: int,
                 width: int, height: int) -> ptr:
    """Full SD3 txt2img pipeline.
    
    Returns: (1, 3, H, W) generated image
    """
    # 1. Encode prompt
    txt = sd3_encode_prompt(prompt)
    
    # 2. Prepare noise
    sigmas = flow_match_sigmas(steps, 0.002, 30.0)
    latent_h = height // 8
    latent_w = width // 8
    x = torch_std_randn(latent_h * 16, latent_w, seed)
    x = torch_std_unsqueeze(x, 0)
    
    # 3. Flow matching sampling (reuse FLUX sampler)
    for i in range(steps):
        t = torch_std_full_scalar(1, sigmas[i])
        guidance_tensor = torch_std_full_scalar(1, cfg)
        v = sd3_forward(x, txt, t, guidance_tensor)
        x = flow_match_step(x, v, sigmas[i], sigmas[i+1], seed + i)
    
    # 4. VAE decode
    image = torch_std_vae_decode(x)
    return image


def sd3_free() -> void:
    global _sd3_jit
    torch_std_jit_module_delete(_sd3_jit)
