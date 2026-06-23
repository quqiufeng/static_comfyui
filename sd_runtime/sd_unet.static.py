# sd_runtime/sd_unet.static.py — SD UNet Forward 编排
# 1:1 对齐 comfyui_ref/comfy/ldm/modules/diffusionmodules/openaimodel.py
#
# C++ 运行时已包含完整 UNet 架构 (ResBlock + Transformer + Attention + Skip)：
#   SD1.5: torch_std_sd15_unet_forward_dict (从 safetensors dict 取权重)
#   SDXL:  torch_std_sdxl_unet_forward (dict + 尺寸嵌入)
#
# 本模块仅做管线编排：加载 safetensors → 调用 C++ forward → 返回结果。
#
# 用法:
#   from sd_unet import sd15_unet_forward, sdxl_unet_forward
#
#   sd_dict = load_safetensors("/model.safetensors")
#   result = sd15_unet_forward(sd_dict, latent, timesteps, text_emb)

from ops import *
import math


# ==============================================================================
# 公共常量
# ==============================================================================

# SD1.5 模型配置
SD15_IN_CHANNELS: int = 4
SD15_MODEL_CHANNELS: int = 320
SD15_OUT_CHANNELS: int = 4

# SDXL 模型配置
SDXL_IN_CHANNELS: int = 4
SDXL_MODEL_CHANNELS: int = 320  # actually 0.5 * 320? No, let me check
SDXL_OUT_CHANNELS: int = 4


# ==============================================================================
# SD1.5 UNet Forward
# ==============================================================================

def sd15_unet_forward(sd_dict: ptr, latent: ptr, timestep: ptr,
                       text_emb: ptr) -> ptr:
    """Run SD1.5 UNet inference.
    
    sd_dict: safetensors dict pointer (from torch_std_safetensors_load)
    latent: (B, 4, H, W) float32 latent
    timestep: (B,) float32 timesteps
    text_emb: (B, 77, 768) CLIP text embeddings
    
    Returns: (B, 4, H, W) denoised latent
    """
    return torch_std_sd15_unet_forward_dict(
        sd_dict, latent, timestep, text_emb,
        null, null, null, 0, 0.0)  # no LoRA


# ==============================================================================
# SDXL UNet Forward
# ==============================================================================

def sdxl_unet_forward(sd_dict: ptr, latent: ptr, timestep: ptr,
                       text_emb: ptr, pooled_emb: ptr,
                       original_size_h: float, original_size_w: float,
                       crop_top: float, crop_left: float,
                       target_size_h: float, target_size_w: float) -> ptr:
    """Run SDXL UNet inference with size conditioning.
    
    sd_dict: weight dict (unordered_map<string, Tensor>) 
    latent: (B, 4, H, W) float32 latent
    timestep: (B,) float32 timesteps
    text_emb: (B, 77, 2048) concat CLIP-L + CLIP-G embeddings
    pooled_emb: (B, 1280) pooled CLIP-G embedding
    original_size_h/w: original image size (for conditioning)
    crop_top/left: crop parameters
    target_size_h/w: target generation size
    
    Returns: (B, 4, H, W) denoised latent
    """
    return torch_std_sdxl_unet_forward(
        sd_dict, latent, timestep, text_emb, pooled_emb,
        original_size_h, original_size_w,
        crop_top, crop_left,
        target_size_h, target_size_w)
