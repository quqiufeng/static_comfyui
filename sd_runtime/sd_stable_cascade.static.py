# sd_runtime/sd_stable_cascade.static.py — Stable Cascade 三级级联扩散 (Phase 15)
# 对位 comfyui_ref/comfy/ldm/cascade/ (1123 行)
#
# Stable Cascade: Stage_A (VAE 压缩) → Stage_B (浅扩散) → Stage_C (主扩散)
# 使用 JIT TorchScript 模型进行各阶段前向.

from ops import *
from sd_vae import *


# ==============================================================================
# Stable Cascade 全局状态
# ==============================================================================

_sc_stage_a: ptr   # VAE-like: 图像 ↔ latent 压缩 (c=4, 8x)
_sc_stage_b: ptr   # 浅扩散: latent → latent (c=4, 4x, 1024x1024 → 256x256)
_sc_stage_c: ptr   # 主扩散: latent → latent with text conditioning
_sc_has_init: int


# ==============================================================================
# 模型管理
# ==============================================================================

def stable_cascade_init(stage_a_path: str, stage_b_path: str,
                         stage_c_path: str) -> void:
    """加载 Stable Cascade 三个阶段的 JIT 模型."""
    global _sc_stage_a, _sc_stage_b, _sc_stage_c, _sc_has_init
    _sc_stage_a = torch_std_jit_load(stage_a_path)
    _sc_stage_b = torch_std_jit_load(stage_b_path)
    _sc_stage_c = torch_std_jit_load(stage_c_path)
    _sc_has_init = 1


# ==============================================================================
# Stage A: VAE 压缩 (pixel ↔ latent, 8x 压缩)
# ==============================================================================

def sc_stage_a_encode(image: ptr) -> ptr:
    """Encode image to Stage A latent (c=4, 8x spatial reduction).
    
    image: (1, 3, H, W) → latent: (1, 4, H/8, W/8)
    """
    global _sc_stage_a
    return torch_std_jit_forward(_sc_stage_a, image)


def sc_stage_a_decode(latent: ptr) -> ptr:
    """Decode Stage A latent back to image.
    
    latent: (1, 4, H, W) → image: (1, 3, H*8, W*8)
    """
    global _sc_stage_a
    return torch_std_jit_forward(_sc_stage_a, latent)


# ==============================================================================
# Stage B: 浅扩散 (c=4, 4x 分辨率, 通常 256x256 → 1024x1024)
# ==============================================================================

def sc_stage_b_forward(latent: ptr, timestep: ptr,
                        r: ptr) -> ptr:
    """Stage B denoising step.
    
    latent: (1, 4, H, W) at 256x256 equivalent
    timestep: (1,) scalar
    r: (1,) resolution conditioning
    Returns: denoised latent
    """
    global _sc_stage_b
    return torch_std_jit_forward(_sc_stage_b, latent, timestep, r)


def sc_stage_b_sample(noise: ptr, steps: int, r_val: float,
                       seed: int) -> ptr:
    """Run full Stage B sampling.
    
    noise: (1, 4, H, W) random noise
    steps: number of denoising steps
    r_val: resolution parameter
    Returns: denoised latent
    """
    dt = 1.0 / float(steps)
    x = noise
    for i in range(steps):
        t_scalar = float(i) * dt
        t_shape = make_int_array(1)
        int_array_set(t_shape, 0, 1)
        t = torch_std_full(t_shape, 1, t_scalar, 0)
        r_shape = make_int_array(1)
        int_array_set(r_shape, 0, 1)
        r = torch_std_full(r_shape, 1, r_val, 0)
        x = sc_stage_b_forward(x, t, r)
    return x


# ==============================================================================
# Stage C: 主扩散 (c=16, 带文本条件)
# ==============================================================================

def sc_stage_c_forward(latent: ptr, timestep: ptr, r: ptr,
                        clip_text: ptr, clip_text_pooled: ptr,
                        clip_img: ptr) -> ptr:
    """Stage C denoising step with full conditioning.
    
    latent: (1, 16, H, W)
    timestep: (1,) scalar
    r: (1,) resolution conditioning
    clip_text: (1, L, 1280) CLIP text embeddings
    clip_text_pooled: (1, 1280) pooled CLIP
    clip_img: optional (1, 4, 768) CLIP image embeddings
    Returns: denoised latent
    """
    global _sc_stage_c
    return torch_std_jit_forward(_sc_stage_c, latent, timestep, r,
                                  clip_text, clip_text_pooled, clip_img)


def sc_stage_c_sample(noise: ptr, steps: int, r_val: float,
                       clip_text: ptr, clip_text_pooled: ptr,
                       seed: int) -> ptr:
    """Run full Stage C sampling."""
    dt = 1.0 / float(steps)
    x = noise
    for i in range(steps):
        t_scalar = float(i) * dt
        t_shape = make_int_array(1)
        int_array_set(t_shape, 0, 1)
        t = torch_std_full(t_shape, 1, t_scalar, 0)
        r_shape = make_int_array(1)
        int_array_set(r_shape, 0, 1)
        r = torch_std_full(r_shape, 1, r_val, 0)
        x = sc_stage_c_forward(x, t, r, clip_text,
                                clip_text_pooled, null)
    return x


# ==============================================================================
# 完整管线: txt2img
# ==============================================================================

def stable_cascade_generate(prompt: str, steps_b: int, steps_c: int,
                             height: int, width: int, seed: int) -> ptr:
    """Full Stable Cascade txt2img pipeline.
    
    Returns: (1, 3, H, W) generated image
    """
    # Stage C: 从噪声生成 16 通道 latent
    latent_h_c = height // 32  # Stage C operates at 32x compression
    latent_w_c = width // 32
    noise = torch_std_randn(16, latent_h_c * latent_w_c, seed)
    noise = torch_std_unsqueeze(noise, 0)
    
    # Dummy text embeddings (placeholder — needs proper CLIP encoding)
    clip_text = null
    clip_text_pooled = null
    
    c_out = sc_stage_c_sample(noise, steps_c, 1.0,
                               clip_text, clip_text_pooled, seed)
    
    # Stage B: 上采样到 4 通道 4x 分辨率
    # (In practice: Stage C→Stage B upsampling is more complex)
    b_out = c_out  # placeholder
    
    # Stage A: decode to image
    image = sc_stage_a_decode(b_out)
    return image


def stable_cascade_free() -> void:
    global _sc_stage_a, _sc_stage_b, _sc_stage_c
    torch_std_jit_module_delete(_sc_stage_a)
    torch_std_jit_module_delete(_sc_stage_b)
    torch_std_jit_module_delete(_sc_stage_c)
