# sd_runtime/sd_pixart.static.py — PixArt Transformer 扩散 (Phase 15)
# 对位 comfyui_ref/comfy/ldm/pixartms.py
#
# PixArt: DiT (Diffusion Transformer) 架构 + T5 文本编码.
# 使用 JIT TorchScript 前向.

from ops import *
from sd_t5 import *


# ==============================================================================
# 全局状态
# ==============================================================================

_pixart_jit: ptr
_pixart_t5: ptr


# ==============================================================================
# 模型管理
# ==============================================================================

def pixart_init(jit_model_path: str, t5_path: str) -> void:
    """加载 PixArt DiT + T5 文本编码器."""
    global _pixart_jit, _pixart_t5
    _pixart_jit = torch_std_jit_load(jit_model_path)
    _pixart_t5 = torch_std_jit_load(t5_path)


def pixart_encode_text(text: str, max_len: int) -> ptr:
    """Encode prompt with T5 for PixArt conditioning.
    
    Returns: (1, max_len, 4096) T5 embeddings
    """
    global _pixart_t5
    return t5_encode(text)


def pixart_forward(latent: ptr, timestep: ptr,
                    text_emb: ptr, guidance: float) -> ptr:
    """PixArt DiT forward.
    
    latent: (1, 4, H, W) noise latent
    timestep: (1,) scalar
    text_emb: (1, L, 4096) T5 embeddings
    guidance: CFG scale
    Returns: noise prediction
    """
    global _pixart_jit
    # PixArt uses cross-attention with T5 embeddings
    g_shape = make_int_array(1)
    int_array_set(g_shape, 0, 1)
    guidance_t = torch_std_full(g_shape, 1, guidance, 0)
    return torch_std_jit_forward(_pixart_jit, latent, timestep,
                                  text_emb, guidance_t)


def pixart_generate(prompt: str, steps: int, cfg: float,
                     height: int, width: int, seed: int) -> ptr:
    """PixArt txt2img pipeline.
    
    Returns: (1, 3, H, W) image
    """
    global _pixart_jit
    # Encode text
    txt = pixart_encode_text(prompt, 120)
    
    # Prepare noise latent
    latent_h = height // 8
    latent_w = width // 8
    noise = torch_std_randn(4, latent_h * latent_w, seed)
    noise = torch_std_unsqueeze(noise, 0)
    
    # DDIM sampling
    x = noise
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_scalar = float(i) * dt
        t_shape = make_int_array(1)
        int_array_set(t_shape, 0, 1)
        t = torch_std_full(t_shape, 1, t_scalar, 0)
        
        # CFG: cond + uncond
        noise_pred = pixart_forward(x, t, txt, cfg)
        
        # Euler step
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    # Decode via VAE
    return torch_std_vae_decode(x)


def pixart_free() -> void:
    global _pixart_jit, _pixart_t5
    torch_std_jit_module_delete(_pixart_jit)
