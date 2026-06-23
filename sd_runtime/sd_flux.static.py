# sd_runtime/sd_flux.static.py — FLUX 管线 (Phase 11)
#
# Flat functions. 全局状态: flux_dict

from ops import *
import math


FLUX_LATENT_CHANNELS: int = 16
FLUX_N_BLOCKS: int = 24
FLUX_N_HEADS_IMG: int = 24
FLUX_N_HEADS_TXT: int = 16
FLUX_HEAD_DIM: int = 128

_flux_dict: ptr


def flux_init(wdict: ptr) -> void:
    """Set FLUX MMDiT weight dict."""
    global _flux_dict
    _flux_dict = wdict


def flux_txt2img(t5_encode_fn, prompt: str, steps: int, guidance: float,
                 height: int, width: int, seed: int,
                 vae_decode_fn) -> ptr:
    """FLUX txt2img pipeline.
    
    t5_encode_fn: function(text, max_len) → embeddings ptr
    vae_decode_fn: function(latent) → image ptr
    """
    global _flux_dict
    
    # 1. T5 encode
    txt_emb = t5_encode_fn(prompt, 512)
    
    # 2. Noise latent
    torch_std_manual_seed(seed)
    latent_h = height // 16
    latent_w = width // 16
    shape = make_int_array(4)
    int_array_set(shape, 0, 1)
    int_array_set(shape, 1, FLUX_LATENT_CHANNELS)
    int_array_set(shape, 2, latent_h)
    int_array_set(shape, 3, latent_w)
    x = torch_std_randn(shape, 4, 0)
    
    # 3. Flow matching Euler sampling
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = float(i) * dt
        scalar_shape = make_int_array(1)
        int_array_set(scalar_shape, 0, 1)
        t_tensor = torch_std_full(scalar_shape, 1, t, 0)
        
        velocity = torch_std_flux_forward(
            _flux_dict, x, txt_emb, t_tensor, null,
            guidance, FLUX_N_BLOCKS,
            FLUX_N_HEADS_IMG, FLUX_N_HEADS_TXT, FLUX_HEAD_DIM)
        
        x = torch_std_fm_step(velocity, x, dt)
    
    # 4. VAE decode
    return vae_decode_fn(x)


def flux_free() -> void:
    pass
