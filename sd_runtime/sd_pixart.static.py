# sd_runtime/sd_pixart.static.py — PixArt DiT (libTorch C++ forward)
# 对位 comfyui_ref/comfy/ldm/pixart/pixartms.py (256 行)
# 使用 torch_std_pixart_forward (纯 libTorch, 28-block DiT)

from ops import *


_pixart_weight_ptrs: ptr
_pixart_n_weights: int


def pixart_init(safetensors_path: str) -> void:
    """加载 PixArt safetensors 权重."""
    global _pixart_weight_ptrs, _pixart_n_weights
    sd_dict = torch_std_safetensors_load(safetensors_path)
    # Load all weight tensors into flat array — order matches C++ weight layout
    n = torch_std_safetensors_count(sd_dict)
    w = make_ptr_array(n)
    for i in range(n):
        ptr_array_set(w, i, torch_std_safetensors_tensor(sd_dict, i))
    _pixart_weight_ptrs = w
    _pixart_n_weights = n


def pixart_forward(latent: ptr, timestep: ptr, text_emb: ptr,
                    height: int, width: int) -> ptr:
    """PixArt DiT forward via libTorch C++.
    
    latent: (1, 4, H, W)
    timestep: (1,) scalar
    text_emb: (1, 120, 4096) T5 embeddings
    """
    global _pixart_weight_ptrs, _pixart_n_weights
    return torch_std_pixart_forward(
        _pixart_weight_ptrs, _pixart_n_weights,
        latent, timestep, text_emb,
        height, width, 2)  # patch_size=2


def pixart_generate(prompt: str, steps: int, cfg: float,
                     height: int, width: int, seed: int) -> ptr:
    """PixArt txt2img via libTorch C++ DiT forward."""
    # T5 encode (120 tokens for PixArt)
    txt = t5_encode(prompt, 120)
    
    # Noise latent
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
        
        # CFG
        noise_pred = pixart_forward(x, t, txt, height, width)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return torch_std_vae_decode(x)


def pixart_free() -> void:
    pass
