# sd_runtime/sd_cosmos.static.py — Cosmos Video 扩散 (libTorch C++)
#
# Cosmos (NVIDIA): 视频扩散模型.
# 使用 torch_std_cosmos_forward (纯 libTorch).

from ops import *


_cosmos_weight_ptrs: ptr
_cosmos_n_weights: int
_cosmos_vae: ptr


def cosmos_init(safetensors_path: str, vae_path: str) -> void:
    global _cosmos_weight_ptrs, _cosmos_n_weights, _cosmos_vae
    sd_dict = torch_std_safetensors_load(safetensors_path)
    n = torch_std_safetensors_count(sd_dict)
    w = make_ptr_array(n)
    for i in range(n):
        ptr_array_set(w, i, torch_std_safetensors_tensor(sd_dict, i))
    _cosmos_weight_ptrs = w
    _cosmos_n_weights = n
    _cosmos_vae = torch_std_jit_load(vae_path)


def cosmos_encode(frames: ptr) -> ptr:
    global _cosmos_vae
    return torch_std_jit_forward(_cosmos_vae, frames)


def cosmos_decode(latent: ptr) -> ptr:
    global _cosmos_vae
    return torch_std_jit_forward(_cosmos_vae, latent)


def cosmos_forward(latent: ptr, timestep: ptr, text_emb: ptr,
                    n_frames: int, height: int, width: int) -> ptr:
    global _cosmos_weight_ptrs, _cosmos_n_weights
    return torch_std_cosmos_forward(
        _cosmos_weight_ptrs, _cosmos_n_weights,
        latent, timestep, text_emb,
        n_frames, height, width)


def cosmos_generate(prompt: str, steps: int,
                     n_frames: int, height: int, width: int,
                     seed: int) -> ptr:
    latent_h = height // 8
    latent_w = width // 8
    noise = torch_std_randn(n_frames * 16 * latent_h * latent_w, 1, seed)
    shape = make_int_array(4)
    int_array_set(shape, 0, n_frames)
    int_array_set(shape, 1, 16)
    int_array_set(shape, 2, latent_h)
    int_array_set(shape, 3, latent_w)
    x = torch_std_reshape(noise, shape, 4)
    
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_scalar = float(i) * dt
        t_shape = make_int_array(1)
        int_array_set(t_shape, 0, 1)
        t = torch_std_full(t_shape, 1, t_scalar, 0)
        noise_pred = cosmos_forward(x, t, null, n_frames, height, width)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return cosmos_decode(x)


def cosmos_free() -> void:
    pass
