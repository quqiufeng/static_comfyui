# sd_runtime/sd_cosmos.static.py — Cosmos Video 扩散 (Phase 15)
#
# Cosmos (NVIDIA): 视频扩散模型.
# 使用 JIT TorchScript 前向.

from ops import *


_cosmos_jit: ptr
_cosmos_vae: ptr


def cosmos_init(jit_model_path: str, vae_path: str) -> void:
    global _cosmos_jit, _cosmos_vae
    _cosmos_jit = torch_std_jit_load(jit_model_path)
    _cosmos_vae = torch_std_jit_load(vae_path)


def cosmos_encode(frames: ptr) -> ptr:
    global _cosmos_vae
    return torch_std_jit_forward(_cosmos_vae, frames)


def cosmos_decode(latent: ptr) -> ptr:
    global _cosmos_vae
    return torch_std_jit_forward(_cosmos_vae, latent)


def cosmos_forward(latent: ptr, timestep: ptr,
                    text_emb: ptr) -> ptr:
    global _cosmos_jit
    return torch_std_jit_forward(_cosmos_jit, latent, timestep, text_emb)


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
        noise_pred = cosmos_forward(x, t, null)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return cosmos_decode(x)


def cosmos_free() -> void:
    global _cosmos_jit, _cosmos_vae
    torch_std_jit_module_delete(_cosmos_jit)
    torch_std_jit_module_delete(_cosmos_vae)
