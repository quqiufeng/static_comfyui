# sd_runtime/sd_wan_video.static.py — Wan Video 扩散 (Phase 15)
#
# Wan Video: 3D UNet with RoPE position encoding.
# 使用 JIT TorchScript 前向.

from ops import *


_wan_jit: ptr
_wan_vae: ptr


def wan_video_init(jit_model_path: str, vae_path: str) -> void:
    global _wan_jit, _wan_vae
    _wan_jit = torch_std_jit_load(jit_model_path)
    _wan_vae = torch_std_jit_load(vae_path)


def wan_video_encode(frames: ptr) -> ptr:
    global _wan_vae
    return torch_std_jit_forward(_wan_vae, frames)


def wan_video_decode(latent: ptr) -> ptr:
    global _wan_vae
    return torch_std_jit_forward(_wan_vae, latent)


def wan_video_forward(latent: ptr, timestep: ptr,
                       text_emb: ptr) -> ptr:
    global _wan_jit
    return torch_std_jit_forward(_wan_jit, latent, timestep, text_emb)


def wan_video_generate(prompt: str, steps: int,
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
        noise_pred = wan_video_forward(x, t, null)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return wan_video_decode(x)


def wan_video_free() -> void:
    global _wan_jit, _wan_vae
    torch_std_jit_module_delete(_wan_jit)
    torch_std_jit_module_delete(_wan_vae)
