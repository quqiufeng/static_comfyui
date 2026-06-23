# sd_runtime/sd_wan_video.static.py — Wan Video 扩散 (libTorch C++)
#
# Wan Video: 3D UNet with RoPE position encoding.
# 使用 torch_std_wan_video_forward (纯 libTorch).

from ops import *


_wan_dict: ptr
_wan_vae: ptr


def wan_video_init(safetensors_path: str, vae_path: str) -> void:
    global _wan_dict, _wan_vae
    _wan_dict = torch_std_safetensors_load(safetensors_path)
    _wan_vae = torch_std_jit_load(vae_path)


def wan_video_encode(frames: ptr) -> ptr:
    global _wan_vae
    return torch_std_jit_forward(_wan_vae, frames)


def wan_video_decode(latent: ptr) -> ptr:
    global _wan_vae
    return torch_std_jit_forward(_wan_vae, latent)


def wan_video_forward(latent: ptr, timestep: ptr, text_emb: ptr,
                       n_frames: int, height: int, width: int) -> ptr:
    global _wan_dict
    return torch_std_wan_video_forward(
        _wan_dict,
        latent, timestep, text_emb,
        n_frames, height, width)


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
        noise_pred = wan_video_forward(x, t, null, n_frames, height, width)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return wan_video_decode(x)


def wan_video_free() -> void:
    pass
