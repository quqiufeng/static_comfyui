# sd_runtime/sd_hunyuan_video.static.py — Hunyuan Video 扩散 (libTorch C++)
#
# Hunyuan Video: 3D UNet with spatio-temporal attention.
# 使用 torch_std_hunyuan_video_forward (纯 libTorch).

from ops import *


_hunyuan_dict: ptr
_hunyuan_vae: ptr


def hunyuan_video_init(safetensors_path: str, vae_path: str) -> void:
    """加载 Hunyuan Video 权重 + VAE."""
    global _hunyuan_dict, _hunyuan_vae
    _hunyuan_dict = torch_std_safetensors_load(safetensors_path)
    _hunyuan_vae = torch_std_jit_load(vae_path)


def hunyuan_video_encode(frames: ptr) -> ptr:
    global _hunyuan_vae
    return torch_std_jit_forward(_hunyuan_vae, frames)


def hunyuan_video_decode(latent: ptr) -> ptr:
    global _hunyuan_vae
    return torch_std_jit_forward(_hunyuan_vae, latent)


def hunyuan_video_forward(latent: ptr, timestep: ptr, text_emb: ptr,
                           n_frames: int, height: int, width: int) -> ptr:
    """Hunyuan Video 3D UNet forward via libTorch C++."""
    global _hunyuan_dict
    return torch_std_hunyuan_video_forward(
        _hunyuan_dict,
        latent, timestep, text_emb,
        n_frames, height, width)


def hunyuan_video_generate(prompt: str, steps: int, cfg: float,
                            n_frames: int, height: int, width: int,
                            seed: int) -> ptr:
    """Generate video from text prompt."""
    latent_h = height // 8
    latent_w = width // 8
    noise = torch_std_randn(n_frames * 4 * latent_h * latent_w, 1, seed)
    shape = make_int_array(4)
    int_array_set(shape, 0, n_frames)
    int_array_set(shape, 1, 4)
    int_array_set(shape, 2, latent_h)
    int_array_set(shape, 3, latent_w)
    x = torch_std_reshape(noise, shape, 4)
    
    dt = 1.0 / float(steps)
    for i in range(steps):
        t_scalar = float(i) * dt
        t_shape = make_int_array(1)
        int_array_set(t_shape, 0, 1)
        t = torch_std_full(t_shape, 1, t_scalar, 0)
        noise_pred = hunyuan_video_forward(x, t, null, n_frames, height, width)
        x = torch_std_sub(x, torch_std_mul_scalar(noise_pred, dt))
    
    return hunyuan_video_decode(x)


def hunyuan_video_free() -> void:
    pass
