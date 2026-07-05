from k_diffusion.sampling import (
    get_sigmas,
    sample_step,
    sample_flow_step,
)
from model_sampling import ModelSamplingType


def prepare_noise(latent_shape, seed: int, device):
    torch.manual_seed(seed)
    return torch.randn(latent_shape)


def cfg_denoise(model_fn, x, sigma, cond, uncond, cfg_scale: float):
    s_in = sigma
    cond_out = model_fn(x, s_in, cond)
    uncond_out = model_fn(x, s_in, uncond)
    return uncond_out + (cond_out - uncond_out) * cfg_scale


def sample(model_fn, noise, steps: int, cfg: float, sampler_name: str,
           cond, uncond, sampler_type: int, sigma_min: float, sigma_max: float,
           denoise: float, seed: int):
    x = noise
    sigmas = get_sigmas(sampler_type, steps, sigma_min, sigma_max)
    if sampler_type == ModelSamplingType.FLOW:
        n = 0
        while n < steps:
            t = sigmas[n]
            t_next = sigmas[n + 1]
            denoised = cfg_denoise(model_fn, x, t, cond, uncond, cfg)
            dt = t_next - t
            x = sample_flow_step(denoised, x, dt)
            n = n + 1
    else:
        n = 0
        old_denoised = None
        while n < steps:
            sigma_t = sigmas[n]
            sigma_prev = sigmas[n + 1]
            denoised = cfg_denoise(model_fn, x, sigma_t, cond, uncond, cfg)
            x = sample_step(sampler_name, denoised, x, sigma_t, sigma_prev, old_denoised)
            old_denoised = denoised
            n = n + 1
    return x


def main():
    pass
