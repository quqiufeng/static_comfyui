def get_sigmas(sampler_type: int, steps: int, sigma_min: float, sigma_max: float):
    if sampler_type == 8:
        return torch.fm_sigmas(steps, sigma_min, sigma_max)
    else:
        return torch.sampler_sigmas(steps, sigma_min, sigma_max, "karras")


def sample_step(sampler_name: str, noise_pred, x_t, sigma_t, sigma_prev, extra=None):
    if sampler_name == "euler":
        return torch.sample_euler(noise_pred, x_t, sigma_t, sigma_prev)
    elif sampler_name == "euler_ancestral":
        return torch.sample_euler_ancestral(noise_pred, x_t, sigma_t, sigma_prev)
    elif sampler_name == "ddim":
        return torch.sample_ddim(noise_pred, x_t, sigma_t, sigma_prev, 0.0)
    elif sampler_name == "dpmpp_2m":
        old_denoised = extra
        is_first = 1 if old_denoised is None else 0
        if old_denoised is None:
            old_denoised = x_t
        return torch.sample_dpmpp_2m(noise_pred, x_t, sigma_t, sigma_prev, old_denoised, is_first)
    else:
        return torch.sample_euler(noise_pred, x_t, sigma_t, sigma_prev)


def sample_flow_step(velocity, x_t, dt: float):
    return torch.fm_step(velocity, x_t, dt)


def main():
    pass
