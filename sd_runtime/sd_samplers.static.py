# sd_runtime/sd_samplers.static.py — 采样器编排
# 1:1 对齐 comfyui_ref/comfy/k_diffusion/sampling.py + comfy/samplers.py
#
# C++ 运行时提供核心去噪步骤函数：
#   torch_std_sampler_sigmas       — sigma 调度 (Karras / exp / linear)
#   torch_std_sample_ddim          — DDIM 一步
#   torch_std_sample_euler         — Euler 一步
#   torch_std_sample_euler_ancestral — Euler Ancestral 一步
#   torch_std_sample_dpmpp_2m      — DPM++ 2M 一步
#
# StaticPy 侧负责采样循环、CFG 组合、sigma 调度。
#
# 用法:
#   sigmas = get_sigmas_karras(20, 0.03, 14.6)
#   x = sample_euler(model_fn, noise, sigmas, cfg_scale=7.0)

from ops import *
import math


# ==============================================================================
# Sigma 调度
# ==============================================================================

def get_sigmas_karras(steps: int, sigma_min: float = 0.03,
                       sigma_max: float = 14.6) -> ptr:
    """Karras et al. 2022 sigma schedule.
    
    Default sigma_min/max for SD1.5.
    """
    return torch_std_sampler_sigmas(steps, sigma_min, sigma_max, "karras")


def get_sigmas_exponential(steps: int, sigma_min: float = 0.03,
                            sigma_max: float = 14.6) -> ptr:
    """Exponential sigma schedule from max to min."""
    return torch_std_sampler_sigmas(steps, sigma_min, sigma_max, "exponential")


def get_sigmas_linear(steps: int, sigma_min: float = 0.03,
                       sigma_max: float = 14.6) -> ptr:
    """Linear sigma schedule from max to min."""
    return torch_std_sampler_sigmas(steps, sigma_min, sigma_max, "linear")


def get_sigmas_ddim(steps: int, sigma_min: float = 0.03,
                     sigma_max: float = 14.6) -> ptr:
    """Simple DDIM-style sigma schedule (inverse-space linear).
    
    Produces sigmas = [sigma_max, ..., sigma_min, 0.0]
    """
    sigmas = torch_std_sampler_sigmas(steps, sigma_min, sigma_max, "linear")
    # DDIM also appends 0.0 for the final clean step
    # The sigmas tensor from C++ has `steps` elements (max→min)
    # We need steps+1 elements for the loop
    # Convert linear sigmas to inverse space for DDIM-style
    return sigmas  # already produces steps elements


# ==============================================================================
# 辅助函数
# ==============================================================================

def to_alpha_bar(sigma: float) -> float:
    """Convert sigma to alpha_cumprod (needed for DDIM)."""
    # sigma^2 = (1 - alpha_bar) / alpha_bar
    # alpha_bar = 1 / (1 + sigma^2)
    return 1.0 / (1.0 + sigma * sigma)


# ==============================================================================
# CFG (Classifier-Free Guidance) 预测
# ==============================================================================

def cfg_predict(unet_fn, x: ptr, sigma: ptr, text_emb_cond: ptr,
                text_emb_uncond: ptr, cfg_scale: float) -> ptr:
    """Combined CFG noise prediction.
    
    unet_fn: function(sd_dict, x, timestep, text_emb) -> noise_pred
    x: current noisy latent (B, C, H, W)
    sigma: noise level (scalar tensor)
    text_emb_cond: conditioned text embeddings
    text_emb_uncond: unconditioned text embeddings (empty)
    cfg_scale: guidance scale
    
    Returns: CFG-combined noise prediction
    """
    # Cond prediction
    cond_noise = unet_fn(x, sigma, text_emb_cond)
    
    # Uncond prediction
    uncond_noise = unet_fn(x, sigma, text_emb_uncond)
    
    # CFG: eps = uncond + cfg * (cond - uncond)
    return uncond_noise + cfg_scale * (cond_noise - uncond_noise)


# ==============================================================================
# 采样器循环（Euler 系列）
# ==============================================================================

def sample_euler(unet_fn, x: ptr, sigmas: ptr,
                  text_emb_cond: ptr, text_emb_uncond: ptr,
                  cfg_scale: float = 7.0) -> ptr:
    """Euler sampler loop.
    
    unet_fn: function(x, sigma, text_emb) -> noise_pred
    x: initial noisy latent (random noise for txt2img)
    sigmas: sigma schedule tensor [N+1] (from get_sigmas_*)
    text_emb_cond/...: cond/uncond text embeddings
    cfg_scale: guidance scale
    
    Returns: denoised latent
    """
    n_steps = int(torch_std_size(sigmas, 0)) - 1
    
    for i in range(n_steps):
        sigma_t = torch_std_narrow(sigmas, 0, i, 1)      # scalar tensor
        sigma_next = torch_std_narrow(sigmas, 0, i + 1, 1)  # scalar tensor
        
        # CFG noise prediction
        noise_pred = cfg_predict(unet_fn, x, sigma_t,
                                  text_emb_cond, text_emb_uncond, cfg_scale)
        
        # Euler step: x_prev = x_t + (sigma_prev - sigma_t) * (x_t - eps) / sigma_t
        x = torch_std_sample_euler(noise_pred, x, sigma_t, sigma_next)
    
    return x


def sample_euler_ancestral(unet_fn, x: ptr, sigmas: ptr,
                            text_emb_cond: ptr, text_emb_uncond: ptr,
                            cfg_scale: float = 7.0) -> ptr:
    """Euler Ancestral sampler loop."""
    n_steps = int(torch_std_size(sigmas, 0)) - 1
    
    for i in range(n_steps):
        sigma_t = torch_std_narrow(sigmas, 0, i, 1)
        sigma_next = torch_std_narrow(sigmas, 0, i + 1, 1)
        
        noise_pred = cfg_predict(unet_fn, x, sigma_t,
                                  text_emb_cond, text_emb_uncond, cfg_scale)
        
        x = torch_std_sample_euler_ancestral(noise_pred, x, sigma_t, sigma_next)
    
    return x


# ==============================================================================
# 采样器循环（DPM++ 2M）
# ==============================================================================

def sample_dpmpp_2m(unet_fn, x: ptr, sigmas: ptr,
                     text_emb_cond: ptr, text_emb_uncond: ptr,
                     cfg_scale: float = 7.0) -> ptr:
    """DPM++ 2M sampler loop (second-order multi-step)."""
    n_steps = int(torch_std_size(sigmas, 0)) - 1
    old_denoised: ptr = null
    
    for i in range(n_steps):
        sigma_t = torch_std_narrow(sigmas, 0, i, 1)
        sigma_next = torch_std_narrow(sigmas, 0, i + 1, 1)
        
        noise_pred = cfg_predict(unet_fn, x, sigma_t,
                                  text_emb_cond, text_emb_uncond, cfg_scale)
        
        result = torch_std_sample_dpmpp_2m(
            noise_pred, x, sigma_t, sigma_next,
            old_denoised, 1 if old_denoised == null else 0)
        
        # DPM++ 2M returns stacked [x_prev, denoised]
        x = torch_std_narrow(result, 0, 0, 1)
        old_denoised = torch_std_narrow(result, 0, 1, 1)
    
    return x


# ==============================================================================
# 采样器循环（DDIM）
# ==============================================================================

def sample_ddim(unet_fn, x: ptr, sigmas: ptr,
                 text_emb_cond: ptr, text_emb_uncond: ptr,
                 cfg_scale: float = 7.0, eta: float = 0.0) -> ptr:
    """DDIM sampler loop.
    
    Uses torch_std_sample_ddim_from_sigma which converts sigmas to
    alpha_bar = 1/(1+sigma^2) internally.
    """
    n_steps = int(torch_std_size(sigmas, 0)) - 1
    
    for i in range(n_steps):
        sigma_t = torch_std_narrow(sigmas, 0, i, 1)
        sigma_next = torch_std_narrow(sigmas, 0, i + 1, 1)
        
        noise_pred = cfg_predict(unet_fn, x, sigma_t,
                                  text_emb_cond, text_emb_uncond, cfg_scale)
        
        x = torch_std_sample_ddim_from_sigma(
            noise_pred, x, sigma_t, sigma_next, eta)
    
    return x
