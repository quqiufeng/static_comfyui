# sd_runtime/sd_pipeline.py — 采样管线（Phase 7: ComfyUI Samplers 对位）
# 1:1 对齐 comfyui_ref/comfy/samplers.py + sample.py
#
# 顶层入口：sample(noise, steps, cfg, sampler_name, scheduler_name,
#                  unet_fn, text_emb_cond, text_emb_uncond)
# 返回 denoised latent。
#
# C++ 函数用于 sigma 调度 + 去噪步骤；StaticPy 侧编排循环 + CFG。

from sd_samplers import *
import math


# ==============================================================================
# Sigma 调度器注册表
# ==============================================================================

def sigmas_karras(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    """Karras sigma schedule (default)."""
    return get_sigmas_karras(steps, sigma_min, sigma_max)


def sigmas_exponential(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    return get_sigmas_exponential(steps, sigma_min, sigma_max)


def sigmas_linear(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    return get_sigmas_linear(steps, sigma_min, sigma_max)


# 注册表
SIGMA_SCHEDULERS = {
    "karras": sigmas_karras,
    "exponential": sigmas_exponential,
    "linear": sigmas_linear,
    "simple": sigmas_karras,  # fallback
}


def get_sigmas(scheduler_name: str, steps: int,
               sigma_min: float = 0.03, sigma_max: float = 14.6) -> ptr:
    """Dispatch to sigma scheduler by name."""
    fn = SIGMA_SCHEDULERS.get(scheduler_name)
    if fn is None:
        fn = SIGMA_SCHEDULERS["karras"]  # default fallback
    return fn(steps, sigma_min, sigma_max)


# ==============================================================================
# LDM ModelSampling 常量（SD1.5 默认值）
# ==============================================================================

# Beta schedule for SD1.5: linear from β_1=0.00085 to β_T=0.012 over T=1000
# These are used for DDIM-style sigma computation from the model_sampling
SD15_BETA_START: float = 0.00085
SD15_BETA_END: float = 0.0120
SD15_NUM_TIMESTEPS: int = 1000


# ==============================================================================
# 模型前向 + CFG 组合 (简化版：不处理 area / control / mask)
# ==============================================================================

def predict_noise(unet_fn, x: ptr, sigma: ptr,
                  text_emb_cond: ptr, text_emb_uncond: ptr,
                  cfg_scale: float) -> ptr:
    """模型前向 + CFG 组合。
    
    unet_fn(sd_dict, x, timestep, text_emb) → noise_pred
    
    如果 cfg_scale == 1.0，则跳过 uncond 分支（优化）。
    """
    if cfg_scale <= 1.0 or text_emb_uncond == null:
        return unet_fn(x, sigma, text_emb_cond)
    
    cond = unet_fn(x, sigma, text_emb_cond)
    uncond = unet_fn(x, sigma, text_emb_uncond)
    
    return uncond + cfg_scale * (cond - uncond)


# ==============================================================================
# 采样器调度
# ==============================================================================

def sample_euler_loop(unet_fn, x: ptr, sigmas: ptr,
                       text_emb_cond: ptr, text_emb_uncond: ptr,
                       cfg_scale: float) -> ptr:
    return sample_euler(unet_fn, x, sigmas,
                         text_emb_cond, text_emb_uncond, cfg_scale)


def sample_euler_ancestral_loop(unet_fn, x: ptr, sigmas: ptr,
                                  text_emb_cond: ptr, text_emb_uncond: ptr,
                                  cfg_scale: float) -> ptr:
    return sample_euler_ancestral(unet_fn, x, sigmas,
                                   text_emb_cond, text_emb_uncond, cfg_scale)


def sample_dpmpp_2m_loop(unet_fn, x: ptr, sigmas: ptr,
                           text_emb_cond: ptr, text_emb_uncond: ptr,
                           cfg_scale: float) -> ptr:
    return sample_dpmpp_2m(unet_fn, x, sigmas,
                            text_emb_cond, text_emb_uncond, cfg_scale)


def sample_ddim_loop(unet_fn, x: ptr, sigmas: ptr,
                      text_emb_cond: ptr, text_emb_uncond: ptr,
                      cfg_scale: float) -> ptr:
    return sample_ddim(unet_fn, x, sigmas,
                        text_emb_cond, text_emb_uncond, cfg_scale)


# 采样器注册表
SAMPLER_FUNCTIONS = {
    "euler": sample_euler_loop,
    "euler_ancestral": sample_euler_ancestral_loop,
    "dpmpp_2m": sample_dpmpp_2m_loop,
    "ddim": sample_ddim_loop,
}


def get_sampler(sampler_name: str):
    """Look up sampler function by name."""
    fn = SAMPLER_FUNCTIONS.get(sampler_name)
    if fn is None:
        fn = SAMPLER_FUNCTIONS["euler"]  # default
    return fn


# ==============================================================================
# KSampler — 主采样类（对位 comfy.samplers.KSampler）
# ==============================================================================

class KSampler:
    """Main sampler class, mirrors comfy.samplers.KSampler.
    
    用法:
        sampler = KSampler(steps=20, sampler_name="euler",
                           scheduler_name="karras")
        result = sampler.sample(noise, steps, cfg,
                                 unet_fn, text_emb_cond, text_emb_uncond)
    """
    
    steps: int
    sampler_name: str
    scheduler_name: str
    sigma_min: float
    sigma_max: float
    
    def __init__(self, steps: int = 20,
                 sampler_name: str = "euler",
                 scheduler_name: str = "karras",
                 sigma_min: float = 0.03,
                 sigma_max: float = 14.6):
        self.steps = steps
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def calculate_sigmas(self, steps: int) -> ptr:
        """Generate sigma schedule for the given step count."""
        return get_sigmas(self.scheduler_name, steps,
                           self.sigma_min, self.sigma_max)
    
    def sample(self, noise: ptr, cfg: float,
                unet_fn, text_emb_cond: ptr, text_emb_uncond: ptr) -> ptr:
        """Run the full denoising loop.
        
        调用采样器循环 + CFG 组合。
        """
        sigmas = self.calculate_sigmas(self.steps)
        sampler_fn = get_sampler(self.sampler_name)
        
        return sampler_fn(unet_fn, noise, sigmas,
                           text_emb_cond, text_emb_uncond, cfg)


# ==============================================================================
# 顶层 sample() 函数（对位 comfy.sample.sample）
# ==============================================================================

def sample(noise: ptr, steps: int, cfg: float,
           sampler_name: str, scheduler_name: str,
           unet_fn,
           text_emb_cond: ptr, text_emb_uncond: ptr,
           sigma_min: float = 0.03,
           sigma_max: float = 14.6,
           denoise: float = 1.0) -> ptr:
    """Top-level sampling function — 主入口。
    
    Args:
        noise: (B, 4, H, W) 随机噪声
        steps: 采样步数 (e.g. 20)
        cfg: 无分类器引导尺度 (e.g. 7.0)
        sampler_name: "euler", "euler_ancestral", "dpmpp_2m", "ddim"
        scheduler_name: "karras", "exponential", "linear", "ddim_uniform"
        unet_fn: function(x, sigma, text_emb) → noise_pred
        text_emb_cond: 条件文本嵌入 (B, 77, 768)
        text_emb_uncond: 无条件文本嵌入 (或 null)
        sigma_min/max: 噪声范围
        denoise: 去噪强度 (1.0 = 完整去噪)
    
    Returns:
        (B, 4, H, W) 去噪后的 latent
    """
    if denoise < 1.0:
        # Partial denoise: use fewer sigmas from the end
        full_steps = int(steps / denoise)
        sigmas = get_sigmas(scheduler_name, full_steps,
                             sigma_min, sigma_max)
        n_total = int(torch_std_size(sigmas, 0))
        # Take the last (steps + 1) sigmas
        start_idx = n_total - steps - 1
        sigmas = torch_std_narrow(sigmas, 0, start_idx, steps + 1)
    else:
        sigmas = get_sigmas(scheduler_name, steps,
                             sigma_min, sigma_max)
    
    sampler_fn = get_sampler(sampler_name)
    return sampler_fn(unet_fn, noise, sigmas,
                       text_emb_cond, text_emb_uncond, cfg)
