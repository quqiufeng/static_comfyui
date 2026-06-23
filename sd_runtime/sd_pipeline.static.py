# sd_runtime/sd_pipeline.static.py — 采样管线 + SD 完整管线 (Phase 7+8)
# Phase 7:  KSampler / sample() / CFG / sigma 调度
# Phase 8:  SD15Pipeline / SDXLPipeline 完整集成
# 对位 comfyui_ref/comfy/sd.py + sample.py + model_base.py
#
# 连接 Phase 1-7 的所有模块为完整的 txt2img/img2img 管线。
# 这是编译后 ELF 的主入口。
#
# 用法:
#   pipeline = SD15Pipeline(sd_dict, clip, vae, unet_fn)
#   image = pipeline.txt2img("a cat", steps=20, cfg=7.0)

from ops import *
from sd_samplers import *
import math


# ==============================================================================
# Phase 7: Sigma 调度器 + CFG + KSampler + 顶层 sample()
# ==============================================================================

def sigmas_karras(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    return get_sigmas_karras(steps, sigma_min, sigma_max)

def sigmas_exponential(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    return get_sigmas_exponential(steps, sigma_min, sigma_max)

def sigmas_linear(steps: int, sigma_min: float, sigma_max: float) -> ptr:
    return get_sigmas_linear(steps, sigma_min, sigma_max)

SIGMA_SCHEDULERS = {
    "karras": sigmas_karras,
    "exponential": sigmas_exponential,
    "linear": sigmas_linear,
    "simple": sigmas_karras,
}

def get_sigmas(scheduler_name: str, steps: int,
               sigma_min: float = 0.03, sigma_max: float = 14.6) -> ptr:
    fn = SIGMA_SCHEDULERS.get(scheduler_name, SIGMA_SCHEDULERS["karras"])
    return fn(steps, sigma_min, sigma_max)


def predict_noise(unet_fn, x: ptr, sigma: ptr,
                  text_emb_cond: ptr, text_emb_uncond: ptr,
                  cfg_scale: float) -> ptr:
    """UNet forward + CFG → denoised (for sampler).
    UNet returns epsilon; convert to denoised: denoised = x - sigma * eps_cfg.
    """
    if cfg_scale <= 1.0 or text_emb_uncond == null:
        eps = unet_fn(x, sigma, text_emb_cond)
    else:
        cond_eps = unet_fn(x, sigma, text_emb_cond)
        uncond_eps = unet_fn(x, sigma, text_emb_uncond)
        eps_diff = torch_std_sub(cond_eps, uncond_eps)
        eps_scaled = torch_std_mul_scalar(eps_diff, cfg_scale)
        eps = torch_std_add(uncond_eps, eps_scaled)
    sigma_eps = torch_std_mul(sigma, eps)
    return torch_std_sub(x, sigma_eps)


def sample_euler_loop(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg):
    return sample_euler(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg)

def sample_euler_ancestral_loop(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg):
    return sample_euler_ancestral(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg)

def sample_dpmpp_2m_loop(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg):
    return sample_dpmpp_2m(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg)

def sample_ddim_loop(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg):
    return sample_ddim(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg)

def sample_dpmpp_sde_loop(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg):
    return sample_dpmpp_sde(unet_fn, x, sigmas, text_emb_cond, text_emb_uncond, cfg)

SAMPLER_FUNCTIONS = {
    "euler": sample_euler_loop,
    "euler_ancestral": sample_euler_ancestral_loop,
    "dpmpp_2m": sample_dpmpp_2m_loop,
    "dpmpp_sde": sample_dpmpp_sde_loop,
    "ddim": sample_ddim_loop,
}

def get_sampler(sampler_name: str):
    return SAMPLER_FUNCTIONS.get(sampler_name, SAMPLER_FUNCTIONS["euler"])


class KSampler:
    """Multi-step denoising sampler (mirrors comfy.samplers.KSampler)."""
    steps: int
    sampler_name: str
    scheduler_name: str
    sigma_min: float
    sigma_max: float

    def __init__(self, steps: int, sampler_name: str,
                 scheduler_name: str, sigma_min: float, sigma_max: float):
        self.steps = steps
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample(self, noise: ptr, cfg: float,
               unet_fn, text_emb_cond: ptr, text_emb_uncond: ptr) -> ptr:
        sigmas = get_sigmas(self.scheduler_name, self.steps,
                             self.sigma_min, self.sigma_max)
        sampler_fn = get_sampler(self.sampler_name)
        return sampler_fn(unet_fn, noise, sigmas,
                           text_emb_cond, text_emb_uncond, cfg)


def sample(noise: ptr, steps: int, cfg: float,
           sampler_name: str, scheduler_name: str,
           unet_fn,
           text_emb_cond: ptr, text_emb_uncond: ptr,
           sigma_min: float, sigma_max: float,
           denoise: float) -> ptr:
    """Top-level sampling. Supports partial denoise."""
    if denoise < 1.0:
        full_steps = int(steps / denoise)
        sigmas = get_sigmas(scheduler_name, full_steps, sigma_min, sigma_max)
        n_total = int(torch_std_size(sigmas, 0))
        start_idx = n_total - steps - 1
        sigmas = torch_std_narrow(sigmas, 0, start_idx, steps + 1)
    else:
        sigmas = get_sigmas(scheduler_name, steps, sigma_min, sigma_max)
    sampler_fn = get_sampler(sampler_name)
    return sampler_fn(unet_fn, noise, sigmas,
                       text_emb_cond, text_emb_uncond, cfg)


# ==============================================================================
# Phase 8: SD1.5 / SDXL 模型常量 + 管线
# ==============================================================================

SD15_LATENT_CHANNELS: int = 4
SD15_IMAGE_SIZE: int = 512
SD15_LATENT_SIZE: int = 64

SDXL_LATENT_CHANNELS: int = 4
SDXL_IMAGE_SIZE: int = 1024
SDXL_LATENT_SIZE: int = 128


# ==============================================================================
# UNet 包装函数
# ==============================================================================

def make_sd15_unet_fn(sd_dict: ptr) -> function:
    """Create UNet forward wrapper for SD1.5.
    
    Returns: fn(x, sigma, text_emb) → epsilon (noise prediction)
    """
    def unet_fn(x: ptr, sigma: ptr, text_emb: ptr) -> ptr:
        # sigma scalar → expand to match batch dim
        # x: (B, 4, H, W)
        # text_emb: (B, 77, 768)
        return torch_std_sd15_unet_forward_dict(
            sd_dict, x, sigma, text_emb,
            null, null, null, 0, 0.0)  # no LoRA
    return unet_fn


def make_sdxl_unet_fn(sd_dict: ptr,
                       original_size_h: float, original_size_w: float,
                       crop_top: float = 0.0, crop_left: float = 0.0,
                       target_size_h: float = 1024.0, target_size_w: float = 1024.0
                       ) -> function:
    """Create UNet forward wrapper for SDXL with size conditioning.
    
    Returns: fn(x, sigma, text_emb, pooled_emb) → epsilon
    """
    def unet_fn(x: ptr, sigma: ptr, text_emb: ptr, pooled_emb: ptr) -> ptr:
        return torch_std_sdxl_unet_forward(
            sd_dict, x, sigma, text_emb, pooled_emb,
            original_size_h, original_size_w,
            crop_top, crop_left,
            target_size_h, target_size_w)
    return unet_fn


# ==============================================================================
# SD1.5 管线
# ==============================================================================

class SD15Pipeline:
    """Complete SD1.5 txt2img pipeline.
    
    组合 CLIP + UNet + Sampler + VAE 为单入口。
    """
    
    sd_dict: ptr
    clip
    vae
    unet_fn
    lora_A: ptr
    lora_B: ptr
    lora_indices: ptr
    n_lora: int
    
    def __init__(self, sd_dict: ptr, clip, vae, unet_fn):
        """Initialize pipeline.
        
        sd_dict: safetensors dict for UNet weights
        clip: SD1.5 CLIP pipeline (sd1_clip.SD15ClipPipeline)
        vae: VAE module
        unet_fn: UNet forward wrapper fn or null for auto
        """
        self.sd_dict = sd_dict
        self.clip = clip
        self.vae = vae
        if unet_fn:
            self.unet_fn = unet_fn
        else:
            self.unet_fn = make_sd15_unet_fn(sd_dict)
        self.lora_A = null
        self.lora_B = null
        self.lora_indices = null
        self.n_lora = 0
    
    def load_lora(self, lora_dict: ptr, n_weights: int,
                   max_lora: int) -> int:
        """Load LoRA weights and match to UNet.
        
        Uses flat function API from sd_lora.
        Returns: number of matched LoRA pairs
        """
        lora_init(n_weights, max_lora)
        lora_load_from_dict(lora_dict)
        n = lora_match()
        if n > 0:
            self.lora_A = lora_get_A()
            self.lora_B = lora_get_B()
            self.lora_indices = lora_get_indices()
            self.n_lora = n
        return n
    
    def encode_prompt(self, text: str) -> ptr:
        """Encode text prompt to embeddings.
        
        Returns: (1, 77, 768) float32 text embeddings
        """
        return sd1_clip_encode(self.clip, text)
    
    def prepare_noise(self, batch_size: int, height: int, width: int,
                       seed: int) -> ptr:
        """Generate initial random noise latent.
        
        Returns: (B, 4, H/8, W/8) float32 noise
        """
        torch_std_manual_seed(seed)
        latent_h = height // 8
        latent_w = width // 8
        shape = make_int_array(4)
        int_array_set(shape, 0, batch_size)
        int_array_set(shape, 1, SD15_LATENT_CHANNELS)
        int_array_set(shape, 2, latent_h)
        int_array_set(shape, 3, latent_w)
        return torch_std_randn(shape, 4, 0)  # 0 = float32
    
    def txt2img(self, prompt: str, steps: int, cfg: float,
                 sampler_name: str, scheduler_name: str,
                 height: int, width: int, seed: int,
                 denoise: float, sigma_min: float, sigma_max: float) -> ptr:
        """Full txt2img pipeline.
        
        Returns: (1, 3, H, W) float32 image
        """
        text_emb = self.encode_prompt(prompt)
        text_emb_uncond = self.encode_prompt("")
        
        noise = self.prepare_noise(1, height, width, seed)
        
        sampler = KSampler(steps, sampler_name, scheduler_name,
                            sigma_min, sigma_max)
        
        def wrapped_fn(x, sigma, te):
            return self.unet_fn(x, sigma, te)
        
        latent = sampler.sample(noise, cfg, wrapped_fn,
                                 text_emb, text_emb_uncond)
        
        image = torch_std_vae_decode(self.vae, latent)
        return image


# ==============================================================================
# SDXL 管线
# ==============================================================================

class SDXLPipeline:
    """Complete SDXL txt2img pipeline with dual CLIP + size conditioning."""
    
    sd_dict: ptr
    clip
    vae
    unet_fn
    
    def __init__(self, sd_dict: ptr, clip, vae, unet_fn,
                 original_size_h: float, original_size_w: float,
                 crop_top: float, crop_left: float,
                 target_size_h: float, target_size_w: float):
        """Initialize SDXL pipeline."""
        self.sd_dict = sd_dict
        self.clip = clip
        self.vae = vae
        if unet_fn:
            self.unet_fn = unet_fn
        else:
            self.unet_fn = make_sdxl_unet_fn(
                sd_dict, original_size_h, original_size_w,
                crop_top, crop_left,
                target_size_h, target_size_w)
    
    def encode_prompt(self, text: str):
        """Encode prompt → (text_emb, pooled_emb) using SDXL dual CLIP."""
        return sdxl_clip_encode(self.clip, text)
    
    def prepare_noise(self, batch_size: int, height: int, width: int,
                       seed: int) -> ptr:
        torch_std_manual_seed(seed)
        latent_h = height // 8
        latent_w = width // 8
        shape = make_int_array(4)
        int_array_set(shape, 0, batch_size)
        int_array_set(shape, 1, SDXL_LATENT_CHANNELS)
        int_array_set(shape, 2, latent_h)
        int_array_set(shape, 3, latent_w)
        return torch_std_randn(shape, 4, 0)
    
    def txt2img(self, prompt: str, steps: int, cfg: float,
                 sampler_name: str, scheduler_name: str,
                 height: int, width: int, seed: int,
                 denoise: float, sigma_min: float, sigma_max: float) -> ptr:
        """Full SDXL txt2img pipeline.
        
        Returns: (1, 3, H, W) float32 image
        """
        text_emb, pooled_emb = self.encode_prompt(prompt)
        text_emb_uncond, pooled_emb_uncond = self.encode_prompt("")
        
        noise = self.prepare_noise(1, height, width, seed)
        
        sampler = KSampler(steps, sampler_name, scheduler_name,
                            sigma_min, sigma_max)
        
        def wrapped_fn(x, sigma, te):
            return self.unet_fn(x, sigma, te, pooled_emb)
        
        latent = sampler.sample(noise, cfg, wrapped_fn,
                                 text_emb, text_emb_uncond)
        
        image = torch_std_vae_decode(self.vae, latent)
        return image
