# sd_runtime/sd_models.static.py — Supported models registry + main entry (Phase 8 gap)
# 对位 comfyui_ref/comfy/supported_models.py (2319 行)
#
# Flat functions. 模型配置注册表 + 主入口。

from ops import *
from sd_core import *
from sd_pipeline import *
from sd_flux import *
from sd_t5 import *
from sd_vae import *


# ==============================================================================
# 模型配置注册表
# ==============================================================================

# SD1.5 基础模型配置
SD15_CONFIGS: ptr  # dict: name -> {unet, clip, vae paths}
# SDXL 基础模型配置  
SDXL_CONFIGS: ptr
# FLUX 模型配置
FLUX_CONFIGS: ptr


def models_init() -> void:
    """Initialize model config registry."""
    global SD15_CONFIGS, SDXL_CONFIGS, FLUX_CONFIGS
    
    sd15 = make_dict()
    dict_set(sd15, "sd_v1.5", "sd_v1.5")
    dict_set(sd15, "sd_v1.4", "sd_v1.4")
    SD15_CONFIGS = sd15
    
    sdxl = make_dict()
    dict_set(sdxl, "sdxl_v1.0", "sdxl_v1.0")
    SDXL_CONFIGS = sdxl
    
    flux_cfg = make_dict()
    dict_set(flux_cfg, "flux_schnell", "flux_schnell")
    dict_set(flux_cfg, "flux_dev", "flux_dev")
    FLUX_CONFIGS = flux_cfg


# ==============================================================================
# 按模型名称检测类型
# ==============================================================================

def detect_model_config(dict_ptr: ptr) -> str:
    """Detect model config name from safetensors keys."""
    mt = detect_model_type(dict_ptr)
    if mt == MODEL_TYPE_FLUX:
        return "flux"
    elif mt == MODEL_TYPE_SDXL:
        return "sdxl"
    elif mt == MODEL_TYPE_SD15:
        return "sd15"
    return "unknown"


# ==============================================================================
# 主入口：自动加载 + 推理
# ==============================================================================

def sd_generate(prompt: str,
                model_path: str, clip_path: str, vae_path: str,
                steps: int, cfg: float, height: int, width: int,
                seed: int, sampler: str, scheduler: str,
                lora_path: str) -> ptr:
    """Top-level txt2img entry point.
    
    Loads model, runs inference, returns image.
    Designed to be the 'static_main' entry for the ELF binary.
    
    Args:
        prompt: text prompt
        model_path: path to SD UNet safetensors
        clip_path: path to CLIP JIT .pt
        vae_path: path to VAE JIT .pt
        steps: sampling steps
        cfg: CFG scale
        height/width: output size
        seed: random seed
        sampler: sampler name (euler/euler_ancestral/dpmpp_2m/ddim)
        scheduler: scheduler name (karras/exponential/linear)
        lora_path: optional LoRA path (empty string to skip)
    
    Returns: (1, 3, H, W) float32 image
    """
    # 1. Init subsystems
    core_init()
    models_init()
    
    # 2. Load UNet weights
    sd_dict = torch_std_safetensors_load(model_path)
    mt = detect_model_type(sd_dict)
    
    # 3. Load CLIP
    clip_state = sd1_clip_init(clip_path, "", "", 768, 0)
    
    # 4. Load VAE
    vae_module = torch_std_jit_load(vae_path)
    
    # 5. Optional LoRA
    lora_count = 0
    if lora_path != "":
        lora_dict = torch_std_safetensors_load(lora_path)
        lora_num = torch_std_lora_match_to_unet(lora_dict, 276, null, null, null, 64)
        lora_count = lora_num
    
    # 6. Create UNet wrapper
    unet_fn = make_sd15_unet_fn(sd_dict)
    
    # 7. Create pipeline
    pipeline = SD15Pipeline(sd_dict, clip_state, vae_module, unet_fn)
    
    # 8. Run txt2img
    image = pipeline.txt2img(prompt, steps, cfg, sampler, scheduler,
                              height, width, seed, 1.0, 0.03, 14.6)
    
    return image


# ==============================================================================
# Static entry point (called by compiled ELF)
# ==============================================================================

def static_main() -> void:
    """Entry point called by Chez Scheme runtime.
    
    Reads CLI args, runs sd_generate, saves image.
    """
    prompt = os_getenv("PROMPT")
    if prompt == null:
        prompt = "a cute cat"
    
    model_path = os_getenv("MODEL")
    clip_path = os_getenv("CLIP")
    vae_path = os_getenv("VAE")
    
    if model_path == null:
        exit_program(1)
    
    steps_str = os_getenv("STEPS")
    steps = 20
    if steps_str != null:
        steps = string_to_int(steps_str)
    
    cfg_str = os_getenv("CFG")
    cfg = 7.0
    if cfg_str != null:
        cfg = string_to_float(cfg_str)
    
    seed_str = os_getenv("SEED")
    seed = 42
    if seed_str != null:
        seed = string_to_int(seed_str)
    
    image = sd_generate(prompt, model_path, clip_path, vae_path,
                         steps, cfg, 512, 512, seed,
                         "euler", "karras", "")
    
    output_path = os_getenv("OUTPUT")
    if output_path == null:
        output_path = "output.png"
    
    save_image(image, output_path)
