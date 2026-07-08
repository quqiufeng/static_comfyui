#!/data/venv/bin/python
"""SDXL txt2img using ComfyUI's native sample() - confirmed working."""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/opt/static_comfyui/ComfyUI')

import torch
import torch.nn.functional as F
from comfy.sd import load_checkpoint_guess_config
import comfy.sample

model_patcher, clip, vae, _ = load_checkpoint_guess_config(
    '/data/models/image/DreamShaperXL_Turbo_v2_1.safetensors')

# Turbo-tuned parameters: lower CFG and fewer steps than standard SDXL
CFG = 3.0
STEPS = 20
HIRES_STEPS = 45
HIRES_STRENGTH = 0.35
SAMPLER = 'euler'
SCHEDULER = 'karras'

# Target 2560x1440 -> latent 320x180
# Low-res pass 1920x1080 -> latent 240x135 (1.33x)
LOW_LATENT_W = 240
LOW_LATENT_H = 135
TARGET_LATENT_W = 320
TARGET_LATENT_H = 180

def encode(text):
    tokens = clip.tokenize(text)
    out, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return out.cuda().half(), pooled.cuda().float()

cond, p_pos = encode("solo,single woman,half body portrait of a young woman, "
                     "soft natural lighting, elegant pose, studio lighting, "
                     "sharp eyes, clean white background, medium close up")
uncond, p_neg = encode("blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, embedding:EasyNegative, embedding:bad-hands-5")

positive = [(cond, {'pooled_output': p_pos})]
negative = [(uncond, {'pooled_output': p_neg})]

# Low-res pass
latent = torch.zeros(1, 4, LOW_LATENT_H, LOW_LATENT_W, dtype=torch.float16, device='cuda')
noise = torch.randn(1, 4, LOW_LATENT_H, LOW_LATENT_W, dtype=torch.float16, device='cuda')

print("Sampling low-res pass (ComfyUI native)...")
samples = comfy.sample.sample(model_patcher, noise, STEPS, CFG, SAMPLER, SCHEDULER,
                               positive, negative, latent, denoise=1.0, seed=42)
print(f"Low-res samples: mean={samples.abs().mean():.4f}")

# Upscale latent to target size
print("Upscaling latent for HiRes pass...")
samples = F.interpolate(samples, size=(TARGET_LATENT_H, TARGET_LATENT_W), mode='bilinear', align_corners=False)

# HiRes pass with new noise at target size
noise_hires = torch.randn(1, 4, TARGET_LATENT_H, TARGET_LATENT_W, dtype=torch.float16, device='cuda')
print("Sampling Hi-res pass...")
samples = comfy.sample.sample(model_patcher, noise_hires, HIRES_STEPS, CFG, SAMPLER, SCHEDULER,
                               positive, negative, samples, denoise=HIRES_STRENGTH, seed=42)
print(f"Hi-res samples: mean={samples.abs().mean():.4f}")

print("Decoding...")
decoded = vae.decode(samples)
img = decoded[0].cpu().float()
if img.dim() == 3 and img.size(0) <= 4:
    img = img.permute(1,2,0)
img = (img * 255).clamp(0, 255).byte().numpy()
from PIL import Image
Image.fromarray(img).save('/home/quqiufeng/python_reference_turbo_2560x1440.png')
print(f"Saved, range={img.min()}-{img.max()}")
