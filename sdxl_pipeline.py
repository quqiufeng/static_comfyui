#!/data/venv/bin/python
"""SDXL txt2img using ComfyUI's native sample() - confirmed working."""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/opt/static_comfyui/ComfyUI')

import torch
from comfy.sd import load_checkpoint_guess_config
import comfy.sample

model_patcher, clip, vae, _ = load_checkpoint_guess_config(
    '/data/models/image/sd_xl_base_1.0.safetensors')

def encode(text):
    tokens = clip.tokenize(text)
    out, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return out.cuda().half(), pooled.cuda().float()

cond, p_pos = encode("solo,single woman,half body portrait of a young woman, "
                     "soft natural lighting, elegant pose, studio lighting, "
                     "sharp eyes, clean white background, medium close up")
uncond, p_neg = encode("blurry, low quality, messy, worst quality, ugly")

positive = [(cond, {'pooled_output': p_pos})]
negative = [(uncond, {'pooled_output': p_neg})]

latent = torch.zeros(1, 4, 128, 128, dtype=torch.float16, device='cuda')
noise = torch.randn(1, 4, 128, 128, dtype=torch.float16, device='cuda')

print("Sampling (ComfyUI native)...")
samples = comfy.sample.sample(model_patcher, noise, 20, 7.0, 'euler', 'normal',
                               positive, negative, latent, denoise=1.0, seed=42)
print(f"Samples: mean={samples.abs().mean():.4f}")

print("Decoding...")
decoded = vae.decode(samples)
img = decoded[0].cpu().float()
if img.dim() == 3 and img.size(0) <= 4:
    img = img.permute(1,2,0)
img = (img * 255).clamp(0, 255).byte().numpy()
from PIL import Image
Image.fromarray(img).save('/home/quqiufeng/python_reference.png')
print(f"Saved, range={img.min()}-{img.max()}")
