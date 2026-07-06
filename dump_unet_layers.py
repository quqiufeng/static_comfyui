#!/data/venv/bin/python
"""Dump SDXL UNet layers manually (hooks don't work with patcher)."""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/opt/static_comfyui/ComfyUI')

import torch
from comfy.sd import load_checkpoint_guess_config
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding

print("Loading checkpoint...")
model_patcher, clip, vae, _ = load_checkpoint_guess_config(
    '/data/models/image/sd_xl_base_1.0.safetensors')
model = model_patcher.model
unet = model.diffusion_model

# Encode real prompts
def encode(text):
    tokens = clip.tokenize(text)
    out, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return out, pooled

cond, pooled = encode("test")
extra = model.extra_conds(pooled_output=pooled, cross_attn=cond,
    width=1024., height=1024., crop_w=0., crop_h=0.,
    target_width=1024., target_height=1024.)
y = extra['y'].cond.cuda().half()
context = extra['c_crossattn'].cond.cuda().half()

# Build inputs matching C++ pipeline
torch.manual_seed(42)
noise = torch.randn(1, 4, 128, 128, dtype=torch.float16, device='cuda')
sigma_val = 14.615
xc = noise / (sigma_val ** 2 + 1.0) ** 0.5
ms = model.model_sampling
t = ms.timestep(torch.tensor([sigma_val])).long().cuda().reshape(1)

# Build time embedding (same as UNet._forward)
t_emb = timestep_embedding(t, unet.model_channels).to(xc.dtype)
emb = unet.time_embed(t_emb)
if y is not None:
    assert y.shape[0] == xc.shape[0]
    emb = emb + unet.label_emb(y)

# Run input_blocks manually
h = xc.cuda().half()
hs = []
for idx, module in enumerate(unet.input_blocks):
    h = forward_timestep_embed(module, h, emb, context, {})
    hs.append(h.detach().cpu().float())
    print(f"ib{idx}: shape={hs[-1].shape}, mean={hs[-1].mean():.4f}, std={hs[-1].std():.4f}")

# Middle
if unet.middle_block is not None:
    h = forward_timestep_embed(unet.middle_block, h, emb, context, {})
    print(f"mid: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")

# Output blocks
for idx, module in enumerate(unet.output_blocks):
    hsp = hs.pop()
    h = torch.cat([h, hsp.cuda().half()], dim=1)
    h = forward_timestep_embed(module, h, emb, context, {})
    print(f"ob{idx}: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")

# Out
h = unet.out(h)
print(f"out: shape={h.shape}, mean={h.mean():.4f}, std={h.std():.4f}")
