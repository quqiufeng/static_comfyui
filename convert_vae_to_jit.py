#!/usr/bin/env python3
"""Convert SDXL VAE safetensors to TorchScript JIT module using ComfyUI model code."""
import sys, os
sys.path.insert(0, '/data/venv/lib/python3.12/site-packages')
sys.path.insert(0, '/opt/static_comfyui/ComfyUI')

import torch
from safetensors.torch import load_file

# Use ComfyUI's VAE decoder architecture
from comfy.ldm.modules.diffusionmodules.model import Decoder, Encoder
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

sd = load_file("/data/models/image/ae.safetensors")
print(f"Loaded {len(sd)} keys")

# SDXL VAE config
config = {
    'in_channels': 4,
    'out_ch': 3,
    'ch': 512,
    'ch_mult': [1, 2, 4, 4],
    'num_res_blocks': 2,
    'attn_resolutions': [],
    'resolution': 256,
    'z_channels': 4,
    'dropout': 0.0,
    'tanh_out': False,
    'use_linear_attn': False,
}

# Build decoder  
print("Building decoder...")
decoder = Decoder(
    ch=config['ch'],
    ignorekwargs={},
    out_ch=config['out_ch'],
    ch_mult=config['ch_mult'],
    num_res_blocks=config['num_res_blocks'],
    attn_resolutions=config['attn_resolutions'],
    in_channels=config['in_channels'],
    resolution=config['resolution'],
    z_channels=config['z_channels'],
    dropout=config['dropout'],
    tanh_out=config['tanh_out'],
    use_linear_attn=config['use_linear_attn'],
)
    ch=config['ch'],
    ch_mult=config['ch_mult'],
    num_res_blocks=config['num_res_blocks'],
    attn_resolutions=config['attn_resolutions'],
    resolution=config['resolution'],
    dropout=config['dropout'],
)
decoder.eval()

# Remap safetensors keys to decoder state_dict
decoder_sd = decoder.state_dict()
remap = {}
for k, v in sd.items():
    if k.startswith('decoder.'):
        new_k = k[len('decoder.'):]
        remap[new_k] = v

# Load weights
missing, unexpected = decoder.load_state_dict(remap, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  First 5 missing: {missing[:5]}")

# Trace and save
if len(missing) < 10:
    example = torch.randn(1, 4, 128, 128)
    traced = torch.jit.trace(decoder, example)
    torch.jit.save(traced, "/data/models/image/ae_decoder_jit.pt")
    print("Saved to /data/models/image/ae_decoder_jit.pt")
else:
    print("Too many missing keys")
