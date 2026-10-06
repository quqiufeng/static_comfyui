#!/usr/bin/env python3
"""Convert a diffusers/PEFT z_image LoRA to sd.cpp-compatible key names.

diffusers dreambooth training saves keys like:
    transformer.layers.0.attention.to_q.lora_A.weight
    (or base_model.model.transformer.layers... after some versions)

sd.cpp (ComfyUI-style z_image LoRA) expects:
    diffusion_model.layers.0.attention.to_q.lora_A.weight

Usage:
  python convert_lora.py <in.safetensors> [out.safetensors]
"""
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert_key(k: str) -> str:
    for pre in ("base_model.model.", "base_model."):
        if k.startswith(pre):
            k = k[len(pre):]
    if k.startswith("transformer."):
        k = "diffusion_model." + k[len("transformer."):]
    elif k.startswith("unet."):
        k = "diffusion_model." + k[len("unet."):]
    return k


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_sdcpp.safetensors")

    sd = load_file(str(src))
    out = {convert_key(k): v for k, v in sd.items()}
    save_file(out, str(dst), metadata={"format": "pt", "converted_by": "convert_lora.py"})

    sample = list(out.keys())[0]
    print(f"{len(out)} tensors -> {dst}")
    print(f"  e.g. {sample}")


if __name__ == "__main__":
    main()
