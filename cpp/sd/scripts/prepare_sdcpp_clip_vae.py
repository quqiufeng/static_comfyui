#!/usr/bin/env python3
"""Prepare CLIP/VAE safetensors compatible with stable-diffusion.cpp loader.

sd.cpp expects SDXL CLIP keys to be prefixed with:
  text_encoders.clip_l.transformer.
  text_encoders.clip_g.transformer.

It expects VAE keys to be prefixed with:
  vae.

The standalone CLIP/VAE files from ComfyUI/WebUI usually lack these prefixes,
so this script adds them.

Run once:
  python3 scripts/prepare_sdcpp_clip_vae.py
"""
import argparse
from safetensors import safe_open
from safetensors.torch import save_file


def add_prefix(input_path: str, output_path: str, prefix: str) -> None:
    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[prefix + key] = f.get_tensor(key)
    if not tensors:
        raise RuntimeError(f"No tensors found in {input_path}")
    save_file(tensors, output_path)
    print(f"Wrote {len(tensors)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/data/models/image")
    args = parser.parse_args()

    model_dir = args.model_dir

    add_prefix(f"{model_dir}/clip_l.safetensors",
               f"{model_dir}/clip_l_sdcpp.safetensors",
               "text_encoders.clip_l.transformer.")
    add_prefix(f"{model_dir}/clip_g.safetensors",
               f"{model_dir}/clip_g_sdcpp.safetensors",
               "text_encoders.clip_g.transformer.")
    add_prefix(f"{model_dir}/ae.safetensors",
               f"{model_dir}/ae_sdcpp.safetensors",
               "vae.")


if __name__ == "__main__":
    main()
