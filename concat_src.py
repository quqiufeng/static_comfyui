#!/usr/bin/env python3
"""Concatenate all comfycli/*.static.py in dependency order for single-file translation."""
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "comfycli")

FILES = [
    "comfy_types.static.py",
    "folder_paths.static.py",
    "cli_args.static.py",
    "supported_models_base.static.py",
    "supported_models.static.py",
    "model_detection.static.py",
    "model_sampling.static.py",
    "latent_formats.static.py",
    "model_base.static.py",
    "model_management.static.py",
    "sd.static.py",
    "lora.static.py",
    "clip_model.static.py",
    "k_diffusion/sampling.static.py",
    "controlnet.static.py",
    "sd_backend.static.py",
    "nodes.static.py",
    "execution.static.py",
    "main.static.py",
]

KEEP_MAIN = {"main.static.py"}

def strip_main_py(content):
    """Remove 'def main():\\n    pass' stub."""
    return re.sub(r'\n\s*def main\(\):\s*\n\s+pass\s*', '', content)

def concat():
    lines = []
    for fname in FILES:
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            content = f.read()
        if fname not in KEEP_MAIN:
            content = strip_main_py(content)
        # Strip from ... import ... (not needed in bundle; all defs inline)
        cleaned = re.sub(r'(?m)^\s*(from\s+\S.*import\s+\S[\s\S]*?)(?=\n\S|\Z)', '', content)
        lines.append(f"# === {fname} ===\n")
        lines.append(cleaned)
        lines.append("\n")
    return "".join(lines)

if __name__ == "__main__":
    output = concat()
    out_path = os.path.join(BASE, "_bundle.static.py")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Written {out_path} ({len(output.splitlines())} lines)")
