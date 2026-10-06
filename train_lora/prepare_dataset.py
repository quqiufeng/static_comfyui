#!/usr/bin/env python3
"""Prepare a z_image style-LoRA dataset.

- copies images into <out>/images/  (training --instance_data_dir, images only)
- optional: auto-caption each image with Florence-2 -> <out>/metadata.jsonl
- writes <out>/instance_prompt.txt  (fixed trigger prompt for training)

Usage:
  python prepare_dataset.py <src_dir> <out_dir> [--trigger mystyle] [--caption]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="source directory with style images")
    ap.add_argument("out", help="output dataset directory")
    ap.add_argument("--trigger", default="mystyle", help="trigger token")
    ap.add_argument("--caption", action="store_true", help="auto-caption with Florence-2")
    ap.add_argument("--img2prompt", default="/opt/static_comfyui/cpp/florence2/img2prompt")
    ap.add_argument("--task", default="<DETAILED_CAPTION>")
    ap.add_argument("--beams", default="3")
    ap.add_argument("--max-side", type=int, default=1536, help="downscale long side to at most this (0=keep)")
    ap.add_argument("--instance-prompt", default=None, help="fixed training prompt (default '<trigger> style')")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        sys.exit(f"src not a directory: {src}")
    imgs_dir = out / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images in {src}")

    # phase 1: resize/copy
    saved = []  # (name, saved_path)
    for i, p in enumerate(files):
        im = Image.open(p).convert("RGB")
        if args.max_side and max(im.size) > args.max_side:
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
        suffix = p.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            name = f"{i:03d}.jpg"
            im.save(imgs_dir / name, quality=95)
        else:
            name = f"{i:03d}.png"
            im.save(imgs_dir / name)
        saved.append((name, imgs_dir / name))
    print(f"  resized/copied {len(saved)} images -> {imgs_dir}")

    # phase 2: batch caption (Florence model loaded once)
    caps = {}
    if args.caption:
        paths = "\n".join(str(sp) for _, sp in saved)
        try:
            proc = subprocess.run(
                [args.img2prompt, "--model-dir", os.environ.get("FLORENCE2_MODEL_DIR", "/data/models/florence2"),
                 "--task", args.task, "--beams", str(args.beams), "--batch"],
                input=paths, capture_output=True, text=True,
            )
            lines = [l.strip() for l in proc.stdout.splitlines()]
            if len(lines) != len(saved):
                print(f"  WARN: got {len(lines)} captions for {len(saved)} images")
            for (name, _), cap in zip(saved, lines):
                caps[name] = cap
        except Exception as e:
            print(f"  caption failed: {e}")

    meta = []
    for name, _ in saved:
        text = args.trigger
        cap = caps.get(name, "")
        if cap:
            text = f"{args.trigger}, {cap}"
        meta.append({"file_name": name, "text": text})
        print(f"  {name}  {text[:80]}")

    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    prompt = args.instance_prompt or f"{args.trigger} style"
    (out / "instance_prompt.txt").write_text(prompt)

    print(f"\n{len(files)} images -> {imgs_dir}")
    print(f"instance_prompt: {prompt}")
    print(f"metadata: {out/'metadata.jsonl'}")


if __name__ == "__main__":
    main()
