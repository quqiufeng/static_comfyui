#!/usr/bin/env python3
"""Convert Z-Image Fun-ControlNet (diffusers naming) to stable-diffusion.cpp naming.

The main Z-Image checkpoint is consumed by sd.cpp with merged qkv and `q_norm`/`k_norm`
names, but Fun-ControlNet safetensors ship in diffusers naming. The only differences for
sd.cpp's split-qkv control blocks are the attention norm names:

    attention.norm_q -> attention.q_norm
    attention.norm_k -> attention.k_norm

Usage:
    convert_controlnet.py INPUT.safetensors OUTPUT.safetensors
"""
import json
import struct
import sys

RENAMES = (
    (".attention.norm_q.", ".attention.q_norm."),
    (".attention.norm_k.", ".attention.k_norm."),
)


def read_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    return header, data


def write_safetensors(path, header, data):
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(data)


def convert(src, dst):
    header, data = read_safetensors(src)
    out = {}
    metadata = None
    for key, value in header.items():
        if key == "__metadata__":
            metadata = value
            continue
        name = key
        for old, new in RENAMES:
            name = name.replace(old, new)
        out[name] = value
    if metadata is not None:
        out["__metadata__"] = metadata
    write_safetensors(dst, out, data)
    print(f"converted {len(out)} tensors -> {dst}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    convert(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
