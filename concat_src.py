#!/usr/bin/env python3
"""Concatenate all comfycli/*.static.py in dependency order for single-file translation."""
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "comfycli")

# 仅保留 sd.cpp 后端实际使用的模块（torch 时代的模型栈已废弃）。
# 顺序 = 依赖顺序（被依赖者在前）。
FILES = [
    "cli_args.static.py",
    "sd_backend.static.py",
    "nodes.static.py",
    "execution.static.py",
    "main.static.py",
]

# 上游 prelude 无 dict_keys，由 comfycli_ffi.scm 提供 Scheme 实现；
# 这里用一条 import 让类型检查器认识该名字（import 本身不生成代码）。
HEADER = "from comfycli_builtins import dict_keys, is_link, path_dirname, path_split\n\n"

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
    return HEADER + "".join(lines)

if __name__ == "__main__":
    output = concat()
    out_path = os.path.join(BASE, "_bundle.static.py")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Written {out_path} ({len(output.splitlines())} lines)")
