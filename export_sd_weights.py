#!/usr/bin/env python3
"""export_sd_weights.py — PyTorch/Safetensors → float32 .bin 权重导出
用法: python export_sd_weights.py model.safetensors output_dir/

导出规则:
  - 所有权重统一转 float32 后按原 tensor 名保存为 {safe_name}.bin
  - 生成 index.json 记录 name/file/shape/nelem
"""
import sys, os, json
import numpy as np
from safetensors import safe_open

def safe_name(name):
    return name.replace('.', '_').replace('[', '_').replace(']', '')

def export_weights(safetensors_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    index = []
    with safe_open(safetensors_path, framework='np') as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            # 统一转 float32
            if tensor.dtype != np.float32:
                tensor = tensor.astype(np.float32)
            bin_path = os.path.join(output_dir, f"{safe_name(name)}.bin")
            tensor.tofile(bin_path)
            n = int(np.prod(tensor.shape))
            index.append({
                'name': name,
                'file': f"{safe_name(name)}.bin",
                'shape': list(tensor.shape),
                'nelem': n,
            })
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Exported {len(index)} tensors (float32) to {output_dir}/")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_sd_weights.py model.safetensors output_dir/")
        sys.exit(1)
    export_weights(sys.argv[1], sys.argv[2])
