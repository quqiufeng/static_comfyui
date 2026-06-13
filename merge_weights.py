#!/usr/bin/env python3
"""merge_weights.py — 将 per-tensor float32 .bin 合并为单个 weights.bin + offset 索引
用法: python merge_weights.py per_tensor_dir/ merged_dir/

索引里的 offset 是 float32 元素偏移，StaticPy 用 w_slice(data, offset, nelem) 切分。
"""
import sys, os, json

def merge(input_dir, output_dir, prefix=None):
    os.makedirs(output_dir, exist_ok=True)
    idx = json.load(open(os.path.join(input_dir, 'index.json')))
    if prefix:
        idx = [e for e in idx if e['name'].startswith(prefix)]
    out_path = os.path.join(output_dir, 'weights.bin')
    offset = 0
    merged = []
    with open(out_path, 'wb') as out:
        for e in idx:
            path = os.path.join(input_dir, e['file'])
            data = open(path, 'rb').read()
            # 必须是 4 字节倍数以对齐 float32
            if len(data) % 4 != 0:
                raise ValueError(f"{e['file']} size {len(data)} not multiple of 4")
            nelem = len(data) // 4
            if 'nelem' in e and e['nelem'] != nelem:
                print(f"Warning: {e['name']} nelem mismatch {e['nelem']} vs {nelem}")
            out.write(data)
            merged.append({
                'name': e['name'],
                'offset': offset,
                'nelem': nelem,
                'shape': e['shape'],
            })
            offset += nelem
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(merged)} tensors -> {out_path} ({offset} floats, {offset*4/1e9:.2f} GB)")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python merge_weights.py input_dir/ output_dir/ [prefix]")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
