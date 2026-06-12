#!/usr/bin/env python3
"""export_sd_weights.py — PyTorch/Safetensors → .bin 权重导出
用法: python export_sd_weights.py model.safetensors output_dir/
"""
import sys, os, json, struct

def export_weights(safetensors_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(safetensors_path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        tensor_data = f.read()
    
    index = []
    for name, info in header.items():
        if isinstance(info, dict) and 'dtype' in info:
            n = 1
            for d in info.get('shape', []):
                n *= d
            safe_name = name.replace('.', '_').replace('[', '_').replace(']', '')
            bin_path = os.path.join(output_dir, f"{safe_name}.bin")
            with open(bin_path, 'wb') as out:
                out.write(tensor_data[info['data_offsets'][0]:info['data_offsets'][1]])
            index.append({'name': name, 'file': f"{safe_name}.bin", 'shape': info['shape']})
    
    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Exported {len(index)} tensors to {output_dir}/")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python export_sd_weights.py model.safetensors output_dir/")
        sys.exit(1)
    export_weights(sys.argv[1], sys.argv[2])
