#!/usr/bin/env python3
"""gen_unet.py — 从权重分析生成 UNet forward StaticPy 代码"""
import sys, json

def main():
    idx = json.load(open(sys.argv[1]))
    unet = [t for t in idx if 'model.diffusion_model' in t['name']]
    
    # Group tensors by block
    blocks = {}
    for t in unet:
        name = t['name'].replace('model.diffusion_model.', '')
        parts = name.split('.')
        zone = parts[0]
        bid = parts[1]
        sub = parts[2] if len(parts) > 2 else ''
        layer = '.'.join(parts[3:])
        key = f"{zone}.{bid}"
        if key not in blocks:
            blocks[key] = {'zone': zone, 'id': bid, 'tensors': []}
        blocks[key]['tensors'].append({'sub': sub, 'layer': layer, 'shape': t['shape'],
            'is_w': 'weight' in name, 'is_b': 'bias' in name})
    
    # Generate code
    print('''
# unet_forward.static.py — SD UNet forward (auto-generated)
# 所有 Conv2d 使用内联 im2col + dgemm 模式

def unet_forward(latent, timestep, context, weights_dir, n, h, w):
    hh: int = h
    ww: int = w
    h_cur: list[float]
''')
    # Load all weights
    for t in unet:
        name = t['name'].replace('model.diffusion_model.', '')
        safe = name.replace('.', '_')
        nelem = 1
        for d in t['shape']: nelem *= d
        print(f'    {safe} = load_bin(weights_dir + "/model_diffusion_model_{safe}.bin", {nelem})')
    
    print()
    print('    # Input Conv: 4→320')
    print('    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1')
    print('    _nc = n * _ho * _wo; _kd = 4 * 3 * 3')
    print('    _col = make_float_array(_nc * _kd)')
    print('    im2col(latent, n, 4, hh, ww, 3, 1, 1, _col)')
    print('    _y = make_float_array(_nc * 320)')
    print('    dgemm_row_auto(_nc, 320, _kd, 1.0, _col, input_blocks_0_0_weight, 0.0, _y)')
    print('    __i = 0')
    print('    while __i < _nc:')
    print('        __j = 0')
    print('        while __j < 320:')
    print('            float_array_set(_y, __i*320+__j, float_array_ref(_y, __i*320+__j) + float_array_ref(input_blocks_0_0_bias, __j))')
    print('            __j = __j + 1')
    print('        __i = __i + 1')
    print('    h_cur = _y')
    
    # Generate all blocks
    for key in sorted(blocks.keys()):
        b = blocks[key]
        # Skip input_blocks.0 (already handled)
        if key == 'input_blocks.0':
            continue
        
        lines = []
        lines.append(f'\n    # {key}')
        for t in b['tensors']:
            lines.append(f'    #   {t["sub"]} {t["layer"]} {t["shape"]}')
        
        is_down = any(t['sub'] == 'op' for t in b['tensors'])
        is_up = any(t['sub'] == 'op' for t in b['tensors'])
        has_attn = any('attn' in t['layer'] for t in b['tensors'])
        has_emb = any('emb' in t['layer'] for t in b['tensors'])
        
        # Find norm1 channels
        norm1_ch = 320
        for t in b['tensors']:
            if 'norm' in t['layer'] and t['is_w'] and t['sub'] == '0':
                norm1_ch = t['shape'][0]
        
        # Find conv1 channels
        conv1_in = norm1_ch
        conv1_out = norm1_ch
        for t in b['tensors']:
            if 'conv' in t['layer'] and t['is_w'] and t['sub'] == '0':
                conv1_in = t['shape'][1]
                conv1_out = t['shape'][0]
        
        lines.append(f'    group_norm(h_cur, input_blocks_{b["id"]}_0_in_layers_0_weight, input_blocks_{b["id"]}_0_in_layers_0_bias, 32, {norm1_ch}, hh*ww)')
        lines.append(f'    arr_silu(h_cur, h_cur, n*{norm1_ch}*hh*ww)')
        
        # Inline conv2d
        lines.append(f'    _ho = (hh + 2*1 - 3)//1 + 1; _wo = (ww + 2*1 - 3)//1 + 1')
        lines.append(f'    _nc = n * _ho * _wo; _kd = {conv1_in} * 3 * 3')
        lines.append(f'    _col = make_float_array(_nc * _kd)')
        lines.append(f'    im2col(h_cur, n, {conv1_in}, hh, ww, 3, 1, 1, _col)')
        lines.append(f'    _y = make_float_array(_nc * {conv1_out})')
        lines.append(f'    dgemm_row_auto(_nc, {conv1_out}, _kd, 1.0, _col, input_blocks_{b["id"]}_0_in_layers_2_weight, 0.0, _y)')
        lines.append(f'    __i = 0')
        lines.append(f'    while __i < _nc:')
        lines.append(f'        __j = 0')
        lines.append(f'        while __j < {conv1_out}:')
        lines.append(f'            float_array_set(_y, __i*{conv1_out}+__j, float_array_ref(_y, __i*{conv1_out}+__j) + float_array_ref(input_blocks_{b["id"]}_0_in_layers_2_bias, __j))')
        lines.append(f'            __j = __j + 1')
        lines.append(f'        __i = __i + 1')
        lines.append(f'    h_cur = _y')
        
        if is_down:
            lines.append(f'    # downsample')
            lines.append(f'    hh = hh//2; ww = ww//2')
        
        # Output line
        lines.append(f'    h_cur = h_cur  # placeholder for skip')
        
        print('\n'.join(lines))
    
    print(f'\n    return h_cur')
    print(f'\n# Total: {len(unet)} tensors, {len(blocks)} blocks', file=sys.stderr)

if __name__ == '__main__':
    main()
