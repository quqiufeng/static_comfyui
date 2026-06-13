#!/usr/bin/env python3
"""gen_unet.py — 从权重偏移生成完整 UNet forward StaticPy 代码"""
import sys, json

def main():
    idx = json.load(open(sys.argv[1]))
    # Build offset map
    offs = {}
    for e in idx:
        name = e['name'].replace('model.diffusion_model.', '')
        offs[name] = (e['offset'], e['nelem'], e['shape'])

    def o(name):
        if name in offs: return offs[name][0]
        full = f'model.diffusion_model.{name}'
        if full in offs: return offs[full][0]
        return -1

    def w(name):
        """ws(data, offset, nelem)"""
        oo = o(name)
        if oo < 0: return None
        n = 1
        for d in offs.get(name, offs.get(f'model.diffusion_model.{name}', [0,0,[0]]))[2]: n *= d
        return f'ws(data,{oo},{n})'

    lines = []
    lines.append('''# unet_forward.static.py — 自动生成，偏移硬编码
def unet_forward(latent, timestep, context, data, n, hh, ww):
    h_cur: list[float]; _s: list[float] = make_float_array(30)
    ws = w_slice

    # time embedding
    emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)
    emb = linear_torch(emb, ws(data,160,320*1280), ws(data,0,320), n, 1280, 1280)
    emb = silu_torch(emb, n*1280)
    emb = linear_torch(emb, ws(data,?,?), ws(data,?,?), n, 1280, 1280)
''')
    # input conv
    i0w = w('input_blocks.0.0.weight')
    i0b = w('input_blocks.0.0.bias')
    if i0w and i0b:
        lines.append(f'    h_cur = conv2d_torch(latent, {i0w}, {i0b}, n, 4, 320, hh, ww, 3, 1, 1)')
        lines.append('    _s[0] = h_cur')

    lines.append('''
    # TODO: generate all 20+ blocks with correct offsets
    return h_cur
''')
    print('\n'.join(lines))
    print(f'// {len(offs)} entries indexed', file=sys.stderr)

if __name__ == '__main__':
    main()
