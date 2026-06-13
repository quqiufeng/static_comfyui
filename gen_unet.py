#!/usr/bin/env python3
"""gen_unet.py — 生成完整 SDXL UNet forward，偏移硬编码 + skip connections"""
import sys, json

def main():
    idx = json.load(open(sys.argv[1]))
    offs = {}
    for e in idx:
        name = e['name'].replace('model.diffusion_model.', '')
        offs[name] = (e['offset'], e['nelem'], e['shape'])

    def o(name):
        if name in offs: return offs[name][0]
        return -1

    O = lambda n: o(n)  # shortcut

    def ws(name):
        """Lookup weight by dotted name"""
        oo = O(name)
        if oo < 0: return f'#MISSING {name}'
        n = 1
        for d in offs.get(name, [0,0,[0]])[2]: n *= d
        return f'ws(data,{oo},{n})'

    # Output all the boilerplate
    print("""# unet_forward.static.py — 自动生成，所有偏移硬编码
def unet_forward(latent, timestep, context, data, n, hh, ww):
    h_cur: list[float]; _s: list[float] = make_float_array(30)
    ws = w_slice

    # time embedding
    emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)
    emb = linear_torch(emb, ws(data,%s,320*1280), ws(data,%s,320), n, 1280, 1280)
    arr_silu(emb, emb, n*1280)
    emb = linear_torch(emb, ws(data,%s,1280*1280), ws(data,%s,1280), n, 1280, 1280)
""" % (O('time_embed.0.weight'), O('time_embed.0.bias'), O('time_embed.2.weight'), O('time_embed.2.bias')))

    # Helper to emit a ResBlock
    def rb(p, ci, co):
        # p is a function that generates the dotted name for each weight
        def n(suffix): return f'{p}.{suffix}'
        print(f'    group_norm_torch(h_cur, {ws(n("in_layers.0.weight"))}, {ws(n("in_layers.0.bias"))}, 32, {ci}, hh*ww)')
        print(f'    silu_torch(h_cur, n*{ci}*hh*ww)')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("in_layers.2.weight"))}, {ws(n("in_layers.2.bias"))}, n, {ci}, {co}, hh, ww, 3, 1, 1)')
        print(f'    _y = linear_torch(emb, {ws(n("emb_layers.1.weight"))}, {ws(n("emb_layers.1.bias"))}, n, 1280, {co}*2)')
        print(f'    apply_scale_shift(h_cur, _y, n, {co}, hh*ww)')
        print(f'    group_norm_torch(h_cur, {ws(n("out_layers.0.weight"))}, {ws(n("out_layers.0.bias"))}, 32, {co}, hh*ww)')
        print(f'    silu_torch(h_cur, n*{co}*hh*ww)')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("out_layers.3.weight"))}, {ws(n("out_layers.3.bias"))}, n, {co}, {co}, hh, ww, 3, 1, 1)')

    def rb_skip(p, ci, co):
        if ci != co:
            print(f'    _sk = conv2d_torch(h_cur_orig, {ws(p+".skip_connection.weight")}, {ws(p+".skip_connection.bias")}, n, {ci}, {co}, hh, ww, 1, 1, 1)')
        rb(p, ci, co)

    # Input conv
    print(f'    h_cur = conv2d_torch(latent, {ws("input_blocks.0.0.weight")}, {ws("input_blocks.0.0.bias")}, n, 4, 320, hh, ww, 3, 1, 1)')
    print('    _s[0] = h_cur\n')

    # Input blocks 1-2
    for bid in [1,2]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320, 320)
        print(f'    _s[{bid}] = h_cur\n')

    # Down 3
    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.3.0.op.weight")}, {ws("input_blocks.3.0.op.bias")}, n, 320, 320, hh, ww, 3, 2, 1)')
    print('    hh = hh//2; ww = ww//2')
    print('    _s[3] = h_cur\n')

    # Input blocks 4-5
    for bid in [4,5]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320 if bid==4 else 640, 640)
        print(f'    _s[{bid}] = h_cur\n')

    # Down 6
    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.6.0.op.weight")}, {ws("input_blocks.6.0.op.bias")}, n, 640, 640, hh, ww, 3, 2, 1)')
    print('    hh = hh//2; ww = ww//2')
    print('    _s[6] = h_cur\n')

    # Input blocks 7-8
    for bid in [7,8]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 640 if bid==7 else 1280, 1280)
        print(f'    _s[{bid}] = h_cur\n')

    # Middle
    print('    # middle_block')
    rb('middle_block.0', 1280, 1280)
    rb('middle_block.2', 1280, 1280)

    # Output blocks with skip connections
    out_config = [
        (0, 8, 1280, 1280, 2560, 1280, False),
        (1, 7, 1280, 1280, 2560, 1280, False),
        (2, 6, 640, 1280, 1920, 1280, True),
        (3, 5, 640, 1280, 1920, 640, False),
        (4, 4, 640, 640, 1280, 640, False),
        (5, 3, 320, 640, 960, 640, True),
        (6, 2, 320, 640, 960, 320, False),
        (7, 1, 320, 320, 640, 320, False),
        (8, 0, 320, 320, 640, 320, False),
    ]
    for obid, skip_bid, skip_ch, cur_ch, cat_ch, out_ch, do_up in out_config:
        print(f'\n    # output_blocks.{obid} (skip from {skip_bid}:{skip_ch}ch, cur:{cur_ch}ch)')
        # Concat
        print(f'    _cat = make_float_array(n*{cat_ch}*hh*ww)')
        print(f'    _i=0')
        print(f'    while _i<n*{cur_ch}*hh*ww:')
        print(f'        float_array_set(_cat,_i,float_array_ref(h_cur,_i))')
        print(f'        _i=_i+1')
        print(f'    _i=0')
        print(f'    while _i<n*{skip_ch}*hh*ww:')
        print(f'        float_array_set(_cat,n*{cur_ch}*hh*ww+_i,float_array_ref(_s[{skip_bid}],_i))')
        print(f'        _i=_i+1')
        print(f'    h_cur = _cat')
        rb(f'output_blocks.{obid}.0', cat_ch, out_ch)
        if do_up:
            print(f'    h_cur = upsample_nearest(h_cur,n,{out_ch},hh,ww,2)')
            print('    hh = hh*2; ww = ww*2')
            print(f'    h_cur = conv2d_torch(h_cur, {ws(f"output_blocks.{obid}.2.conv.weight")}, {ws(f"output_blocks.{obid}.2.conv.bias")}, n, {out_ch}, {out_ch}, hh, ww, 3, 1, 1)')

    # Output conv
    print(f'\n    h_cur = conv2d_torch(h_cur, {ws("out.0.weight")}, {ws("out.0.bias")}, n, 320, 4, hh, ww, 3, 1, 1)')
    print('    return h_cur')

if __name__ == '__main__':
    main()
