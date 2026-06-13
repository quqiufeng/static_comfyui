#!/usr/bin/env python3
"""gen_unet.py — 生成完整 SDXL UNet forward，所有中间张量都是 GPU ptr"""
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

    def ws(name):
        """Lookup weight by dotted name, returns ptr tensor name or 0 for missing"""
        if o(name) < 0:
            return "0"
        return f'_w_{name.replace(".", "_")}'

    def w_shape(name):
        if o(name) < 0: return [0]
        return offs[name][2]

    print('''# unet_forward.static.py — 自动生成，所有偏移硬编码
extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"

def w_slice_1d(data: list[float], offset: int, n: int) -> ptr:
    return st_from_blob_1d(float_array_offset(data, offset), n)

def w_slice_2d(data: list[float], offset: int, d0: int, d1: int) -> ptr:
    return st_from_blob_2d(float_array_offset(data, offset), d0, d1)

def w_slice_4d(data: list[float], offset: int, d0: int, d1: int, d2: int, d3: int) -> ptr:
    return st_from_blob_4d(float_array_offset(data, offset), d0, d1, d2, d3)

extern fn float_array_offset(a: list[float], n: int) -> list[float] from "prelude"

def unet_forward(latent: ptr, timestep: list[float], context: ptr, data: list[float], n: int, hh: int, ww: int) -> ptr:
    h_cur: ptr; _s: ptr = make_ptr_array(30)
    _h_cur_orig: ptr; _sk: ptr; _cat: ptr; _y: ptr; _se: ptr

    # time embedding (inlined)
    _emb = make_float_array(n*320)
    _h = 160
    _i2 = 0
    while _i2 < _h:
        _f = exp(_i2 * (-log(10000.0) / _h))
        _t = float_array_ref(timestep, 0)
        float_array_set(_emb, _i2, cos(_t * _f))
        float_array_set(_emb, _h + _i2, sin(_t * _f))
        _i2 = _i2 + 1
    emb: ptr = st_from_blob_1d(_emb, n*320)''')

    # weight tensor declarations
    for name, (off, nelem, shape) in sorted(offs.items(), key=lambda x: x[1][0]):
        var = f'_w_{name.replace(".", "_")}'
        if len(shape) == 1:
            print(f'    {var}: ptr = w_slice_1d(data, {off}, {shape[0]})')
        elif len(shape) == 2:
            print(f'    {var}: ptr = w_slice_2d(data, {off}, {shape[0]}, {shape[1]})')
        elif len(shape) == 4:
            print(f'    {var}: ptr = w_slice_4d(data, {off}, {shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})')

    o1 = o('time_embed.0.weight'); o2 = o('time_embed.0.bias')
    o3 = o('time_embed.2.weight'); o4 = o('time_embed.2.bias')
    print(f'    emb = linear_torch(emb, {ws("time_embed.0.weight")}, {ws("time_embed.0.bias")}, n, 320, 1280)')
    print('    emb = silu_torch(emb)')
    print(f'    emb = linear_torch(emb, {ws("time_embed.2.weight")}, {ws("time_embed.2.bias")}, n, 1280, 1280)')
    print('    emb = silu_torch(emb)')
    print()

    def rb(p, ci, co):
        def n(suffix): return f'{p}.{suffix}'
        print(f'    _h_cur_orig = st_clone(h_cur)')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(n("in_layers.0.weight"))}, {ws(n("in_layers.0.bias"))}, 32, {ci}, hh, ww)')
        print(f'    h_cur = silu_torch(h_cur)')
        print(f'    print("{p} gn+silu"); print(st_sum(h_cur))')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("in_layers.2.weight"))}, {ws(n("in_layers.2.bias"))}, n, {ci}, {co}, hh, ww, 3, 1, 3//2)')
        print(f'    print("{p} conv1"); print(st_sum(h_cur))')
        print(f'    _se = silu_torch(emb)')
        print(f'    _y = linear_torch(_se, {ws(n("emb_layers.1.weight"))}, {ws(n("emb_layers.1.bias"))}, n, 1280, {co})')
        print(f'    print("{p} emb linear"); print(st_sum(_y))')
        # time emb broadcast add: _y is [n, co], h_cur is [n, co, h, w]
        print(f'    h_cur = add_time_emb_tensor(h_cur, _y, n, {co})')
        print(f'    st_tensor_free(_se); st_tensor_free(_y)')
        print(f'    print("{p} after time add"); print(st_sum(h_cur))')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(n("out_layers.0.weight"))}, {ws(n("out_layers.0.bias"))}, 32, {co}, hh, ww)')
        print(f'    h_cur = silu_torch(h_cur)')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("out_layers.3.weight"))}, {ws(n("out_layers.3.bias"))}, n, {co}, {co}, hh, ww, 3, 1, 3//2)')
        print(f'    print("{p} conv2"); print(st_sum(h_cur))')
        if ci != co:
            print(f'    _sk = conv2d_torch(_h_cur_orig, {ws(p+".skip_connection.weight")}, {ws(p+".skip_connection.bias")}, n, {ci}, {co}, hh, ww, 1, 1, 1//2)')
        else:
            print(f'    _sk = _h_cur_orig')
        print(f'    h_cur = add_tensor(h_cur, _sk)')
        print(f'    st_tensor_free(_sk)')
        if True:
            print(f'    print("{p} rb out"); print(st_sum(h_cur))')

    def spatial_transformer(p, c, inner_dim, n_heads, d_head, hidden, depth):
        print(f'    # SpatialTransformer {p}')
        print(f'    _x_in = st_clone(h_cur)')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(p+".norm.weight")}, {ws(p+".norm.bias")}, 32, {c}, hh, ww)')
        print(f'    _seq = reshape_nchw_to_nlc(h_cur, n, {c}, hh, ww)')
        print(f'    print("{p} st after gn+reshape"); print(st_sum(_seq))')
        print(f'    st_tensor_free(h_cur)')
        print(f'    _seq = linear_torch(_seq, {ws(p+".proj_in.weight")}, {ws(p+".proj_in.bias")}, n*hh*ww, {c}, {inner_dim})')
        print(f'    print("{p} st after proj_in"); print(st_sum(_seq))')
        for d in range(depth):
            pre = f'{p}.transformer_blocks.{d}'
            print(f'    _seq = spatial_transformer_block(_seq, context,')
            print(f'        {ws(pre+".norm1.weight")}, {ws(pre+".norm1.bias")},')
            print(f'        {ws(pre+".norm2.weight")}, {ws(pre+".norm2.bias")},')
            print(f'        {ws(pre+".norm3.weight")}, {ws(pre+".norm3.bias")},')
            print(f'        {ws(pre+".attn1.to_q.weight")}, {ws(pre+".attn1.to_q.bias")},')
            print(f'        {ws(pre+".attn1.to_k.weight")}, {ws(pre+".attn1.to_k.bias")},')
            print(f'        {ws(pre+".attn1.to_v.weight")}, {ws(pre+".attn1.to_v.bias")},')
            print(f'        {ws(pre+".attn1.to_out.0.weight")}, {ws(pre+".attn1.to_out.0.bias")},')
            print(f'        {ws(pre+".attn2.to_q.weight")}, {ws(pre+".attn2.to_q.bias")},')
            print(f'        {ws(pre+".attn2.to_k.weight")}, {ws(pre+".attn2.to_k.bias")},')
            print(f'        {ws(pre+".attn2.to_v.weight")}, {ws(pre+".attn2.to_v.bias")},')
            print(f'        {ws(pre+".attn2.to_out.0.weight")}, {ws(pre+".attn2.to_out.0.bias")},')
            print(f'        {ws(pre+".ff.net.0.proj.weight")}, {ws(pre+".ff.net.0.proj.bias")},')
            print(f'        {ws(pre+".ff.net.2.weight")}, {ws(pre+".ff.net.2.bias")},')
            print(f'        n, hh*ww, {inner_dim}, {n_heads}, {hidden})')
            print(f'    print("{p} st after tb{d}"); print(st_sum(_seq))')
        print(f'    _seq = linear_torch(_seq, {ws(p+".proj_out.weight")}, {ws(p+".proj_out.bias")}, n*hh*ww, {inner_dim}, {c})')
        print(f'    print("{p} st after proj_out"); print(st_sum(_seq))')
        print(f'    _h_img = reshape_nlc_to_nchw(_seq, n, {c}, hh, ww)')
        print(f'    st_tensor_free(_seq)')
        print(f'    h_cur = add_tensor(_x_in, _h_img)')
        print(f'    st_tensor_free(_x_in); st_tensor_free(_h_img)')
        if True:
            print(f'    print("{p} st out"); print(st_sum(h_cur))')

    print(f'    h_cur = conv2d_torch(latent, {ws("input_blocks.0.0.weight")}, {ws("input_blocks.0.0.bias")}, n, 4, 320, hh, ww, 3, 1, 3//2)')
    print('    ptr_array_set(_s, 0, h_cur)\n')

    for bid in [1,2]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320, 320)
        print(f'    ptr_array_set(_s, {bid}, h_cur)\n')

    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.3.0.op.weight")}, {ws("input_blocks.3.0.op.bias")}, n, 320, 320, hh, ww, 3, 2, 1)')
    print('    hh = hh//2; ww = ww//2')
    print('    ptr_array_set(_s, 3, h_cur)\n')

    for bid in [4,5]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320 if bid==4 else 640, 640)
        spatial_transformer(f'input_blocks.{bid}.1', 640, 640, 10, 64, 2560, 2)
        print(f'    ptr_array_set(_s, {bid}, h_cur)\n')

    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.6.0.op.weight")}, {ws("input_blocks.6.0.op.bias")}, n, 640, 640, hh, ww, 3, 2, 1)')
    print('    hh = hh//2; ww = ww//2')
    print('    ptr_array_set(_s, 6, h_cur)\n')

    for bid in [7,8]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 640 if bid==7 else 1280, 1280)
        spatial_transformer(f'input_blocks.{bid}.1', 1280, 1280, 20, 64, 5120, 10)
        print(f'    ptr_array_set(_s, {bid}, h_cur)\n')

    print('    # middle_block')
    rb('middle_block.0', 1280, 1280)
    spatial_transformer('middle_block.1', 1280, 1280, 20, 64, 5120, 10)
    rb('middle_block.2', 1280, 1280)

    out_config = [
        (0, 8, 1280, 1280, 2560, 1280, False),
        (1, 7, 1280, 1280, 2560, 1280, False),
        (2, 6, 1280, 640, 1920, 1280, True),
        (3, 5, 1280, 640, 1920, 640, False),
        (4, 4, 640, 640, 1280, 640, False),
        (5, 3, 640, 640, 1280, 640, True),
        (6, 2, 640, 320, 960, 320, False),
        (7, 1, 320, 320, 640, 320, False),
        (8, 0, 320, 320, 640, 320, False),
    ]
    out_transformers = {
        0: (1280, 1280, 20, 64, 5120, 10),
        1: (1280, 1280, 20, 64, 5120, 10),
        2: (1280, 1280, 20, 64, 5120, 10),
        3: (640, 640, 10, 64, 2560, 2),
        4: (640, 640, 10, 64, 2560, 2),
        5: (640, 640, 10, 64, 2560, 2),
    }

    for obid, skip_bid, skip_ch, cur_ch, cat_ch, out_ch, do_up in out_config:
        print(f'\n    # output_blocks.{obid}')
        print(f'    _cur = h_cur')
        print(f'    _skip = ptr_array_ref(_s, {skip_bid})')
        print(f'    h_cur = cat_channel_tensors(_cur, _skip, n, {cur_ch}, {skip_ch}, hh, ww)')
        print(f'    st_tensor_free(_cur)')
        rb(f'output_blocks.{obid}.0', cat_ch, out_ch)
        if obid in out_transformers:
            cfg = out_transformers[obid]
            spatial_transformer(f'output_blocks.{obid}.1', cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5])
        if do_up:
            print(f'    h_cur = upsample_nearest_torch(h_cur, 2)')
            print('    hh = hh*2; ww = ww*2')
            print(f'    h_cur = conv2d_torch(h_cur, {ws(f"output_blocks.{obid}.2.conv.weight")}, {ws(f"output_blocks.{obid}.2.conv.bias")}, n, {out_ch}, {out_ch}, hh, ww, 3, 1, 1)')

    print(f'\n    h_cur = group_norm_torch(h_cur, {ws("out.0.weight")}, {ws("out.0.bias")}, 32, 320, hh, ww)')
    print(f'    h_cur = silu_torch(h_cur)')
    print(f'    h_cur = conv2d_torch(h_cur, {ws("out.2.weight")}, {ws("out.2.bias")}, n, 320, 4, hh, ww, 3, 1, 1)')
    print('    return h_cur')

if __name__ == '__main__':
    main()
