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
extern fn make_ptr_array(n: int) -> ptr from "prelude"
extern fn ptr_array_set(a: ptr, i: int, v: ptr) -> void from "prelude"
extern fn ptr_array_ref(a: ptr, i: int) -> ptr from "prelude"
extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_4d(data: ptr, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_half_flat(data: ptr, n: int) -> ptr from "staticpy_torch"
extern fn st_view_1d(base: ptr, offset: int, d0: int) -> ptr from "staticpy_torch"
extern fn st_view_2d(base: ptr, offset: int, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_view_4d(base: ptr, offset: int, d0: int, d1: int, d2: int, d3: int) -> ptr from "staticpy_torch"
extern fn st_tensor_to_half(t: ptr) -> ptr from "staticpy_torch"

def unet_view_1d(base: ptr, offset: int, n: int) -> ptr:
    return st_view_1d(base, offset, n)

def unet_view_2d(base: ptr, offset: int, d0: int, d1: int) -> ptr:
    return st_view_2d(base, offset, d0, d1)

def unet_view_4d(base: ptr, offset: int, d0: int, d1: int, d2: int, d3: int) -> ptr:
    return st_view_4d(base, offset, d0, d1, d2, d3)

extern fn float_array_offset(a: list[float], n: int) -> list[float] from "prelude"
''')

    weight_entries = sorted(offs.items(), key=lambda x: x[1][0])
    n_weights = len(weight_entries)
    weight_vars = {}
    for wi, (name, (off, nelem, shape)) in enumerate(weight_entries):
        weight_vars[name] = (f'_w_{name.replace(".", "_")}', wi)

    def ws(name):
        if o(name) < 0:
            return "0"
        return f'ptr_array_ref(weights, {weight_vars[name][1] + 1})'

    print(f'''def load_unet_weights(data: list[float]) -> ptr:
    _base: ptr = st_from_blob_half_flat(data, {sum(e[1][1] for e in weight_entries)})
    _weights: ptr = make_ptr_array({n_weights + 1})
    ptr_array_set(_weights, 0, _base)''')

    for wi, (name, (off, nelem, shape)) in enumerate(weight_entries):
        var = weight_vars[name][0]
        if len(shape) == 1:
            print(f'    {var}: ptr = unet_view_1d(_base, {off}, {shape[0]})')
        elif len(shape) == 2:
            print(f'    {var}: ptr = unet_view_2d(_base, {off}, {shape[0]}, {shape[1]})')
        elif len(shape) == 4:
            print(f'    {var}: ptr = unet_view_4d(_base, {off}, {shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})')
        print(f'    ptr_array_set(_weights, {wi + 1}, {var})')

    print('    return _weights')
    print()

    print(f'''def unet_forward(latent: ptr, timestep: list[float], context: ptr, y: ptr, weights: ptr, n: int, hh: int, ww: int) -> ptr:
    h_cur: ptr; _s: ptr = make_ptr_array(30)
    _h_cur_orig: ptr; _sk: ptr; _cat: ptr; _y: ptr; _se: ptr; _h_old: ptr

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
    emb: ptr = st_from_blob_1d(_emb, n*320)
    emb = st_tensor_to_half(emb)''')

    o1 = o('time_embed.0.weight'); o2 = o('time_embed.0.bias')
    o3 = o('time_embed.2.weight'); o4 = o('time_embed.2.bias')
    print(f'    emb = linear_torch(emb, {ws("time_embed.0.weight")}, {ws("time_embed.0.bias")}, n, 320, 1280)')
    print('    emb = silu_torch(emb)')
    print(f'    emb = linear_torch(emb, {ws("time_embed.2.weight")}, {ws("time_embed.2.bias")}, n, 1280, 1280)')
    print('    emb = silu_torch(emb)')
    print(f'    _y_emb = linear_torch(y, {ws("label_emb.0.0.weight")}, {ws("label_emb.0.0.bias")}, n, 2816, 1280)')
    print('    _y_emb = silu_torch(_y_emb)')
    print(f'    _y_emb = linear_torch(_y_emb, {ws("label_emb.0.2.weight")}, {ws("label_emb.0.2.bias")}, n, 1280, 1280)')
    print('    emb = add_tensor(emb, _y_emb)')
    print('    st_tensor_free(_y_emb)')
    print()

    def rb(p, ci, co):
        def n(suffix): return f'{p}.{suffix}'
        print(f'    _h_cur_orig = st_clone(h_cur)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(n("in_layers.0.weight"))}, {ws(n("in_layers.0.bias"))}, 32, {ci}, hh, ww)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = silu_torch(h_cur)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("in_layers.2.weight"))}, {ws(n("in_layers.2.bias"))}, n, {ci}, {co}, hh, ww, 3, 1, 3//2)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _se = silu_torch(emb)')
        print(f'    _y = linear_torch(_se, {ws(n("emb_layers.1.weight"))}, {ws(n("emb_layers.1.bias"))}, n, 1280, {co})')
        # time emb broadcast add: _y is [n, co], h_cur is [n, co, h, w]
        print(f'    _h_old = h_cur')
        print(f'    h_cur = add_time_emb_tensor(h_cur, _y, n, {co})')
        print(f'    st_tensor_free(_h_old)')
        print(f'    st_tensor_free(_se); st_tensor_free(_y)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(n("out_layers.0.weight"))}, {ws(n("out_layers.0.bias"))}, 32, {co}, hh, ww)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = silu_torch(h_cur)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = conv2d_torch(h_cur, {ws(n("out_layers.3.weight"))}, {ws(n("out_layers.3.bias"))}, n, {co}, {co}, hh, ww, 3, 1, 3//2)')
        print(f'    st_tensor_free(_h_old)')
        if ci != co:
            print(f'    _sk = conv2d_torch(_h_cur_orig, {ws(p+".skip_connection.weight")}, {ws(p+".skip_connection.bias")}, n, {ci}, {co}, hh, ww, 1, 1, 1//2)')
            print(f'    st_tensor_free(_h_cur_orig)')
        else:
            print(f'    _sk = _h_cur_orig')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = add_tensor(h_cur, _sk)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    st_tensor_free(_sk)')

    def spatial_transformer(p, c, inner_dim, n_heads, d_head, hidden, depth):
        print(f'    # SpatialTransformer {p}')
        print(f'    _x_in = st_clone(h_cur)')
        print(f'    _h_old = h_cur')
        print(f'    h_cur = group_norm_torch(h_cur, {ws(p+".norm.weight")}, {ws(p+".norm.bias")}, 32, {c}, hh, ww)')
        print(f'    st_tensor_free(_h_old)')
        print(f'    _seq = reshape_nchw_to_nlc(h_cur, n, {c}, hh, ww)')
        print(f'    st_tensor_free(h_cur)')
        print(f'    _seq = linear_torch(_seq, {ws(p+".proj_in.weight")}, {ws(p+".proj_in.bias")}, n*hh*ww, {c}, {inner_dim})')
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
        print(f'    _seq = linear_torch(_seq, {ws(p+".proj_out.weight")}, {ws(p+".proj_out.bias")}, n*hh*ww, {inner_dim}, {c})')
        print(f'    _h_img = reshape_nlc_to_nchw(_seq, n, {c}, hh, ww)')
        print(f'    st_tensor_free(_seq)')
        print(f'    h_cur = add_tensor(_x_in, _h_img)')
        print(f'    st_tensor_free(_x_in); st_tensor_free(_h_img)')

    print(f'    h_cur = conv2d_torch(latent, {ws("input_blocks.0.0.weight")}, {ws("input_blocks.0.0.bias")}, n, 4, 320, hh, ww, 3, 1, 3//2)')
    print('    _ss_0: ptr = st_clone(h_cur)\n    ptr_array_set(_s, 0, _ss_0)\n')

    for bid in [1,2]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320, 320)
        print(f'    _ss_{bid}: ptr = st_clone(h_cur)\n    ptr_array_set(_s, {bid}, _ss_{bid})\n')

    print(f'    _h_old = h_cur')
    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.3.0.op.weight")}, {ws("input_blocks.3.0.op.bias")}, n, 320, 320, hh, ww, 3, 2, 1)')
    print(f'    st_tensor_free(_h_old)')
    print('    hh = hh//2; ww = ww//2')
    print('    _ss_3: ptr = st_clone(h_cur)\n    ptr_array_set(_s, 3, _ss_3)\n')

    for bid in [4,5]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 320 if bid==4 else 640, 640)
        spatial_transformer(f'input_blocks.{bid}.1', 640, 640, 10, 64, 2560, 2)
        print(f'    _ss_{bid}: ptr = st_clone(h_cur)\n    ptr_array_set(_s, {bid}, _ss_{bid})\n')

    print(f'    _h_old = h_cur')
    print(f'    h_cur = conv2d_torch(h_cur, {ws("input_blocks.6.0.op.weight")}, {ws("input_blocks.6.0.op.bias")}, n, 640, 640, hh, ww, 3, 2, 1)')
    print(f'    st_tensor_free(_h_old)')
    print('    hh = hh//2; ww = ww//2')
    print('    _ss_6: ptr = st_clone(h_cur)\n    ptr_array_set(_s, 6, _ss_6)\n')

    for bid in [7,8]:
        print(f'    # input_blocks.{bid}')
        rb(f'input_blocks.{bid}.0', 640 if bid==7 else 1280, 1280)
        spatial_transformer(f'input_blocks.{bid}.1', 1280, 1280, 20, 64, 5120, 10)
        print(f'    _ss_{bid}: ptr = st_clone(h_cur)\n    ptr_array_set(_s, {bid}, _ss_{bid})\n')

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

    for obid, skip_bid, cur_ch, skip_ch, cat_ch, out_ch, do_up in out_config:
        print(f'\n    # output_blocks.{obid}')
        print(f'    _cur = h_cur')
        print(f'    _skip = ptr_array_ref(_s, {skip_bid})')
        print(f'    h_cur = cat_channel_tensors(_cur, _skip, n, {cur_ch}, {skip_ch}, hh, ww)')
        print(f'    st_tensor_free(_cur); st_tensor_free(_skip)')
        rb(f'output_blocks.{obid}.0', cat_ch, out_ch)
        if obid in out_transformers:
            cfg = out_transformers[obid]
            spatial_transformer(f'output_blocks.{obid}.1', cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5])
        if do_up:
            print(f'    _h_old = h_cur')
            print(f'    h_cur = upsample_nearest_torch(h_cur, 2)')
            print(f'    st_tensor_free(_h_old)')
            print('    hh = hh*2; ww = ww*2')
            print(f'    _h_old = h_cur')
            print(f'    h_cur = conv2d_torch(h_cur, {ws(f"output_blocks.{obid}.2.conv.weight")}, {ws(f"output_blocks.{obid}.2.conv.bias")}, n, {out_ch}, {out_ch}, hh, ww, 3, 1, 1)')
            print(f'    st_tensor_free(_h_old)')

    print(f'\n    _h_out = h_cur')
    print(f'    h_cur = group_norm_torch(h_cur, {ws("out.0.weight")}, {ws("out.0.bias")}, 32, 320, hh, ww)')
    print(f'    st_tensor_free(_h_out)')
    print(f'    _h_out = h_cur')
    print(f'    h_cur = silu_torch(h_cur)')
    print(f'    st_tensor_free(_h_out)')
    print(f'    _h_out = h_cur')
    print(f'    h_cur = conv2d_torch(h_cur, {ws("out.2.weight")}, {ws("out.2.bias")}, n, 320, 4, hh, ww, 3, 1, 1)')
    print(f'    st_tensor_free(_h_out)')
    print('    result: ptr = h_cur')
    print('    return result')

if __name__ == '__main__':
    main()
