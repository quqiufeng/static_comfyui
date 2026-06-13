#!/usr/bin/env python3
"""gen_unet.py v2 — 从权重结构生成正确 UNet forward 代码
Block types:
  .0.* = ResBlock (GN→SiLU→Conv→+temb→GN→SiLU→Conv→+skip)
  .1.* = SpatialTransformer (GN→1x1Conv→Attn×N→1x1Conv→+x)
  .2.* = Down/Upsample op
"""
import sys, json
from collections import defaultdict

def load_idx(path):
    with open(path) as f:
        return json.load(f)

def gen():
    idx = load_idx(sys.argv[1])
    unet = [t for t in idx if 'model.diffusion_model' in t['name']]
    
    # Organize by zone.block.sub
    blocks = defaultdict(lambda: defaultdict(list))
    for t in unet:
        name = t['name'].replace('model.diffusion_model.', '')
        pts = name.split('.')
        zone, bid = pts[0], pts[1]
        sub = int(pts[2]) if len(pts) > 2 and pts[2].isdigit() else -1
        rest = '.'.join(pts[3:]) if len(pts) > 3 else ''
        blocks[(zone, bid)][sub].append({'rest': rest, 'shape': t['shape']})
    
    zones = sorted(set(k[0] for k in blocks))
    bids = {}
    for k in blocks:
        if k[0] not in bids: bids[k[0]] = []
        bids[k[0]].append(int(k[1]))
    for z in bids: bids[z].sort()
    
    out = []
    out.append("# Auto-generated SDXL UNet forward")
    out.append("")
    out.append("def unet_forward(latent, timestep, context, weights_dir, n, h, w):")
    out.append("    hh: int = h; ww: int = w")
    out.append("    h_cur: list[float]")
    out.append("")
    
    # Load all weights
    for t in unet:
        name = t['name'].replace('model.diffusion_model.', '')
        safe = name.replace('.', '_')
        nelem = 1
        for d in t['shape']: nelem *= d
        out.append(f'    {safe} = load_bin(weights_dir + "/model_diffusion_model_{safe}.bin", {nelem})')
    
    out.append("")
    out.append("    # time embedding")
    out.append("    emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)")
    out.append("    _y = make_float_array(n * 1280)")
    out.append("    dgemm_row_auto(n, 1280, 1280, 1.0, emb, time_embed_0_weight, 0.0, _y)")
    out.append("    __i=0; TE = n*1280")
    out.append("    while __i < TE: float_array_set(_y,__i,float_array_ref(_y,__i)+float_array_ref(time_embed_0_bias,__i)); __i=__i+1")
    out.append("    arr_silu(_y,_y,TE)")
    out.append("    dgemm_row_auto(n, 1280, 1280, 1.0, _y, time_embed_2_weight, 0.0, _y)")
    out.append("    __i=0; while __i < TE: float_array_set(_y,__i,float_array_ref(_y,__i)+float_array_ref(time_embed_2_bias,__i)); __i=__i+1")
    out.append("    emb = _y")
    out.append("")
    
    # Input conv: 4→320
    out.append("    # input conv: 4→320")
    out.append("    _ho=(hh+2*1-3)//1+1; _wo=(ww+2*1-3)//1+1; _nc=n*_ho*_wo; _kd=4*3*3")
    out.append("    _col=make_float_array(_nc*_kd); im2col(latent,n,4,hh,ww,3,1,1,_col)")
    out.append("    _y=make_float_array(_nc*320); dgemm_row_auto(_nc,320,_kd,1.0,_col,input_blocks_0_0_weight,0.0,_y)")
    out.append("    add_bias(_y,input_blocks_0_0_bias,_nc,320); h_cur=_y")
    out.append("")
    
    # Helper: generate ResBlock
    def gen_resblock(zone, bid, sub, c_in, c_out, has_skip=0):
        p = f'{zone}_{bid}_{sub}'
        out.append(f'    # ResBlock {zone}.{bid}.{sub}: {c_in}→{c_out}')
        # norm1 → silu → conv1
        out.append(f'    group_norm(h_cur, {p}_in_layers_0_weight, {p}_in_layers_0_bias, 32, {c_in}, hh*ww)')
        out.append(f'    arr_silu(h_cur, h_cur, n*{c_in}*hh*ww)')
        out.append(f'    conv2d_inline(h_cur, {p}_in_layers_2_weight, {p}_in_layers_2_bias, n, {c_in}, {c_out}, hh, ww)')
        # time embedding → scale/shift
        out.append(f'    _y=make_float_array(n*1280); dgemm_row_auto(n,{c_out*2},1280,1.0,emb,{p}_emb_layers_1_weight,0.0,_y)')
        out.append(f'    add_bias(_y,{p}_emb_layers_1_bias,n,{c_out*2})')
        if has_skip:
            out.append(f'    _skip=make_float_array(n*{c_out}*hh*ww)')
            out.append(f'    conv2d_inline_skip(h_cur,_skip,{p}_skip_connection_weight,{p}_skip_connection_bias,n,{c_in},{c_out},hh,ww)')
        # norm2 → silu → conv2
        out.append(f'    group_norm(h_cur, {p}_out_layers_0_weight, {p}_out_layers_0_bias, 32, {c_out}, hh*ww)')
        out.append(f'    arr_silu(h_cur, h_cur, n*{c_out}*hh*ww)')
        out.append(f'    conv2d_inline(h_cur, {p}_out_layers_3_weight, {p}_out_layers_3_bias, n, {c_out}, {c_out}, hh, ww)')
        if has_skip:
            out.append(f'    add_arr(h_cur,_skip,n*{c_out}*hh*ww)')
    
    # Generate all input blocks
    out.append("    # === input blocks ===")
    skips = []
    for bid in bids.get('input_blocks', []):
        subs = blocks[('input_blocks', str(bid))]
        for sub in sorted(subs.keys()):
            tensors = subs[sub]
            is_resblock = any('in_layers' in t['rest'] for t in tensors)
            is_attn = any('attn' in t['rest'] for t in tensors)
            is_down = any('op' in t['rest'] and 'conv' in t['rest'] for t in tensors)
            
            if is_resblock:
                # Determine channels
                c_in = 320; c_out = 320
                for t in tensors:
                    if 'in_layers.2.weight' in t['rest']: c_in = t['shape'][1]; c_out = t['shape'][0]
                gen_resblock('input_blocks', bid, sub, c_in, c_out, 1 if c_in != c_out else 0)
            if is_attn:
                out.append(f'    # SpatialTransformer {zone}.{bid}.{sub}')
    
    out.append("")
    out.append("    return h_cur")
    out.append("")
    out.append("# === helper: inline conv2d (im2col + dgemm + bias) ===")
    out.append("# conv2d_inline(src, w, b, n, c_in, c_out, h, w)")
    out.append("# add_bias(arr, bias, n_rows, n_cols)")
    out.append("# add_arr(a, b, n)")
    
    return '\n'.join(out)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: gen_unet.py <index.json>")
        sys.exit(1)
    print(gen())
