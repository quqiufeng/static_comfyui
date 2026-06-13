# unet_forward.static.py — SDXL UNet complete forward
# 所有块逐一调用，skip connections 明确路由
# 权重: weights_dir/model_diffusion_model_*.bin

def unet_forward(latent, timestep, context, weights_dir, n, hh, ww):
    _y: list[float]; _s: list[float] = make_float_array(30)
    _i: int; _j: int
    total: int = n * 4 * hh * ww
    
    # 时间步嵌入 1280ch
    _emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)
    _y = make_float_array(n * 1280)
    dgemm_row_auto(n, 1280, 1280, 1.0, _emb, time_embed_0_weight, 0.0, _y)
    _i = 0; while _i < n*1280: float_array_set(_y,_i,float_array_ref(_y,_i)+float_array_ref(time_embed_0_bias,_i)); _i=_i+1
    arr_silu(_y,_y,n*1280)
    dgemm_row_auto(n,1280,1280,1.0,_y,time_embed_2_weight,0.0,_y)
    _i = 0; while _i < n*1280: float_array_set(_y,_i,float_array_ref(_y,_i)+float_array_ref(time_embed_2_bias,_i)); _i=_i+1
    emb = _y

    # 输入卷积 4→320
    h_cur = conv2d_inline(latent, input_blocks_0_0_weight, input_blocks_0_0_bias, n, 4, 320, hh, ww)
    _s[0] = h_cur                                    # skip 0: 320@8

    # input_blocks 1-8
    h_cur = rb(h_cur,emb,n,320,320,hh,ww, 'input_blocks_1_0')
    _s[1] = h_cur                                    # skip 1: 320@8
    h_cur = rb(h_cur,emb,n,320,320,hh,ww, 'input_blocks_2_0')
    _s[2] = h_cur                                    # skip 2: 320@8
    # downsample: 320→320, /2
    h_cur = conv2d_inline(h_cur, input_blocks_3_0_op_weight, input_blocks_3_0_op_bias, n, 320, 320, hh, ww)
    hh = hh//2; ww = ww//2
    _s[3] = h_cur                                    # skip 3: 320@4
    # block 4: ResBlock 320→640 + Attention
    h_cur = rb(h_cur,emb,n,320,640,hh,ww, 'input_blocks_4_0')
    h_cur = tf(h_cur,context,n,640,hh,ww, 8, 512, 'input_blocks_4_1')
    _s[4] = h_cur                                    # skip 4: 640@4
    # block 5: ResBlock 640→640 + Attention
    h_cur = rb(h_cur,emb,n,640,640,hh,ww, 'input_blocks_5_0')
    h_cur = tf(h_cur,context,n,640,hh,ww, 8, 512, 'input_blocks_5_1')
    _s[5] = h_cur                                    # skip 5: 640@4
    # downsample: 640→640, /2
    h_cur = conv2d_inline(h_cur, input_blocks_6_0_op_weight, input_blocks_6_0_op_bias, n, 640, 640, hh, ww)
    hh = hh//2; ww = ww//2
    _s[6] = h_cur                                    # skip 6: 640@2
    # block 7: ResBlock 640→1280 + Attention
    h_cur = rb(h_cur,emb,n,640,1280,hh,ww, 'input_blocks_7_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'input_blocks_7_1')
    _s[7] = h_cur                                    # skip 7: 1280@2
    # block 8: ResBlock 1280→1280 + Attention
    h_cur = rb(h_cur,emb,n,1280,1280,hh,ww, 'input_blocks_8_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'input_blocks_8_1')
    _s[8] = h_cur                                    # skip 8: 1280@2

    # middle block: ResBlock + Attention + ResBlock
    h_cur = rb(h_cur,emb,n,1280,1280,hh,ww, 'middle_block_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'middle_block_1')
    h_cur = rb(h_cur,emb,n,1280,1280,hh,ww, 'middle_block_2')

    # output blocks 0-8 (with skip connections from input blocks)
    # out 0: concat _s[8] (1280) + h_cur (1280) = 2560 → 1280
    h_cur = cat(h_cur,_s[8],n,1280,1280,hh,ww)
    h_cur = rb(h_cur,emb,n,2560,1280,hh,ww, 'output_blocks_0_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'output_blocks_0_1')
    # out 1: concat _s[7] (1280) + h_cur (1280) = 2560 → 1280
    h_cur = cat(h_cur,_s[7],n,1280,1280,hh,ww)
    h_cur = rb(h_cur,emb,n,2560,1280,hh,ww, 'output_blocks_1_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'output_blocks_1_1')
    # out 2: concat _s[6] (640) + h_cur (1280) = 1920 → 1280 + upsample
    h_cur = cat(h_cur,_s[6],n,1280,640,hh,ww)
    h_cur = rb(h_cur,emb,n,1920,1280,hh,ww, 'output_blocks_2_0')
    h_cur = tf(h_cur,context,n,1280,hh,ww, 8, 1024, 'output_blocks_2_1')
    h_cur = upsample_nearest(h_cur,n,1280,hh,ww,2); hh=hh*2; ww=ww*2
    h_cur = conv2d_inline(h_cur, output_blocks_2_2_conv_weight, output_blocks_2_2_conv_bias, n, 1280, 1280, hh, ww)
    # out 3: concat _s[5] (640) + h_cur (1280) = 1920 → 640
    h_cur = cat(h_cur,_s[5],n,1280,640,hh,ww)
    h_cur = rb(h_cur,emb,n,1920,640,hh,ww, 'output_blocks_3_0')
    h_cur = tf(h_cur,context,n,640,hh,ww, 8, 512, 'output_blocks_3_1')
    # out 4: concat _s[4] (640) + h_cur (640) = 1280 → 640
    h_cur = cat(h_cur,_s[4],n,640,640,hh,ww)
    h_cur = rb(h_cur,emb,n,1280,640,hh,ww, 'output_blocks_4_0')
    h_cur = tf(h_cur,context,n,640,hh,ww, 8, 512, 'output_blocks_4_1')
    # out 5: concat _s[3] (320) + h_cur (640) = 960 → 640 + upsample
    h_cur = cat(h_cur,_s[3],n,640,320,hh,ww)
    h_cur = rb(h_cur,emb,n,960,640,hh,ww, 'output_blocks_5_0')
    h_cur = tf(h_cur,context,n,640,hh,ww, 8, 512, 'output_blocks_5_1')
    h_cur = upsample_nearest(h_cur,n,640,hh,ww,2); hh=hh*2; ww=ww*2
    h_cur = conv2d_inline(h_cur, output_blocks_5_2_conv_weight, output_blocks_5_2_conv_bias, n, 640, 640, hh, ww)
    # out 6: concat _s[2] (320) + h_cur (640) = 960 → 320
    h_cur = cat(h_cur,_s[2],n,640,320,hh,ww)
    h_cur = rb(h_cur,emb,n,960,320,hh,ww, 'output_blocks_6_0')
    # out 7: concat _s[1] (320) + h_cur (320) = 640 → 320
    h_cur = cat(h_cur,_s[1],n,320,320,hh,ww)
    h_cur = rb(h_cur,emb,n,640,320,hh,ww, 'output_blocks_7_0')
    # out 8: concat _s[0] (320) + h_cur (320) = 640 → 320
    h_cur = cat(h_cur,_s[0],n,320,320,hh,ww)
    h_cur = rb(h_cur,emb,n,640,320,hh,ww, 'output_blocks_8_0')

    # 输出卷积 320→4
    h_cur = conv2d_inline(h_cur, out_0_weight, out_0_bias, n, 320, 4, hh, ww)
    return h_cur

# ═══════ 辅助块函数 ═══════

def rb(x, emb, n, c_in, c_out, hh, ww, p):
    """ResBlock: GN→SiLU→Conv→+temb→GN→SiLU→Conv→+skip"""
    group_norm(x, load_bn(p+'_in_layers_0_weight'), load_bn(p+'_in_layers_0_bias'), 32, c_in, hh*ww)
    arr_silu(x, x, n*c_in*hh*ww)
    x = conv2d_inline(x, load_bn(p+'_in_layers_2_weight'), load_bn(p+'_in_layers_2_bias'), n, c_in, c_out, hh, ww)
    _y = make_float_array(n*c_out*2)
    dgemm_row_auto(n, c_out*2, 1280, 1.0, emb, load_bn(p+'_emb_layers_1_weight'), 0.0, _y)
    _i=0;while _i<n*c_out*2:float_array_set(_y,_i,float_array_ref(_y,_i)+float_array_ref(load_bn(p+'_emb_layers_1_bias'),_i));_i=_i+1
    apply_scale_shift(x, _y, n, c_out, hh*ww)
    group_norm(x, load_bn(p+'_out_layers_0_weight'), load_bn(p+'_out_layers_0_bias'), 32, c_out, hh*ww)
    arr_silu(x, x, n*c_out*hh*ww)
    x = conv2d_inline(x, load_bn(p+'_out_layers_3_weight'), load_bn(p+'_out_layers_3_bias'), n, c_out, c_out, hh, ww)
    if c_in != c_out:
        _sk = conv2d_inline(x, load_bn(p+'_skip_connection_weight'), load_bn(p+'_skip_connection_bias'), n, c_in, c_out, hh, ww)
        _i=0;while _i<n*c_out*hh*ww:float_array_set(x,_i,float_array_ref(x,_i)+float_array_ref(_sk,_i));_i=_i+1
    return x

def tf(x, ctx, n, c, hh, ww, heads, dim, p):
    """SpatialTransformer: GN→proj_in→Attention→proj_out→+x"""
    _src: list[float] = make_float_array(n*c*hh*ww)
    _i=0;while _i<n*c*hh*ww:float_array_set(_src,_i,float_array_ref(x,_i));_i=_i+1
    group_norm(x, load_bn(p+'_norm_weight'), load_bn(p+'_norm_bias'), 32, c, hh*ww)
    x = conv2d_inline(x, load_bn(p+'_proj_in_weight'), load_bn(p+'_proj_in_bias'), n, c, c, hh, ww)
    # reshape [n,c,h,w] → [n,h*w,c]
    _tok = hh*ww; _flat = make_float_array(n*_tok*c)
    _b=0;while _b<n:_ch=0;while _ch<c:_t=0;while _t<_tok:float_array_set(_flat,(_b*_tok+_t)*c+_ch,float_array_ref(x,((_b*c+_ch)*hh)*ww+_t));_t=_t+1;_ch=_ch+1;_b=_b+1
    # attention: QKV
    _q = make_float_array(n*_tok*c); dgemm_row_auto(n*_tok,c,c,1.0,_flat,load_bn(p+'_transformer_blocks_0_attn1_to_q_weight'),0.0,_q)
    _k = make_float_array(n*_tok*c); dgemm_row_auto(n*_tok,c,c,1.0,_flat,load_bn(p+'_transformer_blocks_0_attn1_to_k_weight'),0.0,_k)
    _v = make_float_array(n*_tok*c); dgemm_row_auto(n*_tok,c,c,1.0,_flat,load_bn(p+'_transformer_blocks_0_attn1_to_v_weight'),0.0,_v)
    # self-attention
    _o = attention_sd(_q,_k,_v,n,_tok,_tok,c,heads)
    # proj_out
    _p2 = make_float_array(n*_tok*c); dgemm_row_auto(n*_tok,c,c,1.0,_o,load_bn(p+'_transformer_blocks_0_attn1_to_out_0_weight'),0.0,_p2)
    _i=0;while _i<n*_tok*c:float_array_set(_p2,_i,float_array_ref(_p2,_i)+float_array_ref(load_bn(p+'_transformer_blocks_0_attn1_to_out_0_bias'),_i));_i=_i+1
    # reshape back + skip
    _hb = make_float_array(n*c*hh*ww)
    _b=0;while _b<n:_ch=0;while _ch<c:_t=0;while _t<_tok:float_array_set(_hb,((_b*c+_ch)*hh)*ww+_t,float_array_ref(_p2,(_b*_tok+_t)*c+_ch));_t=_t+1;_ch=_ch+1;_b=_b+1
    _i=0;while _i<n*c*hh*ww:float_array_set(_hb,_i,float_array_ref(_hb,_i)+float_array_ref(_src,_i));_i=_i+1
    return _hb

def cat(a, b, n, ca, cb, hh, ww):
    """在通道维拼接: [n,ca,h,w] + [n,cb,h,w] → [n,ca+cb,h,w]"""
    _r = make_float_array(n*(ca+cb)*hh*ww)
    _i=0;while _i<n*ca*hh*ww:float_array_set(_r,_i,float_array_ref(a,_i));_i=_i+1
    _i=0;while _i<n*cb*hh*ww:float_array_set(_r,n*ca*hh*ww+_i,float_array_ref(b,_i));_i=_i+1
    return _r

def load_bn(p):
    """占位: 实际权重从 weights_dir 加载"""
    return make_float_array(1)
