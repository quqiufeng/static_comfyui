
# unet_forward.static.py — 自动生成，所有权重偏移硬编码
def unet_forward(latent, timestep, context, data, n, hh, ww):
    h_cur: list[float]; _s: list[float] = make_float_array(30)
    ws = w_slice

    # time embedding
    emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)
    emb = linear_torch(emb, ws(data,{o('time_embed.0.weight')},320*1280), ws(data,{o('time_embed.0.bias')},320), n, 1280, 1280)
    arr_silu(emb, emb, n*1280)
    emb = linear_torch(emb, ws(data,{o('time_embed.2.weight')},1280*1280), ws(data,{o('time_embed.2.bias')},1280), n, 1280, 1280)

    h_cur = conv2d_torch(latent, ws(data,160,11520), ws(data,0,320), n, 4, 320, hh, ww, 3, 1, 1)
    _s[0] = h_cur

    # input_blocks

    # input_blocks_1_0
    group_norm_torch(h_cur, # MISSING input_blocks_1_0_in_layers_0_weight, # MISSING input_blocks_1_0_in_layers_0_bias, 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_1_0_in_layers_2_weight, # MISSING input_blocks_1_0_in_layers_2_bias, n, 320, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_1_0_emb_layers_1_weight, # MISSING input_blocks_1_0_emb_layers_1_bias, n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_1_0_out_layers_0_weight, # MISSING input_blocks_1_0_out_layers_0_bias, 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_1_0_out_layers_3_weight, # MISSING input_blocks_1_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 1)

    _s[1] = h_cur

    # input_blocks_2_0
    group_norm_torch(h_cur, # MISSING input_blocks_2_0_in_layers_0_weight, # MISSING input_blocks_2_0_in_layers_0_bias, 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_2_0_in_layers_2_weight, # MISSING input_blocks_2_0_in_layers_2_bias, n, 320, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_2_0_emb_layers_1_weight, # MISSING input_blocks_2_0_emb_layers_1_bias, n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_2_0_out_layers_0_weight, # MISSING input_blocks_2_0_out_layers_0_bias, 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_2_0_out_layers_3_weight, # MISSING input_blocks_2_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 1)

    _s[2] = h_cur
    # downsample 3
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_3_0_op_weight, # MISSING input_blocks_3_0_op_bias, n, 320, 320, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2

    # input_blocks_4_0
    group_norm_torch(h_cur, # MISSING input_blocks_4_0_in_layers_0_weight, # MISSING input_blocks_4_0_in_layers_0_bias, 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_4_0_in_layers_2_weight, # MISSING input_blocks_4_0_in_layers_2_bias, n, 320, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_4_0_emb_layers_1_weight, # MISSING input_blocks_4_0_emb_layers_1_bias, n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_4_0_out_layers_0_weight, # MISSING input_blocks_4_0_out_layers_0_bias, 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_4_0_out_layers_3_weight, # MISSING input_blocks_4_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 1)

    _s[4] = h_cur

    # input_blocks_5_0
    group_norm_torch(h_cur, # MISSING input_blocks_5_0_in_layers_0_weight, # MISSING input_blocks_5_0_in_layers_0_bias, 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_5_0_in_layers_2_weight, # MISSING input_blocks_5_0_in_layers_2_bias, n, 640, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_5_0_emb_layers_1_weight, # MISSING input_blocks_5_0_emb_layers_1_bias, n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_5_0_out_layers_0_weight, # MISSING input_blocks_5_0_out_layers_0_bias, 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_5_0_out_layers_3_weight, # MISSING input_blocks_5_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 1)

    _s[5] = h_cur
    # downsample 6
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_6_0_op_weight, # MISSING input_blocks_6_0_op_bias, n, 640, 640, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2

    # input_blocks_7_0
    group_norm_torch(h_cur, # MISSING input_blocks_7_0_in_layers_0_weight, # MISSING input_blocks_7_0_in_layers_0_bias, 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_7_0_in_layers_2_weight, # MISSING input_blocks_7_0_in_layers_2_bias, n, 640, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_7_0_emb_layers_1_weight, # MISSING input_blocks_7_0_emb_layers_1_bias, n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_7_0_out_layers_0_weight, # MISSING input_blocks_7_0_out_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_7_0_out_layers_3_weight, # MISSING input_blocks_7_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 1)

    _s[7] = h_cur

    # input_blocks_8_0
    group_norm_torch(h_cur, # MISSING input_blocks_8_0_in_layers_0_weight, # MISSING input_blocks_8_0_in_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_8_0_in_layers_2_weight, # MISSING input_blocks_8_0_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING input_blocks_8_0_emb_layers_1_weight, # MISSING input_blocks_8_0_emb_layers_1_bias, n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, # MISSING input_blocks_8_0_out_layers_0_weight, # MISSING input_blocks_8_0_out_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING input_blocks_8_0_out_layers_3_weight, # MISSING input_blocks_8_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 1)

    _s[8] = h_cur
    # middle_block

    # middle_block_0
    group_norm_torch(h_cur, # MISSING middle_block_0_in_layers_0_weight, # MISSING middle_block_0_in_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING middle_block_0_in_layers_2_weight, # MISSING middle_block_0_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING middle_block_0_emb_layers_1_weight, # MISSING middle_block_0_emb_layers_1_bias, n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, # MISSING middle_block_0_out_layers_0_weight, # MISSING middle_block_0_out_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING middle_block_0_out_layers_3_weight, # MISSING middle_block_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 1)


    # middle_block_2
    group_norm_torch(h_cur, # MISSING middle_block_2_in_layers_0_weight, # MISSING middle_block_2_in_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING middle_block_2_in_layers_2_weight, # MISSING middle_block_2_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, # MISSING middle_block_2_emb_layers_1_weight, # MISSING middle_block_2_emb_layers_1_bias, n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, # MISSING middle_block_2_out_layers_0_weight, # MISSING middle_block_2_out_layers_0_bias, 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, # MISSING middle_block_2_out_layers_3_weight, # MISSING middle_block_2_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 1)


    # output blocks (simplified - skip connections not yet wired)
    # TODO: wire up all 9 output blocks with skip connections
    
    h_cur = conv2d_torch(h_cur, out_0_weight_placeholder, out_0_bias_placeholder, n, 320, 4, hh, ww, 3, 1, 1)
    return h_cur

# Generated with 1680 weights
