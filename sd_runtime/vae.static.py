# vae.static.py — VAE Decoder (separate function, inline conv2d)

def vae_decoder_forward(latent, weights_dir, n, height, width):
    h_cur: list[float]
    hh: int = height
    ww: int = width

    conv_in_bias = load_bin(weights_dir + "/decoder_conv_in_bias.bin", 512)
    conv_in_weight = load_bin(weights_dir + "/decoder_conv_in_weight.bin", 73728)
    conv_out_bias = load_bin(weights_dir + "/decoder_conv_out_bias.bin", 3)
    conv_out_weight = load_bin(weights_dir + "/decoder_conv_out_weight.bin", 3456)
    mid_attn_1_k_bias = load_bin(weights_dir + "/decoder_mid_attn_1_k_bias.bin", 512)
    mid_attn_1_k_weight = load_bin(weights_dir + "/decoder_mid_attn_1_k_weight.bin", 262144)
    mid_attn_1_norm_bias = load_bin(weights_dir + "/decoder_mid_attn_1_norm_bias.bin", 512)
    mid_attn_1_norm_weight = load_bin(weights_dir + "/decoder_mid_attn_1_norm_weight.bin", 512)
    mid_attn_1_proj_out_bias = load_bin(weights_dir + "/decoder_mid_attn_1_proj_out_bias.bin", 512)
    mid_attn_1_proj_out_weight = load_bin(weights_dir + "/decoder_mid_attn_1_proj_out_weight.bin", 262144)
    mid_attn_1_q_bias = load_bin(weights_dir + "/decoder_mid_attn_1_q_bias.bin", 512)
    mid_attn_1_q_weight = load_bin(weights_dir + "/decoder_mid_attn_1_q_weight.bin", 262144)
    mid_attn_1_v_bias = load_bin(weights_dir + "/decoder_mid_attn_1_v_bias.bin", 512)
    mid_attn_1_v_weight = load_bin(weights_dir + "/decoder_mid_attn_1_v_weight.bin", 262144)
    mid_block_1_conv1_bias = load_bin(weights_dir + "/decoder_mid_block_1_conv1_bias.bin", 512)
    mid_block_1_conv1_weight = load_bin(weights_dir + "/decoder_mid_block_1_conv1_weight.bin", 2359296)
    mid_block_1_conv2_bias = load_bin(weights_dir + "/decoder_mid_block_1_conv2_bias.bin", 512)
    mid_block_1_conv2_weight = load_bin(weights_dir + "/decoder_mid_block_1_conv2_weight.bin", 2359296)
    mid_block_1_norm1_bias = load_bin(weights_dir + "/decoder_mid_block_1_norm1_bias.bin", 512)
    mid_block_1_norm1_weight = load_bin(weights_dir + "/decoder_mid_block_1_norm1_weight.bin", 512)
    mid_block_1_norm2_bias = load_bin(weights_dir + "/decoder_mid_block_1_norm2_bias.bin", 512)
    mid_block_1_norm2_weight = load_bin(weights_dir + "/decoder_mid_block_1_norm2_weight.bin", 512)
    mid_block_2_conv1_bias = load_bin(weights_dir + "/decoder_mid_block_2_conv1_bias.bin", 512)
    mid_block_2_conv1_weight = load_bin(weights_dir + "/decoder_mid_block_2_conv1_weight.bin", 2359296)
    mid_block_2_conv2_bias = load_bin(weights_dir + "/decoder_mid_block_2_conv2_bias.bin", 512)
    mid_block_2_conv2_weight = load_bin(weights_dir + "/decoder_mid_block_2_conv2_weight.bin", 2359296)
    mid_block_2_norm1_bias = load_bin(weights_dir + "/decoder_mid_block_2_norm1_bias.bin", 512)
    mid_block_2_norm1_weight = load_bin(weights_dir + "/decoder_mid_block_2_norm1_weight.bin", 512)
    mid_block_2_norm2_bias = load_bin(weights_dir + "/decoder_mid_block_2_norm2_bias.bin", 512)
    mid_block_2_norm2_weight = load_bin(weights_dir + "/decoder_mid_block_2_norm2_weight.bin", 512)
    norm_out_bias = load_bin(weights_dir + "/decoder_norm_out_bias.bin", 128)
    norm_out_weight = load_bin(weights_dir + "/decoder_norm_out_weight.bin", 128)
    up_0_block_0_conv1_bias = load_bin(weights_dir + "/decoder_up_0_block_0_conv1_bias.bin", 128)
    up_0_block_0_conv1_weight = load_bin(weights_dir + "/decoder_up_0_block_0_conv1_weight.bin", 294912)
    up_0_block_0_conv2_bias = load_bin(weights_dir + "/decoder_up_0_block_0_conv2_bias.bin", 128)
    up_0_block_0_conv2_weight = load_bin(weights_dir + "/decoder_up_0_block_0_conv2_weight.bin", 147456)
    up_0_block_0_nin_shortcut_bias = load_bin(weights_dir + "/decoder_up_0_block_0_nin_shortcut_bias.bin", 128)
    up_0_block_0_nin_shortcut_weight = load_bin(weights_dir + "/decoder_up_0_block_0_nin_shortcut_weight.bin", 32768)
    up_0_block_0_norm1_bias = load_bin(weights_dir + "/decoder_up_0_block_0_norm1_bias.bin", 256)
    up_0_block_0_norm1_weight = load_bin(weights_dir + "/decoder_up_0_block_0_norm1_weight.bin", 256)
    up_0_block_0_norm2_bias = load_bin(weights_dir + "/decoder_up_0_block_0_norm2_bias.bin", 128)
    up_0_block_0_norm2_weight = load_bin(weights_dir + "/decoder_up_0_block_0_norm2_weight.bin", 128)
    up_0_block_1_conv1_bias = load_bin(weights_dir + "/decoder_up_0_block_1_conv1_bias.bin", 128)
    up_0_block_1_conv1_weight = load_bin(weights_dir + "/decoder_up_0_block_1_conv1_weight.bin", 147456)
    up_0_block_1_conv2_bias = load_bin(weights_dir + "/decoder_up_0_block_1_conv2_bias.bin", 128)
    up_0_block_1_conv2_weight = load_bin(weights_dir + "/decoder_up_0_block_1_conv2_weight.bin", 147456)
    up_0_block_1_norm1_bias = load_bin(weights_dir + "/decoder_up_0_block_1_norm1_bias.bin", 128)
    up_0_block_1_norm1_weight = load_bin(weights_dir + "/decoder_up_0_block_1_norm1_weight.bin", 128)
    up_0_block_1_norm2_bias = load_bin(weights_dir + "/decoder_up_0_block_1_norm2_bias.bin", 128)
    up_0_block_1_norm2_weight = load_bin(weights_dir + "/decoder_up_0_block_1_norm2_weight.bin", 128)
    up_0_block_2_conv1_bias = load_bin(weights_dir + "/decoder_up_0_block_2_conv1_bias.bin", 128)
    up_0_block_2_conv1_weight = load_bin(weights_dir + "/decoder_up_0_block_2_conv1_weight.bin", 147456)
    up_0_block_2_conv2_bias = load_bin(weights_dir + "/decoder_up_0_block_2_conv2_bias.bin", 128)
    up_0_block_2_conv2_weight = load_bin(weights_dir + "/decoder_up_0_block_2_conv2_weight.bin", 147456)
    up_0_block_2_norm1_bias = load_bin(weights_dir + "/decoder_up_0_block_2_norm1_bias.bin", 128)
    up_0_block_2_norm1_weight = load_bin(weights_dir + "/decoder_up_0_block_2_norm1_weight.bin", 128)
    up_0_block_2_norm2_bias = load_bin(weights_dir + "/decoder_up_0_block_2_norm2_bias.bin", 128)
    up_0_block_2_norm2_weight = load_bin(weights_dir + "/decoder_up_0_block_2_norm2_weight.bin", 128)
    up_1_block_0_conv1_bias = load_bin(weights_dir + "/decoder_up_1_block_0_conv1_bias.bin", 256)
    up_1_block_0_conv1_weight = load_bin(weights_dir + "/decoder_up_1_block_0_conv1_weight.bin", 1179648)
    up_1_block_0_conv2_bias = load_bin(weights_dir + "/decoder_up_1_block_0_conv2_bias.bin", 256)
    up_1_block_0_conv2_weight = load_bin(weights_dir + "/decoder_up_1_block_0_conv2_weight.bin", 589824)
    up_1_block_0_nin_shortcut_bias = load_bin(weights_dir + "/decoder_up_1_block_0_nin_shortcut_bias.bin", 256)
    up_1_block_0_nin_shortcut_weight = load_bin(weights_dir + "/decoder_up_1_block_0_nin_shortcut_weight.bin", 131072)
    up_1_block_0_norm1_bias = load_bin(weights_dir + "/decoder_up_1_block_0_norm1_bias.bin", 512)
    up_1_block_0_norm1_weight = load_bin(weights_dir + "/decoder_up_1_block_0_norm1_weight.bin", 512)
    up_1_block_0_norm2_bias = load_bin(weights_dir + "/decoder_up_1_block_0_norm2_bias.bin", 256)
    up_1_block_0_norm2_weight = load_bin(weights_dir + "/decoder_up_1_block_0_norm2_weight.bin", 256)
    up_1_block_1_conv1_bias = load_bin(weights_dir + "/decoder_up_1_block_1_conv1_bias.bin", 256)
    up_1_block_1_conv1_weight = load_bin(weights_dir + "/decoder_up_1_block_1_conv1_weight.bin", 589824)
    up_1_block_1_conv2_bias = load_bin(weights_dir + "/decoder_up_1_block_1_conv2_bias.bin", 256)
    up_1_block_1_conv2_weight = load_bin(weights_dir + "/decoder_up_1_block_1_conv2_weight.bin", 589824)
    up_1_block_1_norm1_bias = load_bin(weights_dir + "/decoder_up_1_block_1_norm1_bias.bin", 256)
    up_1_block_1_norm1_weight = load_bin(weights_dir + "/decoder_up_1_block_1_norm1_weight.bin", 256)
    up_1_block_1_norm2_bias = load_bin(weights_dir + "/decoder_up_1_block_1_norm2_bias.bin", 256)
    up_1_block_1_norm2_weight = load_bin(weights_dir + "/decoder_up_1_block_1_norm2_weight.bin", 256)
    up_1_block_2_conv1_bias = load_bin(weights_dir + "/decoder_up_1_block_2_conv1_bias.bin", 256)
    up_1_block_2_conv1_weight = load_bin(weights_dir + "/decoder_up_1_block_2_conv1_weight.bin", 589824)
    up_1_block_2_conv2_bias = load_bin(weights_dir + "/decoder_up_1_block_2_conv2_bias.bin", 256)
    up_1_block_2_conv2_weight = load_bin(weights_dir + "/decoder_up_1_block_2_conv2_weight.bin", 589824)
    up_1_block_2_norm1_bias = load_bin(weights_dir + "/decoder_up_1_block_2_norm1_bias.bin", 256)
    up_1_block_2_norm1_weight = load_bin(weights_dir + "/decoder_up_1_block_2_norm1_weight.bin", 256)
    up_1_block_2_norm2_bias = load_bin(weights_dir + "/decoder_up_1_block_2_norm2_bias.bin", 256)
    up_1_block_2_norm2_weight = load_bin(weights_dir + "/decoder_up_1_block_2_norm2_weight.bin", 256)
    up_1_upsample_conv_bias = load_bin(weights_dir + "/decoder_up_1_upsample_conv_bias.bin", 256)
    up_1_upsample_conv_weight = load_bin(weights_dir + "/decoder_up_1_upsample_conv_weight.bin", 589824)
    up_2_block_0_conv1_bias = load_bin(weights_dir + "/decoder_up_2_block_0_conv1_bias.bin", 512)
    up_2_block_0_conv1_weight = load_bin(weights_dir + "/decoder_up_2_block_0_conv1_weight.bin", 2359296)
    up_2_block_0_conv2_bias = load_bin(weights_dir + "/decoder_up_2_block_0_conv2_bias.bin", 512)
    up_2_block_0_conv2_weight = load_bin(weights_dir + "/decoder_up_2_block_0_conv2_weight.bin", 2359296)
    up_2_block_0_norm1_bias = load_bin(weights_dir + "/decoder_up_2_block_0_norm1_bias.bin", 512)
    up_2_block_0_norm1_weight = load_bin(weights_dir + "/decoder_up_2_block_0_norm1_weight.bin", 512)
    up_2_block_0_norm2_bias = load_bin(weights_dir + "/decoder_up_2_block_0_norm2_bias.bin", 512)
    up_2_block_0_norm2_weight = load_bin(weights_dir + "/decoder_up_2_block_0_norm2_weight.bin", 512)
    up_2_block_1_conv1_bias = load_bin(weights_dir + "/decoder_up_2_block_1_conv1_bias.bin", 512)
    up_2_block_1_conv1_weight = load_bin(weights_dir + "/decoder_up_2_block_1_conv1_weight.bin", 2359296)
    up_2_block_1_conv2_bias = load_bin(weights_dir + "/decoder_up_2_block_1_conv2_bias.bin", 512)
    up_2_block_1_conv2_weight = load_bin(weights_dir + "/decoder_up_2_block_1_conv2_weight.bin", 2359296)
    up_2_block_1_norm1_bias = load_bin(weights_dir + "/decoder_up_2_block_1_norm1_bias.bin", 512)
    up_2_block_1_norm1_weight = load_bin(weights_dir + "/decoder_up_2_block_1_norm1_weight.bin", 512)
    up_2_block_1_norm2_bias = load_bin(weights_dir + "/decoder_up_2_block_1_norm2_bias.bin", 512)
    up_2_block_1_norm2_weight = load_bin(weights_dir + "/decoder_up_2_block_1_norm2_weight.bin", 512)
    up_2_block_2_conv1_bias = load_bin(weights_dir + "/decoder_up_2_block_2_conv1_bias.bin", 512)
    up_2_block_2_conv1_weight = load_bin(weights_dir + "/decoder_up_2_block_2_conv1_weight.bin", 2359296)
    up_2_block_2_conv2_bias = load_bin(weights_dir + "/decoder_up_2_block_2_conv2_bias.bin", 512)
    up_2_block_2_conv2_weight = load_bin(weights_dir + "/decoder_up_2_block_2_conv2_weight.bin", 2359296)
    up_2_block_2_norm1_bias = load_bin(weights_dir + "/decoder_up_2_block_2_norm1_bias.bin", 512)
    up_2_block_2_norm1_weight = load_bin(weights_dir + "/decoder_up_2_block_2_norm1_weight.bin", 512)
    up_2_block_2_norm2_bias = load_bin(weights_dir + "/decoder_up_2_block_2_norm2_bias.bin", 512)
    up_2_block_2_norm2_weight = load_bin(weights_dir + "/decoder_up_2_block_2_norm2_weight.bin", 512)
    up_2_upsample_conv_bias = load_bin(weights_dir + "/decoder_up_2_upsample_conv_bias.bin", 512)
    up_2_upsample_conv_weight = load_bin(weights_dir + "/decoder_up_2_upsample_conv_weight.bin", 2359296)
    up_3_block_0_conv1_bias = load_bin(weights_dir + "/decoder_up_3_block_0_conv1_bias.bin", 512)
    up_3_block_0_conv1_weight = load_bin(weights_dir + "/decoder_up_3_block_0_conv1_weight.bin", 2359296)
    up_3_block_0_conv2_bias = load_bin(weights_dir + "/decoder_up_3_block_0_conv2_bias.bin", 512)
    up_3_block_0_conv2_weight = load_bin(weights_dir + "/decoder_up_3_block_0_conv2_weight.bin", 2359296)
    up_3_block_0_norm1_bias = load_bin(weights_dir + "/decoder_up_3_block_0_norm1_bias.bin", 512)
    up_3_block_0_norm1_weight = load_bin(weights_dir + "/decoder_up_3_block_0_norm1_weight.bin", 512)
    up_3_block_0_norm2_bias = load_bin(weights_dir + "/decoder_up_3_block_0_norm2_bias.bin", 512)
    up_3_block_0_norm2_weight = load_bin(weights_dir + "/decoder_up_3_block_0_norm2_weight.bin", 512)
    up_3_block_1_conv1_bias = load_bin(weights_dir + "/decoder_up_3_block_1_conv1_bias.bin", 512)
    up_3_block_1_conv1_weight = load_bin(weights_dir + "/decoder_up_3_block_1_conv1_weight.bin", 2359296)
    up_3_block_1_conv2_bias = load_bin(weights_dir + "/decoder_up_3_block_1_conv2_bias.bin", 512)
    up_3_block_1_conv2_weight = load_bin(weights_dir + "/decoder_up_3_block_1_conv2_weight.bin", 2359296)
    up_3_block_1_norm1_bias = load_bin(weights_dir + "/decoder_up_3_block_1_norm1_bias.bin", 512)
    up_3_block_1_norm1_weight = load_bin(weights_dir + "/decoder_up_3_block_1_norm1_weight.bin", 512)
    up_3_block_1_norm2_bias = load_bin(weights_dir + "/decoder_up_3_block_1_norm2_bias.bin", 512)
    up_3_block_1_norm2_weight = load_bin(weights_dir + "/decoder_up_3_block_1_norm2_weight.bin", 512)
    up_3_block_2_conv1_bias = load_bin(weights_dir + "/decoder_up_3_block_2_conv1_bias.bin", 512)
    up_3_block_2_conv1_weight = load_bin(weights_dir + "/decoder_up_3_block_2_conv1_weight.bin", 2359296)
    up_3_block_2_conv2_bias = load_bin(weights_dir + "/decoder_up_3_block_2_conv2_bias.bin", 512)
    up_3_block_2_conv2_weight = load_bin(weights_dir + "/decoder_up_3_block_2_conv2_weight.bin", 2359296)
    up_3_block_2_norm1_bias = load_bin(weights_dir + "/decoder_up_3_block_2_norm1_bias.bin", 512)
    up_3_block_2_norm1_weight = load_bin(weights_dir + "/decoder_up_3_block_2_norm1_weight.bin", 512)
    up_3_block_2_norm2_bias = load_bin(weights_dir + "/decoder_up_3_block_2_norm2_bias.bin", 512)
    up_3_block_2_norm2_weight = load_bin(weights_dir + "/decoder_up_3_block_2_norm2_weight.bin", 512)
    up_3_upsample_conv_bias = load_bin(weights_dir + "/decoder_up_3_upsample_conv_bias.bin", 512)
    up_3_upsample_conv_weight = load_bin(weights_dir + "/decoder_up_3_upsample_conv_weight.bin", 2359296)

    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 16 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(latent, 1, 16, hh, ww, 3, 1, 1, _col)
    print("alloc y2..."); _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, conv_in_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(conv_in_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 1")
    group_norm(h_cur, mid_block_1_norm1_weight, mid_block_1_norm1_bias, 32, 512, hh*ww)
    print("step 2")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); print("mid_conv_start..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, mid_block_1_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(mid_block_1_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 3")
    group_norm(h_cur, mid_block_1_norm2_weight, mid_block_1_norm2_bias, 32, 512, hh*ww)
    print("step 4")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, mid_block_1_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(mid_block_1_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 5")
    group_norm(h_cur, mid_block_2_norm1_weight, mid_block_2_norm1_bias, 32, 512, hh*ww)
    print("step 6")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, mid_block_2_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(mid_block_2_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 7")
    group_norm(h_cur, mid_block_2_norm2_weight, mid_block_2_norm2_bias, 32, 512, hh*ww)
    print("step 8")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, mid_block_2_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(mid_block_2_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 9")
    group_norm(h_cur, up_0_block_0_norm1_weight, up_0_block_0_norm1_bias, 32, 256, hh*ww)
    print("step 10")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_0_block_0_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_0_block_0_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 11")
    group_norm(h_cur, up_0_block_0_norm2_weight, up_0_block_0_norm2_bias, 32, 256, hh*ww)
    print("step 12")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_0_block_0_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_0_block_0_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 13")
    group_norm(h_cur, up_0_block_1_norm1_weight, up_0_block_1_norm1_bias, 32, 128, hh*ww)
    print("step 14")
    arr_silu(h_cur, h_cur, 1*128*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 128 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 128, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 128)
    dgemm_row_auto(_nc, 128, _kd, 1.0, _col, up_0_block_1_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 128:
            float_array_set(_y, __i*128+__j, float_array_ref(_y, __i*128+__j) + float_array_ref(up_0_block_1_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 15")
    group_norm(h_cur, up_0_block_1_norm2_weight, up_0_block_1_norm2_bias, 32, 128, hh*ww)
    print("step 16")
    arr_silu(h_cur, h_cur, 1*128*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 128 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 128, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 128)
    dgemm_row_auto(_nc, 128, _kd, 1.0, _col, up_0_block_1_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 128:
            float_array_set(_y, __i*128+__j, float_array_ref(_y, __i*128+__j) + float_array_ref(up_0_block_1_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 17")
    group_norm(h_cur, up_0_block_2_norm1_weight, up_0_block_2_norm1_bias, 32, 128, hh*ww)
    print("step 18")
    arr_silu(h_cur, h_cur, 1*128*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 128 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 128, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 128)
    dgemm_row_auto(_nc, 128, _kd, 1.0, _col, up_0_block_2_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 128:
            float_array_set(_y, __i*128+__j, float_array_ref(_y, __i*128+__j) + float_array_ref(up_0_block_2_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 19")
    group_norm(h_cur, up_0_block_2_norm2_weight, up_0_block_2_norm2_bias, 32, 128, hh*ww)
    print("step 20")
    arr_silu(h_cur, h_cur, 1*128*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 128 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 128, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 128)
    dgemm_row_auto(_nc, 128, _kd, 1.0, _col, up_0_block_2_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 128:
            float_array_set(_y, __i*128+__j, float_array_ref(_y, __i*128+__j) + float_array_ref(up_0_block_2_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 21")
    h_cur = upsample_nearest(h_cur, 1, 256, hh, ww, 2)
    print("step 22")
    hh = hh*2; ww = ww*2
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_1_upsample_conv_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_1_upsample_conv_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 23")
    group_norm(h_cur, up_1_block_0_norm1_weight, up_1_block_0_norm1_bias, 32, 512, hh*ww)
    print("step 24")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_1_block_0_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_1_block_0_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 25")
    group_norm(h_cur, up_1_block_0_norm2_weight, up_1_block_0_norm2_bias, 32, 512, hh*ww)
    print("step 26")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_1_block_0_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_1_block_0_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 27")
    group_norm(h_cur, up_1_block_1_norm1_weight, up_1_block_1_norm1_bias, 32, 256, hh*ww)
    print("step 28")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_1_block_1_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_1_block_1_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 29")
    group_norm(h_cur, up_1_block_1_norm2_weight, up_1_block_1_norm2_bias, 32, 256, hh*ww)
    print("step 30")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_1_block_1_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_1_block_1_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 31")
    group_norm(h_cur, up_1_block_2_norm1_weight, up_1_block_2_norm1_bias, 32, 256, hh*ww)
    print("step 32")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_1_block_2_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_1_block_2_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 33")
    group_norm(h_cur, up_1_block_2_norm2_weight, up_1_block_2_norm2_bias, 32, 256, hh*ww)
    print("step 34")
    arr_silu(h_cur, h_cur, 1*256*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 256 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 256, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 256)
    dgemm_row_auto(_nc, 256, _kd, 1.0, _col, up_1_block_2_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 256:
            float_array_set(_y, __i*256+__j, float_array_ref(_y, __i*256+__j) + float_array_ref(up_1_block_2_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 35")
    h_cur = upsample_nearest(h_cur, 1, 512, hh, ww, 2)
    print("step 36")
    hh = hh*2; ww = ww*2
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_upsample_conv_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_upsample_conv_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 37")
    group_norm(h_cur, up_2_block_0_norm1_weight, up_2_block_0_norm1_bias, 32, 512, hh*ww)
    print("step 38")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_0_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_0_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 39")
    group_norm(h_cur, up_2_block_0_norm2_weight, up_2_block_0_norm2_bias, 32, 512, hh*ww)
    print("step 40")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_0_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_0_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 41")
    group_norm(h_cur, up_2_block_1_norm1_weight, up_2_block_1_norm1_bias, 32, 512, hh*ww)
    print("step 42")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_1_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_1_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 43")
    group_norm(h_cur, up_2_block_1_norm2_weight, up_2_block_1_norm2_bias, 32, 512, hh*ww)
    print("step 44")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_1_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_1_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 45")
    group_norm(h_cur, up_2_block_2_norm1_weight, up_2_block_2_norm1_bias, 32, 512, hh*ww)
    print("step 46")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_2_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_2_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 47")
    group_norm(h_cur, up_2_block_2_norm2_weight, up_2_block_2_norm2_bias, 32, 512, hh*ww)
    print("step 48")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_2_block_2_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_2_block_2_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 49")
    h_cur = upsample_nearest(h_cur, 1, 512, hh, ww, 2)
    print("step 50")
    hh = hh*2; ww = ww*2
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_upsample_conv_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_upsample_conv_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 51")
    group_norm(h_cur, up_3_block_0_norm1_weight, up_3_block_0_norm1_bias, 32, 512, hh*ww)
    print("step 52")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_0_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_0_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 53")
    group_norm(h_cur, up_3_block_0_norm2_weight, up_3_block_0_norm2_bias, 32, 512, hh*ww)
    print("step 54")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_0_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_0_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 55")
    group_norm(h_cur, up_3_block_1_norm1_weight, up_3_block_1_norm1_bias, 32, 512, hh*ww)
    print("step 56")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_1_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_1_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 57")
    group_norm(h_cur, up_3_block_1_norm2_weight, up_3_block_1_norm2_bias, 32, 512, hh*ww)
    print("step 58")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_1_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_1_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 59")
    group_norm(h_cur, up_3_block_2_norm1_weight, up_3_block_2_norm1_bias, 32, 512, hh*ww)
    print("step 60")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_2_conv1_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_2_conv1_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 61")
    group_norm(h_cur, up_3_block_2_norm2_weight, up_3_block_2_norm2_bias, 32, 512, hh*ww)
    print("step 62")
    arr_silu(h_cur, h_cur, 1*512*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 512 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    print("im2col2..."); im2col(h_cur, 1, 512, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 512)
    dgemm_row_auto(_nc, 512, _kd, 1.0, _col, up_3_block_2_conv2_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 512:
            float_array_set(_y, __i*512+__j, float_array_ref(_y, __i*512+__j) + float_array_ref(up_3_block_2_conv2_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 63")
    group_norm(h_cur, norm_out_weight, norm_out_bias, 32, 128, hh*ww)
    print("step 64")
    arr_silu(h_cur, h_cur, 1*128*hh*ww)
    _ho = (hh + 2*1 - 3)//1 + 1
    _wo = (ww + 2*1 - 3)//1 + 1
    _nc = 1 * _ho * _wo
    _kd = 128 * 3 * 3
    print("alloc col2..."); _col = make_float_array(_nc * _kd)
    im2col(h_cur, 1, 128, hh, ww, 3, 1, 1, _col)
    _y = make_float_array(_nc * 3)
    dgemm_row_auto(_nc, 3, _kd, 1.0, _col, conv_out_weight, 0.0, _y)
    __i = 0
    while __i < _nc:
        __j = 0
        while __j < 3:
            float_array_set(_y, __i*3+__j, float_array_ref(_y, __i*3+__j) + float_array_ref(conv_out_bias, __j))
            __j = __j + 1
        __i = __i + 1
    h_cur = _y
    print("step 65")
    arr_clip(h_cur, 0.0, 1.0, 1*3*hh*ww)
    return h_cur
    # mid blocks only - returning 512ch activation
    return h_cur
