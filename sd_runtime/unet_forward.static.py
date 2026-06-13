# unet_forward.static.py — 自动生成，所有偏移硬编码
def unet_forward(latent, timestep, context, data, n, hh, ww):
    h_cur: list[float]; _s: list[float] = make_float_array(30)
    ws = w_slice

    # time embedding
    emb = timestep_embedding_batch(timestep, 1280, n, 10000.0)
    emb = linear_torch(emb, ws(data,1282707202,320*1280), ws(data,1282706562,320), n, 1280, 1280)
    arr_silu(emb, emb, n*1280)
    emb = linear_torch(emb, ws(data,1282912642,1280*1280), ws(data,1282912002,1280), n, 1280, 1280)

    h_cur = conv2d_torch(latent, ws(data,160,11520), ws(data,0,320), n, 4, 320, hh, ww, 3, 1, 1)
    _s[0] = h_cur

    # input_blocks.1
    group_norm_torch(h_cur, ws(data,211040,320), ws(data,210880,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,211360,921600), ws(data,211200,320), n, 320, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,6080,409600), ws(data,5920,320), n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, ws(data,672320,320), ws(data,672160,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,672640,921600), ws(data,672480,320), n, 320, 320, hh, ww, 3, 1, 1)
    _s[1] = h_cur

    # input_blocks.2
    group_norm_torch(h_cur, ws(data,1338560,320), ws(data,1338400,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1338880,921600), ws(data,1338720,320), n, 320, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1133600,409600), ws(data,1133440,320), n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, ws(data,1799840,320), ws(data,1799680,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1800160,921600), ws(data,1800000,320), n, 320, 320, hh, ww, 3, 1, 1)
    _s[2] = h_cur

    h_cur = conv2d_torch(h_cur, ws(data,2261120,921600), ws(data,2260960,320), n, 320, 320, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2
    _s[3] = h_cur

    # input_blocks.4
    group_norm_torch(h_cur, ws(data,3132000,320), ws(data,3131840,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,3132480,1843200), ws(data,3132160,640), n, 320, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,2722240,819200), ws(data,2721920,640), n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, ws(data,4054400,640), ws(data,4054080,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,4055040,3686400), ws(data,4054720,640), n, 640, 640, hh, ww, 3, 1, 1)
    _s[4] = h_cur

    # input_blocks.5
    group_norm_torch(h_cur, ws(data,16827200,640), ws(data,16826880,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,16827840,3686400), ws(data,16827520,640), n, 640, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,16417280,819200), ws(data,16416960,640), n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, ws(data,18671360,640), ws(data,18671040,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,18672000,3686400), ws(data,18671680,640), n, 640, 640, hh, ww, 3, 1, 1)
    _s[5] = h_cur

    h_cur = conv2d_torch(h_cur, ws(data,30931520,3686400), ws(data,30931200,640), n, 640, 640, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2
    _s[6] = h_cur

    # input_blocks.7
    group_norm_torch(h_cur, ws(data,33594880,640), ws(data,33594560,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,33595840,7372800), ws(data,33595200,1280), n, 640, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,32775360,1638400), ws(data,32774720,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,37282880,1280), ws(data,37282240,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,37284160,14745600), ws(data,37283520,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    _s[7] = h_cur

    # input_blocks.8
    group_norm_torch(h_cur, ws(data,221307840,1280), ws(data,221307200,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,221309120,14745600), ws(data,221308480,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,220488000,1638400), ws(data,220487360,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,228682560,1280), ws(data,228681920,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,228683840,14745600), ws(data,228683200,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    _s[8] = h_cur

    # middle_block
    group_norm_torch(h_cur, ws(data,414920000,1280), ws(data,414919360,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,414921280,14745600), ws(data,414920640,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,414100160,1638400), ws(data,414099520,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,422294720,1280), ws(data,422294080,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,422296000,14745600), ws(data,422295360,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    group_norm_torch(h_cur, ws(data,605909440,1280), ws(data,605908800,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,605910720,14745600), ws(data,605910080,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,605089600,1638400), ws(data,605088960,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,613284160,1280), ws(data,613283520,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,613285440,14745600), ws(data,613284800,1280), n, 1280, 1280, hh, ww, 3, 1, 1)

    # output_blocks.0 (skip from 8:1280ch, cur:1280ch)
    _cat = make_float_array(n*2560*hh*ww)
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,n*1280*hh*ww+_i,float_array_ref(_s[8],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,621485442,2560), ws(data,621484162,2560), 32, 2560, hh*ww)
    silu_torch(h_cur, n*2560*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,621487362,29491200), ws(data,621486722,1280), n, 2560, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,620664962,1638400), ws(data,620664322,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,636233602,1280), ws(data,636232962,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,636234882,14745600), ws(data,636234242,1280), n, 1280, 1280, hh, ww, 3, 1, 1)

    # output_blocks.1 (skip from 7:1280ch, cur:1280ch)
    _cat = make_float_array(n*2560*hh*ww)
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,n*1280*hh*ww+_i,float_array_ref(_s[7],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,821488002,2560), ws(data,821486722,2560), 32, 2560, hh*ww)
    silu_torch(h_cur, n*2560*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,821489922,29491200), ws(data,821489282,1280), n, 2560, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,820667522,1638400), ws(data,820666882,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,836236162,1280), ws(data,836235522,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,836237442,14745600), ws(data,836236802,1280), n, 1280, 1280, hh, ww, 3, 1, 1)

    # output_blocks.2 (skip from 6:640ch, cur:1280ch)
    _cat = make_float_array(n*1920*hh*ww)
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,n*1280*hh*ww+_i,float_array_ref(_s[6],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1021490242,1920), ws(data,1021489282,1920), 32, 1920, hh*ww)
    silu_torch(h_cur, n*1920*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1021491842,22118400), ws(data,1021491202,1280), n, 1920, 1280, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1020670082,1638400), ws(data,1020669442,1280), n, 1280, 1280*2)
    apply_scale_shift(h_cur, _y, n, 1280, hh*ww)
    group_norm_torch(h_cur, ws(data,1032551682,1280), ws(data,1032551042,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1032552962,14745600), ws(data,1032552322,1280), n, 1280, 1280, hh, ww, 3, 1, 1)
    h_cur = upsample_nearest(h_cur,n,1280,hh,ww,2)
    hh = hh*2; ww = ww*2
    h_cur = conv2d_torch(h_cur, ws(data,1216576002,14745600), ws(data,1216575362,1280), n, 1280, 1280, hh, ww, 3, 1, 1)

    # output_blocks.3 (skip from 5:640ch, cur:1280ch)
    _cat = make_float_array(n*1920*hh*ww)
    _i=0
    while _i<n*1280*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,n*1280*hh*ww+_i,float_array_ref(_s[5],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1224359682,1920), ws(data,1224358722,1920), 32, 1920, hh*ww)
    silu_torch(h_cur, n*1920*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1224360962,11059200), ws(data,1224360642,640), n, 1920, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1223949122,819200), ws(data,1223948802,640), n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, ws(data,1229890882,640), ws(data,1229890562,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1229891522,3686400), ws(data,1229891202,640), n, 640, 640, hh, ww, 3, 1, 1)

    # output_blocks.4 (skip from 4:640ch, cur:640ch)
    _cat = make_float_array(n*1280*hh*ww)
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,n*640*hh*ww+_i,float_array_ref(_s[4],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1243176002,1280), ws(data,1243175362,1280), 32, 1280, hh*ww)
    silu_torch(h_cur, n*1280*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1243176962,7372800), ws(data,1243176642,640), n, 1280, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1242765762,819200), ws(data,1242765442,640), n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, ws(data,1246863682,640), ws(data,1246863362,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1246864322,3686400), ws(data,1246864002,640), n, 640, 640, hh, ww, 3, 1, 1)

    # output_blocks.5 (skip from 3:320ch, cur:640ch)
    _cat = make_float_array(n*960*hh*ww)
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,n*640*hh*ww+_i,float_array_ref(_s[3],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1259943842,960), ws(data,1259943362,960), 32, 960, hh*ww)
    silu_torch(h_cur, n*960*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1259944642,5529600), ws(data,1259944322,640), n, 960, 640, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1259533762,819200), ws(data,1259533442,640), n, 1280, 640*2)
    apply_scale_shift(h_cur, _y, n, 640, hh*ww)
    group_norm_torch(h_cur, ws(data,1262709762,640), ws(data,1262709442,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1262710402,3686400), ws(data,1262710082,640), n, 640, 640, hh, ww, 3, 1, 1)
    h_cur = upsample_nearest(h_cur,n,640,hh,ww,2)
    hh = hh*2; ww = ww*2
    h_cur = conv2d_torch(h_cur, ws(data,1275277442,3686400), ws(data,1275277122,640), n, 640, 640, hh, ww, 3, 1, 1)

    # output_blocks.6 (skip from 2:320ch, cur:640ch)
    _cat = make_float_array(n*960*hh*ww)
    _i=0
    while _i<n*640*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,n*640*hh*ww+_i,float_array_ref(_s[2],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1277326082,960), ws(data,1277325602,960), 32, 960, hh*ww)
    silu_torch(h_cur, n*960*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1277326722,2764800), ws(data,1277326562,320), n, 960, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1277120802,409600), ws(data,1277120642,320), n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, ws(data,1278709282,320), ws(data,1278709122,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1278709602,921600), ws(data,1278709442,320), n, 320, 320, hh, ww, 3, 1, 1)

    # output_blocks.7 (skip from 1:320ch, cur:320ch)
    _cat = make_float_array(n*640*hh*ww)
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,n*320*hh*ww+_i,float_array_ref(_s[1],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1279529442,640), ws(data,1279529122,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1279529922,1843200), ws(data,1279529762,320), n, 640, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1279324322,409600), ws(data,1279324162,320), n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, ws(data,1280451682,320), ws(data,1280451522,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1280452002,921600), ws(data,1280451842,320), n, 320, 320, hh, ww, 3, 1, 1)

    # output_blocks.8 (skip from 0:320ch, cur:320ch)
    _cat = make_float_array(n*640*hh*ww)
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,_i,float_array_ref(h_cur,_i))
        _i=_i+1
    _i=0
    while _i<n*320*hh*ww:
        float_array_set(_cat,n*320*hh*ww+_i,float_array_ref(_s[0],_i))
        _i=_i+1
    h_cur = _cat
    group_norm_torch(h_cur, ws(data,1281220642,640), ws(data,1281220322,640), 32, 640, hh*ww)
    silu_torch(h_cur, n*640*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1281221122,1843200), ws(data,1281220962,320), n, 640, 320, hh, ww, 3, 1, 1)
    _y = linear_torch(emb, ws(data,1281015522,409600), ws(data,1281015362,320), n, 1280, 320*2)
    apply_scale_shift(h_cur, _y, n, 320, hh*ww)
    group_norm_torch(h_cur, ws(data,1282142882,320), ws(data,1282142722,320), 32, 320, hh*ww)
    silu_torch(h_cur, n*320*hh*ww)
    h_cur = conv2d_torch(h_cur, ws(data,1282143202,921600), ws(data,1282143042,320), n, 320, 320, hh, ww, 3, 1, 1)

    h_cur = conv2d_torch(h_cur, ws(data,620658400,320), ws(data,620658240,320), n, 320, 4, hh, ww, 3, 1, 1)
    return h_cur
