# unet_forward.static.py — 自动生成，所有偏移硬编码
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
    emb: ptr = st_from_blob_1d(_emb, n*320)
    _w_input_blocks_0_0_bias: ptr = w_slice_1d(data, 0, 320)
    _w_input_blocks_0_0_weight: ptr = w_slice_4d(data, 320, 320, 4, 3, 3)
    _w_input_blocks_1_0_emb_layers_1_bias: ptr = w_slice_1d(data, 11840, 320)
    _w_input_blocks_1_0_emb_layers_1_weight: ptr = w_slice_2d(data, 12160, 320, 1280)
    _w_input_blocks_1_0_in_layers_0_bias: ptr = w_slice_1d(data, 421760, 320)
    _w_input_blocks_1_0_in_layers_0_weight: ptr = w_slice_1d(data, 422080, 320)
    _w_input_blocks_1_0_in_layers_2_bias: ptr = w_slice_1d(data, 422400, 320)
    _w_input_blocks_1_0_in_layers_2_weight: ptr = w_slice_4d(data, 422720, 320, 320, 3, 3)
    _w_input_blocks_1_0_out_layers_0_bias: ptr = w_slice_1d(data, 1344320, 320)
    _w_input_blocks_1_0_out_layers_0_weight: ptr = w_slice_1d(data, 1344640, 320)
    _w_input_blocks_1_0_out_layers_3_bias: ptr = w_slice_1d(data, 1344960, 320)
    _w_input_blocks_1_0_out_layers_3_weight: ptr = w_slice_4d(data, 1345280, 320, 320, 3, 3)
    _w_input_blocks_2_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2266880, 320)
    _w_input_blocks_2_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2267200, 320, 1280)
    _w_input_blocks_2_0_in_layers_0_bias: ptr = w_slice_1d(data, 2676800, 320)
    _w_input_blocks_2_0_in_layers_0_weight: ptr = w_slice_1d(data, 2677120, 320)
    _w_input_blocks_2_0_in_layers_2_bias: ptr = w_slice_1d(data, 2677440, 320)
    _w_input_blocks_2_0_in_layers_2_weight: ptr = w_slice_4d(data, 2677760, 320, 320, 3, 3)
    _w_input_blocks_2_0_out_layers_0_bias: ptr = w_slice_1d(data, 3599360, 320)
    _w_input_blocks_2_0_out_layers_0_weight: ptr = w_slice_1d(data, 3599680, 320)
    _w_input_blocks_2_0_out_layers_3_bias: ptr = w_slice_1d(data, 3600000, 320)
    _w_input_blocks_2_0_out_layers_3_weight: ptr = w_slice_4d(data, 3600320, 320, 320, 3, 3)
    _w_input_blocks_3_0_op_bias: ptr = w_slice_1d(data, 4521920, 320)
    _w_input_blocks_3_0_op_weight: ptr = w_slice_4d(data, 4522240, 320, 320, 3, 3)
    _w_input_blocks_4_0_emb_layers_1_bias: ptr = w_slice_1d(data, 5443840, 640)
    _w_input_blocks_4_0_emb_layers_1_weight: ptr = w_slice_2d(data, 5444480, 640, 1280)
    _w_input_blocks_4_0_in_layers_0_bias: ptr = w_slice_1d(data, 6263680, 320)
    _w_input_blocks_4_0_in_layers_0_weight: ptr = w_slice_1d(data, 6264000, 320)
    _w_input_blocks_4_0_in_layers_2_bias: ptr = w_slice_1d(data, 6264320, 640)
    _w_input_blocks_4_0_in_layers_2_weight: ptr = w_slice_4d(data, 6264960, 640, 320, 3, 3)
    _w_input_blocks_4_0_out_layers_0_bias: ptr = w_slice_1d(data, 8108160, 640)
    _w_input_blocks_4_0_out_layers_0_weight: ptr = w_slice_1d(data, 8108800, 640)
    _w_input_blocks_4_0_out_layers_3_bias: ptr = w_slice_1d(data, 8109440, 640)
    _w_input_blocks_4_0_out_layers_3_weight: ptr = w_slice_4d(data, 8110080, 640, 640, 3, 3)
    _w_input_blocks_4_0_skip_connection_bias: ptr = w_slice_1d(data, 11796480, 640)
    _w_input_blocks_4_0_skip_connection_weight: ptr = w_slice_4d(data, 11797120, 640, 320, 1, 1)
    _w_input_blocks_4_1_norm_bias: ptr = w_slice_1d(data, 12001920, 640)
    _w_input_blocks_4_1_norm_weight: ptr = w_slice_1d(data, 12002560, 640)
    _w_input_blocks_4_1_proj_in_bias: ptr = w_slice_1d(data, 12003200, 640)
    _w_input_blocks_4_1_proj_in_weight: ptr = w_slice_2d(data, 12003840, 640, 640)
    _w_input_blocks_4_1_proj_out_bias: ptr = w_slice_1d(data, 12413440, 640)
    _w_input_blocks_4_1_proj_out_weight: ptr = w_slice_2d(data, 12414080, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 12823680, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 13233280, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 13233920, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 13643520, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 14053120, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 14462720, 640, 2048)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 15773440, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 15774080, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 16183680, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 16593280, 640, 2048)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 17904000, 5120)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 17909120, 5120, 640)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 21185920, 640)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 21186560, 640, 2560)
    _w_input_blocks_4_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 22824960, 640)
    _w_input_blocks_4_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 22825600, 640)
    _w_input_blocks_4_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 22826240, 640)
    _w_input_blocks_4_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 22826880, 640)
    _w_input_blocks_4_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 22827520, 640)
    _w_input_blocks_4_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 22828160, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 22828800, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 23238400, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 23239040, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 23648640, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 24058240, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 24467840, 640, 2048)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 25778560, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 25779200, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 26188800, 640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 26598400, 640, 2048)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 27909120, 5120)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 27914240, 5120, 640)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 31191040, 640)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 31191680, 640, 2560)
    _w_input_blocks_4_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 32830080, 640)
    _w_input_blocks_4_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 32830720, 640)
    _w_input_blocks_4_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 32831360, 640)
    _w_input_blocks_4_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 32832000, 640)
    _w_input_blocks_4_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 32832640, 640)
    _w_input_blocks_4_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 32833280, 640)
    _w_input_blocks_5_0_emb_layers_1_bias: ptr = w_slice_1d(data, 32833920, 640)
    _w_input_blocks_5_0_emb_layers_1_weight: ptr = w_slice_2d(data, 32834560, 640, 1280)
    _w_input_blocks_5_0_in_layers_0_bias: ptr = w_slice_1d(data, 33653760, 640)
    _w_input_blocks_5_0_in_layers_0_weight: ptr = w_slice_1d(data, 33654400, 640)
    _w_input_blocks_5_0_in_layers_2_bias: ptr = w_slice_1d(data, 33655040, 640)
    _w_input_blocks_5_0_in_layers_2_weight: ptr = w_slice_4d(data, 33655680, 640, 640, 3, 3)
    _w_input_blocks_5_0_out_layers_0_bias: ptr = w_slice_1d(data, 37342080, 640)
    _w_input_blocks_5_0_out_layers_0_weight: ptr = w_slice_1d(data, 37342720, 640)
    _w_input_blocks_5_0_out_layers_3_bias: ptr = w_slice_1d(data, 37343360, 640)
    _w_input_blocks_5_0_out_layers_3_weight: ptr = w_slice_4d(data, 37344000, 640, 640, 3, 3)
    _w_input_blocks_5_1_norm_bias: ptr = w_slice_1d(data, 41030400, 640)
    _w_input_blocks_5_1_norm_weight: ptr = w_slice_1d(data, 41031040, 640)
    _w_input_blocks_5_1_proj_in_bias: ptr = w_slice_1d(data, 41031680, 640)
    _w_input_blocks_5_1_proj_in_weight: ptr = w_slice_2d(data, 41032320, 640, 640)
    _w_input_blocks_5_1_proj_out_bias: ptr = w_slice_1d(data, 41441920, 640)
    _w_input_blocks_5_1_proj_out_weight: ptr = w_slice_2d(data, 41442560, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 41852160, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 42261760, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 42262400, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 42672000, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 43081600, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 43491200, 640, 2048)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 44801920, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 44802560, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 45212160, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 45621760, 640, 2048)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 46932480, 5120)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 46937600, 5120, 640)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 50214400, 640)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 50215040, 640, 2560)
    _w_input_blocks_5_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 51853440, 640)
    _w_input_blocks_5_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 51854080, 640)
    _w_input_blocks_5_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 51854720, 640)
    _w_input_blocks_5_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 51855360, 640)
    _w_input_blocks_5_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 51856000, 640)
    _w_input_blocks_5_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 51856640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 51857280, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 52266880, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 52267520, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 52677120, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 53086720, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 53496320, 640, 2048)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 54807040, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 54807680, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 55217280, 640, 640)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 55626880, 640, 2048)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 56937600, 5120)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 56942720, 5120, 640)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 60219520, 640)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 60220160, 640, 2560)
    _w_input_blocks_5_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 61858560, 640)
    _w_input_blocks_5_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 61859200, 640)
    _w_input_blocks_5_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 61859840, 640)
    _w_input_blocks_5_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 61860480, 640)
    _w_input_blocks_5_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 61861120, 640)
    _w_input_blocks_5_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 61861760, 640)
    _w_input_blocks_6_0_op_bias: ptr = w_slice_1d(data, 61862400, 640)
    _w_input_blocks_6_0_op_weight: ptr = w_slice_4d(data, 61863040, 640, 640, 3, 3)
    _w_input_blocks_7_0_emb_layers_1_bias: ptr = w_slice_1d(data, 65549440, 1280)
    _w_input_blocks_7_0_emb_layers_1_weight: ptr = w_slice_2d(data, 65550720, 1280, 1280)
    _w_input_blocks_7_0_in_layers_0_bias: ptr = w_slice_1d(data, 67189120, 640)
    _w_input_blocks_7_0_in_layers_0_weight: ptr = w_slice_1d(data, 67189760, 640)
    _w_input_blocks_7_0_in_layers_2_bias: ptr = w_slice_1d(data, 67190400, 1280)
    _w_input_blocks_7_0_in_layers_2_weight: ptr = w_slice_4d(data, 67191680, 1280, 640, 3, 3)
    _w_input_blocks_7_0_out_layers_0_bias: ptr = w_slice_1d(data, 74564480, 1280)
    _w_input_blocks_7_0_out_layers_0_weight: ptr = w_slice_1d(data, 74565760, 1280)
    _w_input_blocks_7_0_out_layers_3_bias: ptr = w_slice_1d(data, 74567040, 1280)
    _w_input_blocks_7_0_out_layers_3_weight: ptr = w_slice_4d(data, 74568320, 1280, 1280, 3, 3)
    _w_input_blocks_7_0_skip_connection_bias: ptr = w_slice_1d(data, 89313920, 1280)
    _w_input_blocks_7_0_skip_connection_weight: ptr = w_slice_4d(data, 89315200, 1280, 640, 1, 1)
    _w_input_blocks_7_1_norm_bias: ptr = w_slice_1d(data, 90134400, 1280)
    _w_input_blocks_7_1_norm_weight: ptr = w_slice_1d(data, 90135680, 1280)
    _w_input_blocks_7_1_proj_in_bias: ptr = w_slice_1d(data, 90136960, 1280)
    _w_input_blocks_7_1_proj_in_weight: ptr = w_slice_2d(data, 90138240, 1280, 1280)
    _w_input_blocks_7_1_proj_out_bias: ptr = w_slice_1d(data, 91776640, 1280)
    _w_input_blocks_7_1_proj_out_weight: ptr = w_slice_2d(data, 91777920, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 93416320, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 95054720, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 95056000, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 96694400, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 98332800, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 99971200, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 102592640, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 102593920, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 104232320, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 105870720, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 108492160, 10240)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 108502400, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 121609600, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 121610880, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 128164480, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 128165760, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 128167040, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 128168320, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 128169600, 1280)
    _w_input_blocks_7_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 128170880, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 128172160, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 129810560, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 129811840, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 131450240, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 133088640, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 134727040, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 137348480, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 137349760, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 138988160, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 140626560, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 143248000, 10240)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 143258240, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 156365440, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 156366720, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 162920320, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 162921600, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 162922880, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 162924160, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 162925440, 1280)
    _w_input_blocks_7_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 162926720, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 162928000, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 164566400, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 164567680, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 166206080, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 167844480, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 169482880, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 172104320, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 172105600, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 173744000, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 175382400, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 178003840, 10240)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 178014080, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 191121280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 191122560, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 197676160, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 197677440, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 197678720, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 197680000, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 197681280, 1280)
    _w_input_blocks_7_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 197682560, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 197683840, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 199322240, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 199323520, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 200961920, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 202600320, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 204238720, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 206860160, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 206861440, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 208499840, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 210138240, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 212759680, 10240)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 212769920, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 225877120, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 225878400, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 232432000, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 232433280, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 232434560, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 232435840, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 232437120, 1280)
    _w_input_blocks_7_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 232438400, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 232439680, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 234078080, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 234079360, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 235717760, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 237356160, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 238994560, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 241616000, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 241617280, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 243255680, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 244894080, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 247515520, 10240)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 247525760, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 260632960, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 260634240, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 267187840, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 267189120, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 267190400, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 267191680, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 267192960, 1280)
    _w_input_blocks_7_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 267194240, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 267195520, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 268833920, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 268835200, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 270473600, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 272112000, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 273750400, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 276371840, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 276373120, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 278011520, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 279649920, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 282271360, 10240)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 282281600, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 295388800, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 295390080, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 301943680, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 301944960, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 301946240, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 301947520, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 301948800, 1280)
    _w_input_blocks_7_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 301950080, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 301951360, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 303589760, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 303591040, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 305229440, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 306867840, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 308506240, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 311127680, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 311128960, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 312767360, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 314405760, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 317027200, 10240)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 317037440, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 330144640, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 330145920, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 336699520, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 336700800, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 336702080, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 336703360, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 336704640, 1280)
    _w_input_blocks_7_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 336705920, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 336707200, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 338345600, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 338346880, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 339985280, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 341623680, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 343262080, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 345883520, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 345884800, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 347523200, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 349161600, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 351783040, 10240)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 351793280, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 364900480, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 364901760, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 371455360, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 371456640, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 371457920, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 371459200, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 371460480, 1280)
    _w_input_blocks_7_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 371461760, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 371463040, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 373101440, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 373102720, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 374741120, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 376379520, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 378017920, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 380639360, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 380640640, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 382279040, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 383917440, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 386538880, 10240)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 386549120, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 399656320, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 399657600, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 406211200, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 406212480, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 406213760, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 406215040, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 406216320, 1280)
    _w_input_blocks_7_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 406217600, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 406218880, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 407857280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 407858560, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 409496960, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 411135360, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 412773760, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 415395200, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 415396480, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 417034880, 1280, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 418673280, 1280, 2048)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 421294720, 10240)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 421304960, 10240, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 434412160, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 434413440, 1280, 5120)
    _w_input_blocks_7_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 440967040, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 440968320, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 440969600, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 440970880, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 440972160, 1280)
    _w_input_blocks_7_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 440973440, 1280)
    _w_input_blocks_8_0_emb_layers_1_bias: ptr = w_slice_1d(data, 440974720, 1280)
    _w_input_blocks_8_0_emb_layers_1_weight: ptr = w_slice_2d(data, 440976000, 1280, 1280)
    _w_input_blocks_8_0_in_layers_0_bias: ptr = w_slice_1d(data, 442614400, 1280)
    _w_input_blocks_8_0_in_layers_0_weight: ptr = w_slice_1d(data, 442615680, 1280)
    _w_input_blocks_8_0_in_layers_2_bias: ptr = w_slice_1d(data, 442616960, 1280)
    _w_input_blocks_8_0_in_layers_2_weight: ptr = w_slice_4d(data, 442618240, 1280, 1280, 3, 3)
    _w_input_blocks_8_0_out_layers_0_bias: ptr = w_slice_1d(data, 457363840, 1280)
    _w_input_blocks_8_0_out_layers_0_weight: ptr = w_slice_1d(data, 457365120, 1280)
    _w_input_blocks_8_0_out_layers_3_bias: ptr = w_slice_1d(data, 457366400, 1280)
    _w_input_blocks_8_0_out_layers_3_weight: ptr = w_slice_4d(data, 457367680, 1280, 1280, 3, 3)
    _w_input_blocks_8_1_norm_bias: ptr = w_slice_1d(data, 472113280, 1280)
    _w_input_blocks_8_1_norm_weight: ptr = w_slice_1d(data, 472114560, 1280)
    _w_input_blocks_8_1_proj_in_bias: ptr = w_slice_1d(data, 472115840, 1280)
    _w_input_blocks_8_1_proj_in_weight: ptr = w_slice_2d(data, 472117120, 1280, 1280)
    _w_input_blocks_8_1_proj_out_bias: ptr = w_slice_1d(data, 473755520, 1280)
    _w_input_blocks_8_1_proj_out_weight: ptr = w_slice_2d(data, 473756800, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 475395200, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 477033600, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 477034880, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 478673280, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 480311680, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 481950080, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 484571520, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 484572800, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 486211200, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 487849600, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 490471040, 10240)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 490481280, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 503588480, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 503589760, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 510143360, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 510144640, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 510145920, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 510147200, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 510148480, 1280)
    _w_input_blocks_8_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 510149760, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 510151040, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 511789440, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 511790720, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 513429120, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 515067520, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 516705920, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 519327360, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 519328640, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 520967040, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 522605440, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 525226880, 10240)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 525237120, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 538344320, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 538345600, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 544899200, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 544900480, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 544901760, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 544903040, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 544904320, 1280)
    _w_input_blocks_8_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 544905600, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 544906880, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 546545280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 546546560, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 548184960, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 549823360, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 551461760, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 554083200, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 554084480, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 555722880, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 557361280, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 559982720, 10240)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 559992960, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 573100160, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 573101440, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 579655040, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 579656320, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 579657600, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 579658880, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 579660160, 1280)
    _w_input_blocks_8_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 579661440, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 579662720, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 581301120, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 581302400, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 582940800, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 584579200, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 586217600, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 588839040, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 588840320, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 590478720, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 592117120, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 594738560, 10240)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 594748800, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 607856000, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 607857280, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 614410880, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 614412160, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 614413440, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 614414720, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 614416000, 1280)
    _w_input_blocks_8_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 614417280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 614418560, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 616056960, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 616058240, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 617696640, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 619335040, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 620973440, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 623594880, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 623596160, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 625234560, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 626872960, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 629494400, 10240)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 629504640, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 642611840, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 642613120, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 649166720, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 649168000, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 649169280, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 649170560, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 649171840, 1280)
    _w_input_blocks_8_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 649173120, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 649174400, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 650812800, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 650814080, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 652452480, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 654090880, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 655729280, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 658350720, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 658352000, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 659990400, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 661628800, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 664250240, 10240)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 664260480, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 677367680, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 677368960, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 683922560, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 683923840, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 683925120, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 683926400, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 683927680, 1280)
    _w_input_blocks_8_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 683928960, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 683930240, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 685568640, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 685569920, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 687208320, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 688846720, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 690485120, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 693106560, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 693107840, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 694746240, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 696384640, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 699006080, 10240)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 699016320, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 712123520, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 712124800, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 718678400, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 718679680, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 718680960, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 718682240, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 718683520, 1280)
    _w_input_blocks_8_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 718684800, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 718686080, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 720324480, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 720325760, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 721964160, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 723602560, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 725240960, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 727862400, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 727863680, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 729502080, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 731140480, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 733761920, 10240)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 733772160, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 746879360, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 746880640, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 753434240, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 753435520, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 753436800, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 753438080, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 753439360, 1280)
    _w_input_blocks_8_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 753440640, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 753441920, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 755080320, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 755081600, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 756720000, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 758358400, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 759996800, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 762618240, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 762619520, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 764257920, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 765896320, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 768517760, 10240)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 768528000, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 781635200, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 781636480, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 788190080, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 788191360, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 788192640, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 788193920, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 788195200, 1280)
    _w_input_blocks_8_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 788196480, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 788197760, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 789836160, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 789837440, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 791475840, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 793114240, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 794752640, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 797374080, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 797375360, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 799013760, 1280, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 800652160, 1280, 2048)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 803273600, 10240)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 803283840, 10240, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 816391040, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 816392320, 1280, 5120)
    _w_input_blocks_8_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 822945920, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 822947200, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 822948480, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 822949760, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 822951040, 1280)
    _w_input_blocks_8_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 822952320, 1280)
    _w_label_emb_0_0_bias: ptr = w_slice_1d(data, 822953600, 1280)
    _w_label_emb_0_0_weight: ptr = w_slice_2d(data, 822954880, 1280, 2816)
    _w_label_emb_0_2_bias: ptr = w_slice_1d(data, 826559360, 1280)
    _w_label_emb_0_2_weight: ptr = w_slice_2d(data, 826560640, 1280, 1280)
    _w_middle_block_0_emb_layers_1_bias: ptr = w_slice_1d(data, 828199040, 1280)
    _w_middle_block_0_emb_layers_1_weight: ptr = w_slice_2d(data, 828200320, 1280, 1280)
    _w_middle_block_0_in_layers_0_bias: ptr = w_slice_1d(data, 829838720, 1280)
    _w_middle_block_0_in_layers_0_weight: ptr = w_slice_1d(data, 829840000, 1280)
    _w_middle_block_0_in_layers_2_bias: ptr = w_slice_1d(data, 829841280, 1280)
    _w_middle_block_0_in_layers_2_weight: ptr = w_slice_4d(data, 829842560, 1280, 1280, 3, 3)
    _w_middle_block_0_out_layers_0_bias: ptr = w_slice_1d(data, 844588160, 1280)
    _w_middle_block_0_out_layers_0_weight: ptr = w_slice_1d(data, 844589440, 1280)
    _w_middle_block_0_out_layers_3_bias: ptr = w_slice_1d(data, 844590720, 1280)
    _w_middle_block_0_out_layers_3_weight: ptr = w_slice_4d(data, 844592000, 1280, 1280, 3, 3)
    _w_middle_block_1_norm_bias: ptr = w_slice_1d(data, 859337600, 1280)
    _w_middle_block_1_norm_weight: ptr = w_slice_1d(data, 859338880, 1280)
    _w_middle_block_1_proj_in_bias: ptr = w_slice_1d(data, 859340160, 1280)
    _w_middle_block_1_proj_in_weight: ptr = w_slice_2d(data, 859341440, 1280, 1280)
    _w_middle_block_1_proj_out_bias: ptr = w_slice_1d(data, 860979840, 1280)
    _w_middle_block_1_proj_out_weight: ptr = w_slice_2d(data, 860981120, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 862619520, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 864257920, 1280)
    _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 864259200, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 865897600, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 867536000, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 869174400, 1280, 2048)
    _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 871795840, 1280)
    _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 871797120, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 873435520, 1280, 1280)
    _w_middle_block_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 875073920, 1280, 2048)
    _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 877695360, 10240)
    _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 877705600, 10240, 1280)
    _w_middle_block_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 890812800, 1280)
    _w_middle_block_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 890814080, 1280, 5120)
    _w_middle_block_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 897367680, 1280)
    _w_middle_block_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 897368960, 1280)
    _w_middle_block_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 897370240, 1280)
    _w_middle_block_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 897371520, 1280)
    _w_middle_block_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 897372800, 1280)
    _w_middle_block_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 897374080, 1280)
    _w_middle_block_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 897375360, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 899013760, 1280)
    _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 899015040, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 900653440, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 902291840, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 903930240, 1280, 2048)
    _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 906551680, 1280)
    _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 906552960, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 908191360, 1280, 1280)
    _w_middle_block_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 909829760, 1280, 2048)
    _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 912451200, 10240)
    _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 912461440, 10240, 1280)
    _w_middle_block_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 925568640, 1280)
    _w_middle_block_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 925569920, 1280, 5120)
    _w_middle_block_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 932123520, 1280)
    _w_middle_block_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 932124800, 1280)
    _w_middle_block_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 932126080, 1280)
    _w_middle_block_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 932127360, 1280)
    _w_middle_block_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 932128640, 1280)
    _w_middle_block_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 932129920, 1280)
    _w_middle_block_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 932131200, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 933769600, 1280)
    _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 933770880, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 935409280, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 937047680, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 938686080, 1280, 2048)
    _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 941307520, 1280)
    _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 941308800, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 942947200, 1280, 1280)
    _w_middle_block_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 944585600, 1280, 2048)
    _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 947207040, 10240)
    _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 947217280, 10240, 1280)
    _w_middle_block_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 960324480, 1280)
    _w_middle_block_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 960325760, 1280, 5120)
    _w_middle_block_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 966879360, 1280)
    _w_middle_block_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 966880640, 1280)
    _w_middle_block_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 966881920, 1280)
    _w_middle_block_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 966883200, 1280)
    _w_middle_block_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 966884480, 1280)
    _w_middle_block_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 966885760, 1280)
    _w_middle_block_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 966887040, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 968525440, 1280)
    _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 968526720, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 970165120, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 971803520, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 973441920, 1280, 2048)
    _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 976063360, 1280)
    _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 976064640, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 977703040, 1280, 1280)
    _w_middle_block_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 979341440, 1280, 2048)
    _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 981962880, 10240)
    _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 981973120, 10240, 1280)
    _w_middle_block_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 995080320, 1280)
    _w_middle_block_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 995081600, 1280, 5120)
    _w_middle_block_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 1001635200, 1280)
    _w_middle_block_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 1001636480, 1280)
    _w_middle_block_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 1001637760, 1280)
    _w_middle_block_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 1001639040, 1280)
    _w_middle_block_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 1001640320, 1280)
    _w_middle_block_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 1001641600, 1280)
    _w_middle_block_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 1001642880, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1003281280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1003282560, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 1004920960, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 1006559360, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 1008197760, 1280, 2048)
    _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1010819200, 1280)
    _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1010820480, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 1012458880, 1280, 1280)
    _w_middle_block_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 1014097280, 1280, 2048)
    _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1016718720, 10240)
    _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1016728960, 10240, 1280)
    _w_middle_block_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 1029836160, 1280)
    _w_middle_block_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 1029837440, 1280, 5120)
    _w_middle_block_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 1036391040, 1280)
    _w_middle_block_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 1036392320, 1280)
    _w_middle_block_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 1036393600, 1280)
    _w_middle_block_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 1036394880, 1280)
    _w_middle_block_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 1036396160, 1280)
    _w_middle_block_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 1036397440, 1280)
    _w_middle_block_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 1036398720, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1038037120, 1280)
    _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1038038400, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 1039676800, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 1041315200, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 1042953600, 1280, 2048)
    _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1045575040, 1280)
    _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1045576320, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 1047214720, 1280, 1280)
    _w_middle_block_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 1048853120, 1280, 2048)
    _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1051474560, 10240)
    _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1051484800, 10240, 1280)
    _w_middle_block_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 1064592000, 1280)
    _w_middle_block_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 1064593280, 1280, 5120)
    _w_middle_block_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 1071146880, 1280)
    _w_middle_block_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 1071148160, 1280)
    _w_middle_block_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 1071149440, 1280)
    _w_middle_block_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 1071150720, 1280)
    _w_middle_block_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 1071152000, 1280)
    _w_middle_block_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 1071153280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 1071154560, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1072792960, 1280)
    _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1072794240, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 1074432640, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 1076071040, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 1077709440, 1280, 2048)
    _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1080330880, 1280)
    _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1080332160, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 1081970560, 1280, 1280)
    _w_middle_block_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 1083608960, 1280, 2048)
    _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1086230400, 10240)
    _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1086240640, 10240, 1280)
    _w_middle_block_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 1099347840, 1280)
    _w_middle_block_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 1099349120, 1280, 5120)
    _w_middle_block_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 1105902720, 1280)
    _w_middle_block_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 1105904000, 1280)
    _w_middle_block_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 1105905280, 1280)
    _w_middle_block_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 1105906560, 1280)
    _w_middle_block_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 1105907840, 1280)
    _w_middle_block_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 1105909120, 1280)
    _w_middle_block_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 1105910400, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1107548800, 1280)
    _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1107550080, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 1109188480, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 1110826880, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 1112465280, 1280, 2048)
    _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1115086720, 1280)
    _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1115088000, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 1116726400, 1280, 1280)
    _w_middle_block_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 1118364800, 1280, 2048)
    _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1120986240, 10240)
    _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1120996480, 10240, 1280)
    _w_middle_block_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 1134103680, 1280)
    _w_middle_block_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 1134104960, 1280, 5120)
    _w_middle_block_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 1140658560, 1280)
    _w_middle_block_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 1140659840, 1280)
    _w_middle_block_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 1140661120, 1280)
    _w_middle_block_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 1140662400, 1280)
    _w_middle_block_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 1140663680, 1280)
    _w_middle_block_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 1140664960, 1280)
    _w_middle_block_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 1140666240, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1142304640, 1280)
    _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1142305920, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 1143944320, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 1145582720, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 1147221120, 1280, 2048)
    _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1149842560, 1280)
    _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1149843840, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 1151482240, 1280, 1280)
    _w_middle_block_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 1153120640, 1280, 2048)
    _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1155742080, 10240)
    _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1155752320, 10240, 1280)
    _w_middle_block_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 1168859520, 1280)
    _w_middle_block_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 1168860800, 1280, 5120)
    _w_middle_block_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 1175414400, 1280)
    _w_middle_block_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 1175415680, 1280)
    _w_middle_block_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 1175416960, 1280)
    _w_middle_block_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 1175418240, 1280)
    _w_middle_block_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 1175419520, 1280)
    _w_middle_block_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 1175420800, 1280)
    _w_middle_block_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 1175422080, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1177060480, 1280)
    _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1177061760, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 1178700160, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 1180338560, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 1181976960, 1280, 2048)
    _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1184598400, 1280)
    _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1184599680, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 1186238080, 1280, 1280)
    _w_middle_block_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 1187876480, 1280, 2048)
    _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1190497920, 10240)
    _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1190508160, 10240, 1280)
    _w_middle_block_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 1203615360, 1280)
    _w_middle_block_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 1203616640, 1280, 5120)
    _w_middle_block_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 1210170240, 1280)
    _w_middle_block_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 1210171520, 1280)
    _w_middle_block_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 1210172800, 1280)
    _w_middle_block_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 1210174080, 1280)
    _w_middle_block_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 1210175360, 1280)
    _w_middle_block_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 1210176640, 1280)
    _w_middle_block_2_emb_layers_1_bias: ptr = w_slice_1d(data, 1210177920, 1280)
    _w_middle_block_2_emb_layers_1_weight: ptr = w_slice_2d(data, 1210179200, 1280, 1280)
    _w_middle_block_2_in_layers_0_bias: ptr = w_slice_1d(data, 1211817600, 1280)
    _w_middle_block_2_in_layers_0_weight: ptr = w_slice_1d(data, 1211818880, 1280)
    _w_middle_block_2_in_layers_2_bias: ptr = w_slice_1d(data, 1211820160, 1280)
    _w_middle_block_2_in_layers_2_weight: ptr = w_slice_4d(data, 1211821440, 1280, 1280, 3, 3)
    _w_middle_block_2_out_layers_0_bias: ptr = w_slice_1d(data, 1226567040, 1280)
    _w_middle_block_2_out_layers_0_weight: ptr = w_slice_1d(data, 1226568320, 1280)
    _w_middle_block_2_out_layers_3_bias: ptr = w_slice_1d(data, 1226569600, 1280)
    _w_middle_block_2_out_layers_3_weight: ptr = w_slice_4d(data, 1226570880, 1280, 1280, 3, 3)
    _w_out_0_bias: ptr = w_slice_1d(data, 1241316480, 320)
    _w_out_0_weight: ptr = w_slice_1d(data, 1241316800, 320)
    _w_out_2_bias: ptr = w_slice_1d(data, 1241317120, 4)
    _w_out_2_weight: ptr = w_slice_4d(data, 1241317124, 4, 320, 3, 3)
    _w_output_blocks_0_0_emb_layers_1_bias: ptr = w_slice_1d(data, 1241328644, 1280)
    _w_output_blocks_0_0_emb_layers_1_weight: ptr = w_slice_2d(data, 1241329924, 1280, 1280)
    _w_output_blocks_0_0_in_layers_0_bias: ptr = w_slice_1d(data, 1242968324, 2560)
    _w_output_blocks_0_0_in_layers_0_weight: ptr = w_slice_1d(data, 1242970884, 2560)
    _w_output_blocks_0_0_in_layers_2_bias: ptr = w_slice_1d(data, 1242973444, 1280)
    _w_output_blocks_0_0_in_layers_2_weight: ptr = w_slice_4d(data, 1242974724, 1280, 2560, 3, 3)
    _w_output_blocks_0_0_out_layers_0_bias: ptr = w_slice_1d(data, 1272465924, 1280)
    _w_output_blocks_0_0_out_layers_0_weight: ptr = w_slice_1d(data, 1272467204, 1280)
    _w_output_blocks_0_0_out_layers_3_bias: ptr = w_slice_1d(data, 1272468484, 1280)
    _w_output_blocks_0_0_out_layers_3_weight: ptr = w_slice_4d(data, 1272469764, 1280, 1280, 3, 3)
    _w_output_blocks_0_0_skip_connection_bias: ptr = w_slice_1d(data, 1287215364, 1280)
    _w_output_blocks_0_0_skip_connection_weight: ptr = w_slice_4d(data, 1287216644, 1280, 2560, 1, 1)
    _w_output_blocks_0_1_norm_bias: ptr = w_slice_1d(data, 1290493444, 1280)
    _w_output_blocks_0_1_norm_weight: ptr = w_slice_1d(data, 1290494724, 1280)
    _w_output_blocks_0_1_proj_in_bias: ptr = w_slice_1d(data, 1290496004, 1280)
    _w_output_blocks_0_1_proj_in_weight: ptr = w_slice_2d(data, 1290497284, 1280, 1280)
    _w_output_blocks_0_1_proj_out_bias: ptr = w_slice_1d(data, 1292135684, 1280)
    _w_output_blocks_0_1_proj_out_weight: ptr = w_slice_2d(data, 1292136964, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 1293775364, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1295413764, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1295415044, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 1297053444, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 1298691844, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 1300330244, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1302951684, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1302952964, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 1304591364, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 1306229764, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1308851204, 10240)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1308861444, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 1321968644, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 1321969924, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 1328523524, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 1328524804, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 1328526084, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 1328527364, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 1328528644, 1280)
    _w_output_blocks_0_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 1328529924, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 1328531204, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1330169604, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1330170884, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 1331809284, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 1333447684, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 1335086084, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1337707524, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1337708804, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 1339347204, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 1340985604, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1343607044, 10240)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1343617284, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 1356724484, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 1356725764, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 1363279364, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 1363280644, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 1363281924, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 1363283204, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 1363284484, 1280)
    _w_output_blocks_0_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 1363285764, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 1363287044, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1364925444, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1364926724, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 1366565124, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 1368203524, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 1369841924, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1372463364, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1372464644, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 1374103044, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 1375741444, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1378362884, 10240)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1378373124, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 1391480324, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 1391481604, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 1398035204, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 1398036484, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 1398037764, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 1398039044, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 1398040324, 1280)
    _w_output_blocks_0_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 1398041604, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 1398042884, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1399681284, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1399682564, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 1401320964, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 1402959364, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 1404597764, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1407219204, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1407220484, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 1408858884, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 1410497284, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1413118724, 10240)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1413128964, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 1426236164, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 1426237444, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 1432791044, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 1432792324, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 1432793604, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 1432794884, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 1432796164, 1280)
    _w_output_blocks_0_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 1432797444, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 1432798724, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1434437124, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1434438404, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 1436076804, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 1437715204, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 1439353604, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1441975044, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1441976324, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 1443614724, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 1445253124, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1447874564, 10240)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1447884804, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 1460992004, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 1460993284, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 1467546884, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 1467548164, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 1467549444, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 1467550724, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 1467552004, 1280)
    _w_output_blocks_0_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 1467553284, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 1467554564, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1469192964, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1469194244, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 1470832644, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 1472471044, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 1474109444, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1476730884, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1476732164, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 1478370564, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 1480008964, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1482630404, 10240)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1482640644, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 1495747844, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 1495749124, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 1502302724, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 1502304004, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 1502305284, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 1502306564, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 1502307844, 1280)
    _w_output_blocks_0_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 1502309124, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 1502310404, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1503948804, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1503950084, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 1505588484, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 1507226884, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 1508865284, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1511486724, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1511488004, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 1513126404, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 1514764804, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1517386244, 10240)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1517396484, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 1530503684, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 1530504964, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 1537058564, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 1537059844, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 1537061124, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 1537062404, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 1537063684, 1280)
    _w_output_blocks_0_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 1537064964, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 1537066244, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1538704644, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1538705924, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 1540344324, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 1541982724, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 1543621124, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1546242564, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1546243844, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 1547882244, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 1549520644, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1552142084, 10240)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1552152324, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 1565259524, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 1565260804, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 1571814404, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 1571815684, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 1571816964, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 1571818244, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 1571819524, 1280)
    _w_output_blocks_0_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 1571820804, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 1571822084, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1573460484, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1573461764, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 1575100164, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 1576738564, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 1578376964, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1580998404, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1580999684, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 1582638084, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 1584276484, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1586897924, 10240)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1586908164, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 1600015364, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 1600016644, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 1606570244, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 1606571524, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 1606572804, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 1606574084, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 1606575364, 1280)
    _w_output_blocks_0_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 1606576644, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 1606577924, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1608216324, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1608217604, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 1609856004, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 1611494404, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 1613132804, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1615754244, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1615755524, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 1617393924, 1280, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 1619032324, 1280, 2048)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1621653764, 10240)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1621664004, 10240, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 1634771204, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 1634772484, 1280, 5120)
    _w_output_blocks_0_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 1641326084, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 1641327364, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 1641328644, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 1641329924, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 1641331204, 1280)
    _w_output_blocks_0_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 1641332484, 1280)
    _w_output_blocks_1_0_emb_layers_1_bias: ptr = w_slice_1d(data, 1641333764, 1280)
    _w_output_blocks_1_0_emb_layers_1_weight: ptr = w_slice_2d(data, 1641335044, 1280, 1280)
    _w_output_blocks_1_0_in_layers_0_bias: ptr = w_slice_1d(data, 1642973444, 2560)
    _w_output_blocks_1_0_in_layers_0_weight: ptr = w_slice_1d(data, 1642976004, 2560)
    _w_output_blocks_1_0_in_layers_2_bias: ptr = w_slice_1d(data, 1642978564, 1280)
    _w_output_blocks_1_0_in_layers_2_weight: ptr = w_slice_4d(data, 1642979844, 1280, 2560, 3, 3)
    _w_output_blocks_1_0_out_layers_0_bias: ptr = w_slice_1d(data, 1672471044, 1280)
    _w_output_blocks_1_0_out_layers_0_weight: ptr = w_slice_1d(data, 1672472324, 1280)
    _w_output_blocks_1_0_out_layers_3_bias: ptr = w_slice_1d(data, 1672473604, 1280)
    _w_output_blocks_1_0_out_layers_3_weight: ptr = w_slice_4d(data, 1672474884, 1280, 1280, 3, 3)
    _w_output_blocks_1_0_skip_connection_bias: ptr = w_slice_1d(data, 1687220484, 1280)
    _w_output_blocks_1_0_skip_connection_weight: ptr = w_slice_4d(data, 1687221764, 1280, 2560, 1, 1)
    _w_output_blocks_1_1_norm_bias: ptr = w_slice_1d(data, 1690498564, 1280)
    _w_output_blocks_1_1_norm_weight: ptr = w_slice_1d(data, 1690499844, 1280)
    _w_output_blocks_1_1_proj_in_bias: ptr = w_slice_1d(data, 1690501124, 1280)
    _w_output_blocks_1_1_proj_in_weight: ptr = w_slice_2d(data, 1690502404, 1280, 1280)
    _w_output_blocks_1_1_proj_out_bias: ptr = w_slice_1d(data, 1692140804, 1280)
    _w_output_blocks_1_1_proj_out_weight: ptr = w_slice_2d(data, 1692142084, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 1693780484, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1695418884, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1695420164, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 1697058564, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 1698696964, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 1700335364, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1702956804, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1702958084, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 1704596484, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 1706234884, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1708856324, 10240)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1708866564, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 1721973764, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 1721975044, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 1728528644, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 1728529924, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 1728531204, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 1728532484, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 1728533764, 1280)
    _w_output_blocks_1_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 1728535044, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 1728536324, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1730174724, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1730176004, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 1731814404, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 1733452804, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 1735091204, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1737712644, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1737713924, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 1739352324, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 1740990724, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1743612164, 10240)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1743622404, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 1756729604, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 1756730884, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 1763284484, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 1763285764, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 1763287044, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 1763288324, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 1763289604, 1280)
    _w_output_blocks_1_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 1763290884, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 1763292164, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1764930564, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1764931844, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 1766570244, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 1768208644, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 1769847044, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1772468484, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1772469764, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 1774108164, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 1775746564, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1778368004, 10240)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1778378244, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 1791485444, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 1791486724, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 1798040324, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 1798041604, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 1798042884, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 1798044164, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 1798045444, 1280)
    _w_output_blocks_1_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 1798046724, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 1798048004, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1799686404, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1799687684, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 1801326084, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 1802964484, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 1804602884, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1807224324, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1807225604, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 1808864004, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 1810502404, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1813123844, 10240)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1813134084, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 1826241284, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 1826242564, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 1832796164, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 1832797444, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 1832798724, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 1832800004, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 1832801284, 1280)
    _w_output_blocks_1_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 1832802564, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 1832803844, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1834442244, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1834443524, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 1836081924, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 1837720324, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 1839358724, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1841980164, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1841981444, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 1843619844, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 1845258244, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1847879684, 10240)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1847889924, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 1860997124, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 1860998404, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 1867552004, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 1867553284, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 1867554564, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 1867555844, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 1867557124, 1280)
    _w_output_blocks_1_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 1867558404, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 1867559684, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1869198084, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1869199364, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 1870837764, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 1872476164, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 1874114564, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1876736004, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1876737284, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 1878375684, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 1880014084, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1882635524, 10240)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1882645764, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 1895752964, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 1895754244, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 1902307844, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 1902309124, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 1902310404, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 1902311684, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 1902312964, 1280)
    _w_output_blocks_1_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 1902314244, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 1902315524, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1903953924, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1903955204, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 1905593604, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 1907232004, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 1908870404, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1911491844, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1911493124, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 1913131524, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 1914769924, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1917391364, 10240)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1917401604, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 1930508804, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 1930510084, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 1937063684, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 1937064964, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 1937066244, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 1937067524, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 1937068804, 1280)
    _w_output_blocks_1_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 1937070084, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 1937071364, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1938709764, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1938711044, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 1940349444, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 1941987844, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 1943626244, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1946247684, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1946248964, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 1947887364, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 1949525764, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1952147204, 10240)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1952157444, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 1965264644, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 1965265924, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 1971819524, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 1971820804, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 1971822084, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 1971823364, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 1971824644, 1280)
    _w_output_blocks_1_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 1971825924, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 1971827204, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 1973465604, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 1973466884, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 1975105284, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 1976743684, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 1978382084, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 1981003524, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 1981004804, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 1982643204, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 1984281604, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 1986903044, 10240)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 1986913284, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 2000020484, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 2000021764, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 2006575364, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 2006576644, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 2006577924, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 2006579204, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 2006580484, 1280)
    _w_output_blocks_1_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 2006581764, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 2006583044, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2008221444, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2008222724, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 2009861124, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 2011499524, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 2013137924, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2015759364, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2015760644, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 2017399044, 1280, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 2019037444, 1280, 2048)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2021658884, 10240)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2021669124, 10240, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 2034776324, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 2034777604, 1280, 5120)
    _w_output_blocks_1_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 2041331204, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 2041332484, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 2041333764, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 2041335044, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 2041336324, 1280)
    _w_output_blocks_1_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 2041337604, 1280)
    _w_output_blocks_2_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2041338884, 1280)
    _w_output_blocks_2_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2041340164, 1280, 1280)
    _w_output_blocks_2_0_in_layers_0_bias: ptr = w_slice_1d(data, 2042978564, 1920)
    _w_output_blocks_2_0_in_layers_0_weight: ptr = w_slice_1d(data, 2042980484, 1920)
    _w_output_blocks_2_0_in_layers_2_bias: ptr = w_slice_1d(data, 2042982404, 1280)
    _w_output_blocks_2_0_in_layers_2_weight: ptr = w_slice_4d(data, 2042983684, 1280, 1920, 3, 3)
    _w_output_blocks_2_0_out_layers_0_bias: ptr = w_slice_1d(data, 2065102084, 1280)
    _w_output_blocks_2_0_out_layers_0_weight: ptr = w_slice_1d(data, 2065103364, 1280)
    _w_output_blocks_2_0_out_layers_3_bias: ptr = w_slice_1d(data, 2065104644, 1280)
    _w_output_blocks_2_0_out_layers_3_weight: ptr = w_slice_4d(data, 2065105924, 1280, 1280, 3, 3)
    _w_output_blocks_2_0_skip_connection_bias: ptr = w_slice_1d(data, 2079851524, 1280)
    _w_output_blocks_2_0_skip_connection_weight: ptr = w_slice_4d(data, 2079852804, 1280, 1920, 1, 1)
    _w_output_blocks_2_1_norm_bias: ptr = w_slice_1d(data, 2082310404, 1280)
    _w_output_blocks_2_1_norm_weight: ptr = w_slice_1d(data, 2082311684, 1280)
    _w_output_blocks_2_1_proj_in_bias: ptr = w_slice_1d(data, 2082312964, 1280)
    _w_output_blocks_2_1_proj_in_weight: ptr = w_slice_2d(data, 2082314244, 1280, 1280)
    _w_output_blocks_2_1_proj_out_bias: ptr = w_slice_1d(data, 2083952644, 1280)
    _w_output_blocks_2_1_proj_out_weight: ptr = w_slice_2d(data, 2083953924, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 2085592324, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2087230724, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2087232004, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 2088870404, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 2090508804, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 2092147204, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2094768644, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2094769924, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 2096408324, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 2098046724, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2100668164, 10240)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2100678404, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 2113785604, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 2113786884, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 2120340484, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 2120341764, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 2120343044, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 2120344324, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 2120345604, 1280)
    _w_output_blocks_2_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 2120346884, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 2120348164, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2121986564, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2121987844, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 2123626244, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 2125264644, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 2126903044, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2129524484, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2129525764, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 2131164164, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 2132802564, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2135424004, 10240)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2135434244, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 2148541444, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 2148542724, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 2155096324, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 2155097604, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 2155098884, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 2155100164, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 2155101444, 1280)
    _w_output_blocks_2_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 2155102724, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight: ptr = w_slice_2d(data, 2155104004, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2156742404, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2156743684, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight: ptr = w_slice_2d(data, 2158382084, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight: ptr = w_slice_2d(data, 2160020484, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight: ptr = w_slice_2d(data, 2161658884, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2164280324, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2164281604, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight: ptr = w_slice_2d(data, 2165920004, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight: ptr = w_slice_2d(data, 2167558404, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2170179844, 10240)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2170190084, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_bias: ptr = w_slice_1d(data, 2183297284, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_weight: ptr = w_slice_2d(data, 2183298564, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_2_norm1_bias: ptr = w_slice_1d(data, 2189852164, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_norm1_weight: ptr = w_slice_1d(data, 2189853444, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_norm2_bias: ptr = w_slice_1d(data, 2189854724, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_norm2_weight: ptr = w_slice_1d(data, 2189856004, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_norm3_bias: ptr = w_slice_1d(data, 2189857284, 1280)
    _w_output_blocks_2_1_transformer_blocks_2_norm3_weight: ptr = w_slice_1d(data, 2189858564, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight: ptr = w_slice_2d(data, 2189859844, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2191498244, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2191499524, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight: ptr = w_slice_2d(data, 2193137924, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight: ptr = w_slice_2d(data, 2194776324, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight: ptr = w_slice_2d(data, 2196414724, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2199036164, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2199037444, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight: ptr = w_slice_2d(data, 2200675844, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight: ptr = w_slice_2d(data, 2202314244, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2204935684, 10240)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2204945924, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_bias: ptr = w_slice_1d(data, 2218053124, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_weight: ptr = w_slice_2d(data, 2218054404, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_3_norm1_bias: ptr = w_slice_1d(data, 2224608004, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_norm1_weight: ptr = w_slice_1d(data, 2224609284, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_norm2_bias: ptr = w_slice_1d(data, 2224610564, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_norm2_weight: ptr = w_slice_1d(data, 2224611844, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_norm3_bias: ptr = w_slice_1d(data, 2224613124, 1280)
    _w_output_blocks_2_1_transformer_blocks_3_norm3_weight: ptr = w_slice_1d(data, 2224614404, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight: ptr = w_slice_2d(data, 2224615684, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2226254084, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2226255364, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight: ptr = w_slice_2d(data, 2227893764, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight: ptr = w_slice_2d(data, 2229532164, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight: ptr = w_slice_2d(data, 2231170564, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2233792004, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2233793284, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight: ptr = w_slice_2d(data, 2235431684, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight: ptr = w_slice_2d(data, 2237070084, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2239691524, 10240)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2239701764, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_bias: ptr = w_slice_1d(data, 2252808964, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_weight: ptr = w_slice_2d(data, 2252810244, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_4_norm1_bias: ptr = w_slice_1d(data, 2259363844, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_norm1_weight: ptr = w_slice_1d(data, 2259365124, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_norm2_bias: ptr = w_slice_1d(data, 2259366404, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_norm2_weight: ptr = w_slice_1d(data, 2259367684, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_norm3_bias: ptr = w_slice_1d(data, 2259368964, 1280)
    _w_output_blocks_2_1_transformer_blocks_4_norm3_weight: ptr = w_slice_1d(data, 2259370244, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight: ptr = w_slice_2d(data, 2259371524, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2261009924, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2261011204, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight: ptr = w_slice_2d(data, 2262649604, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight: ptr = w_slice_2d(data, 2264288004, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight: ptr = w_slice_2d(data, 2265926404, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2268547844, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2268549124, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight: ptr = w_slice_2d(data, 2270187524, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight: ptr = w_slice_2d(data, 2271825924, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2274447364, 10240)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2274457604, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_bias: ptr = w_slice_1d(data, 2287564804, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_weight: ptr = w_slice_2d(data, 2287566084, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_5_norm1_bias: ptr = w_slice_1d(data, 2294119684, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_norm1_weight: ptr = w_slice_1d(data, 2294120964, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_norm2_bias: ptr = w_slice_1d(data, 2294122244, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_norm2_weight: ptr = w_slice_1d(data, 2294123524, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_norm3_bias: ptr = w_slice_1d(data, 2294124804, 1280)
    _w_output_blocks_2_1_transformer_blocks_5_norm3_weight: ptr = w_slice_1d(data, 2294126084, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight: ptr = w_slice_2d(data, 2294127364, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2295765764, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2295767044, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight: ptr = w_slice_2d(data, 2297405444, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight: ptr = w_slice_2d(data, 2299043844, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight: ptr = w_slice_2d(data, 2300682244, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2303303684, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2303304964, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight: ptr = w_slice_2d(data, 2304943364, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight: ptr = w_slice_2d(data, 2306581764, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2309203204, 10240)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2309213444, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_bias: ptr = w_slice_1d(data, 2322320644, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_weight: ptr = w_slice_2d(data, 2322321924, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_6_norm1_bias: ptr = w_slice_1d(data, 2328875524, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_norm1_weight: ptr = w_slice_1d(data, 2328876804, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_norm2_bias: ptr = w_slice_1d(data, 2328878084, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_norm2_weight: ptr = w_slice_1d(data, 2328879364, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_norm3_bias: ptr = w_slice_1d(data, 2328880644, 1280)
    _w_output_blocks_2_1_transformer_blocks_6_norm3_weight: ptr = w_slice_1d(data, 2328881924, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight: ptr = w_slice_2d(data, 2328883204, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2330521604, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2330522884, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight: ptr = w_slice_2d(data, 2332161284, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight: ptr = w_slice_2d(data, 2333799684, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight: ptr = w_slice_2d(data, 2335438084, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2338059524, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2338060804, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight: ptr = w_slice_2d(data, 2339699204, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight: ptr = w_slice_2d(data, 2341337604, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2343959044, 10240)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2343969284, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_bias: ptr = w_slice_1d(data, 2357076484, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_weight: ptr = w_slice_2d(data, 2357077764, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_7_norm1_bias: ptr = w_slice_1d(data, 2363631364, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_norm1_weight: ptr = w_slice_1d(data, 2363632644, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_norm2_bias: ptr = w_slice_1d(data, 2363633924, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_norm2_weight: ptr = w_slice_1d(data, 2363635204, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_norm3_bias: ptr = w_slice_1d(data, 2363636484, 1280)
    _w_output_blocks_2_1_transformer_blocks_7_norm3_weight: ptr = w_slice_1d(data, 2363637764, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight: ptr = w_slice_2d(data, 2363639044, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2365277444, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2365278724, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight: ptr = w_slice_2d(data, 2366917124, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight: ptr = w_slice_2d(data, 2368555524, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight: ptr = w_slice_2d(data, 2370193924, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2372815364, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2372816644, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight: ptr = w_slice_2d(data, 2374455044, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight: ptr = w_slice_2d(data, 2376093444, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2378714884, 10240)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2378725124, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_bias: ptr = w_slice_1d(data, 2391832324, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_weight: ptr = w_slice_2d(data, 2391833604, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_8_norm1_bias: ptr = w_slice_1d(data, 2398387204, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_norm1_weight: ptr = w_slice_1d(data, 2398388484, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_norm2_bias: ptr = w_slice_1d(data, 2398389764, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_norm2_weight: ptr = w_slice_1d(data, 2398391044, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_norm3_bias: ptr = w_slice_1d(data, 2398392324, 1280)
    _w_output_blocks_2_1_transformer_blocks_8_norm3_weight: ptr = w_slice_1d(data, 2398393604, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight: ptr = w_slice_2d(data, 2398394884, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2400033284, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2400034564, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight: ptr = w_slice_2d(data, 2401672964, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight: ptr = w_slice_2d(data, 2403311364, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight: ptr = w_slice_2d(data, 2404949764, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2407571204, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2407572484, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight: ptr = w_slice_2d(data, 2409210884, 1280, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight: ptr = w_slice_2d(data, 2410849284, 1280, 2048)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2413470724, 10240)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2413480964, 10240, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_bias: ptr = w_slice_1d(data, 2426588164, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_weight: ptr = w_slice_2d(data, 2426589444, 1280, 5120)
    _w_output_blocks_2_1_transformer_blocks_9_norm1_bias: ptr = w_slice_1d(data, 2433143044, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_norm1_weight: ptr = w_slice_1d(data, 2433144324, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_norm2_bias: ptr = w_slice_1d(data, 2433145604, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_norm2_weight: ptr = w_slice_1d(data, 2433146884, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_norm3_bias: ptr = w_slice_1d(data, 2433148164, 1280)
    _w_output_blocks_2_1_transformer_blocks_9_norm3_weight: ptr = w_slice_1d(data, 2433149444, 1280)
    _w_output_blocks_2_2_conv_bias: ptr = w_slice_1d(data, 2433150724, 1280)
    _w_output_blocks_2_2_conv_weight: ptr = w_slice_4d(data, 2433152004, 1280, 1280, 3, 3)
    _w_output_blocks_3_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2447897604, 640)
    _w_output_blocks_3_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2447898244, 640, 1280)
    _w_output_blocks_3_0_in_layers_0_bias: ptr = w_slice_1d(data, 2448717444, 1920)
    _w_output_blocks_3_0_in_layers_0_weight: ptr = w_slice_1d(data, 2448719364, 1920)
    _w_output_blocks_3_0_in_layers_2_bias: ptr = w_slice_1d(data, 2448721284, 640)
    _w_output_blocks_3_0_in_layers_2_weight: ptr = w_slice_4d(data, 2448721924, 640, 1920, 3, 3)
    _w_output_blocks_3_0_out_layers_0_bias: ptr = w_slice_1d(data, 2459781124, 640)
    _w_output_blocks_3_0_out_layers_0_weight: ptr = w_slice_1d(data, 2459781764, 640)
    _w_output_blocks_3_0_out_layers_3_bias: ptr = w_slice_1d(data, 2459782404, 640)
    _w_output_blocks_3_0_out_layers_3_weight: ptr = w_slice_4d(data, 2459783044, 640, 640, 3, 3)
    _w_output_blocks_3_0_skip_connection_bias: ptr = w_slice_1d(data, 2463469444, 640)
    _w_output_blocks_3_0_skip_connection_weight: ptr = w_slice_4d(data, 2463470084, 640, 1920, 1, 1)
    _w_output_blocks_3_1_norm_bias: ptr = w_slice_1d(data, 2464698884, 640)
    _w_output_blocks_3_1_norm_weight: ptr = w_slice_1d(data, 2464699524, 640)
    _w_output_blocks_3_1_proj_in_bias: ptr = w_slice_1d(data, 2464700164, 640)
    _w_output_blocks_3_1_proj_in_weight: ptr = w_slice_2d(data, 2464700804, 640, 640)
    _w_output_blocks_3_1_proj_out_bias: ptr = w_slice_1d(data, 2465110404, 640)
    _w_output_blocks_3_1_proj_out_weight: ptr = w_slice_2d(data, 2465111044, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 2465520644, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2465930244, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2465930884, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 2466340484, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 2466750084, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 2467159684, 640, 2048)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2468470404, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2468471044, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 2468880644, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 2469290244, 640, 2048)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2470600964, 5120)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2470606084, 5120, 640)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 2473882884, 640)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 2473883524, 640, 2560)
    _w_output_blocks_3_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 2475521924, 640)
    _w_output_blocks_3_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 2475522564, 640)
    _w_output_blocks_3_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 2475523204, 640)
    _w_output_blocks_3_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 2475523844, 640)
    _w_output_blocks_3_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 2475524484, 640)
    _w_output_blocks_3_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 2475525124, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 2475525764, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2475935364, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2475936004, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 2476345604, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 2476755204, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 2477164804, 640, 2048)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2478475524, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2478476164, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 2478885764, 640, 640)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 2479295364, 640, 2048)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2480606084, 5120)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2480611204, 5120, 640)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 2483888004, 640)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 2483888644, 640, 2560)
    _w_output_blocks_3_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 2485527044, 640)
    _w_output_blocks_3_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 2485527684, 640)
    _w_output_blocks_3_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 2485528324, 640)
    _w_output_blocks_3_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 2485528964, 640)
    _w_output_blocks_3_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 2485529604, 640)
    _w_output_blocks_3_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 2485530244, 640)
    _w_output_blocks_4_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2485530884, 640)
    _w_output_blocks_4_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2485531524, 640, 1280)
    _w_output_blocks_4_0_in_layers_0_bias: ptr = w_slice_1d(data, 2486350724, 1280)
    _w_output_blocks_4_0_in_layers_0_weight: ptr = w_slice_1d(data, 2486352004, 1280)
    _w_output_blocks_4_0_in_layers_2_bias: ptr = w_slice_1d(data, 2486353284, 640)
    _w_output_blocks_4_0_in_layers_2_weight: ptr = w_slice_4d(data, 2486353924, 640, 1280, 3, 3)
    _w_output_blocks_4_0_out_layers_0_bias: ptr = w_slice_1d(data, 2493726724, 640)
    _w_output_blocks_4_0_out_layers_0_weight: ptr = w_slice_1d(data, 2493727364, 640)
    _w_output_blocks_4_0_out_layers_3_bias: ptr = w_slice_1d(data, 2493728004, 640)
    _w_output_blocks_4_0_out_layers_3_weight: ptr = w_slice_4d(data, 2493728644, 640, 640, 3, 3)
    _w_output_blocks_4_0_skip_connection_bias: ptr = w_slice_1d(data, 2497415044, 640)
    _w_output_blocks_4_0_skip_connection_weight: ptr = w_slice_4d(data, 2497415684, 640, 1280, 1, 1)
    _w_output_blocks_4_1_norm_bias: ptr = w_slice_1d(data, 2498234884, 640)
    _w_output_blocks_4_1_norm_weight: ptr = w_slice_1d(data, 2498235524, 640)
    _w_output_blocks_4_1_proj_in_bias: ptr = w_slice_1d(data, 2498236164, 640)
    _w_output_blocks_4_1_proj_in_weight: ptr = w_slice_2d(data, 2498236804, 640, 640)
    _w_output_blocks_4_1_proj_out_bias: ptr = w_slice_1d(data, 2498646404, 640)
    _w_output_blocks_4_1_proj_out_weight: ptr = w_slice_2d(data, 2498647044, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 2499056644, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2499466244, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2499466884, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 2499876484, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 2500286084, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 2500695684, 640, 2048)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2502006404, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2502007044, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 2502416644, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 2502826244, 640, 2048)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2504136964, 5120)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2504142084, 5120, 640)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 2507418884, 640)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 2507419524, 640, 2560)
    _w_output_blocks_4_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 2509057924, 640)
    _w_output_blocks_4_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 2509058564, 640)
    _w_output_blocks_4_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 2509059204, 640)
    _w_output_blocks_4_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 2509059844, 640)
    _w_output_blocks_4_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 2509060484, 640)
    _w_output_blocks_4_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 2509061124, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 2509061764, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2509471364, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2509472004, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 2509881604, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 2510291204, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 2510700804, 640, 2048)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2512011524, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2512012164, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 2512421764, 640, 640)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 2512831364, 640, 2048)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2514142084, 5120)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2514147204, 5120, 640)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 2517424004, 640)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 2517424644, 640, 2560)
    _w_output_blocks_4_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 2519063044, 640)
    _w_output_blocks_4_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 2519063684, 640)
    _w_output_blocks_4_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 2519064324, 640)
    _w_output_blocks_4_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 2519064964, 640)
    _w_output_blocks_4_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 2519065604, 640)
    _w_output_blocks_4_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 2519066244, 640)
    _w_output_blocks_5_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2519066884, 640)
    _w_output_blocks_5_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2519067524, 640, 1280)
    _w_output_blocks_5_0_in_layers_0_bias: ptr = w_slice_1d(data, 2519886724, 960)
    _w_output_blocks_5_0_in_layers_0_weight: ptr = w_slice_1d(data, 2519887684, 960)
    _w_output_blocks_5_0_in_layers_2_bias: ptr = w_slice_1d(data, 2519888644, 640)
    _w_output_blocks_5_0_in_layers_2_weight: ptr = w_slice_4d(data, 2519889284, 640, 960, 3, 3)
    _w_output_blocks_5_0_out_layers_0_bias: ptr = w_slice_1d(data, 2525418884, 640)
    _w_output_blocks_5_0_out_layers_0_weight: ptr = w_slice_1d(data, 2525419524, 640)
    _w_output_blocks_5_0_out_layers_3_bias: ptr = w_slice_1d(data, 2525420164, 640)
    _w_output_blocks_5_0_out_layers_3_weight: ptr = w_slice_4d(data, 2525420804, 640, 640, 3, 3)
    _w_output_blocks_5_0_skip_connection_bias: ptr = w_slice_1d(data, 2529107204, 640)
    _w_output_blocks_5_0_skip_connection_weight: ptr = w_slice_4d(data, 2529107844, 640, 960, 1, 1)
    _w_output_blocks_5_1_norm_bias: ptr = w_slice_1d(data, 2529722244, 640)
    _w_output_blocks_5_1_norm_weight: ptr = w_slice_1d(data, 2529722884, 640)
    _w_output_blocks_5_1_proj_in_bias: ptr = w_slice_1d(data, 2529723524, 640)
    _w_output_blocks_5_1_proj_in_weight: ptr = w_slice_2d(data, 2529724164, 640, 640)
    _w_output_blocks_5_1_proj_out_bias: ptr = w_slice_1d(data, 2530133764, 640)
    _w_output_blocks_5_1_proj_out_weight: ptr = w_slice_2d(data, 2530134404, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight: ptr = w_slice_2d(data, 2530544004, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2530953604, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2530954244, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight: ptr = w_slice_2d(data, 2531363844, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight: ptr = w_slice_2d(data, 2531773444, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight: ptr = w_slice_2d(data, 2532183044, 640, 2048)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2533493764, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2533494404, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight: ptr = w_slice_2d(data, 2533904004, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight: ptr = w_slice_2d(data, 2534313604, 640, 2048)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2535624324, 5120)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2535629444, 5120, 640)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_bias: ptr = w_slice_1d(data, 2538906244, 640)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_weight: ptr = w_slice_2d(data, 2538906884, 640, 2560)
    _w_output_blocks_5_1_transformer_blocks_0_norm1_bias: ptr = w_slice_1d(data, 2540545284, 640)
    _w_output_blocks_5_1_transformer_blocks_0_norm1_weight: ptr = w_slice_1d(data, 2540545924, 640)
    _w_output_blocks_5_1_transformer_blocks_0_norm2_bias: ptr = w_slice_1d(data, 2540546564, 640)
    _w_output_blocks_5_1_transformer_blocks_0_norm2_weight: ptr = w_slice_1d(data, 2540547204, 640)
    _w_output_blocks_5_1_transformer_blocks_0_norm3_bias: ptr = w_slice_1d(data, 2540547844, 640)
    _w_output_blocks_5_1_transformer_blocks_0_norm3_weight: ptr = w_slice_1d(data, 2540548484, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight: ptr = w_slice_2d(data, 2540549124, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = w_slice_1d(data, 2540958724, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = w_slice_2d(data, 2540959364, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight: ptr = w_slice_2d(data, 2541368964, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight: ptr = w_slice_2d(data, 2541778564, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight: ptr = w_slice_2d(data, 2542188164, 640, 2048)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = w_slice_1d(data, 2543498884, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = w_slice_2d(data, 2543499524, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight: ptr = w_slice_2d(data, 2543909124, 640, 640)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight: ptr = w_slice_2d(data, 2544318724, 640, 2048)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = w_slice_1d(data, 2545629444, 5120)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = w_slice_2d(data, 2545634564, 5120, 640)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_bias: ptr = w_slice_1d(data, 2548911364, 640)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_weight: ptr = w_slice_2d(data, 2548912004, 640, 2560)
    _w_output_blocks_5_1_transformer_blocks_1_norm1_bias: ptr = w_slice_1d(data, 2550550404, 640)
    _w_output_blocks_5_1_transformer_blocks_1_norm1_weight: ptr = w_slice_1d(data, 2550551044, 640)
    _w_output_blocks_5_1_transformer_blocks_1_norm2_bias: ptr = w_slice_1d(data, 2550551684, 640)
    _w_output_blocks_5_1_transformer_blocks_1_norm2_weight: ptr = w_slice_1d(data, 2550552324, 640)
    _w_output_blocks_5_1_transformer_blocks_1_norm3_bias: ptr = w_slice_1d(data, 2550552964, 640)
    _w_output_blocks_5_1_transformer_blocks_1_norm3_weight: ptr = w_slice_1d(data, 2550553604, 640)
    _w_output_blocks_5_2_conv_bias: ptr = w_slice_1d(data, 2550554244, 640)
    _w_output_blocks_5_2_conv_weight: ptr = w_slice_4d(data, 2550554884, 640, 640, 3, 3)
    _w_output_blocks_6_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2554241284, 320)
    _w_output_blocks_6_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2554241604, 320, 1280)
    _w_output_blocks_6_0_in_layers_0_bias: ptr = w_slice_1d(data, 2554651204, 960)
    _w_output_blocks_6_0_in_layers_0_weight: ptr = w_slice_1d(data, 2554652164, 960)
    _w_output_blocks_6_0_in_layers_2_bias: ptr = w_slice_1d(data, 2554653124, 320)
    _w_output_blocks_6_0_in_layers_2_weight: ptr = w_slice_4d(data, 2554653444, 320, 960, 3, 3)
    _w_output_blocks_6_0_out_layers_0_bias: ptr = w_slice_1d(data, 2557418244, 320)
    _w_output_blocks_6_0_out_layers_0_weight: ptr = w_slice_1d(data, 2557418564, 320)
    _w_output_blocks_6_0_out_layers_3_bias: ptr = w_slice_1d(data, 2557418884, 320)
    _w_output_blocks_6_0_out_layers_3_weight: ptr = w_slice_4d(data, 2557419204, 320, 320, 3, 3)
    _w_output_blocks_6_0_skip_connection_bias: ptr = w_slice_1d(data, 2558340804, 320)
    _w_output_blocks_6_0_skip_connection_weight: ptr = w_slice_4d(data, 2558341124, 320, 960, 1, 1)
    _w_output_blocks_7_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2558648324, 320)
    _w_output_blocks_7_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2558648644, 320, 1280)
    _w_output_blocks_7_0_in_layers_0_bias: ptr = w_slice_1d(data, 2559058244, 640)
    _w_output_blocks_7_0_in_layers_0_weight: ptr = w_slice_1d(data, 2559058884, 640)
    _w_output_blocks_7_0_in_layers_2_bias: ptr = w_slice_1d(data, 2559059524, 320)
    _w_output_blocks_7_0_in_layers_2_weight: ptr = w_slice_4d(data, 2559059844, 320, 640, 3, 3)
    _w_output_blocks_7_0_out_layers_0_bias: ptr = w_slice_1d(data, 2560903044, 320)
    _w_output_blocks_7_0_out_layers_0_weight: ptr = w_slice_1d(data, 2560903364, 320)
    _w_output_blocks_7_0_out_layers_3_bias: ptr = w_slice_1d(data, 2560903684, 320)
    _w_output_blocks_7_0_out_layers_3_weight: ptr = w_slice_4d(data, 2560904004, 320, 320, 3, 3)
    _w_output_blocks_7_0_skip_connection_bias: ptr = w_slice_1d(data, 2561825604, 320)
    _w_output_blocks_7_0_skip_connection_weight: ptr = w_slice_4d(data, 2561825924, 320, 640, 1, 1)
    _w_output_blocks_8_0_emb_layers_1_bias: ptr = w_slice_1d(data, 2562030724, 320)
    _w_output_blocks_8_0_emb_layers_1_weight: ptr = w_slice_2d(data, 2562031044, 320, 1280)
    _w_output_blocks_8_0_in_layers_0_bias: ptr = w_slice_1d(data, 2562440644, 640)
    _w_output_blocks_8_0_in_layers_0_weight: ptr = w_slice_1d(data, 2562441284, 640)
    _w_output_blocks_8_0_in_layers_2_bias: ptr = w_slice_1d(data, 2562441924, 320)
    _w_output_blocks_8_0_in_layers_2_weight: ptr = w_slice_4d(data, 2562442244, 320, 640, 3, 3)
    _w_output_blocks_8_0_out_layers_0_bias: ptr = w_slice_1d(data, 2564285444, 320)
    _w_output_blocks_8_0_out_layers_0_weight: ptr = w_slice_1d(data, 2564285764, 320)
    _w_output_blocks_8_0_out_layers_3_bias: ptr = w_slice_1d(data, 2564286084, 320)
    _w_output_blocks_8_0_out_layers_3_weight: ptr = w_slice_4d(data, 2564286404, 320, 320, 3, 3)
    _w_output_blocks_8_0_skip_connection_bias: ptr = w_slice_1d(data, 2565208004, 320)
    _w_output_blocks_8_0_skip_connection_weight: ptr = w_slice_4d(data, 2565208324, 320, 640, 1, 1)
    _w_time_embed_0_bias: ptr = w_slice_1d(data, 2565413124, 1280)
    _w_time_embed_0_weight: ptr = w_slice_2d(data, 2565414404, 1280, 320)
    _w_time_embed_2_bias: ptr = w_slice_1d(data, 2565824004, 1280)
    _w_time_embed_2_weight: ptr = w_slice_2d(data, 2565825284, 1280, 1280)
    emb = linear_torch(emb, _w_time_embed_0_weight, _w_time_embed_0_bias, n, 320, 1280)
    emb = silu_torch(emb)
    emb = linear_torch(emb, _w_time_embed_2_weight, _w_time_embed_2_bias, n, 1280, 1280)
    emb = silu_torch(emb)

    h_cur = conv2d_torch(latent, _w_input_blocks_0_0_weight, _w_input_blocks_0_0_bias, n, 4, 320, hh, ww, 3, 1, 3//2)
    ptr_array_set(_s, 0, h_cur)

    # input_blocks.1
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_1_0_in_layers_0_weight, _w_input_blocks_1_0_in_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_1_0_in_layers_2_weight, _w_input_blocks_1_0_in_layers_2_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_1_0_emb_layers_1_weight, _w_input_blocks_1_0_emb_layers_1_bias, n, 1280, 320)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_1_0_out_layers_0_weight, _w_input_blocks_1_0_out_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_1_0_out_layers_3_weight, _w_input_blocks_1_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    ptr_array_set(_s, 1, h_cur)

    # input_blocks.2
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_2_0_in_layers_0_weight, _w_input_blocks_2_0_in_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_2_0_in_layers_2_weight, _w_input_blocks_2_0_in_layers_2_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_2_0_emb_layers_1_weight, _w_input_blocks_2_0_emb_layers_1_bias, n, 1280, 320)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_2_0_out_layers_0_weight, _w_input_blocks_2_0_out_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_2_0_out_layers_3_weight, _w_input_blocks_2_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    ptr_array_set(_s, 2, h_cur)

    h_cur = conv2d_torch(h_cur, _w_input_blocks_3_0_op_weight, _w_input_blocks_3_0_op_bias, n, 320, 320, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2
    ptr_array_set(_s, 3, h_cur)

    # input_blocks.4
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_4_0_in_layers_0_weight, _w_input_blocks_4_0_in_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_4_0_in_layers_2_weight, _w_input_blocks_4_0_in_layers_2_bias, n, 320, 640, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_4_0_emb_layers_1_weight, _w_input_blocks_4_0_emb_layers_1_bias, n, 1280, 640)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_4_0_out_layers_0_weight, _w_input_blocks_4_0_out_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_4_0_out_layers_3_weight, _w_input_blocks_4_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_input_blocks_4_0_skip_connection_weight, _w_input_blocks_4_0_skip_connection_bias, n, 320, 640, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.4.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_4_1_norm_weight, _w_input_blocks_4_1_norm_bias, 32, 640, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_input_blocks_4_1_proj_in_weight, _w_input_blocks_4_1_proj_in_bias, n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_4_1_transformer_blocks_0_norm1_weight, _w_input_blocks_4_1_transformer_blocks_0_norm1_bias,
        _w_input_blocks_4_1_transformer_blocks_0_norm2_weight, _w_input_blocks_4_1_transformer_blocks_0_norm2_bias,
        _w_input_blocks_4_1_transformer_blocks_0_norm3_weight, _w_input_blocks_4_1_transformer_blocks_0_norm3_bias,
        _w_input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight, _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_weight, _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_4_1_transformer_blocks_1_norm1_weight, _w_input_blocks_4_1_transformer_blocks_1_norm1_bias,
        _w_input_blocks_4_1_transformer_blocks_1_norm2_weight, _w_input_blocks_4_1_transformer_blocks_1_norm2_bias,
        _w_input_blocks_4_1_transformer_blocks_1_norm3_weight, _w_input_blocks_4_1_transformer_blocks_1_norm3_bias,
        _w_input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight, _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_weight, _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, _w_input_blocks_4_1_proj_out_weight, _w_input_blocks_4_1_proj_out_bias, n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    ptr_array_set(_s, 4, h_cur)

    # input_blocks.5
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_5_0_in_layers_0_weight, _w_input_blocks_5_0_in_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_5_0_in_layers_2_weight, _w_input_blocks_5_0_in_layers_2_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_5_0_emb_layers_1_weight, _w_input_blocks_5_0_emb_layers_1_bias, n, 1280, 640)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_5_0_out_layers_0_weight, _w_input_blocks_5_0_out_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_5_0_out_layers_3_weight, _w_input_blocks_5_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.5.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_5_1_norm_weight, _w_input_blocks_5_1_norm_bias, 32, 640, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_input_blocks_5_1_proj_in_weight, _w_input_blocks_5_1_proj_in_bias, n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_5_1_transformer_blocks_0_norm1_weight, _w_input_blocks_5_1_transformer_blocks_0_norm1_bias,
        _w_input_blocks_5_1_transformer_blocks_0_norm2_weight, _w_input_blocks_5_1_transformer_blocks_0_norm2_bias,
        _w_input_blocks_5_1_transformer_blocks_0_norm3_weight, _w_input_blocks_5_1_transformer_blocks_0_norm3_bias,
        _w_input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight, _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_weight, _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_5_1_transformer_blocks_1_norm1_weight, _w_input_blocks_5_1_transformer_blocks_1_norm1_bias,
        _w_input_blocks_5_1_transformer_blocks_1_norm2_weight, _w_input_blocks_5_1_transformer_blocks_1_norm2_bias,
        _w_input_blocks_5_1_transformer_blocks_1_norm3_weight, _w_input_blocks_5_1_transformer_blocks_1_norm3_bias,
        _w_input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight, _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_weight, _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, _w_input_blocks_5_1_proj_out_weight, _w_input_blocks_5_1_proj_out_bias, n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    ptr_array_set(_s, 5, h_cur)

    h_cur = conv2d_torch(h_cur, _w_input_blocks_6_0_op_weight, _w_input_blocks_6_0_op_bias, n, 640, 640, hh, ww, 3, 2, 1)
    hh = hh//2; ww = ww//2
    ptr_array_set(_s, 6, h_cur)

    # input_blocks.7
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_7_0_in_layers_0_weight, _w_input_blocks_7_0_in_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_7_0_in_layers_2_weight, _w_input_blocks_7_0_in_layers_2_bias, n, 640, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_7_0_emb_layers_1_weight, _w_input_blocks_7_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_7_0_out_layers_0_weight, _w_input_blocks_7_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_7_0_out_layers_3_weight, _w_input_blocks_7_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_input_blocks_7_0_skip_connection_weight, _w_input_blocks_7_0_skip_connection_bias, n, 640, 1280, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.7.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_7_1_norm_weight, _w_input_blocks_7_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_input_blocks_7_1_proj_in_weight, _w_input_blocks_7_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_0_norm1_weight, _w_input_blocks_7_1_transformer_blocks_0_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_0_norm2_weight, _w_input_blocks_7_1_transformer_blocks_0_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_0_norm3_weight, _w_input_blocks_7_1_transformer_blocks_0_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_1_norm1_weight, _w_input_blocks_7_1_transformer_blocks_1_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_1_norm2_weight, _w_input_blocks_7_1_transformer_blocks_1_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_1_norm3_weight, _w_input_blocks_7_1_transformer_blocks_1_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_2_norm1_weight, _w_input_blocks_7_1_transformer_blocks_2_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_2_norm2_weight, _w_input_blocks_7_1_transformer_blocks_2_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_2_norm3_weight, _w_input_blocks_7_1_transformer_blocks_2_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_3_norm1_weight, _w_input_blocks_7_1_transformer_blocks_3_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_3_norm2_weight, _w_input_blocks_7_1_transformer_blocks_3_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_3_norm3_weight, _w_input_blocks_7_1_transformer_blocks_3_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_4_norm1_weight, _w_input_blocks_7_1_transformer_blocks_4_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_4_norm2_weight, _w_input_blocks_7_1_transformer_blocks_4_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_4_norm3_weight, _w_input_blocks_7_1_transformer_blocks_4_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_5_norm1_weight, _w_input_blocks_7_1_transformer_blocks_5_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_5_norm2_weight, _w_input_blocks_7_1_transformer_blocks_5_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_5_norm3_weight, _w_input_blocks_7_1_transformer_blocks_5_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_6_norm1_weight, _w_input_blocks_7_1_transformer_blocks_6_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_6_norm2_weight, _w_input_blocks_7_1_transformer_blocks_6_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_6_norm3_weight, _w_input_blocks_7_1_transformer_blocks_6_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_7_norm1_weight, _w_input_blocks_7_1_transformer_blocks_7_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_7_norm2_weight, _w_input_blocks_7_1_transformer_blocks_7_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_7_norm3_weight, _w_input_blocks_7_1_transformer_blocks_7_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_8_norm1_weight, _w_input_blocks_7_1_transformer_blocks_8_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_8_norm2_weight, _w_input_blocks_7_1_transformer_blocks_8_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_8_norm3_weight, _w_input_blocks_7_1_transformer_blocks_8_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_7_1_transformer_blocks_9_norm1_weight, _w_input_blocks_7_1_transformer_blocks_9_norm1_bias,
        _w_input_blocks_7_1_transformer_blocks_9_norm2_weight, _w_input_blocks_7_1_transformer_blocks_9_norm2_bias,
        _w_input_blocks_7_1_transformer_blocks_9_norm3_weight, _w_input_blocks_7_1_transformer_blocks_9_norm3_bias,
        _w_input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight, _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_weight, _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_input_blocks_7_1_proj_out_weight, _w_input_blocks_7_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    ptr_array_set(_s, 7, h_cur)

    # input_blocks.8
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_8_0_in_layers_0_weight, _w_input_blocks_8_0_in_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_8_0_in_layers_2_weight, _w_input_blocks_8_0_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_input_blocks_8_0_emb_layers_1_weight, _w_input_blocks_8_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_8_0_out_layers_0_weight, _w_input_blocks_8_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_input_blocks_8_0_out_layers_3_weight, _w_input_blocks_8_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.8.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_input_blocks_8_1_norm_weight, _w_input_blocks_8_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_input_blocks_8_1_proj_in_weight, _w_input_blocks_8_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_0_norm1_weight, _w_input_blocks_8_1_transformer_blocks_0_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_0_norm2_weight, _w_input_blocks_8_1_transformer_blocks_0_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_0_norm3_weight, _w_input_blocks_8_1_transformer_blocks_0_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_1_norm1_weight, _w_input_blocks_8_1_transformer_blocks_1_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_1_norm2_weight, _w_input_blocks_8_1_transformer_blocks_1_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_1_norm3_weight, _w_input_blocks_8_1_transformer_blocks_1_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_2_norm1_weight, _w_input_blocks_8_1_transformer_blocks_2_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_2_norm2_weight, _w_input_blocks_8_1_transformer_blocks_2_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_2_norm3_weight, _w_input_blocks_8_1_transformer_blocks_2_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_3_norm1_weight, _w_input_blocks_8_1_transformer_blocks_3_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_3_norm2_weight, _w_input_blocks_8_1_transformer_blocks_3_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_3_norm3_weight, _w_input_blocks_8_1_transformer_blocks_3_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_4_norm1_weight, _w_input_blocks_8_1_transformer_blocks_4_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_4_norm2_weight, _w_input_blocks_8_1_transformer_blocks_4_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_4_norm3_weight, _w_input_blocks_8_1_transformer_blocks_4_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_5_norm1_weight, _w_input_blocks_8_1_transformer_blocks_5_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_5_norm2_weight, _w_input_blocks_8_1_transformer_blocks_5_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_5_norm3_weight, _w_input_blocks_8_1_transformer_blocks_5_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_6_norm1_weight, _w_input_blocks_8_1_transformer_blocks_6_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_6_norm2_weight, _w_input_blocks_8_1_transformer_blocks_6_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_6_norm3_weight, _w_input_blocks_8_1_transformer_blocks_6_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_7_norm1_weight, _w_input_blocks_8_1_transformer_blocks_7_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_7_norm2_weight, _w_input_blocks_8_1_transformer_blocks_7_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_7_norm3_weight, _w_input_blocks_8_1_transformer_blocks_7_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_8_norm1_weight, _w_input_blocks_8_1_transformer_blocks_8_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_8_norm2_weight, _w_input_blocks_8_1_transformer_blocks_8_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_8_norm3_weight, _w_input_blocks_8_1_transformer_blocks_8_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_input_blocks_8_1_transformer_blocks_9_norm1_weight, _w_input_blocks_8_1_transformer_blocks_9_norm1_bias,
        _w_input_blocks_8_1_transformer_blocks_9_norm2_weight, _w_input_blocks_8_1_transformer_blocks_9_norm2_bias,
        _w_input_blocks_8_1_transformer_blocks_9_norm3_weight, _w_input_blocks_8_1_transformer_blocks_9_norm3_bias,
        _w_input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight, _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_weight, _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_input_blocks_8_1_proj_out_weight, _w_input_blocks_8_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    ptr_array_set(_s, 8, h_cur)

    # middle_block
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_middle_block_0_in_layers_0_weight, _w_middle_block_0_in_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_middle_block_0_in_layers_2_weight, _w_middle_block_0_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_middle_block_0_emb_layers_1_weight, _w_middle_block_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_middle_block_0_out_layers_0_weight, _w_middle_block_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_middle_block_0_out_layers_3_weight, _w_middle_block_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer middle_block.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_middle_block_1_norm_weight, _w_middle_block_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_middle_block_1_proj_in_weight, _w_middle_block_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_0_norm1_weight, _w_middle_block_1_transformer_blocks_0_norm1_bias,
        _w_middle_block_1_transformer_blocks_0_norm2_weight, _w_middle_block_1_transformer_blocks_0_norm2_bias,
        _w_middle_block_1_transformer_blocks_0_norm3_weight, _w_middle_block_1_transformer_blocks_0_norm3_bias,
        _w_middle_block_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_0_ff_net_2_weight, _w_middle_block_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_1_norm1_weight, _w_middle_block_1_transformer_blocks_1_norm1_bias,
        _w_middle_block_1_transformer_blocks_1_norm2_weight, _w_middle_block_1_transformer_blocks_1_norm2_bias,
        _w_middle_block_1_transformer_blocks_1_norm3_weight, _w_middle_block_1_transformer_blocks_1_norm3_bias,
        _w_middle_block_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_1_ff_net_2_weight, _w_middle_block_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_2_norm1_weight, _w_middle_block_1_transformer_blocks_2_norm1_bias,
        _w_middle_block_1_transformer_blocks_2_norm2_weight, _w_middle_block_1_transformer_blocks_2_norm2_bias,
        _w_middle_block_1_transformer_blocks_2_norm3_weight, _w_middle_block_1_transformer_blocks_2_norm3_bias,
        _w_middle_block_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_2_ff_net_2_weight, _w_middle_block_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_3_norm1_weight, _w_middle_block_1_transformer_blocks_3_norm1_bias,
        _w_middle_block_1_transformer_blocks_3_norm2_weight, _w_middle_block_1_transformer_blocks_3_norm2_bias,
        _w_middle_block_1_transformer_blocks_3_norm3_weight, _w_middle_block_1_transformer_blocks_3_norm3_bias,
        _w_middle_block_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_3_ff_net_2_weight, _w_middle_block_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_4_norm1_weight, _w_middle_block_1_transformer_blocks_4_norm1_bias,
        _w_middle_block_1_transformer_blocks_4_norm2_weight, _w_middle_block_1_transformer_blocks_4_norm2_bias,
        _w_middle_block_1_transformer_blocks_4_norm3_weight, _w_middle_block_1_transformer_blocks_4_norm3_bias,
        _w_middle_block_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_4_ff_net_2_weight, _w_middle_block_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_5_norm1_weight, _w_middle_block_1_transformer_blocks_5_norm1_bias,
        _w_middle_block_1_transformer_blocks_5_norm2_weight, _w_middle_block_1_transformer_blocks_5_norm2_bias,
        _w_middle_block_1_transformer_blocks_5_norm3_weight, _w_middle_block_1_transformer_blocks_5_norm3_bias,
        _w_middle_block_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_5_ff_net_2_weight, _w_middle_block_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_6_norm1_weight, _w_middle_block_1_transformer_blocks_6_norm1_bias,
        _w_middle_block_1_transformer_blocks_6_norm2_weight, _w_middle_block_1_transformer_blocks_6_norm2_bias,
        _w_middle_block_1_transformer_blocks_6_norm3_weight, _w_middle_block_1_transformer_blocks_6_norm3_bias,
        _w_middle_block_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_6_ff_net_2_weight, _w_middle_block_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_7_norm1_weight, _w_middle_block_1_transformer_blocks_7_norm1_bias,
        _w_middle_block_1_transformer_blocks_7_norm2_weight, _w_middle_block_1_transformer_blocks_7_norm2_bias,
        _w_middle_block_1_transformer_blocks_7_norm3_weight, _w_middle_block_1_transformer_blocks_7_norm3_bias,
        _w_middle_block_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_7_ff_net_2_weight, _w_middle_block_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_8_norm1_weight, _w_middle_block_1_transformer_blocks_8_norm1_bias,
        _w_middle_block_1_transformer_blocks_8_norm2_weight, _w_middle_block_1_transformer_blocks_8_norm2_bias,
        _w_middle_block_1_transformer_blocks_8_norm3_weight, _w_middle_block_1_transformer_blocks_8_norm3_bias,
        _w_middle_block_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_8_ff_net_2_weight, _w_middle_block_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_middle_block_1_transformer_blocks_9_norm1_weight, _w_middle_block_1_transformer_blocks_9_norm1_bias,
        _w_middle_block_1_transformer_blocks_9_norm2_weight, _w_middle_block_1_transformer_blocks_9_norm2_bias,
        _w_middle_block_1_transformer_blocks_9_norm3_weight, _w_middle_block_1_transformer_blocks_9_norm3_bias,
        _w_middle_block_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_weight, _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_weight, _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_weight, _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_middle_block_1_transformer_blocks_9_ff_net_2_weight, _w_middle_block_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_middle_block_1_proj_out_weight, _w_middle_block_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_middle_block_2_in_layers_0_weight, _w_middle_block_2_in_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_middle_block_2_in_layers_2_weight, _w_middle_block_2_in_layers_2_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_middle_block_2_emb_layers_1_weight, _w_middle_block_2_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_middle_block_2_out_layers_0_weight, _w_middle_block_2_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_middle_block_2_out_layers_3_weight, _w_middle_block_2_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = _h_cur_orig
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)

    # output_blocks.0
    _cur = h_cur
    _skip = ptr_array_ref(_s, 8)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 1280, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_0_0_in_layers_0_weight, _w_output_blocks_0_0_in_layers_0_bias, 32, 2560, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_0_0_in_layers_2_weight, _w_output_blocks_0_0_in_layers_2_bias, n, 2560, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_0_0_emb_layers_1_weight, _w_output_blocks_0_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_0_0_out_layers_0_weight, _w_output_blocks_0_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_0_0_out_layers_3_weight, _w_output_blocks_0_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_0_0_skip_connection_weight, _w_output_blocks_0_0_skip_connection_bias, n, 2560, 1280, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.0.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_0_1_norm_weight, _w_output_blocks_0_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_0_1_proj_in_weight, _w_output_blocks_0_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_0_norm1_weight, _w_output_blocks_0_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_0_norm2_weight, _w_output_blocks_0_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_0_norm3_weight, _w_output_blocks_0_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_1_norm1_weight, _w_output_blocks_0_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_1_norm2_weight, _w_output_blocks_0_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_1_norm3_weight, _w_output_blocks_0_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_2_norm1_weight, _w_output_blocks_0_1_transformer_blocks_2_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_2_norm2_weight, _w_output_blocks_0_1_transformer_blocks_2_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_2_norm3_weight, _w_output_blocks_0_1_transformer_blocks_2_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_3_norm1_weight, _w_output_blocks_0_1_transformer_blocks_3_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_3_norm2_weight, _w_output_blocks_0_1_transformer_blocks_3_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_3_norm3_weight, _w_output_blocks_0_1_transformer_blocks_3_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_4_norm1_weight, _w_output_blocks_0_1_transformer_blocks_4_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_4_norm2_weight, _w_output_blocks_0_1_transformer_blocks_4_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_4_norm3_weight, _w_output_blocks_0_1_transformer_blocks_4_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_5_norm1_weight, _w_output_blocks_0_1_transformer_blocks_5_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_5_norm2_weight, _w_output_blocks_0_1_transformer_blocks_5_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_5_norm3_weight, _w_output_blocks_0_1_transformer_blocks_5_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_6_norm1_weight, _w_output_blocks_0_1_transformer_blocks_6_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_6_norm2_weight, _w_output_blocks_0_1_transformer_blocks_6_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_6_norm3_weight, _w_output_blocks_0_1_transformer_blocks_6_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_7_norm1_weight, _w_output_blocks_0_1_transformer_blocks_7_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_7_norm2_weight, _w_output_blocks_0_1_transformer_blocks_7_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_7_norm3_weight, _w_output_blocks_0_1_transformer_blocks_7_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_8_norm1_weight, _w_output_blocks_0_1_transformer_blocks_8_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_8_norm2_weight, _w_output_blocks_0_1_transformer_blocks_8_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_8_norm3_weight, _w_output_blocks_0_1_transformer_blocks_8_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_0_1_transformer_blocks_9_norm1_weight, _w_output_blocks_0_1_transformer_blocks_9_norm1_bias,
        _w_output_blocks_0_1_transformer_blocks_9_norm2_weight, _w_output_blocks_0_1_transformer_blocks_9_norm2_bias,
        _w_output_blocks_0_1_transformer_blocks_9_norm3_weight, _w_output_blocks_0_1_transformer_blocks_9_norm3_bias,
        _w_output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight, _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_weight, _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_output_blocks_0_1_proj_out_weight, _w_output_blocks_0_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.1
    _cur = h_cur
    _skip = ptr_array_ref(_s, 7)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 1280, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_1_0_in_layers_0_weight, _w_output_blocks_1_0_in_layers_0_bias, 32, 2560, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_1_0_in_layers_2_weight, _w_output_blocks_1_0_in_layers_2_bias, n, 2560, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_1_0_emb_layers_1_weight, _w_output_blocks_1_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_1_0_out_layers_0_weight, _w_output_blocks_1_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_1_0_out_layers_3_weight, _w_output_blocks_1_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_1_0_skip_connection_weight, _w_output_blocks_1_0_skip_connection_bias, n, 2560, 1280, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.1.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_1_1_norm_weight, _w_output_blocks_1_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_1_1_proj_in_weight, _w_output_blocks_1_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_0_norm1_weight, _w_output_blocks_1_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_0_norm2_weight, _w_output_blocks_1_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_0_norm3_weight, _w_output_blocks_1_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_1_norm1_weight, _w_output_blocks_1_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_1_norm2_weight, _w_output_blocks_1_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_1_norm3_weight, _w_output_blocks_1_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_2_norm1_weight, _w_output_blocks_1_1_transformer_blocks_2_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_2_norm2_weight, _w_output_blocks_1_1_transformer_blocks_2_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_2_norm3_weight, _w_output_blocks_1_1_transformer_blocks_2_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_3_norm1_weight, _w_output_blocks_1_1_transformer_blocks_3_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_3_norm2_weight, _w_output_blocks_1_1_transformer_blocks_3_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_3_norm3_weight, _w_output_blocks_1_1_transformer_blocks_3_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_4_norm1_weight, _w_output_blocks_1_1_transformer_blocks_4_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_4_norm2_weight, _w_output_blocks_1_1_transformer_blocks_4_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_4_norm3_weight, _w_output_blocks_1_1_transformer_blocks_4_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_5_norm1_weight, _w_output_blocks_1_1_transformer_blocks_5_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_5_norm2_weight, _w_output_blocks_1_1_transformer_blocks_5_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_5_norm3_weight, _w_output_blocks_1_1_transformer_blocks_5_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_6_norm1_weight, _w_output_blocks_1_1_transformer_blocks_6_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_6_norm2_weight, _w_output_blocks_1_1_transformer_blocks_6_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_6_norm3_weight, _w_output_blocks_1_1_transformer_blocks_6_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_7_norm1_weight, _w_output_blocks_1_1_transformer_blocks_7_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_7_norm2_weight, _w_output_blocks_1_1_transformer_blocks_7_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_7_norm3_weight, _w_output_blocks_1_1_transformer_blocks_7_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_8_norm1_weight, _w_output_blocks_1_1_transformer_blocks_8_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_8_norm2_weight, _w_output_blocks_1_1_transformer_blocks_8_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_8_norm3_weight, _w_output_blocks_1_1_transformer_blocks_8_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_1_1_transformer_blocks_9_norm1_weight, _w_output_blocks_1_1_transformer_blocks_9_norm1_bias,
        _w_output_blocks_1_1_transformer_blocks_9_norm2_weight, _w_output_blocks_1_1_transformer_blocks_9_norm2_bias,
        _w_output_blocks_1_1_transformer_blocks_9_norm3_weight, _w_output_blocks_1_1_transformer_blocks_9_norm3_bias,
        _w_output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight, _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_weight, _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_output_blocks_1_1_proj_out_weight, _w_output_blocks_1_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.2
    _cur = h_cur
    _skip = ptr_array_ref(_s, 6)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 1280, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_2_0_in_layers_0_weight, _w_output_blocks_2_0_in_layers_0_bias, 32, 1920, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_2_0_in_layers_2_weight, _w_output_blocks_2_0_in_layers_2_bias, n, 1920, 1280, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_2_0_emb_layers_1_weight, _w_output_blocks_2_0_emb_layers_1_bias, n, 1280, 1280)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_2_0_out_layers_0_weight, _w_output_blocks_2_0_out_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_2_0_out_layers_3_weight, _w_output_blocks_2_0_out_layers_3_bias, n, 1280, 1280, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_2_0_skip_connection_weight, _w_output_blocks_2_0_skip_connection_bias, n, 1920, 1280, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.2.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_2_1_norm_weight, _w_output_blocks_2_1_norm_bias, 32, 1280, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_2_1_proj_in_weight, _w_output_blocks_2_1_proj_in_bias, n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_0_norm1_weight, _w_output_blocks_2_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_0_norm2_weight, _w_output_blocks_2_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_0_norm3_weight, _w_output_blocks_2_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_1_norm1_weight, _w_output_blocks_2_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_1_norm2_weight, _w_output_blocks_2_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_1_norm3_weight, _w_output_blocks_2_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_2_norm1_weight, _w_output_blocks_2_1_transformer_blocks_2_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_2_norm2_weight, _w_output_blocks_2_1_transformer_blocks_2_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_2_norm3_weight, _w_output_blocks_2_1_transformer_blocks_2_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_3_norm1_weight, _w_output_blocks_2_1_transformer_blocks_3_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_3_norm2_weight, _w_output_blocks_2_1_transformer_blocks_3_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_3_norm3_weight, _w_output_blocks_2_1_transformer_blocks_3_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_4_norm1_weight, _w_output_blocks_2_1_transformer_blocks_4_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_4_norm2_weight, _w_output_blocks_2_1_transformer_blocks_4_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_4_norm3_weight, _w_output_blocks_2_1_transformer_blocks_4_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_5_norm1_weight, _w_output_blocks_2_1_transformer_blocks_5_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_5_norm2_weight, _w_output_blocks_2_1_transformer_blocks_5_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_5_norm3_weight, _w_output_blocks_2_1_transformer_blocks_5_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_6_norm1_weight, _w_output_blocks_2_1_transformer_blocks_6_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_6_norm2_weight, _w_output_blocks_2_1_transformer_blocks_6_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_6_norm3_weight, _w_output_blocks_2_1_transformer_blocks_6_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_7_norm1_weight, _w_output_blocks_2_1_transformer_blocks_7_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_7_norm2_weight, _w_output_blocks_2_1_transformer_blocks_7_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_7_norm3_weight, _w_output_blocks_2_1_transformer_blocks_7_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_8_norm1_weight, _w_output_blocks_2_1_transformer_blocks_8_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_8_norm2_weight, _w_output_blocks_2_1_transformer_blocks_8_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_8_norm3_weight, _w_output_blocks_2_1_transformer_blocks_8_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_2_1_transformer_blocks_9_norm1_weight, _w_output_blocks_2_1_transformer_blocks_9_norm1_bias,
        _w_output_blocks_2_1_transformer_blocks_9_norm2_weight, _w_output_blocks_2_1_transformer_blocks_9_norm2_bias,
        _w_output_blocks_2_1_transformer_blocks_9_norm3_weight, _w_output_blocks_2_1_transformer_blocks_9_norm3_bias,
        _w_output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight, 0,
        _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias,
        _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight, _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias,
        _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_weight, _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_bias,
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, _w_output_blocks_2_1_proj_out_weight, _w_output_blocks_2_1_proj_out_bias, n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    h_cur = upsample_nearest_torch(h_cur, 2)
    hh = hh*2; ww = ww*2
    h_cur = conv2d_torch(h_cur, _w_output_blocks_2_2_conv_weight, _w_output_blocks_2_2_conv_bias, n, 1280, 1280, hh, ww, 3, 1, 1)

    # output_blocks.3
    _cur = h_cur
    _skip = ptr_array_ref(_s, 5)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 1280, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_3_0_in_layers_0_weight, _w_output_blocks_3_0_in_layers_0_bias, 32, 1920, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_3_0_in_layers_2_weight, _w_output_blocks_3_0_in_layers_2_bias, n, 1920, 640, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_3_0_emb_layers_1_weight, _w_output_blocks_3_0_emb_layers_1_bias, n, 1280, 640)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_3_0_out_layers_0_weight, _w_output_blocks_3_0_out_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_3_0_out_layers_3_weight, _w_output_blocks_3_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_3_0_skip_connection_weight, _w_output_blocks_3_0_skip_connection_bias, n, 1920, 640, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.3.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_3_1_norm_weight, _w_output_blocks_3_1_norm_bias, 32, 640, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_3_1_proj_in_weight, _w_output_blocks_3_1_proj_in_bias, n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_3_1_transformer_blocks_0_norm1_weight, _w_output_blocks_3_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_3_1_transformer_blocks_0_norm2_weight, _w_output_blocks_3_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_3_1_transformer_blocks_0_norm3_weight, _w_output_blocks_3_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_3_1_transformer_blocks_1_norm1_weight, _w_output_blocks_3_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_3_1_transformer_blocks_1_norm2_weight, _w_output_blocks_3_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_3_1_transformer_blocks_1_norm3_weight, _w_output_blocks_3_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, _w_output_blocks_3_1_proj_out_weight, _w_output_blocks_3_1_proj_out_bias, n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.4
    _cur = h_cur
    _skip = ptr_array_ref(_s, 4)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 640, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_4_0_in_layers_0_weight, _w_output_blocks_4_0_in_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_4_0_in_layers_2_weight, _w_output_blocks_4_0_in_layers_2_bias, n, 1280, 640, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_4_0_emb_layers_1_weight, _w_output_blocks_4_0_emb_layers_1_bias, n, 1280, 640)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_4_0_out_layers_0_weight, _w_output_blocks_4_0_out_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_4_0_out_layers_3_weight, _w_output_blocks_4_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_4_0_skip_connection_weight, _w_output_blocks_4_0_skip_connection_bias, n, 1280, 640, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.4.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_4_1_norm_weight, _w_output_blocks_4_1_norm_bias, 32, 640, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_4_1_proj_in_weight, _w_output_blocks_4_1_proj_in_bias, n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_4_1_transformer_blocks_0_norm1_weight, _w_output_blocks_4_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_4_1_transformer_blocks_0_norm2_weight, _w_output_blocks_4_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_4_1_transformer_blocks_0_norm3_weight, _w_output_blocks_4_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_4_1_transformer_blocks_1_norm1_weight, _w_output_blocks_4_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_4_1_transformer_blocks_1_norm2_weight, _w_output_blocks_4_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_4_1_transformer_blocks_1_norm3_weight, _w_output_blocks_4_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, _w_output_blocks_4_1_proj_out_weight, _w_output_blocks_4_1_proj_out_bias, n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.5
    _cur = h_cur
    _skip = ptr_array_ref(_s, 3)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 640, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_5_0_in_layers_0_weight, _w_output_blocks_5_0_in_layers_0_bias, 32, 1280, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_5_0_in_layers_2_weight, _w_output_blocks_5_0_in_layers_2_bias, n, 1280, 640, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_5_0_emb_layers_1_weight, _w_output_blocks_5_0_emb_layers_1_bias, n, 1280, 640)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_5_0_out_layers_0_weight, _w_output_blocks_5_0_out_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_5_0_out_layers_3_weight, _w_output_blocks_5_0_out_layers_3_bias, n, 640, 640, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_5_0_skip_connection_weight, _w_output_blocks_5_0_skip_connection_bias, n, 1280, 640, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.5.1
    _x_in = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_5_1_norm_weight, _w_output_blocks_5_1_norm_bias, 32, 640, hh, ww)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, _w_output_blocks_5_1_proj_in_weight, _w_output_blocks_5_1_proj_in_bias, n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_5_1_transformer_blocks_0_norm1_weight, _w_output_blocks_5_1_transformer_blocks_0_norm1_bias,
        _w_output_blocks_5_1_transformer_blocks_0_norm2_weight, _w_output_blocks_5_1_transformer_blocks_0_norm2_bias,
        _w_output_blocks_5_1_transformer_blocks_0_norm3_weight, _w_output_blocks_5_1_transformer_blocks_0_norm3_bias,
        _w_output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias,
        _w_output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias,
        _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight, _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias,
        _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_weight, _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        _w_output_blocks_5_1_transformer_blocks_1_norm1_weight, _w_output_blocks_5_1_transformer_blocks_1_norm1_bias,
        _w_output_blocks_5_1_transformer_blocks_1_norm2_weight, _w_output_blocks_5_1_transformer_blocks_1_norm2_bias,
        _w_output_blocks_5_1_transformer_blocks_1_norm3_weight, _w_output_blocks_5_1_transformer_blocks_1_norm3_bias,
        _w_output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias,
        _w_output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight, 0,
        _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias,
        _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight, _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias,
        _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_weight, _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_bias,
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, _w_output_blocks_5_1_proj_out_weight, _w_output_blocks_5_1_proj_out_bias, n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    h_cur = upsample_nearest_torch(h_cur, 2)
    hh = hh*2; ww = ww*2
    h_cur = conv2d_torch(h_cur, _w_output_blocks_5_2_conv_weight, _w_output_blocks_5_2_conv_bias, n, 640, 640, hh, ww, 3, 1, 1)

    # output_blocks.6
    _cur = h_cur
    _skip = ptr_array_ref(_s, 2)
    h_cur = cat_channel_tensors(_cur, _skip, n, 320, 640, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_6_0_in_layers_0_weight, _w_output_blocks_6_0_in_layers_0_bias, 32, 960, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_6_0_in_layers_2_weight, _w_output_blocks_6_0_in_layers_2_bias, n, 960, 320, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_6_0_emb_layers_1_weight, _w_output_blocks_6_0_emb_layers_1_bias, n, 1280, 320)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_6_0_out_layers_0_weight, _w_output_blocks_6_0_out_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_6_0_out_layers_3_weight, _w_output_blocks_6_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_6_0_skip_connection_weight, _w_output_blocks_6_0_skip_connection_bias, n, 960, 320, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)

    # output_blocks.7
    _cur = h_cur
    _skip = ptr_array_ref(_s, 1)
    h_cur = cat_channel_tensors(_cur, _skip, n, 320, 320, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_7_0_in_layers_0_weight, _w_output_blocks_7_0_in_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_7_0_in_layers_2_weight, _w_output_blocks_7_0_in_layers_2_bias, n, 640, 320, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_7_0_emb_layers_1_weight, _w_output_blocks_7_0_emb_layers_1_bias, n, 1280, 320)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_7_0_out_layers_0_weight, _w_output_blocks_7_0_out_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_7_0_out_layers_3_weight, _w_output_blocks_7_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_7_0_skip_connection_weight, _w_output_blocks_7_0_skip_connection_bias, n, 640, 320, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)

    # output_blocks.8
    _cur = h_cur
    _skip = ptr_array_ref(_s, 0)
    h_cur = cat_channel_tensors(_cur, _skip, n, 320, 320, hh, ww)
    st_tensor_free(_cur)
    _h_cur_orig = st_clone(h_cur)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_8_0_in_layers_0_weight, _w_output_blocks_8_0_in_layers_0_bias, 32, 640, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_8_0_in_layers_2_weight, _w_output_blocks_8_0_in_layers_2_bias, n, 640, 320, hh, ww, 3, 1, 3//2)
    _se = silu_torch(emb)
    _y = linear_torch(_se, _w_output_blocks_8_0_emb_layers_1_weight, _w_output_blocks_8_0_emb_layers_1_bias, n, 1280, 320)
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_se); st_tensor_free(_y)
    h_cur = group_norm_torch(h_cur, _w_output_blocks_8_0_out_layers_0_weight, _w_output_blocks_8_0_out_layers_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_output_blocks_8_0_out_layers_3_weight, _w_output_blocks_8_0_out_layers_3_bias, n, 320, 320, hh, ww, 3, 1, 3//2)
    _sk = conv2d_torch(_h_cur_orig, _w_output_blocks_8_0_skip_connection_weight, _w_output_blocks_8_0_skip_connection_bias, n, 640, 320, hh, ww, 1, 1, 1//2)
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_sk)

    h_cur = group_norm_torch(h_cur, _w_out_0_weight, _w_out_0_bias, 32, 320, hh, ww)
    h_cur = silu_torch(h_cur)
    h_cur = conv2d_torch(h_cur, _w_out_2_weight, _w_out_2_bias, n, 320, 4, hh, ww, 3, 1, 1)
    return h_cur
