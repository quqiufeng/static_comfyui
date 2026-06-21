# unet_forward.static.py — 自动生成，所有偏移硬编码
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

def load_unet_weights(data: list[float]) -> ptr:
    _base: ptr = st_from_blob_half_flat(data, 2567463684)
    _weights: ptr = make_ptr_array(1681)
    ptr_array_set(_weights, 0, _base)
    _w_input_blocks_0_0_bias: ptr = unet_view_1d(_base, 0, 320)
    ptr_array_set(_weights, 1, _w_input_blocks_0_0_bias)
    _w_input_blocks_0_0_weight: ptr = unet_view_4d(_base, 320, 320, 4, 3, 3)
    ptr_array_set(_weights, 2, _w_input_blocks_0_0_weight)
    _w_input_blocks_1_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 11840, 320)
    ptr_array_set(_weights, 3, _w_input_blocks_1_0_emb_layers_1_bias)
    _w_input_blocks_1_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 12160, 320, 1280)
    ptr_array_set(_weights, 4, _w_input_blocks_1_0_emb_layers_1_weight)
    _w_input_blocks_1_0_in_layers_0_bias: ptr = unet_view_1d(_base, 421760, 320)
    ptr_array_set(_weights, 5, _w_input_blocks_1_0_in_layers_0_bias)
    _w_input_blocks_1_0_in_layers_0_weight: ptr = unet_view_1d(_base, 422080, 320)
    ptr_array_set(_weights, 6, _w_input_blocks_1_0_in_layers_0_weight)
    _w_input_blocks_1_0_in_layers_2_bias: ptr = unet_view_1d(_base, 422400, 320)
    ptr_array_set(_weights, 7, _w_input_blocks_1_0_in_layers_2_bias)
    _w_input_blocks_1_0_in_layers_2_weight: ptr = unet_view_4d(_base, 422720, 320, 320, 3, 3)
    ptr_array_set(_weights, 8, _w_input_blocks_1_0_in_layers_2_weight)
    _w_input_blocks_1_0_out_layers_0_bias: ptr = unet_view_1d(_base, 1344320, 320)
    ptr_array_set(_weights, 9, _w_input_blocks_1_0_out_layers_0_bias)
    _w_input_blocks_1_0_out_layers_0_weight: ptr = unet_view_1d(_base, 1344640, 320)
    ptr_array_set(_weights, 10, _w_input_blocks_1_0_out_layers_0_weight)
    _w_input_blocks_1_0_out_layers_3_bias: ptr = unet_view_1d(_base, 1344960, 320)
    ptr_array_set(_weights, 11, _w_input_blocks_1_0_out_layers_3_bias)
    _w_input_blocks_1_0_out_layers_3_weight: ptr = unet_view_4d(_base, 1345280, 320, 320, 3, 3)
    ptr_array_set(_weights, 12, _w_input_blocks_1_0_out_layers_3_weight)
    _w_input_blocks_2_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2266880, 320)
    ptr_array_set(_weights, 13, _w_input_blocks_2_0_emb_layers_1_bias)
    _w_input_blocks_2_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2267200, 320, 1280)
    ptr_array_set(_weights, 14, _w_input_blocks_2_0_emb_layers_1_weight)
    _w_input_blocks_2_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2676800, 320)
    ptr_array_set(_weights, 15, _w_input_blocks_2_0_in_layers_0_bias)
    _w_input_blocks_2_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2677120, 320)
    ptr_array_set(_weights, 16, _w_input_blocks_2_0_in_layers_0_weight)
    _w_input_blocks_2_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2677440, 320)
    ptr_array_set(_weights, 17, _w_input_blocks_2_0_in_layers_2_bias)
    _w_input_blocks_2_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2677760, 320, 320, 3, 3)
    ptr_array_set(_weights, 18, _w_input_blocks_2_0_in_layers_2_weight)
    _w_input_blocks_2_0_out_layers_0_bias: ptr = unet_view_1d(_base, 3599360, 320)
    ptr_array_set(_weights, 19, _w_input_blocks_2_0_out_layers_0_bias)
    _w_input_blocks_2_0_out_layers_0_weight: ptr = unet_view_1d(_base, 3599680, 320)
    ptr_array_set(_weights, 20, _w_input_blocks_2_0_out_layers_0_weight)
    _w_input_blocks_2_0_out_layers_3_bias: ptr = unet_view_1d(_base, 3600000, 320)
    ptr_array_set(_weights, 21, _w_input_blocks_2_0_out_layers_3_bias)
    _w_input_blocks_2_0_out_layers_3_weight: ptr = unet_view_4d(_base, 3600320, 320, 320, 3, 3)
    ptr_array_set(_weights, 22, _w_input_blocks_2_0_out_layers_3_weight)
    _w_input_blocks_3_0_op_bias: ptr = unet_view_1d(_base, 4521920, 320)
    ptr_array_set(_weights, 23, _w_input_blocks_3_0_op_bias)
    _w_input_blocks_3_0_op_weight: ptr = unet_view_4d(_base, 4522240, 320, 320, 3, 3)
    ptr_array_set(_weights, 24, _w_input_blocks_3_0_op_weight)
    _w_input_blocks_4_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 5443840, 640)
    ptr_array_set(_weights, 25, _w_input_blocks_4_0_emb_layers_1_bias)
    _w_input_blocks_4_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 5444480, 640, 1280)
    ptr_array_set(_weights, 26, _w_input_blocks_4_0_emb_layers_1_weight)
    _w_input_blocks_4_0_in_layers_0_bias: ptr = unet_view_1d(_base, 6263680, 320)
    ptr_array_set(_weights, 27, _w_input_blocks_4_0_in_layers_0_bias)
    _w_input_blocks_4_0_in_layers_0_weight: ptr = unet_view_1d(_base, 6264000, 320)
    ptr_array_set(_weights, 28, _w_input_blocks_4_0_in_layers_0_weight)
    _w_input_blocks_4_0_in_layers_2_bias: ptr = unet_view_1d(_base, 6264320, 640)
    ptr_array_set(_weights, 29, _w_input_blocks_4_0_in_layers_2_bias)
    _w_input_blocks_4_0_in_layers_2_weight: ptr = unet_view_4d(_base, 6264960, 640, 320, 3, 3)
    ptr_array_set(_weights, 30, _w_input_blocks_4_0_in_layers_2_weight)
    _w_input_blocks_4_0_out_layers_0_bias: ptr = unet_view_1d(_base, 8108160, 640)
    ptr_array_set(_weights, 31, _w_input_blocks_4_0_out_layers_0_bias)
    _w_input_blocks_4_0_out_layers_0_weight: ptr = unet_view_1d(_base, 8108800, 640)
    ptr_array_set(_weights, 32, _w_input_blocks_4_0_out_layers_0_weight)
    _w_input_blocks_4_0_out_layers_3_bias: ptr = unet_view_1d(_base, 8109440, 640)
    ptr_array_set(_weights, 33, _w_input_blocks_4_0_out_layers_3_bias)
    _w_input_blocks_4_0_out_layers_3_weight: ptr = unet_view_4d(_base, 8110080, 640, 640, 3, 3)
    ptr_array_set(_weights, 34, _w_input_blocks_4_0_out_layers_3_weight)
    _w_input_blocks_4_0_skip_connection_bias: ptr = unet_view_1d(_base, 11796480, 640)
    ptr_array_set(_weights, 35, _w_input_blocks_4_0_skip_connection_bias)
    _w_input_blocks_4_0_skip_connection_weight: ptr = unet_view_4d(_base, 11797120, 640, 320, 1, 1)
    ptr_array_set(_weights, 36, _w_input_blocks_4_0_skip_connection_weight)
    _w_input_blocks_4_1_norm_bias: ptr = unet_view_1d(_base, 12001920, 640)
    ptr_array_set(_weights, 37, _w_input_blocks_4_1_norm_bias)
    _w_input_blocks_4_1_norm_weight: ptr = unet_view_1d(_base, 12002560, 640)
    ptr_array_set(_weights, 38, _w_input_blocks_4_1_norm_weight)
    _w_input_blocks_4_1_proj_in_bias: ptr = unet_view_1d(_base, 12003200, 640)
    ptr_array_set(_weights, 39, _w_input_blocks_4_1_proj_in_bias)
    _w_input_blocks_4_1_proj_in_weight: ptr = unet_view_2d(_base, 12003840, 640, 640)
    ptr_array_set(_weights, 40, _w_input_blocks_4_1_proj_in_weight)
    _w_input_blocks_4_1_proj_out_bias: ptr = unet_view_1d(_base, 12413440, 640)
    ptr_array_set(_weights, 41, _w_input_blocks_4_1_proj_out_bias)
    _w_input_blocks_4_1_proj_out_weight: ptr = unet_view_2d(_base, 12414080, 640, 640)
    ptr_array_set(_weights, 42, _w_input_blocks_4_1_proj_out_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 12823680, 640, 640)
    ptr_array_set(_weights, 43, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_k_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 13233280, 640)
    ptr_array_set(_weights, 44, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 13233920, 640, 640)
    ptr_array_set(_weights, 45, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 13643520, 640, 640)
    ptr_array_set(_weights, 46, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_q_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 14053120, 640, 640)
    ptr_array_set(_weights, 47, _w_input_blocks_4_1_transformer_blocks_0_attn1_to_v_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 14462720, 640, 2048)
    ptr_array_set(_weights, 48, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_k_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 15773440, 640)
    ptr_array_set(_weights, 49, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 15774080, 640, 640)
    ptr_array_set(_weights, 50, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 16183680, 640, 640)
    ptr_array_set(_weights, 51, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_q_weight)
    _w_input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 16593280, 640, 2048)
    ptr_array_set(_weights, 52, _w_input_blocks_4_1_transformer_blocks_0_attn2_to_v_weight)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 17904000, 5120)
    ptr_array_set(_weights, 53, _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 17909120, 5120, 640)
    ptr_array_set(_weights, 54, _w_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 21185920, 640)
    ptr_array_set(_weights, 55, _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_bias)
    _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 21186560, 640, 2560)
    ptr_array_set(_weights, 56, _w_input_blocks_4_1_transformer_blocks_0_ff_net_2_weight)
    _w_input_blocks_4_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 22824960, 640)
    ptr_array_set(_weights, 57, _w_input_blocks_4_1_transformer_blocks_0_norm1_bias)
    _w_input_blocks_4_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 22825600, 640)
    ptr_array_set(_weights, 58, _w_input_blocks_4_1_transformer_blocks_0_norm1_weight)
    _w_input_blocks_4_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 22826240, 640)
    ptr_array_set(_weights, 59, _w_input_blocks_4_1_transformer_blocks_0_norm2_bias)
    _w_input_blocks_4_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 22826880, 640)
    ptr_array_set(_weights, 60, _w_input_blocks_4_1_transformer_blocks_0_norm2_weight)
    _w_input_blocks_4_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 22827520, 640)
    ptr_array_set(_weights, 61, _w_input_blocks_4_1_transformer_blocks_0_norm3_bias)
    _w_input_blocks_4_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 22828160, 640)
    ptr_array_set(_weights, 62, _w_input_blocks_4_1_transformer_blocks_0_norm3_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 22828800, 640, 640)
    ptr_array_set(_weights, 63, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_k_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 23238400, 640)
    ptr_array_set(_weights, 64, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 23239040, 640, 640)
    ptr_array_set(_weights, 65, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 23648640, 640, 640)
    ptr_array_set(_weights, 66, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_q_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 24058240, 640, 640)
    ptr_array_set(_weights, 67, _w_input_blocks_4_1_transformer_blocks_1_attn1_to_v_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 24467840, 640, 2048)
    ptr_array_set(_weights, 68, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_k_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 25778560, 640)
    ptr_array_set(_weights, 69, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 25779200, 640, 640)
    ptr_array_set(_weights, 70, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 26188800, 640, 640)
    ptr_array_set(_weights, 71, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_q_weight)
    _w_input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 26598400, 640, 2048)
    ptr_array_set(_weights, 72, _w_input_blocks_4_1_transformer_blocks_1_attn2_to_v_weight)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 27909120, 5120)
    ptr_array_set(_weights, 73, _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 27914240, 5120, 640)
    ptr_array_set(_weights, 74, _w_input_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 31191040, 640)
    ptr_array_set(_weights, 75, _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_bias)
    _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 31191680, 640, 2560)
    ptr_array_set(_weights, 76, _w_input_blocks_4_1_transformer_blocks_1_ff_net_2_weight)
    _w_input_blocks_4_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 32830080, 640)
    ptr_array_set(_weights, 77, _w_input_blocks_4_1_transformer_blocks_1_norm1_bias)
    _w_input_blocks_4_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 32830720, 640)
    ptr_array_set(_weights, 78, _w_input_blocks_4_1_transformer_blocks_1_norm1_weight)
    _w_input_blocks_4_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 32831360, 640)
    ptr_array_set(_weights, 79, _w_input_blocks_4_1_transformer_blocks_1_norm2_bias)
    _w_input_blocks_4_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 32832000, 640)
    ptr_array_set(_weights, 80, _w_input_blocks_4_1_transformer_blocks_1_norm2_weight)
    _w_input_blocks_4_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 32832640, 640)
    ptr_array_set(_weights, 81, _w_input_blocks_4_1_transformer_blocks_1_norm3_bias)
    _w_input_blocks_4_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 32833280, 640)
    ptr_array_set(_weights, 82, _w_input_blocks_4_1_transformer_blocks_1_norm3_weight)
    _w_input_blocks_5_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 32833920, 640)
    ptr_array_set(_weights, 83, _w_input_blocks_5_0_emb_layers_1_bias)
    _w_input_blocks_5_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 32834560, 640, 1280)
    ptr_array_set(_weights, 84, _w_input_blocks_5_0_emb_layers_1_weight)
    _w_input_blocks_5_0_in_layers_0_bias: ptr = unet_view_1d(_base, 33653760, 640)
    ptr_array_set(_weights, 85, _w_input_blocks_5_0_in_layers_0_bias)
    _w_input_blocks_5_0_in_layers_0_weight: ptr = unet_view_1d(_base, 33654400, 640)
    ptr_array_set(_weights, 86, _w_input_blocks_5_0_in_layers_0_weight)
    _w_input_blocks_5_0_in_layers_2_bias: ptr = unet_view_1d(_base, 33655040, 640)
    ptr_array_set(_weights, 87, _w_input_blocks_5_0_in_layers_2_bias)
    _w_input_blocks_5_0_in_layers_2_weight: ptr = unet_view_4d(_base, 33655680, 640, 640, 3, 3)
    ptr_array_set(_weights, 88, _w_input_blocks_5_0_in_layers_2_weight)
    _w_input_blocks_5_0_out_layers_0_bias: ptr = unet_view_1d(_base, 37342080, 640)
    ptr_array_set(_weights, 89, _w_input_blocks_5_0_out_layers_0_bias)
    _w_input_blocks_5_0_out_layers_0_weight: ptr = unet_view_1d(_base, 37342720, 640)
    ptr_array_set(_weights, 90, _w_input_blocks_5_0_out_layers_0_weight)
    _w_input_blocks_5_0_out_layers_3_bias: ptr = unet_view_1d(_base, 37343360, 640)
    ptr_array_set(_weights, 91, _w_input_blocks_5_0_out_layers_3_bias)
    _w_input_blocks_5_0_out_layers_3_weight: ptr = unet_view_4d(_base, 37344000, 640, 640, 3, 3)
    ptr_array_set(_weights, 92, _w_input_blocks_5_0_out_layers_3_weight)
    _w_input_blocks_5_1_norm_bias: ptr = unet_view_1d(_base, 41030400, 640)
    ptr_array_set(_weights, 93, _w_input_blocks_5_1_norm_bias)
    _w_input_blocks_5_1_norm_weight: ptr = unet_view_1d(_base, 41031040, 640)
    ptr_array_set(_weights, 94, _w_input_blocks_5_1_norm_weight)
    _w_input_blocks_5_1_proj_in_bias: ptr = unet_view_1d(_base, 41031680, 640)
    ptr_array_set(_weights, 95, _w_input_blocks_5_1_proj_in_bias)
    _w_input_blocks_5_1_proj_in_weight: ptr = unet_view_2d(_base, 41032320, 640, 640)
    ptr_array_set(_weights, 96, _w_input_blocks_5_1_proj_in_weight)
    _w_input_blocks_5_1_proj_out_bias: ptr = unet_view_1d(_base, 41441920, 640)
    ptr_array_set(_weights, 97, _w_input_blocks_5_1_proj_out_bias)
    _w_input_blocks_5_1_proj_out_weight: ptr = unet_view_2d(_base, 41442560, 640, 640)
    ptr_array_set(_weights, 98, _w_input_blocks_5_1_proj_out_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 41852160, 640, 640)
    ptr_array_set(_weights, 99, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_k_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 42261760, 640)
    ptr_array_set(_weights, 100, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 42262400, 640, 640)
    ptr_array_set(_weights, 101, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 42672000, 640, 640)
    ptr_array_set(_weights, 102, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_q_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 43081600, 640, 640)
    ptr_array_set(_weights, 103, _w_input_blocks_5_1_transformer_blocks_0_attn1_to_v_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 43491200, 640, 2048)
    ptr_array_set(_weights, 104, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_k_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 44801920, 640)
    ptr_array_set(_weights, 105, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 44802560, 640, 640)
    ptr_array_set(_weights, 106, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 45212160, 640, 640)
    ptr_array_set(_weights, 107, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_q_weight)
    _w_input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 45621760, 640, 2048)
    ptr_array_set(_weights, 108, _w_input_blocks_5_1_transformer_blocks_0_attn2_to_v_weight)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 46932480, 5120)
    ptr_array_set(_weights, 109, _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 46937600, 5120, 640)
    ptr_array_set(_weights, 110, _w_input_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 50214400, 640)
    ptr_array_set(_weights, 111, _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_bias)
    _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 50215040, 640, 2560)
    ptr_array_set(_weights, 112, _w_input_blocks_5_1_transformer_blocks_0_ff_net_2_weight)
    _w_input_blocks_5_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 51853440, 640)
    ptr_array_set(_weights, 113, _w_input_blocks_5_1_transformer_blocks_0_norm1_bias)
    _w_input_blocks_5_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 51854080, 640)
    ptr_array_set(_weights, 114, _w_input_blocks_5_1_transformer_blocks_0_norm1_weight)
    _w_input_blocks_5_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 51854720, 640)
    ptr_array_set(_weights, 115, _w_input_blocks_5_1_transformer_blocks_0_norm2_bias)
    _w_input_blocks_5_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 51855360, 640)
    ptr_array_set(_weights, 116, _w_input_blocks_5_1_transformer_blocks_0_norm2_weight)
    _w_input_blocks_5_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 51856000, 640)
    ptr_array_set(_weights, 117, _w_input_blocks_5_1_transformer_blocks_0_norm3_bias)
    _w_input_blocks_5_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 51856640, 640)
    ptr_array_set(_weights, 118, _w_input_blocks_5_1_transformer_blocks_0_norm3_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 51857280, 640, 640)
    ptr_array_set(_weights, 119, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_k_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 52266880, 640)
    ptr_array_set(_weights, 120, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 52267520, 640, 640)
    ptr_array_set(_weights, 121, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 52677120, 640, 640)
    ptr_array_set(_weights, 122, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_q_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 53086720, 640, 640)
    ptr_array_set(_weights, 123, _w_input_blocks_5_1_transformer_blocks_1_attn1_to_v_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 53496320, 640, 2048)
    ptr_array_set(_weights, 124, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_k_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 54807040, 640)
    ptr_array_set(_weights, 125, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 54807680, 640, 640)
    ptr_array_set(_weights, 126, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 55217280, 640, 640)
    ptr_array_set(_weights, 127, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_q_weight)
    _w_input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 55626880, 640, 2048)
    ptr_array_set(_weights, 128, _w_input_blocks_5_1_transformer_blocks_1_attn2_to_v_weight)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 56937600, 5120)
    ptr_array_set(_weights, 129, _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 56942720, 5120, 640)
    ptr_array_set(_weights, 130, _w_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 60219520, 640)
    ptr_array_set(_weights, 131, _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_bias)
    _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 60220160, 640, 2560)
    ptr_array_set(_weights, 132, _w_input_blocks_5_1_transformer_blocks_1_ff_net_2_weight)
    _w_input_blocks_5_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 61858560, 640)
    ptr_array_set(_weights, 133, _w_input_blocks_5_1_transformer_blocks_1_norm1_bias)
    _w_input_blocks_5_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 61859200, 640)
    ptr_array_set(_weights, 134, _w_input_blocks_5_1_transformer_blocks_1_norm1_weight)
    _w_input_blocks_5_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 61859840, 640)
    ptr_array_set(_weights, 135, _w_input_blocks_5_1_transformer_blocks_1_norm2_bias)
    _w_input_blocks_5_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 61860480, 640)
    ptr_array_set(_weights, 136, _w_input_blocks_5_1_transformer_blocks_1_norm2_weight)
    _w_input_blocks_5_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 61861120, 640)
    ptr_array_set(_weights, 137, _w_input_blocks_5_1_transformer_blocks_1_norm3_bias)
    _w_input_blocks_5_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 61861760, 640)
    ptr_array_set(_weights, 138, _w_input_blocks_5_1_transformer_blocks_1_norm3_weight)
    _w_input_blocks_6_0_op_bias: ptr = unet_view_1d(_base, 61862400, 640)
    ptr_array_set(_weights, 139, _w_input_blocks_6_0_op_bias)
    _w_input_blocks_6_0_op_weight: ptr = unet_view_4d(_base, 61863040, 640, 640, 3, 3)
    ptr_array_set(_weights, 140, _w_input_blocks_6_0_op_weight)
    _w_input_blocks_7_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 65549440, 1280)
    ptr_array_set(_weights, 141, _w_input_blocks_7_0_emb_layers_1_bias)
    _w_input_blocks_7_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 65550720, 1280, 1280)
    ptr_array_set(_weights, 142, _w_input_blocks_7_0_emb_layers_1_weight)
    _w_input_blocks_7_0_in_layers_0_bias: ptr = unet_view_1d(_base, 67189120, 640)
    ptr_array_set(_weights, 143, _w_input_blocks_7_0_in_layers_0_bias)
    _w_input_blocks_7_0_in_layers_0_weight: ptr = unet_view_1d(_base, 67189760, 640)
    ptr_array_set(_weights, 144, _w_input_blocks_7_0_in_layers_0_weight)
    _w_input_blocks_7_0_in_layers_2_bias: ptr = unet_view_1d(_base, 67190400, 1280)
    ptr_array_set(_weights, 145, _w_input_blocks_7_0_in_layers_2_bias)
    _w_input_blocks_7_0_in_layers_2_weight: ptr = unet_view_4d(_base, 67191680, 1280, 640, 3, 3)
    ptr_array_set(_weights, 146, _w_input_blocks_7_0_in_layers_2_weight)
    _w_input_blocks_7_0_out_layers_0_bias: ptr = unet_view_1d(_base, 74564480, 1280)
    ptr_array_set(_weights, 147, _w_input_blocks_7_0_out_layers_0_bias)
    _w_input_blocks_7_0_out_layers_0_weight: ptr = unet_view_1d(_base, 74565760, 1280)
    ptr_array_set(_weights, 148, _w_input_blocks_7_0_out_layers_0_weight)
    _w_input_blocks_7_0_out_layers_3_bias: ptr = unet_view_1d(_base, 74567040, 1280)
    ptr_array_set(_weights, 149, _w_input_blocks_7_0_out_layers_3_bias)
    _w_input_blocks_7_0_out_layers_3_weight: ptr = unet_view_4d(_base, 74568320, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 150, _w_input_blocks_7_0_out_layers_3_weight)
    _w_input_blocks_7_0_skip_connection_bias: ptr = unet_view_1d(_base, 89313920, 1280)
    ptr_array_set(_weights, 151, _w_input_blocks_7_0_skip_connection_bias)
    _w_input_blocks_7_0_skip_connection_weight: ptr = unet_view_4d(_base, 89315200, 1280, 640, 1, 1)
    ptr_array_set(_weights, 152, _w_input_blocks_7_0_skip_connection_weight)
    _w_input_blocks_7_1_norm_bias: ptr = unet_view_1d(_base, 90134400, 1280)
    ptr_array_set(_weights, 153, _w_input_blocks_7_1_norm_bias)
    _w_input_blocks_7_1_norm_weight: ptr = unet_view_1d(_base, 90135680, 1280)
    ptr_array_set(_weights, 154, _w_input_blocks_7_1_norm_weight)
    _w_input_blocks_7_1_proj_in_bias: ptr = unet_view_1d(_base, 90136960, 1280)
    ptr_array_set(_weights, 155, _w_input_blocks_7_1_proj_in_bias)
    _w_input_blocks_7_1_proj_in_weight: ptr = unet_view_2d(_base, 90138240, 1280, 1280)
    ptr_array_set(_weights, 156, _w_input_blocks_7_1_proj_in_weight)
    _w_input_blocks_7_1_proj_out_bias: ptr = unet_view_1d(_base, 91776640, 1280)
    ptr_array_set(_weights, 157, _w_input_blocks_7_1_proj_out_bias)
    _w_input_blocks_7_1_proj_out_weight: ptr = unet_view_2d(_base, 91777920, 1280, 1280)
    ptr_array_set(_weights, 158, _w_input_blocks_7_1_proj_out_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 93416320, 1280, 1280)
    ptr_array_set(_weights, 159, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 95054720, 1280)
    ptr_array_set(_weights, 160, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 95056000, 1280, 1280)
    ptr_array_set(_weights, 161, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 96694400, 1280, 1280)
    ptr_array_set(_weights, 162, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 98332800, 1280, 1280)
    ptr_array_set(_weights, 163, _w_input_blocks_7_1_transformer_blocks_0_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 99971200, 1280, 2048)
    ptr_array_set(_weights, 164, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 102592640, 1280)
    ptr_array_set(_weights, 165, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 102593920, 1280, 1280)
    ptr_array_set(_weights, 166, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 104232320, 1280, 1280)
    ptr_array_set(_weights, 167, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 105870720, 1280, 2048)
    ptr_array_set(_weights, 168, _w_input_blocks_7_1_transformer_blocks_0_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 108492160, 10240)
    ptr_array_set(_weights, 169, _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 108502400, 10240, 1280)
    ptr_array_set(_weights, 170, _w_input_blocks_7_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 121609600, 1280)
    ptr_array_set(_weights, 171, _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 121610880, 1280, 5120)
    ptr_array_set(_weights, 172, _w_input_blocks_7_1_transformer_blocks_0_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 128164480, 1280)
    ptr_array_set(_weights, 173, _w_input_blocks_7_1_transformer_blocks_0_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 128165760, 1280)
    ptr_array_set(_weights, 174, _w_input_blocks_7_1_transformer_blocks_0_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 128167040, 1280)
    ptr_array_set(_weights, 175, _w_input_blocks_7_1_transformer_blocks_0_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 128168320, 1280)
    ptr_array_set(_weights, 176, _w_input_blocks_7_1_transformer_blocks_0_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 128169600, 1280)
    ptr_array_set(_weights, 177, _w_input_blocks_7_1_transformer_blocks_0_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 128170880, 1280)
    ptr_array_set(_weights, 178, _w_input_blocks_7_1_transformer_blocks_0_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 128172160, 1280, 1280)
    ptr_array_set(_weights, 179, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 129810560, 1280)
    ptr_array_set(_weights, 180, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 129811840, 1280, 1280)
    ptr_array_set(_weights, 181, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 131450240, 1280, 1280)
    ptr_array_set(_weights, 182, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 133088640, 1280, 1280)
    ptr_array_set(_weights, 183, _w_input_blocks_7_1_transformer_blocks_1_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 134727040, 1280, 2048)
    ptr_array_set(_weights, 184, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 137348480, 1280)
    ptr_array_set(_weights, 185, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 137349760, 1280, 1280)
    ptr_array_set(_weights, 186, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 138988160, 1280, 1280)
    ptr_array_set(_weights, 187, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 140626560, 1280, 2048)
    ptr_array_set(_weights, 188, _w_input_blocks_7_1_transformer_blocks_1_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 143248000, 10240)
    ptr_array_set(_weights, 189, _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 143258240, 10240, 1280)
    ptr_array_set(_weights, 190, _w_input_blocks_7_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 156365440, 1280)
    ptr_array_set(_weights, 191, _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 156366720, 1280, 5120)
    ptr_array_set(_weights, 192, _w_input_blocks_7_1_transformer_blocks_1_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 162920320, 1280)
    ptr_array_set(_weights, 193, _w_input_blocks_7_1_transformer_blocks_1_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 162921600, 1280)
    ptr_array_set(_weights, 194, _w_input_blocks_7_1_transformer_blocks_1_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 162922880, 1280)
    ptr_array_set(_weights, 195, _w_input_blocks_7_1_transformer_blocks_1_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 162924160, 1280)
    ptr_array_set(_weights, 196, _w_input_blocks_7_1_transformer_blocks_1_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 162925440, 1280)
    ptr_array_set(_weights, 197, _w_input_blocks_7_1_transformer_blocks_1_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 162926720, 1280)
    ptr_array_set(_weights, 198, _w_input_blocks_7_1_transformer_blocks_1_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 162928000, 1280, 1280)
    ptr_array_set(_weights, 199, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 164566400, 1280)
    ptr_array_set(_weights, 200, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 164567680, 1280, 1280)
    ptr_array_set(_weights, 201, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 166206080, 1280, 1280)
    ptr_array_set(_weights, 202, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 167844480, 1280, 1280)
    ptr_array_set(_weights, 203, _w_input_blocks_7_1_transformer_blocks_2_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 169482880, 1280, 2048)
    ptr_array_set(_weights, 204, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 172104320, 1280)
    ptr_array_set(_weights, 205, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 172105600, 1280, 1280)
    ptr_array_set(_weights, 206, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 173744000, 1280, 1280)
    ptr_array_set(_weights, 207, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 175382400, 1280, 2048)
    ptr_array_set(_weights, 208, _w_input_blocks_7_1_transformer_blocks_2_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 178003840, 10240)
    ptr_array_set(_weights, 209, _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 178014080, 10240, 1280)
    ptr_array_set(_weights, 210, _w_input_blocks_7_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 191121280, 1280)
    ptr_array_set(_weights, 211, _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 191122560, 1280, 5120)
    ptr_array_set(_weights, 212, _w_input_blocks_7_1_transformer_blocks_2_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 197676160, 1280)
    ptr_array_set(_weights, 213, _w_input_blocks_7_1_transformer_blocks_2_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 197677440, 1280)
    ptr_array_set(_weights, 214, _w_input_blocks_7_1_transformer_blocks_2_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 197678720, 1280)
    ptr_array_set(_weights, 215, _w_input_blocks_7_1_transformer_blocks_2_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 197680000, 1280)
    ptr_array_set(_weights, 216, _w_input_blocks_7_1_transformer_blocks_2_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 197681280, 1280)
    ptr_array_set(_weights, 217, _w_input_blocks_7_1_transformer_blocks_2_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 197682560, 1280)
    ptr_array_set(_weights, 218, _w_input_blocks_7_1_transformer_blocks_2_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 197683840, 1280, 1280)
    ptr_array_set(_weights, 219, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 199322240, 1280)
    ptr_array_set(_weights, 220, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 199323520, 1280, 1280)
    ptr_array_set(_weights, 221, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 200961920, 1280, 1280)
    ptr_array_set(_weights, 222, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 202600320, 1280, 1280)
    ptr_array_set(_weights, 223, _w_input_blocks_7_1_transformer_blocks_3_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 204238720, 1280, 2048)
    ptr_array_set(_weights, 224, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 206860160, 1280)
    ptr_array_set(_weights, 225, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 206861440, 1280, 1280)
    ptr_array_set(_weights, 226, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 208499840, 1280, 1280)
    ptr_array_set(_weights, 227, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 210138240, 1280, 2048)
    ptr_array_set(_weights, 228, _w_input_blocks_7_1_transformer_blocks_3_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 212759680, 10240)
    ptr_array_set(_weights, 229, _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 212769920, 10240, 1280)
    ptr_array_set(_weights, 230, _w_input_blocks_7_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 225877120, 1280)
    ptr_array_set(_weights, 231, _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 225878400, 1280, 5120)
    ptr_array_set(_weights, 232, _w_input_blocks_7_1_transformer_blocks_3_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 232432000, 1280)
    ptr_array_set(_weights, 233, _w_input_blocks_7_1_transformer_blocks_3_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 232433280, 1280)
    ptr_array_set(_weights, 234, _w_input_blocks_7_1_transformer_blocks_3_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 232434560, 1280)
    ptr_array_set(_weights, 235, _w_input_blocks_7_1_transformer_blocks_3_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 232435840, 1280)
    ptr_array_set(_weights, 236, _w_input_blocks_7_1_transformer_blocks_3_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 232437120, 1280)
    ptr_array_set(_weights, 237, _w_input_blocks_7_1_transformer_blocks_3_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 232438400, 1280)
    ptr_array_set(_weights, 238, _w_input_blocks_7_1_transformer_blocks_3_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 232439680, 1280, 1280)
    ptr_array_set(_weights, 239, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 234078080, 1280)
    ptr_array_set(_weights, 240, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 234079360, 1280, 1280)
    ptr_array_set(_weights, 241, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 235717760, 1280, 1280)
    ptr_array_set(_weights, 242, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 237356160, 1280, 1280)
    ptr_array_set(_weights, 243, _w_input_blocks_7_1_transformer_blocks_4_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 238994560, 1280, 2048)
    ptr_array_set(_weights, 244, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 241616000, 1280)
    ptr_array_set(_weights, 245, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 241617280, 1280, 1280)
    ptr_array_set(_weights, 246, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 243255680, 1280, 1280)
    ptr_array_set(_weights, 247, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 244894080, 1280, 2048)
    ptr_array_set(_weights, 248, _w_input_blocks_7_1_transformer_blocks_4_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 247515520, 10240)
    ptr_array_set(_weights, 249, _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 247525760, 10240, 1280)
    ptr_array_set(_weights, 250, _w_input_blocks_7_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 260632960, 1280)
    ptr_array_set(_weights, 251, _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 260634240, 1280, 5120)
    ptr_array_set(_weights, 252, _w_input_blocks_7_1_transformer_blocks_4_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 267187840, 1280)
    ptr_array_set(_weights, 253, _w_input_blocks_7_1_transformer_blocks_4_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 267189120, 1280)
    ptr_array_set(_weights, 254, _w_input_blocks_7_1_transformer_blocks_4_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 267190400, 1280)
    ptr_array_set(_weights, 255, _w_input_blocks_7_1_transformer_blocks_4_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 267191680, 1280)
    ptr_array_set(_weights, 256, _w_input_blocks_7_1_transformer_blocks_4_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 267192960, 1280)
    ptr_array_set(_weights, 257, _w_input_blocks_7_1_transformer_blocks_4_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 267194240, 1280)
    ptr_array_set(_weights, 258, _w_input_blocks_7_1_transformer_blocks_4_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 267195520, 1280, 1280)
    ptr_array_set(_weights, 259, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 268833920, 1280)
    ptr_array_set(_weights, 260, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 268835200, 1280, 1280)
    ptr_array_set(_weights, 261, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 270473600, 1280, 1280)
    ptr_array_set(_weights, 262, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 272112000, 1280, 1280)
    ptr_array_set(_weights, 263, _w_input_blocks_7_1_transformer_blocks_5_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 273750400, 1280, 2048)
    ptr_array_set(_weights, 264, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 276371840, 1280)
    ptr_array_set(_weights, 265, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 276373120, 1280, 1280)
    ptr_array_set(_weights, 266, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 278011520, 1280, 1280)
    ptr_array_set(_weights, 267, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 279649920, 1280, 2048)
    ptr_array_set(_weights, 268, _w_input_blocks_7_1_transformer_blocks_5_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 282271360, 10240)
    ptr_array_set(_weights, 269, _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 282281600, 10240, 1280)
    ptr_array_set(_weights, 270, _w_input_blocks_7_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 295388800, 1280)
    ptr_array_set(_weights, 271, _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 295390080, 1280, 5120)
    ptr_array_set(_weights, 272, _w_input_blocks_7_1_transformer_blocks_5_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 301943680, 1280)
    ptr_array_set(_weights, 273, _w_input_blocks_7_1_transformer_blocks_5_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 301944960, 1280)
    ptr_array_set(_weights, 274, _w_input_blocks_7_1_transformer_blocks_5_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 301946240, 1280)
    ptr_array_set(_weights, 275, _w_input_blocks_7_1_transformer_blocks_5_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 301947520, 1280)
    ptr_array_set(_weights, 276, _w_input_blocks_7_1_transformer_blocks_5_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 301948800, 1280)
    ptr_array_set(_weights, 277, _w_input_blocks_7_1_transformer_blocks_5_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 301950080, 1280)
    ptr_array_set(_weights, 278, _w_input_blocks_7_1_transformer_blocks_5_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 301951360, 1280, 1280)
    ptr_array_set(_weights, 279, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 303589760, 1280)
    ptr_array_set(_weights, 280, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 303591040, 1280, 1280)
    ptr_array_set(_weights, 281, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 305229440, 1280, 1280)
    ptr_array_set(_weights, 282, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 306867840, 1280, 1280)
    ptr_array_set(_weights, 283, _w_input_blocks_7_1_transformer_blocks_6_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 308506240, 1280, 2048)
    ptr_array_set(_weights, 284, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 311127680, 1280)
    ptr_array_set(_weights, 285, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 311128960, 1280, 1280)
    ptr_array_set(_weights, 286, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 312767360, 1280, 1280)
    ptr_array_set(_weights, 287, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 314405760, 1280, 2048)
    ptr_array_set(_weights, 288, _w_input_blocks_7_1_transformer_blocks_6_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 317027200, 10240)
    ptr_array_set(_weights, 289, _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 317037440, 10240, 1280)
    ptr_array_set(_weights, 290, _w_input_blocks_7_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 330144640, 1280)
    ptr_array_set(_weights, 291, _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 330145920, 1280, 5120)
    ptr_array_set(_weights, 292, _w_input_blocks_7_1_transformer_blocks_6_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 336699520, 1280)
    ptr_array_set(_weights, 293, _w_input_blocks_7_1_transformer_blocks_6_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 336700800, 1280)
    ptr_array_set(_weights, 294, _w_input_blocks_7_1_transformer_blocks_6_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 336702080, 1280)
    ptr_array_set(_weights, 295, _w_input_blocks_7_1_transformer_blocks_6_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 336703360, 1280)
    ptr_array_set(_weights, 296, _w_input_blocks_7_1_transformer_blocks_6_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 336704640, 1280)
    ptr_array_set(_weights, 297, _w_input_blocks_7_1_transformer_blocks_6_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 336705920, 1280)
    ptr_array_set(_weights, 298, _w_input_blocks_7_1_transformer_blocks_6_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 336707200, 1280, 1280)
    ptr_array_set(_weights, 299, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 338345600, 1280)
    ptr_array_set(_weights, 300, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 338346880, 1280, 1280)
    ptr_array_set(_weights, 301, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 339985280, 1280, 1280)
    ptr_array_set(_weights, 302, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 341623680, 1280, 1280)
    ptr_array_set(_weights, 303, _w_input_blocks_7_1_transformer_blocks_7_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 343262080, 1280, 2048)
    ptr_array_set(_weights, 304, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 345883520, 1280)
    ptr_array_set(_weights, 305, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 345884800, 1280, 1280)
    ptr_array_set(_weights, 306, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 347523200, 1280, 1280)
    ptr_array_set(_weights, 307, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 349161600, 1280, 2048)
    ptr_array_set(_weights, 308, _w_input_blocks_7_1_transformer_blocks_7_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 351783040, 10240)
    ptr_array_set(_weights, 309, _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 351793280, 10240, 1280)
    ptr_array_set(_weights, 310, _w_input_blocks_7_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 364900480, 1280)
    ptr_array_set(_weights, 311, _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 364901760, 1280, 5120)
    ptr_array_set(_weights, 312, _w_input_blocks_7_1_transformer_blocks_7_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 371455360, 1280)
    ptr_array_set(_weights, 313, _w_input_blocks_7_1_transformer_blocks_7_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 371456640, 1280)
    ptr_array_set(_weights, 314, _w_input_blocks_7_1_transformer_blocks_7_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 371457920, 1280)
    ptr_array_set(_weights, 315, _w_input_blocks_7_1_transformer_blocks_7_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 371459200, 1280)
    ptr_array_set(_weights, 316, _w_input_blocks_7_1_transformer_blocks_7_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 371460480, 1280)
    ptr_array_set(_weights, 317, _w_input_blocks_7_1_transformer_blocks_7_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 371461760, 1280)
    ptr_array_set(_weights, 318, _w_input_blocks_7_1_transformer_blocks_7_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 371463040, 1280, 1280)
    ptr_array_set(_weights, 319, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 373101440, 1280)
    ptr_array_set(_weights, 320, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 373102720, 1280, 1280)
    ptr_array_set(_weights, 321, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 374741120, 1280, 1280)
    ptr_array_set(_weights, 322, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 376379520, 1280, 1280)
    ptr_array_set(_weights, 323, _w_input_blocks_7_1_transformer_blocks_8_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 378017920, 1280, 2048)
    ptr_array_set(_weights, 324, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 380639360, 1280)
    ptr_array_set(_weights, 325, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 380640640, 1280, 1280)
    ptr_array_set(_weights, 326, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 382279040, 1280, 1280)
    ptr_array_set(_weights, 327, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 383917440, 1280, 2048)
    ptr_array_set(_weights, 328, _w_input_blocks_7_1_transformer_blocks_8_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 386538880, 10240)
    ptr_array_set(_weights, 329, _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 386549120, 10240, 1280)
    ptr_array_set(_weights, 330, _w_input_blocks_7_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 399656320, 1280)
    ptr_array_set(_weights, 331, _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 399657600, 1280, 5120)
    ptr_array_set(_weights, 332, _w_input_blocks_7_1_transformer_blocks_8_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 406211200, 1280)
    ptr_array_set(_weights, 333, _w_input_blocks_7_1_transformer_blocks_8_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 406212480, 1280)
    ptr_array_set(_weights, 334, _w_input_blocks_7_1_transformer_blocks_8_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 406213760, 1280)
    ptr_array_set(_weights, 335, _w_input_blocks_7_1_transformer_blocks_8_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 406215040, 1280)
    ptr_array_set(_weights, 336, _w_input_blocks_7_1_transformer_blocks_8_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 406216320, 1280)
    ptr_array_set(_weights, 337, _w_input_blocks_7_1_transformer_blocks_8_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 406217600, 1280)
    ptr_array_set(_weights, 338, _w_input_blocks_7_1_transformer_blocks_8_norm3_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 406218880, 1280, 1280)
    ptr_array_set(_weights, 339, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 407857280, 1280)
    ptr_array_set(_weights, 340, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 407858560, 1280, 1280)
    ptr_array_set(_weights, 341, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 409496960, 1280, 1280)
    ptr_array_set(_weights, 342, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 411135360, 1280, 1280)
    ptr_array_set(_weights, 343, _w_input_blocks_7_1_transformer_blocks_9_attn1_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 412773760, 1280, 2048)
    ptr_array_set(_weights, 344, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_k_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 415395200, 1280)
    ptr_array_set(_weights, 345, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 415396480, 1280, 1280)
    ptr_array_set(_weights, 346, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 417034880, 1280, 1280)
    ptr_array_set(_weights, 347, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_q_weight)
    _w_input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 418673280, 1280, 2048)
    ptr_array_set(_weights, 348, _w_input_blocks_7_1_transformer_blocks_9_attn2_to_v_weight)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 421294720, 10240)
    ptr_array_set(_weights, 349, _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 421304960, 10240, 1280)
    ptr_array_set(_weights, 350, _w_input_blocks_7_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 434412160, 1280)
    ptr_array_set(_weights, 351, _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_bias)
    _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 434413440, 1280, 5120)
    ptr_array_set(_weights, 352, _w_input_blocks_7_1_transformer_blocks_9_ff_net_2_weight)
    _w_input_blocks_7_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 440967040, 1280)
    ptr_array_set(_weights, 353, _w_input_blocks_7_1_transformer_blocks_9_norm1_bias)
    _w_input_blocks_7_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 440968320, 1280)
    ptr_array_set(_weights, 354, _w_input_blocks_7_1_transformer_blocks_9_norm1_weight)
    _w_input_blocks_7_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 440969600, 1280)
    ptr_array_set(_weights, 355, _w_input_blocks_7_1_transformer_blocks_9_norm2_bias)
    _w_input_blocks_7_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 440970880, 1280)
    ptr_array_set(_weights, 356, _w_input_blocks_7_1_transformer_blocks_9_norm2_weight)
    _w_input_blocks_7_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 440972160, 1280)
    ptr_array_set(_weights, 357, _w_input_blocks_7_1_transformer_blocks_9_norm3_bias)
    _w_input_blocks_7_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 440973440, 1280)
    ptr_array_set(_weights, 358, _w_input_blocks_7_1_transformer_blocks_9_norm3_weight)
    _w_input_blocks_8_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 440974720, 1280)
    ptr_array_set(_weights, 359, _w_input_blocks_8_0_emb_layers_1_bias)
    _w_input_blocks_8_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 440976000, 1280, 1280)
    ptr_array_set(_weights, 360, _w_input_blocks_8_0_emb_layers_1_weight)
    _w_input_blocks_8_0_in_layers_0_bias: ptr = unet_view_1d(_base, 442614400, 1280)
    ptr_array_set(_weights, 361, _w_input_blocks_8_0_in_layers_0_bias)
    _w_input_blocks_8_0_in_layers_0_weight: ptr = unet_view_1d(_base, 442615680, 1280)
    ptr_array_set(_weights, 362, _w_input_blocks_8_0_in_layers_0_weight)
    _w_input_blocks_8_0_in_layers_2_bias: ptr = unet_view_1d(_base, 442616960, 1280)
    ptr_array_set(_weights, 363, _w_input_blocks_8_0_in_layers_2_bias)
    _w_input_blocks_8_0_in_layers_2_weight: ptr = unet_view_4d(_base, 442618240, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 364, _w_input_blocks_8_0_in_layers_2_weight)
    _w_input_blocks_8_0_out_layers_0_bias: ptr = unet_view_1d(_base, 457363840, 1280)
    ptr_array_set(_weights, 365, _w_input_blocks_8_0_out_layers_0_bias)
    _w_input_blocks_8_0_out_layers_0_weight: ptr = unet_view_1d(_base, 457365120, 1280)
    ptr_array_set(_weights, 366, _w_input_blocks_8_0_out_layers_0_weight)
    _w_input_blocks_8_0_out_layers_3_bias: ptr = unet_view_1d(_base, 457366400, 1280)
    ptr_array_set(_weights, 367, _w_input_blocks_8_0_out_layers_3_bias)
    _w_input_blocks_8_0_out_layers_3_weight: ptr = unet_view_4d(_base, 457367680, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 368, _w_input_blocks_8_0_out_layers_3_weight)
    _w_input_blocks_8_1_norm_bias: ptr = unet_view_1d(_base, 472113280, 1280)
    ptr_array_set(_weights, 369, _w_input_blocks_8_1_norm_bias)
    _w_input_blocks_8_1_norm_weight: ptr = unet_view_1d(_base, 472114560, 1280)
    ptr_array_set(_weights, 370, _w_input_blocks_8_1_norm_weight)
    _w_input_blocks_8_1_proj_in_bias: ptr = unet_view_1d(_base, 472115840, 1280)
    ptr_array_set(_weights, 371, _w_input_blocks_8_1_proj_in_bias)
    _w_input_blocks_8_1_proj_in_weight: ptr = unet_view_2d(_base, 472117120, 1280, 1280)
    ptr_array_set(_weights, 372, _w_input_blocks_8_1_proj_in_weight)
    _w_input_blocks_8_1_proj_out_bias: ptr = unet_view_1d(_base, 473755520, 1280)
    ptr_array_set(_weights, 373, _w_input_blocks_8_1_proj_out_bias)
    _w_input_blocks_8_1_proj_out_weight: ptr = unet_view_2d(_base, 473756800, 1280, 1280)
    ptr_array_set(_weights, 374, _w_input_blocks_8_1_proj_out_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 475395200, 1280, 1280)
    ptr_array_set(_weights, 375, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 477033600, 1280)
    ptr_array_set(_weights, 376, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 477034880, 1280, 1280)
    ptr_array_set(_weights, 377, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 478673280, 1280, 1280)
    ptr_array_set(_weights, 378, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 480311680, 1280, 1280)
    ptr_array_set(_weights, 379, _w_input_blocks_8_1_transformer_blocks_0_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 481950080, 1280, 2048)
    ptr_array_set(_weights, 380, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 484571520, 1280)
    ptr_array_set(_weights, 381, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 484572800, 1280, 1280)
    ptr_array_set(_weights, 382, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 486211200, 1280, 1280)
    ptr_array_set(_weights, 383, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 487849600, 1280, 2048)
    ptr_array_set(_weights, 384, _w_input_blocks_8_1_transformer_blocks_0_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 490471040, 10240)
    ptr_array_set(_weights, 385, _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 490481280, 10240, 1280)
    ptr_array_set(_weights, 386, _w_input_blocks_8_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 503588480, 1280)
    ptr_array_set(_weights, 387, _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 503589760, 1280, 5120)
    ptr_array_set(_weights, 388, _w_input_blocks_8_1_transformer_blocks_0_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 510143360, 1280)
    ptr_array_set(_weights, 389, _w_input_blocks_8_1_transformer_blocks_0_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 510144640, 1280)
    ptr_array_set(_weights, 390, _w_input_blocks_8_1_transformer_blocks_0_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 510145920, 1280)
    ptr_array_set(_weights, 391, _w_input_blocks_8_1_transformer_blocks_0_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 510147200, 1280)
    ptr_array_set(_weights, 392, _w_input_blocks_8_1_transformer_blocks_0_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 510148480, 1280)
    ptr_array_set(_weights, 393, _w_input_blocks_8_1_transformer_blocks_0_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 510149760, 1280)
    ptr_array_set(_weights, 394, _w_input_blocks_8_1_transformer_blocks_0_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 510151040, 1280, 1280)
    ptr_array_set(_weights, 395, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 511789440, 1280)
    ptr_array_set(_weights, 396, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 511790720, 1280, 1280)
    ptr_array_set(_weights, 397, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 513429120, 1280, 1280)
    ptr_array_set(_weights, 398, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 515067520, 1280, 1280)
    ptr_array_set(_weights, 399, _w_input_blocks_8_1_transformer_blocks_1_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 516705920, 1280, 2048)
    ptr_array_set(_weights, 400, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 519327360, 1280)
    ptr_array_set(_weights, 401, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 519328640, 1280, 1280)
    ptr_array_set(_weights, 402, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 520967040, 1280, 1280)
    ptr_array_set(_weights, 403, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 522605440, 1280, 2048)
    ptr_array_set(_weights, 404, _w_input_blocks_8_1_transformer_blocks_1_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 525226880, 10240)
    ptr_array_set(_weights, 405, _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 525237120, 10240, 1280)
    ptr_array_set(_weights, 406, _w_input_blocks_8_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 538344320, 1280)
    ptr_array_set(_weights, 407, _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 538345600, 1280, 5120)
    ptr_array_set(_weights, 408, _w_input_blocks_8_1_transformer_blocks_1_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 544899200, 1280)
    ptr_array_set(_weights, 409, _w_input_blocks_8_1_transformer_blocks_1_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 544900480, 1280)
    ptr_array_set(_weights, 410, _w_input_blocks_8_1_transformer_blocks_1_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 544901760, 1280)
    ptr_array_set(_weights, 411, _w_input_blocks_8_1_transformer_blocks_1_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 544903040, 1280)
    ptr_array_set(_weights, 412, _w_input_blocks_8_1_transformer_blocks_1_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 544904320, 1280)
    ptr_array_set(_weights, 413, _w_input_blocks_8_1_transformer_blocks_1_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 544905600, 1280)
    ptr_array_set(_weights, 414, _w_input_blocks_8_1_transformer_blocks_1_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 544906880, 1280, 1280)
    ptr_array_set(_weights, 415, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 546545280, 1280)
    ptr_array_set(_weights, 416, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 546546560, 1280, 1280)
    ptr_array_set(_weights, 417, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 548184960, 1280, 1280)
    ptr_array_set(_weights, 418, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 549823360, 1280, 1280)
    ptr_array_set(_weights, 419, _w_input_blocks_8_1_transformer_blocks_2_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 551461760, 1280, 2048)
    ptr_array_set(_weights, 420, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 554083200, 1280)
    ptr_array_set(_weights, 421, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 554084480, 1280, 1280)
    ptr_array_set(_weights, 422, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 555722880, 1280, 1280)
    ptr_array_set(_weights, 423, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 557361280, 1280, 2048)
    ptr_array_set(_weights, 424, _w_input_blocks_8_1_transformer_blocks_2_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 559982720, 10240)
    ptr_array_set(_weights, 425, _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 559992960, 10240, 1280)
    ptr_array_set(_weights, 426, _w_input_blocks_8_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 573100160, 1280)
    ptr_array_set(_weights, 427, _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 573101440, 1280, 5120)
    ptr_array_set(_weights, 428, _w_input_blocks_8_1_transformer_blocks_2_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 579655040, 1280)
    ptr_array_set(_weights, 429, _w_input_blocks_8_1_transformer_blocks_2_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 579656320, 1280)
    ptr_array_set(_weights, 430, _w_input_blocks_8_1_transformer_blocks_2_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 579657600, 1280)
    ptr_array_set(_weights, 431, _w_input_blocks_8_1_transformer_blocks_2_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 579658880, 1280)
    ptr_array_set(_weights, 432, _w_input_blocks_8_1_transformer_blocks_2_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 579660160, 1280)
    ptr_array_set(_weights, 433, _w_input_blocks_8_1_transformer_blocks_2_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 579661440, 1280)
    ptr_array_set(_weights, 434, _w_input_blocks_8_1_transformer_blocks_2_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 579662720, 1280, 1280)
    ptr_array_set(_weights, 435, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 581301120, 1280)
    ptr_array_set(_weights, 436, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 581302400, 1280, 1280)
    ptr_array_set(_weights, 437, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 582940800, 1280, 1280)
    ptr_array_set(_weights, 438, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 584579200, 1280, 1280)
    ptr_array_set(_weights, 439, _w_input_blocks_8_1_transformer_blocks_3_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 586217600, 1280, 2048)
    ptr_array_set(_weights, 440, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 588839040, 1280)
    ptr_array_set(_weights, 441, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 588840320, 1280, 1280)
    ptr_array_set(_weights, 442, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 590478720, 1280, 1280)
    ptr_array_set(_weights, 443, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 592117120, 1280, 2048)
    ptr_array_set(_weights, 444, _w_input_blocks_8_1_transformer_blocks_3_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 594738560, 10240)
    ptr_array_set(_weights, 445, _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 594748800, 10240, 1280)
    ptr_array_set(_weights, 446, _w_input_blocks_8_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 607856000, 1280)
    ptr_array_set(_weights, 447, _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 607857280, 1280, 5120)
    ptr_array_set(_weights, 448, _w_input_blocks_8_1_transformer_blocks_3_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 614410880, 1280)
    ptr_array_set(_weights, 449, _w_input_blocks_8_1_transformer_blocks_3_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 614412160, 1280)
    ptr_array_set(_weights, 450, _w_input_blocks_8_1_transformer_blocks_3_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 614413440, 1280)
    ptr_array_set(_weights, 451, _w_input_blocks_8_1_transformer_blocks_3_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 614414720, 1280)
    ptr_array_set(_weights, 452, _w_input_blocks_8_1_transformer_blocks_3_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 614416000, 1280)
    ptr_array_set(_weights, 453, _w_input_blocks_8_1_transformer_blocks_3_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 614417280, 1280)
    ptr_array_set(_weights, 454, _w_input_blocks_8_1_transformer_blocks_3_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 614418560, 1280, 1280)
    ptr_array_set(_weights, 455, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 616056960, 1280)
    ptr_array_set(_weights, 456, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 616058240, 1280, 1280)
    ptr_array_set(_weights, 457, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 617696640, 1280, 1280)
    ptr_array_set(_weights, 458, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 619335040, 1280, 1280)
    ptr_array_set(_weights, 459, _w_input_blocks_8_1_transformer_blocks_4_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 620973440, 1280, 2048)
    ptr_array_set(_weights, 460, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 623594880, 1280)
    ptr_array_set(_weights, 461, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 623596160, 1280, 1280)
    ptr_array_set(_weights, 462, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 625234560, 1280, 1280)
    ptr_array_set(_weights, 463, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 626872960, 1280, 2048)
    ptr_array_set(_weights, 464, _w_input_blocks_8_1_transformer_blocks_4_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 629494400, 10240)
    ptr_array_set(_weights, 465, _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 629504640, 10240, 1280)
    ptr_array_set(_weights, 466, _w_input_blocks_8_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 642611840, 1280)
    ptr_array_set(_weights, 467, _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 642613120, 1280, 5120)
    ptr_array_set(_weights, 468, _w_input_blocks_8_1_transformer_blocks_4_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 649166720, 1280)
    ptr_array_set(_weights, 469, _w_input_blocks_8_1_transformer_blocks_4_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 649168000, 1280)
    ptr_array_set(_weights, 470, _w_input_blocks_8_1_transformer_blocks_4_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 649169280, 1280)
    ptr_array_set(_weights, 471, _w_input_blocks_8_1_transformer_blocks_4_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 649170560, 1280)
    ptr_array_set(_weights, 472, _w_input_blocks_8_1_transformer_blocks_4_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 649171840, 1280)
    ptr_array_set(_weights, 473, _w_input_blocks_8_1_transformer_blocks_4_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 649173120, 1280)
    ptr_array_set(_weights, 474, _w_input_blocks_8_1_transformer_blocks_4_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 649174400, 1280, 1280)
    ptr_array_set(_weights, 475, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 650812800, 1280)
    ptr_array_set(_weights, 476, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 650814080, 1280, 1280)
    ptr_array_set(_weights, 477, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 652452480, 1280, 1280)
    ptr_array_set(_weights, 478, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 654090880, 1280, 1280)
    ptr_array_set(_weights, 479, _w_input_blocks_8_1_transformer_blocks_5_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 655729280, 1280, 2048)
    ptr_array_set(_weights, 480, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 658350720, 1280)
    ptr_array_set(_weights, 481, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 658352000, 1280, 1280)
    ptr_array_set(_weights, 482, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 659990400, 1280, 1280)
    ptr_array_set(_weights, 483, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 661628800, 1280, 2048)
    ptr_array_set(_weights, 484, _w_input_blocks_8_1_transformer_blocks_5_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 664250240, 10240)
    ptr_array_set(_weights, 485, _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 664260480, 10240, 1280)
    ptr_array_set(_weights, 486, _w_input_blocks_8_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 677367680, 1280)
    ptr_array_set(_weights, 487, _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 677368960, 1280, 5120)
    ptr_array_set(_weights, 488, _w_input_blocks_8_1_transformer_blocks_5_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 683922560, 1280)
    ptr_array_set(_weights, 489, _w_input_blocks_8_1_transformer_blocks_5_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 683923840, 1280)
    ptr_array_set(_weights, 490, _w_input_blocks_8_1_transformer_blocks_5_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 683925120, 1280)
    ptr_array_set(_weights, 491, _w_input_blocks_8_1_transformer_blocks_5_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 683926400, 1280)
    ptr_array_set(_weights, 492, _w_input_blocks_8_1_transformer_blocks_5_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 683927680, 1280)
    ptr_array_set(_weights, 493, _w_input_blocks_8_1_transformer_blocks_5_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 683928960, 1280)
    ptr_array_set(_weights, 494, _w_input_blocks_8_1_transformer_blocks_5_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 683930240, 1280, 1280)
    ptr_array_set(_weights, 495, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 685568640, 1280)
    ptr_array_set(_weights, 496, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 685569920, 1280, 1280)
    ptr_array_set(_weights, 497, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 687208320, 1280, 1280)
    ptr_array_set(_weights, 498, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 688846720, 1280, 1280)
    ptr_array_set(_weights, 499, _w_input_blocks_8_1_transformer_blocks_6_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 690485120, 1280, 2048)
    ptr_array_set(_weights, 500, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 693106560, 1280)
    ptr_array_set(_weights, 501, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 693107840, 1280, 1280)
    ptr_array_set(_weights, 502, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 694746240, 1280, 1280)
    ptr_array_set(_weights, 503, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 696384640, 1280, 2048)
    ptr_array_set(_weights, 504, _w_input_blocks_8_1_transformer_blocks_6_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 699006080, 10240)
    ptr_array_set(_weights, 505, _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 699016320, 10240, 1280)
    ptr_array_set(_weights, 506, _w_input_blocks_8_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 712123520, 1280)
    ptr_array_set(_weights, 507, _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 712124800, 1280, 5120)
    ptr_array_set(_weights, 508, _w_input_blocks_8_1_transformer_blocks_6_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 718678400, 1280)
    ptr_array_set(_weights, 509, _w_input_blocks_8_1_transformer_blocks_6_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 718679680, 1280)
    ptr_array_set(_weights, 510, _w_input_blocks_8_1_transformer_blocks_6_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 718680960, 1280)
    ptr_array_set(_weights, 511, _w_input_blocks_8_1_transformer_blocks_6_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 718682240, 1280)
    ptr_array_set(_weights, 512, _w_input_blocks_8_1_transformer_blocks_6_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 718683520, 1280)
    ptr_array_set(_weights, 513, _w_input_blocks_8_1_transformer_blocks_6_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 718684800, 1280)
    ptr_array_set(_weights, 514, _w_input_blocks_8_1_transformer_blocks_6_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 718686080, 1280, 1280)
    ptr_array_set(_weights, 515, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 720324480, 1280)
    ptr_array_set(_weights, 516, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 720325760, 1280, 1280)
    ptr_array_set(_weights, 517, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 721964160, 1280, 1280)
    ptr_array_set(_weights, 518, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 723602560, 1280, 1280)
    ptr_array_set(_weights, 519, _w_input_blocks_8_1_transformer_blocks_7_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 725240960, 1280, 2048)
    ptr_array_set(_weights, 520, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 727862400, 1280)
    ptr_array_set(_weights, 521, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 727863680, 1280, 1280)
    ptr_array_set(_weights, 522, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 729502080, 1280, 1280)
    ptr_array_set(_weights, 523, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 731140480, 1280, 2048)
    ptr_array_set(_weights, 524, _w_input_blocks_8_1_transformer_blocks_7_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 733761920, 10240)
    ptr_array_set(_weights, 525, _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 733772160, 10240, 1280)
    ptr_array_set(_weights, 526, _w_input_blocks_8_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 746879360, 1280)
    ptr_array_set(_weights, 527, _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 746880640, 1280, 5120)
    ptr_array_set(_weights, 528, _w_input_blocks_8_1_transformer_blocks_7_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 753434240, 1280)
    ptr_array_set(_weights, 529, _w_input_blocks_8_1_transformer_blocks_7_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 753435520, 1280)
    ptr_array_set(_weights, 530, _w_input_blocks_8_1_transformer_blocks_7_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 753436800, 1280)
    ptr_array_set(_weights, 531, _w_input_blocks_8_1_transformer_blocks_7_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 753438080, 1280)
    ptr_array_set(_weights, 532, _w_input_blocks_8_1_transformer_blocks_7_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 753439360, 1280)
    ptr_array_set(_weights, 533, _w_input_blocks_8_1_transformer_blocks_7_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 753440640, 1280)
    ptr_array_set(_weights, 534, _w_input_blocks_8_1_transformer_blocks_7_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 753441920, 1280, 1280)
    ptr_array_set(_weights, 535, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 755080320, 1280)
    ptr_array_set(_weights, 536, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 755081600, 1280, 1280)
    ptr_array_set(_weights, 537, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 756720000, 1280, 1280)
    ptr_array_set(_weights, 538, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 758358400, 1280, 1280)
    ptr_array_set(_weights, 539, _w_input_blocks_8_1_transformer_blocks_8_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 759996800, 1280, 2048)
    ptr_array_set(_weights, 540, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 762618240, 1280)
    ptr_array_set(_weights, 541, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 762619520, 1280, 1280)
    ptr_array_set(_weights, 542, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 764257920, 1280, 1280)
    ptr_array_set(_weights, 543, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 765896320, 1280, 2048)
    ptr_array_set(_weights, 544, _w_input_blocks_8_1_transformer_blocks_8_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 768517760, 10240)
    ptr_array_set(_weights, 545, _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 768528000, 10240, 1280)
    ptr_array_set(_weights, 546, _w_input_blocks_8_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 781635200, 1280)
    ptr_array_set(_weights, 547, _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 781636480, 1280, 5120)
    ptr_array_set(_weights, 548, _w_input_blocks_8_1_transformer_blocks_8_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 788190080, 1280)
    ptr_array_set(_weights, 549, _w_input_blocks_8_1_transformer_blocks_8_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 788191360, 1280)
    ptr_array_set(_weights, 550, _w_input_blocks_8_1_transformer_blocks_8_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 788192640, 1280)
    ptr_array_set(_weights, 551, _w_input_blocks_8_1_transformer_blocks_8_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 788193920, 1280)
    ptr_array_set(_weights, 552, _w_input_blocks_8_1_transformer_blocks_8_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 788195200, 1280)
    ptr_array_set(_weights, 553, _w_input_blocks_8_1_transformer_blocks_8_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 788196480, 1280)
    ptr_array_set(_weights, 554, _w_input_blocks_8_1_transformer_blocks_8_norm3_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 788197760, 1280, 1280)
    ptr_array_set(_weights, 555, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 789836160, 1280)
    ptr_array_set(_weights, 556, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 789837440, 1280, 1280)
    ptr_array_set(_weights, 557, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 791475840, 1280, 1280)
    ptr_array_set(_weights, 558, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 793114240, 1280, 1280)
    ptr_array_set(_weights, 559, _w_input_blocks_8_1_transformer_blocks_9_attn1_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 794752640, 1280, 2048)
    ptr_array_set(_weights, 560, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_k_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 797374080, 1280)
    ptr_array_set(_weights, 561, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 797375360, 1280, 1280)
    ptr_array_set(_weights, 562, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 799013760, 1280, 1280)
    ptr_array_set(_weights, 563, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_q_weight)
    _w_input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 800652160, 1280, 2048)
    ptr_array_set(_weights, 564, _w_input_blocks_8_1_transformer_blocks_9_attn2_to_v_weight)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 803273600, 10240)
    ptr_array_set(_weights, 565, _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 803283840, 10240, 1280)
    ptr_array_set(_weights, 566, _w_input_blocks_8_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 816391040, 1280)
    ptr_array_set(_weights, 567, _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_bias)
    _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 816392320, 1280, 5120)
    ptr_array_set(_weights, 568, _w_input_blocks_8_1_transformer_blocks_9_ff_net_2_weight)
    _w_input_blocks_8_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 822945920, 1280)
    ptr_array_set(_weights, 569, _w_input_blocks_8_1_transformer_blocks_9_norm1_bias)
    _w_input_blocks_8_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 822947200, 1280)
    ptr_array_set(_weights, 570, _w_input_blocks_8_1_transformer_blocks_9_norm1_weight)
    _w_input_blocks_8_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 822948480, 1280)
    ptr_array_set(_weights, 571, _w_input_blocks_8_1_transformer_blocks_9_norm2_bias)
    _w_input_blocks_8_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 822949760, 1280)
    ptr_array_set(_weights, 572, _w_input_blocks_8_1_transformer_blocks_9_norm2_weight)
    _w_input_blocks_8_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 822951040, 1280)
    ptr_array_set(_weights, 573, _w_input_blocks_8_1_transformer_blocks_9_norm3_bias)
    _w_input_blocks_8_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 822952320, 1280)
    ptr_array_set(_weights, 574, _w_input_blocks_8_1_transformer_blocks_9_norm3_weight)
    _w_label_emb_0_0_bias: ptr = unet_view_1d(_base, 822953600, 1280)
    ptr_array_set(_weights, 575, _w_label_emb_0_0_bias)
    _w_label_emb_0_0_weight: ptr = unet_view_2d(_base, 822954880, 1280, 2816)
    ptr_array_set(_weights, 576, _w_label_emb_0_0_weight)
    _w_label_emb_0_2_bias: ptr = unet_view_1d(_base, 826559360, 1280)
    ptr_array_set(_weights, 577, _w_label_emb_0_2_bias)
    _w_label_emb_0_2_weight: ptr = unet_view_2d(_base, 826560640, 1280, 1280)
    ptr_array_set(_weights, 578, _w_label_emb_0_2_weight)
    _w_middle_block_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 828199040, 1280)
    ptr_array_set(_weights, 579, _w_middle_block_0_emb_layers_1_bias)
    _w_middle_block_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 828200320, 1280, 1280)
    ptr_array_set(_weights, 580, _w_middle_block_0_emb_layers_1_weight)
    _w_middle_block_0_in_layers_0_bias: ptr = unet_view_1d(_base, 829838720, 1280)
    ptr_array_set(_weights, 581, _w_middle_block_0_in_layers_0_bias)
    _w_middle_block_0_in_layers_0_weight: ptr = unet_view_1d(_base, 829840000, 1280)
    ptr_array_set(_weights, 582, _w_middle_block_0_in_layers_0_weight)
    _w_middle_block_0_in_layers_2_bias: ptr = unet_view_1d(_base, 829841280, 1280)
    ptr_array_set(_weights, 583, _w_middle_block_0_in_layers_2_bias)
    _w_middle_block_0_in_layers_2_weight: ptr = unet_view_4d(_base, 829842560, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 584, _w_middle_block_0_in_layers_2_weight)
    _w_middle_block_0_out_layers_0_bias: ptr = unet_view_1d(_base, 844588160, 1280)
    ptr_array_set(_weights, 585, _w_middle_block_0_out_layers_0_bias)
    _w_middle_block_0_out_layers_0_weight: ptr = unet_view_1d(_base, 844589440, 1280)
    ptr_array_set(_weights, 586, _w_middle_block_0_out_layers_0_weight)
    _w_middle_block_0_out_layers_3_bias: ptr = unet_view_1d(_base, 844590720, 1280)
    ptr_array_set(_weights, 587, _w_middle_block_0_out_layers_3_bias)
    _w_middle_block_0_out_layers_3_weight: ptr = unet_view_4d(_base, 844592000, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 588, _w_middle_block_0_out_layers_3_weight)
    _w_middle_block_1_norm_bias: ptr = unet_view_1d(_base, 859337600, 1280)
    ptr_array_set(_weights, 589, _w_middle_block_1_norm_bias)
    _w_middle_block_1_norm_weight: ptr = unet_view_1d(_base, 859338880, 1280)
    ptr_array_set(_weights, 590, _w_middle_block_1_norm_weight)
    _w_middle_block_1_proj_in_bias: ptr = unet_view_1d(_base, 859340160, 1280)
    ptr_array_set(_weights, 591, _w_middle_block_1_proj_in_bias)
    _w_middle_block_1_proj_in_weight: ptr = unet_view_2d(_base, 859341440, 1280, 1280)
    ptr_array_set(_weights, 592, _w_middle_block_1_proj_in_weight)
    _w_middle_block_1_proj_out_bias: ptr = unet_view_1d(_base, 860979840, 1280)
    ptr_array_set(_weights, 593, _w_middle_block_1_proj_out_bias)
    _w_middle_block_1_proj_out_weight: ptr = unet_view_2d(_base, 860981120, 1280, 1280)
    ptr_array_set(_weights, 594, _w_middle_block_1_proj_out_weight)
    _w_middle_block_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 862619520, 1280, 1280)
    ptr_array_set(_weights, 595, _w_middle_block_1_transformer_blocks_0_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 864257920, 1280)
    ptr_array_set(_weights, 596, _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 864259200, 1280, 1280)
    ptr_array_set(_weights, 597, _w_middle_block_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 865897600, 1280, 1280)
    ptr_array_set(_weights, 598, _w_middle_block_1_transformer_blocks_0_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 867536000, 1280, 1280)
    ptr_array_set(_weights, 599, _w_middle_block_1_transformer_blocks_0_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 869174400, 1280, 2048)
    ptr_array_set(_weights, 600, _w_middle_block_1_transformer_blocks_0_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 871795840, 1280)
    ptr_array_set(_weights, 601, _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 871797120, 1280, 1280)
    ptr_array_set(_weights, 602, _w_middle_block_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 873435520, 1280, 1280)
    ptr_array_set(_weights, 603, _w_middle_block_1_transformer_blocks_0_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 875073920, 1280, 2048)
    ptr_array_set(_weights, 604, _w_middle_block_1_transformer_blocks_0_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 877695360, 10240)
    ptr_array_set(_weights, 605, _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 877705600, 10240, 1280)
    ptr_array_set(_weights, 606, _w_middle_block_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 890812800, 1280)
    ptr_array_set(_weights, 607, _w_middle_block_1_transformer_blocks_0_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 890814080, 1280, 5120)
    ptr_array_set(_weights, 608, _w_middle_block_1_transformer_blocks_0_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 897367680, 1280)
    ptr_array_set(_weights, 609, _w_middle_block_1_transformer_blocks_0_norm1_bias)
    _w_middle_block_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 897368960, 1280)
    ptr_array_set(_weights, 610, _w_middle_block_1_transformer_blocks_0_norm1_weight)
    _w_middle_block_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 897370240, 1280)
    ptr_array_set(_weights, 611, _w_middle_block_1_transformer_blocks_0_norm2_bias)
    _w_middle_block_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 897371520, 1280)
    ptr_array_set(_weights, 612, _w_middle_block_1_transformer_blocks_0_norm2_weight)
    _w_middle_block_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 897372800, 1280)
    ptr_array_set(_weights, 613, _w_middle_block_1_transformer_blocks_0_norm3_bias)
    _w_middle_block_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 897374080, 1280)
    ptr_array_set(_weights, 614, _w_middle_block_1_transformer_blocks_0_norm3_weight)
    _w_middle_block_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 897375360, 1280, 1280)
    ptr_array_set(_weights, 615, _w_middle_block_1_transformer_blocks_1_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 899013760, 1280)
    ptr_array_set(_weights, 616, _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 899015040, 1280, 1280)
    ptr_array_set(_weights, 617, _w_middle_block_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 900653440, 1280, 1280)
    ptr_array_set(_weights, 618, _w_middle_block_1_transformer_blocks_1_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 902291840, 1280, 1280)
    ptr_array_set(_weights, 619, _w_middle_block_1_transformer_blocks_1_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 903930240, 1280, 2048)
    ptr_array_set(_weights, 620, _w_middle_block_1_transformer_blocks_1_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 906551680, 1280)
    ptr_array_set(_weights, 621, _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 906552960, 1280, 1280)
    ptr_array_set(_weights, 622, _w_middle_block_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 908191360, 1280, 1280)
    ptr_array_set(_weights, 623, _w_middle_block_1_transformer_blocks_1_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 909829760, 1280, 2048)
    ptr_array_set(_weights, 624, _w_middle_block_1_transformer_blocks_1_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 912451200, 10240)
    ptr_array_set(_weights, 625, _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 912461440, 10240, 1280)
    ptr_array_set(_weights, 626, _w_middle_block_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 925568640, 1280)
    ptr_array_set(_weights, 627, _w_middle_block_1_transformer_blocks_1_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 925569920, 1280, 5120)
    ptr_array_set(_weights, 628, _w_middle_block_1_transformer_blocks_1_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 932123520, 1280)
    ptr_array_set(_weights, 629, _w_middle_block_1_transformer_blocks_1_norm1_bias)
    _w_middle_block_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 932124800, 1280)
    ptr_array_set(_weights, 630, _w_middle_block_1_transformer_blocks_1_norm1_weight)
    _w_middle_block_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 932126080, 1280)
    ptr_array_set(_weights, 631, _w_middle_block_1_transformer_blocks_1_norm2_bias)
    _w_middle_block_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 932127360, 1280)
    ptr_array_set(_weights, 632, _w_middle_block_1_transformer_blocks_1_norm2_weight)
    _w_middle_block_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 932128640, 1280)
    ptr_array_set(_weights, 633, _w_middle_block_1_transformer_blocks_1_norm3_bias)
    _w_middle_block_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 932129920, 1280)
    ptr_array_set(_weights, 634, _w_middle_block_1_transformer_blocks_1_norm3_weight)
    _w_middle_block_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 932131200, 1280, 1280)
    ptr_array_set(_weights, 635, _w_middle_block_1_transformer_blocks_2_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 933769600, 1280)
    ptr_array_set(_weights, 636, _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 933770880, 1280, 1280)
    ptr_array_set(_weights, 637, _w_middle_block_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 935409280, 1280, 1280)
    ptr_array_set(_weights, 638, _w_middle_block_1_transformer_blocks_2_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 937047680, 1280, 1280)
    ptr_array_set(_weights, 639, _w_middle_block_1_transformer_blocks_2_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 938686080, 1280, 2048)
    ptr_array_set(_weights, 640, _w_middle_block_1_transformer_blocks_2_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 941307520, 1280)
    ptr_array_set(_weights, 641, _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 941308800, 1280, 1280)
    ptr_array_set(_weights, 642, _w_middle_block_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 942947200, 1280, 1280)
    ptr_array_set(_weights, 643, _w_middle_block_1_transformer_blocks_2_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 944585600, 1280, 2048)
    ptr_array_set(_weights, 644, _w_middle_block_1_transformer_blocks_2_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 947207040, 10240)
    ptr_array_set(_weights, 645, _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 947217280, 10240, 1280)
    ptr_array_set(_weights, 646, _w_middle_block_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 960324480, 1280)
    ptr_array_set(_weights, 647, _w_middle_block_1_transformer_blocks_2_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 960325760, 1280, 5120)
    ptr_array_set(_weights, 648, _w_middle_block_1_transformer_blocks_2_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 966879360, 1280)
    ptr_array_set(_weights, 649, _w_middle_block_1_transformer_blocks_2_norm1_bias)
    _w_middle_block_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 966880640, 1280)
    ptr_array_set(_weights, 650, _w_middle_block_1_transformer_blocks_2_norm1_weight)
    _w_middle_block_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 966881920, 1280)
    ptr_array_set(_weights, 651, _w_middle_block_1_transformer_blocks_2_norm2_bias)
    _w_middle_block_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 966883200, 1280)
    ptr_array_set(_weights, 652, _w_middle_block_1_transformer_blocks_2_norm2_weight)
    _w_middle_block_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 966884480, 1280)
    ptr_array_set(_weights, 653, _w_middle_block_1_transformer_blocks_2_norm3_bias)
    _w_middle_block_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 966885760, 1280)
    ptr_array_set(_weights, 654, _w_middle_block_1_transformer_blocks_2_norm3_weight)
    _w_middle_block_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 966887040, 1280, 1280)
    ptr_array_set(_weights, 655, _w_middle_block_1_transformer_blocks_3_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 968525440, 1280)
    ptr_array_set(_weights, 656, _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 968526720, 1280, 1280)
    ptr_array_set(_weights, 657, _w_middle_block_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 970165120, 1280, 1280)
    ptr_array_set(_weights, 658, _w_middle_block_1_transformer_blocks_3_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 971803520, 1280, 1280)
    ptr_array_set(_weights, 659, _w_middle_block_1_transformer_blocks_3_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 973441920, 1280, 2048)
    ptr_array_set(_weights, 660, _w_middle_block_1_transformer_blocks_3_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 976063360, 1280)
    ptr_array_set(_weights, 661, _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 976064640, 1280, 1280)
    ptr_array_set(_weights, 662, _w_middle_block_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 977703040, 1280, 1280)
    ptr_array_set(_weights, 663, _w_middle_block_1_transformer_blocks_3_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 979341440, 1280, 2048)
    ptr_array_set(_weights, 664, _w_middle_block_1_transformer_blocks_3_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 981962880, 10240)
    ptr_array_set(_weights, 665, _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 981973120, 10240, 1280)
    ptr_array_set(_weights, 666, _w_middle_block_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 995080320, 1280)
    ptr_array_set(_weights, 667, _w_middle_block_1_transformer_blocks_3_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 995081600, 1280, 5120)
    ptr_array_set(_weights, 668, _w_middle_block_1_transformer_blocks_3_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 1001635200, 1280)
    ptr_array_set(_weights, 669, _w_middle_block_1_transformer_blocks_3_norm1_bias)
    _w_middle_block_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 1001636480, 1280)
    ptr_array_set(_weights, 670, _w_middle_block_1_transformer_blocks_3_norm1_weight)
    _w_middle_block_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 1001637760, 1280)
    ptr_array_set(_weights, 671, _w_middle_block_1_transformer_blocks_3_norm2_bias)
    _w_middle_block_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 1001639040, 1280)
    ptr_array_set(_weights, 672, _w_middle_block_1_transformer_blocks_3_norm2_weight)
    _w_middle_block_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 1001640320, 1280)
    ptr_array_set(_weights, 673, _w_middle_block_1_transformer_blocks_3_norm3_bias)
    _w_middle_block_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 1001641600, 1280)
    ptr_array_set(_weights, 674, _w_middle_block_1_transformer_blocks_3_norm3_weight)
    _w_middle_block_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 1001642880, 1280, 1280)
    ptr_array_set(_weights, 675, _w_middle_block_1_transformer_blocks_4_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1003281280, 1280)
    ptr_array_set(_weights, 676, _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1003282560, 1280, 1280)
    ptr_array_set(_weights, 677, _w_middle_block_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 1004920960, 1280, 1280)
    ptr_array_set(_weights, 678, _w_middle_block_1_transformer_blocks_4_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 1006559360, 1280, 1280)
    ptr_array_set(_weights, 679, _w_middle_block_1_transformer_blocks_4_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 1008197760, 1280, 2048)
    ptr_array_set(_weights, 680, _w_middle_block_1_transformer_blocks_4_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1010819200, 1280)
    ptr_array_set(_weights, 681, _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1010820480, 1280, 1280)
    ptr_array_set(_weights, 682, _w_middle_block_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 1012458880, 1280, 1280)
    ptr_array_set(_weights, 683, _w_middle_block_1_transformer_blocks_4_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 1014097280, 1280, 2048)
    ptr_array_set(_weights, 684, _w_middle_block_1_transformer_blocks_4_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1016718720, 10240)
    ptr_array_set(_weights, 685, _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1016728960, 10240, 1280)
    ptr_array_set(_weights, 686, _w_middle_block_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 1029836160, 1280)
    ptr_array_set(_weights, 687, _w_middle_block_1_transformer_blocks_4_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 1029837440, 1280, 5120)
    ptr_array_set(_weights, 688, _w_middle_block_1_transformer_blocks_4_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 1036391040, 1280)
    ptr_array_set(_weights, 689, _w_middle_block_1_transformer_blocks_4_norm1_bias)
    _w_middle_block_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 1036392320, 1280)
    ptr_array_set(_weights, 690, _w_middle_block_1_transformer_blocks_4_norm1_weight)
    _w_middle_block_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 1036393600, 1280)
    ptr_array_set(_weights, 691, _w_middle_block_1_transformer_blocks_4_norm2_bias)
    _w_middle_block_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 1036394880, 1280)
    ptr_array_set(_weights, 692, _w_middle_block_1_transformer_blocks_4_norm2_weight)
    _w_middle_block_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 1036396160, 1280)
    ptr_array_set(_weights, 693, _w_middle_block_1_transformer_blocks_4_norm3_bias)
    _w_middle_block_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 1036397440, 1280)
    ptr_array_set(_weights, 694, _w_middle_block_1_transformer_blocks_4_norm3_weight)
    _w_middle_block_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 1036398720, 1280, 1280)
    ptr_array_set(_weights, 695, _w_middle_block_1_transformer_blocks_5_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1038037120, 1280)
    ptr_array_set(_weights, 696, _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1038038400, 1280, 1280)
    ptr_array_set(_weights, 697, _w_middle_block_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 1039676800, 1280, 1280)
    ptr_array_set(_weights, 698, _w_middle_block_1_transformer_blocks_5_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 1041315200, 1280, 1280)
    ptr_array_set(_weights, 699, _w_middle_block_1_transformer_blocks_5_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 1042953600, 1280, 2048)
    ptr_array_set(_weights, 700, _w_middle_block_1_transformer_blocks_5_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1045575040, 1280)
    ptr_array_set(_weights, 701, _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1045576320, 1280, 1280)
    ptr_array_set(_weights, 702, _w_middle_block_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 1047214720, 1280, 1280)
    ptr_array_set(_weights, 703, _w_middle_block_1_transformer_blocks_5_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 1048853120, 1280, 2048)
    ptr_array_set(_weights, 704, _w_middle_block_1_transformer_blocks_5_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1051474560, 10240)
    ptr_array_set(_weights, 705, _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1051484800, 10240, 1280)
    ptr_array_set(_weights, 706, _w_middle_block_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 1064592000, 1280)
    ptr_array_set(_weights, 707, _w_middle_block_1_transformer_blocks_5_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 1064593280, 1280, 5120)
    ptr_array_set(_weights, 708, _w_middle_block_1_transformer_blocks_5_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 1071146880, 1280)
    ptr_array_set(_weights, 709, _w_middle_block_1_transformer_blocks_5_norm1_bias)
    _w_middle_block_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 1071148160, 1280)
    ptr_array_set(_weights, 710, _w_middle_block_1_transformer_blocks_5_norm1_weight)
    _w_middle_block_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 1071149440, 1280)
    ptr_array_set(_weights, 711, _w_middle_block_1_transformer_blocks_5_norm2_bias)
    _w_middle_block_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 1071150720, 1280)
    ptr_array_set(_weights, 712, _w_middle_block_1_transformer_blocks_5_norm2_weight)
    _w_middle_block_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 1071152000, 1280)
    ptr_array_set(_weights, 713, _w_middle_block_1_transformer_blocks_5_norm3_bias)
    _w_middle_block_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 1071153280, 1280)
    ptr_array_set(_weights, 714, _w_middle_block_1_transformer_blocks_5_norm3_weight)
    _w_middle_block_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 1071154560, 1280, 1280)
    ptr_array_set(_weights, 715, _w_middle_block_1_transformer_blocks_6_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1072792960, 1280)
    ptr_array_set(_weights, 716, _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1072794240, 1280, 1280)
    ptr_array_set(_weights, 717, _w_middle_block_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 1074432640, 1280, 1280)
    ptr_array_set(_weights, 718, _w_middle_block_1_transformer_blocks_6_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 1076071040, 1280, 1280)
    ptr_array_set(_weights, 719, _w_middle_block_1_transformer_blocks_6_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 1077709440, 1280, 2048)
    ptr_array_set(_weights, 720, _w_middle_block_1_transformer_blocks_6_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1080330880, 1280)
    ptr_array_set(_weights, 721, _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1080332160, 1280, 1280)
    ptr_array_set(_weights, 722, _w_middle_block_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 1081970560, 1280, 1280)
    ptr_array_set(_weights, 723, _w_middle_block_1_transformer_blocks_6_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 1083608960, 1280, 2048)
    ptr_array_set(_weights, 724, _w_middle_block_1_transformer_blocks_6_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1086230400, 10240)
    ptr_array_set(_weights, 725, _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1086240640, 10240, 1280)
    ptr_array_set(_weights, 726, _w_middle_block_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 1099347840, 1280)
    ptr_array_set(_weights, 727, _w_middle_block_1_transformer_blocks_6_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 1099349120, 1280, 5120)
    ptr_array_set(_weights, 728, _w_middle_block_1_transformer_blocks_6_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 1105902720, 1280)
    ptr_array_set(_weights, 729, _w_middle_block_1_transformer_blocks_6_norm1_bias)
    _w_middle_block_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 1105904000, 1280)
    ptr_array_set(_weights, 730, _w_middle_block_1_transformer_blocks_6_norm1_weight)
    _w_middle_block_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 1105905280, 1280)
    ptr_array_set(_weights, 731, _w_middle_block_1_transformer_blocks_6_norm2_bias)
    _w_middle_block_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 1105906560, 1280)
    ptr_array_set(_weights, 732, _w_middle_block_1_transformer_blocks_6_norm2_weight)
    _w_middle_block_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 1105907840, 1280)
    ptr_array_set(_weights, 733, _w_middle_block_1_transformer_blocks_6_norm3_bias)
    _w_middle_block_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 1105909120, 1280)
    ptr_array_set(_weights, 734, _w_middle_block_1_transformer_blocks_6_norm3_weight)
    _w_middle_block_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 1105910400, 1280, 1280)
    ptr_array_set(_weights, 735, _w_middle_block_1_transformer_blocks_7_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1107548800, 1280)
    ptr_array_set(_weights, 736, _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1107550080, 1280, 1280)
    ptr_array_set(_weights, 737, _w_middle_block_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 1109188480, 1280, 1280)
    ptr_array_set(_weights, 738, _w_middle_block_1_transformer_blocks_7_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 1110826880, 1280, 1280)
    ptr_array_set(_weights, 739, _w_middle_block_1_transformer_blocks_7_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 1112465280, 1280, 2048)
    ptr_array_set(_weights, 740, _w_middle_block_1_transformer_blocks_7_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1115086720, 1280)
    ptr_array_set(_weights, 741, _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1115088000, 1280, 1280)
    ptr_array_set(_weights, 742, _w_middle_block_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 1116726400, 1280, 1280)
    ptr_array_set(_weights, 743, _w_middle_block_1_transformer_blocks_7_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 1118364800, 1280, 2048)
    ptr_array_set(_weights, 744, _w_middle_block_1_transformer_blocks_7_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1120986240, 10240)
    ptr_array_set(_weights, 745, _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1120996480, 10240, 1280)
    ptr_array_set(_weights, 746, _w_middle_block_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 1134103680, 1280)
    ptr_array_set(_weights, 747, _w_middle_block_1_transformer_blocks_7_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 1134104960, 1280, 5120)
    ptr_array_set(_weights, 748, _w_middle_block_1_transformer_blocks_7_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 1140658560, 1280)
    ptr_array_set(_weights, 749, _w_middle_block_1_transformer_blocks_7_norm1_bias)
    _w_middle_block_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 1140659840, 1280)
    ptr_array_set(_weights, 750, _w_middle_block_1_transformer_blocks_7_norm1_weight)
    _w_middle_block_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 1140661120, 1280)
    ptr_array_set(_weights, 751, _w_middle_block_1_transformer_blocks_7_norm2_bias)
    _w_middle_block_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 1140662400, 1280)
    ptr_array_set(_weights, 752, _w_middle_block_1_transformer_blocks_7_norm2_weight)
    _w_middle_block_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 1140663680, 1280)
    ptr_array_set(_weights, 753, _w_middle_block_1_transformer_blocks_7_norm3_bias)
    _w_middle_block_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 1140664960, 1280)
    ptr_array_set(_weights, 754, _w_middle_block_1_transformer_blocks_7_norm3_weight)
    _w_middle_block_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 1140666240, 1280, 1280)
    ptr_array_set(_weights, 755, _w_middle_block_1_transformer_blocks_8_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1142304640, 1280)
    ptr_array_set(_weights, 756, _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1142305920, 1280, 1280)
    ptr_array_set(_weights, 757, _w_middle_block_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 1143944320, 1280, 1280)
    ptr_array_set(_weights, 758, _w_middle_block_1_transformer_blocks_8_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 1145582720, 1280, 1280)
    ptr_array_set(_weights, 759, _w_middle_block_1_transformer_blocks_8_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 1147221120, 1280, 2048)
    ptr_array_set(_weights, 760, _w_middle_block_1_transformer_blocks_8_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1149842560, 1280)
    ptr_array_set(_weights, 761, _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1149843840, 1280, 1280)
    ptr_array_set(_weights, 762, _w_middle_block_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 1151482240, 1280, 1280)
    ptr_array_set(_weights, 763, _w_middle_block_1_transformer_blocks_8_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 1153120640, 1280, 2048)
    ptr_array_set(_weights, 764, _w_middle_block_1_transformer_blocks_8_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1155742080, 10240)
    ptr_array_set(_weights, 765, _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1155752320, 10240, 1280)
    ptr_array_set(_weights, 766, _w_middle_block_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 1168859520, 1280)
    ptr_array_set(_weights, 767, _w_middle_block_1_transformer_blocks_8_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 1168860800, 1280, 5120)
    ptr_array_set(_weights, 768, _w_middle_block_1_transformer_blocks_8_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 1175414400, 1280)
    ptr_array_set(_weights, 769, _w_middle_block_1_transformer_blocks_8_norm1_bias)
    _w_middle_block_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 1175415680, 1280)
    ptr_array_set(_weights, 770, _w_middle_block_1_transformer_blocks_8_norm1_weight)
    _w_middle_block_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 1175416960, 1280)
    ptr_array_set(_weights, 771, _w_middle_block_1_transformer_blocks_8_norm2_bias)
    _w_middle_block_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 1175418240, 1280)
    ptr_array_set(_weights, 772, _w_middle_block_1_transformer_blocks_8_norm2_weight)
    _w_middle_block_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 1175419520, 1280)
    ptr_array_set(_weights, 773, _w_middle_block_1_transformer_blocks_8_norm3_bias)
    _w_middle_block_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 1175420800, 1280)
    ptr_array_set(_weights, 774, _w_middle_block_1_transformer_blocks_8_norm3_weight)
    _w_middle_block_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 1175422080, 1280, 1280)
    ptr_array_set(_weights, 775, _w_middle_block_1_transformer_blocks_9_attn1_to_k_weight)
    _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1177060480, 1280)
    ptr_array_set(_weights, 776, _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1177061760, 1280, 1280)
    ptr_array_set(_weights, 777, _w_middle_block_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 1178700160, 1280, 1280)
    ptr_array_set(_weights, 778, _w_middle_block_1_transformer_blocks_9_attn1_to_q_weight)
    _w_middle_block_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 1180338560, 1280, 1280)
    ptr_array_set(_weights, 779, _w_middle_block_1_transformer_blocks_9_attn1_to_v_weight)
    _w_middle_block_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 1181976960, 1280, 2048)
    ptr_array_set(_weights, 780, _w_middle_block_1_transformer_blocks_9_attn2_to_k_weight)
    _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1184598400, 1280)
    ptr_array_set(_weights, 781, _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1184599680, 1280, 1280)
    ptr_array_set(_weights, 782, _w_middle_block_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_middle_block_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 1186238080, 1280, 1280)
    ptr_array_set(_weights, 783, _w_middle_block_1_transformer_blocks_9_attn2_to_q_weight)
    _w_middle_block_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 1187876480, 1280, 2048)
    ptr_array_set(_weights, 784, _w_middle_block_1_transformer_blocks_9_attn2_to_v_weight)
    _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1190497920, 10240)
    ptr_array_set(_weights, 785, _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1190508160, 10240, 1280)
    ptr_array_set(_weights, 786, _w_middle_block_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_middle_block_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 1203615360, 1280)
    ptr_array_set(_weights, 787, _w_middle_block_1_transformer_blocks_9_ff_net_2_bias)
    _w_middle_block_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 1203616640, 1280, 5120)
    ptr_array_set(_weights, 788, _w_middle_block_1_transformer_blocks_9_ff_net_2_weight)
    _w_middle_block_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 1210170240, 1280)
    ptr_array_set(_weights, 789, _w_middle_block_1_transformer_blocks_9_norm1_bias)
    _w_middle_block_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 1210171520, 1280)
    ptr_array_set(_weights, 790, _w_middle_block_1_transformer_blocks_9_norm1_weight)
    _w_middle_block_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 1210172800, 1280)
    ptr_array_set(_weights, 791, _w_middle_block_1_transformer_blocks_9_norm2_bias)
    _w_middle_block_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 1210174080, 1280)
    ptr_array_set(_weights, 792, _w_middle_block_1_transformer_blocks_9_norm2_weight)
    _w_middle_block_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 1210175360, 1280)
    ptr_array_set(_weights, 793, _w_middle_block_1_transformer_blocks_9_norm3_bias)
    _w_middle_block_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 1210176640, 1280)
    ptr_array_set(_weights, 794, _w_middle_block_1_transformer_blocks_9_norm3_weight)
    _w_middle_block_2_emb_layers_1_bias: ptr = unet_view_1d(_base, 1210177920, 1280)
    ptr_array_set(_weights, 795, _w_middle_block_2_emb_layers_1_bias)
    _w_middle_block_2_emb_layers_1_weight: ptr = unet_view_2d(_base, 1210179200, 1280, 1280)
    ptr_array_set(_weights, 796, _w_middle_block_2_emb_layers_1_weight)
    _w_middle_block_2_in_layers_0_bias: ptr = unet_view_1d(_base, 1211817600, 1280)
    ptr_array_set(_weights, 797, _w_middle_block_2_in_layers_0_bias)
    _w_middle_block_2_in_layers_0_weight: ptr = unet_view_1d(_base, 1211818880, 1280)
    ptr_array_set(_weights, 798, _w_middle_block_2_in_layers_0_weight)
    _w_middle_block_2_in_layers_2_bias: ptr = unet_view_1d(_base, 1211820160, 1280)
    ptr_array_set(_weights, 799, _w_middle_block_2_in_layers_2_bias)
    _w_middle_block_2_in_layers_2_weight: ptr = unet_view_4d(_base, 1211821440, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 800, _w_middle_block_2_in_layers_2_weight)
    _w_middle_block_2_out_layers_0_bias: ptr = unet_view_1d(_base, 1226567040, 1280)
    ptr_array_set(_weights, 801, _w_middle_block_2_out_layers_0_bias)
    _w_middle_block_2_out_layers_0_weight: ptr = unet_view_1d(_base, 1226568320, 1280)
    ptr_array_set(_weights, 802, _w_middle_block_2_out_layers_0_weight)
    _w_middle_block_2_out_layers_3_bias: ptr = unet_view_1d(_base, 1226569600, 1280)
    ptr_array_set(_weights, 803, _w_middle_block_2_out_layers_3_bias)
    _w_middle_block_2_out_layers_3_weight: ptr = unet_view_4d(_base, 1226570880, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 804, _w_middle_block_2_out_layers_3_weight)
    _w_out_0_bias: ptr = unet_view_1d(_base, 1241316480, 320)
    ptr_array_set(_weights, 805, _w_out_0_bias)
    _w_out_0_weight: ptr = unet_view_1d(_base, 1241316800, 320)
    ptr_array_set(_weights, 806, _w_out_0_weight)
    _w_out_2_bias: ptr = unet_view_1d(_base, 1241317120, 4)
    ptr_array_set(_weights, 807, _w_out_2_bias)
    _w_out_2_weight: ptr = unet_view_4d(_base, 1241317124, 4, 320, 3, 3)
    ptr_array_set(_weights, 808, _w_out_2_weight)
    _w_output_blocks_0_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 1241328644, 1280)
    ptr_array_set(_weights, 809, _w_output_blocks_0_0_emb_layers_1_bias)
    _w_output_blocks_0_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 1241329924, 1280, 1280)
    ptr_array_set(_weights, 810, _w_output_blocks_0_0_emb_layers_1_weight)
    _w_output_blocks_0_0_in_layers_0_bias: ptr = unet_view_1d(_base, 1242968324, 2560)
    ptr_array_set(_weights, 811, _w_output_blocks_0_0_in_layers_0_bias)
    _w_output_blocks_0_0_in_layers_0_weight: ptr = unet_view_1d(_base, 1242970884, 2560)
    ptr_array_set(_weights, 812, _w_output_blocks_0_0_in_layers_0_weight)
    _w_output_blocks_0_0_in_layers_2_bias: ptr = unet_view_1d(_base, 1242973444, 1280)
    ptr_array_set(_weights, 813, _w_output_blocks_0_0_in_layers_2_bias)
    _w_output_blocks_0_0_in_layers_2_weight: ptr = unet_view_4d(_base, 1242974724, 1280, 2560, 3, 3)
    ptr_array_set(_weights, 814, _w_output_blocks_0_0_in_layers_2_weight)
    _w_output_blocks_0_0_out_layers_0_bias: ptr = unet_view_1d(_base, 1272465924, 1280)
    ptr_array_set(_weights, 815, _w_output_blocks_0_0_out_layers_0_bias)
    _w_output_blocks_0_0_out_layers_0_weight: ptr = unet_view_1d(_base, 1272467204, 1280)
    ptr_array_set(_weights, 816, _w_output_blocks_0_0_out_layers_0_weight)
    _w_output_blocks_0_0_out_layers_3_bias: ptr = unet_view_1d(_base, 1272468484, 1280)
    ptr_array_set(_weights, 817, _w_output_blocks_0_0_out_layers_3_bias)
    _w_output_blocks_0_0_out_layers_3_weight: ptr = unet_view_4d(_base, 1272469764, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 818, _w_output_blocks_0_0_out_layers_3_weight)
    _w_output_blocks_0_0_skip_connection_bias: ptr = unet_view_1d(_base, 1287215364, 1280)
    ptr_array_set(_weights, 819, _w_output_blocks_0_0_skip_connection_bias)
    _w_output_blocks_0_0_skip_connection_weight: ptr = unet_view_4d(_base, 1287216644, 1280, 2560, 1, 1)
    ptr_array_set(_weights, 820, _w_output_blocks_0_0_skip_connection_weight)
    _w_output_blocks_0_1_norm_bias: ptr = unet_view_1d(_base, 1290493444, 1280)
    ptr_array_set(_weights, 821, _w_output_blocks_0_1_norm_bias)
    _w_output_blocks_0_1_norm_weight: ptr = unet_view_1d(_base, 1290494724, 1280)
    ptr_array_set(_weights, 822, _w_output_blocks_0_1_norm_weight)
    _w_output_blocks_0_1_proj_in_bias: ptr = unet_view_1d(_base, 1290496004, 1280)
    ptr_array_set(_weights, 823, _w_output_blocks_0_1_proj_in_bias)
    _w_output_blocks_0_1_proj_in_weight: ptr = unet_view_2d(_base, 1290497284, 1280, 1280)
    ptr_array_set(_weights, 824, _w_output_blocks_0_1_proj_in_weight)
    _w_output_blocks_0_1_proj_out_bias: ptr = unet_view_1d(_base, 1292135684, 1280)
    ptr_array_set(_weights, 825, _w_output_blocks_0_1_proj_out_bias)
    _w_output_blocks_0_1_proj_out_weight: ptr = unet_view_2d(_base, 1292136964, 1280, 1280)
    ptr_array_set(_weights, 826, _w_output_blocks_0_1_proj_out_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 1293775364, 1280, 1280)
    ptr_array_set(_weights, 827, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1295413764, 1280)
    ptr_array_set(_weights, 828, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1295415044, 1280, 1280)
    ptr_array_set(_weights, 829, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 1297053444, 1280, 1280)
    ptr_array_set(_weights, 830, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 1298691844, 1280, 1280)
    ptr_array_set(_weights, 831, _w_output_blocks_0_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 1300330244, 1280, 2048)
    ptr_array_set(_weights, 832, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1302951684, 1280)
    ptr_array_set(_weights, 833, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1302952964, 1280, 1280)
    ptr_array_set(_weights, 834, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 1304591364, 1280, 1280)
    ptr_array_set(_weights, 835, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 1306229764, 1280, 2048)
    ptr_array_set(_weights, 836, _w_output_blocks_0_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1308851204, 10240)
    ptr_array_set(_weights, 837, _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1308861444, 10240, 1280)
    ptr_array_set(_weights, 838, _w_output_blocks_0_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 1321968644, 1280)
    ptr_array_set(_weights, 839, _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 1321969924, 1280, 5120)
    ptr_array_set(_weights, 840, _w_output_blocks_0_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 1328523524, 1280)
    ptr_array_set(_weights, 841, _w_output_blocks_0_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 1328524804, 1280)
    ptr_array_set(_weights, 842, _w_output_blocks_0_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 1328526084, 1280)
    ptr_array_set(_weights, 843, _w_output_blocks_0_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 1328527364, 1280)
    ptr_array_set(_weights, 844, _w_output_blocks_0_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 1328528644, 1280)
    ptr_array_set(_weights, 845, _w_output_blocks_0_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 1328529924, 1280)
    ptr_array_set(_weights, 846, _w_output_blocks_0_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 1328531204, 1280, 1280)
    ptr_array_set(_weights, 847, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1330169604, 1280)
    ptr_array_set(_weights, 848, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1330170884, 1280, 1280)
    ptr_array_set(_weights, 849, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 1331809284, 1280, 1280)
    ptr_array_set(_weights, 850, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 1333447684, 1280, 1280)
    ptr_array_set(_weights, 851, _w_output_blocks_0_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 1335086084, 1280, 2048)
    ptr_array_set(_weights, 852, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1337707524, 1280)
    ptr_array_set(_weights, 853, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1337708804, 1280, 1280)
    ptr_array_set(_weights, 854, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 1339347204, 1280, 1280)
    ptr_array_set(_weights, 855, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 1340985604, 1280, 2048)
    ptr_array_set(_weights, 856, _w_output_blocks_0_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1343607044, 10240)
    ptr_array_set(_weights, 857, _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1343617284, 10240, 1280)
    ptr_array_set(_weights, 858, _w_output_blocks_0_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 1356724484, 1280)
    ptr_array_set(_weights, 859, _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 1356725764, 1280, 5120)
    ptr_array_set(_weights, 860, _w_output_blocks_0_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 1363279364, 1280)
    ptr_array_set(_weights, 861, _w_output_blocks_0_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 1363280644, 1280)
    ptr_array_set(_weights, 862, _w_output_blocks_0_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 1363281924, 1280)
    ptr_array_set(_weights, 863, _w_output_blocks_0_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 1363283204, 1280)
    ptr_array_set(_weights, 864, _w_output_blocks_0_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 1363284484, 1280)
    ptr_array_set(_weights, 865, _w_output_blocks_0_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 1363285764, 1280)
    ptr_array_set(_weights, 866, _w_output_blocks_0_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 1363287044, 1280, 1280)
    ptr_array_set(_weights, 867, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1364925444, 1280)
    ptr_array_set(_weights, 868, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1364926724, 1280, 1280)
    ptr_array_set(_weights, 869, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 1366565124, 1280, 1280)
    ptr_array_set(_weights, 870, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 1368203524, 1280, 1280)
    ptr_array_set(_weights, 871, _w_output_blocks_0_1_transformer_blocks_2_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 1369841924, 1280, 2048)
    ptr_array_set(_weights, 872, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1372463364, 1280)
    ptr_array_set(_weights, 873, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1372464644, 1280, 1280)
    ptr_array_set(_weights, 874, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 1374103044, 1280, 1280)
    ptr_array_set(_weights, 875, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 1375741444, 1280, 2048)
    ptr_array_set(_weights, 876, _w_output_blocks_0_1_transformer_blocks_2_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1378362884, 10240)
    ptr_array_set(_weights, 877, _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1378373124, 10240, 1280)
    ptr_array_set(_weights, 878, _w_output_blocks_0_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 1391480324, 1280)
    ptr_array_set(_weights, 879, _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 1391481604, 1280, 5120)
    ptr_array_set(_weights, 880, _w_output_blocks_0_1_transformer_blocks_2_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 1398035204, 1280)
    ptr_array_set(_weights, 881, _w_output_blocks_0_1_transformer_blocks_2_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 1398036484, 1280)
    ptr_array_set(_weights, 882, _w_output_blocks_0_1_transformer_blocks_2_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 1398037764, 1280)
    ptr_array_set(_weights, 883, _w_output_blocks_0_1_transformer_blocks_2_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 1398039044, 1280)
    ptr_array_set(_weights, 884, _w_output_blocks_0_1_transformer_blocks_2_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 1398040324, 1280)
    ptr_array_set(_weights, 885, _w_output_blocks_0_1_transformer_blocks_2_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 1398041604, 1280)
    ptr_array_set(_weights, 886, _w_output_blocks_0_1_transformer_blocks_2_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 1398042884, 1280, 1280)
    ptr_array_set(_weights, 887, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1399681284, 1280)
    ptr_array_set(_weights, 888, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1399682564, 1280, 1280)
    ptr_array_set(_weights, 889, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 1401320964, 1280, 1280)
    ptr_array_set(_weights, 890, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 1402959364, 1280, 1280)
    ptr_array_set(_weights, 891, _w_output_blocks_0_1_transformer_blocks_3_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 1404597764, 1280, 2048)
    ptr_array_set(_weights, 892, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1407219204, 1280)
    ptr_array_set(_weights, 893, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1407220484, 1280, 1280)
    ptr_array_set(_weights, 894, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 1408858884, 1280, 1280)
    ptr_array_set(_weights, 895, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 1410497284, 1280, 2048)
    ptr_array_set(_weights, 896, _w_output_blocks_0_1_transformer_blocks_3_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1413118724, 10240)
    ptr_array_set(_weights, 897, _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1413128964, 10240, 1280)
    ptr_array_set(_weights, 898, _w_output_blocks_0_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 1426236164, 1280)
    ptr_array_set(_weights, 899, _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 1426237444, 1280, 5120)
    ptr_array_set(_weights, 900, _w_output_blocks_0_1_transformer_blocks_3_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 1432791044, 1280)
    ptr_array_set(_weights, 901, _w_output_blocks_0_1_transformer_blocks_3_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 1432792324, 1280)
    ptr_array_set(_weights, 902, _w_output_blocks_0_1_transformer_blocks_3_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 1432793604, 1280)
    ptr_array_set(_weights, 903, _w_output_blocks_0_1_transformer_blocks_3_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 1432794884, 1280)
    ptr_array_set(_weights, 904, _w_output_blocks_0_1_transformer_blocks_3_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 1432796164, 1280)
    ptr_array_set(_weights, 905, _w_output_blocks_0_1_transformer_blocks_3_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 1432797444, 1280)
    ptr_array_set(_weights, 906, _w_output_blocks_0_1_transformer_blocks_3_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 1432798724, 1280, 1280)
    ptr_array_set(_weights, 907, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1434437124, 1280)
    ptr_array_set(_weights, 908, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1434438404, 1280, 1280)
    ptr_array_set(_weights, 909, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 1436076804, 1280, 1280)
    ptr_array_set(_weights, 910, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 1437715204, 1280, 1280)
    ptr_array_set(_weights, 911, _w_output_blocks_0_1_transformer_blocks_4_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 1439353604, 1280, 2048)
    ptr_array_set(_weights, 912, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1441975044, 1280)
    ptr_array_set(_weights, 913, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1441976324, 1280, 1280)
    ptr_array_set(_weights, 914, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 1443614724, 1280, 1280)
    ptr_array_set(_weights, 915, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 1445253124, 1280, 2048)
    ptr_array_set(_weights, 916, _w_output_blocks_0_1_transformer_blocks_4_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1447874564, 10240)
    ptr_array_set(_weights, 917, _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1447884804, 10240, 1280)
    ptr_array_set(_weights, 918, _w_output_blocks_0_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 1460992004, 1280)
    ptr_array_set(_weights, 919, _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 1460993284, 1280, 5120)
    ptr_array_set(_weights, 920, _w_output_blocks_0_1_transformer_blocks_4_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 1467546884, 1280)
    ptr_array_set(_weights, 921, _w_output_blocks_0_1_transformer_blocks_4_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 1467548164, 1280)
    ptr_array_set(_weights, 922, _w_output_blocks_0_1_transformer_blocks_4_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 1467549444, 1280)
    ptr_array_set(_weights, 923, _w_output_blocks_0_1_transformer_blocks_4_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 1467550724, 1280)
    ptr_array_set(_weights, 924, _w_output_blocks_0_1_transformer_blocks_4_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 1467552004, 1280)
    ptr_array_set(_weights, 925, _w_output_blocks_0_1_transformer_blocks_4_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 1467553284, 1280)
    ptr_array_set(_weights, 926, _w_output_blocks_0_1_transformer_blocks_4_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 1467554564, 1280, 1280)
    ptr_array_set(_weights, 927, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1469192964, 1280)
    ptr_array_set(_weights, 928, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1469194244, 1280, 1280)
    ptr_array_set(_weights, 929, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 1470832644, 1280, 1280)
    ptr_array_set(_weights, 930, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 1472471044, 1280, 1280)
    ptr_array_set(_weights, 931, _w_output_blocks_0_1_transformer_blocks_5_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 1474109444, 1280, 2048)
    ptr_array_set(_weights, 932, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1476730884, 1280)
    ptr_array_set(_weights, 933, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1476732164, 1280, 1280)
    ptr_array_set(_weights, 934, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 1478370564, 1280, 1280)
    ptr_array_set(_weights, 935, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 1480008964, 1280, 2048)
    ptr_array_set(_weights, 936, _w_output_blocks_0_1_transformer_blocks_5_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1482630404, 10240)
    ptr_array_set(_weights, 937, _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1482640644, 10240, 1280)
    ptr_array_set(_weights, 938, _w_output_blocks_0_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 1495747844, 1280)
    ptr_array_set(_weights, 939, _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 1495749124, 1280, 5120)
    ptr_array_set(_weights, 940, _w_output_blocks_0_1_transformer_blocks_5_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 1502302724, 1280)
    ptr_array_set(_weights, 941, _w_output_blocks_0_1_transformer_blocks_5_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 1502304004, 1280)
    ptr_array_set(_weights, 942, _w_output_blocks_0_1_transformer_blocks_5_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 1502305284, 1280)
    ptr_array_set(_weights, 943, _w_output_blocks_0_1_transformer_blocks_5_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 1502306564, 1280)
    ptr_array_set(_weights, 944, _w_output_blocks_0_1_transformer_blocks_5_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 1502307844, 1280)
    ptr_array_set(_weights, 945, _w_output_blocks_0_1_transformer_blocks_5_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 1502309124, 1280)
    ptr_array_set(_weights, 946, _w_output_blocks_0_1_transformer_blocks_5_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 1502310404, 1280, 1280)
    ptr_array_set(_weights, 947, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1503948804, 1280)
    ptr_array_set(_weights, 948, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1503950084, 1280, 1280)
    ptr_array_set(_weights, 949, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 1505588484, 1280, 1280)
    ptr_array_set(_weights, 950, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 1507226884, 1280, 1280)
    ptr_array_set(_weights, 951, _w_output_blocks_0_1_transformer_blocks_6_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 1508865284, 1280, 2048)
    ptr_array_set(_weights, 952, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1511486724, 1280)
    ptr_array_set(_weights, 953, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1511488004, 1280, 1280)
    ptr_array_set(_weights, 954, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 1513126404, 1280, 1280)
    ptr_array_set(_weights, 955, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 1514764804, 1280, 2048)
    ptr_array_set(_weights, 956, _w_output_blocks_0_1_transformer_blocks_6_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1517386244, 10240)
    ptr_array_set(_weights, 957, _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1517396484, 10240, 1280)
    ptr_array_set(_weights, 958, _w_output_blocks_0_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 1530503684, 1280)
    ptr_array_set(_weights, 959, _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 1530504964, 1280, 5120)
    ptr_array_set(_weights, 960, _w_output_blocks_0_1_transformer_blocks_6_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 1537058564, 1280)
    ptr_array_set(_weights, 961, _w_output_blocks_0_1_transformer_blocks_6_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 1537059844, 1280)
    ptr_array_set(_weights, 962, _w_output_blocks_0_1_transformer_blocks_6_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 1537061124, 1280)
    ptr_array_set(_weights, 963, _w_output_blocks_0_1_transformer_blocks_6_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 1537062404, 1280)
    ptr_array_set(_weights, 964, _w_output_blocks_0_1_transformer_blocks_6_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 1537063684, 1280)
    ptr_array_set(_weights, 965, _w_output_blocks_0_1_transformer_blocks_6_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 1537064964, 1280)
    ptr_array_set(_weights, 966, _w_output_blocks_0_1_transformer_blocks_6_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 1537066244, 1280, 1280)
    ptr_array_set(_weights, 967, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1538704644, 1280)
    ptr_array_set(_weights, 968, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1538705924, 1280, 1280)
    ptr_array_set(_weights, 969, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 1540344324, 1280, 1280)
    ptr_array_set(_weights, 970, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 1541982724, 1280, 1280)
    ptr_array_set(_weights, 971, _w_output_blocks_0_1_transformer_blocks_7_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 1543621124, 1280, 2048)
    ptr_array_set(_weights, 972, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1546242564, 1280)
    ptr_array_set(_weights, 973, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1546243844, 1280, 1280)
    ptr_array_set(_weights, 974, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 1547882244, 1280, 1280)
    ptr_array_set(_weights, 975, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 1549520644, 1280, 2048)
    ptr_array_set(_weights, 976, _w_output_blocks_0_1_transformer_blocks_7_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1552142084, 10240)
    ptr_array_set(_weights, 977, _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1552152324, 10240, 1280)
    ptr_array_set(_weights, 978, _w_output_blocks_0_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 1565259524, 1280)
    ptr_array_set(_weights, 979, _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 1565260804, 1280, 5120)
    ptr_array_set(_weights, 980, _w_output_blocks_0_1_transformer_blocks_7_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 1571814404, 1280)
    ptr_array_set(_weights, 981, _w_output_blocks_0_1_transformer_blocks_7_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 1571815684, 1280)
    ptr_array_set(_weights, 982, _w_output_blocks_0_1_transformer_blocks_7_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 1571816964, 1280)
    ptr_array_set(_weights, 983, _w_output_blocks_0_1_transformer_blocks_7_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 1571818244, 1280)
    ptr_array_set(_weights, 984, _w_output_blocks_0_1_transformer_blocks_7_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 1571819524, 1280)
    ptr_array_set(_weights, 985, _w_output_blocks_0_1_transformer_blocks_7_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 1571820804, 1280)
    ptr_array_set(_weights, 986, _w_output_blocks_0_1_transformer_blocks_7_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 1571822084, 1280, 1280)
    ptr_array_set(_weights, 987, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1573460484, 1280)
    ptr_array_set(_weights, 988, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1573461764, 1280, 1280)
    ptr_array_set(_weights, 989, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 1575100164, 1280, 1280)
    ptr_array_set(_weights, 990, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 1576738564, 1280, 1280)
    ptr_array_set(_weights, 991, _w_output_blocks_0_1_transformer_blocks_8_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 1578376964, 1280, 2048)
    ptr_array_set(_weights, 992, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1580998404, 1280)
    ptr_array_set(_weights, 993, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1580999684, 1280, 1280)
    ptr_array_set(_weights, 994, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 1582638084, 1280, 1280)
    ptr_array_set(_weights, 995, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 1584276484, 1280, 2048)
    ptr_array_set(_weights, 996, _w_output_blocks_0_1_transformer_blocks_8_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1586897924, 10240)
    ptr_array_set(_weights, 997, _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1586908164, 10240, 1280)
    ptr_array_set(_weights, 998, _w_output_blocks_0_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 1600015364, 1280)
    ptr_array_set(_weights, 999, _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 1600016644, 1280, 5120)
    ptr_array_set(_weights, 1000, _w_output_blocks_0_1_transformer_blocks_8_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 1606570244, 1280)
    ptr_array_set(_weights, 1001, _w_output_blocks_0_1_transformer_blocks_8_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 1606571524, 1280)
    ptr_array_set(_weights, 1002, _w_output_blocks_0_1_transformer_blocks_8_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 1606572804, 1280)
    ptr_array_set(_weights, 1003, _w_output_blocks_0_1_transformer_blocks_8_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 1606574084, 1280)
    ptr_array_set(_weights, 1004, _w_output_blocks_0_1_transformer_blocks_8_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 1606575364, 1280)
    ptr_array_set(_weights, 1005, _w_output_blocks_0_1_transformer_blocks_8_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 1606576644, 1280)
    ptr_array_set(_weights, 1006, _w_output_blocks_0_1_transformer_blocks_8_norm3_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 1606577924, 1280, 1280)
    ptr_array_set(_weights, 1007, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1608216324, 1280)
    ptr_array_set(_weights, 1008, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1608217604, 1280, 1280)
    ptr_array_set(_weights, 1009, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 1609856004, 1280, 1280)
    ptr_array_set(_weights, 1010, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 1611494404, 1280, 1280)
    ptr_array_set(_weights, 1011, _w_output_blocks_0_1_transformer_blocks_9_attn1_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 1613132804, 1280, 2048)
    ptr_array_set(_weights, 1012, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_k_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1615754244, 1280)
    ptr_array_set(_weights, 1013, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1615755524, 1280, 1280)
    ptr_array_set(_weights, 1014, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 1617393924, 1280, 1280)
    ptr_array_set(_weights, 1015, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_q_weight)
    _w_output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 1619032324, 1280, 2048)
    ptr_array_set(_weights, 1016, _w_output_blocks_0_1_transformer_blocks_9_attn2_to_v_weight)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1621653764, 10240)
    ptr_array_set(_weights, 1017, _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1621664004, 10240, 1280)
    ptr_array_set(_weights, 1018, _w_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 1634771204, 1280)
    ptr_array_set(_weights, 1019, _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_bias)
    _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 1634772484, 1280, 5120)
    ptr_array_set(_weights, 1020, _w_output_blocks_0_1_transformer_blocks_9_ff_net_2_weight)
    _w_output_blocks_0_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 1641326084, 1280)
    ptr_array_set(_weights, 1021, _w_output_blocks_0_1_transformer_blocks_9_norm1_bias)
    _w_output_blocks_0_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 1641327364, 1280)
    ptr_array_set(_weights, 1022, _w_output_blocks_0_1_transformer_blocks_9_norm1_weight)
    _w_output_blocks_0_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 1641328644, 1280)
    ptr_array_set(_weights, 1023, _w_output_blocks_0_1_transformer_blocks_9_norm2_bias)
    _w_output_blocks_0_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 1641329924, 1280)
    ptr_array_set(_weights, 1024, _w_output_blocks_0_1_transformer_blocks_9_norm2_weight)
    _w_output_blocks_0_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 1641331204, 1280)
    ptr_array_set(_weights, 1025, _w_output_blocks_0_1_transformer_blocks_9_norm3_bias)
    _w_output_blocks_0_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 1641332484, 1280)
    ptr_array_set(_weights, 1026, _w_output_blocks_0_1_transformer_blocks_9_norm3_weight)
    _w_output_blocks_1_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 1641333764, 1280)
    ptr_array_set(_weights, 1027, _w_output_blocks_1_0_emb_layers_1_bias)
    _w_output_blocks_1_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 1641335044, 1280, 1280)
    ptr_array_set(_weights, 1028, _w_output_blocks_1_0_emb_layers_1_weight)
    _w_output_blocks_1_0_in_layers_0_bias: ptr = unet_view_1d(_base, 1642973444, 2560)
    ptr_array_set(_weights, 1029, _w_output_blocks_1_0_in_layers_0_bias)
    _w_output_blocks_1_0_in_layers_0_weight: ptr = unet_view_1d(_base, 1642976004, 2560)
    ptr_array_set(_weights, 1030, _w_output_blocks_1_0_in_layers_0_weight)
    _w_output_blocks_1_0_in_layers_2_bias: ptr = unet_view_1d(_base, 1642978564, 1280)
    ptr_array_set(_weights, 1031, _w_output_blocks_1_0_in_layers_2_bias)
    _w_output_blocks_1_0_in_layers_2_weight: ptr = unet_view_4d(_base, 1642979844, 1280, 2560, 3, 3)
    ptr_array_set(_weights, 1032, _w_output_blocks_1_0_in_layers_2_weight)
    _w_output_blocks_1_0_out_layers_0_bias: ptr = unet_view_1d(_base, 1672471044, 1280)
    ptr_array_set(_weights, 1033, _w_output_blocks_1_0_out_layers_0_bias)
    _w_output_blocks_1_0_out_layers_0_weight: ptr = unet_view_1d(_base, 1672472324, 1280)
    ptr_array_set(_weights, 1034, _w_output_blocks_1_0_out_layers_0_weight)
    _w_output_blocks_1_0_out_layers_3_bias: ptr = unet_view_1d(_base, 1672473604, 1280)
    ptr_array_set(_weights, 1035, _w_output_blocks_1_0_out_layers_3_bias)
    _w_output_blocks_1_0_out_layers_3_weight: ptr = unet_view_4d(_base, 1672474884, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 1036, _w_output_blocks_1_0_out_layers_3_weight)
    _w_output_blocks_1_0_skip_connection_bias: ptr = unet_view_1d(_base, 1687220484, 1280)
    ptr_array_set(_weights, 1037, _w_output_blocks_1_0_skip_connection_bias)
    _w_output_blocks_1_0_skip_connection_weight: ptr = unet_view_4d(_base, 1687221764, 1280, 2560, 1, 1)
    ptr_array_set(_weights, 1038, _w_output_blocks_1_0_skip_connection_weight)
    _w_output_blocks_1_1_norm_bias: ptr = unet_view_1d(_base, 1690498564, 1280)
    ptr_array_set(_weights, 1039, _w_output_blocks_1_1_norm_bias)
    _w_output_blocks_1_1_norm_weight: ptr = unet_view_1d(_base, 1690499844, 1280)
    ptr_array_set(_weights, 1040, _w_output_blocks_1_1_norm_weight)
    _w_output_blocks_1_1_proj_in_bias: ptr = unet_view_1d(_base, 1690501124, 1280)
    ptr_array_set(_weights, 1041, _w_output_blocks_1_1_proj_in_bias)
    _w_output_blocks_1_1_proj_in_weight: ptr = unet_view_2d(_base, 1690502404, 1280, 1280)
    ptr_array_set(_weights, 1042, _w_output_blocks_1_1_proj_in_weight)
    _w_output_blocks_1_1_proj_out_bias: ptr = unet_view_1d(_base, 1692140804, 1280)
    ptr_array_set(_weights, 1043, _w_output_blocks_1_1_proj_out_bias)
    _w_output_blocks_1_1_proj_out_weight: ptr = unet_view_2d(_base, 1692142084, 1280, 1280)
    ptr_array_set(_weights, 1044, _w_output_blocks_1_1_proj_out_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 1693780484, 1280, 1280)
    ptr_array_set(_weights, 1045, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1695418884, 1280)
    ptr_array_set(_weights, 1046, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1695420164, 1280, 1280)
    ptr_array_set(_weights, 1047, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 1697058564, 1280, 1280)
    ptr_array_set(_weights, 1048, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 1698696964, 1280, 1280)
    ptr_array_set(_weights, 1049, _w_output_blocks_1_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 1700335364, 1280, 2048)
    ptr_array_set(_weights, 1050, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1702956804, 1280)
    ptr_array_set(_weights, 1051, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1702958084, 1280, 1280)
    ptr_array_set(_weights, 1052, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 1704596484, 1280, 1280)
    ptr_array_set(_weights, 1053, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 1706234884, 1280, 2048)
    ptr_array_set(_weights, 1054, _w_output_blocks_1_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1708856324, 10240)
    ptr_array_set(_weights, 1055, _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1708866564, 10240, 1280)
    ptr_array_set(_weights, 1056, _w_output_blocks_1_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 1721973764, 1280)
    ptr_array_set(_weights, 1057, _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 1721975044, 1280, 5120)
    ptr_array_set(_weights, 1058, _w_output_blocks_1_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 1728528644, 1280)
    ptr_array_set(_weights, 1059, _w_output_blocks_1_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 1728529924, 1280)
    ptr_array_set(_weights, 1060, _w_output_blocks_1_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 1728531204, 1280)
    ptr_array_set(_weights, 1061, _w_output_blocks_1_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 1728532484, 1280)
    ptr_array_set(_weights, 1062, _w_output_blocks_1_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 1728533764, 1280)
    ptr_array_set(_weights, 1063, _w_output_blocks_1_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 1728535044, 1280)
    ptr_array_set(_weights, 1064, _w_output_blocks_1_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 1728536324, 1280, 1280)
    ptr_array_set(_weights, 1065, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1730174724, 1280)
    ptr_array_set(_weights, 1066, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1730176004, 1280, 1280)
    ptr_array_set(_weights, 1067, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 1731814404, 1280, 1280)
    ptr_array_set(_weights, 1068, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 1733452804, 1280, 1280)
    ptr_array_set(_weights, 1069, _w_output_blocks_1_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 1735091204, 1280, 2048)
    ptr_array_set(_weights, 1070, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1737712644, 1280)
    ptr_array_set(_weights, 1071, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1737713924, 1280, 1280)
    ptr_array_set(_weights, 1072, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 1739352324, 1280, 1280)
    ptr_array_set(_weights, 1073, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 1740990724, 1280, 2048)
    ptr_array_set(_weights, 1074, _w_output_blocks_1_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1743612164, 10240)
    ptr_array_set(_weights, 1075, _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1743622404, 10240, 1280)
    ptr_array_set(_weights, 1076, _w_output_blocks_1_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 1756729604, 1280)
    ptr_array_set(_weights, 1077, _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 1756730884, 1280, 5120)
    ptr_array_set(_weights, 1078, _w_output_blocks_1_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 1763284484, 1280)
    ptr_array_set(_weights, 1079, _w_output_blocks_1_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 1763285764, 1280)
    ptr_array_set(_weights, 1080, _w_output_blocks_1_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 1763287044, 1280)
    ptr_array_set(_weights, 1081, _w_output_blocks_1_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 1763288324, 1280)
    ptr_array_set(_weights, 1082, _w_output_blocks_1_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 1763289604, 1280)
    ptr_array_set(_weights, 1083, _w_output_blocks_1_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 1763290884, 1280)
    ptr_array_set(_weights, 1084, _w_output_blocks_1_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 1763292164, 1280, 1280)
    ptr_array_set(_weights, 1085, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1764930564, 1280)
    ptr_array_set(_weights, 1086, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1764931844, 1280, 1280)
    ptr_array_set(_weights, 1087, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 1766570244, 1280, 1280)
    ptr_array_set(_weights, 1088, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 1768208644, 1280, 1280)
    ptr_array_set(_weights, 1089, _w_output_blocks_1_1_transformer_blocks_2_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 1769847044, 1280, 2048)
    ptr_array_set(_weights, 1090, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1772468484, 1280)
    ptr_array_set(_weights, 1091, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1772469764, 1280, 1280)
    ptr_array_set(_weights, 1092, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 1774108164, 1280, 1280)
    ptr_array_set(_weights, 1093, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 1775746564, 1280, 2048)
    ptr_array_set(_weights, 1094, _w_output_blocks_1_1_transformer_blocks_2_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1778368004, 10240)
    ptr_array_set(_weights, 1095, _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1778378244, 10240, 1280)
    ptr_array_set(_weights, 1096, _w_output_blocks_1_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 1791485444, 1280)
    ptr_array_set(_weights, 1097, _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 1791486724, 1280, 5120)
    ptr_array_set(_weights, 1098, _w_output_blocks_1_1_transformer_blocks_2_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 1798040324, 1280)
    ptr_array_set(_weights, 1099, _w_output_blocks_1_1_transformer_blocks_2_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 1798041604, 1280)
    ptr_array_set(_weights, 1100, _w_output_blocks_1_1_transformer_blocks_2_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 1798042884, 1280)
    ptr_array_set(_weights, 1101, _w_output_blocks_1_1_transformer_blocks_2_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 1798044164, 1280)
    ptr_array_set(_weights, 1102, _w_output_blocks_1_1_transformer_blocks_2_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 1798045444, 1280)
    ptr_array_set(_weights, 1103, _w_output_blocks_1_1_transformer_blocks_2_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 1798046724, 1280)
    ptr_array_set(_weights, 1104, _w_output_blocks_1_1_transformer_blocks_2_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 1798048004, 1280, 1280)
    ptr_array_set(_weights, 1105, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1799686404, 1280)
    ptr_array_set(_weights, 1106, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1799687684, 1280, 1280)
    ptr_array_set(_weights, 1107, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 1801326084, 1280, 1280)
    ptr_array_set(_weights, 1108, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 1802964484, 1280, 1280)
    ptr_array_set(_weights, 1109, _w_output_blocks_1_1_transformer_blocks_3_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 1804602884, 1280, 2048)
    ptr_array_set(_weights, 1110, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1807224324, 1280)
    ptr_array_set(_weights, 1111, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1807225604, 1280, 1280)
    ptr_array_set(_weights, 1112, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 1808864004, 1280, 1280)
    ptr_array_set(_weights, 1113, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 1810502404, 1280, 2048)
    ptr_array_set(_weights, 1114, _w_output_blocks_1_1_transformer_blocks_3_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1813123844, 10240)
    ptr_array_set(_weights, 1115, _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1813134084, 10240, 1280)
    ptr_array_set(_weights, 1116, _w_output_blocks_1_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 1826241284, 1280)
    ptr_array_set(_weights, 1117, _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 1826242564, 1280, 5120)
    ptr_array_set(_weights, 1118, _w_output_blocks_1_1_transformer_blocks_3_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 1832796164, 1280)
    ptr_array_set(_weights, 1119, _w_output_blocks_1_1_transformer_blocks_3_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 1832797444, 1280)
    ptr_array_set(_weights, 1120, _w_output_blocks_1_1_transformer_blocks_3_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 1832798724, 1280)
    ptr_array_set(_weights, 1121, _w_output_blocks_1_1_transformer_blocks_3_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 1832800004, 1280)
    ptr_array_set(_weights, 1122, _w_output_blocks_1_1_transformer_blocks_3_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 1832801284, 1280)
    ptr_array_set(_weights, 1123, _w_output_blocks_1_1_transformer_blocks_3_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 1832802564, 1280)
    ptr_array_set(_weights, 1124, _w_output_blocks_1_1_transformer_blocks_3_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 1832803844, 1280, 1280)
    ptr_array_set(_weights, 1125, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1834442244, 1280)
    ptr_array_set(_weights, 1126, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1834443524, 1280, 1280)
    ptr_array_set(_weights, 1127, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 1836081924, 1280, 1280)
    ptr_array_set(_weights, 1128, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 1837720324, 1280, 1280)
    ptr_array_set(_weights, 1129, _w_output_blocks_1_1_transformer_blocks_4_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 1839358724, 1280, 2048)
    ptr_array_set(_weights, 1130, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1841980164, 1280)
    ptr_array_set(_weights, 1131, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1841981444, 1280, 1280)
    ptr_array_set(_weights, 1132, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 1843619844, 1280, 1280)
    ptr_array_set(_weights, 1133, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 1845258244, 1280, 2048)
    ptr_array_set(_weights, 1134, _w_output_blocks_1_1_transformer_blocks_4_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1847879684, 10240)
    ptr_array_set(_weights, 1135, _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1847889924, 10240, 1280)
    ptr_array_set(_weights, 1136, _w_output_blocks_1_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 1860997124, 1280)
    ptr_array_set(_weights, 1137, _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 1860998404, 1280, 5120)
    ptr_array_set(_weights, 1138, _w_output_blocks_1_1_transformer_blocks_4_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 1867552004, 1280)
    ptr_array_set(_weights, 1139, _w_output_blocks_1_1_transformer_blocks_4_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 1867553284, 1280)
    ptr_array_set(_weights, 1140, _w_output_blocks_1_1_transformer_blocks_4_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 1867554564, 1280)
    ptr_array_set(_weights, 1141, _w_output_blocks_1_1_transformer_blocks_4_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 1867555844, 1280)
    ptr_array_set(_weights, 1142, _w_output_blocks_1_1_transformer_blocks_4_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 1867557124, 1280)
    ptr_array_set(_weights, 1143, _w_output_blocks_1_1_transformer_blocks_4_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 1867558404, 1280)
    ptr_array_set(_weights, 1144, _w_output_blocks_1_1_transformer_blocks_4_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 1867559684, 1280, 1280)
    ptr_array_set(_weights, 1145, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1869198084, 1280)
    ptr_array_set(_weights, 1146, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1869199364, 1280, 1280)
    ptr_array_set(_weights, 1147, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 1870837764, 1280, 1280)
    ptr_array_set(_weights, 1148, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 1872476164, 1280, 1280)
    ptr_array_set(_weights, 1149, _w_output_blocks_1_1_transformer_blocks_5_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 1874114564, 1280, 2048)
    ptr_array_set(_weights, 1150, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1876736004, 1280)
    ptr_array_set(_weights, 1151, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1876737284, 1280, 1280)
    ptr_array_set(_weights, 1152, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 1878375684, 1280, 1280)
    ptr_array_set(_weights, 1153, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 1880014084, 1280, 2048)
    ptr_array_set(_weights, 1154, _w_output_blocks_1_1_transformer_blocks_5_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1882635524, 10240)
    ptr_array_set(_weights, 1155, _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1882645764, 10240, 1280)
    ptr_array_set(_weights, 1156, _w_output_blocks_1_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 1895752964, 1280)
    ptr_array_set(_weights, 1157, _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 1895754244, 1280, 5120)
    ptr_array_set(_weights, 1158, _w_output_blocks_1_1_transformer_blocks_5_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 1902307844, 1280)
    ptr_array_set(_weights, 1159, _w_output_blocks_1_1_transformer_blocks_5_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 1902309124, 1280)
    ptr_array_set(_weights, 1160, _w_output_blocks_1_1_transformer_blocks_5_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 1902310404, 1280)
    ptr_array_set(_weights, 1161, _w_output_blocks_1_1_transformer_blocks_5_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 1902311684, 1280)
    ptr_array_set(_weights, 1162, _w_output_blocks_1_1_transformer_blocks_5_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 1902312964, 1280)
    ptr_array_set(_weights, 1163, _w_output_blocks_1_1_transformer_blocks_5_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 1902314244, 1280)
    ptr_array_set(_weights, 1164, _w_output_blocks_1_1_transformer_blocks_5_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 1902315524, 1280, 1280)
    ptr_array_set(_weights, 1165, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1903953924, 1280)
    ptr_array_set(_weights, 1166, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1903955204, 1280, 1280)
    ptr_array_set(_weights, 1167, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 1905593604, 1280, 1280)
    ptr_array_set(_weights, 1168, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 1907232004, 1280, 1280)
    ptr_array_set(_weights, 1169, _w_output_blocks_1_1_transformer_blocks_6_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 1908870404, 1280, 2048)
    ptr_array_set(_weights, 1170, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1911491844, 1280)
    ptr_array_set(_weights, 1171, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1911493124, 1280, 1280)
    ptr_array_set(_weights, 1172, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 1913131524, 1280, 1280)
    ptr_array_set(_weights, 1173, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 1914769924, 1280, 2048)
    ptr_array_set(_weights, 1174, _w_output_blocks_1_1_transformer_blocks_6_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1917391364, 10240)
    ptr_array_set(_weights, 1175, _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1917401604, 10240, 1280)
    ptr_array_set(_weights, 1176, _w_output_blocks_1_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 1930508804, 1280)
    ptr_array_set(_weights, 1177, _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 1930510084, 1280, 5120)
    ptr_array_set(_weights, 1178, _w_output_blocks_1_1_transformer_blocks_6_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 1937063684, 1280)
    ptr_array_set(_weights, 1179, _w_output_blocks_1_1_transformer_blocks_6_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 1937064964, 1280)
    ptr_array_set(_weights, 1180, _w_output_blocks_1_1_transformer_blocks_6_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 1937066244, 1280)
    ptr_array_set(_weights, 1181, _w_output_blocks_1_1_transformer_blocks_6_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 1937067524, 1280)
    ptr_array_set(_weights, 1182, _w_output_blocks_1_1_transformer_blocks_6_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 1937068804, 1280)
    ptr_array_set(_weights, 1183, _w_output_blocks_1_1_transformer_blocks_6_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 1937070084, 1280)
    ptr_array_set(_weights, 1184, _w_output_blocks_1_1_transformer_blocks_6_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 1937071364, 1280, 1280)
    ptr_array_set(_weights, 1185, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1938709764, 1280)
    ptr_array_set(_weights, 1186, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1938711044, 1280, 1280)
    ptr_array_set(_weights, 1187, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 1940349444, 1280, 1280)
    ptr_array_set(_weights, 1188, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 1941987844, 1280, 1280)
    ptr_array_set(_weights, 1189, _w_output_blocks_1_1_transformer_blocks_7_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 1943626244, 1280, 2048)
    ptr_array_set(_weights, 1190, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1946247684, 1280)
    ptr_array_set(_weights, 1191, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1946248964, 1280, 1280)
    ptr_array_set(_weights, 1192, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 1947887364, 1280, 1280)
    ptr_array_set(_weights, 1193, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 1949525764, 1280, 2048)
    ptr_array_set(_weights, 1194, _w_output_blocks_1_1_transformer_blocks_7_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1952147204, 10240)
    ptr_array_set(_weights, 1195, _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1952157444, 10240, 1280)
    ptr_array_set(_weights, 1196, _w_output_blocks_1_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 1965264644, 1280)
    ptr_array_set(_weights, 1197, _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 1965265924, 1280, 5120)
    ptr_array_set(_weights, 1198, _w_output_blocks_1_1_transformer_blocks_7_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 1971819524, 1280)
    ptr_array_set(_weights, 1199, _w_output_blocks_1_1_transformer_blocks_7_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 1971820804, 1280)
    ptr_array_set(_weights, 1200, _w_output_blocks_1_1_transformer_blocks_7_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 1971822084, 1280)
    ptr_array_set(_weights, 1201, _w_output_blocks_1_1_transformer_blocks_7_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 1971823364, 1280)
    ptr_array_set(_weights, 1202, _w_output_blocks_1_1_transformer_blocks_7_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 1971824644, 1280)
    ptr_array_set(_weights, 1203, _w_output_blocks_1_1_transformer_blocks_7_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 1971825924, 1280)
    ptr_array_set(_weights, 1204, _w_output_blocks_1_1_transformer_blocks_7_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 1971827204, 1280, 1280)
    ptr_array_set(_weights, 1205, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 1973465604, 1280)
    ptr_array_set(_weights, 1206, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 1973466884, 1280, 1280)
    ptr_array_set(_weights, 1207, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 1975105284, 1280, 1280)
    ptr_array_set(_weights, 1208, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 1976743684, 1280, 1280)
    ptr_array_set(_weights, 1209, _w_output_blocks_1_1_transformer_blocks_8_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 1978382084, 1280, 2048)
    ptr_array_set(_weights, 1210, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 1981003524, 1280)
    ptr_array_set(_weights, 1211, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 1981004804, 1280, 1280)
    ptr_array_set(_weights, 1212, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 1982643204, 1280, 1280)
    ptr_array_set(_weights, 1213, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 1984281604, 1280, 2048)
    ptr_array_set(_weights, 1214, _w_output_blocks_1_1_transformer_blocks_8_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 1986903044, 10240)
    ptr_array_set(_weights, 1215, _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 1986913284, 10240, 1280)
    ptr_array_set(_weights, 1216, _w_output_blocks_1_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 2000020484, 1280)
    ptr_array_set(_weights, 1217, _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 2000021764, 1280, 5120)
    ptr_array_set(_weights, 1218, _w_output_blocks_1_1_transformer_blocks_8_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 2006575364, 1280)
    ptr_array_set(_weights, 1219, _w_output_blocks_1_1_transformer_blocks_8_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 2006576644, 1280)
    ptr_array_set(_weights, 1220, _w_output_blocks_1_1_transformer_blocks_8_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 2006577924, 1280)
    ptr_array_set(_weights, 1221, _w_output_blocks_1_1_transformer_blocks_8_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 2006579204, 1280)
    ptr_array_set(_weights, 1222, _w_output_blocks_1_1_transformer_blocks_8_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 2006580484, 1280)
    ptr_array_set(_weights, 1223, _w_output_blocks_1_1_transformer_blocks_8_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 2006581764, 1280)
    ptr_array_set(_weights, 1224, _w_output_blocks_1_1_transformer_blocks_8_norm3_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 2006583044, 1280, 1280)
    ptr_array_set(_weights, 1225, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2008221444, 1280)
    ptr_array_set(_weights, 1226, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2008222724, 1280, 1280)
    ptr_array_set(_weights, 1227, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 2009861124, 1280, 1280)
    ptr_array_set(_weights, 1228, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 2011499524, 1280, 1280)
    ptr_array_set(_weights, 1229, _w_output_blocks_1_1_transformer_blocks_9_attn1_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 2013137924, 1280, 2048)
    ptr_array_set(_weights, 1230, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_k_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2015759364, 1280)
    ptr_array_set(_weights, 1231, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2015760644, 1280, 1280)
    ptr_array_set(_weights, 1232, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 2017399044, 1280, 1280)
    ptr_array_set(_weights, 1233, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_q_weight)
    _w_output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 2019037444, 1280, 2048)
    ptr_array_set(_weights, 1234, _w_output_blocks_1_1_transformer_blocks_9_attn2_to_v_weight)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2021658884, 10240)
    ptr_array_set(_weights, 1235, _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2021669124, 10240, 1280)
    ptr_array_set(_weights, 1236, _w_output_blocks_1_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 2034776324, 1280)
    ptr_array_set(_weights, 1237, _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_bias)
    _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 2034777604, 1280, 5120)
    ptr_array_set(_weights, 1238, _w_output_blocks_1_1_transformer_blocks_9_ff_net_2_weight)
    _w_output_blocks_1_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 2041331204, 1280)
    ptr_array_set(_weights, 1239, _w_output_blocks_1_1_transformer_blocks_9_norm1_bias)
    _w_output_blocks_1_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 2041332484, 1280)
    ptr_array_set(_weights, 1240, _w_output_blocks_1_1_transformer_blocks_9_norm1_weight)
    _w_output_blocks_1_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 2041333764, 1280)
    ptr_array_set(_weights, 1241, _w_output_blocks_1_1_transformer_blocks_9_norm2_bias)
    _w_output_blocks_1_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 2041335044, 1280)
    ptr_array_set(_weights, 1242, _w_output_blocks_1_1_transformer_blocks_9_norm2_weight)
    _w_output_blocks_1_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 2041336324, 1280)
    ptr_array_set(_weights, 1243, _w_output_blocks_1_1_transformer_blocks_9_norm3_bias)
    _w_output_blocks_1_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 2041337604, 1280)
    ptr_array_set(_weights, 1244, _w_output_blocks_1_1_transformer_blocks_9_norm3_weight)
    _w_output_blocks_2_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2041338884, 1280)
    ptr_array_set(_weights, 1245, _w_output_blocks_2_0_emb_layers_1_bias)
    _w_output_blocks_2_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2041340164, 1280, 1280)
    ptr_array_set(_weights, 1246, _w_output_blocks_2_0_emb_layers_1_weight)
    _w_output_blocks_2_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2042978564, 1920)
    ptr_array_set(_weights, 1247, _w_output_blocks_2_0_in_layers_0_bias)
    _w_output_blocks_2_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2042980484, 1920)
    ptr_array_set(_weights, 1248, _w_output_blocks_2_0_in_layers_0_weight)
    _w_output_blocks_2_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2042982404, 1280)
    ptr_array_set(_weights, 1249, _w_output_blocks_2_0_in_layers_2_bias)
    _w_output_blocks_2_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2042983684, 1280, 1920, 3, 3)
    ptr_array_set(_weights, 1250, _w_output_blocks_2_0_in_layers_2_weight)
    _w_output_blocks_2_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2065102084, 1280)
    ptr_array_set(_weights, 1251, _w_output_blocks_2_0_out_layers_0_bias)
    _w_output_blocks_2_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2065103364, 1280)
    ptr_array_set(_weights, 1252, _w_output_blocks_2_0_out_layers_0_weight)
    _w_output_blocks_2_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2065104644, 1280)
    ptr_array_set(_weights, 1253, _w_output_blocks_2_0_out_layers_3_bias)
    _w_output_blocks_2_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2065105924, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 1254, _w_output_blocks_2_0_out_layers_3_weight)
    _w_output_blocks_2_0_skip_connection_bias: ptr = unet_view_1d(_base, 2079851524, 1280)
    ptr_array_set(_weights, 1255, _w_output_blocks_2_0_skip_connection_bias)
    _w_output_blocks_2_0_skip_connection_weight: ptr = unet_view_4d(_base, 2079852804, 1280, 1920, 1, 1)
    ptr_array_set(_weights, 1256, _w_output_blocks_2_0_skip_connection_weight)
    _w_output_blocks_2_1_norm_bias: ptr = unet_view_1d(_base, 2082310404, 1280)
    ptr_array_set(_weights, 1257, _w_output_blocks_2_1_norm_bias)
    _w_output_blocks_2_1_norm_weight: ptr = unet_view_1d(_base, 2082311684, 1280)
    ptr_array_set(_weights, 1258, _w_output_blocks_2_1_norm_weight)
    _w_output_blocks_2_1_proj_in_bias: ptr = unet_view_1d(_base, 2082312964, 1280)
    ptr_array_set(_weights, 1259, _w_output_blocks_2_1_proj_in_bias)
    _w_output_blocks_2_1_proj_in_weight: ptr = unet_view_2d(_base, 2082314244, 1280, 1280)
    ptr_array_set(_weights, 1260, _w_output_blocks_2_1_proj_in_weight)
    _w_output_blocks_2_1_proj_out_bias: ptr = unet_view_1d(_base, 2083952644, 1280)
    ptr_array_set(_weights, 1261, _w_output_blocks_2_1_proj_out_bias)
    _w_output_blocks_2_1_proj_out_weight: ptr = unet_view_2d(_base, 2083953924, 1280, 1280)
    ptr_array_set(_weights, 1262, _w_output_blocks_2_1_proj_out_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 2085592324, 1280, 1280)
    ptr_array_set(_weights, 1263, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2087230724, 1280)
    ptr_array_set(_weights, 1264, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2087232004, 1280, 1280)
    ptr_array_set(_weights, 1265, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 2088870404, 1280, 1280)
    ptr_array_set(_weights, 1266, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 2090508804, 1280, 1280)
    ptr_array_set(_weights, 1267, _w_output_blocks_2_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 2092147204, 1280, 2048)
    ptr_array_set(_weights, 1268, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2094768644, 1280)
    ptr_array_set(_weights, 1269, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2094769924, 1280, 1280)
    ptr_array_set(_weights, 1270, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 2096408324, 1280, 1280)
    ptr_array_set(_weights, 1271, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 2098046724, 1280, 2048)
    ptr_array_set(_weights, 1272, _w_output_blocks_2_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2100668164, 10240)
    ptr_array_set(_weights, 1273, _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2100678404, 10240, 1280)
    ptr_array_set(_weights, 1274, _w_output_blocks_2_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 2113785604, 1280)
    ptr_array_set(_weights, 1275, _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 2113786884, 1280, 5120)
    ptr_array_set(_weights, 1276, _w_output_blocks_2_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 2120340484, 1280)
    ptr_array_set(_weights, 1277, _w_output_blocks_2_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 2120341764, 1280)
    ptr_array_set(_weights, 1278, _w_output_blocks_2_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 2120343044, 1280)
    ptr_array_set(_weights, 1279, _w_output_blocks_2_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 2120344324, 1280)
    ptr_array_set(_weights, 1280, _w_output_blocks_2_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 2120345604, 1280)
    ptr_array_set(_weights, 1281, _w_output_blocks_2_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 2120346884, 1280)
    ptr_array_set(_weights, 1282, _w_output_blocks_2_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 2120348164, 1280, 1280)
    ptr_array_set(_weights, 1283, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2121986564, 1280)
    ptr_array_set(_weights, 1284, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2121987844, 1280, 1280)
    ptr_array_set(_weights, 1285, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 2123626244, 1280, 1280)
    ptr_array_set(_weights, 1286, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 2125264644, 1280, 1280)
    ptr_array_set(_weights, 1287, _w_output_blocks_2_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 2126903044, 1280, 2048)
    ptr_array_set(_weights, 1288, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2129524484, 1280)
    ptr_array_set(_weights, 1289, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2129525764, 1280, 1280)
    ptr_array_set(_weights, 1290, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 2131164164, 1280, 1280)
    ptr_array_set(_weights, 1291, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 2132802564, 1280, 2048)
    ptr_array_set(_weights, 1292, _w_output_blocks_2_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2135424004, 10240)
    ptr_array_set(_weights, 1293, _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2135434244, 10240, 1280)
    ptr_array_set(_weights, 1294, _w_output_blocks_2_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 2148541444, 1280)
    ptr_array_set(_weights, 1295, _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 2148542724, 1280, 5120)
    ptr_array_set(_weights, 1296, _w_output_blocks_2_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 2155096324, 1280)
    ptr_array_set(_weights, 1297, _w_output_blocks_2_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 2155097604, 1280)
    ptr_array_set(_weights, 1298, _w_output_blocks_2_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 2155098884, 1280)
    ptr_array_set(_weights, 1299, _w_output_blocks_2_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 2155100164, 1280)
    ptr_array_set(_weights, 1300, _w_output_blocks_2_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 2155101444, 1280)
    ptr_array_set(_weights, 1301, _w_output_blocks_2_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 2155102724, 1280)
    ptr_array_set(_weights, 1302, _w_output_blocks_2_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight: ptr = unet_view_2d(_base, 2155104004, 1280, 1280)
    ptr_array_set(_weights, 1303, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2156742404, 1280)
    ptr_array_set(_weights, 1304, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2156743684, 1280, 1280)
    ptr_array_set(_weights, 1305, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight: ptr = unet_view_2d(_base, 2158382084, 1280, 1280)
    ptr_array_set(_weights, 1306, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight: ptr = unet_view_2d(_base, 2160020484, 1280, 1280)
    ptr_array_set(_weights, 1307, _w_output_blocks_2_1_transformer_blocks_2_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight: ptr = unet_view_2d(_base, 2161658884, 1280, 2048)
    ptr_array_set(_weights, 1308, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2164280324, 1280)
    ptr_array_set(_weights, 1309, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2164281604, 1280, 1280)
    ptr_array_set(_weights, 1310, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight: ptr = unet_view_2d(_base, 2165920004, 1280, 1280)
    ptr_array_set(_weights, 1311, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight: ptr = unet_view_2d(_base, 2167558404, 1280, 2048)
    ptr_array_set(_weights, 1312, _w_output_blocks_2_1_transformer_blocks_2_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2170179844, 10240)
    ptr_array_set(_weights, 1313, _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2170190084, 10240, 1280)
    ptr_array_set(_weights, 1314, _w_output_blocks_2_1_transformer_blocks_2_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_bias: ptr = unet_view_1d(_base, 2183297284, 1280)
    ptr_array_set(_weights, 1315, _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_weight: ptr = unet_view_2d(_base, 2183298564, 1280, 5120)
    ptr_array_set(_weights, 1316, _w_output_blocks_2_1_transformer_blocks_2_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_2_norm1_bias: ptr = unet_view_1d(_base, 2189852164, 1280)
    ptr_array_set(_weights, 1317, _w_output_blocks_2_1_transformer_blocks_2_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_2_norm1_weight: ptr = unet_view_1d(_base, 2189853444, 1280)
    ptr_array_set(_weights, 1318, _w_output_blocks_2_1_transformer_blocks_2_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_2_norm2_bias: ptr = unet_view_1d(_base, 2189854724, 1280)
    ptr_array_set(_weights, 1319, _w_output_blocks_2_1_transformer_blocks_2_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_2_norm2_weight: ptr = unet_view_1d(_base, 2189856004, 1280)
    ptr_array_set(_weights, 1320, _w_output_blocks_2_1_transformer_blocks_2_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_2_norm3_bias: ptr = unet_view_1d(_base, 2189857284, 1280)
    ptr_array_set(_weights, 1321, _w_output_blocks_2_1_transformer_blocks_2_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_2_norm3_weight: ptr = unet_view_1d(_base, 2189858564, 1280)
    ptr_array_set(_weights, 1322, _w_output_blocks_2_1_transformer_blocks_2_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight: ptr = unet_view_2d(_base, 2189859844, 1280, 1280)
    ptr_array_set(_weights, 1323, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2191498244, 1280)
    ptr_array_set(_weights, 1324, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2191499524, 1280, 1280)
    ptr_array_set(_weights, 1325, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight: ptr = unet_view_2d(_base, 2193137924, 1280, 1280)
    ptr_array_set(_weights, 1326, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight: ptr = unet_view_2d(_base, 2194776324, 1280, 1280)
    ptr_array_set(_weights, 1327, _w_output_blocks_2_1_transformer_blocks_3_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight: ptr = unet_view_2d(_base, 2196414724, 1280, 2048)
    ptr_array_set(_weights, 1328, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2199036164, 1280)
    ptr_array_set(_weights, 1329, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2199037444, 1280, 1280)
    ptr_array_set(_weights, 1330, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight: ptr = unet_view_2d(_base, 2200675844, 1280, 1280)
    ptr_array_set(_weights, 1331, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight: ptr = unet_view_2d(_base, 2202314244, 1280, 2048)
    ptr_array_set(_weights, 1332, _w_output_blocks_2_1_transformer_blocks_3_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2204935684, 10240)
    ptr_array_set(_weights, 1333, _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2204945924, 10240, 1280)
    ptr_array_set(_weights, 1334, _w_output_blocks_2_1_transformer_blocks_3_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_bias: ptr = unet_view_1d(_base, 2218053124, 1280)
    ptr_array_set(_weights, 1335, _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_weight: ptr = unet_view_2d(_base, 2218054404, 1280, 5120)
    ptr_array_set(_weights, 1336, _w_output_blocks_2_1_transformer_blocks_3_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_3_norm1_bias: ptr = unet_view_1d(_base, 2224608004, 1280)
    ptr_array_set(_weights, 1337, _w_output_blocks_2_1_transformer_blocks_3_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_3_norm1_weight: ptr = unet_view_1d(_base, 2224609284, 1280)
    ptr_array_set(_weights, 1338, _w_output_blocks_2_1_transformer_blocks_3_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_3_norm2_bias: ptr = unet_view_1d(_base, 2224610564, 1280)
    ptr_array_set(_weights, 1339, _w_output_blocks_2_1_transformer_blocks_3_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_3_norm2_weight: ptr = unet_view_1d(_base, 2224611844, 1280)
    ptr_array_set(_weights, 1340, _w_output_blocks_2_1_transformer_blocks_3_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_3_norm3_bias: ptr = unet_view_1d(_base, 2224613124, 1280)
    ptr_array_set(_weights, 1341, _w_output_blocks_2_1_transformer_blocks_3_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_3_norm3_weight: ptr = unet_view_1d(_base, 2224614404, 1280)
    ptr_array_set(_weights, 1342, _w_output_blocks_2_1_transformer_blocks_3_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight: ptr = unet_view_2d(_base, 2224615684, 1280, 1280)
    ptr_array_set(_weights, 1343, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2226254084, 1280)
    ptr_array_set(_weights, 1344, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2226255364, 1280, 1280)
    ptr_array_set(_weights, 1345, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight: ptr = unet_view_2d(_base, 2227893764, 1280, 1280)
    ptr_array_set(_weights, 1346, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight: ptr = unet_view_2d(_base, 2229532164, 1280, 1280)
    ptr_array_set(_weights, 1347, _w_output_blocks_2_1_transformer_blocks_4_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight: ptr = unet_view_2d(_base, 2231170564, 1280, 2048)
    ptr_array_set(_weights, 1348, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2233792004, 1280)
    ptr_array_set(_weights, 1349, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2233793284, 1280, 1280)
    ptr_array_set(_weights, 1350, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight: ptr = unet_view_2d(_base, 2235431684, 1280, 1280)
    ptr_array_set(_weights, 1351, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight: ptr = unet_view_2d(_base, 2237070084, 1280, 2048)
    ptr_array_set(_weights, 1352, _w_output_blocks_2_1_transformer_blocks_4_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2239691524, 10240)
    ptr_array_set(_weights, 1353, _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2239701764, 10240, 1280)
    ptr_array_set(_weights, 1354, _w_output_blocks_2_1_transformer_blocks_4_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_bias: ptr = unet_view_1d(_base, 2252808964, 1280)
    ptr_array_set(_weights, 1355, _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_weight: ptr = unet_view_2d(_base, 2252810244, 1280, 5120)
    ptr_array_set(_weights, 1356, _w_output_blocks_2_1_transformer_blocks_4_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_4_norm1_bias: ptr = unet_view_1d(_base, 2259363844, 1280)
    ptr_array_set(_weights, 1357, _w_output_blocks_2_1_transformer_blocks_4_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_4_norm1_weight: ptr = unet_view_1d(_base, 2259365124, 1280)
    ptr_array_set(_weights, 1358, _w_output_blocks_2_1_transformer_blocks_4_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_4_norm2_bias: ptr = unet_view_1d(_base, 2259366404, 1280)
    ptr_array_set(_weights, 1359, _w_output_blocks_2_1_transformer_blocks_4_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_4_norm2_weight: ptr = unet_view_1d(_base, 2259367684, 1280)
    ptr_array_set(_weights, 1360, _w_output_blocks_2_1_transformer_blocks_4_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_4_norm3_bias: ptr = unet_view_1d(_base, 2259368964, 1280)
    ptr_array_set(_weights, 1361, _w_output_blocks_2_1_transformer_blocks_4_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_4_norm3_weight: ptr = unet_view_1d(_base, 2259370244, 1280)
    ptr_array_set(_weights, 1362, _w_output_blocks_2_1_transformer_blocks_4_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight: ptr = unet_view_2d(_base, 2259371524, 1280, 1280)
    ptr_array_set(_weights, 1363, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2261009924, 1280)
    ptr_array_set(_weights, 1364, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2261011204, 1280, 1280)
    ptr_array_set(_weights, 1365, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight: ptr = unet_view_2d(_base, 2262649604, 1280, 1280)
    ptr_array_set(_weights, 1366, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight: ptr = unet_view_2d(_base, 2264288004, 1280, 1280)
    ptr_array_set(_weights, 1367, _w_output_blocks_2_1_transformer_blocks_5_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight: ptr = unet_view_2d(_base, 2265926404, 1280, 2048)
    ptr_array_set(_weights, 1368, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2268547844, 1280)
    ptr_array_set(_weights, 1369, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2268549124, 1280, 1280)
    ptr_array_set(_weights, 1370, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight: ptr = unet_view_2d(_base, 2270187524, 1280, 1280)
    ptr_array_set(_weights, 1371, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight: ptr = unet_view_2d(_base, 2271825924, 1280, 2048)
    ptr_array_set(_weights, 1372, _w_output_blocks_2_1_transformer_blocks_5_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2274447364, 10240)
    ptr_array_set(_weights, 1373, _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2274457604, 10240, 1280)
    ptr_array_set(_weights, 1374, _w_output_blocks_2_1_transformer_blocks_5_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_bias: ptr = unet_view_1d(_base, 2287564804, 1280)
    ptr_array_set(_weights, 1375, _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_weight: ptr = unet_view_2d(_base, 2287566084, 1280, 5120)
    ptr_array_set(_weights, 1376, _w_output_blocks_2_1_transformer_blocks_5_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_5_norm1_bias: ptr = unet_view_1d(_base, 2294119684, 1280)
    ptr_array_set(_weights, 1377, _w_output_blocks_2_1_transformer_blocks_5_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_5_norm1_weight: ptr = unet_view_1d(_base, 2294120964, 1280)
    ptr_array_set(_weights, 1378, _w_output_blocks_2_1_transformer_blocks_5_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_5_norm2_bias: ptr = unet_view_1d(_base, 2294122244, 1280)
    ptr_array_set(_weights, 1379, _w_output_blocks_2_1_transformer_blocks_5_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_5_norm2_weight: ptr = unet_view_1d(_base, 2294123524, 1280)
    ptr_array_set(_weights, 1380, _w_output_blocks_2_1_transformer_blocks_5_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_5_norm3_bias: ptr = unet_view_1d(_base, 2294124804, 1280)
    ptr_array_set(_weights, 1381, _w_output_blocks_2_1_transformer_blocks_5_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_5_norm3_weight: ptr = unet_view_1d(_base, 2294126084, 1280)
    ptr_array_set(_weights, 1382, _w_output_blocks_2_1_transformer_blocks_5_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight: ptr = unet_view_2d(_base, 2294127364, 1280, 1280)
    ptr_array_set(_weights, 1383, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2295765764, 1280)
    ptr_array_set(_weights, 1384, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2295767044, 1280, 1280)
    ptr_array_set(_weights, 1385, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight: ptr = unet_view_2d(_base, 2297405444, 1280, 1280)
    ptr_array_set(_weights, 1386, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight: ptr = unet_view_2d(_base, 2299043844, 1280, 1280)
    ptr_array_set(_weights, 1387, _w_output_blocks_2_1_transformer_blocks_6_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight: ptr = unet_view_2d(_base, 2300682244, 1280, 2048)
    ptr_array_set(_weights, 1388, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2303303684, 1280)
    ptr_array_set(_weights, 1389, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2303304964, 1280, 1280)
    ptr_array_set(_weights, 1390, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight: ptr = unet_view_2d(_base, 2304943364, 1280, 1280)
    ptr_array_set(_weights, 1391, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight: ptr = unet_view_2d(_base, 2306581764, 1280, 2048)
    ptr_array_set(_weights, 1392, _w_output_blocks_2_1_transformer_blocks_6_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2309203204, 10240)
    ptr_array_set(_weights, 1393, _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2309213444, 10240, 1280)
    ptr_array_set(_weights, 1394, _w_output_blocks_2_1_transformer_blocks_6_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_bias: ptr = unet_view_1d(_base, 2322320644, 1280)
    ptr_array_set(_weights, 1395, _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_weight: ptr = unet_view_2d(_base, 2322321924, 1280, 5120)
    ptr_array_set(_weights, 1396, _w_output_blocks_2_1_transformer_blocks_6_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_6_norm1_bias: ptr = unet_view_1d(_base, 2328875524, 1280)
    ptr_array_set(_weights, 1397, _w_output_blocks_2_1_transformer_blocks_6_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_6_norm1_weight: ptr = unet_view_1d(_base, 2328876804, 1280)
    ptr_array_set(_weights, 1398, _w_output_blocks_2_1_transformer_blocks_6_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_6_norm2_bias: ptr = unet_view_1d(_base, 2328878084, 1280)
    ptr_array_set(_weights, 1399, _w_output_blocks_2_1_transformer_blocks_6_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_6_norm2_weight: ptr = unet_view_1d(_base, 2328879364, 1280)
    ptr_array_set(_weights, 1400, _w_output_blocks_2_1_transformer_blocks_6_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_6_norm3_bias: ptr = unet_view_1d(_base, 2328880644, 1280)
    ptr_array_set(_weights, 1401, _w_output_blocks_2_1_transformer_blocks_6_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_6_norm3_weight: ptr = unet_view_1d(_base, 2328881924, 1280)
    ptr_array_set(_weights, 1402, _w_output_blocks_2_1_transformer_blocks_6_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight: ptr = unet_view_2d(_base, 2328883204, 1280, 1280)
    ptr_array_set(_weights, 1403, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2330521604, 1280)
    ptr_array_set(_weights, 1404, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2330522884, 1280, 1280)
    ptr_array_set(_weights, 1405, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight: ptr = unet_view_2d(_base, 2332161284, 1280, 1280)
    ptr_array_set(_weights, 1406, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight: ptr = unet_view_2d(_base, 2333799684, 1280, 1280)
    ptr_array_set(_weights, 1407, _w_output_blocks_2_1_transformer_blocks_7_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight: ptr = unet_view_2d(_base, 2335438084, 1280, 2048)
    ptr_array_set(_weights, 1408, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2338059524, 1280)
    ptr_array_set(_weights, 1409, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2338060804, 1280, 1280)
    ptr_array_set(_weights, 1410, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight: ptr = unet_view_2d(_base, 2339699204, 1280, 1280)
    ptr_array_set(_weights, 1411, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight: ptr = unet_view_2d(_base, 2341337604, 1280, 2048)
    ptr_array_set(_weights, 1412, _w_output_blocks_2_1_transformer_blocks_7_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2343959044, 10240)
    ptr_array_set(_weights, 1413, _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2343969284, 10240, 1280)
    ptr_array_set(_weights, 1414, _w_output_blocks_2_1_transformer_blocks_7_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_bias: ptr = unet_view_1d(_base, 2357076484, 1280)
    ptr_array_set(_weights, 1415, _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_weight: ptr = unet_view_2d(_base, 2357077764, 1280, 5120)
    ptr_array_set(_weights, 1416, _w_output_blocks_2_1_transformer_blocks_7_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_7_norm1_bias: ptr = unet_view_1d(_base, 2363631364, 1280)
    ptr_array_set(_weights, 1417, _w_output_blocks_2_1_transformer_blocks_7_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_7_norm1_weight: ptr = unet_view_1d(_base, 2363632644, 1280)
    ptr_array_set(_weights, 1418, _w_output_blocks_2_1_transformer_blocks_7_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_7_norm2_bias: ptr = unet_view_1d(_base, 2363633924, 1280)
    ptr_array_set(_weights, 1419, _w_output_blocks_2_1_transformer_blocks_7_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_7_norm2_weight: ptr = unet_view_1d(_base, 2363635204, 1280)
    ptr_array_set(_weights, 1420, _w_output_blocks_2_1_transformer_blocks_7_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_7_norm3_bias: ptr = unet_view_1d(_base, 2363636484, 1280)
    ptr_array_set(_weights, 1421, _w_output_blocks_2_1_transformer_blocks_7_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_7_norm3_weight: ptr = unet_view_1d(_base, 2363637764, 1280)
    ptr_array_set(_weights, 1422, _w_output_blocks_2_1_transformer_blocks_7_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight: ptr = unet_view_2d(_base, 2363639044, 1280, 1280)
    ptr_array_set(_weights, 1423, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2365277444, 1280)
    ptr_array_set(_weights, 1424, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2365278724, 1280, 1280)
    ptr_array_set(_weights, 1425, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight: ptr = unet_view_2d(_base, 2366917124, 1280, 1280)
    ptr_array_set(_weights, 1426, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight: ptr = unet_view_2d(_base, 2368555524, 1280, 1280)
    ptr_array_set(_weights, 1427, _w_output_blocks_2_1_transformer_blocks_8_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight: ptr = unet_view_2d(_base, 2370193924, 1280, 2048)
    ptr_array_set(_weights, 1428, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2372815364, 1280)
    ptr_array_set(_weights, 1429, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2372816644, 1280, 1280)
    ptr_array_set(_weights, 1430, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight: ptr = unet_view_2d(_base, 2374455044, 1280, 1280)
    ptr_array_set(_weights, 1431, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight: ptr = unet_view_2d(_base, 2376093444, 1280, 2048)
    ptr_array_set(_weights, 1432, _w_output_blocks_2_1_transformer_blocks_8_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2378714884, 10240)
    ptr_array_set(_weights, 1433, _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2378725124, 10240, 1280)
    ptr_array_set(_weights, 1434, _w_output_blocks_2_1_transformer_blocks_8_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_bias: ptr = unet_view_1d(_base, 2391832324, 1280)
    ptr_array_set(_weights, 1435, _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_weight: ptr = unet_view_2d(_base, 2391833604, 1280, 5120)
    ptr_array_set(_weights, 1436, _w_output_blocks_2_1_transformer_blocks_8_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_8_norm1_bias: ptr = unet_view_1d(_base, 2398387204, 1280)
    ptr_array_set(_weights, 1437, _w_output_blocks_2_1_transformer_blocks_8_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_8_norm1_weight: ptr = unet_view_1d(_base, 2398388484, 1280)
    ptr_array_set(_weights, 1438, _w_output_blocks_2_1_transformer_blocks_8_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_8_norm2_bias: ptr = unet_view_1d(_base, 2398389764, 1280)
    ptr_array_set(_weights, 1439, _w_output_blocks_2_1_transformer_blocks_8_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_8_norm2_weight: ptr = unet_view_1d(_base, 2398391044, 1280)
    ptr_array_set(_weights, 1440, _w_output_blocks_2_1_transformer_blocks_8_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_8_norm3_bias: ptr = unet_view_1d(_base, 2398392324, 1280)
    ptr_array_set(_weights, 1441, _w_output_blocks_2_1_transformer_blocks_8_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_8_norm3_weight: ptr = unet_view_1d(_base, 2398393604, 1280)
    ptr_array_set(_weights, 1442, _w_output_blocks_2_1_transformer_blocks_8_norm3_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight: ptr = unet_view_2d(_base, 2398394884, 1280, 1280)
    ptr_array_set(_weights, 1443, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2400033284, 1280)
    ptr_array_set(_weights, 1444, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2400034564, 1280, 1280)
    ptr_array_set(_weights, 1445, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight: ptr = unet_view_2d(_base, 2401672964, 1280, 1280)
    ptr_array_set(_weights, 1446, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight: ptr = unet_view_2d(_base, 2403311364, 1280, 1280)
    ptr_array_set(_weights, 1447, _w_output_blocks_2_1_transformer_blocks_9_attn1_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight: ptr = unet_view_2d(_base, 2404949764, 1280, 2048)
    ptr_array_set(_weights, 1448, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_k_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2407571204, 1280)
    ptr_array_set(_weights, 1449, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_bias)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2407572484, 1280, 1280)
    ptr_array_set(_weights, 1450, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_out_0_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight: ptr = unet_view_2d(_base, 2409210884, 1280, 1280)
    ptr_array_set(_weights, 1451, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_q_weight)
    _w_output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight: ptr = unet_view_2d(_base, 2410849284, 1280, 2048)
    ptr_array_set(_weights, 1452, _w_output_blocks_2_1_transformer_blocks_9_attn2_to_v_weight)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2413470724, 10240)
    ptr_array_set(_weights, 1453, _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_bias)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2413480964, 10240, 1280)
    ptr_array_set(_weights, 1454, _w_output_blocks_2_1_transformer_blocks_9_ff_net_0_proj_weight)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_bias: ptr = unet_view_1d(_base, 2426588164, 1280)
    ptr_array_set(_weights, 1455, _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_bias)
    _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_weight: ptr = unet_view_2d(_base, 2426589444, 1280, 5120)
    ptr_array_set(_weights, 1456, _w_output_blocks_2_1_transformer_blocks_9_ff_net_2_weight)
    _w_output_blocks_2_1_transformer_blocks_9_norm1_bias: ptr = unet_view_1d(_base, 2433143044, 1280)
    ptr_array_set(_weights, 1457, _w_output_blocks_2_1_transformer_blocks_9_norm1_bias)
    _w_output_blocks_2_1_transformer_blocks_9_norm1_weight: ptr = unet_view_1d(_base, 2433144324, 1280)
    ptr_array_set(_weights, 1458, _w_output_blocks_2_1_transformer_blocks_9_norm1_weight)
    _w_output_blocks_2_1_transformer_blocks_9_norm2_bias: ptr = unet_view_1d(_base, 2433145604, 1280)
    ptr_array_set(_weights, 1459, _w_output_blocks_2_1_transformer_blocks_9_norm2_bias)
    _w_output_blocks_2_1_transformer_blocks_9_norm2_weight: ptr = unet_view_1d(_base, 2433146884, 1280)
    ptr_array_set(_weights, 1460, _w_output_blocks_2_1_transformer_blocks_9_norm2_weight)
    _w_output_blocks_2_1_transformer_blocks_9_norm3_bias: ptr = unet_view_1d(_base, 2433148164, 1280)
    ptr_array_set(_weights, 1461, _w_output_blocks_2_1_transformer_blocks_9_norm3_bias)
    _w_output_blocks_2_1_transformer_blocks_9_norm3_weight: ptr = unet_view_1d(_base, 2433149444, 1280)
    ptr_array_set(_weights, 1462, _w_output_blocks_2_1_transformer_blocks_9_norm3_weight)
    _w_output_blocks_2_2_conv_bias: ptr = unet_view_1d(_base, 2433150724, 1280)
    ptr_array_set(_weights, 1463, _w_output_blocks_2_2_conv_bias)
    _w_output_blocks_2_2_conv_weight: ptr = unet_view_4d(_base, 2433152004, 1280, 1280, 3, 3)
    ptr_array_set(_weights, 1464, _w_output_blocks_2_2_conv_weight)
    _w_output_blocks_3_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2447897604, 640)
    ptr_array_set(_weights, 1465, _w_output_blocks_3_0_emb_layers_1_bias)
    _w_output_blocks_3_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2447898244, 640, 1280)
    ptr_array_set(_weights, 1466, _w_output_blocks_3_0_emb_layers_1_weight)
    _w_output_blocks_3_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2448717444, 1920)
    ptr_array_set(_weights, 1467, _w_output_blocks_3_0_in_layers_0_bias)
    _w_output_blocks_3_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2448719364, 1920)
    ptr_array_set(_weights, 1468, _w_output_blocks_3_0_in_layers_0_weight)
    _w_output_blocks_3_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2448721284, 640)
    ptr_array_set(_weights, 1469, _w_output_blocks_3_0_in_layers_2_bias)
    _w_output_blocks_3_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2448721924, 640, 1920, 3, 3)
    ptr_array_set(_weights, 1470, _w_output_blocks_3_0_in_layers_2_weight)
    _w_output_blocks_3_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2459781124, 640)
    ptr_array_set(_weights, 1471, _w_output_blocks_3_0_out_layers_0_bias)
    _w_output_blocks_3_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2459781764, 640)
    ptr_array_set(_weights, 1472, _w_output_blocks_3_0_out_layers_0_weight)
    _w_output_blocks_3_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2459782404, 640)
    ptr_array_set(_weights, 1473, _w_output_blocks_3_0_out_layers_3_bias)
    _w_output_blocks_3_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2459783044, 640, 640, 3, 3)
    ptr_array_set(_weights, 1474, _w_output_blocks_3_0_out_layers_3_weight)
    _w_output_blocks_3_0_skip_connection_bias: ptr = unet_view_1d(_base, 2463469444, 640)
    ptr_array_set(_weights, 1475, _w_output_blocks_3_0_skip_connection_bias)
    _w_output_blocks_3_0_skip_connection_weight: ptr = unet_view_4d(_base, 2463470084, 640, 1920, 1, 1)
    ptr_array_set(_weights, 1476, _w_output_blocks_3_0_skip_connection_weight)
    _w_output_blocks_3_1_norm_bias: ptr = unet_view_1d(_base, 2464698884, 640)
    ptr_array_set(_weights, 1477, _w_output_blocks_3_1_norm_bias)
    _w_output_blocks_3_1_norm_weight: ptr = unet_view_1d(_base, 2464699524, 640)
    ptr_array_set(_weights, 1478, _w_output_blocks_3_1_norm_weight)
    _w_output_blocks_3_1_proj_in_bias: ptr = unet_view_1d(_base, 2464700164, 640)
    ptr_array_set(_weights, 1479, _w_output_blocks_3_1_proj_in_bias)
    _w_output_blocks_3_1_proj_in_weight: ptr = unet_view_2d(_base, 2464700804, 640, 640)
    ptr_array_set(_weights, 1480, _w_output_blocks_3_1_proj_in_weight)
    _w_output_blocks_3_1_proj_out_bias: ptr = unet_view_1d(_base, 2465110404, 640)
    ptr_array_set(_weights, 1481, _w_output_blocks_3_1_proj_out_bias)
    _w_output_blocks_3_1_proj_out_weight: ptr = unet_view_2d(_base, 2465111044, 640, 640)
    ptr_array_set(_weights, 1482, _w_output_blocks_3_1_proj_out_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 2465520644, 640, 640)
    ptr_array_set(_weights, 1483, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2465930244, 640)
    ptr_array_set(_weights, 1484, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2465930884, 640, 640)
    ptr_array_set(_weights, 1485, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 2466340484, 640, 640)
    ptr_array_set(_weights, 1486, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 2466750084, 640, 640)
    ptr_array_set(_weights, 1487, _w_output_blocks_3_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 2467159684, 640, 2048)
    ptr_array_set(_weights, 1488, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2468470404, 640)
    ptr_array_set(_weights, 1489, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2468471044, 640, 640)
    ptr_array_set(_weights, 1490, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 2468880644, 640, 640)
    ptr_array_set(_weights, 1491, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 2469290244, 640, 2048)
    ptr_array_set(_weights, 1492, _w_output_blocks_3_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2470600964, 5120)
    ptr_array_set(_weights, 1493, _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2470606084, 5120, 640)
    ptr_array_set(_weights, 1494, _w_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 2473882884, 640)
    ptr_array_set(_weights, 1495, _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 2473883524, 640, 2560)
    ptr_array_set(_weights, 1496, _w_output_blocks_3_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_3_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 2475521924, 640)
    ptr_array_set(_weights, 1497, _w_output_blocks_3_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_3_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 2475522564, 640)
    ptr_array_set(_weights, 1498, _w_output_blocks_3_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_3_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 2475523204, 640)
    ptr_array_set(_weights, 1499, _w_output_blocks_3_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_3_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 2475523844, 640)
    ptr_array_set(_weights, 1500, _w_output_blocks_3_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_3_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 2475524484, 640)
    ptr_array_set(_weights, 1501, _w_output_blocks_3_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_3_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 2475525124, 640)
    ptr_array_set(_weights, 1502, _w_output_blocks_3_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 2475525764, 640, 640)
    ptr_array_set(_weights, 1503, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2475935364, 640)
    ptr_array_set(_weights, 1504, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2475936004, 640, 640)
    ptr_array_set(_weights, 1505, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 2476345604, 640, 640)
    ptr_array_set(_weights, 1506, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 2476755204, 640, 640)
    ptr_array_set(_weights, 1507, _w_output_blocks_3_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 2477164804, 640, 2048)
    ptr_array_set(_weights, 1508, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2478475524, 640)
    ptr_array_set(_weights, 1509, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2478476164, 640, 640)
    ptr_array_set(_weights, 1510, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 2478885764, 640, 640)
    ptr_array_set(_weights, 1511, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 2479295364, 640, 2048)
    ptr_array_set(_weights, 1512, _w_output_blocks_3_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2480606084, 5120)
    ptr_array_set(_weights, 1513, _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2480611204, 5120, 640)
    ptr_array_set(_weights, 1514, _w_output_blocks_3_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 2483888004, 640)
    ptr_array_set(_weights, 1515, _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 2483888644, 640, 2560)
    ptr_array_set(_weights, 1516, _w_output_blocks_3_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_3_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 2485527044, 640)
    ptr_array_set(_weights, 1517, _w_output_blocks_3_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_3_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 2485527684, 640)
    ptr_array_set(_weights, 1518, _w_output_blocks_3_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_3_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 2485528324, 640)
    ptr_array_set(_weights, 1519, _w_output_blocks_3_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_3_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 2485528964, 640)
    ptr_array_set(_weights, 1520, _w_output_blocks_3_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_3_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 2485529604, 640)
    ptr_array_set(_weights, 1521, _w_output_blocks_3_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_3_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 2485530244, 640)
    ptr_array_set(_weights, 1522, _w_output_blocks_3_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_4_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2485530884, 640)
    ptr_array_set(_weights, 1523, _w_output_blocks_4_0_emb_layers_1_bias)
    _w_output_blocks_4_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2485531524, 640, 1280)
    ptr_array_set(_weights, 1524, _w_output_blocks_4_0_emb_layers_1_weight)
    _w_output_blocks_4_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2486350724, 1280)
    ptr_array_set(_weights, 1525, _w_output_blocks_4_0_in_layers_0_bias)
    _w_output_blocks_4_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2486352004, 1280)
    ptr_array_set(_weights, 1526, _w_output_blocks_4_0_in_layers_0_weight)
    _w_output_blocks_4_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2486353284, 640)
    ptr_array_set(_weights, 1527, _w_output_blocks_4_0_in_layers_2_bias)
    _w_output_blocks_4_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2486353924, 640, 1280, 3, 3)
    ptr_array_set(_weights, 1528, _w_output_blocks_4_0_in_layers_2_weight)
    _w_output_blocks_4_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2493726724, 640)
    ptr_array_set(_weights, 1529, _w_output_blocks_4_0_out_layers_0_bias)
    _w_output_blocks_4_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2493727364, 640)
    ptr_array_set(_weights, 1530, _w_output_blocks_4_0_out_layers_0_weight)
    _w_output_blocks_4_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2493728004, 640)
    ptr_array_set(_weights, 1531, _w_output_blocks_4_0_out_layers_3_bias)
    _w_output_blocks_4_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2493728644, 640, 640, 3, 3)
    ptr_array_set(_weights, 1532, _w_output_blocks_4_0_out_layers_3_weight)
    _w_output_blocks_4_0_skip_connection_bias: ptr = unet_view_1d(_base, 2497415044, 640)
    ptr_array_set(_weights, 1533, _w_output_blocks_4_0_skip_connection_bias)
    _w_output_blocks_4_0_skip_connection_weight: ptr = unet_view_4d(_base, 2497415684, 640, 1280, 1, 1)
    ptr_array_set(_weights, 1534, _w_output_blocks_4_0_skip_connection_weight)
    _w_output_blocks_4_1_norm_bias: ptr = unet_view_1d(_base, 2498234884, 640)
    ptr_array_set(_weights, 1535, _w_output_blocks_4_1_norm_bias)
    _w_output_blocks_4_1_norm_weight: ptr = unet_view_1d(_base, 2498235524, 640)
    ptr_array_set(_weights, 1536, _w_output_blocks_4_1_norm_weight)
    _w_output_blocks_4_1_proj_in_bias: ptr = unet_view_1d(_base, 2498236164, 640)
    ptr_array_set(_weights, 1537, _w_output_blocks_4_1_proj_in_bias)
    _w_output_blocks_4_1_proj_in_weight: ptr = unet_view_2d(_base, 2498236804, 640, 640)
    ptr_array_set(_weights, 1538, _w_output_blocks_4_1_proj_in_weight)
    _w_output_blocks_4_1_proj_out_bias: ptr = unet_view_1d(_base, 2498646404, 640)
    ptr_array_set(_weights, 1539, _w_output_blocks_4_1_proj_out_bias)
    _w_output_blocks_4_1_proj_out_weight: ptr = unet_view_2d(_base, 2498647044, 640, 640)
    ptr_array_set(_weights, 1540, _w_output_blocks_4_1_proj_out_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 2499056644, 640, 640)
    ptr_array_set(_weights, 1541, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2499466244, 640)
    ptr_array_set(_weights, 1542, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2499466884, 640, 640)
    ptr_array_set(_weights, 1543, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 2499876484, 640, 640)
    ptr_array_set(_weights, 1544, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 2500286084, 640, 640)
    ptr_array_set(_weights, 1545, _w_output_blocks_4_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 2500695684, 640, 2048)
    ptr_array_set(_weights, 1546, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2502006404, 640)
    ptr_array_set(_weights, 1547, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2502007044, 640, 640)
    ptr_array_set(_weights, 1548, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 2502416644, 640, 640)
    ptr_array_set(_weights, 1549, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 2502826244, 640, 2048)
    ptr_array_set(_weights, 1550, _w_output_blocks_4_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2504136964, 5120)
    ptr_array_set(_weights, 1551, _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2504142084, 5120, 640)
    ptr_array_set(_weights, 1552, _w_output_blocks_4_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 2507418884, 640)
    ptr_array_set(_weights, 1553, _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 2507419524, 640, 2560)
    ptr_array_set(_weights, 1554, _w_output_blocks_4_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_4_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 2509057924, 640)
    ptr_array_set(_weights, 1555, _w_output_blocks_4_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_4_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 2509058564, 640)
    ptr_array_set(_weights, 1556, _w_output_blocks_4_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_4_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 2509059204, 640)
    ptr_array_set(_weights, 1557, _w_output_blocks_4_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_4_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 2509059844, 640)
    ptr_array_set(_weights, 1558, _w_output_blocks_4_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_4_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 2509060484, 640)
    ptr_array_set(_weights, 1559, _w_output_blocks_4_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_4_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 2509061124, 640)
    ptr_array_set(_weights, 1560, _w_output_blocks_4_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 2509061764, 640, 640)
    ptr_array_set(_weights, 1561, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2509471364, 640)
    ptr_array_set(_weights, 1562, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2509472004, 640, 640)
    ptr_array_set(_weights, 1563, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 2509881604, 640, 640)
    ptr_array_set(_weights, 1564, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 2510291204, 640, 640)
    ptr_array_set(_weights, 1565, _w_output_blocks_4_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 2510700804, 640, 2048)
    ptr_array_set(_weights, 1566, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2512011524, 640)
    ptr_array_set(_weights, 1567, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2512012164, 640, 640)
    ptr_array_set(_weights, 1568, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 2512421764, 640, 640)
    ptr_array_set(_weights, 1569, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 2512831364, 640, 2048)
    ptr_array_set(_weights, 1570, _w_output_blocks_4_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2514142084, 5120)
    ptr_array_set(_weights, 1571, _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2514147204, 5120, 640)
    ptr_array_set(_weights, 1572, _w_output_blocks_4_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 2517424004, 640)
    ptr_array_set(_weights, 1573, _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 2517424644, 640, 2560)
    ptr_array_set(_weights, 1574, _w_output_blocks_4_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_4_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 2519063044, 640)
    ptr_array_set(_weights, 1575, _w_output_blocks_4_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_4_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 2519063684, 640)
    ptr_array_set(_weights, 1576, _w_output_blocks_4_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_4_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 2519064324, 640)
    ptr_array_set(_weights, 1577, _w_output_blocks_4_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_4_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 2519064964, 640)
    ptr_array_set(_weights, 1578, _w_output_blocks_4_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_4_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 2519065604, 640)
    ptr_array_set(_weights, 1579, _w_output_blocks_4_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_4_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 2519066244, 640)
    ptr_array_set(_weights, 1580, _w_output_blocks_4_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_5_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2519066884, 640)
    ptr_array_set(_weights, 1581, _w_output_blocks_5_0_emb_layers_1_bias)
    _w_output_blocks_5_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2519067524, 640, 1280)
    ptr_array_set(_weights, 1582, _w_output_blocks_5_0_emb_layers_1_weight)
    _w_output_blocks_5_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2519886724, 960)
    ptr_array_set(_weights, 1583, _w_output_blocks_5_0_in_layers_0_bias)
    _w_output_blocks_5_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2519887684, 960)
    ptr_array_set(_weights, 1584, _w_output_blocks_5_0_in_layers_0_weight)
    _w_output_blocks_5_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2519888644, 640)
    ptr_array_set(_weights, 1585, _w_output_blocks_5_0_in_layers_2_bias)
    _w_output_blocks_5_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2519889284, 640, 960, 3, 3)
    ptr_array_set(_weights, 1586, _w_output_blocks_5_0_in_layers_2_weight)
    _w_output_blocks_5_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2525418884, 640)
    ptr_array_set(_weights, 1587, _w_output_blocks_5_0_out_layers_0_bias)
    _w_output_blocks_5_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2525419524, 640)
    ptr_array_set(_weights, 1588, _w_output_blocks_5_0_out_layers_0_weight)
    _w_output_blocks_5_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2525420164, 640)
    ptr_array_set(_weights, 1589, _w_output_blocks_5_0_out_layers_3_bias)
    _w_output_blocks_5_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2525420804, 640, 640, 3, 3)
    ptr_array_set(_weights, 1590, _w_output_blocks_5_0_out_layers_3_weight)
    _w_output_blocks_5_0_skip_connection_bias: ptr = unet_view_1d(_base, 2529107204, 640)
    ptr_array_set(_weights, 1591, _w_output_blocks_5_0_skip_connection_bias)
    _w_output_blocks_5_0_skip_connection_weight: ptr = unet_view_4d(_base, 2529107844, 640, 960, 1, 1)
    ptr_array_set(_weights, 1592, _w_output_blocks_5_0_skip_connection_weight)
    _w_output_blocks_5_1_norm_bias: ptr = unet_view_1d(_base, 2529722244, 640)
    ptr_array_set(_weights, 1593, _w_output_blocks_5_1_norm_bias)
    _w_output_blocks_5_1_norm_weight: ptr = unet_view_1d(_base, 2529722884, 640)
    ptr_array_set(_weights, 1594, _w_output_blocks_5_1_norm_weight)
    _w_output_blocks_5_1_proj_in_bias: ptr = unet_view_1d(_base, 2529723524, 640)
    ptr_array_set(_weights, 1595, _w_output_blocks_5_1_proj_in_bias)
    _w_output_blocks_5_1_proj_in_weight: ptr = unet_view_2d(_base, 2529724164, 640, 640)
    ptr_array_set(_weights, 1596, _w_output_blocks_5_1_proj_in_weight)
    _w_output_blocks_5_1_proj_out_bias: ptr = unet_view_1d(_base, 2530133764, 640)
    ptr_array_set(_weights, 1597, _w_output_blocks_5_1_proj_out_bias)
    _w_output_blocks_5_1_proj_out_weight: ptr = unet_view_2d(_base, 2530134404, 640, 640)
    ptr_array_set(_weights, 1598, _w_output_blocks_5_1_proj_out_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight: ptr = unet_view_2d(_base, 2530544004, 640, 640)
    ptr_array_set(_weights, 1599, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_k_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2530953604, 640)
    ptr_array_set(_weights, 1600, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_bias)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2530954244, 640, 640)
    ptr_array_set(_weights, 1601, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_out_0_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight: ptr = unet_view_2d(_base, 2531363844, 640, 640)
    ptr_array_set(_weights, 1602, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_q_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight: ptr = unet_view_2d(_base, 2531773444, 640, 640)
    ptr_array_set(_weights, 1603, _w_output_blocks_5_1_transformer_blocks_0_attn1_to_v_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight: ptr = unet_view_2d(_base, 2532183044, 640, 2048)
    ptr_array_set(_weights, 1604, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_k_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2533493764, 640)
    ptr_array_set(_weights, 1605, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_bias)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2533494404, 640, 640)
    ptr_array_set(_weights, 1606, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight: ptr = unet_view_2d(_base, 2533904004, 640, 640)
    ptr_array_set(_weights, 1607, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_q_weight)
    _w_output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight: ptr = unet_view_2d(_base, 2534313604, 640, 2048)
    ptr_array_set(_weights, 1608, _w_output_blocks_5_1_transformer_blocks_0_attn2_to_v_weight)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2535624324, 5120)
    ptr_array_set(_weights, 1609, _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_bias)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2535629444, 5120, 640)
    ptr_array_set(_weights, 1610, _w_output_blocks_5_1_transformer_blocks_0_ff_net_0_proj_weight)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_bias: ptr = unet_view_1d(_base, 2538906244, 640)
    ptr_array_set(_weights, 1611, _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_bias)
    _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_weight: ptr = unet_view_2d(_base, 2538906884, 640, 2560)
    ptr_array_set(_weights, 1612, _w_output_blocks_5_1_transformer_blocks_0_ff_net_2_weight)
    _w_output_blocks_5_1_transformer_blocks_0_norm1_bias: ptr = unet_view_1d(_base, 2540545284, 640)
    ptr_array_set(_weights, 1613, _w_output_blocks_5_1_transformer_blocks_0_norm1_bias)
    _w_output_blocks_5_1_transformer_blocks_0_norm1_weight: ptr = unet_view_1d(_base, 2540545924, 640)
    ptr_array_set(_weights, 1614, _w_output_blocks_5_1_transformer_blocks_0_norm1_weight)
    _w_output_blocks_5_1_transformer_blocks_0_norm2_bias: ptr = unet_view_1d(_base, 2540546564, 640)
    ptr_array_set(_weights, 1615, _w_output_blocks_5_1_transformer_blocks_0_norm2_bias)
    _w_output_blocks_5_1_transformer_blocks_0_norm2_weight: ptr = unet_view_1d(_base, 2540547204, 640)
    ptr_array_set(_weights, 1616, _w_output_blocks_5_1_transformer_blocks_0_norm2_weight)
    _w_output_blocks_5_1_transformer_blocks_0_norm3_bias: ptr = unet_view_1d(_base, 2540547844, 640)
    ptr_array_set(_weights, 1617, _w_output_blocks_5_1_transformer_blocks_0_norm3_bias)
    _w_output_blocks_5_1_transformer_blocks_0_norm3_weight: ptr = unet_view_1d(_base, 2540548484, 640)
    ptr_array_set(_weights, 1618, _w_output_blocks_5_1_transformer_blocks_0_norm3_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight: ptr = unet_view_2d(_base, 2540549124, 640, 640)
    ptr_array_set(_weights, 1619, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_k_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias: ptr = unet_view_1d(_base, 2540958724, 640)
    ptr_array_set(_weights, 1620, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_bias)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight: ptr = unet_view_2d(_base, 2540959364, 640, 640)
    ptr_array_set(_weights, 1621, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_out_0_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight: ptr = unet_view_2d(_base, 2541368964, 640, 640)
    ptr_array_set(_weights, 1622, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_q_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight: ptr = unet_view_2d(_base, 2541778564, 640, 640)
    ptr_array_set(_weights, 1623, _w_output_blocks_5_1_transformer_blocks_1_attn1_to_v_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight: ptr = unet_view_2d(_base, 2542188164, 640, 2048)
    ptr_array_set(_weights, 1624, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_k_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias: ptr = unet_view_1d(_base, 2543498884, 640)
    ptr_array_set(_weights, 1625, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_bias)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight: ptr = unet_view_2d(_base, 2543499524, 640, 640)
    ptr_array_set(_weights, 1626, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_out_0_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight: ptr = unet_view_2d(_base, 2543909124, 640, 640)
    ptr_array_set(_weights, 1627, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_q_weight)
    _w_output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight: ptr = unet_view_2d(_base, 2544318724, 640, 2048)
    ptr_array_set(_weights, 1628, _w_output_blocks_5_1_transformer_blocks_1_attn2_to_v_weight)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias: ptr = unet_view_1d(_base, 2545629444, 5120)
    ptr_array_set(_weights, 1629, _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_bias)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight: ptr = unet_view_2d(_base, 2545634564, 5120, 640)
    ptr_array_set(_weights, 1630, _w_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj_weight)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_bias: ptr = unet_view_1d(_base, 2548911364, 640)
    ptr_array_set(_weights, 1631, _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_bias)
    _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_weight: ptr = unet_view_2d(_base, 2548912004, 640, 2560)
    ptr_array_set(_weights, 1632, _w_output_blocks_5_1_transformer_blocks_1_ff_net_2_weight)
    _w_output_blocks_5_1_transformer_blocks_1_norm1_bias: ptr = unet_view_1d(_base, 2550550404, 640)
    ptr_array_set(_weights, 1633, _w_output_blocks_5_1_transformer_blocks_1_norm1_bias)
    _w_output_blocks_5_1_transformer_blocks_1_norm1_weight: ptr = unet_view_1d(_base, 2550551044, 640)
    ptr_array_set(_weights, 1634, _w_output_blocks_5_1_transformer_blocks_1_norm1_weight)
    _w_output_blocks_5_1_transformer_blocks_1_norm2_bias: ptr = unet_view_1d(_base, 2550551684, 640)
    ptr_array_set(_weights, 1635, _w_output_blocks_5_1_transformer_blocks_1_norm2_bias)
    _w_output_blocks_5_1_transformer_blocks_1_norm2_weight: ptr = unet_view_1d(_base, 2550552324, 640)
    ptr_array_set(_weights, 1636, _w_output_blocks_5_1_transformer_blocks_1_norm2_weight)
    _w_output_blocks_5_1_transformer_blocks_1_norm3_bias: ptr = unet_view_1d(_base, 2550552964, 640)
    ptr_array_set(_weights, 1637, _w_output_blocks_5_1_transformer_blocks_1_norm3_bias)
    _w_output_blocks_5_1_transformer_blocks_1_norm3_weight: ptr = unet_view_1d(_base, 2550553604, 640)
    ptr_array_set(_weights, 1638, _w_output_blocks_5_1_transformer_blocks_1_norm3_weight)
    _w_output_blocks_5_2_conv_bias: ptr = unet_view_1d(_base, 2550554244, 640)
    ptr_array_set(_weights, 1639, _w_output_blocks_5_2_conv_bias)
    _w_output_blocks_5_2_conv_weight: ptr = unet_view_4d(_base, 2550554884, 640, 640, 3, 3)
    ptr_array_set(_weights, 1640, _w_output_blocks_5_2_conv_weight)
    _w_output_blocks_6_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2554241284, 320)
    ptr_array_set(_weights, 1641, _w_output_blocks_6_0_emb_layers_1_bias)
    _w_output_blocks_6_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2554241604, 320, 1280)
    ptr_array_set(_weights, 1642, _w_output_blocks_6_0_emb_layers_1_weight)
    _w_output_blocks_6_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2554651204, 960)
    ptr_array_set(_weights, 1643, _w_output_blocks_6_0_in_layers_0_bias)
    _w_output_blocks_6_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2554652164, 960)
    ptr_array_set(_weights, 1644, _w_output_blocks_6_0_in_layers_0_weight)
    _w_output_blocks_6_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2554653124, 320)
    ptr_array_set(_weights, 1645, _w_output_blocks_6_0_in_layers_2_bias)
    _w_output_blocks_6_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2554653444, 320, 960, 3, 3)
    ptr_array_set(_weights, 1646, _w_output_blocks_6_0_in_layers_2_weight)
    _w_output_blocks_6_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2557418244, 320)
    ptr_array_set(_weights, 1647, _w_output_blocks_6_0_out_layers_0_bias)
    _w_output_blocks_6_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2557418564, 320)
    ptr_array_set(_weights, 1648, _w_output_blocks_6_0_out_layers_0_weight)
    _w_output_blocks_6_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2557418884, 320)
    ptr_array_set(_weights, 1649, _w_output_blocks_6_0_out_layers_3_bias)
    _w_output_blocks_6_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2557419204, 320, 320, 3, 3)
    ptr_array_set(_weights, 1650, _w_output_blocks_6_0_out_layers_3_weight)
    _w_output_blocks_6_0_skip_connection_bias: ptr = unet_view_1d(_base, 2558340804, 320)
    ptr_array_set(_weights, 1651, _w_output_blocks_6_0_skip_connection_bias)
    _w_output_blocks_6_0_skip_connection_weight: ptr = unet_view_4d(_base, 2558341124, 320, 960, 1, 1)
    ptr_array_set(_weights, 1652, _w_output_blocks_6_0_skip_connection_weight)
    _w_output_blocks_7_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2558648324, 320)
    ptr_array_set(_weights, 1653, _w_output_blocks_7_0_emb_layers_1_bias)
    _w_output_blocks_7_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2558648644, 320, 1280)
    ptr_array_set(_weights, 1654, _w_output_blocks_7_0_emb_layers_1_weight)
    _w_output_blocks_7_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2559058244, 640)
    ptr_array_set(_weights, 1655, _w_output_blocks_7_0_in_layers_0_bias)
    _w_output_blocks_7_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2559058884, 640)
    ptr_array_set(_weights, 1656, _w_output_blocks_7_0_in_layers_0_weight)
    _w_output_blocks_7_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2559059524, 320)
    ptr_array_set(_weights, 1657, _w_output_blocks_7_0_in_layers_2_bias)
    _w_output_blocks_7_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2559059844, 320, 640, 3, 3)
    ptr_array_set(_weights, 1658, _w_output_blocks_7_0_in_layers_2_weight)
    _w_output_blocks_7_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2560903044, 320)
    ptr_array_set(_weights, 1659, _w_output_blocks_7_0_out_layers_0_bias)
    _w_output_blocks_7_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2560903364, 320)
    ptr_array_set(_weights, 1660, _w_output_blocks_7_0_out_layers_0_weight)
    _w_output_blocks_7_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2560903684, 320)
    ptr_array_set(_weights, 1661, _w_output_blocks_7_0_out_layers_3_bias)
    _w_output_blocks_7_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2560904004, 320, 320, 3, 3)
    ptr_array_set(_weights, 1662, _w_output_blocks_7_0_out_layers_3_weight)
    _w_output_blocks_7_0_skip_connection_bias: ptr = unet_view_1d(_base, 2561825604, 320)
    ptr_array_set(_weights, 1663, _w_output_blocks_7_0_skip_connection_bias)
    _w_output_blocks_7_0_skip_connection_weight: ptr = unet_view_4d(_base, 2561825924, 320, 640, 1, 1)
    ptr_array_set(_weights, 1664, _w_output_blocks_7_0_skip_connection_weight)
    _w_output_blocks_8_0_emb_layers_1_bias: ptr = unet_view_1d(_base, 2562030724, 320)
    ptr_array_set(_weights, 1665, _w_output_blocks_8_0_emb_layers_1_bias)
    _w_output_blocks_8_0_emb_layers_1_weight: ptr = unet_view_2d(_base, 2562031044, 320, 1280)
    ptr_array_set(_weights, 1666, _w_output_blocks_8_0_emb_layers_1_weight)
    _w_output_blocks_8_0_in_layers_0_bias: ptr = unet_view_1d(_base, 2562440644, 640)
    ptr_array_set(_weights, 1667, _w_output_blocks_8_0_in_layers_0_bias)
    _w_output_blocks_8_0_in_layers_0_weight: ptr = unet_view_1d(_base, 2562441284, 640)
    ptr_array_set(_weights, 1668, _w_output_blocks_8_0_in_layers_0_weight)
    _w_output_blocks_8_0_in_layers_2_bias: ptr = unet_view_1d(_base, 2562441924, 320)
    ptr_array_set(_weights, 1669, _w_output_blocks_8_0_in_layers_2_bias)
    _w_output_blocks_8_0_in_layers_2_weight: ptr = unet_view_4d(_base, 2562442244, 320, 640, 3, 3)
    ptr_array_set(_weights, 1670, _w_output_blocks_8_0_in_layers_2_weight)
    _w_output_blocks_8_0_out_layers_0_bias: ptr = unet_view_1d(_base, 2564285444, 320)
    ptr_array_set(_weights, 1671, _w_output_blocks_8_0_out_layers_0_bias)
    _w_output_blocks_8_0_out_layers_0_weight: ptr = unet_view_1d(_base, 2564285764, 320)
    ptr_array_set(_weights, 1672, _w_output_blocks_8_0_out_layers_0_weight)
    _w_output_blocks_8_0_out_layers_3_bias: ptr = unet_view_1d(_base, 2564286084, 320)
    ptr_array_set(_weights, 1673, _w_output_blocks_8_0_out_layers_3_bias)
    _w_output_blocks_8_0_out_layers_3_weight: ptr = unet_view_4d(_base, 2564286404, 320, 320, 3, 3)
    ptr_array_set(_weights, 1674, _w_output_blocks_8_0_out_layers_3_weight)
    _w_output_blocks_8_0_skip_connection_bias: ptr = unet_view_1d(_base, 2565208004, 320)
    ptr_array_set(_weights, 1675, _w_output_blocks_8_0_skip_connection_bias)
    _w_output_blocks_8_0_skip_connection_weight: ptr = unet_view_4d(_base, 2565208324, 320, 640, 1, 1)
    ptr_array_set(_weights, 1676, _w_output_blocks_8_0_skip_connection_weight)
    _w_time_embed_0_bias: ptr = unet_view_1d(_base, 2565413124, 1280)
    ptr_array_set(_weights, 1677, _w_time_embed_0_bias)
    _w_time_embed_0_weight: ptr = unet_view_2d(_base, 2565414404, 1280, 320)
    ptr_array_set(_weights, 1678, _w_time_embed_0_weight)
    _w_time_embed_2_bias: ptr = unet_view_1d(_base, 2565824004, 1280)
    ptr_array_set(_weights, 1679, _w_time_embed_2_bias)
    _w_time_embed_2_weight: ptr = unet_view_2d(_base, 2565825284, 1280, 1280)
    ptr_array_set(_weights, 1680, _w_time_embed_2_weight)
    return _weights

# attention mode: manual (ComfyUI source-aligned q@k^T / softmax / attn@v)
def attention_torch(q: ptr, k: ptr, v: ptr, batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> ptr:
    return attention_torch_manual(q, k, v, batch, tokens_q, tokens_k, dim, heads)

def unet_forward(latent: ptr, timestep: list[float], context: ptr, y: ptr, weights: ptr, n: int, hh: int, ww: int) -> ptr:
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
    emb = st_tensor_to_half(emb)
    emb = linear_torch(emb, ptr_array_ref(weights, 1678), ptr_array_ref(weights, 1677), n, 320, 1280)
    emb = silu_torch(emb)
    emb = linear_torch(emb, ptr_array_ref(weights, 1680), ptr_array_ref(weights, 1679), n, 1280, 1280)
    emb = silu_torch(emb)
    _y_emb = linear_torch(y, ptr_array_ref(weights, 576), ptr_array_ref(weights, 575), n, 2816, 1280)
    _y_emb = silu_torch(_y_emb)
    _y_emb = linear_torch(_y_emb, ptr_array_ref(weights, 578), ptr_array_ref(weights, 577), n, 1280, 1280)
    emb = add_tensor(emb, _y_emb)
    st_tensor_free(_y_emb)

    h_cur = conv2d_torch(latent, ptr_array_ref(weights, 2), ptr_array_ref(weights, 1), n, 4, 320, hh, ww, 3, 1, 3//2)
    _ss_0: ptr = st_clone(h_cur)
    ptr_array_set(_s, 0, _ss_0)

    # input_blocks.1
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 6), ptr_array_ref(weights, 5), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 8), ptr_array_ref(weights, 7), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 4), ptr_array_ref(weights, 3), n, 1280, 320)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 10), ptr_array_ref(weights, 9), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 12), ptr_array_ref(weights, 11), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    _ss_1: ptr = st_clone(h_cur)
    ptr_array_set(_s, 1, _ss_1)

    # input_blocks.2
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 16), ptr_array_ref(weights, 15), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 18), ptr_array_ref(weights, 17), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 14), ptr_array_ref(weights, 13), n, 1280, 320)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 20), ptr_array_ref(weights, 19), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 22), ptr_array_ref(weights, 21), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    _ss_2: ptr = st_clone(h_cur)
    ptr_array_set(_s, 2, _ss_2)

    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 24), ptr_array_ref(weights, 23), n, 320, 320, hh, ww, 3, 2, 1)
    st_tensor_free(_h_old)
    hh = hh//2; ww = ww//2
    _ss_3: ptr = st_clone(h_cur)
    ptr_array_set(_s, 3, _ss_3)

    # input_blocks.4
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 28), ptr_array_ref(weights, 27), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 30), ptr_array_ref(weights, 29), n, 320, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 26), ptr_array_ref(weights, 25), n, 1280, 640)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 32), ptr_array_ref(weights, 31), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 34), ptr_array_ref(weights, 33), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 36), ptr_array_ref(weights, 35), n, 320, 640, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.4.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 38), ptr_array_ref(weights, 37), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 40), ptr_array_ref(weights, 39), n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 58), ptr_array_ref(weights, 57),
        ptr_array_ref(weights, 60), ptr_array_ref(weights, 59),
        ptr_array_ref(weights, 62), ptr_array_ref(weights, 61),
        ptr_array_ref(weights, 46), 0,
        ptr_array_ref(weights, 43), 0,
        ptr_array_ref(weights, 47), 0,
        ptr_array_ref(weights, 45), ptr_array_ref(weights, 44),
        ptr_array_ref(weights, 51), 0,
        ptr_array_ref(weights, 48), 0,
        ptr_array_ref(weights, 52), 0,
        ptr_array_ref(weights, 50), ptr_array_ref(weights, 49),
        ptr_array_ref(weights, 54), ptr_array_ref(weights, 53),
        ptr_array_ref(weights, 56), ptr_array_ref(weights, 55),
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 78), ptr_array_ref(weights, 77),
        ptr_array_ref(weights, 80), ptr_array_ref(weights, 79),
        ptr_array_ref(weights, 82), ptr_array_ref(weights, 81),
        ptr_array_ref(weights, 66), 0,
        ptr_array_ref(weights, 63), 0,
        ptr_array_ref(weights, 67), 0,
        ptr_array_ref(weights, 65), ptr_array_ref(weights, 64),
        ptr_array_ref(weights, 71), 0,
        ptr_array_ref(weights, 68), 0,
        ptr_array_ref(weights, 72), 0,
        ptr_array_ref(weights, 70), ptr_array_ref(weights, 69),
        ptr_array_ref(weights, 74), ptr_array_ref(weights, 73),
        ptr_array_ref(weights, 76), ptr_array_ref(weights, 75),
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 42), ptr_array_ref(weights, 41), n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _ss_4: ptr = st_clone(h_cur)
    ptr_array_set(_s, 4, _ss_4)

    # input_blocks.5
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 86), ptr_array_ref(weights, 85), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 88), ptr_array_ref(weights, 87), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 84), ptr_array_ref(weights, 83), n, 1280, 640)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 90), ptr_array_ref(weights, 89), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 92), ptr_array_ref(weights, 91), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.5.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 94), ptr_array_ref(weights, 93), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 96), ptr_array_ref(weights, 95), n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 114), ptr_array_ref(weights, 113),
        ptr_array_ref(weights, 116), ptr_array_ref(weights, 115),
        ptr_array_ref(weights, 118), ptr_array_ref(weights, 117),
        ptr_array_ref(weights, 102), 0,
        ptr_array_ref(weights, 99), 0,
        ptr_array_ref(weights, 103), 0,
        ptr_array_ref(weights, 101), ptr_array_ref(weights, 100),
        ptr_array_ref(weights, 107), 0,
        ptr_array_ref(weights, 104), 0,
        ptr_array_ref(weights, 108), 0,
        ptr_array_ref(weights, 106), ptr_array_ref(weights, 105),
        ptr_array_ref(weights, 110), ptr_array_ref(weights, 109),
        ptr_array_ref(weights, 112), ptr_array_ref(weights, 111),
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 134), ptr_array_ref(weights, 133),
        ptr_array_ref(weights, 136), ptr_array_ref(weights, 135),
        ptr_array_ref(weights, 138), ptr_array_ref(weights, 137),
        ptr_array_ref(weights, 122), 0,
        ptr_array_ref(weights, 119), 0,
        ptr_array_ref(weights, 123), 0,
        ptr_array_ref(weights, 121), ptr_array_ref(weights, 120),
        ptr_array_ref(weights, 127), 0,
        ptr_array_ref(weights, 124), 0,
        ptr_array_ref(weights, 128), 0,
        ptr_array_ref(weights, 126), ptr_array_ref(weights, 125),
        ptr_array_ref(weights, 130), ptr_array_ref(weights, 129),
        ptr_array_ref(weights, 132), ptr_array_ref(weights, 131),
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 98), ptr_array_ref(weights, 97), n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _ss_5: ptr = st_clone(h_cur)
    ptr_array_set(_s, 5, _ss_5)

    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 140), ptr_array_ref(weights, 139), n, 640, 640, hh, ww, 3, 2, 1)
    st_tensor_free(_h_old)
    hh = hh//2; ww = ww//2
    _ss_6: ptr = st_clone(h_cur)
    ptr_array_set(_s, 6, _ss_6)

    # input_blocks.7
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 144), ptr_array_ref(weights, 143), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 146), ptr_array_ref(weights, 145), n, 640, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 142), ptr_array_ref(weights, 141), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 148), ptr_array_ref(weights, 147), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 150), ptr_array_ref(weights, 149), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 152), ptr_array_ref(weights, 151), n, 640, 1280, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.7.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 154), ptr_array_ref(weights, 153), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 156), ptr_array_ref(weights, 155), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 174), ptr_array_ref(weights, 173),
        ptr_array_ref(weights, 176), ptr_array_ref(weights, 175),
        ptr_array_ref(weights, 178), ptr_array_ref(weights, 177),
        ptr_array_ref(weights, 162), 0,
        ptr_array_ref(weights, 159), 0,
        ptr_array_ref(weights, 163), 0,
        ptr_array_ref(weights, 161), ptr_array_ref(weights, 160),
        ptr_array_ref(weights, 167), 0,
        ptr_array_ref(weights, 164), 0,
        ptr_array_ref(weights, 168), 0,
        ptr_array_ref(weights, 166), ptr_array_ref(weights, 165),
        ptr_array_ref(weights, 170), ptr_array_ref(weights, 169),
        ptr_array_ref(weights, 172), ptr_array_ref(weights, 171),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 194), ptr_array_ref(weights, 193),
        ptr_array_ref(weights, 196), ptr_array_ref(weights, 195),
        ptr_array_ref(weights, 198), ptr_array_ref(weights, 197),
        ptr_array_ref(weights, 182), 0,
        ptr_array_ref(weights, 179), 0,
        ptr_array_ref(weights, 183), 0,
        ptr_array_ref(weights, 181), ptr_array_ref(weights, 180),
        ptr_array_ref(weights, 187), 0,
        ptr_array_ref(weights, 184), 0,
        ptr_array_ref(weights, 188), 0,
        ptr_array_ref(weights, 186), ptr_array_ref(weights, 185),
        ptr_array_ref(weights, 190), ptr_array_ref(weights, 189),
        ptr_array_ref(weights, 192), ptr_array_ref(weights, 191),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 214), ptr_array_ref(weights, 213),
        ptr_array_ref(weights, 216), ptr_array_ref(weights, 215),
        ptr_array_ref(weights, 218), ptr_array_ref(weights, 217),
        ptr_array_ref(weights, 202), 0,
        ptr_array_ref(weights, 199), 0,
        ptr_array_ref(weights, 203), 0,
        ptr_array_ref(weights, 201), ptr_array_ref(weights, 200),
        ptr_array_ref(weights, 207), 0,
        ptr_array_ref(weights, 204), 0,
        ptr_array_ref(weights, 208), 0,
        ptr_array_ref(weights, 206), ptr_array_ref(weights, 205),
        ptr_array_ref(weights, 210), ptr_array_ref(weights, 209),
        ptr_array_ref(weights, 212), ptr_array_ref(weights, 211),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 234), ptr_array_ref(weights, 233),
        ptr_array_ref(weights, 236), ptr_array_ref(weights, 235),
        ptr_array_ref(weights, 238), ptr_array_ref(weights, 237),
        ptr_array_ref(weights, 222), 0,
        ptr_array_ref(weights, 219), 0,
        ptr_array_ref(weights, 223), 0,
        ptr_array_ref(weights, 221), ptr_array_ref(weights, 220),
        ptr_array_ref(weights, 227), 0,
        ptr_array_ref(weights, 224), 0,
        ptr_array_ref(weights, 228), 0,
        ptr_array_ref(weights, 226), ptr_array_ref(weights, 225),
        ptr_array_ref(weights, 230), ptr_array_ref(weights, 229),
        ptr_array_ref(weights, 232), ptr_array_ref(weights, 231),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 254), ptr_array_ref(weights, 253),
        ptr_array_ref(weights, 256), ptr_array_ref(weights, 255),
        ptr_array_ref(weights, 258), ptr_array_ref(weights, 257),
        ptr_array_ref(weights, 242), 0,
        ptr_array_ref(weights, 239), 0,
        ptr_array_ref(weights, 243), 0,
        ptr_array_ref(weights, 241), ptr_array_ref(weights, 240),
        ptr_array_ref(weights, 247), 0,
        ptr_array_ref(weights, 244), 0,
        ptr_array_ref(weights, 248), 0,
        ptr_array_ref(weights, 246), ptr_array_ref(weights, 245),
        ptr_array_ref(weights, 250), ptr_array_ref(weights, 249),
        ptr_array_ref(weights, 252), ptr_array_ref(weights, 251),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 274), ptr_array_ref(weights, 273),
        ptr_array_ref(weights, 276), ptr_array_ref(weights, 275),
        ptr_array_ref(weights, 278), ptr_array_ref(weights, 277),
        ptr_array_ref(weights, 262), 0,
        ptr_array_ref(weights, 259), 0,
        ptr_array_ref(weights, 263), 0,
        ptr_array_ref(weights, 261), ptr_array_ref(weights, 260),
        ptr_array_ref(weights, 267), 0,
        ptr_array_ref(weights, 264), 0,
        ptr_array_ref(weights, 268), 0,
        ptr_array_ref(weights, 266), ptr_array_ref(weights, 265),
        ptr_array_ref(weights, 270), ptr_array_ref(weights, 269),
        ptr_array_ref(weights, 272), ptr_array_ref(weights, 271),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 294), ptr_array_ref(weights, 293),
        ptr_array_ref(weights, 296), ptr_array_ref(weights, 295),
        ptr_array_ref(weights, 298), ptr_array_ref(weights, 297),
        ptr_array_ref(weights, 282), 0,
        ptr_array_ref(weights, 279), 0,
        ptr_array_ref(weights, 283), 0,
        ptr_array_ref(weights, 281), ptr_array_ref(weights, 280),
        ptr_array_ref(weights, 287), 0,
        ptr_array_ref(weights, 284), 0,
        ptr_array_ref(weights, 288), 0,
        ptr_array_ref(weights, 286), ptr_array_ref(weights, 285),
        ptr_array_ref(weights, 290), ptr_array_ref(weights, 289),
        ptr_array_ref(weights, 292), ptr_array_ref(weights, 291),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 314), ptr_array_ref(weights, 313),
        ptr_array_ref(weights, 316), ptr_array_ref(weights, 315),
        ptr_array_ref(weights, 318), ptr_array_ref(weights, 317),
        ptr_array_ref(weights, 302), 0,
        ptr_array_ref(weights, 299), 0,
        ptr_array_ref(weights, 303), 0,
        ptr_array_ref(weights, 301), ptr_array_ref(weights, 300),
        ptr_array_ref(weights, 307), 0,
        ptr_array_ref(weights, 304), 0,
        ptr_array_ref(weights, 308), 0,
        ptr_array_ref(weights, 306), ptr_array_ref(weights, 305),
        ptr_array_ref(weights, 310), ptr_array_ref(weights, 309),
        ptr_array_ref(weights, 312), ptr_array_ref(weights, 311),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 334), ptr_array_ref(weights, 333),
        ptr_array_ref(weights, 336), ptr_array_ref(weights, 335),
        ptr_array_ref(weights, 338), ptr_array_ref(weights, 337),
        ptr_array_ref(weights, 322), 0,
        ptr_array_ref(weights, 319), 0,
        ptr_array_ref(weights, 323), 0,
        ptr_array_ref(weights, 321), ptr_array_ref(weights, 320),
        ptr_array_ref(weights, 327), 0,
        ptr_array_ref(weights, 324), 0,
        ptr_array_ref(weights, 328), 0,
        ptr_array_ref(weights, 326), ptr_array_ref(weights, 325),
        ptr_array_ref(weights, 330), ptr_array_ref(weights, 329),
        ptr_array_ref(weights, 332), ptr_array_ref(weights, 331),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 354), ptr_array_ref(weights, 353),
        ptr_array_ref(weights, 356), ptr_array_ref(weights, 355),
        ptr_array_ref(weights, 358), ptr_array_ref(weights, 357),
        ptr_array_ref(weights, 342), 0,
        ptr_array_ref(weights, 339), 0,
        ptr_array_ref(weights, 343), 0,
        ptr_array_ref(weights, 341), ptr_array_ref(weights, 340),
        ptr_array_ref(weights, 347), 0,
        ptr_array_ref(weights, 344), 0,
        ptr_array_ref(weights, 348), 0,
        ptr_array_ref(weights, 346), ptr_array_ref(weights, 345),
        ptr_array_ref(weights, 350), ptr_array_ref(weights, 349),
        ptr_array_ref(weights, 352), ptr_array_ref(weights, 351),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 158), ptr_array_ref(weights, 157), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _ss_7: ptr = st_clone(h_cur)
    ptr_array_set(_s, 7, _ss_7)

    # input_blocks.8
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 362), ptr_array_ref(weights, 361), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 364), ptr_array_ref(weights, 363), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 360), ptr_array_ref(weights, 359), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 366), ptr_array_ref(weights, 365), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 368), ptr_array_ref(weights, 367), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer input_blocks.8.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 370), ptr_array_ref(weights, 369), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 372), ptr_array_ref(weights, 371), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 390), ptr_array_ref(weights, 389),
        ptr_array_ref(weights, 392), ptr_array_ref(weights, 391),
        ptr_array_ref(weights, 394), ptr_array_ref(weights, 393),
        ptr_array_ref(weights, 378), 0,
        ptr_array_ref(weights, 375), 0,
        ptr_array_ref(weights, 379), 0,
        ptr_array_ref(weights, 377), ptr_array_ref(weights, 376),
        ptr_array_ref(weights, 383), 0,
        ptr_array_ref(weights, 380), 0,
        ptr_array_ref(weights, 384), 0,
        ptr_array_ref(weights, 382), ptr_array_ref(weights, 381),
        ptr_array_ref(weights, 386), ptr_array_ref(weights, 385),
        ptr_array_ref(weights, 388), ptr_array_ref(weights, 387),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 410), ptr_array_ref(weights, 409),
        ptr_array_ref(weights, 412), ptr_array_ref(weights, 411),
        ptr_array_ref(weights, 414), ptr_array_ref(weights, 413),
        ptr_array_ref(weights, 398), 0,
        ptr_array_ref(weights, 395), 0,
        ptr_array_ref(weights, 399), 0,
        ptr_array_ref(weights, 397), ptr_array_ref(weights, 396),
        ptr_array_ref(weights, 403), 0,
        ptr_array_ref(weights, 400), 0,
        ptr_array_ref(weights, 404), 0,
        ptr_array_ref(weights, 402), ptr_array_ref(weights, 401),
        ptr_array_ref(weights, 406), ptr_array_ref(weights, 405),
        ptr_array_ref(weights, 408), ptr_array_ref(weights, 407),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 430), ptr_array_ref(weights, 429),
        ptr_array_ref(weights, 432), ptr_array_ref(weights, 431),
        ptr_array_ref(weights, 434), ptr_array_ref(weights, 433),
        ptr_array_ref(weights, 418), 0,
        ptr_array_ref(weights, 415), 0,
        ptr_array_ref(weights, 419), 0,
        ptr_array_ref(weights, 417), ptr_array_ref(weights, 416),
        ptr_array_ref(weights, 423), 0,
        ptr_array_ref(weights, 420), 0,
        ptr_array_ref(weights, 424), 0,
        ptr_array_ref(weights, 422), ptr_array_ref(weights, 421),
        ptr_array_ref(weights, 426), ptr_array_ref(weights, 425),
        ptr_array_ref(weights, 428), ptr_array_ref(weights, 427),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 450), ptr_array_ref(weights, 449),
        ptr_array_ref(weights, 452), ptr_array_ref(weights, 451),
        ptr_array_ref(weights, 454), ptr_array_ref(weights, 453),
        ptr_array_ref(weights, 438), 0,
        ptr_array_ref(weights, 435), 0,
        ptr_array_ref(weights, 439), 0,
        ptr_array_ref(weights, 437), ptr_array_ref(weights, 436),
        ptr_array_ref(weights, 443), 0,
        ptr_array_ref(weights, 440), 0,
        ptr_array_ref(weights, 444), 0,
        ptr_array_ref(weights, 442), ptr_array_ref(weights, 441),
        ptr_array_ref(weights, 446), ptr_array_ref(weights, 445),
        ptr_array_ref(weights, 448), ptr_array_ref(weights, 447),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 470), ptr_array_ref(weights, 469),
        ptr_array_ref(weights, 472), ptr_array_ref(weights, 471),
        ptr_array_ref(weights, 474), ptr_array_ref(weights, 473),
        ptr_array_ref(weights, 458), 0,
        ptr_array_ref(weights, 455), 0,
        ptr_array_ref(weights, 459), 0,
        ptr_array_ref(weights, 457), ptr_array_ref(weights, 456),
        ptr_array_ref(weights, 463), 0,
        ptr_array_ref(weights, 460), 0,
        ptr_array_ref(weights, 464), 0,
        ptr_array_ref(weights, 462), ptr_array_ref(weights, 461),
        ptr_array_ref(weights, 466), ptr_array_ref(weights, 465),
        ptr_array_ref(weights, 468), ptr_array_ref(weights, 467),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 490), ptr_array_ref(weights, 489),
        ptr_array_ref(weights, 492), ptr_array_ref(weights, 491),
        ptr_array_ref(weights, 494), ptr_array_ref(weights, 493),
        ptr_array_ref(weights, 478), 0,
        ptr_array_ref(weights, 475), 0,
        ptr_array_ref(weights, 479), 0,
        ptr_array_ref(weights, 477), ptr_array_ref(weights, 476),
        ptr_array_ref(weights, 483), 0,
        ptr_array_ref(weights, 480), 0,
        ptr_array_ref(weights, 484), 0,
        ptr_array_ref(weights, 482), ptr_array_ref(weights, 481),
        ptr_array_ref(weights, 486), ptr_array_ref(weights, 485),
        ptr_array_ref(weights, 488), ptr_array_ref(weights, 487),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 510), ptr_array_ref(weights, 509),
        ptr_array_ref(weights, 512), ptr_array_ref(weights, 511),
        ptr_array_ref(weights, 514), ptr_array_ref(weights, 513),
        ptr_array_ref(weights, 498), 0,
        ptr_array_ref(weights, 495), 0,
        ptr_array_ref(weights, 499), 0,
        ptr_array_ref(weights, 497), ptr_array_ref(weights, 496),
        ptr_array_ref(weights, 503), 0,
        ptr_array_ref(weights, 500), 0,
        ptr_array_ref(weights, 504), 0,
        ptr_array_ref(weights, 502), ptr_array_ref(weights, 501),
        ptr_array_ref(weights, 506), ptr_array_ref(weights, 505),
        ptr_array_ref(weights, 508), ptr_array_ref(weights, 507),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 530), ptr_array_ref(weights, 529),
        ptr_array_ref(weights, 532), ptr_array_ref(weights, 531),
        ptr_array_ref(weights, 534), ptr_array_ref(weights, 533),
        ptr_array_ref(weights, 518), 0,
        ptr_array_ref(weights, 515), 0,
        ptr_array_ref(weights, 519), 0,
        ptr_array_ref(weights, 517), ptr_array_ref(weights, 516),
        ptr_array_ref(weights, 523), 0,
        ptr_array_ref(weights, 520), 0,
        ptr_array_ref(weights, 524), 0,
        ptr_array_ref(weights, 522), ptr_array_ref(weights, 521),
        ptr_array_ref(weights, 526), ptr_array_ref(weights, 525),
        ptr_array_ref(weights, 528), ptr_array_ref(weights, 527),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 550), ptr_array_ref(weights, 549),
        ptr_array_ref(weights, 552), ptr_array_ref(weights, 551),
        ptr_array_ref(weights, 554), ptr_array_ref(weights, 553),
        ptr_array_ref(weights, 538), 0,
        ptr_array_ref(weights, 535), 0,
        ptr_array_ref(weights, 539), 0,
        ptr_array_ref(weights, 537), ptr_array_ref(weights, 536),
        ptr_array_ref(weights, 543), 0,
        ptr_array_ref(weights, 540), 0,
        ptr_array_ref(weights, 544), 0,
        ptr_array_ref(weights, 542), ptr_array_ref(weights, 541),
        ptr_array_ref(weights, 546), ptr_array_ref(weights, 545),
        ptr_array_ref(weights, 548), ptr_array_ref(weights, 547),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 570), ptr_array_ref(weights, 569),
        ptr_array_ref(weights, 572), ptr_array_ref(weights, 571),
        ptr_array_ref(weights, 574), ptr_array_ref(weights, 573),
        ptr_array_ref(weights, 558), 0,
        ptr_array_ref(weights, 555), 0,
        ptr_array_ref(weights, 559), 0,
        ptr_array_ref(weights, 557), ptr_array_ref(weights, 556),
        ptr_array_ref(weights, 563), 0,
        ptr_array_ref(weights, 560), 0,
        ptr_array_ref(weights, 564), 0,
        ptr_array_ref(weights, 562), ptr_array_ref(weights, 561),
        ptr_array_ref(weights, 566), ptr_array_ref(weights, 565),
        ptr_array_ref(weights, 568), ptr_array_ref(weights, 567),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 374), ptr_array_ref(weights, 373), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _ss_8: ptr = st_clone(h_cur)
    ptr_array_set(_s, 8, _ss_8)

    # middle_block
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 582), ptr_array_ref(weights, 581), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 584), ptr_array_ref(weights, 583), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 580), ptr_array_ref(weights, 579), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 586), ptr_array_ref(weights, 585), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 588), ptr_array_ref(weights, 587), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer middle_block.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 590), ptr_array_ref(weights, 589), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 592), ptr_array_ref(weights, 591), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 610), ptr_array_ref(weights, 609),
        ptr_array_ref(weights, 612), ptr_array_ref(weights, 611),
        ptr_array_ref(weights, 614), ptr_array_ref(weights, 613),
        ptr_array_ref(weights, 598), 0,
        ptr_array_ref(weights, 595), 0,
        ptr_array_ref(weights, 599), 0,
        ptr_array_ref(weights, 597), ptr_array_ref(weights, 596),
        ptr_array_ref(weights, 603), 0,
        ptr_array_ref(weights, 600), 0,
        ptr_array_ref(weights, 604), 0,
        ptr_array_ref(weights, 602), ptr_array_ref(weights, 601),
        ptr_array_ref(weights, 606), ptr_array_ref(weights, 605),
        ptr_array_ref(weights, 608), ptr_array_ref(weights, 607),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 630), ptr_array_ref(weights, 629),
        ptr_array_ref(weights, 632), ptr_array_ref(weights, 631),
        ptr_array_ref(weights, 634), ptr_array_ref(weights, 633),
        ptr_array_ref(weights, 618), 0,
        ptr_array_ref(weights, 615), 0,
        ptr_array_ref(weights, 619), 0,
        ptr_array_ref(weights, 617), ptr_array_ref(weights, 616),
        ptr_array_ref(weights, 623), 0,
        ptr_array_ref(weights, 620), 0,
        ptr_array_ref(weights, 624), 0,
        ptr_array_ref(weights, 622), ptr_array_ref(weights, 621),
        ptr_array_ref(weights, 626), ptr_array_ref(weights, 625),
        ptr_array_ref(weights, 628), ptr_array_ref(weights, 627),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 650), ptr_array_ref(weights, 649),
        ptr_array_ref(weights, 652), ptr_array_ref(weights, 651),
        ptr_array_ref(weights, 654), ptr_array_ref(weights, 653),
        ptr_array_ref(weights, 638), 0,
        ptr_array_ref(weights, 635), 0,
        ptr_array_ref(weights, 639), 0,
        ptr_array_ref(weights, 637), ptr_array_ref(weights, 636),
        ptr_array_ref(weights, 643), 0,
        ptr_array_ref(weights, 640), 0,
        ptr_array_ref(weights, 644), 0,
        ptr_array_ref(weights, 642), ptr_array_ref(weights, 641),
        ptr_array_ref(weights, 646), ptr_array_ref(weights, 645),
        ptr_array_ref(weights, 648), ptr_array_ref(weights, 647),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 670), ptr_array_ref(weights, 669),
        ptr_array_ref(weights, 672), ptr_array_ref(weights, 671),
        ptr_array_ref(weights, 674), ptr_array_ref(weights, 673),
        ptr_array_ref(weights, 658), 0,
        ptr_array_ref(weights, 655), 0,
        ptr_array_ref(weights, 659), 0,
        ptr_array_ref(weights, 657), ptr_array_ref(weights, 656),
        ptr_array_ref(weights, 663), 0,
        ptr_array_ref(weights, 660), 0,
        ptr_array_ref(weights, 664), 0,
        ptr_array_ref(weights, 662), ptr_array_ref(weights, 661),
        ptr_array_ref(weights, 666), ptr_array_ref(weights, 665),
        ptr_array_ref(weights, 668), ptr_array_ref(weights, 667),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 690), ptr_array_ref(weights, 689),
        ptr_array_ref(weights, 692), ptr_array_ref(weights, 691),
        ptr_array_ref(weights, 694), ptr_array_ref(weights, 693),
        ptr_array_ref(weights, 678), 0,
        ptr_array_ref(weights, 675), 0,
        ptr_array_ref(weights, 679), 0,
        ptr_array_ref(weights, 677), ptr_array_ref(weights, 676),
        ptr_array_ref(weights, 683), 0,
        ptr_array_ref(weights, 680), 0,
        ptr_array_ref(weights, 684), 0,
        ptr_array_ref(weights, 682), ptr_array_ref(weights, 681),
        ptr_array_ref(weights, 686), ptr_array_ref(weights, 685),
        ptr_array_ref(weights, 688), ptr_array_ref(weights, 687),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 710), ptr_array_ref(weights, 709),
        ptr_array_ref(weights, 712), ptr_array_ref(weights, 711),
        ptr_array_ref(weights, 714), ptr_array_ref(weights, 713),
        ptr_array_ref(weights, 698), 0,
        ptr_array_ref(weights, 695), 0,
        ptr_array_ref(weights, 699), 0,
        ptr_array_ref(weights, 697), ptr_array_ref(weights, 696),
        ptr_array_ref(weights, 703), 0,
        ptr_array_ref(weights, 700), 0,
        ptr_array_ref(weights, 704), 0,
        ptr_array_ref(weights, 702), ptr_array_ref(weights, 701),
        ptr_array_ref(weights, 706), ptr_array_ref(weights, 705),
        ptr_array_ref(weights, 708), ptr_array_ref(weights, 707),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 730), ptr_array_ref(weights, 729),
        ptr_array_ref(weights, 732), ptr_array_ref(weights, 731),
        ptr_array_ref(weights, 734), ptr_array_ref(weights, 733),
        ptr_array_ref(weights, 718), 0,
        ptr_array_ref(weights, 715), 0,
        ptr_array_ref(weights, 719), 0,
        ptr_array_ref(weights, 717), ptr_array_ref(weights, 716),
        ptr_array_ref(weights, 723), 0,
        ptr_array_ref(weights, 720), 0,
        ptr_array_ref(weights, 724), 0,
        ptr_array_ref(weights, 722), ptr_array_ref(weights, 721),
        ptr_array_ref(weights, 726), ptr_array_ref(weights, 725),
        ptr_array_ref(weights, 728), ptr_array_ref(weights, 727),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 750), ptr_array_ref(weights, 749),
        ptr_array_ref(weights, 752), ptr_array_ref(weights, 751),
        ptr_array_ref(weights, 754), ptr_array_ref(weights, 753),
        ptr_array_ref(weights, 738), 0,
        ptr_array_ref(weights, 735), 0,
        ptr_array_ref(weights, 739), 0,
        ptr_array_ref(weights, 737), ptr_array_ref(weights, 736),
        ptr_array_ref(weights, 743), 0,
        ptr_array_ref(weights, 740), 0,
        ptr_array_ref(weights, 744), 0,
        ptr_array_ref(weights, 742), ptr_array_ref(weights, 741),
        ptr_array_ref(weights, 746), ptr_array_ref(weights, 745),
        ptr_array_ref(weights, 748), ptr_array_ref(weights, 747),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 770), ptr_array_ref(weights, 769),
        ptr_array_ref(weights, 772), ptr_array_ref(weights, 771),
        ptr_array_ref(weights, 774), ptr_array_ref(weights, 773),
        ptr_array_ref(weights, 758), 0,
        ptr_array_ref(weights, 755), 0,
        ptr_array_ref(weights, 759), 0,
        ptr_array_ref(weights, 757), ptr_array_ref(weights, 756),
        ptr_array_ref(weights, 763), 0,
        ptr_array_ref(weights, 760), 0,
        ptr_array_ref(weights, 764), 0,
        ptr_array_ref(weights, 762), ptr_array_ref(weights, 761),
        ptr_array_ref(weights, 766), ptr_array_ref(weights, 765),
        ptr_array_ref(weights, 768), ptr_array_ref(weights, 767),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 790), ptr_array_ref(weights, 789),
        ptr_array_ref(weights, 792), ptr_array_ref(weights, 791),
        ptr_array_ref(weights, 794), ptr_array_ref(weights, 793),
        ptr_array_ref(weights, 778), 0,
        ptr_array_ref(weights, 775), 0,
        ptr_array_ref(weights, 779), 0,
        ptr_array_ref(weights, 777), ptr_array_ref(weights, 776),
        ptr_array_ref(weights, 783), 0,
        ptr_array_ref(weights, 780), 0,
        ptr_array_ref(weights, 784), 0,
        ptr_array_ref(weights, 782), ptr_array_ref(weights, 781),
        ptr_array_ref(weights, 786), ptr_array_ref(weights, 785),
        ptr_array_ref(weights, 788), ptr_array_ref(weights, 787),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 594), ptr_array_ref(weights, 593), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 798), ptr_array_ref(weights, 797), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 800), ptr_array_ref(weights, 799), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 796), ptr_array_ref(weights, 795), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 802), ptr_array_ref(weights, 801), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 804), ptr_array_ref(weights, 803), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = _h_cur_orig
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)

    # output_blocks.0
    _cur = h_cur
    _skip = ptr_array_ref(_s, 8)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 1280, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 812), ptr_array_ref(weights, 811), 32, 2560, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 814), ptr_array_ref(weights, 813), n, 2560, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 810), ptr_array_ref(weights, 809), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 816), ptr_array_ref(weights, 815), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 818), ptr_array_ref(weights, 817), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 820), ptr_array_ref(weights, 819), n, 2560, 1280, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.0.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 822), ptr_array_ref(weights, 821), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 824), ptr_array_ref(weights, 823), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 842), ptr_array_ref(weights, 841),
        ptr_array_ref(weights, 844), ptr_array_ref(weights, 843),
        ptr_array_ref(weights, 846), ptr_array_ref(weights, 845),
        ptr_array_ref(weights, 830), 0,
        ptr_array_ref(weights, 827), 0,
        ptr_array_ref(weights, 831), 0,
        ptr_array_ref(weights, 829), ptr_array_ref(weights, 828),
        ptr_array_ref(weights, 835), 0,
        ptr_array_ref(weights, 832), 0,
        ptr_array_ref(weights, 836), 0,
        ptr_array_ref(weights, 834), ptr_array_ref(weights, 833),
        ptr_array_ref(weights, 838), ptr_array_ref(weights, 837),
        ptr_array_ref(weights, 840), ptr_array_ref(weights, 839),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 862), ptr_array_ref(weights, 861),
        ptr_array_ref(weights, 864), ptr_array_ref(weights, 863),
        ptr_array_ref(weights, 866), ptr_array_ref(weights, 865),
        ptr_array_ref(weights, 850), 0,
        ptr_array_ref(weights, 847), 0,
        ptr_array_ref(weights, 851), 0,
        ptr_array_ref(weights, 849), ptr_array_ref(weights, 848),
        ptr_array_ref(weights, 855), 0,
        ptr_array_ref(weights, 852), 0,
        ptr_array_ref(weights, 856), 0,
        ptr_array_ref(weights, 854), ptr_array_ref(weights, 853),
        ptr_array_ref(weights, 858), ptr_array_ref(weights, 857),
        ptr_array_ref(weights, 860), ptr_array_ref(weights, 859),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 882), ptr_array_ref(weights, 881),
        ptr_array_ref(weights, 884), ptr_array_ref(weights, 883),
        ptr_array_ref(weights, 886), ptr_array_ref(weights, 885),
        ptr_array_ref(weights, 870), 0,
        ptr_array_ref(weights, 867), 0,
        ptr_array_ref(weights, 871), 0,
        ptr_array_ref(weights, 869), ptr_array_ref(weights, 868),
        ptr_array_ref(weights, 875), 0,
        ptr_array_ref(weights, 872), 0,
        ptr_array_ref(weights, 876), 0,
        ptr_array_ref(weights, 874), ptr_array_ref(weights, 873),
        ptr_array_ref(weights, 878), ptr_array_ref(weights, 877),
        ptr_array_ref(weights, 880), ptr_array_ref(weights, 879),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 902), ptr_array_ref(weights, 901),
        ptr_array_ref(weights, 904), ptr_array_ref(weights, 903),
        ptr_array_ref(weights, 906), ptr_array_ref(weights, 905),
        ptr_array_ref(weights, 890), 0,
        ptr_array_ref(weights, 887), 0,
        ptr_array_ref(weights, 891), 0,
        ptr_array_ref(weights, 889), ptr_array_ref(weights, 888),
        ptr_array_ref(weights, 895), 0,
        ptr_array_ref(weights, 892), 0,
        ptr_array_ref(weights, 896), 0,
        ptr_array_ref(weights, 894), ptr_array_ref(weights, 893),
        ptr_array_ref(weights, 898), ptr_array_ref(weights, 897),
        ptr_array_ref(weights, 900), ptr_array_ref(weights, 899),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 922), ptr_array_ref(weights, 921),
        ptr_array_ref(weights, 924), ptr_array_ref(weights, 923),
        ptr_array_ref(weights, 926), ptr_array_ref(weights, 925),
        ptr_array_ref(weights, 910), 0,
        ptr_array_ref(weights, 907), 0,
        ptr_array_ref(weights, 911), 0,
        ptr_array_ref(weights, 909), ptr_array_ref(weights, 908),
        ptr_array_ref(weights, 915), 0,
        ptr_array_ref(weights, 912), 0,
        ptr_array_ref(weights, 916), 0,
        ptr_array_ref(weights, 914), ptr_array_ref(weights, 913),
        ptr_array_ref(weights, 918), ptr_array_ref(weights, 917),
        ptr_array_ref(weights, 920), ptr_array_ref(weights, 919),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 942), ptr_array_ref(weights, 941),
        ptr_array_ref(weights, 944), ptr_array_ref(weights, 943),
        ptr_array_ref(weights, 946), ptr_array_ref(weights, 945),
        ptr_array_ref(weights, 930), 0,
        ptr_array_ref(weights, 927), 0,
        ptr_array_ref(weights, 931), 0,
        ptr_array_ref(weights, 929), ptr_array_ref(weights, 928),
        ptr_array_ref(weights, 935), 0,
        ptr_array_ref(weights, 932), 0,
        ptr_array_ref(weights, 936), 0,
        ptr_array_ref(weights, 934), ptr_array_ref(weights, 933),
        ptr_array_ref(weights, 938), ptr_array_ref(weights, 937),
        ptr_array_ref(weights, 940), ptr_array_ref(weights, 939),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 962), ptr_array_ref(weights, 961),
        ptr_array_ref(weights, 964), ptr_array_ref(weights, 963),
        ptr_array_ref(weights, 966), ptr_array_ref(weights, 965),
        ptr_array_ref(weights, 950), 0,
        ptr_array_ref(weights, 947), 0,
        ptr_array_ref(weights, 951), 0,
        ptr_array_ref(weights, 949), ptr_array_ref(weights, 948),
        ptr_array_ref(weights, 955), 0,
        ptr_array_ref(weights, 952), 0,
        ptr_array_ref(weights, 956), 0,
        ptr_array_ref(weights, 954), ptr_array_ref(weights, 953),
        ptr_array_ref(weights, 958), ptr_array_ref(weights, 957),
        ptr_array_ref(weights, 960), ptr_array_ref(weights, 959),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 982), ptr_array_ref(weights, 981),
        ptr_array_ref(weights, 984), ptr_array_ref(weights, 983),
        ptr_array_ref(weights, 986), ptr_array_ref(weights, 985),
        ptr_array_ref(weights, 970), 0,
        ptr_array_ref(weights, 967), 0,
        ptr_array_ref(weights, 971), 0,
        ptr_array_ref(weights, 969), ptr_array_ref(weights, 968),
        ptr_array_ref(weights, 975), 0,
        ptr_array_ref(weights, 972), 0,
        ptr_array_ref(weights, 976), 0,
        ptr_array_ref(weights, 974), ptr_array_ref(weights, 973),
        ptr_array_ref(weights, 978), ptr_array_ref(weights, 977),
        ptr_array_ref(weights, 980), ptr_array_ref(weights, 979),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1002), ptr_array_ref(weights, 1001),
        ptr_array_ref(weights, 1004), ptr_array_ref(weights, 1003),
        ptr_array_ref(weights, 1006), ptr_array_ref(weights, 1005),
        ptr_array_ref(weights, 990), 0,
        ptr_array_ref(weights, 987), 0,
        ptr_array_ref(weights, 991), 0,
        ptr_array_ref(weights, 989), ptr_array_ref(weights, 988),
        ptr_array_ref(weights, 995), 0,
        ptr_array_ref(weights, 992), 0,
        ptr_array_ref(weights, 996), 0,
        ptr_array_ref(weights, 994), ptr_array_ref(weights, 993),
        ptr_array_ref(weights, 998), ptr_array_ref(weights, 997),
        ptr_array_ref(weights, 1000), ptr_array_ref(weights, 999),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1022), ptr_array_ref(weights, 1021),
        ptr_array_ref(weights, 1024), ptr_array_ref(weights, 1023),
        ptr_array_ref(weights, 1026), ptr_array_ref(weights, 1025),
        ptr_array_ref(weights, 1010), 0,
        ptr_array_ref(weights, 1007), 0,
        ptr_array_ref(weights, 1011), 0,
        ptr_array_ref(weights, 1009), ptr_array_ref(weights, 1008),
        ptr_array_ref(weights, 1015), 0,
        ptr_array_ref(weights, 1012), 0,
        ptr_array_ref(weights, 1016), 0,
        ptr_array_ref(weights, 1014), ptr_array_ref(weights, 1013),
        ptr_array_ref(weights, 1018), ptr_array_ref(weights, 1017),
        ptr_array_ref(weights, 1020), ptr_array_ref(weights, 1019),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 826), ptr_array_ref(weights, 825), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.1
    _cur = h_cur
    _skip = ptr_array_ref(_s, 7)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 1280, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1030), ptr_array_ref(weights, 1029), 32, 2560, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1032), ptr_array_ref(weights, 1031), n, 2560, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1028), ptr_array_ref(weights, 1027), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1034), ptr_array_ref(weights, 1033), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1036), ptr_array_ref(weights, 1035), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1038), ptr_array_ref(weights, 1037), n, 2560, 1280, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.1.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1040), ptr_array_ref(weights, 1039), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1042), ptr_array_ref(weights, 1041), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1060), ptr_array_ref(weights, 1059),
        ptr_array_ref(weights, 1062), ptr_array_ref(weights, 1061),
        ptr_array_ref(weights, 1064), ptr_array_ref(weights, 1063),
        ptr_array_ref(weights, 1048), 0,
        ptr_array_ref(weights, 1045), 0,
        ptr_array_ref(weights, 1049), 0,
        ptr_array_ref(weights, 1047), ptr_array_ref(weights, 1046),
        ptr_array_ref(weights, 1053), 0,
        ptr_array_ref(weights, 1050), 0,
        ptr_array_ref(weights, 1054), 0,
        ptr_array_ref(weights, 1052), ptr_array_ref(weights, 1051),
        ptr_array_ref(weights, 1056), ptr_array_ref(weights, 1055),
        ptr_array_ref(weights, 1058), ptr_array_ref(weights, 1057),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1080), ptr_array_ref(weights, 1079),
        ptr_array_ref(weights, 1082), ptr_array_ref(weights, 1081),
        ptr_array_ref(weights, 1084), ptr_array_ref(weights, 1083),
        ptr_array_ref(weights, 1068), 0,
        ptr_array_ref(weights, 1065), 0,
        ptr_array_ref(weights, 1069), 0,
        ptr_array_ref(weights, 1067), ptr_array_ref(weights, 1066),
        ptr_array_ref(weights, 1073), 0,
        ptr_array_ref(weights, 1070), 0,
        ptr_array_ref(weights, 1074), 0,
        ptr_array_ref(weights, 1072), ptr_array_ref(weights, 1071),
        ptr_array_ref(weights, 1076), ptr_array_ref(weights, 1075),
        ptr_array_ref(weights, 1078), ptr_array_ref(weights, 1077),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1100), ptr_array_ref(weights, 1099),
        ptr_array_ref(weights, 1102), ptr_array_ref(weights, 1101),
        ptr_array_ref(weights, 1104), ptr_array_ref(weights, 1103),
        ptr_array_ref(weights, 1088), 0,
        ptr_array_ref(weights, 1085), 0,
        ptr_array_ref(weights, 1089), 0,
        ptr_array_ref(weights, 1087), ptr_array_ref(weights, 1086),
        ptr_array_ref(weights, 1093), 0,
        ptr_array_ref(weights, 1090), 0,
        ptr_array_ref(weights, 1094), 0,
        ptr_array_ref(weights, 1092), ptr_array_ref(weights, 1091),
        ptr_array_ref(weights, 1096), ptr_array_ref(weights, 1095),
        ptr_array_ref(weights, 1098), ptr_array_ref(weights, 1097),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1120), ptr_array_ref(weights, 1119),
        ptr_array_ref(weights, 1122), ptr_array_ref(weights, 1121),
        ptr_array_ref(weights, 1124), ptr_array_ref(weights, 1123),
        ptr_array_ref(weights, 1108), 0,
        ptr_array_ref(weights, 1105), 0,
        ptr_array_ref(weights, 1109), 0,
        ptr_array_ref(weights, 1107), ptr_array_ref(weights, 1106),
        ptr_array_ref(weights, 1113), 0,
        ptr_array_ref(weights, 1110), 0,
        ptr_array_ref(weights, 1114), 0,
        ptr_array_ref(weights, 1112), ptr_array_ref(weights, 1111),
        ptr_array_ref(weights, 1116), ptr_array_ref(weights, 1115),
        ptr_array_ref(weights, 1118), ptr_array_ref(weights, 1117),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1140), ptr_array_ref(weights, 1139),
        ptr_array_ref(weights, 1142), ptr_array_ref(weights, 1141),
        ptr_array_ref(weights, 1144), ptr_array_ref(weights, 1143),
        ptr_array_ref(weights, 1128), 0,
        ptr_array_ref(weights, 1125), 0,
        ptr_array_ref(weights, 1129), 0,
        ptr_array_ref(weights, 1127), ptr_array_ref(weights, 1126),
        ptr_array_ref(weights, 1133), 0,
        ptr_array_ref(weights, 1130), 0,
        ptr_array_ref(weights, 1134), 0,
        ptr_array_ref(weights, 1132), ptr_array_ref(weights, 1131),
        ptr_array_ref(weights, 1136), ptr_array_ref(weights, 1135),
        ptr_array_ref(weights, 1138), ptr_array_ref(weights, 1137),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1160), ptr_array_ref(weights, 1159),
        ptr_array_ref(weights, 1162), ptr_array_ref(weights, 1161),
        ptr_array_ref(weights, 1164), ptr_array_ref(weights, 1163),
        ptr_array_ref(weights, 1148), 0,
        ptr_array_ref(weights, 1145), 0,
        ptr_array_ref(weights, 1149), 0,
        ptr_array_ref(weights, 1147), ptr_array_ref(weights, 1146),
        ptr_array_ref(weights, 1153), 0,
        ptr_array_ref(weights, 1150), 0,
        ptr_array_ref(weights, 1154), 0,
        ptr_array_ref(weights, 1152), ptr_array_ref(weights, 1151),
        ptr_array_ref(weights, 1156), ptr_array_ref(weights, 1155),
        ptr_array_ref(weights, 1158), ptr_array_ref(weights, 1157),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1180), ptr_array_ref(weights, 1179),
        ptr_array_ref(weights, 1182), ptr_array_ref(weights, 1181),
        ptr_array_ref(weights, 1184), ptr_array_ref(weights, 1183),
        ptr_array_ref(weights, 1168), 0,
        ptr_array_ref(weights, 1165), 0,
        ptr_array_ref(weights, 1169), 0,
        ptr_array_ref(weights, 1167), ptr_array_ref(weights, 1166),
        ptr_array_ref(weights, 1173), 0,
        ptr_array_ref(weights, 1170), 0,
        ptr_array_ref(weights, 1174), 0,
        ptr_array_ref(weights, 1172), ptr_array_ref(weights, 1171),
        ptr_array_ref(weights, 1176), ptr_array_ref(weights, 1175),
        ptr_array_ref(weights, 1178), ptr_array_ref(weights, 1177),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1200), ptr_array_ref(weights, 1199),
        ptr_array_ref(weights, 1202), ptr_array_ref(weights, 1201),
        ptr_array_ref(weights, 1204), ptr_array_ref(weights, 1203),
        ptr_array_ref(weights, 1188), 0,
        ptr_array_ref(weights, 1185), 0,
        ptr_array_ref(weights, 1189), 0,
        ptr_array_ref(weights, 1187), ptr_array_ref(weights, 1186),
        ptr_array_ref(weights, 1193), 0,
        ptr_array_ref(weights, 1190), 0,
        ptr_array_ref(weights, 1194), 0,
        ptr_array_ref(weights, 1192), ptr_array_ref(weights, 1191),
        ptr_array_ref(weights, 1196), ptr_array_ref(weights, 1195),
        ptr_array_ref(weights, 1198), ptr_array_ref(weights, 1197),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1220), ptr_array_ref(weights, 1219),
        ptr_array_ref(weights, 1222), ptr_array_ref(weights, 1221),
        ptr_array_ref(weights, 1224), ptr_array_ref(weights, 1223),
        ptr_array_ref(weights, 1208), 0,
        ptr_array_ref(weights, 1205), 0,
        ptr_array_ref(weights, 1209), 0,
        ptr_array_ref(weights, 1207), ptr_array_ref(weights, 1206),
        ptr_array_ref(weights, 1213), 0,
        ptr_array_ref(weights, 1210), 0,
        ptr_array_ref(weights, 1214), 0,
        ptr_array_ref(weights, 1212), ptr_array_ref(weights, 1211),
        ptr_array_ref(weights, 1216), ptr_array_ref(weights, 1215),
        ptr_array_ref(weights, 1218), ptr_array_ref(weights, 1217),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1240), ptr_array_ref(weights, 1239),
        ptr_array_ref(weights, 1242), ptr_array_ref(weights, 1241),
        ptr_array_ref(weights, 1244), ptr_array_ref(weights, 1243),
        ptr_array_ref(weights, 1228), 0,
        ptr_array_ref(weights, 1225), 0,
        ptr_array_ref(weights, 1229), 0,
        ptr_array_ref(weights, 1227), ptr_array_ref(weights, 1226),
        ptr_array_ref(weights, 1233), 0,
        ptr_array_ref(weights, 1230), 0,
        ptr_array_ref(weights, 1234), 0,
        ptr_array_ref(weights, 1232), ptr_array_ref(weights, 1231),
        ptr_array_ref(weights, 1236), ptr_array_ref(weights, 1235),
        ptr_array_ref(weights, 1238), ptr_array_ref(weights, 1237),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1044), ptr_array_ref(weights, 1043), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.2
    _cur = h_cur
    _skip = ptr_array_ref(_s, 6)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 640, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1248), ptr_array_ref(weights, 1247), 32, 1920, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1250), ptr_array_ref(weights, 1249), n, 1920, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1246), ptr_array_ref(weights, 1245), n, 1280, 1280)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 1280)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1252), ptr_array_ref(weights, 1251), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1254), ptr_array_ref(weights, 1253), n, 1280, 1280, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1256), ptr_array_ref(weights, 1255), n, 1920, 1280, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.2.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1258), ptr_array_ref(weights, 1257), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 1280, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1260), ptr_array_ref(weights, 1259), n*hh*ww, 1280, 1280)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1278), ptr_array_ref(weights, 1277),
        ptr_array_ref(weights, 1280), ptr_array_ref(weights, 1279),
        ptr_array_ref(weights, 1282), ptr_array_ref(weights, 1281),
        ptr_array_ref(weights, 1266), 0,
        ptr_array_ref(weights, 1263), 0,
        ptr_array_ref(weights, 1267), 0,
        ptr_array_ref(weights, 1265), ptr_array_ref(weights, 1264),
        ptr_array_ref(weights, 1271), 0,
        ptr_array_ref(weights, 1268), 0,
        ptr_array_ref(weights, 1272), 0,
        ptr_array_ref(weights, 1270), ptr_array_ref(weights, 1269),
        ptr_array_ref(weights, 1274), ptr_array_ref(weights, 1273),
        ptr_array_ref(weights, 1276), ptr_array_ref(weights, 1275),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1298), ptr_array_ref(weights, 1297),
        ptr_array_ref(weights, 1300), ptr_array_ref(weights, 1299),
        ptr_array_ref(weights, 1302), ptr_array_ref(weights, 1301),
        ptr_array_ref(weights, 1286), 0,
        ptr_array_ref(weights, 1283), 0,
        ptr_array_ref(weights, 1287), 0,
        ptr_array_ref(weights, 1285), ptr_array_ref(weights, 1284),
        ptr_array_ref(weights, 1291), 0,
        ptr_array_ref(weights, 1288), 0,
        ptr_array_ref(weights, 1292), 0,
        ptr_array_ref(weights, 1290), ptr_array_ref(weights, 1289),
        ptr_array_ref(weights, 1294), ptr_array_ref(weights, 1293),
        ptr_array_ref(weights, 1296), ptr_array_ref(weights, 1295),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1318), ptr_array_ref(weights, 1317),
        ptr_array_ref(weights, 1320), ptr_array_ref(weights, 1319),
        ptr_array_ref(weights, 1322), ptr_array_ref(weights, 1321),
        ptr_array_ref(weights, 1306), 0,
        ptr_array_ref(weights, 1303), 0,
        ptr_array_ref(weights, 1307), 0,
        ptr_array_ref(weights, 1305), ptr_array_ref(weights, 1304),
        ptr_array_ref(weights, 1311), 0,
        ptr_array_ref(weights, 1308), 0,
        ptr_array_ref(weights, 1312), 0,
        ptr_array_ref(weights, 1310), ptr_array_ref(weights, 1309),
        ptr_array_ref(weights, 1314), ptr_array_ref(weights, 1313),
        ptr_array_ref(weights, 1316), ptr_array_ref(weights, 1315),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1338), ptr_array_ref(weights, 1337),
        ptr_array_ref(weights, 1340), ptr_array_ref(weights, 1339),
        ptr_array_ref(weights, 1342), ptr_array_ref(weights, 1341),
        ptr_array_ref(weights, 1326), 0,
        ptr_array_ref(weights, 1323), 0,
        ptr_array_ref(weights, 1327), 0,
        ptr_array_ref(weights, 1325), ptr_array_ref(weights, 1324),
        ptr_array_ref(weights, 1331), 0,
        ptr_array_ref(weights, 1328), 0,
        ptr_array_ref(weights, 1332), 0,
        ptr_array_ref(weights, 1330), ptr_array_ref(weights, 1329),
        ptr_array_ref(weights, 1334), ptr_array_ref(weights, 1333),
        ptr_array_ref(weights, 1336), ptr_array_ref(weights, 1335),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1358), ptr_array_ref(weights, 1357),
        ptr_array_ref(weights, 1360), ptr_array_ref(weights, 1359),
        ptr_array_ref(weights, 1362), ptr_array_ref(weights, 1361),
        ptr_array_ref(weights, 1346), 0,
        ptr_array_ref(weights, 1343), 0,
        ptr_array_ref(weights, 1347), 0,
        ptr_array_ref(weights, 1345), ptr_array_ref(weights, 1344),
        ptr_array_ref(weights, 1351), 0,
        ptr_array_ref(weights, 1348), 0,
        ptr_array_ref(weights, 1352), 0,
        ptr_array_ref(weights, 1350), ptr_array_ref(weights, 1349),
        ptr_array_ref(weights, 1354), ptr_array_ref(weights, 1353),
        ptr_array_ref(weights, 1356), ptr_array_ref(weights, 1355),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1378), ptr_array_ref(weights, 1377),
        ptr_array_ref(weights, 1380), ptr_array_ref(weights, 1379),
        ptr_array_ref(weights, 1382), ptr_array_ref(weights, 1381),
        ptr_array_ref(weights, 1366), 0,
        ptr_array_ref(weights, 1363), 0,
        ptr_array_ref(weights, 1367), 0,
        ptr_array_ref(weights, 1365), ptr_array_ref(weights, 1364),
        ptr_array_ref(weights, 1371), 0,
        ptr_array_ref(weights, 1368), 0,
        ptr_array_ref(weights, 1372), 0,
        ptr_array_ref(weights, 1370), ptr_array_ref(weights, 1369),
        ptr_array_ref(weights, 1374), ptr_array_ref(weights, 1373),
        ptr_array_ref(weights, 1376), ptr_array_ref(weights, 1375),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1398), ptr_array_ref(weights, 1397),
        ptr_array_ref(weights, 1400), ptr_array_ref(weights, 1399),
        ptr_array_ref(weights, 1402), ptr_array_ref(weights, 1401),
        ptr_array_ref(weights, 1386), 0,
        ptr_array_ref(weights, 1383), 0,
        ptr_array_ref(weights, 1387), 0,
        ptr_array_ref(weights, 1385), ptr_array_ref(weights, 1384),
        ptr_array_ref(weights, 1391), 0,
        ptr_array_ref(weights, 1388), 0,
        ptr_array_ref(weights, 1392), 0,
        ptr_array_ref(weights, 1390), ptr_array_ref(weights, 1389),
        ptr_array_ref(weights, 1394), ptr_array_ref(weights, 1393),
        ptr_array_ref(weights, 1396), ptr_array_ref(weights, 1395),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1418), ptr_array_ref(weights, 1417),
        ptr_array_ref(weights, 1420), ptr_array_ref(weights, 1419),
        ptr_array_ref(weights, 1422), ptr_array_ref(weights, 1421),
        ptr_array_ref(weights, 1406), 0,
        ptr_array_ref(weights, 1403), 0,
        ptr_array_ref(weights, 1407), 0,
        ptr_array_ref(weights, 1405), ptr_array_ref(weights, 1404),
        ptr_array_ref(weights, 1411), 0,
        ptr_array_ref(weights, 1408), 0,
        ptr_array_ref(weights, 1412), 0,
        ptr_array_ref(weights, 1410), ptr_array_ref(weights, 1409),
        ptr_array_ref(weights, 1414), ptr_array_ref(weights, 1413),
        ptr_array_ref(weights, 1416), ptr_array_ref(weights, 1415),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1438), ptr_array_ref(weights, 1437),
        ptr_array_ref(weights, 1440), ptr_array_ref(weights, 1439),
        ptr_array_ref(weights, 1442), ptr_array_ref(weights, 1441),
        ptr_array_ref(weights, 1426), 0,
        ptr_array_ref(weights, 1423), 0,
        ptr_array_ref(weights, 1427), 0,
        ptr_array_ref(weights, 1425), ptr_array_ref(weights, 1424),
        ptr_array_ref(weights, 1431), 0,
        ptr_array_ref(weights, 1428), 0,
        ptr_array_ref(weights, 1432), 0,
        ptr_array_ref(weights, 1430), ptr_array_ref(weights, 1429),
        ptr_array_ref(weights, 1434), ptr_array_ref(weights, 1433),
        ptr_array_ref(weights, 1436), ptr_array_ref(weights, 1435),
        n, hh*ww, 1280, 20, 5120)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1458), ptr_array_ref(weights, 1457),
        ptr_array_ref(weights, 1460), ptr_array_ref(weights, 1459),
        ptr_array_ref(weights, 1462), ptr_array_ref(weights, 1461),
        ptr_array_ref(weights, 1446), 0,
        ptr_array_ref(weights, 1443), 0,
        ptr_array_ref(weights, 1447), 0,
        ptr_array_ref(weights, 1445), ptr_array_ref(weights, 1444),
        ptr_array_ref(weights, 1451), 0,
        ptr_array_ref(weights, 1448), 0,
        ptr_array_ref(weights, 1452), 0,
        ptr_array_ref(weights, 1450), ptr_array_ref(weights, 1449),
        ptr_array_ref(weights, 1454), ptr_array_ref(weights, 1453),
        ptr_array_ref(weights, 1456), ptr_array_ref(weights, 1455),
        n, hh*ww, 1280, 20, 5120)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1262), ptr_array_ref(weights, 1261), n*hh*ww, 1280, 1280)
    _h_img = reshape_nlc_to_nchw(_seq, n, 1280, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _h_old = h_cur
    h_cur = upsample_nearest_torch(h_cur, 2)
    st_tensor_free(_h_old)
    hh = hh*2; ww = ww*2
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1464), ptr_array_ref(weights, 1463), n, 1280, 1280, hh, ww, 3, 1, 1)
    st_tensor_free(_h_old)

    # output_blocks.3
    _cur = h_cur
    _skip = ptr_array_ref(_s, 5)
    h_cur = cat_channel_tensors(_cur, _skip, n, 1280, 640, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1468), ptr_array_ref(weights, 1467), 32, 1920, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1470), ptr_array_ref(weights, 1469), n, 1920, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1466), ptr_array_ref(weights, 1465), n, 1280, 640)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1472), ptr_array_ref(weights, 1471), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1474), ptr_array_ref(weights, 1473), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1476), ptr_array_ref(weights, 1475), n, 1920, 640, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.3.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1478), ptr_array_ref(weights, 1477), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1480), ptr_array_ref(weights, 1479), n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1498), ptr_array_ref(weights, 1497),
        ptr_array_ref(weights, 1500), ptr_array_ref(weights, 1499),
        ptr_array_ref(weights, 1502), ptr_array_ref(weights, 1501),
        ptr_array_ref(weights, 1486), 0,
        ptr_array_ref(weights, 1483), 0,
        ptr_array_ref(weights, 1487), 0,
        ptr_array_ref(weights, 1485), ptr_array_ref(weights, 1484),
        ptr_array_ref(weights, 1491), 0,
        ptr_array_ref(weights, 1488), 0,
        ptr_array_ref(weights, 1492), 0,
        ptr_array_ref(weights, 1490), ptr_array_ref(weights, 1489),
        ptr_array_ref(weights, 1494), ptr_array_ref(weights, 1493),
        ptr_array_ref(weights, 1496), ptr_array_ref(weights, 1495),
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1518), ptr_array_ref(weights, 1517),
        ptr_array_ref(weights, 1520), ptr_array_ref(weights, 1519),
        ptr_array_ref(weights, 1522), ptr_array_ref(weights, 1521),
        ptr_array_ref(weights, 1506), 0,
        ptr_array_ref(weights, 1503), 0,
        ptr_array_ref(weights, 1507), 0,
        ptr_array_ref(weights, 1505), ptr_array_ref(weights, 1504),
        ptr_array_ref(weights, 1511), 0,
        ptr_array_ref(weights, 1508), 0,
        ptr_array_ref(weights, 1512), 0,
        ptr_array_ref(weights, 1510), ptr_array_ref(weights, 1509),
        ptr_array_ref(weights, 1514), ptr_array_ref(weights, 1513),
        ptr_array_ref(weights, 1516), ptr_array_ref(weights, 1515),
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1482), ptr_array_ref(weights, 1481), n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.4
    _cur = h_cur
    _skip = ptr_array_ref(_s, 4)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 640, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1526), ptr_array_ref(weights, 1525), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1528), ptr_array_ref(weights, 1527), n, 1280, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1524), ptr_array_ref(weights, 1523), n, 1280, 640)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1530), ptr_array_ref(weights, 1529), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1532), ptr_array_ref(weights, 1531), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1534), ptr_array_ref(weights, 1533), n, 1280, 640, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.4.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1536), ptr_array_ref(weights, 1535), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1538), ptr_array_ref(weights, 1537), n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1556), ptr_array_ref(weights, 1555),
        ptr_array_ref(weights, 1558), ptr_array_ref(weights, 1557),
        ptr_array_ref(weights, 1560), ptr_array_ref(weights, 1559),
        ptr_array_ref(weights, 1544), 0,
        ptr_array_ref(weights, 1541), 0,
        ptr_array_ref(weights, 1545), 0,
        ptr_array_ref(weights, 1543), ptr_array_ref(weights, 1542),
        ptr_array_ref(weights, 1549), 0,
        ptr_array_ref(weights, 1546), 0,
        ptr_array_ref(weights, 1550), 0,
        ptr_array_ref(weights, 1548), ptr_array_ref(weights, 1547),
        ptr_array_ref(weights, 1552), ptr_array_ref(weights, 1551),
        ptr_array_ref(weights, 1554), ptr_array_ref(weights, 1553),
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1576), ptr_array_ref(weights, 1575),
        ptr_array_ref(weights, 1578), ptr_array_ref(weights, 1577),
        ptr_array_ref(weights, 1580), ptr_array_ref(weights, 1579),
        ptr_array_ref(weights, 1564), 0,
        ptr_array_ref(weights, 1561), 0,
        ptr_array_ref(weights, 1565), 0,
        ptr_array_ref(weights, 1563), ptr_array_ref(weights, 1562),
        ptr_array_ref(weights, 1569), 0,
        ptr_array_ref(weights, 1566), 0,
        ptr_array_ref(weights, 1570), 0,
        ptr_array_ref(weights, 1568), ptr_array_ref(weights, 1567),
        ptr_array_ref(weights, 1572), ptr_array_ref(weights, 1571),
        ptr_array_ref(weights, 1574), ptr_array_ref(weights, 1573),
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1540), ptr_array_ref(weights, 1539), n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)

    # output_blocks.5
    _cur = h_cur
    _skip = ptr_array_ref(_s, 3)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 640, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1584), ptr_array_ref(weights, 1583), 32, 1280, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1586), ptr_array_ref(weights, 1585), n, 1280, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1582), ptr_array_ref(weights, 1581), n, 1280, 640)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 640)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1588), ptr_array_ref(weights, 1587), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1590), ptr_array_ref(weights, 1589), n, 640, 640, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1592), ptr_array_ref(weights, 1591), n, 1280, 640, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)
    # SpatialTransformer output_blocks.5.1
    _x_in = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1594), ptr_array_ref(weights, 1593), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _seq = reshape_nchw_to_nlc(h_cur, n, 640, hh, ww)
    st_tensor_free(h_cur)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1596), ptr_array_ref(weights, 1595), n*hh*ww, 640, 640)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1614), ptr_array_ref(weights, 1613),
        ptr_array_ref(weights, 1616), ptr_array_ref(weights, 1615),
        ptr_array_ref(weights, 1618), ptr_array_ref(weights, 1617),
        ptr_array_ref(weights, 1602), 0,
        ptr_array_ref(weights, 1599), 0,
        ptr_array_ref(weights, 1603), 0,
        ptr_array_ref(weights, 1601), ptr_array_ref(weights, 1600),
        ptr_array_ref(weights, 1607), 0,
        ptr_array_ref(weights, 1604), 0,
        ptr_array_ref(weights, 1608), 0,
        ptr_array_ref(weights, 1606), ptr_array_ref(weights, 1605),
        ptr_array_ref(weights, 1610), ptr_array_ref(weights, 1609),
        ptr_array_ref(weights, 1612), ptr_array_ref(weights, 1611),
        n, hh*ww, 640, 10, 2560)
    _seq = spatial_transformer_block(_seq, context,
        ptr_array_ref(weights, 1634), ptr_array_ref(weights, 1633),
        ptr_array_ref(weights, 1636), ptr_array_ref(weights, 1635),
        ptr_array_ref(weights, 1638), ptr_array_ref(weights, 1637),
        ptr_array_ref(weights, 1622), 0,
        ptr_array_ref(weights, 1619), 0,
        ptr_array_ref(weights, 1623), 0,
        ptr_array_ref(weights, 1621), ptr_array_ref(weights, 1620),
        ptr_array_ref(weights, 1627), 0,
        ptr_array_ref(weights, 1624), 0,
        ptr_array_ref(weights, 1628), 0,
        ptr_array_ref(weights, 1626), ptr_array_ref(weights, 1625),
        ptr_array_ref(weights, 1630), ptr_array_ref(weights, 1629),
        ptr_array_ref(weights, 1632), ptr_array_ref(weights, 1631),
        n, hh*ww, 640, 10, 2560)
    _seq = linear_torch(_seq, ptr_array_ref(weights, 1598), ptr_array_ref(weights, 1597), n*hh*ww, 640, 640)
    _h_img = reshape_nlc_to_nchw(_seq, n, 640, hh, ww)
    st_tensor_free(_seq)
    h_cur = add_tensor(_x_in, _h_img)
    st_tensor_free(_x_in); st_tensor_free(_h_img)
    _h_old = h_cur
    h_cur = upsample_nearest_torch(h_cur, 2)
    st_tensor_free(_h_old)
    hh = hh*2; ww = ww*2
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1640), ptr_array_ref(weights, 1639), n, 640, 640, hh, ww, 3, 1, 1)
    st_tensor_free(_h_old)

    # output_blocks.6
    _cur = h_cur
    _skip = ptr_array_ref(_s, 2)
    h_cur = cat_channel_tensors(_cur, _skip, n, 640, 320, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1644), ptr_array_ref(weights, 1643), 32, 960, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1646), ptr_array_ref(weights, 1645), n, 960, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1642), ptr_array_ref(weights, 1641), n, 1280, 320)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1648), ptr_array_ref(weights, 1647), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1650), ptr_array_ref(weights, 1649), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1652), ptr_array_ref(weights, 1651), n, 960, 320, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)

    # output_blocks.7
    _cur = h_cur
    _skip = ptr_array_ref(_s, 1)
    h_cur = cat_channel_tensors(_cur, _skip, n, 320, 320, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1656), ptr_array_ref(weights, 1655), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1658), ptr_array_ref(weights, 1657), n, 640, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1654), ptr_array_ref(weights, 1653), n, 1280, 320)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1660), ptr_array_ref(weights, 1659), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1662), ptr_array_ref(weights, 1661), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1664), ptr_array_ref(weights, 1663), n, 640, 320, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)

    # output_blocks.8
    _cur = h_cur
    _skip = ptr_array_ref(_s, 0)
    h_cur = cat_channel_tensors(_cur, _skip, n, 320, 320, hh, ww)
    st_tensor_free(_cur); st_tensor_free(_skip)
    _h_cur_orig = st_clone(h_cur)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1668), ptr_array_ref(weights, 1667), 32, 640, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1670), ptr_array_ref(weights, 1669), n, 640, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _se = silu_torch(emb)
    _y = linear_torch(_se, ptr_array_ref(weights, 1666), ptr_array_ref(weights, 1665), n, 1280, 320)
    _h_old = h_cur
    h_cur = add_time_emb_tensor(h_cur, _y, n, 320)
    st_tensor_free(_h_old)
    st_tensor_free(_se); st_tensor_free(_y)
    _h_old = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 1672), ptr_array_ref(weights, 1671), 32, 320, hh, ww)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_old)
    _h_old = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 1674), ptr_array_ref(weights, 1673), n, 320, 320, hh, ww, 3, 1, 3//2)
    st_tensor_free(_h_old)
    _sk = conv2d_torch(_h_cur_orig, ptr_array_ref(weights, 1676), ptr_array_ref(weights, 1675), n, 640, 320, hh, ww, 1, 1, 1//2)
    st_tensor_free(_h_cur_orig)
    _h_old = h_cur
    h_cur = add_tensor(h_cur, _sk)
    st_tensor_free(_h_old)
    st_tensor_free(_sk)

    _h_out = h_cur
    h_cur = group_norm_torch(h_cur, ptr_array_ref(weights, 806), ptr_array_ref(weights, 805), 32, 320, hh, ww)
    st_tensor_free(_h_out)
    _h_out = h_cur
    h_cur = silu_torch(h_cur)
    st_tensor_free(_h_out)
    _h_out = h_cur
    h_cur = conv2d_torch(h_cur, ptr_array_ref(weights, 808), ptr_array_ref(weights, 807), n, 320, 4, hh, ww, 3, 1, 1)
    st_tensor_free(_h_out)
    result: ptr = h_cur
    return result
