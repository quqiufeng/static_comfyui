# transformer_ops.static.py — SpatialTransformer / BasicTransformerBlock（GPU ptr 版本）
# 所有张量都是 torch GPU Tensor 指针（ptr）

# x: [N, C, H, W] GPU tensor
# 返回 [N, H*W, C]，数值顺序与 PyTorch x.permute(0,2,3,1).reshape(N, H*W, C) 一致
def reshape_nchw_to_nlc(x: ptr, n: int, c: int, h: int, w: int) -> ptr:
    x = st_transpose(x, 1, 2)  # [N, H, C, W]
    x = st_transpose(x, 2, 3)  # [N, H, W, C]
    _dims: list[int] = make_int_array(3)
    int_array_set(_dims, 0, n)
    int_array_set(_dims, 1, h * w)
    int_array_set(_dims, 2, c)
    return st_reshape(x, _dims, 3)

# x: [N, H*W, C], 输出 [N, C, H, W]，数值顺序与 PyTorch x.reshape(N, H, W, C).permute(0,3,1,2) 一致
def reshape_nlc_to_nchw(x: ptr, n: int, c: int, h: int, w: int) -> ptr:
    _dims: list[int] = make_int_array(4)
    int_array_set(_dims, 0, n)
    int_array_set(_dims, 1, h)
    int_array_set(_dims, 2, w)
    int_array_set(_dims, 3, c)
    x = st_reshape(x, _dims, 4)  # [N, H, W, C]
    x = st_transpose(x, 2, 3)    # [N, H, C, W]
    x = st_transpose(x, 1, 2)    # [N, C, H, W]
    return x

def attn_self(x: ptr,
              to_q_w: ptr, to_q_b: ptr,
              to_k_w: ptr, to_k_b: ptr,
              to_v_w: ptr, to_v_b: ptr,
              to_out_w: ptr, to_out_b: ptr,
              batch: int, tokens: int, dim: int, heads: int) -> ptr:
    q: ptr = st_linear(x, to_q_w, to_q_b)
    k: ptr = st_linear(x, to_k_w, to_k_b)
    v: ptr = st_linear(x, to_v_w, to_v_b)
    o: ptr = attention_torch(q, k, v, batch, tokens, tokens, dim, heads)
    r: ptr = st_linear(o, to_out_w, to_out_b)
    st_tensor_free(q); st_tensor_free(k); st_tensor_free(v); st_tensor_free(o)
    return r

def attn_cross(x: ptr, ctx: ptr,
               to_q_w: ptr, to_q_b: ptr,
               to_k_w: ptr, to_k_b: ptr,
               to_v_w: ptr, to_v_b: ptr,
               to_out_w: ptr, to_out_b: ptr,
               batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> ptr:
    q: ptr = st_linear(x, to_q_w, to_q_b)
    k: ptr = st_linear(ctx, to_k_w, to_k_b)
    v: ptr = st_linear(ctx, to_v_w, to_v_b)
    o: ptr = attention_torch(q, k, v, batch, tokens_q, tokens_k, dim, heads)
    r: ptr = st_linear(o, to_out_w, to_out_b)
    st_tensor_free(q); st_tensor_free(k); st_tensor_free(v); st_tensor_free(o)
    return r

def ff_geglu(x: ptr,
             w0: ptr, b0: ptr,
             w2: ptr, b2: ptr,
             rows: int, dim: int, hidden: int) -> ptr:
    tmp: ptr = st_linear(x, w0, b0)
    a: ptr = st_slice(tmp, -1, 0, hidden)
    g: ptr = st_slice(tmp, -1, hidden, hidden * 2)
    gg: ptr = st_gelu(g)
    ag: ptr = st_mul_tensor(a, gg)
    r: ptr = st_linear(ag, w2, b2)
    st_tensor_free(tmp); st_tensor_free(a); st_tensor_free(g); st_tensor_free(gg); st_tensor_free(ag)
    return r

def spatial_transformer_block(x: ptr, ctx: ptr,
                              norm1_w: ptr, norm1_b: ptr,
                              norm2_w: ptr, norm2_b: ptr,
                              norm3_w: ptr, norm3_b: ptr,
                              to_q1_w: ptr, to_q1_b: ptr,
                              to_k1_w: ptr, to_k1_b: ptr,
                              to_v1_w: ptr, to_v1_b: ptr,
                              to_out1_w: ptr, to_out1_b: ptr,
                              to_q2_w: ptr, to_q2_b: ptr,
                              to_k2_w: ptr, to_k2_b: ptr,
                              to_v2_w: ptr, to_v2_b: ptr,
                              to_out2_w: ptr, to_out2_b: ptr,
                              ff0_w: ptr, ff0_b: ptr,
                              ff2_w: ptr, ff2_b: ptr,
                              batch: int, tokens: int, dim: int, heads: int, hidden: int) -> ptr:
    _n1: ptr = st_layer_norm(x, norm1_w, norm1_b, 1e-5)
    o1: ptr = attn_self(_n1, to_q1_w, to_q1_b, to_k1_w, to_k1_b, to_v1_w, to_v1_b, to_out1_w, to_out1_b, batch, tokens, dim, heads)
    r1: ptr = st_add_tensor(x, o1)
    st_tensor_free(_n1); st_tensor_free(o1); st_tensor_free(x)

    _n2: ptr = st_layer_norm(r1, norm2_w, norm2_b, 1e-5)
    o2: ptr = attn_cross(_n2, ctx, to_q2_w, to_q2_b, to_k2_w, to_k2_b, to_v2_w, to_v2_b, to_out2_w, to_out2_b, batch, tokens, 77, dim, heads)
    r2: ptr = st_add_tensor(r1, o2)
    st_tensor_free(_n2); st_tensor_free(o2); st_tensor_free(r1)

    _n3: ptr = st_layer_norm(r2, norm3_w, norm3_b, 1e-5)
    o3: ptr = ff_geglu(_n3, ff0_w, ff0_b, ff2_w, ff2_b, batch * tokens, dim, hidden)
    r3: ptr = st_add_tensor(r2, o3)
    st_tensor_free(_n3); st_tensor_free(o3); st_tensor_free(r2)
    return r3

# attention: q,k,v 都是 [batch*tokens, dim]
# SDPA 版本（显存友好，数值可能与 ComfyUI manual attention 有差异）
def attention_torch(q: ptr, k: ptr, v: ptr, batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> ptr:
    dim_head: int = dim // heads

    _dims4: list[int] = make_int_array(4)
    int_array_set(_dims4, 0, batch)
    int_array_set(_dims4, 1, tokens_q)
    int_array_set(_dims4, 2, heads)
    int_array_set(_dims4, 3, dim_head)
    q4: ptr = st_reshape(q, _dims4, 4)
    q4 = st_transpose(q4, 1, 2)  # [batch, heads, tokens_q, dim_head]

    int_array_set(_dims4, 1, tokens_k)
    k4: ptr = st_reshape(k, _dims4, 4)
    k4 = st_transpose(k4, 1, 2)  # [batch, heads, tokens_k, dim_head]

    v4: ptr = st_reshape(v, _dims4, 4)
    v4 = st_transpose(v4, 1, 2)  # [batch, heads, tokens_k, dim_head]

    out4: ptr = st_scaled_dot_product_attention(q4, k4, v4, 0)
    st_tensor_free(q4); st_tensor_free(k4); st_tensor_free(v4)

    out4 = st_transpose(out4, 1, 2)  # [batch, tokens_q, heads, dim_head]
    int_array_set(_dims4, 1, tokens_q)
    int_array_set(_dims4, 2, heads)
    out: ptr = st_reshape(out4, _dims4, 4)
    st_tensor_free(out4)

    _dims2: list[int] = make_int_array(2)
    int_array_set(_dims2, 0, batch * tokens_q)
    int_array_set(_dims2, 1, dim)
    out2: ptr = st_reshape(out, _dims2, 2)
    st_tensor_free(out)
    return out2

# Manual attention 版本，与 ComfyUI 源码 q @ k^T / softmax / attn @ v 完全一致
# 用于数值对齐验证
def attention_torch_manual(q: ptr, k: ptr, v: ptr, batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int) -> ptr:
    dim_head: int = dim // heads
    scale: float = 1.0 / sqrt(float(dim_head))

    # 将 2D [batch*tokens, dim] 按 heads 拆分：
    # 参考 PyTorch: x.reshape(batch, tokens, heads, dim_head).permute(0,2,1,3).reshape(batch*heads, tokens, dim_head)
    _dims4: list[int] = make_int_array(4)
    int_array_set(_dims4, 0, batch)
    int_array_set(_dims4, 1, tokens_q)
    int_array_set(_dims4, 2, heads)
    int_array_set(_dims4, 3, dim_head)
    q4: ptr = st_reshape(q, _dims4, 4)
    q4 = st_transpose(q4, 1, 2)  # [batch, heads, tokens_q, dim_head]

    int_array_set(_dims4, 1, tokens_k)
    k4: ptr = st_reshape(k, _dims4, 4)
    k4 = st_transpose(k4, 1, 2)  # [batch, heads, tokens_k, dim_head]

    v4: ptr = st_reshape(v, _dims4, 4)
    v4 = st_transpose(v4, 1, 2)  # [batch, heads, tokens_k, dim_head]

    # 统一 reshape为 3D [batch*heads, tokens, dim_head]
    _dims3: list[int] = make_int_array(3)
    int_array_set(_dims3, 0, batch * heads)
    int_array_set(_dims3, 1, tokens_q)
    int_array_set(_dims3, 2, dim_head)
    q_h: ptr = st_reshape(q4, _dims3, 3)
    st_tensor_free(q4)
    int_array_set(_dims3, 1, tokens_k)
    k_h: ptr = st_reshape(k4, _dims3, 3)
    st_tensor_free(k4)
    v_h: ptr = st_reshape(v4, _dims3, 3)
    st_tensor_free(v4)

    # sim = q @ k^T: [batch*heads, tokens_q, tokens_k]
    kt: ptr = st_transpose(k_h, 1, 2)
    sim: ptr = st_bmm(q_h, kt)
    st_tensor_free(kt)
    # scale
    sim_scaled: ptr = st_mul_scalar_tensor(sim, scale)
    st_tensor_free(sim)
    # softmax on last dim
    attn: ptr = st_softmax(sim_scaled, -1)
    st_tensor_free(sim_scaled)
    # out = attn @ v: [batch*heads, tokens_q, dim_head]
    out_h: ptr = st_bmm(attn, v_h)
    st_tensor_free(attn); st_tensor_free(q_h); st_tensor_free(k_h); st_tensor_free(v_h)

    # reshape back: [batch*heads, tokens_q, dim_head] -> [batch, heads, tokens_q, dim_head]
    # -> [batch, tokens_q, heads, dim_head] -> [batch*tokens_q, dim]
    int_array_set(_dims4, 0, batch)
    int_array_set(_dims4, 1, heads)
    int_array_set(_dims4, 2, tokens_q)
    int_array_set(_dims4, 3, dim_head)
    out4: ptr = st_reshape(out_h, _dims4, 4)
    out4 = st_transpose(out4, 1, 2)  # [batch, tokens_q, heads, dim_head]
    _dims2: list[int] = make_int_array(2)
    int_array_set(_dims2, 0, batch * tokens_q)
    int_array_set(_dims2, 1, dim)
    out: ptr = st_reshape(out4, _dims2, 2)
    st_tensor_free(out_h); st_tensor_free(out4)
    return out

# Masked version: mask is [tokens_q, tokens_k] 2D tensor
def attention_torch_masked(q: ptr, k: ptr, v: ptr, batch: int, tokens_q: int, tokens_k: int, dim: int, heads: int, mask: ptr) -> ptr:
    dim_head: int = dim // heads
    scale: float = 1.0 / sqrt(float(dim_head))

    _dims4: list[int] = make_int_array(4)
    int_array_set(_dims4, 0, batch)
    int_array_set(_dims4, 1, tokens_q)
    int_array_set(_dims4, 2, heads)
    int_array_set(_dims4, 3, dim_head)
    q4: ptr = st_reshape(q, _dims4, 4)
    q4 = st_transpose(q4, 1, 2)

    int_array_set(_dims4, 1, tokens_k)
    k4: ptr = st_reshape(k, _dims4, 4)
    k4 = st_transpose(k4, 1, 2)

    v4: ptr = st_reshape(v, _dims4, 4)
    v4 = st_transpose(v4, 1, 2)

    _dims3: list[int] = make_int_array(3)
    int_array_set(_dims3, 0, batch * heads)
    int_array_set(_dims3, 1, tokens_q)
    int_array_set(_dims3, 2, dim_head)
    q_h: ptr = st_reshape(q4, _dims3, 3)
    st_tensor_free(q4)
    int_array_set(_dims3, 1, tokens_k)
    k_h: ptr = st_reshape(k4, _dims3, 3)
    st_tensor_free(k4)
    v_h: ptr = st_reshape(v4, _dims3, 3)
    st_tensor_free(v4)

    kt: ptr = st_transpose(k_h, 1, 2)
    sim: ptr = st_bmm(q_h, kt)
    st_tensor_free(kt)
    sim_scaled: ptr = st_mul_scalar_tensor(sim, scale)
    st_tensor_free(sim)

    # Add mask: mask is already [batch*heads, tokens_q, tokens_k]
    sim_masked: ptr = st_add_tensor(sim_scaled, mask)
    st_tensor_free(sim_scaled)

    attn: ptr = st_softmax(sim_masked, -1)
    st_tensor_free(sim_masked)
    out_h: ptr = st_bmm(attn, v_h)
    st_tensor_free(attn); st_tensor_free(q_h); st_tensor_free(k_h); st_tensor_free(v_h)

    int_array_set(_dims4, 0, batch)
    int_array_set(_dims4, 1, heads)
    int_array_set(_dims4, 2, tokens_q)
    int_array_set(_dims4, 3, dim_head)
    out4: ptr = st_reshape(out_h, _dims4, 4)
    out4 = st_transpose(out4, 1, 2)
    _dims2: list[int] = make_int_array(2)
    int_array_set(_dims2, 0, batch * tokens_q)
    int_array_set(_dims2, 1, dim)
    out: ptr = st_reshape(out4, _dims2, 2)
    st_tensor_free(out_h); st_tensor_free(out4)
    return out

# 假设 st_transpose 和 st_bmm 已声明；如果未声明需要补
extern fn st_transpose(t: ptr, dim0: int, dim1: int) -> ptr from "staticpy_torch"
extern fn st_bmm(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_mul_scalar_tensor(t: ptr, s: float) -> ptr from "staticpy_torch"
extern fn st_slice(t: ptr, dim: int, start: int, end: int) -> ptr from "staticpy_torch"
extern fn st_clone(t: ptr) -> ptr from "staticpy_torch"
