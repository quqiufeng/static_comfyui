# clip.static.py — CLIP-L/G text encode (GPU ptr version, auto-generated)

extern fn make_float_array(n: int) -> list[float] from "prelude"
extern fn float_array_set(a: list[float], i: int, v: float) -> void from "prelude"
extern fn float_array_ref(a: list[float], i: int) -> float from "prelude"
extern fn float_array_offset(a: list[float], n: int) -> list[float] from "prelude"

extern fn st_from_blob_1d(data: ptr, d0: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_2d(data: ptr, d0: int, d1: int) -> ptr from "staticpy_torch"
extern fn st_from_blob_3d(data: ptr, d0: int, d1: int, d2: int) -> ptr from "staticpy_torch"
extern fn st_layer_norm(input: ptr, weight: ptr, bias: ptr, eps: float) -> ptr from "staticpy_torch"
extern fn st_linear(input: ptr, weight: ptr, bias: ptr) -> ptr from "staticpy_torch"
extern fn st_gelu(input: ptr) -> ptr from "staticpy_torch"
extern fn st_sigmoid(input: ptr) -> ptr from "staticpy_torch"
extern fn st_mul_tensor(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_add_tensor(a: ptr, b: ptr) -> ptr from "staticpy_torch"
extern fn st_cat_dim(a: ptr, b: ptr, dim: int) -> ptr from "staticpy_torch"
extern fn st_reshape(t: ptr, dims: ptr, ndim: int) -> ptr from "staticpy_torch"
extern fn st_slice(t: ptr, dim: int, start: int, end: int) -> ptr from "staticpy_torch"
extern fn st_tensor_free(t: ptr) -> void from "staticpy_torch"
extern fn st_tensor_save(t: ptr, path: str) -> void from "staticpy_torch"
extern fn st_clone(t: ptr) -> ptr from "staticpy_torch"

def w_slice_1d(data: list[float], offset: int, n: int) -> ptr:
    return st_from_blob_1d(float_array_offset(data, offset), n)

def w_slice_2d(data: list[float], offset: int, d0: int, d1: int) -> ptr:
    return st_from_blob_2d(float_array_offset(data, offset), d0, d1)

def quick_gelu(x: ptr) -> ptr:
    scaled: ptr = st_mul_scalar_tensor(x, 1.702)
    sig: ptr = st_sigmoid(scaled)
    r: ptr = st_mul_tensor(x, sig)
    st_tensor_free(scaled); st_tensor_free(sig)
    return r

def clip_embed(tokens: list[float], data: list[float], tok_offset: int, pos_offset: int, seq_len: int, dim: int) -> ptr:
    cpu: list[float] = make_float_array(seq_len * dim)
    pos: int = 0
    while pos < seq_len:
        tok_id: int = int(float_array_ref(tokens, pos))
        i: int = 0
        while i < dim:
            v: float = float_array_ref(data, tok_offset + tok_id * dim + i) + float_array_ref(data, pos_offset + pos * dim + i)
            float_array_set(cpu, pos * dim + i, v)
            i = i + 1
        pos = pos + 1
    return st_from_blob_3d(cpu, 1, seq_len, dim)

def make_causal_mask(seq_len: int, heads: int) -> ptr:
    cpu: list[float] = make_float_array(heads * seq_len * seq_len)
    h: int = 0
    while h < heads:
        r: int = 0
        while r < seq_len:
            c: int = 0
            while c < seq_len:
                v: float = 0.0
                if c > r:
                    v = -3.4028235e38
                float_array_set(cpu, (h * seq_len + r) * seq_len + c, v)
                c = c + 1
            r = r + 1
        h = h + 1
    return st_from_blob_3d(cpu, heads, seq_len, seq_len)

def clip_layer(x: ptr, mask: ptr,
               ln1_w: ptr, ln1_b: ptr,
               q_w: ptr, q_b: ptr, k_w: ptr, k_b: ptr, v_w: ptr, v_b: ptr,
               out_w: ptr, out_b: ptr,
               ln2_w: ptr, ln2_b: ptr,
               fc1_w: ptr, fc1_b: ptr, fc2_w: ptr, fc2_b: ptr,
               dim: int, heads: int, hidden: int, use_gelu: int) -> ptr:
    n1: ptr = st_layer_norm(x, ln1_w, ln1_b, 1e-5)
    q: ptr = st_linear(n1, q_w, q_b)
    k: ptr = st_linear(n1, k_w, k_b)
    v: ptr = st_linear(n1, v_w, v_b)
    o: ptr = attention_torch_masked(q, k, v, 1, 77, 77, dim, heads, mask)
    out: ptr = st_linear(o, out_w, out_b)
    r1: ptr = st_add_tensor(x, out)
    st_tensor_free(n1); st_tensor_free(q); st_tensor_free(k); st_tensor_free(v)
    st_tensor_free(o); st_tensor_free(out); st_tensor_free(x)
    n2: ptr = st_layer_norm(r1, ln2_w, ln2_b, 1e-5)
    h: ptr = st_linear(n2, fc1_w, fc1_b)
    if use_gelu == 1:
        a: ptr = st_gelu(h)
    else:
        a: ptr = quick_gelu(h)
    h2: ptr = st_linear(a, fc2_w, fc2_b)
    r2: ptr = st_add_tensor(r1, h2)
    st_tensor_free(n2); st_tensor_free(h); st_tensor_free(a); st_tensor_free(h2); st_tensor_free(r1)
    return r2

def clip_g_encode(tokens: list[float], data: list[float]) -> ptr:
    seq_len: int = 77
    dim: int = 1280
    heads: int = 20
    hidden: int = 5120
    x: ptr = clip_embed(tokens, data, 98560, 0, seq_len, dim)
    mask: ptr = make_causal_mask(seq_len, heads)
    penultimate: ptr
    # Layer 0
    _ln1_w_0: ptr = w_slice_1d(data, 63342080, 1280)
    _ln1_b_0: ptr = w_slice_1d(data, 63340800, 1280)
    _q_w_0: ptr = w_slice_2d(data, 79740160, 1280, 1280)
    _q_b_0: ptr = w_slice_1d(data, 79738880, 1280)
    _k_w_0: ptr = w_slice_2d(data, 76460800, 1280, 1280)
    _k_b_0: ptr = w_slice_1d(data, 76459520, 1280)
    _v_w_0: ptr = w_slice_2d(data, 81379840, 1280, 1280)
    _v_b_0: ptr = w_slice_1d(data, 81378560, 1280)
    _out_w_0: ptr = w_slice_2d(data, 78100480, 1280, 1280)
    _out_b_0: ptr = w_slice_1d(data, 78099200, 1280)
    _ln2_w_0: ptr = w_slice_1d(data, 63344640, 1280)
    _ln2_b_0: ptr = w_slice_1d(data, 63343360, 1280)
    _fc1_w_0: ptr = w_slice_2d(data, 63351040, 5120, 1280)
    _fc1_b_0: ptr = w_slice_1d(data, 63345920, 5120)
    _fc2_w_0: ptr = w_slice_2d(data, 69905920, 1280, 5120)
    _fc2_b_0: ptr = w_slice_1d(data, 69904640, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_0, _ln1_b_0,
                   _q_w_0, _q_b_0, _k_w_0, _k_b_0, _v_w_0, _v_b_0,
                   _out_w_0, _out_b_0,
                   _ln2_w_0, _ln2_b_0,
                   _fc1_w_0, _fc1_b_0, _fc2_w_0, _fc2_b_0,
                   dim, heads, hidden, 1)
    # Layer 1
    _ln1_w_1: ptr = w_slice_1d(data, 83019520, 1280)
    _ln1_b_1: ptr = w_slice_1d(data, 83018240, 1280)
    _q_w_1: ptr = w_slice_2d(data, 99417600, 1280, 1280)
    _q_b_1: ptr = w_slice_1d(data, 99416320, 1280)
    _k_w_1: ptr = w_slice_2d(data, 96138240, 1280, 1280)
    _k_b_1: ptr = w_slice_1d(data, 96136960, 1280)
    _v_w_1: ptr = w_slice_2d(data, 101057280, 1280, 1280)
    _v_b_1: ptr = w_slice_1d(data, 101056000, 1280)
    _out_w_1: ptr = w_slice_2d(data, 97777920, 1280, 1280)
    _out_b_1: ptr = w_slice_1d(data, 97776640, 1280)
    _ln2_w_1: ptr = w_slice_1d(data, 83022080, 1280)
    _ln2_b_1: ptr = w_slice_1d(data, 83020800, 1280)
    _fc1_w_1: ptr = w_slice_2d(data, 83028480, 5120, 1280)
    _fc1_b_1: ptr = w_slice_1d(data, 83023360, 5120)
    _fc2_w_1: ptr = w_slice_2d(data, 89583360, 1280, 5120)
    _fc2_b_1: ptr = w_slice_1d(data, 89582080, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_1, _ln1_b_1,
                   _q_w_1, _q_b_1, _k_w_1, _k_b_1, _v_w_1, _v_b_1,
                   _out_w_1, _out_b_1,
                   _ln2_w_1, _ln2_b_1,
                   _fc1_w_1, _fc1_b_1, _fc2_w_1, _fc2_b_1,
                   dim, heads, hidden, 1)
    # Layer 2
    _ln1_w_2: ptr = w_slice_1d(data, 299471360, 1280)
    _ln1_b_2: ptr = w_slice_1d(data, 299470080, 1280)
    _q_w_2: ptr = w_slice_2d(data, 315869440, 1280, 1280)
    _q_b_2: ptr = w_slice_1d(data, 315868160, 1280)
    _k_w_2: ptr = w_slice_2d(data, 312590080, 1280, 1280)
    _k_b_2: ptr = w_slice_1d(data, 312588800, 1280)
    _v_w_2: ptr = w_slice_2d(data, 317509120, 1280, 1280)
    _v_b_2: ptr = w_slice_1d(data, 317507840, 1280)
    _out_w_2: ptr = w_slice_2d(data, 314229760, 1280, 1280)
    _out_b_2: ptr = w_slice_1d(data, 314228480, 1280)
    _ln2_w_2: ptr = w_slice_1d(data, 299473920, 1280)
    _ln2_b_2: ptr = w_slice_1d(data, 299472640, 1280)
    _fc1_w_2: ptr = w_slice_2d(data, 299480320, 5120, 1280)
    _fc1_b_2: ptr = w_slice_1d(data, 299475200, 5120)
    _fc2_w_2: ptr = w_slice_2d(data, 306035200, 1280, 5120)
    _fc2_b_2: ptr = w_slice_1d(data, 306033920, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_2, _ln1_b_2,
                   _q_w_2, _q_b_2, _k_w_2, _k_b_2, _v_w_2, _v_b_2,
                   _out_w_2, _out_b_2,
                   _ln2_w_2, _ln2_b_2,
                   _fc1_w_2, _fc1_b_2, _fc2_w_2, _fc2_b_2,
                   dim, heads, hidden, 1)
    # Layer 3
    _ln1_w_3: ptr = w_slice_1d(data, 515923200, 1280)
    _ln1_b_3: ptr = w_slice_1d(data, 515921920, 1280)
    _q_w_3: ptr = w_slice_2d(data, 532321280, 1280, 1280)
    _q_b_3: ptr = w_slice_1d(data, 532320000, 1280)
    _k_w_3: ptr = w_slice_2d(data, 529041920, 1280, 1280)
    _k_b_3: ptr = w_slice_1d(data, 529040640, 1280)
    _v_w_3: ptr = w_slice_2d(data, 533960960, 1280, 1280)
    _v_b_3: ptr = w_slice_1d(data, 533959680, 1280)
    _out_w_3: ptr = w_slice_2d(data, 530681600, 1280, 1280)
    _out_b_3: ptr = w_slice_1d(data, 530680320, 1280)
    _ln2_w_3: ptr = w_slice_1d(data, 515925760, 1280)
    _ln2_b_3: ptr = w_slice_1d(data, 515924480, 1280)
    _fc1_w_3: ptr = w_slice_2d(data, 515932160, 5120, 1280)
    _fc1_b_3: ptr = w_slice_1d(data, 515927040, 5120)
    _fc2_w_3: ptr = w_slice_2d(data, 522487040, 1280, 5120)
    _fc2_b_3: ptr = w_slice_1d(data, 522485760, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_3, _ln1_b_3,
                   _q_w_3, _q_b_3, _k_w_3, _k_b_3, _v_w_3, _v_b_3,
                   _out_w_3, _out_b_3,
                   _ln2_w_3, _ln2_b_3,
                   _fc1_w_3, _fc1_b_3, _fc2_w_3, _fc2_b_3,
                   dim, heads, hidden, 1)
    # Layer 4
    _ln1_w_4: ptr = w_slice_1d(data, 574955520, 1280)
    _ln1_b_4: ptr = w_slice_1d(data, 574954240, 1280)
    _q_w_4: ptr = w_slice_2d(data, 591353600, 1280, 1280)
    _q_b_4: ptr = w_slice_1d(data, 591352320, 1280)
    _k_w_4: ptr = w_slice_2d(data, 588074240, 1280, 1280)
    _k_b_4: ptr = w_slice_1d(data, 588072960, 1280)
    _v_w_4: ptr = w_slice_2d(data, 592993280, 1280, 1280)
    _v_b_4: ptr = w_slice_1d(data, 592992000, 1280)
    _out_w_4: ptr = w_slice_2d(data, 589713920, 1280, 1280)
    _out_b_4: ptr = w_slice_1d(data, 589712640, 1280)
    _ln2_w_4: ptr = w_slice_1d(data, 574958080, 1280)
    _ln2_b_4: ptr = w_slice_1d(data, 574956800, 1280)
    _fc1_w_4: ptr = w_slice_2d(data, 574964480, 5120, 1280)
    _fc1_b_4: ptr = w_slice_1d(data, 574959360, 5120)
    _fc2_w_4: ptr = w_slice_2d(data, 581519360, 1280, 5120)
    _fc2_b_4: ptr = w_slice_1d(data, 581518080, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_4, _ln1_b_4,
                   _q_w_4, _q_b_4, _k_w_4, _k_b_4, _v_w_4, _v_b_4,
                   _out_w_4, _out_b_4,
                   _ln2_w_4, _ln2_b_4,
                   _fc1_w_4, _fc1_b_4, _fc2_w_4, _fc2_b_4,
                   dim, heads, hidden, 1)
    # Layer 5
    _ln1_w_5: ptr = w_slice_1d(data, 594632960, 1280)
    _ln1_b_5: ptr = w_slice_1d(data, 594631680, 1280)
    _q_w_5: ptr = w_slice_2d(data, 611031040, 1280, 1280)
    _q_b_5: ptr = w_slice_1d(data, 611029760, 1280)
    _k_w_5: ptr = w_slice_2d(data, 607751680, 1280, 1280)
    _k_b_5: ptr = w_slice_1d(data, 607750400, 1280)
    _v_w_5: ptr = w_slice_2d(data, 612670720, 1280, 1280)
    _v_b_5: ptr = w_slice_1d(data, 612669440, 1280)
    _out_w_5: ptr = w_slice_2d(data, 609391360, 1280, 1280)
    _out_b_5: ptr = w_slice_1d(data, 609390080, 1280)
    _ln2_w_5: ptr = w_slice_1d(data, 594635520, 1280)
    _ln2_b_5: ptr = w_slice_1d(data, 594634240, 1280)
    _fc1_w_5: ptr = w_slice_2d(data, 594641920, 5120, 1280)
    _fc1_b_5: ptr = w_slice_1d(data, 594636800, 5120)
    _fc2_w_5: ptr = w_slice_2d(data, 601196800, 1280, 5120)
    _fc2_b_5: ptr = w_slice_1d(data, 601195520, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_5, _ln1_b_5,
                   _q_w_5, _q_b_5, _k_w_5, _k_b_5, _v_w_5, _v_b_5,
                   _out_w_5, _out_b_5,
                   _ln2_w_5, _ln2_b_5,
                   _fc1_w_5, _fc1_b_5, _fc2_w_5, _fc2_b_5,
                   dim, heads, hidden, 1)
    # Layer 6
    _ln1_w_6: ptr = w_slice_1d(data, 614310400, 1280)
    _ln1_b_6: ptr = w_slice_1d(data, 614309120, 1280)
    _q_w_6: ptr = w_slice_2d(data, 630708480, 1280, 1280)
    _q_b_6: ptr = w_slice_1d(data, 630707200, 1280)
    _k_w_6: ptr = w_slice_2d(data, 627429120, 1280, 1280)
    _k_b_6: ptr = w_slice_1d(data, 627427840, 1280)
    _v_w_6: ptr = w_slice_2d(data, 632348160, 1280, 1280)
    _v_b_6: ptr = w_slice_1d(data, 632346880, 1280)
    _out_w_6: ptr = w_slice_2d(data, 629068800, 1280, 1280)
    _out_b_6: ptr = w_slice_1d(data, 629067520, 1280)
    _ln2_w_6: ptr = w_slice_1d(data, 614312960, 1280)
    _ln2_b_6: ptr = w_slice_1d(data, 614311680, 1280)
    _fc1_w_6: ptr = w_slice_2d(data, 614319360, 5120, 1280)
    _fc1_b_6: ptr = w_slice_1d(data, 614314240, 5120)
    _fc2_w_6: ptr = w_slice_2d(data, 620874240, 1280, 5120)
    _fc2_b_6: ptr = w_slice_1d(data, 620872960, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_6, _ln1_b_6,
                   _q_w_6, _q_b_6, _k_w_6, _k_b_6, _v_w_6, _v_b_6,
                   _out_w_6, _out_b_6,
                   _ln2_w_6, _ln2_b_6,
                   _fc1_w_6, _fc1_b_6, _fc2_w_6, _fc2_b_6,
                   dim, heads, hidden, 1)
    # Layer 7
    _ln1_w_7: ptr = w_slice_1d(data, 633987840, 1280)
    _ln1_b_7: ptr = w_slice_1d(data, 633986560, 1280)
    _q_w_7: ptr = w_slice_2d(data, 650385920, 1280, 1280)
    _q_b_7: ptr = w_slice_1d(data, 650384640, 1280)
    _k_w_7: ptr = w_slice_2d(data, 647106560, 1280, 1280)
    _k_b_7: ptr = w_slice_1d(data, 647105280, 1280)
    _v_w_7: ptr = w_slice_2d(data, 652025600, 1280, 1280)
    _v_b_7: ptr = w_slice_1d(data, 652024320, 1280)
    _out_w_7: ptr = w_slice_2d(data, 648746240, 1280, 1280)
    _out_b_7: ptr = w_slice_1d(data, 648744960, 1280)
    _ln2_w_7: ptr = w_slice_1d(data, 633990400, 1280)
    _ln2_b_7: ptr = w_slice_1d(data, 633989120, 1280)
    _fc1_w_7: ptr = w_slice_2d(data, 633996800, 5120, 1280)
    _fc1_b_7: ptr = w_slice_1d(data, 633991680, 5120)
    _fc2_w_7: ptr = w_slice_2d(data, 640551680, 1280, 5120)
    _fc2_b_7: ptr = w_slice_1d(data, 640550400, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_7, _ln1_b_7,
                   _q_w_7, _q_b_7, _k_w_7, _k_b_7, _v_w_7, _v_b_7,
                   _out_w_7, _out_b_7,
                   _ln2_w_7, _ln2_b_7,
                   _fc1_w_7, _fc1_b_7, _fc2_w_7, _fc2_b_7,
                   dim, heads, hidden, 1)
    # Layer 8
    _ln1_w_8: ptr = w_slice_1d(data, 653665280, 1280)
    _ln1_b_8: ptr = w_slice_1d(data, 653664000, 1280)
    _q_w_8: ptr = w_slice_2d(data, 670063360, 1280, 1280)
    _q_b_8: ptr = w_slice_1d(data, 670062080, 1280)
    _k_w_8: ptr = w_slice_2d(data, 666784000, 1280, 1280)
    _k_b_8: ptr = w_slice_1d(data, 666782720, 1280)
    _v_w_8: ptr = w_slice_2d(data, 671703040, 1280, 1280)
    _v_b_8: ptr = w_slice_1d(data, 671701760, 1280)
    _out_w_8: ptr = w_slice_2d(data, 668423680, 1280, 1280)
    _out_b_8: ptr = w_slice_1d(data, 668422400, 1280)
    _ln2_w_8: ptr = w_slice_1d(data, 653667840, 1280)
    _ln2_b_8: ptr = w_slice_1d(data, 653666560, 1280)
    _fc1_w_8: ptr = w_slice_2d(data, 653674240, 5120, 1280)
    _fc1_b_8: ptr = w_slice_1d(data, 653669120, 5120)
    _fc2_w_8: ptr = w_slice_2d(data, 660229120, 1280, 5120)
    _fc2_b_8: ptr = w_slice_1d(data, 660227840, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_8, _ln1_b_8,
                   _q_w_8, _q_b_8, _k_w_8, _k_b_8, _v_w_8, _v_b_8,
                   _out_w_8, _out_b_8,
                   _ln2_w_8, _ln2_b_8,
                   _fc1_w_8, _fc1_b_8, _fc2_w_8, _fc2_b_8,
                   dim, heads, hidden, 1)
    # Layer 9
    _ln1_w_9: ptr = w_slice_1d(data, 673342720, 1280)
    _ln1_b_9: ptr = w_slice_1d(data, 673341440, 1280)
    _q_w_9: ptr = w_slice_2d(data, 689740800, 1280, 1280)
    _q_b_9: ptr = w_slice_1d(data, 689739520, 1280)
    _k_w_9: ptr = w_slice_2d(data, 686461440, 1280, 1280)
    _k_b_9: ptr = w_slice_1d(data, 686460160, 1280)
    _v_w_9: ptr = w_slice_2d(data, 691380480, 1280, 1280)
    _v_b_9: ptr = w_slice_1d(data, 691379200, 1280)
    _out_w_9: ptr = w_slice_2d(data, 688101120, 1280, 1280)
    _out_b_9: ptr = w_slice_1d(data, 688099840, 1280)
    _ln2_w_9: ptr = w_slice_1d(data, 673345280, 1280)
    _ln2_b_9: ptr = w_slice_1d(data, 673344000, 1280)
    _fc1_w_9: ptr = w_slice_2d(data, 673351680, 5120, 1280)
    _fc1_b_9: ptr = w_slice_1d(data, 673346560, 5120)
    _fc2_w_9: ptr = w_slice_2d(data, 679906560, 1280, 5120)
    _fc2_b_9: ptr = w_slice_1d(data, 679905280, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_9, _ln1_b_9,
                   _q_w_9, _q_b_9, _k_w_9, _k_b_9, _v_w_9, _v_b_9,
                   _out_w_9, _out_b_9,
                   _ln2_w_9, _ln2_b_9,
                   _fc1_w_9, _fc1_b_9, _fc2_w_9, _fc2_b_9,
                   dim, heads, hidden, 1)
    # Layer 10
    _ln1_w_10: ptr = w_slice_1d(data, 102696960, 1280)
    _ln1_b_10: ptr = w_slice_1d(data, 102695680, 1280)
    _q_w_10: ptr = w_slice_2d(data, 119095040, 1280, 1280)
    _q_b_10: ptr = w_slice_1d(data, 119093760, 1280)
    _k_w_10: ptr = w_slice_2d(data, 115815680, 1280, 1280)
    _k_b_10: ptr = w_slice_1d(data, 115814400, 1280)
    _v_w_10: ptr = w_slice_2d(data, 120734720, 1280, 1280)
    _v_b_10: ptr = w_slice_1d(data, 120733440, 1280)
    _out_w_10: ptr = w_slice_2d(data, 117455360, 1280, 1280)
    _out_b_10: ptr = w_slice_1d(data, 117454080, 1280)
    _ln2_w_10: ptr = w_slice_1d(data, 102699520, 1280)
    _ln2_b_10: ptr = w_slice_1d(data, 102698240, 1280)
    _fc1_w_10: ptr = w_slice_2d(data, 102705920, 5120, 1280)
    _fc1_b_10: ptr = w_slice_1d(data, 102700800, 5120)
    _fc2_w_10: ptr = w_slice_2d(data, 109260800, 1280, 5120)
    _fc2_b_10: ptr = w_slice_1d(data, 109259520, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_10, _ln1_b_10,
                   _q_w_10, _q_b_10, _k_w_10, _k_b_10, _v_w_10, _v_b_10,
                   _out_w_10, _out_b_10,
                   _ln2_w_10, _ln2_b_10,
                   _fc1_w_10, _fc1_b_10, _fc2_w_10, _fc2_b_10,
                   dim, heads, hidden, 1)
    # Layer 11
    _ln1_w_11: ptr = w_slice_1d(data, 122374400, 1280)
    _ln1_b_11: ptr = w_slice_1d(data, 122373120, 1280)
    _q_w_11: ptr = w_slice_2d(data, 138772480, 1280, 1280)
    _q_b_11: ptr = w_slice_1d(data, 138771200, 1280)
    _k_w_11: ptr = w_slice_2d(data, 135493120, 1280, 1280)
    _k_b_11: ptr = w_slice_1d(data, 135491840, 1280)
    _v_w_11: ptr = w_slice_2d(data, 140412160, 1280, 1280)
    _v_b_11: ptr = w_slice_1d(data, 140410880, 1280)
    _out_w_11: ptr = w_slice_2d(data, 137132800, 1280, 1280)
    _out_b_11: ptr = w_slice_1d(data, 137131520, 1280)
    _ln2_w_11: ptr = w_slice_1d(data, 122376960, 1280)
    _ln2_b_11: ptr = w_slice_1d(data, 122375680, 1280)
    _fc1_w_11: ptr = w_slice_2d(data, 122383360, 5120, 1280)
    _fc1_b_11: ptr = w_slice_1d(data, 122378240, 5120)
    _fc2_w_11: ptr = w_slice_2d(data, 128938240, 1280, 5120)
    _fc2_b_11: ptr = w_slice_1d(data, 128936960, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_11, _ln1_b_11,
                   _q_w_11, _q_b_11, _k_w_11, _k_b_11, _v_w_11, _v_b_11,
                   _out_w_11, _out_b_11,
                   _ln2_w_11, _ln2_b_11,
                   _fc1_w_11, _fc1_b_11, _fc2_w_11, _fc2_b_11,
                   dim, heads, hidden, 1)
    # Layer 12
    _ln1_w_12: ptr = w_slice_1d(data, 142051840, 1280)
    _ln1_b_12: ptr = w_slice_1d(data, 142050560, 1280)
    _q_w_12: ptr = w_slice_2d(data, 158449920, 1280, 1280)
    _q_b_12: ptr = w_slice_1d(data, 158448640, 1280)
    _k_w_12: ptr = w_slice_2d(data, 155170560, 1280, 1280)
    _k_b_12: ptr = w_slice_1d(data, 155169280, 1280)
    _v_w_12: ptr = w_slice_2d(data, 160089600, 1280, 1280)
    _v_b_12: ptr = w_slice_1d(data, 160088320, 1280)
    _out_w_12: ptr = w_slice_2d(data, 156810240, 1280, 1280)
    _out_b_12: ptr = w_slice_1d(data, 156808960, 1280)
    _ln2_w_12: ptr = w_slice_1d(data, 142054400, 1280)
    _ln2_b_12: ptr = w_slice_1d(data, 142053120, 1280)
    _fc1_w_12: ptr = w_slice_2d(data, 142060800, 5120, 1280)
    _fc1_b_12: ptr = w_slice_1d(data, 142055680, 5120)
    _fc2_w_12: ptr = w_slice_2d(data, 148615680, 1280, 5120)
    _fc2_b_12: ptr = w_slice_1d(data, 148614400, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_12, _ln1_b_12,
                   _q_w_12, _q_b_12, _k_w_12, _k_b_12, _v_w_12, _v_b_12,
                   _out_w_12, _out_b_12,
                   _ln2_w_12, _ln2_b_12,
                   _fc1_w_12, _fc1_b_12, _fc2_w_12, _fc2_b_12,
                   dim, heads, hidden, 1)
    # Layer 13
    _ln1_w_13: ptr = w_slice_1d(data, 161729280, 1280)
    _ln1_b_13: ptr = w_slice_1d(data, 161728000, 1280)
    _q_w_13: ptr = w_slice_2d(data, 178127360, 1280, 1280)
    _q_b_13: ptr = w_slice_1d(data, 178126080, 1280)
    _k_w_13: ptr = w_slice_2d(data, 174848000, 1280, 1280)
    _k_b_13: ptr = w_slice_1d(data, 174846720, 1280)
    _v_w_13: ptr = w_slice_2d(data, 179767040, 1280, 1280)
    _v_b_13: ptr = w_slice_1d(data, 179765760, 1280)
    _out_w_13: ptr = w_slice_2d(data, 176487680, 1280, 1280)
    _out_b_13: ptr = w_slice_1d(data, 176486400, 1280)
    _ln2_w_13: ptr = w_slice_1d(data, 161731840, 1280)
    _ln2_b_13: ptr = w_slice_1d(data, 161730560, 1280)
    _fc1_w_13: ptr = w_slice_2d(data, 161738240, 5120, 1280)
    _fc1_b_13: ptr = w_slice_1d(data, 161733120, 5120)
    _fc2_w_13: ptr = w_slice_2d(data, 168293120, 1280, 5120)
    _fc2_b_13: ptr = w_slice_1d(data, 168291840, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_13, _ln1_b_13,
                   _q_w_13, _q_b_13, _k_w_13, _k_b_13, _v_w_13, _v_b_13,
                   _out_w_13, _out_b_13,
                   _ln2_w_13, _ln2_b_13,
                   _fc1_w_13, _fc1_b_13, _fc2_w_13, _fc2_b_13,
                   dim, heads, hidden, 1)
    # Layer 14
    _ln1_w_14: ptr = w_slice_1d(data, 181406720, 1280)
    _ln1_b_14: ptr = w_slice_1d(data, 181405440, 1280)
    _q_w_14: ptr = w_slice_2d(data, 197804800, 1280, 1280)
    _q_b_14: ptr = w_slice_1d(data, 197803520, 1280)
    _k_w_14: ptr = w_slice_2d(data, 194525440, 1280, 1280)
    _k_b_14: ptr = w_slice_1d(data, 194524160, 1280)
    _v_w_14: ptr = w_slice_2d(data, 199444480, 1280, 1280)
    _v_b_14: ptr = w_slice_1d(data, 199443200, 1280)
    _out_w_14: ptr = w_slice_2d(data, 196165120, 1280, 1280)
    _out_b_14: ptr = w_slice_1d(data, 196163840, 1280)
    _ln2_w_14: ptr = w_slice_1d(data, 181409280, 1280)
    _ln2_b_14: ptr = w_slice_1d(data, 181408000, 1280)
    _fc1_w_14: ptr = w_slice_2d(data, 181415680, 5120, 1280)
    _fc1_b_14: ptr = w_slice_1d(data, 181410560, 5120)
    _fc2_w_14: ptr = w_slice_2d(data, 187970560, 1280, 5120)
    _fc2_b_14: ptr = w_slice_1d(data, 187969280, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_14, _ln1_b_14,
                   _q_w_14, _q_b_14, _k_w_14, _k_b_14, _v_w_14, _v_b_14,
                   _out_w_14, _out_b_14,
                   _ln2_w_14, _ln2_b_14,
                   _fc1_w_14, _fc1_b_14, _fc2_w_14, _fc2_b_14,
                   dim, heads, hidden, 1)
    # Layer 15
    _ln1_w_15: ptr = w_slice_1d(data, 201084160, 1280)
    _ln1_b_15: ptr = w_slice_1d(data, 201082880, 1280)
    _q_w_15: ptr = w_slice_2d(data, 217482240, 1280, 1280)
    _q_b_15: ptr = w_slice_1d(data, 217480960, 1280)
    _k_w_15: ptr = w_slice_2d(data, 214202880, 1280, 1280)
    _k_b_15: ptr = w_slice_1d(data, 214201600, 1280)
    _v_w_15: ptr = w_slice_2d(data, 219121920, 1280, 1280)
    _v_b_15: ptr = w_slice_1d(data, 219120640, 1280)
    _out_w_15: ptr = w_slice_2d(data, 215842560, 1280, 1280)
    _out_b_15: ptr = w_slice_1d(data, 215841280, 1280)
    _ln2_w_15: ptr = w_slice_1d(data, 201086720, 1280)
    _ln2_b_15: ptr = w_slice_1d(data, 201085440, 1280)
    _fc1_w_15: ptr = w_slice_2d(data, 201093120, 5120, 1280)
    _fc1_b_15: ptr = w_slice_1d(data, 201088000, 5120)
    _fc2_w_15: ptr = w_slice_2d(data, 207648000, 1280, 5120)
    _fc2_b_15: ptr = w_slice_1d(data, 207646720, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_15, _ln1_b_15,
                   _q_w_15, _q_b_15, _k_w_15, _k_b_15, _v_w_15, _v_b_15,
                   _out_w_15, _out_b_15,
                   _ln2_w_15, _ln2_b_15,
                   _fc1_w_15, _fc1_b_15, _fc2_w_15, _fc2_b_15,
                   dim, heads, hidden, 1)
    # Layer 16
    _ln1_w_16: ptr = w_slice_1d(data, 220761600, 1280)
    _ln1_b_16: ptr = w_slice_1d(data, 220760320, 1280)
    _q_w_16: ptr = w_slice_2d(data, 237159680, 1280, 1280)
    _q_b_16: ptr = w_slice_1d(data, 237158400, 1280)
    _k_w_16: ptr = w_slice_2d(data, 233880320, 1280, 1280)
    _k_b_16: ptr = w_slice_1d(data, 233879040, 1280)
    _v_w_16: ptr = w_slice_2d(data, 238799360, 1280, 1280)
    _v_b_16: ptr = w_slice_1d(data, 238798080, 1280)
    _out_w_16: ptr = w_slice_2d(data, 235520000, 1280, 1280)
    _out_b_16: ptr = w_slice_1d(data, 235518720, 1280)
    _ln2_w_16: ptr = w_slice_1d(data, 220764160, 1280)
    _ln2_b_16: ptr = w_slice_1d(data, 220762880, 1280)
    _fc1_w_16: ptr = w_slice_2d(data, 220770560, 5120, 1280)
    _fc1_b_16: ptr = w_slice_1d(data, 220765440, 5120)
    _fc2_w_16: ptr = w_slice_2d(data, 227325440, 1280, 5120)
    _fc2_b_16: ptr = w_slice_1d(data, 227324160, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_16, _ln1_b_16,
                   _q_w_16, _q_b_16, _k_w_16, _k_b_16, _v_w_16, _v_b_16,
                   _out_w_16, _out_b_16,
                   _ln2_w_16, _ln2_b_16,
                   _fc1_w_16, _fc1_b_16, _fc2_w_16, _fc2_b_16,
                   dim, heads, hidden, 1)
    # Layer 17
    _ln1_w_17: ptr = w_slice_1d(data, 240439040, 1280)
    _ln1_b_17: ptr = w_slice_1d(data, 240437760, 1280)
    _q_w_17: ptr = w_slice_2d(data, 256837120, 1280, 1280)
    _q_b_17: ptr = w_slice_1d(data, 256835840, 1280)
    _k_w_17: ptr = w_slice_2d(data, 253557760, 1280, 1280)
    _k_b_17: ptr = w_slice_1d(data, 253556480, 1280)
    _v_w_17: ptr = w_slice_2d(data, 258476800, 1280, 1280)
    _v_b_17: ptr = w_slice_1d(data, 258475520, 1280)
    _out_w_17: ptr = w_slice_2d(data, 255197440, 1280, 1280)
    _out_b_17: ptr = w_slice_1d(data, 255196160, 1280)
    _ln2_w_17: ptr = w_slice_1d(data, 240441600, 1280)
    _ln2_b_17: ptr = w_slice_1d(data, 240440320, 1280)
    _fc1_w_17: ptr = w_slice_2d(data, 240448000, 5120, 1280)
    _fc1_b_17: ptr = w_slice_1d(data, 240442880, 5120)
    _fc2_w_17: ptr = w_slice_2d(data, 247002880, 1280, 5120)
    _fc2_b_17: ptr = w_slice_1d(data, 247001600, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_17, _ln1_b_17,
                   _q_w_17, _q_b_17, _k_w_17, _k_b_17, _v_w_17, _v_b_17,
                   _out_w_17, _out_b_17,
                   _ln2_w_17, _ln2_b_17,
                   _fc1_w_17, _fc1_b_17, _fc2_w_17, _fc2_b_17,
                   dim, heads, hidden, 1)
    # Layer 18
    _ln1_w_18: ptr = w_slice_1d(data, 260116480, 1280)
    _ln1_b_18: ptr = w_slice_1d(data, 260115200, 1280)
    _q_w_18: ptr = w_slice_2d(data, 276514560, 1280, 1280)
    _q_b_18: ptr = w_slice_1d(data, 276513280, 1280)
    _k_w_18: ptr = w_slice_2d(data, 273235200, 1280, 1280)
    _k_b_18: ptr = w_slice_1d(data, 273233920, 1280)
    _v_w_18: ptr = w_slice_2d(data, 278154240, 1280, 1280)
    _v_b_18: ptr = w_slice_1d(data, 278152960, 1280)
    _out_w_18: ptr = w_slice_2d(data, 274874880, 1280, 1280)
    _out_b_18: ptr = w_slice_1d(data, 274873600, 1280)
    _ln2_w_18: ptr = w_slice_1d(data, 260119040, 1280)
    _ln2_b_18: ptr = w_slice_1d(data, 260117760, 1280)
    _fc1_w_18: ptr = w_slice_2d(data, 260125440, 5120, 1280)
    _fc1_b_18: ptr = w_slice_1d(data, 260120320, 5120)
    _fc2_w_18: ptr = w_slice_2d(data, 266680320, 1280, 5120)
    _fc2_b_18: ptr = w_slice_1d(data, 266679040, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_18, _ln1_b_18,
                   _q_w_18, _q_b_18, _k_w_18, _k_b_18, _v_w_18, _v_b_18,
                   _out_w_18, _out_b_18,
                   _ln2_w_18, _ln2_b_18,
                   _fc1_w_18, _fc1_b_18, _fc2_w_18, _fc2_b_18,
                   dim, heads, hidden, 1)
    # Layer 19
    _ln1_w_19: ptr = w_slice_1d(data, 279793920, 1280)
    _ln1_b_19: ptr = w_slice_1d(data, 279792640, 1280)
    _q_w_19: ptr = w_slice_2d(data, 296192000, 1280, 1280)
    _q_b_19: ptr = w_slice_1d(data, 296190720, 1280)
    _k_w_19: ptr = w_slice_2d(data, 292912640, 1280, 1280)
    _k_b_19: ptr = w_slice_1d(data, 292911360, 1280)
    _v_w_19: ptr = w_slice_2d(data, 297831680, 1280, 1280)
    _v_b_19: ptr = w_slice_1d(data, 297830400, 1280)
    _out_w_19: ptr = w_slice_2d(data, 294552320, 1280, 1280)
    _out_b_19: ptr = w_slice_1d(data, 294551040, 1280)
    _ln2_w_19: ptr = w_slice_1d(data, 279796480, 1280)
    _ln2_b_19: ptr = w_slice_1d(data, 279795200, 1280)
    _fc1_w_19: ptr = w_slice_2d(data, 279802880, 5120, 1280)
    _fc1_b_19: ptr = w_slice_1d(data, 279797760, 5120)
    _fc2_w_19: ptr = w_slice_2d(data, 286357760, 1280, 5120)
    _fc2_b_19: ptr = w_slice_1d(data, 286356480, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_19, _ln1_b_19,
                   _q_w_19, _q_b_19, _k_w_19, _k_b_19, _v_w_19, _v_b_19,
                   _out_w_19, _out_b_19,
                   _ln2_w_19, _ln2_b_19,
                   _fc1_w_19, _fc1_b_19, _fc2_w_19, _fc2_b_19,
                   dim, heads, hidden, 1)
    # Layer 20
    _ln1_w_20: ptr = w_slice_1d(data, 319148800, 1280)
    _ln1_b_20: ptr = w_slice_1d(data, 319147520, 1280)
    _q_w_20: ptr = w_slice_2d(data, 335546880, 1280, 1280)
    _q_b_20: ptr = w_slice_1d(data, 335545600, 1280)
    _k_w_20: ptr = w_slice_2d(data, 332267520, 1280, 1280)
    _k_b_20: ptr = w_slice_1d(data, 332266240, 1280)
    _v_w_20: ptr = w_slice_2d(data, 337186560, 1280, 1280)
    _v_b_20: ptr = w_slice_1d(data, 337185280, 1280)
    _out_w_20: ptr = w_slice_2d(data, 333907200, 1280, 1280)
    _out_b_20: ptr = w_slice_1d(data, 333905920, 1280)
    _ln2_w_20: ptr = w_slice_1d(data, 319151360, 1280)
    _ln2_b_20: ptr = w_slice_1d(data, 319150080, 1280)
    _fc1_w_20: ptr = w_slice_2d(data, 319157760, 5120, 1280)
    _fc1_b_20: ptr = w_slice_1d(data, 319152640, 5120)
    _fc2_w_20: ptr = w_slice_2d(data, 325712640, 1280, 5120)
    _fc2_b_20: ptr = w_slice_1d(data, 325711360, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_20, _ln1_b_20,
                   _q_w_20, _q_b_20, _k_w_20, _k_b_20, _v_w_20, _v_b_20,
                   _out_w_20, _out_b_20,
                   _ln2_w_20, _ln2_b_20,
                   _fc1_w_20, _fc1_b_20, _fc2_w_20, _fc2_b_20,
                   dim, heads, hidden, 1)
    # Layer 21
    _ln1_w_21: ptr = w_slice_1d(data, 338826240, 1280)
    _ln1_b_21: ptr = w_slice_1d(data, 338824960, 1280)
    _q_w_21: ptr = w_slice_2d(data, 355224320, 1280, 1280)
    _q_b_21: ptr = w_slice_1d(data, 355223040, 1280)
    _k_w_21: ptr = w_slice_2d(data, 351944960, 1280, 1280)
    _k_b_21: ptr = w_slice_1d(data, 351943680, 1280)
    _v_w_21: ptr = w_slice_2d(data, 356864000, 1280, 1280)
    _v_b_21: ptr = w_slice_1d(data, 356862720, 1280)
    _out_w_21: ptr = w_slice_2d(data, 353584640, 1280, 1280)
    _out_b_21: ptr = w_slice_1d(data, 353583360, 1280)
    _ln2_w_21: ptr = w_slice_1d(data, 338828800, 1280)
    _ln2_b_21: ptr = w_slice_1d(data, 338827520, 1280)
    _fc1_w_21: ptr = w_slice_2d(data, 338835200, 5120, 1280)
    _fc1_b_21: ptr = w_slice_1d(data, 338830080, 5120)
    _fc2_w_21: ptr = w_slice_2d(data, 345390080, 1280, 5120)
    _fc2_b_21: ptr = w_slice_1d(data, 345388800, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_21, _ln1_b_21,
                   _q_w_21, _q_b_21, _k_w_21, _k_b_21, _v_w_21, _v_b_21,
                   _out_w_21, _out_b_21,
                   _ln2_w_21, _ln2_b_21,
                   _fc1_w_21, _fc1_b_21, _fc2_w_21, _fc2_b_21,
                   dim, heads, hidden, 1)
    # Layer 22
    _ln1_w_22: ptr = w_slice_1d(data, 358503680, 1280)
    _ln1_b_22: ptr = w_slice_1d(data, 358502400, 1280)
    _q_w_22: ptr = w_slice_2d(data, 374901760, 1280, 1280)
    _q_b_22: ptr = w_slice_1d(data, 374900480, 1280)
    _k_w_22: ptr = w_slice_2d(data, 371622400, 1280, 1280)
    _k_b_22: ptr = w_slice_1d(data, 371621120, 1280)
    _v_w_22: ptr = w_slice_2d(data, 376541440, 1280, 1280)
    _v_b_22: ptr = w_slice_1d(data, 376540160, 1280)
    _out_w_22: ptr = w_slice_2d(data, 373262080, 1280, 1280)
    _out_b_22: ptr = w_slice_1d(data, 373260800, 1280)
    _ln2_w_22: ptr = w_slice_1d(data, 358506240, 1280)
    _ln2_b_22: ptr = w_slice_1d(data, 358504960, 1280)
    _fc1_w_22: ptr = w_slice_2d(data, 358512640, 5120, 1280)
    _fc1_b_22: ptr = w_slice_1d(data, 358507520, 5120)
    _fc2_w_22: ptr = w_slice_2d(data, 365067520, 1280, 5120)
    _fc2_b_22: ptr = w_slice_1d(data, 365066240, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_22, _ln1_b_22,
                   _q_w_22, _q_b_22, _k_w_22, _k_b_22, _v_w_22, _v_b_22,
                   _out_w_22, _out_b_22,
                   _ln2_w_22, _ln2_b_22,
                   _fc1_w_22, _fc1_b_22, _fc2_w_22, _fc2_b_22,
                   dim, heads, hidden, 1)
    # Layer 23
    _ln1_w_23: ptr = w_slice_1d(data, 378181120, 1280)
    _ln1_b_23: ptr = w_slice_1d(data, 378179840, 1280)
    _q_w_23: ptr = w_slice_2d(data, 394579200, 1280, 1280)
    _q_b_23: ptr = w_slice_1d(data, 394577920, 1280)
    _k_w_23: ptr = w_slice_2d(data, 391299840, 1280, 1280)
    _k_b_23: ptr = w_slice_1d(data, 391298560, 1280)
    _v_w_23: ptr = w_slice_2d(data, 396218880, 1280, 1280)
    _v_b_23: ptr = w_slice_1d(data, 396217600, 1280)
    _out_w_23: ptr = w_slice_2d(data, 392939520, 1280, 1280)
    _out_b_23: ptr = w_slice_1d(data, 392938240, 1280)
    _ln2_w_23: ptr = w_slice_1d(data, 378183680, 1280)
    _ln2_b_23: ptr = w_slice_1d(data, 378182400, 1280)
    _fc1_w_23: ptr = w_slice_2d(data, 378190080, 5120, 1280)
    _fc1_b_23: ptr = w_slice_1d(data, 378184960, 5120)
    _fc2_w_23: ptr = w_slice_2d(data, 384744960, 1280, 5120)
    _fc2_b_23: ptr = w_slice_1d(data, 384743680, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_23, _ln1_b_23,
                   _q_w_23, _q_b_23, _k_w_23, _k_b_23, _v_w_23, _v_b_23,
                   _out_w_23, _out_b_23,
                   _ln2_w_23, _ln2_b_23,
                   _fc1_w_23, _fc1_b_23, _fc2_w_23, _fc2_b_23,
                   dim, heads, hidden, 1)
    # Layer 24
    _ln1_w_24: ptr = w_slice_1d(data, 397858560, 1280)
    _ln1_b_24: ptr = w_slice_1d(data, 397857280, 1280)
    _q_w_24: ptr = w_slice_2d(data, 414256640, 1280, 1280)
    _q_b_24: ptr = w_slice_1d(data, 414255360, 1280)
    _k_w_24: ptr = w_slice_2d(data, 410977280, 1280, 1280)
    _k_b_24: ptr = w_slice_1d(data, 410976000, 1280)
    _v_w_24: ptr = w_slice_2d(data, 415896320, 1280, 1280)
    _v_b_24: ptr = w_slice_1d(data, 415895040, 1280)
    _out_w_24: ptr = w_slice_2d(data, 412616960, 1280, 1280)
    _out_b_24: ptr = w_slice_1d(data, 412615680, 1280)
    _ln2_w_24: ptr = w_slice_1d(data, 397861120, 1280)
    _ln2_b_24: ptr = w_slice_1d(data, 397859840, 1280)
    _fc1_w_24: ptr = w_slice_2d(data, 397867520, 5120, 1280)
    _fc1_b_24: ptr = w_slice_1d(data, 397862400, 5120)
    _fc2_w_24: ptr = w_slice_2d(data, 404422400, 1280, 5120)
    _fc2_b_24: ptr = w_slice_1d(data, 404421120, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_24, _ln1_b_24,
                   _q_w_24, _q_b_24, _k_w_24, _k_b_24, _v_w_24, _v_b_24,
                   _out_w_24, _out_b_24,
                   _ln2_w_24, _ln2_b_24,
                   _fc1_w_24, _fc1_b_24, _fc2_w_24, _fc2_b_24,
                   dim, heads, hidden, 1)
    # Layer 25
    _ln1_w_25: ptr = w_slice_1d(data, 417536000, 1280)
    _ln1_b_25: ptr = w_slice_1d(data, 417534720, 1280)
    _q_w_25: ptr = w_slice_2d(data, 433934080, 1280, 1280)
    _q_b_25: ptr = w_slice_1d(data, 433932800, 1280)
    _k_w_25: ptr = w_slice_2d(data, 430654720, 1280, 1280)
    _k_b_25: ptr = w_slice_1d(data, 430653440, 1280)
    _v_w_25: ptr = w_slice_2d(data, 435573760, 1280, 1280)
    _v_b_25: ptr = w_slice_1d(data, 435572480, 1280)
    _out_w_25: ptr = w_slice_2d(data, 432294400, 1280, 1280)
    _out_b_25: ptr = w_slice_1d(data, 432293120, 1280)
    _ln2_w_25: ptr = w_slice_1d(data, 417538560, 1280)
    _ln2_b_25: ptr = w_slice_1d(data, 417537280, 1280)
    _fc1_w_25: ptr = w_slice_2d(data, 417544960, 5120, 1280)
    _fc1_b_25: ptr = w_slice_1d(data, 417539840, 5120)
    _fc2_w_25: ptr = w_slice_2d(data, 424099840, 1280, 5120)
    _fc2_b_25: ptr = w_slice_1d(data, 424098560, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_25, _ln1_b_25,
                   _q_w_25, _q_b_25, _k_w_25, _k_b_25, _v_w_25, _v_b_25,
                   _out_w_25, _out_b_25,
                   _ln2_w_25, _ln2_b_25,
                   _fc1_w_25, _fc1_b_25, _fc2_w_25, _fc2_b_25,
                   dim, heads, hidden, 1)
    # Layer 26
    _ln1_w_26: ptr = w_slice_1d(data, 437213440, 1280)
    _ln1_b_26: ptr = w_slice_1d(data, 437212160, 1280)
    _q_w_26: ptr = w_slice_2d(data, 453611520, 1280, 1280)
    _q_b_26: ptr = w_slice_1d(data, 453610240, 1280)
    _k_w_26: ptr = w_slice_2d(data, 450332160, 1280, 1280)
    _k_b_26: ptr = w_slice_1d(data, 450330880, 1280)
    _v_w_26: ptr = w_slice_2d(data, 455251200, 1280, 1280)
    _v_b_26: ptr = w_slice_1d(data, 455249920, 1280)
    _out_w_26: ptr = w_slice_2d(data, 451971840, 1280, 1280)
    _out_b_26: ptr = w_slice_1d(data, 451970560, 1280)
    _ln2_w_26: ptr = w_slice_1d(data, 437216000, 1280)
    _ln2_b_26: ptr = w_slice_1d(data, 437214720, 1280)
    _fc1_w_26: ptr = w_slice_2d(data, 437222400, 5120, 1280)
    _fc1_b_26: ptr = w_slice_1d(data, 437217280, 5120)
    _fc2_w_26: ptr = w_slice_2d(data, 443777280, 1280, 5120)
    _fc2_b_26: ptr = w_slice_1d(data, 443776000, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_26, _ln1_b_26,
                   _q_w_26, _q_b_26, _k_w_26, _k_b_26, _v_w_26, _v_b_26,
                   _out_w_26, _out_b_26,
                   _ln2_w_26, _ln2_b_26,
                   _fc1_w_26, _fc1_b_26, _fc2_w_26, _fc2_b_26,
                   dim, heads, hidden, 1)
    # Layer 27
    _ln1_w_27: ptr = w_slice_1d(data, 456890880, 1280)
    _ln1_b_27: ptr = w_slice_1d(data, 456889600, 1280)
    _q_w_27: ptr = w_slice_2d(data, 473288960, 1280, 1280)
    _q_b_27: ptr = w_slice_1d(data, 473287680, 1280)
    _k_w_27: ptr = w_slice_2d(data, 470009600, 1280, 1280)
    _k_b_27: ptr = w_slice_1d(data, 470008320, 1280)
    _v_w_27: ptr = w_slice_2d(data, 474928640, 1280, 1280)
    _v_b_27: ptr = w_slice_1d(data, 474927360, 1280)
    _out_w_27: ptr = w_slice_2d(data, 471649280, 1280, 1280)
    _out_b_27: ptr = w_slice_1d(data, 471648000, 1280)
    _ln2_w_27: ptr = w_slice_1d(data, 456893440, 1280)
    _ln2_b_27: ptr = w_slice_1d(data, 456892160, 1280)
    _fc1_w_27: ptr = w_slice_2d(data, 456899840, 5120, 1280)
    _fc1_b_27: ptr = w_slice_1d(data, 456894720, 5120)
    _fc2_w_27: ptr = w_slice_2d(data, 463454720, 1280, 5120)
    _fc2_b_27: ptr = w_slice_1d(data, 463453440, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_27, _ln1_b_27,
                   _q_w_27, _q_b_27, _k_w_27, _k_b_27, _v_w_27, _v_b_27,
                   _out_w_27, _out_b_27,
                   _ln2_w_27, _ln2_b_27,
                   _fc1_w_27, _fc1_b_27, _fc2_w_27, _fc2_b_27,
                   dim, heads, hidden, 1)
    # Layer 28
    _ln1_w_28: ptr = w_slice_1d(data, 476568320, 1280)
    _ln1_b_28: ptr = w_slice_1d(data, 476567040, 1280)
    _q_w_28: ptr = w_slice_2d(data, 492966400, 1280, 1280)
    _q_b_28: ptr = w_slice_1d(data, 492965120, 1280)
    _k_w_28: ptr = w_slice_2d(data, 489687040, 1280, 1280)
    _k_b_28: ptr = w_slice_1d(data, 489685760, 1280)
    _v_w_28: ptr = w_slice_2d(data, 494606080, 1280, 1280)
    _v_b_28: ptr = w_slice_1d(data, 494604800, 1280)
    _out_w_28: ptr = w_slice_2d(data, 491326720, 1280, 1280)
    _out_b_28: ptr = w_slice_1d(data, 491325440, 1280)
    _ln2_w_28: ptr = w_slice_1d(data, 476570880, 1280)
    _ln2_b_28: ptr = w_slice_1d(data, 476569600, 1280)
    _fc1_w_28: ptr = w_slice_2d(data, 476577280, 5120, 1280)
    _fc1_b_28: ptr = w_slice_1d(data, 476572160, 5120)
    _fc2_w_28: ptr = w_slice_2d(data, 483132160, 1280, 5120)
    _fc2_b_28: ptr = w_slice_1d(data, 483130880, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_28, _ln1_b_28,
                   _q_w_28, _q_b_28, _k_w_28, _k_b_28, _v_w_28, _v_b_28,
                   _out_w_28, _out_b_28,
                   _ln2_w_28, _ln2_b_28,
                   _fc1_w_28, _fc1_b_28, _fc2_w_28, _fc2_b_28,
                   dim, heads, hidden, 1)
    # Layer 29
    _ln1_w_29: ptr = w_slice_1d(data, 496245760, 1280)
    _ln1_b_29: ptr = w_slice_1d(data, 496244480, 1280)
    _q_w_29: ptr = w_slice_2d(data, 512643840, 1280, 1280)
    _q_b_29: ptr = w_slice_1d(data, 512642560, 1280)
    _k_w_29: ptr = w_slice_2d(data, 509364480, 1280, 1280)
    _k_b_29: ptr = w_slice_1d(data, 509363200, 1280)
    _v_w_29: ptr = w_slice_2d(data, 514283520, 1280, 1280)
    _v_b_29: ptr = w_slice_1d(data, 514282240, 1280)
    _out_w_29: ptr = w_slice_2d(data, 511004160, 1280, 1280)
    _out_b_29: ptr = w_slice_1d(data, 511002880, 1280)
    _ln2_w_29: ptr = w_slice_1d(data, 496248320, 1280)
    _ln2_b_29: ptr = w_slice_1d(data, 496247040, 1280)
    _fc1_w_29: ptr = w_slice_2d(data, 496254720, 5120, 1280)
    _fc1_b_29: ptr = w_slice_1d(data, 496249600, 5120)
    _fc2_w_29: ptr = w_slice_2d(data, 502809600, 1280, 5120)
    _fc2_b_29: ptr = w_slice_1d(data, 502808320, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_29, _ln1_b_29,
                   _q_w_29, _q_b_29, _k_w_29, _k_b_29, _v_w_29, _v_b_29,
                   _out_w_29, _out_b_29,
                   _ln2_w_29, _ln2_b_29,
                   _fc1_w_29, _fc1_b_29, _fc2_w_29, _fc2_b_29,
                   dim, heads, hidden, 1)
    # Layer 30
    _ln1_w_30: ptr = w_slice_1d(data, 535600640, 1280)
    _ln1_b_30: ptr = w_slice_1d(data, 535599360, 1280)
    _q_w_30: ptr = w_slice_2d(data, 551998720, 1280, 1280)
    _q_b_30: ptr = w_slice_1d(data, 551997440, 1280)
    _k_w_30: ptr = w_slice_2d(data, 548719360, 1280, 1280)
    _k_b_30: ptr = w_slice_1d(data, 548718080, 1280)
    _v_w_30: ptr = w_slice_2d(data, 553638400, 1280, 1280)
    _v_b_30: ptr = w_slice_1d(data, 553637120, 1280)
    _out_w_30: ptr = w_slice_2d(data, 550359040, 1280, 1280)
    _out_b_30: ptr = w_slice_1d(data, 550357760, 1280)
    _ln2_w_30: ptr = w_slice_1d(data, 535603200, 1280)
    _ln2_b_30: ptr = w_slice_1d(data, 535601920, 1280)
    _fc1_w_30: ptr = w_slice_2d(data, 535609600, 5120, 1280)
    _fc1_b_30: ptr = w_slice_1d(data, 535604480, 5120)
    _fc2_w_30: ptr = w_slice_2d(data, 542164480, 1280, 5120)
    _fc2_b_30: ptr = w_slice_1d(data, 542163200, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_30, _ln1_b_30,
                   _q_w_30, _q_b_30, _k_w_30, _k_b_30, _v_w_30, _v_b_30,
                   _out_w_30, _out_b_30,
                   _ln2_w_30, _ln2_b_30,
                   _fc1_w_30, _fc1_b_30, _fc2_w_30, _fc2_b_30,
                   dim, heads, hidden, 1)
    penultimate = st_clone(x)
    # Layer 31
    _ln1_w_31: ptr = w_slice_1d(data, 555278080, 1280)
    _ln1_b_31: ptr = w_slice_1d(data, 555276800, 1280)
    _q_w_31: ptr = w_slice_2d(data, 571676160, 1280, 1280)
    _q_b_31: ptr = w_slice_1d(data, 571674880, 1280)
    _k_w_31: ptr = w_slice_2d(data, 568396800, 1280, 1280)
    _k_b_31: ptr = w_slice_1d(data, 568395520, 1280)
    _v_w_31: ptr = w_slice_2d(data, 573315840, 1280, 1280)
    _v_b_31: ptr = w_slice_1d(data, 573314560, 1280)
    _out_w_31: ptr = w_slice_2d(data, 570036480, 1280, 1280)
    _out_b_31: ptr = w_slice_1d(data, 570035200, 1280)
    _ln2_w_31: ptr = w_slice_1d(data, 555280640, 1280)
    _ln2_b_31: ptr = w_slice_1d(data, 555279360, 1280)
    _fc1_w_31: ptr = w_slice_2d(data, 555287040, 5120, 1280)
    _fc1_b_31: ptr = w_slice_1d(data, 555281920, 5120)
    _fc2_w_31: ptr = w_slice_2d(data, 561841920, 1280, 5120)
    _fc2_b_31: ptr = w_slice_1d(data, 561840640, 1280)
    x = clip_layer(x, mask,
                   _ln1_w_31, _ln1_b_31,
                   _q_w_31, _q_b_31, _k_w_31, _k_b_31, _v_w_31, _v_b_31,
                   _out_w_31, _out_b_31,
                   _ln2_w_31, _ln2_b_31,
                   _fc1_w_31, _fc1_b_31, _fc2_w_31, _fc2_b_31,
                   dim, heads, hidden, 1)
    _fn_w: ptr = w_slice_1d(data, 693020160, 1280)
    _fn_b: ptr = w_slice_1d(data, 693018880, 1280)
    _fn_out: ptr = st_layer_norm(x, _fn_w, _fn_b, 1e-5)
    # extract pooled at EOT position 4 -> [1,1,1280]
    _pooled_raw: ptr = st_slice(_fn_out, 1, 4, 5)
    _dims: list[int] = make_int_array(2)
    int_array_set(_dims, 0, 1)
    int_array_set(_dims, 1, 1280)
    _pooled_2d: ptr = st_reshape(_pooled_raw, _dims, 2)
    # text projection: [1280,1280]
    _text_proj_w: ptr = w_slice_2d(data, 693021440, 1280, 1280)
    _pooled: ptr = st_linear(_pooled_2d, _text_proj_w, 0)
    st_tensor_free(_text_proj_w)
    st_tensor_free(_pooled_2d)
    st_tensor_free(_pooled_raw)
    st_tensor_free(mask); st_tensor_free(_fn_out); st_tensor_free(x)
    _out_arr: ptr = make_ptr_array(2)
    ptr_array_set(_out_arr, 0, penultimate)
    ptr_array_set(_out_arr, 1, _pooled)
    return _out_arr

def clip_l_encode(tokens: list[float], data: list[float]) -> ptr:
    seq_len: int = 77
    dim: int = 768
    heads: int = 12
    hidden: int = 3072
    x: ptr = clip_embed(tokens, data, 694719053, 694659840, seq_len, dim)
    mask: ptr = make_causal_mask(seq_len, heads)
    penultimate: ptr
    # Layer 0
    _ln1_w_0: ptr = w_slice_1d(data, 732665165, 768)
    _ln1_b_0: ptr = w_slice_1d(data, 732664397, 768)
    _q_w_0: ptr = w_slice_2d(data, 738571853, 768, 768)
    _q_b_0: ptr = w_slice_1d(data, 738571085, 768)
    _k_w_0: ptr = w_slice_2d(data, 737390669, 768, 768)
    _k_b_0: ptr = w_slice_1d(data, 737389901, 768)
    _v_w_0: ptr = w_slice_2d(data, 739162445, 768, 768)
    _v_b_0: ptr = w_slice_1d(data, 739161677, 768)
    _out_w_0: ptr = w_slice_2d(data, 737981261, 768, 768)
    _out_b_0: ptr = w_slice_1d(data, 737980493, 768)
    _ln2_w_0: ptr = w_slice_1d(data, 732666701, 768)
    _ln2_b_0: ptr = w_slice_1d(data, 732665933, 768)
    _fc1_w_0: ptr = w_slice_2d(data, 732670541, 3072, 768)
    _fc1_b_0: ptr = w_slice_1d(data, 732667469, 3072)
    _fc2_w_0: ptr = w_slice_2d(data, 735030605, 768, 3072)
    _fc2_b_0: ptr = w_slice_1d(data, 735029837, 768)
    x = clip_layer(x, mask,
                   _ln1_w_0, _ln1_b_0,
                   _q_w_0, _q_b_0, _k_w_0, _k_b_0, _v_w_0, _v_b_0,
                   _out_w_0, _out_b_0,
                   _ln2_w_0, _ln2_b_0,
                   _fc1_w_0, _fc1_b_0, _fc2_w_0, _fc2_b_0,
                   dim, heads, hidden, 0)
    # Layer 1
    _ln1_w_1: ptr = w_slice_1d(data, 739753037, 768)
    _ln1_b_1: ptr = w_slice_1d(data, 739752269, 768)
    _q_w_1: ptr = w_slice_2d(data, 745659725, 768, 768)
    _q_b_1: ptr = w_slice_1d(data, 745658957, 768)
    _k_w_1: ptr = w_slice_2d(data, 744478541, 768, 768)
    _k_b_1: ptr = w_slice_1d(data, 744477773, 768)
    _v_w_1: ptr = w_slice_2d(data, 746250317, 768, 768)
    _v_b_1: ptr = w_slice_1d(data, 746249549, 768)
    _out_w_1: ptr = w_slice_2d(data, 745069133, 768, 768)
    _out_b_1: ptr = w_slice_1d(data, 745068365, 768)
    _ln2_w_1: ptr = w_slice_1d(data, 739754573, 768)
    _ln2_b_1: ptr = w_slice_1d(data, 739753805, 768)
    _fc1_w_1: ptr = w_slice_2d(data, 739758413, 3072, 768)
    _fc1_b_1: ptr = w_slice_1d(data, 739755341, 3072)
    _fc2_w_1: ptr = w_slice_2d(data, 742118477, 768, 3072)
    _fc2_b_1: ptr = w_slice_1d(data, 742117709, 768)
    x = clip_layer(x, mask,
                   _ln1_w_1, _ln1_b_1,
                   _q_w_1, _q_b_1, _k_w_1, _k_b_1, _v_w_1, _v_b_1,
                   _out_w_1, _out_b_1,
                   _ln2_w_1, _ln2_b_1,
                   _fc1_w_1, _fc1_b_1, _fc2_w_1, _fc2_b_1,
                   dim, heads, hidden, 0)
    # Layer 2
    _ln1_w_2: ptr = w_slice_1d(data, 761016653, 768)
    _ln1_b_2: ptr = w_slice_1d(data, 761015885, 768)
    _q_w_2: ptr = w_slice_2d(data, 766923341, 768, 768)
    _q_b_2: ptr = w_slice_1d(data, 766922573, 768)
    _k_w_2: ptr = w_slice_2d(data, 765742157, 768, 768)
    _k_b_2: ptr = w_slice_1d(data, 765741389, 768)
    _v_w_2: ptr = w_slice_2d(data, 767513933, 768, 768)
    _v_b_2: ptr = w_slice_1d(data, 767513165, 768)
    _out_w_2: ptr = w_slice_2d(data, 766332749, 768, 768)
    _out_b_2: ptr = w_slice_1d(data, 766331981, 768)
    _ln2_w_2: ptr = w_slice_1d(data, 761018189, 768)
    _ln2_b_2: ptr = w_slice_1d(data, 761017421, 768)
    _fc1_w_2: ptr = w_slice_2d(data, 761022029, 3072, 768)
    _fc1_b_2: ptr = w_slice_1d(data, 761018957, 3072)
    _fc2_w_2: ptr = w_slice_2d(data, 763382093, 768, 3072)
    _fc2_b_2: ptr = w_slice_1d(data, 763381325, 768)
    x = clip_layer(x, mask,
                   _ln1_w_2, _ln1_b_2,
                   _q_w_2, _q_b_2, _k_w_2, _k_b_2, _v_w_2, _v_b_2,
                   _out_w_2, _out_b_2,
                   _ln2_w_2, _ln2_b_2,
                   _fc1_w_2, _fc1_b_2, _fc2_w_2, _fc2_b_2,
                   dim, heads, hidden, 0)
    # Layer 3
    _ln1_w_3: ptr = w_slice_1d(data, 768104525, 768)
    _ln1_b_3: ptr = w_slice_1d(data, 768103757, 768)
    _q_w_3: ptr = w_slice_2d(data, 774011213, 768, 768)
    _q_b_3: ptr = w_slice_1d(data, 774010445, 768)
    _k_w_3: ptr = w_slice_2d(data, 772830029, 768, 768)
    _k_b_3: ptr = w_slice_1d(data, 772829261, 768)
    _v_w_3: ptr = w_slice_2d(data, 774601805, 768, 768)
    _v_b_3: ptr = w_slice_1d(data, 774601037, 768)
    _out_w_3: ptr = w_slice_2d(data, 773420621, 768, 768)
    _out_b_3: ptr = w_slice_1d(data, 773419853, 768)
    _ln2_w_3: ptr = w_slice_1d(data, 768106061, 768)
    _ln2_b_3: ptr = w_slice_1d(data, 768105293, 768)
    _fc1_w_3: ptr = w_slice_2d(data, 768109901, 3072, 768)
    _fc1_b_3: ptr = w_slice_1d(data, 768106829, 3072)
    _fc2_w_3: ptr = w_slice_2d(data, 770469965, 768, 3072)
    _fc2_b_3: ptr = w_slice_1d(data, 770469197, 768)
    x = clip_layer(x, mask,
                   _ln1_w_3, _ln1_b_3,
                   _q_w_3, _q_b_3, _k_w_3, _k_b_3, _v_w_3, _v_b_3,
                   _out_w_3, _out_b_3,
                   _ln2_w_3, _ln2_b_3,
                   _fc1_w_3, _fc1_b_3, _fc2_w_3, _fc2_b_3,
                   dim, heads, hidden, 0)
    # Layer 4
    _ln1_w_4: ptr = w_slice_1d(data, 775192397, 768)
    _ln1_b_4: ptr = w_slice_1d(data, 775191629, 768)
    _q_w_4: ptr = w_slice_2d(data, 781099085, 768, 768)
    _q_b_4: ptr = w_slice_1d(data, 781098317, 768)
    _k_w_4: ptr = w_slice_2d(data, 779917901, 768, 768)
    _k_b_4: ptr = w_slice_1d(data, 779917133, 768)
    _v_w_4: ptr = w_slice_2d(data, 781689677, 768, 768)
    _v_b_4: ptr = w_slice_1d(data, 781688909, 768)
    _out_w_4: ptr = w_slice_2d(data, 780508493, 768, 768)
    _out_b_4: ptr = w_slice_1d(data, 780507725, 768)
    _ln2_w_4: ptr = w_slice_1d(data, 775193933, 768)
    _ln2_b_4: ptr = w_slice_1d(data, 775193165, 768)
    _fc1_w_4: ptr = w_slice_2d(data, 775197773, 3072, 768)
    _fc1_b_4: ptr = w_slice_1d(data, 775194701, 3072)
    _fc2_w_4: ptr = w_slice_2d(data, 777557837, 768, 3072)
    _fc2_b_4: ptr = w_slice_1d(data, 777557069, 768)
    x = clip_layer(x, mask,
                   _ln1_w_4, _ln1_b_4,
                   _q_w_4, _q_b_4, _k_w_4, _k_b_4, _v_w_4, _v_b_4,
                   _out_w_4, _out_b_4,
                   _ln2_w_4, _ln2_b_4,
                   _fc1_w_4, _fc1_b_4, _fc2_w_4, _fc2_b_4,
                   dim, heads, hidden, 0)
    # Layer 5
    _ln1_w_5: ptr = w_slice_1d(data, 782280269, 768)
    _ln1_b_5: ptr = w_slice_1d(data, 782279501, 768)
    _q_w_5: ptr = w_slice_2d(data, 788186957, 768, 768)
    _q_b_5: ptr = w_slice_1d(data, 788186189, 768)
    _k_w_5: ptr = w_slice_2d(data, 787005773, 768, 768)
    _k_b_5: ptr = w_slice_1d(data, 787005005, 768)
    _v_w_5: ptr = w_slice_2d(data, 788777549, 768, 768)
    _v_b_5: ptr = w_slice_1d(data, 788776781, 768)
    _out_w_5: ptr = w_slice_2d(data, 787596365, 768, 768)
    _out_b_5: ptr = w_slice_1d(data, 787595597, 768)
    _ln2_w_5: ptr = w_slice_1d(data, 782281805, 768)
    _ln2_b_5: ptr = w_slice_1d(data, 782281037, 768)
    _fc1_w_5: ptr = w_slice_2d(data, 782285645, 3072, 768)
    _fc1_b_5: ptr = w_slice_1d(data, 782282573, 3072)
    _fc2_w_5: ptr = w_slice_2d(data, 784645709, 768, 3072)
    _fc2_b_5: ptr = w_slice_1d(data, 784644941, 768)
    x = clip_layer(x, mask,
                   _ln1_w_5, _ln1_b_5,
                   _q_w_5, _q_b_5, _k_w_5, _k_b_5, _v_w_5, _v_b_5,
                   _out_w_5, _out_b_5,
                   _ln2_w_5, _ln2_b_5,
                   _fc1_w_5, _fc1_b_5, _fc2_w_5, _fc2_b_5,
                   dim, heads, hidden, 0)
    # Layer 6
    _ln1_w_6: ptr = w_slice_1d(data, 789368141, 768)
    _ln1_b_6: ptr = w_slice_1d(data, 789367373, 768)
    _q_w_6: ptr = w_slice_2d(data, 795274829, 768, 768)
    _q_b_6: ptr = w_slice_1d(data, 795274061, 768)
    _k_w_6: ptr = w_slice_2d(data, 794093645, 768, 768)
    _k_b_6: ptr = w_slice_1d(data, 794092877, 768)
    _v_w_6: ptr = w_slice_2d(data, 795865421, 768, 768)
    _v_b_6: ptr = w_slice_1d(data, 795864653, 768)
    _out_w_6: ptr = w_slice_2d(data, 794684237, 768, 768)
    _out_b_6: ptr = w_slice_1d(data, 794683469, 768)
    _ln2_w_6: ptr = w_slice_1d(data, 789369677, 768)
    _ln2_b_6: ptr = w_slice_1d(data, 789368909, 768)
    _fc1_w_6: ptr = w_slice_2d(data, 789373517, 3072, 768)
    _fc1_b_6: ptr = w_slice_1d(data, 789370445, 3072)
    _fc2_w_6: ptr = w_slice_2d(data, 791733581, 768, 3072)
    _fc2_b_6: ptr = w_slice_1d(data, 791732813, 768)
    x = clip_layer(x, mask,
                   _ln1_w_6, _ln1_b_6,
                   _q_w_6, _q_b_6, _k_w_6, _k_b_6, _v_w_6, _v_b_6,
                   _out_w_6, _out_b_6,
                   _ln2_w_6, _ln2_b_6,
                   _fc1_w_6, _fc1_b_6, _fc2_w_6, _fc2_b_6,
                   dim, heads, hidden, 0)
    # Layer 7
    _ln1_w_7: ptr = w_slice_1d(data, 796456013, 768)
    _ln1_b_7: ptr = w_slice_1d(data, 796455245, 768)
    _q_w_7: ptr = w_slice_2d(data, 802362701, 768, 768)
    _q_b_7: ptr = w_slice_1d(data, 802361933, 768)
    _k_w_7: ptr = w_slice_2d(data, 801181517, 768, 768)
    _k_b_7: ptr = w_slice_1d(data, 801180749, 768)
    _v_w_7: ptr = w_slice_2d(data, 802953293, 768, 768)
    _v_b_7: ptr = w_slice_1d(data, 802952525, 768)
    _out_w_7: ptr = w_slice_2d(data, 801772109, 768, 768)
    _out_b_7: ptr = w_slice_1d(data, 801771341, 768)
    _ln2_w_7: ptr = w_slice_1d(data, 796457549, 768)
    _ln2_b_7: ptr = w_slice_1d(data, 796456781, 768)
    _fc1_w_7: ptr = w_slice_2d(data, 796461389, 3072, 768)
    _fc1_b_7: ptr = w_slice_1d(data, 796458317, 3072)
    _fc2_w_7: ptr = w_slice_2d(data, 798821453, 768, 3072)
    _fc2_b_7: ptr = w_slice_1d(data, 798820685, 768)
    x = clip_layer(x, mask,
                   _ln1_w_7, _ln1_b_7,
                   _q_w_7, _q_b_7, _k_w_7, _k_b_7, _v_w_7, _v_b_7,
                   _out_w_7, _out_b_7,
                   _ln2_w_7, _ln2_b_7,
                   _fc1_w_7, _fc1_b_7, _fc2_w_7, _fc2_b_7,
                   dim, heads, hidden, 0)
    # Layer 8
    _ln1_w_8: ptr = w_slice_1d(data, 803543885, 768)
    _ln1_b_8: ptr = w_slice_1d(data, 803543117, 768)
    _q_w_8: ptr = w_slice_2d(data, 809450573, 768, 768)
    _q_b_8: ptr = w_slice_1d(data, 809449805, 768)
    _k_w_8: ptr = w_slice_2d(data, 808269389, 768, 768)
    _k_b_8: ptr = w_slice_1d(data, 808268621, 768)
    _v_w_8: ptr = w_slice_2d(data, 810041165, 768, 768)
    _v_b_8: ptr = w_slice_1d(data, 810040397, 768)
    _out_w_8: ptr = w_slice_2d(data, 808859981, 768, 768)
    _out_b_8: ptr = w_slice_1d(data, 808859213, 768)
    _ln2_w_8: ptr = w_slice_1d(data, 803545421, 768)
    _ln2_b_8: ptr = w_slice_1d(data, 803544653, 768)
    _fc1_w_8: ptr = w_slice_2d(data, 803549261, 3072, 768)
    _fc1_b_8: ptr = w_slice_1d(data, 803546189, 3072)
    _fc2_w_8: ptr = w_slice_2d(data, 805909325, 768, 3072)
    _fc2_b_8: ptr = w_slice_1d(data, 805908557, 768)
    x = clip_layer(x, mask,
                   _ln1_w_8, _ln1_b_8,
                   _q_w_8, _q_b_8, _k_w_8, _k_b_8, _v_w_8, _v_b_8,
                   _out_w_8, _out_b_8,
                   _ln2_w_8, _ln2_b_8,
                   _fc1_w_8, _fc1_b_8, _fc2_w_8, _fc2_b_8,
                   dim, heads, hidden, 0)
    # Layer 9
    _ln1_w_9: ptr = w_slice_1d(data, 810631757, 768)
    _ln1_b_9: ptr = w_slice_1d(data, 810630989, 768)
    _q_w_9: ptr = w_slice_2d(data, 816538445, 768, 768)
    _q_b_9: ptr = w_slice_1d(data, 816537677, 768)
    _k_w_9: ptr = w_slice_2d(data, 815357261, 768, 768)
    _k_b_9: ptr = w_slice_1d(data, 815356493, 768)
    _v_w_9: ptr = w_slice_2d(data, 817129037, 768, 768)
    _v_b_9: ptr = w_slice_1d(data, 817128269, 768)
    _out_w_9: ptr = w_slice_2d(data, 815947853, 768, 768)
    _out_b_9: ptr = w_slice_1d(data, 815947085, 768)
    _ln2_w_9: ptr = w_slice_1d(data, 810633293, 768)
    _ln2_b_9: ptr = w_slice_1d(data, 810632525, 768)
    _fc1_w_9: ptr = w_slice_2d(data, 810637133, 3072, 768)
    _fc1_b_9: ptr = w_slice_1d(data, 810634061, 3072)
    _fc2_w_9: ptr = w_slice_2d(data, 812997197, 768, 3072)
    _fc2_b_9: ptr = w_slice_1d(data, 812996429, 768)
    x = clip_layer(x, mask,
                   _ln1_w_9, _ln1_b_9,
                   _q_w_9, _q_b_9, _k_w_9, _k_b_9, _v_w_9, _v_b_9,
                   _out_w_9, _out_b_9,
                   _ln2_w_9, _ln2_b_9,
                   _fc1_w_9, _fc1_b_9, _fc2_w_9, _fc2_b_9,
                   dim, heads, hidden, 0)
    # Layer 10
    _ln1_w_10: ptr = w_slice_1d(data, 746840909, 768)
    _ln1_b_10: ptr = w_slice_1d(data, 746840141, 768)
    _q_w_10: ptr = w_slice_2d(data, 752747597, 768, 768)
    _q_b_10: ptr = w_slice_1d(data, 752746829, 768)
    _k_w_10: ptr = w_slice_2d(data, 751566413, 768, 768)
    _k_b_10: ptr = w_slice_1d(data, 751565645, 768)
    _v_w_10: ptr = w_slice_2d(data, 753338189, 768, 768)
    _v_b_10: ptr = w_slice_1d(data, 753337421, 768)
    _out_w_10: ptr = w_slice_2d(data, 752157005, 768, 768)
    _out_b_10: ptr = w_slice_1d(data, 752156237, 768)
    _ln2_w_10: ptr = w_slice_1d(data, 746842445, 768)
    _ln2_b_10: ptr = w_slice_1d(data, 746841677, 768)
    _fc1_w_10: ptr = w_slice_2d(data, 746846285, 3072, 768)
    _fc1_b_10: ptr = w_slice_1d(data, 746843213, 3072)
    _fc2_w_10: ptr = w_slice_2d(data, 749206349, 768, 3072)
    _fc2_b_10: ptr = w_slice_1d(data, 749205581, 768)
    x = clip_layer(x, mask,
                   _ln1_w_10, _ln1_b_10,
                   _q_w_10, _q_b_10, _k_w_10, _k_b_10, _v_w_10, _v_b_10,
                   _out_w_10, _out_b_10,
                   _ln2_w_10, _ln2_b_10,
                   _fc1_w_10, _fc1_b_10, _fc2_w_10, _fc2_b_10,
                   dim, heads, hidden, 0)
    penultimate = st_clone(x)
    # Layer 11
    _ln1_w_11: ptr = w_slice_1d(data, 753928781, 768)
    _ln1_b_11: ptr = w_slice_1d(data, 753928013, 768)
    _q_w_11: ptr = w_slice_2d(data, 759835469, 768, 768)
    _q_b_11: ptr = w_slice_1d(data, 759834701, 768)
    _k_w_11: ptr = w_slice_2d(data, 758654285, 768, 768)
    _k_b_11: ptr = w_slice_1d(data, 758653517, 768)
    _v_w_11: ptr = w_slice_2d(data, 760426061, 768, 768)
    _v_b_11: ptr = w_slice_1d(data, 760425293, 768)
    _out_w_11: ptr = w_slice_2d(data, 759244877, 768, 768)
    _out_b_11: ptr = w_slice_1d(data, 759244109, 768)
    _ln2_w_11: ptr = w_slice_1d(data, 753930317, 768)
    _ln2_b_11: ptr = w_slice_1d(data, 753929549, 768)
    _fc1_w_11: ptr = w_slice_2d(data, 753934157, 3072, 768)
    _fc1_b_11: ptr = w_slice_1d(data, 753931085, 3072)
    _fc2_w_11: ptr = w_slice_2d(data, 756294221, 768, 3072)
    _fc2_b_11: ptr = w_slice_1d(data, 756293453, 768)
    x = clip_layer(x, mask,
                   _ln1_w_11, _ln1_b_11,
                   _q_w_11, _q_b_11, _k_w_11, _k_b_11, _v_w_11, _v_b_11,
                   _out_w_11, _out_b_11,
                   _ln2_w_11, _ln2_b_11,
                   _fc1_w_11, _fc1_b_11, _fc2_w_11, _fc2_b_11,
                   dim, heads, hidden, 0)
    _fn_w: ptr = w_slice_1d(data, 817719629, 768)
    _fn_b: ptr = w_slice_1d(data, 817718861, 768)
    _fn_out: ptr = st_layer_norm(x, _fn_w, _fn_b, 1e-5)
    st_tensor_free(mask); st_tensor_free(_fn_out); st_tensor_free(x)
    return penultimate

def clip_encode_lg(tokens_l: list[float], tokens_g: list[float], data: list[float]) -> ptr:
    out_l: ptr = clip_l_encode(tokens_l, data)
    out_g_arr: ptr = clip_g_encode(tokens_g, data)
    out_g: ptr = ptr_array_ref(out_g_arr, 0)
    pooled_g: ptr = ptr_array_ref(out_g_arr, 1)
    ctx: ptr = st_cat_dim(out_l, out_g, 2)  # [1,77,2048]
    st_tensor_free(out_l); st_tensor_free(out_g)
    _out_arr: ptr = make_ptr_array(2)
    ptr_array_set(_out_arr, 0, ctx)
    ptr_array_set(_out_arr, 1, pooled_g)
    return _out_arr
