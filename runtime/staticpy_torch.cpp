// staticpy_torch.cpp — PyTorch C++ API → C 函数包装
// 编译: g++ -O2 -shared -fPIC -o /tmp/libstaticpy_torch.so staticpy_torch.cpp \
//        -I/data/venv/lib/python3.12/site-packages/torch/include \
//        -I/data/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
//        -L/data/venv/lib/python3.12/site-packages/torch/lib -ltorch -lc10 \
//        -Wl,-rpath,/data/venv/lib/python3.12/site-packages/torch/lib
//
// 产出: libstaticpy_torch.so — 被 StaticPy 的 extern fn 调用

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include <cstring>

static int staticpy_device = -1;

static int get_device() {
    if (staticpy_device < 0) {
        try {
            if (torch::cuda::is_available()) staticpy_device = 0;
            else staticpy_device = -2;
        } catch (...) {
            staticpy_device = -2;
        }
    }
    return staticpy_device;
}

static torch::Tensor to_dev(const torch::Tensor& t) {
    if (get_device() >= 0) return t.cuda();
    return t;
}

static torch::Tensor* new_t(const torch::Tensor& t) {
    return new torch::Tensor(to_dev(t));
}

extern "C" {

// ===== Tensor 生命周期 =====

void* st_tensor_create(float* data, int64_t* dims, int ndim) {
    // 从 C float 数组创建 torch tensor
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::from_blob(data, sizes, opts).clone();
    return (void*)new_t(tensor);
}

void* st_tensor_zeros(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::zeros(sizes, torch::kFloat32);
    return (void*)new_t(tensor);
}

void st_tensor_free(void* t) {
    delete (torch::Tensor*)t;
}

int64_t st_tensor_numel(void* t) {
    return ((torch::Tensor*)t)->numel();
}

float* st_tensor_data(void* t) {
    return ((torch::Tensor*)t)->data_ptr<float>();
}

void st_tensor_to_cpu(void* t) {
    *((torch::Tensor*)t) = ((torch::Tensor*)t)->to(torch::kCPU);
}

void st_tensor_to_cuda(void* t) {
    int d = get_device();
    if (d >= 0) *((torch::Tensor*)t) = ((torch::Tensor*)t)->cuda();
}

// ===== 设备管理 =====

int st_cuda_available() {
    return torch::cuda::is_available() ? 1 : 0;
}

int st_cuda_device_count() {
    return torch::cuda::device_count();
}

// ===== 神经网络算子 =====

void* st_conv2d(void* input, void* weight, void* bias,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 int dilation_h, int dilation_w,
                 int groups) {
    auto& inp = *(torch::Tensor*)input;
    auto& w = *(torch::Tensor*)weight;
    c10::optional<torch::Tensor> b;
    if (bias) b = *(torch::Tensor*)bias;
    auto result = torch::conv2d(inp, w, b,
        {stride_h, stride_w}, {pad_h, pad_w}, {dilation_h, dilation_w}, groups);
    return (void*)new torch::Tensor(result);
}

void* st_linear(void* input, void* weight, void* bias) {
    auto& inp = *(torch::Tensor*)input;
    auto& w = *(torch::Tensor*)weight;
    auto result = inp.matmul(w.t());
    if (bias) result = result + *(torch::Tensor*)bias;
    return (void*)new torch::Tensor(result.contiguous());
}

void* st_group_norm(void* input, int num_groups, void* weight, void* bias, double eps) {
    auto& inp = *(torch::Tensor*)input;
    c10::optional<torch::Tensor> w, b;
    if (weight) w = *(torch::Tensor*)weight;
    if (bias) b = *(torch::Tensor*)bias;
    fprintf(stderr, "[st_group_norm] input shape=[");
    for (int i=0;i<inp.dim();i++) fprintf(stderr, "%ld%s", inp.size(i), i+1<inp.dim()?",":"");
    fprintf(stderr, "] sum=%f num_groups=%d eps=%f weight=%p bias=%p\n", inp.sum().item<double>(), num_groups, eps, weight, bias);
    if (w) fprintf(stderr, "  weight sum=%f\n", w->sum().item<double>());
    if (b) fprintf(stderr, "  bias sum=%f\n", b->sum().item<double>());
    auto result = torch::group_norm(inp, num_groups, w, b, eps);
    fprintf(stderr, "[st_group_norm] output sum=%f\n", result.sum().item<double>());
    return (void*)new torch::Tensor(result);
}

void* st_layer_norm(void* input, void* weight, void* bias, double eps) {
    auto& inp = *(torch::Tensor*)input;
    c10::optional<torch::Tensor> w, b;
    if (weight) w = *(torch::Tensor*)weight;
    if (bias) b = *(torch::Tensor*)bias;
    auto result = torch::layer_norm(inp, {inp.size(-1)}, w, b, eps);
    return (void*)new torch::Tensor(result);
}

void* st_mm(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    auto result = ta.mm(tb);
    return (void*)new torch::Tensor(result);
}

void* st_matmul(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    auto result = ta.matmul(tb);
    return (void*)new torch::Tensor(result);
}

void* st_silu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::silu(inp);
    return (void*)new torch::Tensor(result);
}

void* st_relu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::relu(inp);
    return (void*)new torch::Tensor(result);
}

void* st_gelu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::gelu(inp);
    return (void*)new torch::Tensor(result);
}

void* st_sigmoid(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::sigmoid(inp);
    return (void*)new torch::Tensor(result);
}

void* st_tanh(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::tanh(inp);
    return (void*)new torch::Tensor(result);
}

void* st_softmax(void* input, int dim) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::softmax(inp, dim);
    return (void*)new torch::Tensor(result);
}

void* st_max_pool2d(void* input, int k_size, int stride, int pad) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::max_pool2d(inp, {k_size, k_size}, {stride, stride}, {pad, pad});
    return (void*)new torch::Tensor(result);
}

void* st_avg_pool2d(void* input, int k_size, int stride, int pad) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::avg_pool2d(inp, {k_size, k_size}, {stride, stride}, {pad, pad});
    return (void*)new torch::Tensor(result);
}

void* st_upsample_nearest(void* input, int scale_h, int scale_w) {
    auto& inp = *(torch::Tensor*)input;
    auto h = inp.size(-2) * scale_h;
    auto w = inp.size(-1) * scale_w;
    auto result = torch::upsample_nearest2d(inp, {h, w});
    return (void*)new torch::Tensor(result);
}

// ===== 标量运算 =====

void st_add(void* a, void* b) {
    *((torch::Tensor*)a) = *((torch::Tensor*)a) + *((torch::Tensor*)b);
}

void st_mul(void* a, void* b) {
    *((torch::Tensor*)a) = *((torch::Tensor*)a) * *((torch::Tensor*)b);
}

void* st_add_tensor(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    try {
        auto result = ta + tb;
        return (void*)new_t(result);
    } catch (const c10::Error& e) {
        fprintf(stderr, "[st_add_tensor] shape a=[");
        for (int i = 0; i < ta.dim(); i++) {
            if (i) fprintf(stderr, ",");
            fprintf(stderr, "%ld", (long)ta.size(i));
        }
        fprintf(stderr, "] b=[");
        for (int i = 0; i < tb.dim(); i++) {
            if (i) fprintf(stderr, ",");
            fprintf(stderr, "%ld", (long)tb.size(i));
        }
        fprintf(stderr, "]\n");
        throw;
    }
}

void* st_sub_tensor(void* a, void* b) {
    auto result = *((torch::Tensor*)a) - *((torch::Tensor*)b);
    return (void*)new_t(result);
}

void* st_mul_tensor(void* a, void* b) {
    auto result = *((torch::Tensor*)a) * *((torch::Tensor*)b);
    return (void*)new_t(result);
}

void* st_mul_scalar_tensor(void* t, float s) {
    auto result = (*((torch::Tensor*)t)) * s;
    return (void*)new_t(result);
}

void* st_transpose(void* t, int dim0, int dim1) {
    auto result = ((torch::Tensor*)t)->transpose(dim0, dim1).contiguous();
    return (void*)new_t(result);
}

void* st_bmm(void* a, void* b) {
    auto result = ((torch::Tensor*)a)->bmm(*((torch::Tensor*)b));
    return (void*)new_t(result);
}

// ===== 创建/转换 =====

void* st_from_blob(float* data, int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::from_blob(data, sizes, torch::kFloat32).clone();
    return (void*)new_t(tensor);
}

void* st_from_blob_1d(float* data, int64_t d0) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return (void*)new_t(torch::from_blob(data, {d0}, opts).clone());
}

void* st_from_blob_2d(float* data, int64_t d0, int64_t d1) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    // data is row-major [d0, d1]; create [d0, d1] tensor directly (PyTorch default row-major)
    return (void*)new_t(torch::from_blob(data, {d0, d1}, opts).clone());
}

void* st_from_blob_3d(float* data, int64_t d0, int64_t d1, int64_t d2) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return (void*)new_t(torch::from_blob(data, {d0, d1, d2}, opts).clone());
}

void* st_from_blob_4d(float* data, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    return (void*)new_t(torch::from_blob(data, {d0, d1, d2, d3}, opts).clone());
}

void* st_arange(int start, int end, int step) {
    auto result = torch::arange(start, end, step, torch::kFloat32);
    return (void*)new_t(result);
}

void* st_ones(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    return (void*)new_t(torch::ones(sizes, torch::kFloat32));
}

void* st_cat_channel(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    auto result = torch::cat({ta, tb}, 1);
    return (void*)new_t(result);
}

void* st_new(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    return (void*)new_t(torch::empty(sizes, torch::kFloat32));
}

void* st_slice(void* t, int dim, int start, int end) {
    auto result = ((torch::Tensor*)t)->slice(dim, start, end);
    return (void*)new_t(result);
}

void* st_reshape(void* t, int64_t* dims, int ndim) {
    std::vector<int64_t> sizes;
    for (int i = 0; i < ndim; i++) sizes.push_back(dims[i]);
    auto result = ((torch::Tensor*)t)->reshape(sizes);
    return (void*)new_t(result);
}

void* st_clone(void* t) {
    return (void*)new_t(((torch::Tensor*)t)->clone());
}

void st_tensor_save(void* t, const char* path) {
    auto tensor = ((torch::Tensor*)t)->clone().to(torch::kCPU);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(tensor.data_ptr<float>(), sizeof(float), tensor.numel(), f);
        fclose(f);
    }
}

double st_sum(void* t) {
    return ((torch::Tensor*)t)->sum().item<double>();
}

float st_get(void* t, int64_t i) {
    return ((torch::Tensor*)t)->data_ptr<float>()[i];
}

void st_print_shape(void* t) {
    auto& tensor = *(torch::Tensor*)t;
    fprintf(stderr, "[");
    for (int i = 0; i < tensor.dim(); i++) {
        if (i) fprintf(stderr, ",");
        fprintf(stderr, "%ld", (long)tensor.size(i));
    }
    fprintf(stderr, "]\n");
}

double st_max(void* t) {
    return ((torch::Tensor*)t)->max().item<double>();
}

double st_min(void* t) {
    return ((torch::Tensor*)t)->min().item<double>();
}

double st_mean(void* t) {
    return ((torch::Tensor*)t)->mean().item<double>();
}

} // extern "C"
