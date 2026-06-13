// staticpy_torch.cpp — PyTorch C++ API → C 函数包装
// 编译: g++ -O2 -shared -fPIC -o /tmp/libstaticpy_torch.so staticpy_torch.cpp \
//        -I/data/venv/lib/python3.12/site-packages/torch/include \
//        -I/data/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
//        -L/data/venv/lib/python3.12/site-packages/torch/lib -ltorch -lc10 \
//        -Wl,-rpath,/data/venv/lib/python3.12/site-packages/torch/lib
//
// 产出: libstaticpy_torch.so — 被 StaticPy 的 extern fn 调用

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Optional.h>
#include <vector>
#include <string>

extern "C" {

// ===== Tensor 生命周期 =====

void* st_tensor_create(float* data, int64_t* dims, int ndim) {
    // 从 C float 数组创建 torch tensor
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::from_blob(data, sizes, opts).clone();
    auto ptr = new torch::Tensor(tensor);
    return (void*)ptr;
}

void* st_tensor_zeros(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::zeros(sizes, torch::kFloat32);
    auto ptr = new torch::Tensor(tensor);
    return (void*)ptr;
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

void st_tensor_to_cuda(void* t) {
    *((torch::Tensor*)t) = ((torch::Tensor*)t)->cuda();
}

void st_tensor_to_cpu(void* t) {
    *((torch::Tensor*)t) = ((torch::Tensor*)t)->cpu();
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
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_linear(void* input, void* weight, void* bias) {
    auto& inp = *(torch::Tensor*)input;
    auto& w = *(torch::Tensor*)weight;
    c10::optional<torch::Tensor> b;
    if (bias) b = *(torch::Tensor*)bias;
    
    auto result = torch::linear(inp, w, b);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_group_norm(void* input, int num_groups, void* weight, void* bias, float eps) {
    auto& inp = *(torch::Tensor*)input;
    c10::optional<torch::Tensor> w, b;
    if (weight) w = *(torch::Tensor*)weight;
    if (bias) b = *(torch::Tensor*)bias;
    
    auto result = torch::group_norm(inp, num_groups, w, b, eps);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_layer_norm(void* input, void* weight, void* bias, float eps) {
    auto& inp = *(torch::Tensor*)input;
    c10::optional<torch::Tensor> w, b;
    if (weight) w = *(torch::Tensor*)weight;
    if (bias) b = *(torch::Tensor*)bias;
    
    auto result = torch::layer_norm(inp, {inp.size(-1)}, w, b, eps);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_mm(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    auto result = ta.mm(tb);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_matmul(void* a, void* b) {
    auto& ta = *(torch::Tensor*)a;
    auto& tb = *(torch::Tensor*)b;
    auto result = ta.matmul(tb);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_silu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::silu(inp);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_relu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::relu(inp);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_gelu(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::gelu(inp);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_sigmoid(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::sigmoid(inp);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_tanh(void* input) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::tanh(inp);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_softmax(void* input, int dim) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::softmax(inp, dim);
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_max_pool2d(void* input, int k_size, int stride, int pad) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::max_pool2d(inp, {k_size, k_size}, {stride, stride}, {pad, pad});
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_avg_pool2d(void* input, int k_size, int stride, int pad) {
    auto& inp = *(torch::Tensor*)input;
    auto result = torch::avg_pool2d(inp, {k_size, k_size}, {stride, stride}, {pad, pad});
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

void* st_upsample_nearest(void* input, int scale_h, int scale_w) {
    auto& inp = *(torch::Tensor*)input;
    auto h = inp.size(-2) * scale_h;
    auto w = inp.size(-1) * scale_w;
    auto result = torch::upsample_nearest2d(inp, {h, w});
    auto ptr = new torch::Tensor(result);
    return (void*)ptr;
}

// ===== 标量运算 =====

void st_add(void* a, void* b) {
    *((torch::Tensor*)a) = *((torch::Tensor*)a) + *((torch::Tensor*)b);
}

void st_mul(void* a, void* b) {
    *((torch::Tensor*)a) = *((torch::Tensor*)a) * *((torch::Tensor*)b);
}

void* st_add_tensor(void* a, void* b) {
    auto result = *((torch::Tensor*)a) + *((torch::Tensor*)b);
    return (void*)new torch::Tensor(result);
}

void* st_sub_tensor(void* a, void* b) {
    auto result = *((torch::Tensor*)a) - *((torch::Tensor*)b);
    return (void*)new torch::Tensor(result);
}

void* st_mul_tensor(void* a, void* b) {
    auto result = *((torch::Tensor*)a) * *((torch::Tensor*)b);
    return (void*)new torch::Tensor(result);
}

// ===== 创建/转换 =====

void* st_from_blob(float* data, int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto tensor = torch::from_blob(data, sizes, torch::kFloat32).clone();
    return (void*)new torch::Tensor(tensor);
}

void* st_arange(int start, int end, int step) {
    auto result = torch::arange(start, end, step, torch::kFloat32);
    return (void*)new torch::Tensor(result);
}

void* st_ones(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    return (void*)new torch::Tensor(torch::ones(sizes, torch::kFloat32));
}

void* st_new(int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    return (void*)new torch::Tensor(torch::empty(sizes, torch::kFloat32));
}

void* st_slice(void* t, int dim, int start, int end) {
    auto result = ((torch::Tensor*)t)->slice(dim, start, end);
    return (void*)new torch::Tensor(result);
}

void* st_reshape(void* t, int64_t* dims, int ndim) {
    std::vector<int64_t> sizes(dims, dims + ndim);
    auto result = ((torch::Tensor*)t)->reshape(sizes);
    return (void*)new torch::Tensor(result);
}

void* st_clone(void* t) {
    return (void*)new torch::Tensor(((torch::Tensor*)t)->clone());
}

float st_sum(void* t) {
    return ((torch::Tensor*)t)->sum().item<float>();
}

float st_mean(void* t) {
    return ((torch::Tensor*)t)->mean().item<float>();
}

} // extern "C"
