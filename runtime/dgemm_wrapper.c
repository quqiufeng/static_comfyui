#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

/*
 * dgemm_wrapper.c — 运行时检测 CUDA，GPU 不可用时 CPU 回退
 *
 * 编译: gcc -O2 -shared -fPIC -o libdgemm_row.so dgemm_wrapper.c -ldl
 * 零外部依赖（不需要 CUDA 头文件/库）
 *
 * 策略:
 *   运行时 dlopen("libcublas.so.12")，成功 → GPU SGEMM
 *   失败 → 纯 C 实现矩阵乘（慢但可用）
 *
 * 注意: 为节省内存，StaticPy 的 float[] 是 4 字节 float，
 *       所以该 wrapper 内部按 SGEMM（float）处理。
 */

/* ===== 运行时 CUDA 句柄 ===== */
static void *cublas_handle = NULL;
static void *cuda_lib = NULL;

/* cuBLAS 函数指针类型 */
typedef int (*cublasCreate_t)(void **);
typedef int (*cublasSgemm_t)(void *, int, int, int, int, int,
                             float *, float *, int, float *, int,
                             float *, float *, int);
typedef int (*cudaMalloc_t)(void **, size_t);
typedef int (*cudaMemcpy_t)(void *, const void *, size_t, int);
typedef int (*cudaFree_t)(void *);

static cublasCreate_t p_cublasCreate = NULL;
static cublasSgemm_t p_cublasSgemm = NULL;
static cudaMalloc_t p_cudaMalloc = NULL;
static cudaMemcpy_t p_cudaMemcpy = NULL;
static cudaFree_t p_cudaFree = NULL;

static int gpu_available = -1; /* -1=未检测, 0=不可用, 1=可用 */
static void *gpu_blas_handle = NULL;

/* 初始化 GPU — 尝试加载 CUDA 库 */
static int try_init_gpu(void) {
    if (gpu_available >= 0) return gpu_available;

    cuda_lib = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda_lib) {
        cuda_lib = dlopen("libcublas.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!cuda_lib) {
        gpu_available = 0;
        return 0;
    }

    /* 加载函数指针 */
    p_cublasCreate = (cublasCreate_t)dlsym(cuda_lib, "cublasCreate_v2");
    p_cublasSgemm = (cublasSgemm_t)dlsym(cuda_lib, "cublasSgemm_v2");
    p_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_DEFAULT, "cudaMalloc");
    p_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_DEFAULT, "cudaMemcpy");
    p_cudaFree = (cudaFree_t)dlsym(RTLD_DEFAULT, "cudaFree");

    if (!p_cublasCreate || !p_cublasSgemm || !p_cudaMalloc || !p_cudaMemcpy || !p_cudaFree) {
        dlclose(cuda_lib);
        cuda_lib = NULL;
        gpu_available = 0;
        return 0;
    }

    /* 创建 cuBLAS handle */
    if (p_cublasCreate(&gpu_blas_handle) != 0) {
        dlclose(cuda_lib);
        cuda_lib = NULL;
        gpu_available = 0;
        return 0;
    }

    gpu_available = 1;
    return 1;
}

/* ===== GPU SGEMM 实现 ===== */
static void gemm_gpu(int m, int n, int k,
                     float alpha, float *A, float *B,
                     float beta, float *C) {
    float *dA, *dB, *dC;
    size_t sA = (size_t)m * k * sizeof(float);
    size_t sB = (size_t)k * n * sizeof(float);
    size_t sC = (size_t)m * n * sizeof(float);

    p_cudaMalloc((void**)&dA, sA);
    p_cudaMalloc((void**)&dB, sB);
    p_cudaMalloc((void**)&dC, sC);
    p_cudaMemcpy(dA, A, sA, 1); /* 1 = cudaMemcpyHostToDevice */
    p_cudaMemcpy(dB, B, sB, 1);
    p_cudaMemcpy(dC, C, sC, 1);

    /* Column-major: C^T(n,m) = B^T(n,k) * A^T(k,m) */
    p_cublasSgemm(gpu_blas_handle, 0, 0, n, m, k,
                  &alpha, dB, n, dA, k, &beta, dC, n);

    p_cudaMemcpy(C, dC, sC, 2); /* 2 = cudaMemcpyDeviceToHost */
    p_cudaFree(dA); p_cudaFree(dB); p_cudaFree(dC);
}

/* ===== CPU SGEMM 实现（纯 C，零依赖） ===== */
static void gemm_cpu(int m, int n, int k,
                     float alpha, float *A, float *B,
                     float beta, float *C) {
    /* Row-major: C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n) */
    int i, j, l;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            float sum = 0.0f;
            for (l = 0; l < k; l++) {
                sum += A[(size_t)i * k + l] * B[(size_t)l * n + j];
            }
            C[(size_t)i * n + j] = alpha * sum + beta * C[(size_t)i * n + j];
        }
    }
}

/* ===== 公开接口 ===== */

/* 行主序 SGEMM，自动选择 GPU/CPU。
   保持 double alpha/beta 以兼容现有 FFI 声明。 */
void dgemm_row_auto(int m, int n, int k,
                     double alpha, float *A, float *B,
                     double beta, float *C) {
    float a = (float)alpha;
    float b = (float)beta;
    if (try_init_gpu()) {
        gemm_gpu(m, n, k, a, A, B, b, C);
    } else {
        gemm_cpu(m, n, k, a, A, B, b, C);
    }
}

/* 返回 1 表示 GPU 可用，0 表示 CPU 模式 */
int dgemm_gpu_available(void) {
    try_init_gpu();
    return gpu_available == 1 ? 1 : 0;
}
