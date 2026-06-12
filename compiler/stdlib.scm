;; static_stdlib.scm — StaticPy 标准库
;; 默认包含: OpenBLAS, LAPACK, libTorch (aoti)
;; Python 风格包装: np.*, torch.*, la.*, ml.*
;;
;; 在 static_build.sh 中自动并入每个编译

(import (chezscheme))

;; ============================================================
;; Part 1: OpenBLAS (numpy 底层) — 可选加载，不可用时跳过
;; ============================================================
(define *openblas-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libopenblas.so.0")
        (set! *openblas-available* #t)))))

(define cblas-sdot  #f)
(define cblas-ddot  #f)
(define cblas-saxpy #f)
(define cblas-daxpy #f)
(define cblas-scopy #f)
(define cblas-dcopy #f)
(define cblas-sscal #f)
(define cblas-dscal #f)
(define cblas-sgemv #f)
(define cblas-dgemv #f)
(define cblas-sgemm #f)
(define cblas-dgemm #f)

(when *openblas-available*
  (set! cblas-sdot  (foreign-procedure "cblas_sdot" (int void* int void* int) float))
  (set! cblas-ddot  (foreign-procedure "cblas_ddot" (int void* int void* int) double))
  (set! cblas-saxpy (foreign-procedure "cblas_saxpy" (int float void* int void* int) void))
  (set! cblas-daxpy (foreign-procedure "cblas_daxpy" (int double void* int void* int) void))
  (set! cblas-scopy (foreign-procedure "cblas_scopy" (int void* int void* int) void))
  (set! cblas-dcopy (foreign-procedure "cblas_dcopy" (int void* int void* int) void))
  (set! cblas-sscal (foreign-procedure "cblas_sscal" (int float void* int) void))
  (set! cblas-dscal (foreign-procedure "cblas_dscal" (int double void* int) void))
  (set! cblas-sgemv (foreign-procedure "cblas_sgemv" (int int int float void* int void* int float void* int) void))
  (set! cblas-dgemv (foreign-procedure "cblas_dgemv" (int int int double void* int void* int double void* int) void))
  (set! cblas-sgemm (foreign-procedure "cblas_sgemm" (int int int int int int float void* int void* int float void* int) void))
  (set! cblas-dgemm (foreign-procedure "cblas_dgemm" (int int int int int int double void* int void* int double void* int) void)))

;; ============================================================
;; Part 2: LAPACK — 可选加载
;; ============================================================
(define *lapack-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "liblapacke.so.3")
        (set! *lapack-available* #t)))))

(define lapack-dgesv  #f)
(define lapack-sgesv  #f)
(define lapack-dgesdd #f)
(define lapack-sgesdd #f)
(define lapack-dsyevd #f)
(define lapack-dgeqrf #f)

(when *lapack-available*
  (set! lapack-dgesv  (foreign-procedure "LAPACKE_dgesv" (int int int void* int void* void* int) int))
  (set! lapack-sgesv  (foreign-procedure "LAPACKE_sgesv" (int int int void* int void* void* int) int))
  (set! lapack-dgesdd (foreign-procedure "LAPACKE_dgesdd" (int int int int void* int void* void* int void* int) int))
  (set! lapack-sgesdd (foreign-procedure "LAPACKE_sgesdd" (int int int int void* int void* void* int void* int) int))
  (set! lapack-dsyevd (foreign-procedure "LAPACKE_dsyevd" (int int int int void* int void*) int))
  (set! lapack-dgeqrf (foreign-procedure "LAPACKE_dgeqrf" (int int int void* int void*) int)))

;; ============================================================
;; Part 3: CUDA cuBLAS — GPU 矩阵运算
;; ============================================================
(define *cuda-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcublas.so.12")
        (load-shared-object "libcudart.so.12")
        (set! *cuda-available* #t)))))

(define cublas-handle #f)
(define cublas-create #f)
(define cublas-destroy #f)
(define cublas-sgemm #f)
(define cublas-dgemm #f)
(define cublas-saxpy #f)
(define cublas-daxpy #f)
(define cublas-sdot #f)
(define cublas-ddot #f)
(define cublas-dcopy #f)

(when *cuda-available*
  (set! cublas-create (foreign-procedure "cublasCreate_v2" (void*) int))
  (set! cublas-destroy (foreign-procedure "cublasDestroy_v2" (void*) int))
  (set! cublas-sgemm (foreign-procedure "cublasSgemm_v2"
    (void* int int int int int int float void* int void* int float void* int) int))
  (set! cublas-dgemm (foreign-procedure "cublasDgemm_v2"
    (void* int int int int int int void* void* int void* int void* void* int) int))
  (set! cublas-saxpy (foreign-procedure "cublasSaxpy_v2"
    (void* int float void* int void* int) int))
  (set! cublas-daxpy (foreign-procedure "cublasDaxpy_v2"
    (void* int double void* int void* int) int))
  (set! cublas-ddot (foreign-procedure "cublasDdot_v2"
    (void* int void* int void* int) double))
  (set! cublas-dcopy (foreign-procedure "cublasDcopy_v2"
    (void* int void* int void* int) int))
  ;; 初始化 cuBLAS handle
  (let ((h (foreign-alloc 8)))
    (cublas-create h)
    (set! cublas-handle (foreign-ref 'void* h 0))
    (foreign-free h)))

;; ============================================================
;; Part 3a: cuRAND — GPU 随机数生成（权重初始化/dropout）
;; ============================================================
(define *curand-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcurand.so.12")
        (set! *curand-available* #t)))))

(define curand-create #f)
(define curand-destroy #f)
(define curand-set-seed #f)
(define curand-uniform #f)
(define curand-normal #f)

(when *curand-available*
  (set! curand-create (foreign-procedure "curandCreateGenerator" (void* int) int))
  (set! curand-destroy (foreign-procedure "curandDestroyGenerator" (void*) int))
  (set! curand-set-seed (foreign-procedure "curandSetPseudoRandomGeneratorSeed" (void* int) int))
  (set! curand-uniform (foreign-procedure "curandGenerateUniform" (void* void* int) int))
  (set! curand-normal (foreign-procedure "curandGenerateNormal" (void* void* int double double) int)))

;; ============================================================
;; Part 4: cuSOLVER — GPU 矩阵分解（SVD/Cholesky）
;; ============================================================
(define *cusolver-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcusolver.so.12")
        (set! *cusolver-available* #t)))))

(define cusolver-dn-create #f)
(define cusolver-dn-destroy #f)
(define cusolver-dgesvd #f)
(define cusolver-dpotrf #f)

(when *cusolver-available*
  (set! cusolver-dn-create (foreign-procedure "cusolverDnCreate" (void*) int))
  (set! cusolver-dn-destroy (foreign-procedure "cusolverDnDestroy" (void*) int))
  (set! cusolver-dgesvd (foreign-procedure "cusolverDnDgesvd"
    (void* int int int int void* int void* void* int void* int void* int void* void*) int))
  (set! cusolver-dpotrf (foreign-procedure "cusolverDnDpotrf"
    (void* int int void* int void* int void*) int)))

;; ============================================================
;; Part 5: cuDNN — GPU 深度学习原语
;; ============================================================
(define *cudnn-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcudnn.so.9")
        (set! *cudnn-available* #t)))))

(define cudnn-create #f)
(define cudnn-destroy #f)
(define cudnn-create-tensor-desc #f)
(define cudnn-set-tensor-4d #f)
(define cudnn-create-filter-desc #f)
(define cudnn-set-filter-4d #f)
(define cudnn-create-conv-desc #f)
(define cudnn-set-conv-2d #f)
(define cudnn-conv-forward #f)
(define cudnn-activation-forward #f)
(define cudnn-activation-backward #f)
(define cudnn-pooling-forward #f)
(define cudnn-softmax-forward #f)
(define cudnn-batchnorm-forward #f)

(when *cudnn-available*
  (call-with-current-continuation
    (lambda (return)
      (with-exception-handler
        (lambda (e) #f (return))
        (lambda ()
          (set! cudnn-create (foreign-procedure "cudnnCreate" (void*) int))
          (set! cudnn-destroy (foreign-procedure "cudnnDestroy" (void*) int))
          (set! cudnn-create-tensor-desc (foreign-procedure "cudnnCreateTensorDescriptor" (void*) int))
          (set! cudnn-set-tensor-4d (foreign-procedure "cudnnSetTensor4dDescriptor" (void* int int int int int int) int))
          (set! cudnn-create-filter-desc (foreign-procedure "cudnnCreateFilterDescriptor" (void*) int))
          (set! cudnn-set-filter-4d (foreign-procedure "cudnnSetFilter4dDescriptor" (void* int int int int int int) int))
          (set! cudnn-create-conv-desc (foreign-procedure "cudnnCreateConvolutionDescriptor" (void*) int))
          (set! cudnn-set-conv-2d (foreign-procedure "cudnnSetConvolution2dDescriptor" (void* int int int int int int int int) int))
          (set! cudnn-conv-forward (foreign-procedure "cudnnConvolutionForward"
            (void* void* void* void* void* void* void* int void* int void* void* void*) int))
          (set! cudnn-activation-forward (foreign-procedure "cudnnActivationForward"
            (void* int void* void* void* void* void* void*) int))
          (set! cudnn-activation-backward (foreign-procedure "cudnnActivationBackward"
            (void* int void* void* void* void* void* void* void* void* void* void*) int))
          (set! cudnn-pooling-forward (foreign-procedure "cudnnPoolingForward"
            (void* void* void* void* void* void* void* void*) int))
          (set! cudnn-softmax-forward (foreign-procedure "cudnnSoftmaxForward"
            (void* int int void* void* void* void* void* void*) int))
          (set! cudnn-batchnorm-forward (foreign-procedure "cudnnBatchNormalizationForwardTraining"
            (void* int void* void* void* void* void* void* void* void* void* void* void* double) int)))))))

;; ============================================================
;; Part 6: libcurl — HTTP 数据获取
;; ============================================================
(define *curl-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcurl.so.4")
        (set! *curl-available* #t)))))

(define curl-easy-init #f)
(define curl-easy-setopt #f)
(define curl-easy-perform #f)
(define curl-easy-cleanup #f)
(define curl-slist-append #f)

(when *curl-available*
  (set! curl-easy-init (foreign-procedure "curl_easy_init" () void*))
  (set! curl-easy-setopt (foreign-procedure "curl_easy_setopt" (void* int void*) int))
  (set! curl-easy-perform (foreign-procedure "curl_easy_perform" (void*) int))
  (set! curl-easy-cleanup (foreign-procedure "curl_easy_cleanup" (void*) void))
  (set! curl-slist-append (foreign-procedure "curl_slist_append" (void* string) void*)))

;; ============================================================
;; Part 7: libcjson — JSON 解析
;; ============================================================
(define *cjson-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libcjson.so.1")
        (set! *cjson-available* #t)))))

(define cjson-parse #f)
(define cjson-print #f)
(define cjson-delete #f)
(define cjson-get-object-item #f)
(define cjson-get-array-item #f)
(define cjson-get-array-size #f)
(define cjson-is-number #f)
(define cjson-is-string #f)
(define cjson-get-number #f)
(define cjson-get-string #f)

(when *cjson-available*
  (set! cjson-parse (foreign-procedure "cJSON_Parse" (string) void*))
  (set! cjson-print (foreign-procedure "cJSON_Print" (void*) string))
  (set! cjson-delete (foreign-procedure "cJSON_Delete" (void*) void))
  (set! cjson-get-object-item (foreign-procedure "cJSON_GetObjectItem" (void* string) void*))
  (set! cjson-get-array-item (foreign-procedure "cJSON_GetArrayItem" (void* int) void*))
  (set! cjson-get-array-size (foreign-procedure "cJSON_GetArraySize" (void*) int))
  (set! cjson-is-number (foreign-procedure "cJSON_IsNumber" (void*) int))
  (set! cjson-is-string (foreign-procedure "cJSON_IsString" (void*) int))
  (set! cjson-get-number (foreign-procedure "cJSON_GetNumberValue" (void*) double))
  (set! cjson-get-string (foreign-procedure "cJSON_GetStringValue" (void*) string)))

;; ============================================================
;; Part 8: XGBoost — 树模型训练/推理
;; ============================================================
(define *xgboost-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libxgboost.so")
        (set! *xgboost-available* #t)))))

(define xgb-create #f)
(define xgb-set-param #f)
(define xgb-train #f)
(define xgb-predict #f)
(define xgb-dmatrix-create #f)
(define xgb-dmatrix-set-float #f)

(when *xgboost-available*
  (set! xgb-create (foreign-procedure "XGBoosterCreate" (void* int) int))
  (set! xgb-set-param (foreign-procedure "XGBoosterSetParam" (void* string string) int))
  (set! xgb-train (foreign-procedure "XGBoosterTrainOneIter" (void* void*) int))
  (set! xgb-predict (foreign-procedure "XGBoosterPredict"
    (void* void* int int void* void*) int))
  (set! xgb-dmatrix-create (foreign-procedure "XGDMatrixCreateFromMat"
    (void* int int double void*) int))
  (set! xgb-dmatrix-set-float (foreign-procedure "XGDMatrixSetFloatInfo"
    (void* string void* int) int)))

;; ============================================================
;; Part 9: LightGBM — 树模型（替代方案）
;; ============================================================
(define *lightgbm-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "liblightgbm.so")
        (set! *lightgbm-available* #t)))))

(define lgbm-create #f)
(define lgbm-predict #f)
(define lgbm-save #f)
(define lgbm-load #f)

(when *lightgbm-available*
  (set! lgbm-create (foreign-procedure "LGBM_BoosterCreateFromMat"
    (void* int int string void*) int))
  (set! lgbm-predict (foreign-procedure "LGBM_BoosterPredictForMat"
    (void* void* int int int int string void* void*) int))
  (set! lgbm-save (foreign-procedure "LGBM_BoosterSaveModel"
    (void* int string) int))
  (set! lgbm-load (foreign-procedure "LGBM_BoosterLoadModel"
    (string void*) int)))

;; ============================================================
;; Part 10: ONNX Runtime — 模型推理部署
;; ============================================================
(define *onnx-available* #f)
(call-with-current-continuation
  (lambda (return)
    (with-exception-handler
      (lambda (e) #f (return))
      (lambda ()
        (load-shared-object "libonnxruntime.so")
        (set! *onnx-available* #t)))))

(define onnx-create-session #f)
(define onnx-run #f)

(when *onnx-available*
  (call-with-current-continuation
    (lambda (return)
      (with-exception-handler
        (lambda (e) #f (return))
        (lambda ()
          (set! onnx-create-session (foreign-procedure "OrtCreateSession"
            (void* string void* void*) int))
          (set! onnx-run (foreign-procedure "OrtRun"
            (void* void* void* void* int void* void* int) int)))))))
;; ============================================================
;; Part 4: Python 风格包装 — numpy 兼容层（需要 OpenBLAS）
;; ============================================================

(define (np-dot a b n)
  "向量点积 a·b"
  (if *openblas-available* (cblas-ddot n a 1 b 1) (begin (display "OpenBLAS not available\n") 0.0)))

(define (np-daxpy n alpha x y)
  "向量线性组合 y = alpha*x + y"
  (if *openblas-available* (cblas-daxpy n alpha x 1 y 1) -1))

(define (np-copy n x y)
  "向量拷贝 y = x"
  (if *openblas-available* (cblas-dcopy n x 1 y 1) -1))

(define (np-scal n alpha x)
  "向量缩放 x = alpha*x"
  (if *openblas-available* (cblas-dscal n alpha x 1) -1))

(define (np-gemv trans m n alpha a x beta y)
  "矩阵×向量 y = alpha*A*x + beta*y"
  (if *openblas-available* (cblas-dgemv 101 trans m n alpha a n x 1 beta y 1) -1))

(define (np-gemm trans-a trans-b m n k alpha a b beta c)
  "矩阵乘法 C = alpha*A*B + beta*C"
  (if *openblas-available* (cblas-dgemm 101 trans-a trans-b m n k alpha a k b n beta c n) -1))

;; ====== numpy 数学函数（纯 Scheme，用 float-array 操作） ======

(define (np-sum ptr n)
  "求和"
  (do ((i 0 (+ i 1)) (s 0.0 (+ s (foreign-ref 'double ptr (* i 8))))) ((= i n) s)))

(define (np-mean ptr n)
  "平均值"
  (/ (np-sum ptr n) n))

(define (np-max ptr n)
  "最大值"
  (do ((i 1 (+ i 1)) (m (foreign-ref 'double ptr 0) (max m (foreign-ref 'double ptr (* i 8))))) ((= i n) m)))

(define (np-min ptr n)
  "最小值"
  (do ((i 1 (+ i 1)) (m (foreign-ref 'double ptr 0) (min m (foreign-ref 'double ptr (* i 8))))) ((= i n) m)))

(define (np-zeros n)
  "创建全零数组"
  (make-float-array n))

(define (np-ones n)
  "创建全一数组"
  (let ((ptr (make-float-array n)))
    (do ((i 0 (+ i 1))) ((= i n) ptr)
      (foreign-set! 'double ptr (* i 8) 1.0))
    ptr))

(define (np-sqrt ptr n)
  "逐元素平方根"
  (let ((r (make-float-array n)))
    (do ((i 0 (+ i 1))) ((= i n) r)
      (foreign-set! 'double r (* i 8) (sqrt (foreign-ref 'double ptr (* i 8)))))
    r))

(define (np-exp ptr n)
  "逐元素指数"
  (let ((r (make-float-array n)))
    (do ((i 0 (+ i 1))) ((= i n) r)
      (foreign-set! 'double r (* i 8) (exp (foreign-ref 'double ptr (* i 8)))))
    r))

(define (np-abs ptr n)
  "逐元素绝对值"
  (let ((r (make-float-array n)))
    (do ((i 0 (+ i 1))) ((= i n) r)
      (foreign-set! 'double r (* i 8) (abs (foreign-ref 'double ptr (* i 8)))))
    r))

(define (np-arange start end step)
  "arange: 从 start 到 end，步长 step"
  (let* ((n (max 0 (inexact->exact (ceiling (/ (- end start) step)))))
         (ptr (make-float-array n)))
    (do ((i 0 (+ i 1)) (v start (+ v step))) ((= i n) ptr)
      (foreign-set! 'double ptr (* i 8) v))))

(define (np-linspace start end n)
  "linspace: 从 start 到 end，均匀 n 个点"
  (let ((ptr (make-float-array n))
        (step (/ (- end start) (max 1 (- n 1)))))
    (do ((i 0 (+ i 1)) (v start (+ v step))) ((= i n) ptr)
      (foreign-set! 'double ptr (* i 8) v))))

(define (np-concatenate ptrs n-arrays sizes)
  "拼接多个数组"
  (let* ((total-size 0)
         (offsets (make-vector n-arrays 0)))
    (do ((i 0 (+ i 1)))
        ((= i n-arrays) (set! total-size total-size))
      (vector-set! offsets i total-size)
      (set! total-size (+ total-size (vector-ref sizes i))))
    (let ((result (make-float-array total-size)))
      (do ((arr-idx 0 (+ arr-idx 1)))
          ((= arr-idx n-arrays) result)
        (let ((src (vector-ref ptrs arr-idx))
              (offset (vector-ref offsets arr-idx))
              (len (vector-ref sizes arr-idx)))
          (do ((i 0 (+ i 1))) ((= i len))
            (foreign-set! 'double result (* (+ offset i) 8)
              (foreign-ref 'double src (* i 8)))))))))

(define (np-clip ptr n lo hi)
  "截断值到 [lo, hi] 范围"
  (let ((r (make-float-array n)))
    (do ((i 0 (+ i 1))) ((= i n) r)
      (foreign-set! 'double r (* i 8)
        (max lo (min hi (foreign-ref 'double ptr (* i 8))))))))

(define (np-argmax ptr n)
  "最大值的索引"
  (do ((i 1 (+ i 1)) (mi 0) (mv (foreign-ref 'double ptr 0)))
      ((= i n) mi)
    (let ((v (foreign-ref 'double ptr (* i 8))))
      (when (> v mv) (set! mi i) (set! mv v)))))

;; ============================================================
;; Part 5: Python 风格包装 — scipy.linalg 兼容层
;; ============================================================

(define (la-solve n a b)
  "解线性方程组 Ax = b"
  (lapack-dgesv 101 n 1 a n b 1))

(define (la-svd m n a s u vt)
  "SVD 分解 A = U*S*V^T"
  (lapack-dgesdd 101 65 m n a n s u m vt n))

(define (la-eigvals n a w)
  "特征值分解 (对称矩阵)"
  (lapack-dsyevd 101 78 78 n a n w))

;; ============================================================
;; Part 6: Python 风格包装 — PyTorch 兼容层
;; ============================================================

;; torch 包装函数（顶层 define，内部检查 torch 是否可用）
(define (torch-add a b)
  (if *torch-available* (aoti-add-tensor a b) (begin (display "torch not available\n") #f)))
(define (torch-mul a b)
  (if *torch-available* (aoti-mul-tensor a b) (begin (display "torch not available\n") #f)))
(define (torch-sub a b)
  (if *torch-available* (aoti-subtract a b) (begin (display "torch not available\n") #f)))
(define (torch-clone a)
  (if *torch-available* (aoti-clone a) (begin (display "torch not available\n") #f)))
(define (torch-reshape a s)
  (if *torch-available* (aoti-reshape a s) (begin (display "torch not available\n") #f)))
(define (torch-matmul a b)
  (if *torch-available*
    (let ((c (aoti-create-tensor-from-blob 0 0 0 0 0 0 0)))
      (aoti-mm a b c) c)
    (begin (display "torch not available\n") #f)))
