;; comfycli_ffi.scm — sd.cpp 适配层共享库 + 项目内置
;;
;; 通过 static_build.sh 的第 3 个参数引入，拼接在 static_stdlib.scm 之后、
;; 用户代码之前。这样 static_translate.py 无需为 "sdcpp_adapter" 打补丁。
;;
;; 注意：libsdcpp_adapter.so 通过 LD_LIBRARY_PATH 解析（本地 cpp/sd/build，
;; 部署时由 run.sh 设置）。

(load-shared-object "libsdcpp_adapter.so")

;; libcomfycli_torch.so — 权重级操作（模型/CLIP 合并、权重导出）。
;; 故意不使用 libtorch_std_helper.so 之名：上游 static_stdlib.scm 会探测该名并
;; 声明完整 torch API，而本项目只需精简子集。
;; 可选依赖：未加载时相关函数退化为报错桩，避免 AOT 载入期因符号缺失而整体失败。
(define torch_std_safetensors_load #f)
(define torch_std_safetensors_count #f)
(define torch_std_safetensors_save #f)
(define torch_std_safetensors_merge #f)
(define torch_std_safetensors_free #f)
(define torch_std_copy_file #f)

(define torch-helper-loaded?
  (guard (e (#t #f))
    (load-shared-object "libcomfycli_torch.so")
    #t))

(if torch-helper-loaded?
    (begin
      (set! torch_std_safetensors_load (foreign-procedure "torch_std_safetensors_load" (string) void*))
      (set! torch_std_safetensors_count (foreign-procedure "torch_std_safetensors_count" (void*) int))
      (set! torch_std_safetensors_save (foreign-procedure "torch_std_safetensors_save" (void* string) int))
      (set! torch_std_safetensors_merge (foreign-procedure "torch_std_safetensors_merge" (void* void* int string string double string) void*))
      (set! torch_std_safetensors_free (foreign-procedure "torch_std_safetensors_free" (void*) void))
      (set! torch_std_copy_file (foreign-procedure "torch_std_copy_file" (string string) int)))
    (begin
      (display "warning: libcomfycli_torch.so not loaded (merge/save nodes unavailable)\n")
      (set! torch_std_safetensors_load (lambda (path) (error "libcomfycli_torch.so not loaded")))
      (set! torch_std_safetensors_count (lambda (d) (error "libcomfycli_torch.so not loaded")))
      (set! torch_std_safetensors_save (lambda (d path) (error "libcomfycli_torch.so not loaded")))
      (set! torch_std_safetensors_merge (lambda (a b mode p r dr sp) (error "libcomfycli_torch.so not loaded")))
      (set! torch_std_safetensors_free (lambda (d) #f))
      (set! torch_std_copy_file (lambda (s d) (error "libcomfycli_torch.so not loaded")))))

;; 上游 prelude 未提供 dict_keys（comfycli execution 需要）。
;; StaticPy 的 list 即 Scheme vector，故直接返回 hashtable-keys 的 vector。
(define (dict_keys d)
  (hashtable-keys d))

;; workflow 链接 [node_id, output_index] 判定
(define (is_link x)
  (and (vector? x) (= (vector-length x) 2)
       (string? (vector-ref x 0)) (number? (vector-ref x 1))))

;; 上游 prelude 无路径函数
(define (path_dirname p)
  (let loop ((i (- (string-length p) 1)))
    (cond ((< i 0) "")
          ((char=? (string-ref p i) #\/) (substring p 0 i))
          (else (loop (- i 1))))))

(define (path_split p)
  (let loop ((i (- (string-length p) 1)))
    (cond ((< i 0) (vector "" p))
          ((char=? (string-ref p i) #\/)
           (vector (substring p 0 i) (substring p (+ i 1) (string-length p))))
          (else (loop (- i 1))))))
