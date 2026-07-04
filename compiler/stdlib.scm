;; stdlib.scm — StaticPy 标准库
;;
;; 提供 Scheme 层级的工具函数和常用 FFI 函数。
;; 核心 C++ 运行时在 libtorch_std_helper 中，
;; extern fn 声明在用户 .static.py 中，由 translate.py 自动生成 foreign-procedure 绑定。
;;
;; 此文件只包含纯 Scheme 辅助函数，不重复声明 FFI。

;; ====== 数组工具 ======

(define (arr-fill! arr n val)
  "填充 float-array 前 n 个元素为 val"
  (do ((i 0 (+ i 1))) ((>= i n))
    (float-array-set arr i val)))

(define (arr-copy! dst src n)
  "复制 float-array"
  (do ((i 0 (+ i 1))) ((>= i n))
    (float-array-set dst i (float-array-ref src i))))

(define (arr-scale! arr n s)
  "float-array 乘以标量"
  (do ((i 0 (+ i 1))) ((>= i n))
    (float-array-set arr i (* (float-array-ref arr i) s))))

(define (arr-add! arr n s)
  "float-array 加上标量"
  (do ((i 0 (+ i 1))) ((>= i n))
    (float-array-set arr i (+ (float-array-ref arr i) s))))

;; ====== 形状工具 ======

(define (make-shape . dims)
  "创建 int64 形状数组，例如 (make-shape 1 64 128 128)"
  (let ((arr (make-int-array (length dims)))
        (dims-list dims))
    (let loop ((i 0) (d dims-list))
      (when (not (null? d))
        (int-array-set arr i (car d))
        (loop (+ i 1) (cdr d))))
    arr))

;; ====== 时间/性能 ======

(define (time-ms)
  "返回当前时间戳（毫秒）"
  (* (current-second) 1000))

(define (benchmark label thunk)
  "执行 thunk 并打印耗时"
  (let ((start (time-ms)))
    (let ((result (thunk)))
      (display label)
      (display ": ")
      (display (- (time-ms) start))
      (display " ms")
      (newline)
      result)))

;; ====== 条件等待 ======

(define (wait-cond pred timeout-ms)
  "等待 pred 为真，超时返回 #f"
  (let ((deadline (+ (time-ms) timeout-ms)))
    (let loop ()
      (if (pred) #t
          (if (> (time-ms) deadline) #f
              (loop))))))

;; ====== 简单断言 ======

(define (assert cond msg)
  "断言，失败时打印错误信息"
  (when (not cond)
    (display "ASSERTION FAILED: ")
    (display msg)
    (newline)
    (exit 1)))

;; ====== list 工具（用于 Python 列表字面量） ======

(define (list-length lst)
  (length lst))

(define (list-ref lst i)
  (if (vector? lst)
      (vector-ref lst i)
      (list-ref lst i)))

(define (vec-ref v i)
  (vector-ref v i))

;; Helper: convert C char* (as void*) to Scheme string
(define (c-pointer->string ptr)
  (if (and ptr (not (eq? ptr 0)))
      ((foreign-procedure "strdup" (void*) string) ptr)
      #f))

(define (os_getcwd)
  (let ((buf (foreign-alloc 1024)))
    ((foreign-procedure "getcwd" (void* unsigned-64) void*) buf 1024)
    (let ((result (c-pointer->string buf)))
      (foreign-free buf)
      result)))

(define (os_getenv name)
  (c-pointer->string ((foreign-procedure "getenv" (string) void*) name)))
