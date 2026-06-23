;; static_prelude.scm — StaticPy 值类型运行时
;;
;; 值类型: int=fixnum, float=flonum, bool=boolean
;; 数组类型: float[] = foreign-alloc 分配的 C float*
;;           ptr[] = foreign-alloc 分配的 C void* (张量指针)
;;           自动 GC 回收（通过 guardian + 主动 drain）

(import (chezscheme))

;; ====== GC 跟踪：自动释放 C 内存 ======
(define *gc-guardian* (make-guardian))

(define (gc-register-ptr ptr)
  "注册 C 指针到 GC 跟踪，不再使用时自动 free"
  (*gc-guardian* ptr)
  ptr)

(define (gc-drain)
  "主动释放所有回收的 C 内存。需在 GC 后或定期调用。"
  (let loop ()
    (let ((ptr (*gc-guardian*)))
      (when ptr
        (foreign-free ptr)
        (loop)))))

;; 注册 GC hook: Chez Scheme 每次 GC 后自动调用 gc-drain
;; 使用 guard 兼容不支持 collect-rendezvous 的旧版本
(guard (ex (else (display "Warning: collect-rendezvous not available, C arrays may leak\n") (newline)))
  (collect-rendezvous (lambda () (gc-drain))))

;; ====== 浮点数组 ======

(define (make-float-array n)
  "创建 n 个 float 的 C 数组，自动 GC 回收"
  (let ((ptr (foreign-alloc (* n 4))))
    (do ((i 0 (+ i 1))) ((= i n))
      (foreign-set! 'float ptr (* i 4) 0.0))
    (gc-register-ptr ptr)
    ptr))

(define (float-array-set ptr i val)
  (foreign-set! 'float ptr (* i 4) val))

(define (float-array-ref ptr i)
  (foreign-ref 'float ptr (* i 4)))

(define (float-array-offset ptr n)
  "按 float 元素个数偏移指针 (n*4 bytes)"
  (+ ptr (* n 4)))

(define (float-array-free ptr)
  "手动释放（也可等 GC 自动回收）"
  (foreign-free ptr))

;; ====== 指针数组（用于保存中间张量指针） ======

(define (make-ptr-array n)
  "创建 n 个 void* 的 C 数组，自动 GC 回收"
  (let ((ptr (foreign-alloc (* n 8))))
    (do ((i 0 (+ i 1))) ((= i n))
      (foreign-set! 'void* ptr (* i 8) 0))
    (gc-register-ptr ptr)
    ptr))

(define (ptr-array-set ptr i val)
  (foreign-set! 'void* ptr (* i 8) val))

(define (ptr-array-ref ptr i)
  (foreign-ref 'void* ptr (* i 8)))

;; ====== 整数数组（用于 shape / index） ======

(define (make-int-array n)
  "创建 n 个 int64 的 C 数组，自动 GC 回收"
  (let ((ptr (foreign-alloc (* n 8))))
    (do ((i 0 (+ i 1))) ((= i n))
      (foreign-set! 'signed-64 ptr (* i 8) 0))
    (gc-register-ptr ptr)
    ptr))

(define (int-array-set ptr i val)
  (foreign-set! 'signed-64 ptr (* i 8) val))

(define (int-array-ref ptr i)
  (foreign-ref 'signed-64 ptr (* i 8)))

;; ====== 字典（hashtable） ======

(define (make-dict)
  "创建空字典"
  (make-hashtable equal-hash equal?))

(define (dict-get d key)
  "读取字典 d[key]"
  (hashtable-ref d key #f))

(define (dict-set! d key val)
  "写入字典 d[key] = val"
  (hashtable-set! d key val))

(define (dict-contains? d key)
  "检查 key 是否在字典中"
  (hashtable-contains? d key))

;; ====== C ABI 包装（供 extern fn from "prelude" 调用） ======
(define make_float_array (lambda (n) (make-float-array n)))
(define float_array_set (lambda (ptr i val) (float-array-set ptr i val)))
(define float_array_ref (lambda (ptr i) (float-array-ref ptr i)))
(define float_array_offset (lambda (ptr n) (float-array-offset ptr n)))
(define make_ptr_array (lambda (n) (make-ptr-array n)))
(define ptr_array_set (lambda (ptr i val) (ptr-array-set ptr i val)))
(define ptr_array_ref (lambda (ptr i) (ptr-array-ref ptr i)))
(define make_int_array (lambda (n) (make-int-array n)))
(define int_array_set (lambda (ptr i val) (int-array-set ptr i val)))
(define make_dict (lambda () (make-dict)))
(define dict_get (lambda (d key) (dict-get d key)))
(define dict_set (lambda (d key val) (dict-set! d key val)))

;; ====== 文件 I/O（基于 libc） ======

(load-shared-object "libc.so.6")

(define libc-fopen
  (foreign-procedure "fopen" (string string) void*))
(define libc-fclose
  (foreign-procedure "fclose" (void*) int))
(define libc-fread
  (foreign-procedure "fread" (void* unsigned-64 unsigned-64 void*) unsigned-64))
(define libc-fwrite
  (foreign-procedure "fwrite" (void* unsigned-64 unsigned-64 void*) unsigned-64))
(define libc-fprintf
  (foreign-procedure "fprintf" (void* string) int))
(define libc-fgets
  (foreign-procedure "fgets" (void* int void*) void*))

(define (file-open path mode)
  (libc-fopen path mode))

(define (file-close fp)
  (libc-fclose fp))

(define (file-write fp s)
  (let* ((bv (string->utf8 s))
         (len (bytevector-length bv))
         (buf (foreign-alloc len)))
    (do ((i 0 (+ i 1))) ((= i len))
      (foreign-set! 'unsigned-8 buf i (bytevector-u8-ref bv i)))
    (libc-fwrite buf 1 len fp)
    (foreign-free buf)))

(define (file-read-all fp)
  (let* ((buf-size 4096)
         (buf (foreign-alloc buf-size))
         (out (open-output-string)))
    (let loop ()
      (let ((n (libc-fread buf 1 (- buf-size 1) fp)))
        (when (> n 0)
          (foreign-set! 'unsigned-8 buf n 0)  ;; null-terminate
          (display (pointer->string buf) out)
          (loop))))
    (foreign-free buf)
    (get-output-string out)))

(define (file-read-floats path n)
  "读取 n 个 float32 到 float-array"
  (let ((arr (make-float-array n))
        (fp (libc-fopen path "rb")))
    (when (not fp) (error 'file-read-floats "cannot open file" path))
    (libc-fread arr 4 n fp)
    (libc-fclose fp)
    arr))

(define (file-exists? path)
  (let ((fp (libc-fopen path "r")))
    (when fp (libc-fclose fp))
    (not (not fp))))

;; C ABI wrappers
(define file_open (lambda (path mode) (file-open path mode)))
(define file_close (lambda (fp) (file-close fp)))
(define file_read_all (lambda (fp) (file-read-all fp)))
(define file_read_floats (lambda (path n) (file-read-floats path n)))
(define file_write (lambda (fp s) (file-write fp s)))
(define file_exists (lambda (path) (file-exists? path)))

;; ====== 数学函数（libm） ======

(load-shared-object "libm.so.6")
(define libm-sin   (foreign-procedure "sin" (double) double))
(define libm-cos   (foreign-procedure "cos" (double) double))
(define libm-tan   (foreign-procedure "tan" (double) double))
(define libm-log   (foreign-procedure "log" (double) double))
(define libm-log2  (foreign-procedure "log2" (double) double))
(define libm-log10 (foreign-procedure "log10" (double) double))
(define libm-exp   (foreign-procedure "exp" (double) double))
(define libm-sqrt  (foreign-procedure "sqrt" (double) double))
(define libm-floor (foreign-procedure "floor" (double) double))
(define libm-ceil  (foreign-procedure "ceil" (double) double))
(define libm-round (foreign-procedure "round" (double) double))
(define libm-pow   (foreign-procedure "pow" (double double) double))
(define libm-fmod  (foreign-procedure "fmod" (double double) double))
(define libm-sinh  (foreign-procedure "sinh" (double) double))
(define libm-cosh  (foreign-procedure "cosh" (double) double))
(define libm-tanh  (foreign-procedure "tanh" (double) double))
(define libm-atan2 (foreign-procedure "atan2" (double double) double))
(define libm-asin  (foreign-procedure "asin" (double) double))
(define libm-acos  (foreign-procedure "acos" (double) double))
(define libm-atan  (foreign-procedure "atan" (double) double))

;; ====== JSON 解析器（纯 Scheme，无依赖） ======

(define (parse-json s)
  "解析 JSON 字符串 → Scheme 值"
  (letrec
    ((skip-whitespace
       (lambda (i)
         (let loop ((i i))
           (if (< i (string-length s))
               (let ((c (string-ref s i)))
                 (if (or (char=? c #\space) (char=? c #\newline)
                         (char=? c #\tab) (char=? c #\return))
                     (loop (+ i 1))
                     i))
               i))))
     (parse-value
       (lambda (i)
         (let ((i (skip-whitespace i)))
           (if (>= i (string-length s)) (values #f i)
             (let ((c (string-ref s i)))
               (cond ((char=? c #\{) (parse-object (+ i 1)))
                     ((char=? c #\[) (parse-array (+ i 1)))
                     ((char=? c #\") (parse-string (+ i 1)))
                     ((char=? c #\t) (values #t (+ i 4)))
                     ((char=? c #\f) (values #f (+ i 5)))
                     ((char=? c #\n) (values '() (+ i 4)))
                     ((or (char=? c #\-) (char=? c #\+)
                          (char<=? #\0 c #\9))
                      (parse-number i))
                     (else (values #f i))))))))
     (parse-object
       (lambda (i)
         (let ((ht (make-hashtable equal-hash equal?)))
           (let loop ((i i))
             (let ((i (skip-whitespace i)))
               (if (and (< i (string-length s)) (char=? (string-ref s i) #\}))
                   (values ht (+ i 1))
                   (let-values (((key i) (parse-value i)))
                     (let ((i (skip-whitespace i)))
                       (if (and (< i (string-length s))
                                (char=? (string-ref s i) #\:))
                           (let-values (((val i) (parse-value (+ i 1))))
                             (hashtable-set! ht key val)
                             (let ((i (skip-whitespace i)))
                               (if (and (< i (string-length s))
                                        (char=? (string-ref s i) #\,))
                                   (loop (+ i 1))
                                   (loop i)))))
                           (values ht i)))))))))
     (parse-array
       (lambda (i)
         (let loop ((i i) (acc '()))
           (let ((i (skip-whitespace i)))
             (if (and (< i (string-length s)) (char=? (string-ref s i) #\]))
                 (values (reverse acc) (+ i 1))
                 (let-values (((val i) (parse-value i)))
                   (let ((i (skip-whitespace i)))
                     (if (and (< i (string-length s))
                              (char=? (string-ref s i) #\,))
                         (loop (+ i 1) (cons val acc))
                         (loop i (cons val acc))))))))))
     (parse-string
       (lambda (i)
         (let loop ((i i) (acc '()))
           (if (>= i (string-length s))
               (values (apply string (reverse acc)) i)
               (let ((c (string-ref s i)))
                 (cond ((char=? c #\") (values (apply string (reverse acc)) (+ i 1)))
                       ((char=? c #\\)
                        (if (< (+ i 1) (string-length s))
                            (let ((c2 (string-ref s (+ i 1))))
                              (case c2
                                ((#\") (loop (+ i 2) (cons #\" acc)))
                                ((#\\) (loop (+ i 2) (cons #\\ acc)))
                                ((#\/) (loop (+ i 2) (cons #\/ acc)))
                                ((#\b) (loop (+ i 2) (cons #\backspace acc)))
                                ((#\f) (loop (+ i 2) (cons #\page acc)))
                                ((#\n) (loop (+ i 2) (cons #\newline acc)))
                                ((#\r) (loop (+ i 2) (cons #\return acc)))
                                ((#\t) (loop (+ i 2) (cons #\tab acc)))
                                (else (loop (+ i 2) (cons c2 acc)))))
                            (loop (+ i 1) (cons c acc))))
                       (else (loop (+ i 1) (cons c acc)))))))))
     (parse-number
       (lambda (i)
         (let ((start i))
           (if (and (< i (string-length s))
                    (char=? (string-ref s i) #\-))
               (set! i (+ i 1)))
           (let loop ()
             (if (and (< i (string-length s))
                      (char<=? #\0 (string-ref s i) #\9))
                 (begin (set! i (+ i 1)) (loop))
                 i))
           (if (and (< i (string-length s))
                    (char=? (string-ref s i) #\.))
               (let ((start-frac (+ i 1)))
                 (set! i start-frac)
                 (let loop ()
                   (if (and (< i (string-length s))
                            (char<=? #\0 (string-ref s i) #\9))
                       (begin (set! i (+ i 1)) (loop))
                       i))
                 (let ((num-str (substring s start i)))
                   (values (string->number num-str) i)))
               (let ((num-str (substring s start i)))
                 (values (string->number num-str) i)))))))
    (let-values (((val i) (parse-value 0)))
      val)))

;; ====== 实用工具函数 ======

(define (string_split s sep)
  (let ((parts '())
        (start 0))
    (let loop ((pos 0))
      (if (>= pos (string-length s))
          (reverse (cons (substring s start pos) parts))
          (if (char=? (string-ref s pos) sep)
              (begin
                (set! parts (cons (substring s start pos) parts))
                (set! start (+ pos 1))
                (loop (+ pos 1)))
              (loop (+ pos 1)))))))

(define (string_to_float s)
  (string->number s))

(define (string_to_int s)
  (string->number s))

(define (string_join parts sep)
  "连接字符串列表, O(n) 通过 string port"
  (let ((out (open-output-string)))
    (let loop ((p parts) (first #t))
      (when (not (null? p))
        (unless first (display sep out))
        (display (car p) out)
        (loop (cdr p) #f)))
    (get-output-string out)))

(define (string_trim s)
  (let ((len (string-length s)))
    (let loop-start ((i 0))
      (if (>= i len) ""
          (if (char-whitespace? (string-ref s i))
              (loop-start (+ i 1))
              (let loop-end ((j (- len 1)))
                (if (char-whitespace? (string-ref s j))
                    (loop-end (- j 1))
                    (substring s i (+ j 1)))))))))

(define (string_lower s)
  (string-downcase s))

(define (string_upper s)
  (string-upcase s))

(define (string_contains s substr)
  (if (string-contains s substr) #t #f))

(define (string_starts_with s prefix)
  (let ((plen (string-length prefix)))
    (and (>= (string-length s) plen)
         (string=? (substring s 0 plen) prefix))))

(define (string_ends_with s suffix)
  (let ((slen (string-length s))
        (sufflen (string-length suffix)))
    (and (>= slen sufflen)
         (string=? (substring s (- slen sufflen) slen) suffix))))

;; C ABI wrappers
(define str_split (lambda (s sep) (string_split s sep)))
(define str_join (lambda (parts sep) (string_join parts sep)))
(define str_trim (lambda (s) (string_trim s)))
(define str_lower (lambda (s) (string_lower s)))
(define str_upper (lambda (s) (string_upper s)))
(define str_contains (lambda (s substr) (string_contains s substr)))
(define str_starts_with (lambda (s prefix) (string_starts_with s prefix)))
(define str_ends_with (lambda (s suffix) (string_ends_with s suffix)))

;; ====== 打印浮点数组（调试用） ======

(define (print-float-array ptr n)
  (display "#float[")
  (let loop ((i 0))
    (when (< i n)
      (when (> i 0) (display " "))
      (display (float-array-ref ptr i))
      (loop (+ i 1))))
  (display "]")
  (newline))

;; ====== sleep / clock ======

(define libc-nanosleep
  (foreign-procedure "nanosleep" (void* void*) int))

(define (sleep ms)
  (let* ((sec (quotient ms 1000))
         (nsec (* (remainder ms 1000) 1000000))
         (req (foreign-alloc 16)))
    (foreign-set! 'signed-64 req 0 sec)
    (foreign-set! 'signed-64 req 8 nsec)
    (libc-nanosleep req 0)
    (foreign-free req)))

(define (clock)
  (current-second))

(define libc-exit
  (foreign-procedure "exit" (int) void))

(define (exit_program code)
  (libc-exit code))
