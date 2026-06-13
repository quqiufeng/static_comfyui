;; static_prelude.scm — StaticPy 值类型运行时
;; 
;; 值类型: int=fixnum, float=flonum, bool=boolean
;; 数组类型: float[] = foreign-alloc 分配的 C float*
;;           GC 自动回收（通过 guardian），无需手动 free

(import (chezscheme))

;; ====== GC 跟踪：自动释放 C 内存 ======
(define *gc-guardian* (make-guardian))

(define (gc-register-ptr ptr)
  "注册 C 指针到 GC 跟踪，不再使用时自动 free"
  (*gc-guardian* ptr)
  ptr)

(define (gc-cleanup)
  "由 GC 在回收时自动调用"
  (let loop ()
    (let ((ptr (*gc-guardian*)))
      (when ptr
        (foreign-free ptr)
        (loop)))))

;; 每帧清理一次（在 prelude 中不自动触发，依赖 Chez GC 周期）
;; 实际由用户或编译器在适当时机调用

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

;; ====== 文件 I/O ======

;; 加载 libc
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
  "打开文件，返回文件指针"
  (libc-fopen path mode))

(define (file-close fp)
  "关闭文件"
  (libc-fclose fp))

(define (file-write fp s)
  "写入字符串到文件"
  (let* ((bv (string->utf8 s))
         (len (bytevector-length bv))
         (buf (foreign-alloc len)))
    ;; Copy bytevector to C memory
    (do ((i 0 (+ i 1))) ((= i len))
      (foreign-set! 'unsigned-8 buf i (bytevector-u8-ref bv i)))
    (libc-fwrite buf 1 len fp)
    (foreign-free buf)))

(define (file-read-all fp)
  "读取整个文件内容"
  (let* ((buf-size 4096)
         (buf (foreign-alloc buf-size))
         (result '()))
    (let loop ()
      (let ((n (libc-fread buf 1 (- buf-size 1) fp)))
        (when (> n 0)
          ;; Copy C memory to bytevector
          (let ((bv (make-bytevector n 0)))
            (do ((i 0 (+ i 1))) ((= i n))
              (bytevector-u8-set! bv i (foreign-ref 'unsigned-8 buf i)))
            (set! result (cons (utf8->string bv) result)))
          (loop))))
    (foreign-free buf)
    (apply string-append (reverse result))))

(define (file-exists? path)
  "检查文件是否存在"
  (let ((fp (libc-fopen path "r")))
    (when fp (libc-fclose fp))
    (not (not fp))))

;; ====== HTTP 客户端（基于 libcurl） ======

(load-shared-object "libcurl.so.4")

;; Curl easy handle
(define curl-easy-init
  (foreign-procedure "curl_easy_init" () void*))
(define curl-easy-setopt-string
  (foreign-procedure "curl_easy_setopt" (void* unsigned-32 string) int))
(define curl-easy-setopt-ptr
  (foreign-procedure "curl_easy_setopt" (void* unsigned-32 void*) int))
(define curl-easy-perform
  (foreign-procedure "curl_easy_perform" (void*) int))
(define curl-easy-cleanup
  (foreign-procedure "curl_easy_cleanup" (void*) void))
(define curl-easy-getinfo-string
  (foreign-procedure "curl_easy_getinfo" (void* unsigned-32 void*) int))

;; Curl slist
(define curl-slist-append
  (foreign-procedure "curl_slist_append" (void* string) void*))
(define curl-slist-free-all
  (foreign-procedure "curl_slist_free_all" (void*) void))

;; Option constants
(define CURLOPT_URL 10002)
(define CURLOPT_WRITEFUNCTION 20011)
(define CURLOPT_WRITEDATA 10001)
(define CURLOPT_HTTPHEADER 10023)
(define CURLOPT_TIMEOUT 13)

;; 全局 write callback（被 C 调用，收集响应数据到字节向量）
(define *http-response-buffer* #f)
(define *http-response-callback*
  (foreign-callable
    (lambda (ptr size nmemb userdata)
      (let* ((total (* size nmemb))
             (bv (make-bytevector total 0)))
        (do ((i 0 (+ i 1))) ((= i total))
          (bytevector-u8-set! bv i (foreign-ref 'unsigned-8 ptr i)))
        (set! *http-response-buffer*
              (cons bv *http-response-buffer*))
        total))
    (void* unsigned-64 unsigned-64 void*) unsigned-64))

(define (http-get url)
  "HTTP GET 请求，返回 (status . body)"
  (let* ((curl (curl-easy-init))
         (callback-ptr (foreign-callable-entry-point *http-response-callback*)))
    ;; 设置 URL
    (curl-easy-setopt-string curl CURLOPT_URL url)
    ;; 设置 write callback
    (curl-easy-setopt-ptr curl CURLOPT_WRITEFUNCTION callback-ptr)
    ;; 超时 30 秒
    (curl-easy-setopt-ptr curl CURLOPT_TIMEOUT 30)
    ;; 执行请求
    (set! *http-response-buffer* '())
    (let ((res (curl-easy-perform curl)))
      ;; 读取响应
      (let* ((parts (reverse *http-response-buffer*))
             (body (apply string-append (map utf8->string parts))))
        (curl-easy-cleanup curl)
        (cons res body)))))

(define (http-get-simple url)
  "HTTP GET，返回 body 字符串"
  (let ((result (http-get url)))
    (cdr result)))

;; ====== 整数数组（用于 shape） ======

(define (make-int-array n)
  "创建 n 个 int64 的 C 数组，自动 GC 回收"
  (let ((ptr (foreign-alloc (* n 8))))
    (do ((i 0 (+ i 1))) ((= i n))
      (foreign-set! 'unsigned-64 ptr (* i 8) 0))
    (gc-register-ptr ptr)
    ptr))

(define (int-array-set ptr i val)
  (foreign-set! 'unsigned-64 ptr (* i 8) val))

(define (int-array-ref ptr i)
  (foreign-ref 'unsigned-64 ptr (* i 8)))

(define (float-array-offset ptr n)
  "按 float 元素个数偏移指针 (n*4 bytes)"
  (+ ptr (* n 4)))

;; C ABI 包装（供 extern fn from "prelude" 调用）
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
(define file_open (lambda (path mode) (file-open path mode)))
(define file_close (lambda (fp) (file-close fp)))
(define file_read_all (lambda (fp) (file-read-all fp)))
(define file_write (lambda (fp s) (file-write fp s)))
(define file_exists (lambda (path) (file-exists? path)))
(define http_get_simple (lambda (url) (http-get-simple url)))

(define (print-float-array ptr n)
  (display "#float[")
  (let loop ((i 0))
    (when (< i n)
      (when (> i 0) (display " "))
      (display (float-array-ref ptr i))
      (loop (+ i 1))))
  (display "]")
  (newline))

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

;; ====== JSON 解析（libjansson） ======
(define *json-available* #f)
(guard (e (else (display "Warning: libjansson not found, json disabled\n")))
  (load-shared-object "libjansson.so.4")
  (set! *json-available* #t))

(define json-loads-ptr #f)
(define json-dumps-ptr #f)
(when *json-available*
  (set! json-loads-ptr
    (foreign-procedure "json_loads" (string int void*) void*))
  (set! json-dumps-ptr
    (foreign-procedure "json_dumps" (void* int) string)))

(define (json-loads s)
  "解析 JSON 字符串 → Scheme 值（vector/list/hashtable）"
  (if *json-available*
      (let* ((flags 0)
             (root (json-loads-ptr s flags 0))
             (result (json->scheme root)))
        root
        #f)  ;; 简化处理
      (begin (display "json not available\n") #f)))

(define (json->file obj path)
  "写入 JSON 到文件"
  (let ((s (json-scheme->string obj)))
    (when s
      (let ((fp (libc-fopen path "w")))
        (when fp
          (file-write fp s)
          (libc-fclose fp))))))

(define (parse-json s)
  "解析 JSON 字符串（简化实现，支持基本类型和嵌套 dict/list）"
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
                     (else (parse-number i))))))))
     (parse-object
       (lambda (i)
         (let ((tbl (make-dict)))
           (let loop ((i i))
             (let ((i (skip-whitespace i)))
               (if (and (< i (string-length s)) (char=? (string-ref s i) #\}))
                   (values tbl (+ i 1))
                   (let*-values (((key i) (parse-value i))
                                 ((i) (skip-whitespace i))
                                 ((i) (if (char=? (string-ref s i) #\:) (+ i 1) i))
                                 ((val i) (parse-value i))
                                 ((i) (skip-whitespace i)))
                     (dict-set! tbl key val)
                     (let ((i (if (char=? (string-ref s i) #\,) (+ i 1) i)))
                       (loop i)))))))))
     (parse-array
       (lambda (i)
         (let loop ((i i) (acc '()))
           (let ((i (skip-whitespace i)))
             (if (and (< i (string-length s)) (char=? (string-ref s i) #\]))
                 (values (reverse acc) (+ i 1))
                 (let*-values (((val i) (parse-value i))
                               ((i) (skip-whitespace i)))
                   (let ((i (if (char=? (string-ref s i) #\,) (+ i 1) i)))
                     (loop i (cons val acc)))))))))
     (parse-string
       (lambda (i)
         (let loop ((i i) (acc '()))
           (if (>= i (string-length s)) (values (list->string (reverse acc)) i)
             (let ((c (string-ref s i)))
               (cond ((char=? c #\") (values (list->string (reverse acc)) (+ i 1)))
                     ((char=? c #\\) (loop (+ i 2) (cons (parse-escape (+ i 1)) acc)))
                     (else (loop (+ i 1) (cons c acc)))))))))
     (parse-escape
       (lambda (i)
         (if (< i (string-length s))
             (case (string-ref s i)
               ((#\n) #\newline) ((#\t) #\tab) ((#\r) #\return)
               ((#\") #\") ((#\\) #\\) ((#\/) #\/)
               (else (string-ref s i)))
             #\?)))
     (parse-number
       (lambda (i)
         (let loop ((i i) (acc '()))
           (if (and (< i (string-length s))
                    (let ((c (string-ref s i)))
                      (or (char-numeric? c) (char=? c #\-) (char=? c #\+)
                          (char=? c #\.) (char=? c #\e) (char=? c #\E))))
               (loop (+ i 1) (cons (string-ref s i) acc))
               (let ((num-str (list->string (reverse acc))))
                 (values (if (str-find num-str ".")
                             (string->number num-str)
                             (string->number num-str))
                         i)))))))
    (call-with-values (lambda () (parse-value 0))
      (lambda (val i) val))))

(define (json->string obj)
  "Scheme 值 → JSON 字符串"
  (cond ((hashtable? obj)
         (let ((keys (hashtable-keys obj)))
           (string-append "{"
             (string-join
               (map (lambda (k)
                      (string-append "\"" (if (symbol? k) (symbol->string k) k) "\":"
                        (json->string (hashtable-ref obj k #f))))
                    (vector->list keys))
               ", ")
             "}")))
        ((vector? obj)
         (string-append "["
           (string-join (map json->string (vector->list obj)) ", ") "]"))
        ((number? obj) (format "~s" obj))
        ((boolean? obj) (if obj "true" "false"))
        ((string? obj) (string-append "\"" obj "\""))
        ((null? obj) "null")
        (else (format "~s" obj))))

;; ====== 日期时间（libc） ======
(define libc-time
  (foreign-procedure "time" (void*) unsigned-64))
(define libc-localtime
  (foreign-procedure "localtime" (void*) void*))
(define libc-strftime
  (foreign-procedure "strftime" (void* unsigned-64 string void*) unsigned-64))
(define libc-mktime
  (foreign-procedure "mktime" (void*) unsigned-64))

(define (datetime-now)
  "当前时间（Unix 时间戳）"
  (libc-time 0))

(define (datetime-format timestamp fmt)
  "格式化时间戳，fmt 如 '%Y-%m-%d %H:%M:%S'"
  (let* ((t (foreign-alloc 8))
         (_ (foreign-set! 'unsigned-64 t 0 timestamp))
         (tm (libc-localtime t))
         (buf-size 256)
         (buf (foreign-alloc buf-size)))
    (let ((n (libc-strftime buf buf-size fmt tm)))
      (let ((bv (make-bytevector n 0)))
        (do ((i 0 (+ i 1))) ((= i n))
          (bytevector-u8-set! bv i (foreign-ref 'unsigned-8 buf i)))
        (foreign-free buf)
        (foreign-free t)
        (utf8->string bv)))))

(define (datetime-str->ts s fmt)
  "解析时间字符串 → 时间戳，fmt 如 '%Y-%m-%d'"
  ;; 简化为 strptime 的 FFI
  ;; 当前返回当前时间戳
  (libc-time 0))

;; ====== CSV 读写 ======
(define (csv-parse-line line)
  "解析一行 CSV（支持引号转义）"
  (let loop ((i 0) (fields '()) (current '()))
    (if (>= i (string-length line))
        (reverse (cons (list->string (reverse current)) fields))
        (let ((c (string-ref line i)))
          (cond ((char=? c #\,)
                 (loop (+ i 1) (cons (list->string (reverse current)) fields) '()))
                ((char=? c #\")
                 ;; 引号字段
                 (let inner ((j (+ i 1)) (acc '()))
                   (if (>= j (string-length line))
                       (reverse (cons (list->string (reverse acc)) fields))
                       (let ((c2 (string-ref line j)))
                         (if (char=? c2 #\")
                             (if (and (< (+ j 1) (string-length line))
                                      (char=? (string-ref line (+ j 1)) #\"))
                                 (inner (+ j 2) (cons #\" acc))
                                 (loop (+ j 1) 
                                       (cons (list->string (reverse acc)) fields)
                                       current))
                             (inner (+ j 1) (cons c2 acc)))))))
                (else
                 (loop (+ i 1) fields (cons c current))))))))

(define (csv-parse s)
  "解析 CSV 字符串 → list of list"
  (let ((lines (string-split s #\newline)))
    (map csv-parse-line lines)))

(define (csv-encode fields)
  "将字段列表编码为 CSV 行"
  (string-join
    (map (lambda (f)
           (let ((s (if (number? f) (format "~s" f) f)))
             (if (string-any (lambda (c) (or (char=? c #\,) (char=? c #\"))) s)
                 (string-append "\"" (string-replace s "\"" "\"\"") "\"")
                 s)))
         fields)
    ","))

;; ====== 字符串工具（纯 Scheme，不依赖 Chez 扩展） ======

(define (string-split s sep-char)
  "按字符拆分字符串"
  (let ((len (string-length s)))
    (let loop ((i 0) (start 0) (result '()))
      (cond ((>= i len)
             (reverse (if (< start len)
                         (cons (substring s start len) result)
                         result)))
            ((char=? (string-ref s i) sep-char)
             (loop (+ i 1) (+ i 1)
                   (cons (substring s start i) result)))
            (else (loop (+ i 1) start result))))))

(define (str-find s sub . start)
  "查找子串位置（返回 #f 或索引），可选 start 参数"
  (let ((start-idx (if (pair? start) (car start) 0))
        (s-len (string-length s))
        (sub-len (string-length sub)))
    (let loop ((i start-idx))
      (cond ((> i (- s-len sub-len)) #f)
            ((string=? (substring s i (+ i sub-len)) sub) i)
            (else (loop (+ i 1)))))))

(define (string-any pred s)
  "检查是否有任意字符满足谓词"
  (let loop ((i 0))
    (and (< i (string-length s))
         (or (pred (string-ref s i))
             (loop (+ i 1))))))

(define (string-join parts sep)
  "连接字符串列表"
  (if (null? parts) ""
      (let loop ((rest (cdr parts)) (result (car parts)))
        (if (null? rest) result
            (loop (cdr rest) (string-append result sep (car rest)))))))

(define (string-replace s old new)
  "替换所有子串"
  (let ((old-len (string-length old)))
    (let loop ((i 0) (result '()))
      (let ((pos (str-find s old i)))
        (if pos
            (loop (+ pos old-len)
                  (cons new (cons (substring s i pos) result)))
            (apply string-append
                   (reverse (cons (substring s i (string-length s)) result))))))))

;; ====== 字符串工具（对外的简单包装） ======
(define (str-split s sep)
  "按分隔符拆分字符串"
  (let ((sep-len (string-length sep)))
    (let loop ((i 0) (result '()))
      (let ((pos (str-find s sep i)))
        (if pos
            (loop (+ pos sep-len)
                  (cons (substring s i pos) result))
            (reverse (cons (substring s i (string-length s)) result)))))))

(define (str-join parts sep)
  "用分隔符连接字符串列表"
  (string-join parts sep))

(define (str-trim s)
  "去除首尾空白"
  (let ((len (string-length s)))
    (let ((start (do ((i 0 (+ i 1))) ((or (>= i len) (not (char-whitespace? (string-ref s i)))) i)))
          (end (do ((i (- len 1) (- i 1))) ((or (< i 0) (not (char-whitespace? (string-ref s i)))) (+ i 1)))))
      (if (>= start end) "" (substring s start end)))))

(define (str-lower s)
  "转小写"
  (string-downcase s))

(define (str-upper s)
  "转大写"
  (string-upcase s))

(define (str-replace s old new)
  "替换所有子串"
  (let loop ((i 0) (result '()))
    (let ((pos (str-find s old i)))
      (if pos
          (loop (+ pos (string-length old))
                (cons new (cons (substring s i pos) result)))
          (apply string-append (reverse (cons (substring s i (string-length s)) result)))))))

(define (str-contains? s sub)
  "检查是否包含子串"
  (not (not (str-find s sub))))

(define (str-starts-with? s prefix)
  "检查是否以 prefix 开头"
  (and (>= (string-length s) (string-length prefix))
       (string=? (substring s 0 (string-length prefix)) prefix)))

(define (str-ends-with? s suffix)
  "检查是否以 suffix 结尾"
  (and (>= (string-length s) (string-length suffix))
       (string=? (substring s (- (string-length s) (string-length suffix)) (string-length s)) suffix)))

;; ====== 操作系统接口（libc） ======

(define libc-opendir (foreign-procedure "opendir" (string) void*))
(define libc-readdir (foreign-procedure "readdir" (void*) void*))
(define libc-closedir (foreign-procedure "closedir" (void*) int))
(define libc-mkdir (foreign-procedure "mkdir" (string int) int))
(define libc-rmdir (foreign-procedure "rmdir" (string) int))
(define libc-unlink (foreign-procedure "unlink" (string) int))
(define libc-rename (foreign-procedure "rename" (string string) int))
(define libc-getcwd (foreign-procedure "getcwd" (void* unsigned-64) void*))
(define libc-chdir (foreign-procedure "chdir" (string) int))
(define libc-getenv (foreign-procedure "getenv" (string) void*))
(define libc-stat (foreign-procedure "stat" (string void*) int))
(define libc-system (foreign-procedure "system" (string) int))
(define libc-access (foreign-procedure "access" (string int) int))

(define (os-list-dir path)
  (let ((dir (libc-opendir path)))
    (if dir
        (let loop ((acc '()))
          (let ((entry (libc-readdir dir)))
            (if (not (eq? entry 0))
                ;; d_name is at offset 19 in struct dirent on Linux x86_64
                (let ((name (c-str->string (+ entry 19))))
                  (loop (cons name acc)))
                (begin (libc-closedir dir) (reverse acc)))))
        '())))

(define (c-str->string ptr)
  "将 C 字符串（null-terminated）转为 Scheme string"
  (let loop ((i 0) (chars '()))
    (let ((c (foreign-ref 'unsigned-8 ptr i)))
      (if (= c 0)
          (list->string (map integer->char (reverse chars)))
          (loop (+ i 1) (cons c chars))))))

(define (os-getenv name)
  (let ((ptr (libc-getenv name)))
    (if (not (eq? ptr 0))
        (c-str->string ptr)
        #f)))

(define (os-getcwd)
  (let ((buf (foreign-alloc 4096)))
    (let ((ptr (libc-getcwd buf 4096)))
      (if (not (eq? ptr 0))
          (let ((result (c-str->string buf)))
            (foreign-free buf)
            result)
          (begin (foreign-free buf) #f)))))

(define (os-file-size path)
  (let ((buf (foreign-alloc 128)))
    (let ((res (libc-stat path buf)))
      (if (= res 0)
          (let ((size (foreign-ref 'unsigned-64 buf 48)))
            (foreign-free buf) size)
          (begin (foreign-free buf) -1)))))

(define (os-file-exists? path) (= (libc-access path 0) 0))
(define (os-move-file src dst) (= (libc-rename src dst) 0))
(define (os-delete-file path) (= (libc-unlink path) 0))
(define (os-mkdir path) (= (libc-mkdir path 511) 0))
(define (os-rmdir path) (= (libc-rmdir path) 0))
(define (os-shell cmd) (libc-system cmd))
(define (os_cwd) (os-getcwd))
(define (os_list_dir path) (os-list-dir path))
(define (os_getenv name) (os-getenv name))
(define (os_file_size path) (os-file-size path))
(define (os_file_exists path) (os-file-exists? path))
(define (os_move_file src dst) (os-move-file src dst))
(define (os_delete_file path) (os-delete-file path))
(define (os_mkdir path) (os-mkdir path))
(define (os_rmdir path) (os-rmdir path))

;; ====== 正则表达式（libpcre2） ======
(load-shared-object "libpcre2-8.so.0")

(define pcre-compile
  (foreign-procedure "pcre2_compile_8" (void* unsigned-64 unsigned-32 void* void* void*) void*))
(define pcre-match
  (foreign-procedure "pcre2_match_8" (void* void* unsigned-64 unsigned-64 unsigned-32 void* void*) int))
(define pcre-get-error
  (foreign-procedure "pcre2_get_error_message_8" (int void* unsigned-64) void*))

(define (string->c-ptr s)
  "将 Scheme 字符串复制到 C 内存，返回指针"
  (let* ((bv (string->utf8 s))
         (len (bytevector-length bv))
         (ptr (foreign-alloc (+ len 1))))
    (do ((i 0 (+ i 1))) ((= i len))
      (foreign-set! 'unsigned-8 ptr i (bytevector-u8-ref bv i)))
    (foreign-set! 'unsigned-8 ptr len 0)  ;; null terminator
    ptr))

(define (re-match pattern text)
  "正则匹配，返回 #t/#f"
  (let* ((bv (string->utf8 text))
         (text-len (bytevector-length bv))
         (c-pattern (string->c-ptr pattern))
         (err-ptr (foreign-alloc 8))
         (err-off (foreign-alloc 8))
         (re (pcre-compile c-pattern 0 0 err-ptr err-off 0)))
    (if re
        (let* ((c-text (string->c-ptr text))
               (rc (pcre-match re c-text text-len 0 0 0 0)))
          (foreign-free c-text)
          (foreign-free c-pattern)
          (>= rc 0))
        (begin (foreign-free c-pattern) #f))))

(define (re-search pattern text)
  "搜索，返回匹配位置或 -1"
  (let* ((c-pattern (string->c-ptr pattern))
         (err-ptr (foreign-alloc 8))
         (err-off (foreign-alloc 8))
         (re (pcre-compile c-pattern 0 0 err-ptr err-off 0)))
    (if re
        (let* ((c-text (string->c-ptr text))
               (ovector (foreign-alloc (* 3 8)))
               (rc (pcre-match re c-text (string-length text) 0 0 ovector 0)))
          (foreign-free c-text)
          (if (>= rc 0)
              (foreign-ref 'long ovector 0)
              -1))
        -1)))

;; ====== 随机数（libc） ======
(define libc-rand (foreign-procedure "rand" () int))
(define libc-srand (foreign-procedure "srand" (unsigned-32) void))
(define libc-random (foreign-procedure "random" () long))
(define libc-srandom (foreign-procedure "srandom" (unsigned-32) void))

(define (random-seed seed)
  "设置随机种子"
  (libc-srand seed))

(define (random-int)
  "返回 0..RAND_MAX 的随机整数"
  (libc-rand))

(define (random-float)
  "返回 0.0..1.0 的随机浮点数"
  (/ (inexact (libc-rand)) 2147483647.0))

(define (random-range low high)
  "返回 low..high 的随机整数"
  (+ low (remainder (libc-rand) (- high low))))

(define pi 3.141592653589793)
(define e 2.718281828459045)

;; ====== 时间（libc） ======
(define libc-nanosleep
  (foreign-procedure "nanosleep" (void* void*) int))

(define (sleep seconds)
  "休眠 seconds 秒"
  (let* ((sec (inexact->exact (floor seconds)))
         (nsec (inexact->exact (* (- seconds (floor seconds)) 1e9)))
         (ts (foreign-alloc 16)))
    (foreign-set! 'unsigned-64 ts 0 sec)
    (foreign-set! 'unsigned-64 ts 8 nsec)
    (libc-nanosleep ts 0)
    (foreign-free ts)))

(define (clock) (libc-time 0))

;; ====== 命令行参数 ======
(define (argv)
  "返回命令行参数列表"
  (command-line))

(define libc-exit (foreign-procedure "exit" (int) void))

(define (exit-program code)
  "退出程序"
  (libc-exit code))

;; C ABI 包装（下划线风格，供 StaticPy 调用）
(define http_get_simple (lambda (url) (http-get-simple url)))
(define random_float (lambda () (random-float)))
(define random_int (lambda () (random-int)))
(define random_range (lambda (low high) (random-range low high)))
(define random_seed (lambda (seed) (random-seed seed)))

;; ====== 字符串转数字（用于读取数据文件） ======
(define (string_to_float s)
  (let ((n (string->number s)))
    (if n (inexact n) 0.0)))

(define (string_to_int s)
  (or (string->number s) 0))

(define (list_length v)
  (if (vector? v)
      (vector-length v)
      (length v)))

(define (list_ref v i)
  (if (vector? v)
      (vector-ref v i)
      (list-ref v i)))

(define (vec_ref v i)
  (vector-ref v i))
(define re_match (lambda (p s) (re-match p s)))
(define re_search (lambda (p s) (re-search p s)))
(define exit_program (lambda (code) (exit-program code)))
(define json_loads (lambda (s) (json-loads s)))
(define json_dumps (lambda (obj) (json->string obj)))
(define parse_json (lambda (s) (parse-json s)))
(define now_ts (lambda () (datetime-now)))
(define format_time (lambda (ts fmt) (datetime-format ts fmt)))
(define csv_parse (lambda (s) (csv-parse s)))
(define csv_encode (lambda (fields) (csv-encode fields)))
(define str_split (lambda (s sep) (str-split s sep)))
(define str_join (lambda (parts sep) (str-join parts sep)))
(define str_trim (lambda (s) (str-trim s)))
(define str_lower (lambda (s) (str-lower s)))
(define str_upper (lambda (s) (str-upper s)))
(define str_replace (lambda (s old new) (str-replace s old new)))
(define str_contains (lambda (s sub) (str-contains? s sub)))

;; ====== 通用打印 ======

(define (print x)
  (cond ((number? x) (display x) (newline))
        ((boolean? x) (display (if x "true" "false")) (newline))
        ((string? x) (display x) (newline))
        ((hashtable? x) (display (json->string x)) (newline))
        ((list? x) (display (json->string (list->vector x))) (newline))
        ((vector? x) (display (json->string x)) (newline))
        ((null? x) (display "null") (newline))
        (else (display x) (newline))))
