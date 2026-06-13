;; static_prelude.scm — StaticPy 值类型运行时
;; 
;; 值类型: int=fixnum, float=flonum, bool=boolean
;; 数组类型: float[] = foreign-alloc 分配的 C double*
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
  "创建 n 个 double 的 C 数组，自动 GC 回收"
  (let ((ptr (foreign-alloc (* n 8))))
    (do ((i 0 (+ i 1))) ((= i n))
      (foreign-set! 'double ptr (* i 8) 0.0))
    (gc-register-ptr ptr)
    ptr))

(define (float-array-set ptr i val)
  (foreign-set! 'double ptr (* i 8) val))

(define (float-array-ref ptr i)
  (foreign-ref 'double ptr (* i 8)))

(define (float-array-free ptr)
  "手动释放（也可等 GC 自动回收）"
  (foreign-free ptr))

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

;; C ABI 包装（供 extern fn from "prelude" 调用）
(define make_float_array (lambda (n) (make-float-array n)))
(define float_array_set (lambda (ptr i val) (float-array-set ptr i val)))
(define float_array_ref (lambda (ptr i) (float-array-ref ptr i)))
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

;; Source: <input>

;; StaticPy functions
(define (static_arr_fill a v n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set a i v) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_copy dst src n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set dst i (float_array_ref src i)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_add a b r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (+ (float_array_ref a i) (float_array_ref b i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_sub a b r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (- (float_array_ref a i) (float_array_ref b i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_mul a b r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (* (float_array_ref a i) (float_array_ref b i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_mul_scalar a s n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set a i (* (float_array_ref a i) s)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_add_scalar a s n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set a i (+ (float_array_ref a i) s)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_div a b r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (/ (float_array_ref a i) (float_array_ref b i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_div_scalar a s n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set a i (/ (float_array_ref a i) s)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_pow a e r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (pow (float_array_ref a i) e)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_exp a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (exp (float_array_ref a i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_sqrt a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (sqrt (float_array_ref a i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_rsqrt a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (/ 1.0 (sqrt (float_array_ref a i)))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_clip a lo hi n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (set! v (float_array_ref a i)) (if (< v lo)
        (begin (float_array_set a i lo))
        (begin (if (> v hi)
        (begin (float_array_set a i hi))))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_sigm a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (set! x (float_array_ref a i)) (float_array_set r i (/ 1.0 (+ 1.0 (exp (- x))))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_tanh a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (set! x (float_array_ref a i)) (if (> x 20.0)
        (begin (float_array_set r i 1.0))
        (begin (if (< x (- 20.0))
        (begin (float_array_set r i (- 1.0)))
        (begin (set! e2x (exp (* 2.0 x))) (float_array_set r i (/ (- e2x 1.0) (+ e2x 1.0))))))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_relu a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (set! v (float_array_ref a i)) (float_array_set r i (if (> v 0.0) v 0.0)) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_silu a r n)
  (begin
    "SiLU = x * sigmoid(x)"
    (set! i 0)
    (let loop () (if (< i n) (begin (set! x (float_array_ref a i)) (float_array_set r i (/ x (+ 1.0 (exp (- x))))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_gelu a r n)
  (begin
    "GELU = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"
    (set! coeff 0.7978845608028654)
    (set! i 0)
    (let loop () (if (< i n) (begin (set! x (float_array_ref a i)) (set! x3 (* (* x x) x)) (set! inner (* coeff (+ x (* 0.044715 x3)))) (if (> inner 20.0)
        (begin (set! tanh_v 1.0))
        (begin (if (< inner (- 20.0))
        (begin (set! tanh_v (- 1.0)))
        (begin (set! e2x (exp (* 2.0 inner))) (set! tanh_v (/ (- e2x 1.0) (+ e2x 1.0))))))) (float_array_set r i (* (* 0.5 x) (+ 1.0 tanh_v))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_neg a r n)
  (begin
    (set! i 0)
    (let loop () (if (< i n) (begin (float_array_set r i (- (float_array_ref a i))) (set! i (+ i 1)) (loop))))
  ))
(define (static_arr_sum a n)
  (begin
    (set! s 0.0)
    (set! i 0)
    (let loop () (if (< i n) (begin (set! s (+ s (float_array_ref a i))) (set! i (+ i 1)) (loop))))
    s
  ))
(define (static_arr_mean a n)
  (/ (static_arr_sum a n) n))
(define (static_arr_max a n)
  (begin
    (set! m (float_array_ref a 0))
    (set! i 1)
    (let loop () (if (< i n) (begin (set! v (float_array_ref a i)) (if (> v m)
        (begin (set! m v))) (set! i (+ i 1)) (loop))))
    m
  ))
(define (static_arr_min a n)
  (begin
    (set! m (float_array_ref a 0))
    (set! i 1)
    (let loop () (if (< i n) (begin (set! v (float_array_ref a i)) (if (< v m)
        (begin (set! m v))) (set! i (+ i 1)) (loop))))
    m
  ))
(define (static_softmax x n)
  (begin
    "就地 softmax 沿最后一维"
    (set! max_val (static_arr_max x n))
    (static_arr_add_scalar x (- max_val) n)
    (static_arr_exp x x n)
    (set! s (static_arr_sum x n))
    (if (> s 0.0)
    (begin (static_arr_div_scalar x s n)))
  ))
(define (static_softmax_2d x rows cols)
  (begin
    "softmax 对每行分别做"
    (set! r 0)
    (let loop () (if (< r rows) (begin (set! offset (* r cols)) (set! max_val (float_array_ref x offset)) (set! c 1) (let loop () (if (< c cols) (begin (set! v (float_array_ref x (+ offset c))) (if (> v max_val)
        (begin (set! max_val v))) (set! c (+ c 1)) (loop)))) (set! s 0.0) (set! c 0) (let loop () (if (< c cols) (begin (set! ev (exp (- (float_array_ref x (+ offset c)) max_val))) (float_array_set x (+ offset c) ev) (set! s (+ s ev)) (set! c (+ c 1)) (loop)))) (if (> s 0.0)
        (begin (set! c 0) (let loop () (if (< c cols) (begin (float_array_set x (+ offset c) (/ (float_array_ref x (+ offset c)) s)) (set! c (+ c 1)) (loop)))))) (set! r (+ r 1)) (loop))))
  ))
(define (static_layer_norm x gamma beta n_features n_elements)
  (begin
    "LayerNorm: 对每个样本的特征维做归一化
    x: [n_elements] 展平的
    n_features: 特征维度大小
    n_elements / n_features = batch_size * spatial_dims
    "
    (set! step n_features)
    (set! n_samples (quotient n_elements n_features))
    (set! s 0)
    (let loop () (if (< s n_samples) (begin (set! offset (* s step)) (set! mean 0.0) (set! i 0) (let loop () (if (< i step) (begin (set! mean (+ mean (float_array_ref x (+ offset i)))) (set! i (+ i 1)) (loop)))) (set! mean (/ mean step)) (set! var 0.0) (set! i 0) (let loop () (if (< i step) (begin (set! diff (- (float_array_ref x (+ offset i)) mean)) (set! var (+ var (* diff diff))) (set! i (+ i 1)) (loop)))) (set! var (/ var step)) (set! inv_std (/ 1.0 (sqrt (+ var 1e-05)))) (set! i 0) (let loop () (if (< i step) (begin (set! norm (* (- (float_array_ref x (+ offset i)) mean) inv_std)) (float_array_set x (+ offset i) (+ (* norm (float_array_ref gamma i)) (float_array_ref beta i))) (set! i (+ i 1)) (loop)))) (set! s (+ s 1)) (loop))))
  ))
(define (static_group_norm x gamma beta n_groups n_channels spatial)
  (begin
    "GroupNorm: 对每组 channels 做归一化
    x: [n_channels * spatial] 展平的
    每组大小 = n_channels // n_groups * spatial
    "
    (set! group_size (* (quotient n_channels n_groups) spatial))
    (set! g 0)
    (let loop () (if (< g n_groups) (begin (set! offset (* g group_size)) (set! mean 0.0) (set! i 0) (let loop () (if (< i group_size) (begin (set! mean (+ mean (float_array_ref x (+ offset i)))) (set! i (+ i 1)) (loop)))) (set! mean (/ mean group_size)) (set! var 0.0) (set! i 0) (let loop () (if (< i group_size) (begin (set! diff (- (float_array_ref x (+ offset i)) mean)) (set! var (+ var (* diff diff))) (set! i (+ i 1)) (loop)))) (set! var (/ var group_size)) (set! inv_std (/ 1.0 (sqrt (+ var 1e-05)))) (set! i 0) (let loop () (if (< i group_size) (begin (set! ch (quotient (+ offset i) spatial)) (set! norm (* (- (float_array_ref x (+ offset i)) mean) inv_std)) (float_array_set x (+ offset i) (+ (* norm (float_array_ref gamma ch)) (float_array_ref beta ch))) (set! i (+ i 1)) (loop)))) (set! g (+ g 1)) (loop))))
  ))

;; Entry point
(static_main)
