;; comfycli_ffi.scm — sd.cpp 适配层共享库 + 项目内置
;;
;; 通过 static_build.sh 的第 3 个参数引入，拼接在 static_stdlib.scm 之后、
;; 用户代码之前。这样 static_translate.py 无需为 "sdcpp_adapter" 打补丁。
;;
;; 注意：libsdcpp_adapter.so 通过 LD_LIBRARY_PATH 解析（本地 cpp/sd/build，
;; 部署时由 run.sh 设置）。

(load-shared-object "libsdcpp_adapter.so")

;; 上游 prelude 未提供 dict_keys（comfycli execution 需要）。
;; StaticPy 的 list 即 Scheme vector，故直接返回 hashtable-keys 的 vector。
(define (dict_keys d)
  (hashtable-keys d))

;; 判空谓词。上游翻译器不支持 `is None`/`is not None`（会生成非法代码），
;; 且 dict_get 缺失时返回 #f，故用 eq? #f 表示 None。
(define (is_none x) (eq? x #f))
(define (is_some x) (not (eq? x #f)))

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
