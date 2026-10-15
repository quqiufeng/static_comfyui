#!/usr/bin/env python3
"""concat.py — 把若干 .static.py 合并为一个 bundle（供 static_build 编译）。

用法: concat.py lib1.static.py lib2.static.py user.static.py > bundle.static.py
规则：去掉 `import`/`from ... import ...` 行（名字由合并后同一作用域提供）；
      只应有一个文件定义 main()（通常是用户程序或 CLI）。
"""
import sys, re
parts = []
for f in sys.argv[1:]:
    s = open(f).read()
    s = re.sub(r'(?m)^\s*(from\s+\S+\s+import\s+[^\n]*|import\s+[^\n]*)\s*$', '', s)
    parts.append(f"# ==== {f} ====\n" + s)
sys.stdout.write("\n\n".join(parts))
