# torch_helper.static.py — libcomfycli_torch.so 绑定说明
#
# 这些函数不在 StaticPy 里用 extern 声明，而是在 comfycli_ffi.scm 中
# 按共享库是否加载来定义（未加载时退化为报错桩），避免 foreign-procedure
# 在 AOT 载入期因符号缺失而让整个程序起不来。
#
# 名字由 concat_src.py 的 HEADER import 提供给类型检查器：
#   torch_std_safetensors_load / _count / _save / _merge / _free / torch_std_copy_file
