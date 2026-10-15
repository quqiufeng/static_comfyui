# imgproc —— 图像处理库的 StaticPy 重写区

把此前用 **Python/PIL** 做的图像处理，改用 **StaticPy** 重写：处理逻辑写在
`.static.py` 里，编解码等底层能力交给一个运行时加载的 **C++ 库**。产物是
**独立 ELF**（不依赖 Python / pip / venv / PIL）。

和推理栈（`comfycli` / `sdcpp_adapter` / torch）**完全无关**，独立成一块。

## 目录

```
imgproc/
├── README.md
├── build.sh                 # 一键构建：C++ 库 + StaticPy ELF
├── imgproc_ffi.scm          # 运行时加载 libimgproc.so（static_build 的第 3 个参数）
├── mirror_gif.static.py     # 示例：逐帧水平镜像 GIF（PIL-free）
├── mirror_gif + .so         # 构建产物（ELF / Chez AOT ），已 gitignore
└── ../cpp/imgproc/
    ├── imgproc.h/.cpp       # C ABI 图像库（GIF 解码/编码 + 像素缓冲）
    ├── thirdparty/giflib/gif_lib.h  # 随附的 giflib 头
    └── build/libimgproc.so  # 构建产物，动态依赖 libgif.so.7（已 gitignore）
```

## 构建 / 运行

```bash
bash imgproc/build.sh

LD_LIBRARY_PATH=$PWD/cpp/imgproc/build \
  ./imgproc/mirror_gif IN.gif OUT.gif
```

## 架构（三层，各改各的）

1. **C++ 底层库** `cpp/imgproc/imgproc.cpp` → `libimgproc.so`
   - C ABI：`img_open/img_clone_blank/img_frame_ptr/img_write/img_free` 等。
   - 只依赖 **libgif.so.7**（运行时动态解析）+ 标准库。
   - 帧模型：**保留原调色板 + 索引**（bpp=1），或 RGBA（bpp=4）。GIF 解码保留
     逐帧色表，因此镜像等逐像素操作**无损**（不重新量化、不变色）。
2. **FFI 加载** `imgproc/imgproc_ffi.scm`
   - 只做 `(load-shared-object "libimgproc.so")`；具体 `foreign-procedure`
     绑定由 `static_translate.py` 依据 `.py` 里的 `extern fn ... from "imgproc"` 生成。
3. **StaticPy 程序** `imgproc/mirror_gif.static.py`
   - 真正的「镜像」逻辑在这一层：在原始像素缓冲上逐像素搬运。

## 写新算子

1. 需要的新底层能力 → 加到 `cpp/imgproc/imgproc.{h,cpp}`（C ABI）。
2. `.py` 顶部用 `extern fn name(args) -> ret from "imgproc"` 声明。
3. 主逻辑写在 `def main():` 里（**不要在顶层调用 main**；翻译器自动把
   `(static_main)` 当入口，重复调用会跑两遍）。
4. `bash imgproc/build.sh`。

`mirror_gif.static.py` 里的类型细节（都别删）：
- FFI 类型表没有 `void`，所以返回 void 的 C 函数一律返回 `int`；
- 入参用 `str / int / ptr`；`ptr` 映射为 `void*`；
- 取到的指针用 `img_u8_get/set(ptr, off)` 读写字节（C 侧按偏移索引）。

## 备注

- 输入 GIF 的像素**逐像素完全一致**地镜像（用 PIL 对照验证 maxdiff = 0，
  比 PIL「转 RGB 再存」的路径更准——后者会重新量化）。
- 目前 `img_open` 只认 GIF；扩 PNG/JPEG 时在同一层加分支即可。
