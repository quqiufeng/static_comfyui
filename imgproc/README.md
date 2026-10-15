# imgproc —— 图像处理库的 StaticPy 重写区（对标 Pillow）

把此前用 **Python/PIL** 做的图像处理，改用 **StaticPy** 重写：处理逻辑写在
`.static.py` 里，编解码/卷积等底层能力交给一个**运行时加载的 C++ 库**
（`libimgproc.so`，内部动态链接 `libgif.so.7` + 随附 stb 单头）。产物是**独立 ELF**
（不依赖 Python / venv / PIL）。与推理栈（comfycli / sdcpp_adapter）**无关**。

## 三层结构（各改各的）

| 层 | 文件 | 说明 |
|---|---|---|
| C++ 核心 | `../cpp/imgproc/*.cpp` → `libimgproc.so` | 对标 Pillow 的 C 核心 `libImaging` + 编解码 |
| FFI | `imgproc_ffi.scm` | 仅 `load-shared-object`；`foreign-procedure` 由 `.py` 的 `extern fn ... from "imgproc"` 生成 |
| StaticPy API | `*.static.py` | 处理逻辑（如镜像） |

C ABI 是稳定面，**多前端可绑**：StaticPy ELF、LuaJIT FFI、Python ctypes 都能直接调同一份 `.so`。

## 目录

```
imgproc/
├── build.sh                 # 一键：C++ 库 + StaticPy ELF
├── imgproc_ffi.scm
├── mirror_gif.static.py     # 示例算子：GIF 逐帧镜像（无损，走索引帧）
├── pyimgproc.py             # libimgproc.so 的 ctypes 绑定（差分测试用）
├── pil_surface.py           # 自省 Pillow 公开 API → 覆盖面清单
├── pil_api.md               # 上面生成的清单
├── difftest.py              # 自动化差分测试：PIL vs 我们，逐像素比对
└── ../cpp/imgproc/
    ├── core.cpp geom.cpp color.cpp ops.cpp filters.cpp draw.cpp
    ├── imgproc.h imgproc_internal.h
    ├── build_cpp.sh
    └── thirdparty/{giflib/stb}
```

## 构建 / 运行

```bash
bash imgproc/build.sh
LD_LIBRARY_PATH=$PWD/cpp/imgproc/build ./imgproc/mirror_gif IN.gif OUT.gif
```

## 自动化迁移夹具（自省 → 铺桩 → 差分对齐）

整套方法论见 **[MIGRATION.md](./MIGRATION.md)**。要点：Python 自省把"接口面"机器枚举出来，
差分测试把"语义"自动验证——人只写实现，对不齐由机器报。

```bash
# 1) 自省 Pillow API → 清单（每个名字 = 一个待迁移接口）
/data/venv/bin/python imgproc/pil_surface.py > imgproc/pil_api.md

# 2) 差分测试：同一操作 PIL / 我们各跑一遍，逐像素比对
/data/venv/bin/python imgproc/difftest.py -v
```

`difftest.py` 是"再调"的驱动：改动 C++ 后重跑，**绿了就对齐了**（当前 38/38 全绿）。
对不上的项照 Pillow 源码对齐（`Filter.c` / `Resample.c` / `ImageOps.py` …）。

## 当前覆盖 / 状态

- **编解码**：GIF（多帧，保留索引+调色板，无损）、PNG/JPEG/BMP/TGA（stb）。
- **Image 核心**：open/save/new/copy/convert(RGB·RGBA·L·LA·P)/split/merge/getchannel/getbbox/frombytes/to_bytes。
- **几何**：crop · resize（+带 box）· thumbnail · rotate · transpose(7) · pad · scale · paste · paste(mask) · composite · alpha_composite(带位置) · blend · fit · contain。
- **ImageOps**：invert · grayscale · posterize · solarize · autocontrast · equalize · colorize · mirror · expand · fit · contain · pad。
- **ImageChops**：add/subtract/difference/multiply/screen/darker/lighter/blend · **add_modulo/subtract_modulo** · **soft_light/hard_light/overlay** · logical_and/or/xor · const · offset。**ImageColor**：getrgb。
- **ImageEnhance**：brightness/contrast/color/sharpness。
- **ImageFilter**：BLUR·SMOOTH(S)·SHARPEN·DETAIL·EDGE_ENHANCE(MORE)·FIND_EDGES·CONTOUR·EMBOSS·UNSHARP·BoxBlur·GaussianBlur·Median/Min/Max/Mode/Rank —— **卷积已按 Pillow `Filter.c` 精确对齐（含浮点/边界/3×3 翻转）**。
- **ImageStat**：mean/median/extrema/stddev/sum。
- **ImageDraw**：point/line/rectangle/rounded_rectangle/ellipse/polygon/arc。**ImageFont**：stb_truetype 文字。

差分测试当前 **63 项全部通过（max=0）**——含 `resize.{nearest,bilinear,lanczos}`（已按 Pillow
`Resample.c` 定点算法精确对齐）、全部 transpose / Ops / Chops / Enhance / Filter。GIF 镜像与源逐像素 **maxdiff=0**。

## 写新算子

1. 新底层能力 → 加到 `cpp/imgproc/`（C ABI 在 `imgproc.h`）。
2. `.py` 顶部 `extern fn name(args) -> ret from "imgproc"` 声明。
3. 主逻辑写 `def main():`（**别在顶层调用 main**，翻译器自动加 `(static_main)`）。
4. `bash imgproc/build.sh`；`difftest.py` 验证。

FFI 细节：类型表**无 `void`**，返回 void 的 C 函数一律返回 `int`；入参 `str/int/float/ptr`；
多值参数（点列/LUT/调色板）用 CSV 字符串传。
