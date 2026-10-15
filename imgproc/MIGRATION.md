# 用 Python 自省驱动 API 对齐（自动化迁移方法）

把 Python 库（Pillow、numpy、cv2 …）迁到 StaticPy 的**可复制流程**。核心思想：
Python 的自省（`inspect` / `dir()` / 签名）能**机器枚举出全部接口**，于是"覆盖面"
可以自动化；语义则用**差分测试**（跑同一个操作，逐像素比对）来自动验证。两者配一对，
就是一条"自省 → 铺桩 → 差分对齐"的闭环——**人只写实现，对不齐由机器报**。

```
   ┌── 1. 自省 ──────────────┐   inspect / dir / signature
   │  枚举目标库全部函数/类/签名 │   → API 清单（每个名字 = 一个待迁接口）
   └──────────────┬──────────┘
                  ▼
   ┌── 2. 铺桩 ──────────────┐   清单 → extern fn 声明 + StaticPy 骨架 + C ABI
   │  一次性把接口面铺满          │   （"一口气全铺完"）
   └──────────────┬──────────┘
                  ▼
   ┌── 3. 差分对齐 ──────────┐   同输入 → PIL 结果 vs 我们结果 → 逐像素 diff
   │  绿=对齐，红=有差异         │   （"再调"：红项逐个修，直到全绿）
   └──────────────┬──────────┘
                  │
            回到 1（扩下一批接口 / 下一个库）
```

## 1. 自省：把 API 面变成清单

`imgproc/pil_surface.py` 遍历 PIL 各模块的公开名字，用 `inspect.signature` 取签名，
产出 `imgproc/pil_api.md`：

```bash
/data/venv/bin/python imgproc/pil_surface.py > imgproc/pil_api.md
```

- `dir(mod)` 过滤 `_` 前缀 → 公开名；
- `inspect.isclass / isfunction` 分类；`inspect.signature(obj)` 取参数表；
- 输出 markdown 表（函数/签名 + 类/成员）。

**这一步只给"接口"（名字/签名/文档），给不了"语义"。** 语义交给下一步。

## 2. 铺桩：清单 → FFI 声明 + 骨架

对清单里每个待迁接口：

- **C++ 底层库**（`cpp/imgproc/*.cpp`）：实现 C ABI（`imgproc.h`）。这是"重活"，对标
  Pillow 的 C 核心 `libImaging` + 编解码库。FFI 不擅长数组，故多值参数（点列/LUT/调色板）
  走 **CSV 字符串**；返回 `void` 的 C 函数一律**返回 `int`**（翻译器类型表没有 `void`）。
- **FFI 加载**（`imgproc_ffi.scm`）：只 `load-shared-object`；`foreign-procedure` 由 `.py` 的
  `extern fn name(args) -> ret from "imgproc"` 自动生成。
- **StaticPy API**（`*.static.py`）：`def main()` 里编排（**不在顶层调用 main**）。

> 关键：**照着自省出来的签名写**，别凭记忆——签名/默认值/参数顺序都以清单为准。

## 3. 差分对齐：机器报差异

`imgproc/difftest.py` 对每个算子：同一输入分别用 **PIL** 与 **我们的库**（经 `pyimgproc.py`
的 ctypes 绑定）跑一遍，转成 numpy 做**逐像素 diff**，报 `max/mean`：

```bash
/data/venv/bin/python imgproc/difftest.py -v      # 全绿即对齐
```

- 用 ctypes 直连 `.so` 跑差分，比走 StaticPy ELF 快，迭代顺（生产仍走 StaticPy）。
- 对不上的项，**去读 Pillow 源码对齐算法**（这正是自动化的价值——不用猜）：
  - 卷积：`src/libImaging/Filter.c`（浮点 `+0.5` 取整、**边界复制**、3×3 垂直翻转）；
  - 重采样：`src/libImaging/Resample.c`（定点系数 `PRECISION_BITS=22`、两趟 separable）；
  - `ImageOps/ImageEnhance`：`.py` 源（如 Brightness = `blend(black, image, factor)`）。
- 改完重编重跑，**直到 max=0**。

## 现状

`difftest.py`：**38/38 全绿（max=0）**，含 `resize.{nearest,bilinear,lanczos}`、全部
`transpose`、`ImageOps`、`ImageChops`、`ImageEnhance`、`ImageFilter`（卷积精确对齐）。

## 复用到别的库

把 `pil_surface.py` 的模块列表一换（numpy/cv2/…）、`difftest.py` 的用例一换即可。
C ABI 是稳定面，**多前端可绑**：StaticPy ELF、LuaJIT FFI、Python ctypes 调同一份 `.so`。
