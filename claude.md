# static_comfyui — 开发最佳实践（2026-06-23 复盘总结）

## 0. 核心架构

```
StaticPy (管线编排) → C++ libTorch (模型 forward) → Chez AOT (ELF 编译)
                ↑                      ↑
         translate.py            libtorch_std_helper.cpp/.h
         (StaticPy→Scheme)       (extern "C" API)
```

**铁律**: 模型 forward 永远用 C++ libTorch 实现，不用 JIT TorchScript。
- ✅ C++ libTorch = `at::Tensor`, `at::linear()`, `at::scaled_dot_product_attention()`, 权重从 safetensors 加载
- ❌ JIT TorchScript = `torch::jit::load()` + `module->forward()` (只允许 T5 tokenizer、CLIP Vision 等工具模块)

---

## 1. 添加新模型的完整工作流

### Step 1: 在 comfyui_ref 找到源码
```
comfyui_ref/comfy/
├── ldm/pixart/pixartms.py      ← PixArt DiT 架构
├── ldm/cascade/stage_c.py      ← Stable Cascade Stage C
├── ldm/hunyuan_video/model.py  ← Hunyuan Video (注意 import: from comfy.ldm.flux.layers import ...)
├── ldm/wan/model.py            ← Wan Video
├── ldm/cosmos/model.py         ← Cosmos
├── gligen.py                   ← GLIGEN gated attention
```
**关键**: 先看 `import` 语句，看模型是否复用了已有组件（如 Hunyuan 复用 FLUX 的 DoubleStreamBlock）

### Step 2: 在 libtorch_std_helper.h 添加 extern "C" 函数声明
```c
void* torch_std_NEWMODEL_forward(
    void* sd_dict,          // safetensors dict (unordered_map<string, Tensor>)
    void* x,                // (B, C, H, W) input
    void* timestep,         // (B,) timesteps
    void* y,                // text embeddings
    ...model-specific params...);
```
**签名规范**:
- 张量传 `void*` (内部 cast 为 `at::Tensor*`)
- 标量传 `int`/`double`/`float`
- 权重用 `void* sd_dict` (指向 `std::unordered_map<std::string, at::Tensor>`)
- 返回值 `void*` (指向 `new at::Tensor(...)`)

### Step 3: 在 libtorch_std_helper.cpp 实现 C++ forward
```cpp
void* torch_std_NEWMODEL_forward(...) {
    try {
        auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);
        auto& x = unwrap(x_ptr);  // at::Tensor&
        
        // 使用 sdxl_get_weight / sdxl_linear / sdxl_conv2d 按名取权重
        auto h = sdxl_conv2d(x, d, "x_embedder.proj", 2, 0);
        auto w = sdxl_get_weight(d, "some_tensor");
        auto h = sdxl_linear(h, d, "some_layer");
        
        // 核心 forward 逻辑...
        
        return new at::Tensor(h);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return nullptr;
    }
}
```

**可用 C++ 辅助函数**:
| 函数 | 用途 |
|------|------|
| `sdxl_get_weight(d, name)` | 从 dict 按名取权重 Tensor |
| `sdxl_linear(x, d, name)` | 自动取 weight+bias 做 linear |
| `sdxl_conv2d(x, d, name, stride, pad)` | 自动取 weight+bias 做 conv2d |
| `timestep_embedding(t, dim)` | sin-cos timestep 编码 |
| `flux_block(img, txt, te, rope, d, prefix, ...)` | FLUX DoubleStreamBlock |
| `flux_rope(h, w, dim, ref_tensor)` | 2D RoPE 位置编码 |
| `wrap(t)` / `unwrap(p)` | void* ↔ Tensor 转换 |

### Step 4: 在 ops.static.py 添加 extern fn 声明
```python
extern fn torch_std_NEWMODEL_forward(
    sd_dict: ptr,
    x: ptr, timestep: ptr, y: ptr,
    ...) -> ptr from "torch_std_helper"
```

### Step 5: 编写 StaticPy 模块 (管线编排)
```python
# sd_runtime/sd_newmodel.static.py
from ops import *

_model_dict: ptr

def newmodel_init(safetensors_path: str) -> void:
    global _model_dict
    _model_dict = torch_std_safetensors_load(safetensors_path)

def newmodel_forward(latent: ptr, t: ptr, text: ptr) -> ptr:
    global _model_dict
    return torch_std_newmodel_forward(_model_dict, latent, t, text)
```

### Step 6: 添加 build_all.sh + 验证
```bash
# build_all.sh 添加新文件
python3 compiler/translate.py < combined_input.py  # 验证翻译无错
```

---

## 2. 已发现的严重 Bug (不要再犯)

### 🔴 translate.py 缺失 AST 节点处理

| 节点 | 旧行为 | 修复 |
|------|--------|------|
| `ast.BoolOp` (or/and) | `(void)` → 条件永远为真 | `(or ...)` / `(and ...)` |
| `ast.Continue` | `;; Continue:` 注释 → 循环不跳转 | `call/cc` + `(continue)` |
| `ast.Pass` | `;; Pass:` 注释 | 跳过 |
| `ast.Global` | `;; Global:` 注释 | 跳过 |

### 🔴 extern fn 声明缺失

症状: 函数调用被加上 `static_` 前缀 → 运行时 `undefined identifier`

**根因**: 每个 `.static.py` 单独过 translate.py 时，前一个文件的 extern fn 声明不共享。

**修复**: `build.sh` 拼接所有源文件 → 一次 translate.py 调用:
```bash
cat file1.py file2.py ... | python3 translate.py > output.ss
```

**检查**: 编译后 grep `static_torch_std_`，非 silu/gelu 的都是 bug。

### 🔴 `int()` 翻译语义错误

| Python | 旧 Scheme | 新 Scheme |
|--------|----------|----------|
| `int(torch_std_size(x,0))` | `(exact (round (torch_std_size x 0)))` | `(torch_std_size x 0)` |
| `int(steps / denoise)` | `(exact (round (/ steps denoise)))` | `(exact (truncate (/ steps denoise)))` |

- `round` 四舍五入 ≠ Python `int()` 截断
- extern fn 返回 int 的跳过 `exact/truncate`

### 🔴 `math.sqrt()` / `math.log()` → `(math sqrt x)` 调用不存在函数

**修复**: `ast.Attribute` 对 `obj_name == "math"` → `(libm-{attr} ...)`

### 🔴 JIT forward 用于模型 forward (架构错误)

```
❌ sd_pixart_forward: torch_std_jit_forward(jit_model, x, t, y)
✅ sd_pixart_forward: torch_std_pixart_forward(weight_dict, x, t, y)
                      → 内部用 at::conv2d + at::linear + at::scaled_dot_product_attention
```

**规则**: 所有生成式模型 (UNet/DiT/MMDiT) 必须在 C++ 里用 libTorch 搭 forward。

---

## 3. 已知运行时问题 (待修复)

### prelude.scm GC guardian
- `make_int_array` / `make_float_array` / `make_ptr_array` 注册到 guardian
- `collect-rendezvous` 在 GC 后自动 drain
- 如果 Chez 版本不支持 `collect-rendezvous`，需手动调用 `gc-drain`

### prelude.scm 文件 I/O
- `file-read-all` 用 `pointer->string` + `display` 到 string port
- 依赖 `/proc/self/exe` 在 launcher 中确定运行目录

---

## 4. C++ libTorch 实现规范

### 权重加载模式
```cpp
// 用 unordered_map 按名查找
auto& d = *static_cast<std::unordered_map<std::string, at::Tensor>*>(sd_dict_ptr);

// 推荐: 使用辅助函数 (sdxl_get_weight/sdxl_linear/sdxl_conv2d)
auto h = sdxl_linear(x, d, "transformer_blocks.0.attn.qkv");

// 如果权重可能缺失，用 defined() 检查
auto w = sdxl_get_weight(d, "optional_weight");
if (w.defined()) { ... }
```

### 函数签名规范
```
void* func_name(
    void* sd_dict,     // safetensors weight dict
    void* tensor1,      // 输入张量
    void* tensor2,      ...
    int param1,         // 标量参数
    double param2);
```

### 异常处理
```cpp
try {
    ... forward logic ...
    return new at::Tensor(result);
} catch (const std::exception& e) {
    std::cerr << "func_name error: " << e.what() << std::endl;
    return nullptr;
}
```

### 复用模式
- FLUX 的 `flux_block()` 是 static 函数 → **所有 FLUX 变体直接复用**
- Hunyuan Video = FLUX + 3D patch embed + 额外 conditioning
- Wan Video = FLUX + RoPE
- Cosmos = FLUX-based
- SD3 MMDiT = Joint attention (img+txt 拼接注意力) — 不共享 FLUX block

---

## 5. StaticPy 编程规范

### 函数级编译
- 所有函数翻译后加 `static_` 前缀
- `global _var` → Scheme 的 `(define _var #f)` + `(set! _var ...)`
- `from ops import *` → translate.py 忽略，仅用于 IDE 类型检查

### 模块模式
```python
# Flat functions + 全局 ptr 变量 (单例兼容)
_my_state: ptr

def my_init(path: str) -> void:
    global _my_state
    _my_state = torch_std_safetensors_load(path)

def my_forward(x: ptr, t: ptr) -> ptr:
    global _my_state
    return torch_std_my_forward(_my_state, x, t)
```

### 与 C++ 交互
- 张量传 `ptr` (void*)
- extern fn 返回值也是 `ptr`
- 标量用 `int`/`float`/`str`
- C++ 返回 `new at::Tensor(...)` 给 StaticPy

---

## 6. 开发检查清单

### 提交前
- [ ] `cat *.static.py | python3 compiler/translate.py` 无错误
- [ ] `grep static_torch_std_` 结果只有 silu/gelu
- [ ] `grep '(math '` 结果为空
- [ ] `grep '(or \|(and '` 存在 (BoolOp 正确翻译)
- [ ] 新 extern fn 在 ops.static.py 有声明
- [ ] build_all.sh 包含新模块
- [ ] task.md 状态更新

### 模型实现检查
- [ ] Python 源码在 comfyui_ref 找到并阅读
- [ ] C++ forward 用 libTorch (at::Tensor), 非 JIT
- [ ] 权重按名查找 (sdxl_get_weight)
- [ ] 异常处理 (try/catch)
- [ ] StaticPy 模块用 dict-based 签名
- [ ] ops.static.py extern fn 签名与 C++ 一致

---

## 7. 文件职责

| 文件 | 职责 | 语言 |
|------|------|------|
| `cpp_runtime/libtorch_std_helper.h` | extern "C" API 声明 | C |
| `cpp_runtime/libtorch_std_helper.cpp` | 所有模型 forward 实现 | C++ (libTorch) |
| `ops.static.py` | 所有 extern fn 声明 | StaticPy |
| `sd_runtime/*.static.py` | 管线编排 + 模型封装 | StaticPy |
| `compiler/translate.py` | StaticPy → Scheme 翻译 | Python |
| `compiler/prelude.scm` | 运行时 (C 数组/GC/文件I/O) | Scheme |
| `build.sh` | 编译管线 | Bash |
| `build_all.sh` | 一键编译全部模块 | Bash |
| `deliver.sh` | ELF 部署包打包 | Bash |

---

## 8. 快速参考

### 查看 extern fn 声明 (ops.static.py)
```bash
grep "^extern fn" sd_runtime/ops.static.py | wc -l  # 应 = C++ 头文件函数数
```

### 检查翻译正确性
```bash
cat sd_runtime/*.static.py | python3 compiler/translate.py 2>&1 > /tmp/out.scm
grep -c 'define (static_' /tmp/out.scm    # 函数数
grep -c 'define torch_std_' /tmp/out.scm  # extern FFI 数
grep '^;;  ! ' /tmp/out.scm | wc -l        # 警告数
```

### 验证无 JIT
```bash
grep 'torch_std_jit_forward' /tmp/out.scm | grep -v "foreign-procedure\|t5\|vae\|clip\|_cv_\|_adapter_"
# 应无输出 (所有模型 forward 是纯 libTorch)
```

---

> 一句话：**所有模型 forward 用 C++ libTorch，StaticPy 只做管线编排，translate.py 只做语言翻译。**
