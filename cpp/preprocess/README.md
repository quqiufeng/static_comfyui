# cpp/preprocess — ControlNet 条件图预处理器

把普通图片转成 Z-Image Fun-ControlNet 需要的**控制图**（canny / depth / pose / mlsd / lineart / gray / hed）。
纯 C++：OpenCV（零模型）+ ONNX Runtime（depth / pose）。

## 构建

```bash
./build.sh          # -> preprocesscli
# ORT_DIR 默认 /data/venv/onnxruntime-linux-x64-gpu-1.26.0
```

## 用法

```bash
preprocesscli <mode> <input> <output> [options]
# modes: canny | lineart | gray | hed | mlsd | depth | pose
```

| mode | 依赖 | 说明 |
|------|------|------|
| `canny` | OpenCV | 边缘检测（`--canny-low/--canny-high`）|
| `lineart` | OpenCV | 近似线稿（黑线白底）|
| `gray` | OpenCV | 灰度 |
| `hed` | HED ONNX（无则回退 OpenCV 近似）| 精确软边缘（lllyasviel ControlNetHED）|
| `mlsd` | M-LSD ONNX（无则回退 OpenCV LSD）| 精确直线段（lllyasviel M-LSD large）|
| `depth` | Depth-Anything-V2-Small ONNX | 深度图（白=近）|
| `pose` | DWPose (YOLOX + DW) ONNX | OpenPose 18 骨架图 |

`hed_approx` / `mlsd_approx` 可强制使用 OpenCV 近似版。

通用选项：`--target <int>`（长边缩放，0=原图）、`--invert`。

示例：

```bash
# 边缘 → 出图（单阶段）
./preprocesscli canny photo.png /tmp/canny.png --target 1024
CONTROL_NET=/data/models/image/z_image_turbo_fun_controlnet_union_2.1_lite_2601_8steps_sdcpp.safetensors \
CONTROL_IMAGE=/tmp/canny.png CONTROL_STRENGTH=0.75 NO_QUALITY_PREFIX=1 HIRES=0 STEPS=8 CFG=1.0 \
  ../sd/backup.sh "a cat" /tmp/out.png 1024 1024

# 深度
./preprocesscli depth photo.png /tmp/depth.png --target 1024

# 姿态
./preprocesscli pose photo.png /tmp/pose.png
```

## 模型

| 模式 | 默认路径 | 来源 |
|------|----------|------|
| depth | `/data/models/image/depth_anything_v2_small/model.onnx` (+ `.onnx_data`) | `onnx-community/depth-anything-v2-small-ONNX` |
| pose (det) | `/data/models/image/dwpose/yolox_l.onnx` | `yzd-v/DWPose` |
| pose (est) | `/data/models/image/dwpose/dw-ll_ucoco_384.onnx` | `yzd-v/DWPose` |
| hed | `/data/models/image/annotators/hed.onnx` (+ `.data`) | `lllyasviel/Annotators` ControlNetHED（自转 ONNX）|
| mlsd | `/data/models/image/annotators/mlsd.onnx` (+ `.data`) | `lllyasviel/Annotators` MLSd large（自转 ONNX）|

可用 `--depth-model/--pose-det/--pose-est/--mlsd-model` 覆盖；`--cpu` 关闭 CUDA provider。

## 备注

- depth 预处理用 ImageNet 归一化、518×518（14 的倍数）；输出按 min/max 归一化到 0–255。
- pose 用 YOLOX 检测人物（阈值 0.3 + NMS 0.45），DWPose 关键点 `simcc argmax/2`，
  取 COCO-17 身体点插值成 OpenPose-18 并绘制标准配色骨架。
- `mlsd`/`hed` 为 OpenCV 近似（`ForserX` 只有完整 ControlNet、`lllyasviel/Annotators` 只有 `.pth`）；
  如需更精确可后续接入 M-LSD / HED 的 ONNX。
