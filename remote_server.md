# 远程 GPU 部署：Xiangongyun 一键推理

## 概述

ComfyCLI 编译为独立 ELF 二进制，零 Python 依赖。远程部署只需：
1. 本地编译出 `comfycli-bin` + `libsdcpp_adapter.so`
2. 打包所需依赖（sd.cpp / ggml / CUDA 后端）
3. SCP 到远程 GPU 实例
4. 远程执行推理

整个过程由 `comfycli_remote.sh` 一键完成。

---

## 前置配置（.env）

```bash
XGC_API_TOKEN=your_api_token
XGC_IMAGE_ID=c5b511af-75b7-4f51-8115-f6157f3cade8
XGC_PASSWORD=your_password
```

---

## 核心脚本

| 脚本 | 职责 |
|------|------|
| `xgc_ctl.py` | Xiangongyun 实例生命周期管理（部署/等待/SSH/销毁） |
| `comfycli_remote.sh` | 端到端自动化：编译 → 打包 → 部署 → 同步模型 → 上传 → 运行 → 下载 |
| `deploy.sh` | 打包部署包（含 sd.cpp/ggml/CUDA 后端等运行时 .so） |

---

## 典型工作流

### 1. 工作流模式（推荐）

```bash
bash comfycli_remote.sh \
  --workflow test_remote_2560.json \
  --output-dir ./output \
  --name comfy-4090 \
  --with-cuda \
  --run
```

流程：
1. 本地编译 `comfycli-bin` + `libsdcpp_adapter.so`
2. 打包依赖（sd.cpp/ggml/可选 CUDA Runtime）
3. 调用 `xgc_ctl.py deploy` 创建远程 GPU 实例
4. 等待实例进入 `running` 状态
5. 解析 workflow JSON，提取模型文件名
6. 打印模型 SCP 命令（`--sync-models` 自动同步）
7. 上传部署包 + workflow JSON + 模型
8. SSH 执行推理
9. 下载输出图片到本地 `./output/`

### 2. prompt-only 模式

```bash
bash comfycli_remote.sh \
  --checkpoint /data/models/image/z_image_turbo-Q5_K_M.gguf \
  --prompt "a portrait of a young woman" \
  --output ./result.png \
  --width 2560 --height 1440 \
  --steps 20 --cfg 2.5 \
  --name comfy-4090
```

### 3. 已有实例模式

```bash
INSTANCE_ID=xxxxxx
bash comfycli_remote.sh \
  --instance-id $INSTANCE_ID \
  --workflow test_remote_2560.json \
  --output-dir ./output \
  --run
```

---

## 手动部署流程

### 1. 本地编译并打包

```bash
./build.sh
./deploy.sh               # 含 CUDA 后端插件，远程需有 CUDA Runtime
WITH_CUDA=1 ./deploy.sh   # 含 CUDA Runtime，远程无需额外安装
```

产物在 `dist/` 目录：
```
comfycli-bin
comfycli-bin.so
libsdcpp_adapter.so
libggml.so
libggml-base.so
libggml-cuda.so        # GPU 后端插件
libcudart.so.12        # WITH_CUDA=1 时包含
libcublas.so.12        # 同上
libcublasLt.so.12      # 同上
run.sh
```

### 2. 部署实例

```bash
python3 xgc_ctl.py deploy --name comfy-4090
```

### 3. 等待启动

```bash
python3 xgc_ctl.py wait <instance_id> --status running
```

### 4. 上传文件

```bash
# 上传部署包
scp -P <port> dist/package.tar.gz root@<domain>:/opt/

# 上传模型（需先有本地模型文件）
scp -P <port> /data/models/image/*.gguf root@<domain>:/data/models/image/
```

### 5. 远程运行

```bash
ssh -p <port> root@<domain>
cd /opt/comfycli
LD_LIBRARY_PATH=. GGML_BACKEND_PATH=./libggml-cuda.so \
  ./comfycli-bin test_workflow.json --output-dir ./output
```

### 6. 销毁止损

```bash
python3 xgc_ctl.py shutdown_destroy <instance_id>
```

---

## CUDA Runtime 说明

| 模式 | 包大小 | 远程环境要求 | 命令 |
|------|--------|-------------|------|
| GPU（不含 CUDA Runtime / ONNX CPU） | ~79MB | 需已安装 CUDA 12.x | `./deploy.sh` |
| GPU（含 ONNX Runtime CUDA provider） | ~742MB | 需已安装 CUDA 12.x | `WITH_ONNX_CUDA=1 ./deploy.sh` |
| GPU（含 CUDA Runtime） | ~439MB | 仅需 NVIDIA 驱动 | `WITH_CUDA=1 ./deploy.sh` |
| CPU-only | ~35MB | 无需 GPU | `WITH_CUDA_BACKEND=0 ./deploy.sh` |

- `comfycli_remote.sh` 默认使用 `WITH_CUDA=1` 模式（兼容性最好）
- 如远程镜像已预装 CUDA Runtime，可用 `--no-with-cuda` 跳过打包

---

## 模型文件

所有模型位于远程的 `/data/models/image/`：
- `z_image_turbo-Q5_K_M.gguf` — 扩散模型
- `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` — LLM
- `ae.safetensors` — VAE

`comfycli_remote.sh` 会自动从 workflow JSON 解析模型文件名，打印 SCP 命令。
加 `--sync-models` 参数可自动执行模型同步。

---

## Xiangongyun API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/open/instance/deploy` | 部署新实例 |
| `GET`  | `/open/instances` | 查询实例列表 |
| `POST` | `/open/instance/shutdown_destroy` | 关机并销毁（停止计费） |

`xgc_ctl.py` 子命令：

| 命令 | 作用 |
|------|------|
| `list` | 列出所有实例 |
| `deploy` | 部署新实例（默认 RTX 4090 x1） |
| `status <id>` | 查看实例状态 |
| `info <id>` | 查看 SSH 地址/端口/密码 |
| `wait <id>` | 轮询等待实例就绪 |
| `ssh <id>` | 打印 SSH 命令 |
| `shutdown_destroy <id>` | 关机并销毁 |

---

## 注意事项

1. 部署后需等待实例进入 `running` 状态才能获取 SSH 信息
2. 镜像默认预装 CUDA Runtime 和 NVIDIA 驱动
3. 推理完成后及时执行 `shutdown_destroy` 停止计费
4. 脚本默认使用浙江一区 RTX 4090 x1，可在 `deploy` 命令中覆盖
