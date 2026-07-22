# 远程服务器编排：Xiangongyun API 自动化

## 核心脚本

- `xgc_ctl.py`：专门用于操作 Xiangongyun **服务器资源**的 Python 脚本。
  - 只负责实例生命周期：部署、查询状态、等待启动、生成 SSH 命令、关机、销毁、关机并销毁。
  - 自动读取项目根目录 `.env` 中的 `XGC_API_TOKEN` 和 `XGC_IMAGE_ID`。
  - **不负责训练部署本身**：训练环境的准备、代码/数据上传、二进制运行等操作，仍需要通过 SSH 登录后由人工或 AI 逐步执行。
  - 训练结束后通过 `shutdown_destroy` 一键停止按小时计费。

## 前置配置（.env）

在项目根目录创建 `.env`：

```bash
XGC_API_TOKEN=your_api_token
XGC_IMAGE_ID=your_private_image_id
```

脚本会自动读取 `.env` 中的变量，无需手动 `export`。

---

## 目标场景

- 在本地完成代码调试后，需要把训练任务放到远程 GPU 实例上长时间运行。
- 训练结束后立即关机并销毁实例，停止按小时计费。
- 整个流程通过 Python 脚本自动化，避免手动在控制台点击。

---

## 主要 API 端点

| 方法 | 路径 | 说明 | 主要请求参数 |
|------|------|------|--------------|
| `POST` | `/open/instance/deploy` | 部署新实例 | `gpu_model`, `gpu_count`, `data_center_id`, `image`, `image_type`, `name` |
| `GET`  | `/open/instances` | 查询实例列表 | 无 |
| `POST` | `/open/instance/shutdown_release_gpu` | 关机并释放 GPU | `id` |
| `POST` | `/open/instance/destroy` | 销毁实例 | `id` |
| `POST` | `/open/instance/shutdown_destroy` | 关机并销毁（停止计费） | `id` |

返回统一格式：

```json
{
  "code": 200,
  "data": { ... },
  "success": true
}
```

---

## 脚本说明

文件：`xgc_ctl.py`

封装了所有实例生命周期操作，支持以下子命令：

| 命令 | 作用 |
|------|------|
| `list` | 列出所有实例 |
| `deploy` | 部署新实例 |
| `status <id>` | 查看单个实例原始 JSON 状态 |
| `info <id>` | 查看实例关键访问信息（ID、状态、SSH 地址、端口、用户名、密码、Jupyter URL 等） |
| `wait <id>` | 轮询等待实例到达指定状态 |
| `ssh <id>` | 打印可直接执行的 SSH 命令（含密码） |
| `shutdown <id>` | 关机释放 GPU |
| `destroy <id>` | 销毁实例 |
| `shutdown_destroy <id>` | 关机并销毁（训练结束后使用） |

## 镜像说明

私有镜像 ID：`59b0e19d-ca61-4668-a9f2-6c23df208d52`

该镜像已预装训练依赖环境，部署实例后无需额外安装 Python、pip、torch、CUDA toolkit 或同步大型 `.so` 包。实例启动后：

- `/opt/lib/` — 已包含 torch 运行时 `.so`
- `/opt/nvidia/` — 已包含 NVIDIA CUDA/cuDNN 等 `.so`
- `/opt/stock/` — 已包含历史训练二进制（建议用本地新编译的版本覆盖）
- `/root/stock_data.tar` — 已包含 10 年 A 股日线训练数据

因此，从该镜像部署实例后，**只需在本地重新编译 stock 项目，再把二进制文件 scp 到 `/opt/stock/`，然后解压数据即可运行训练**。

---

## 典型工作流

### 1. 部署实例

```bash
python3 xgc_ctl.py deploy --name stock-rl-4090
```

输出会包含实例 ID，例如：

```json
{
  "code": 200,
  "data": { "id": "cfwbjwa0soug79w2" },
  "success": true
}
```

### 2. 等待实例启动完成

```bash
python3 xgc_ctl.py wait cfwbjwa0soug79w2 --status running
```

### 3. 获取 SSH 命令并登录

```bash
python3 xgc_ctl.py ssh cfwbjwa0soug79w2
```

复制输出即可登录：

```bash
sshpass -p '...' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 44775 root@cfwbjwa0soug79w2.ssh.x-gpu.com
```

### 4. 本地编译并同步二进制到远程

由于镜像 `59b0e19d-ca61-4668-a9f2-6c23df208d52` 已预装 torch/CUDA 依赖，只需重新编译并覆盖 `/opt/stock/` 下的二进制：

```bash
# 本地编译（在 /opt/ReScheme 目录下）
bash stock/build.sh

# scp 二进制和 .so 到远程
scp -P 33251 /opt/ReScheme/stock_rl_train \
             /opt/ReScheme/stock_rl_train.so \
             /opt/ReScheme/stock_rl_backtest \
             /opt/ReScheme/stock_rl_backtest.so \
             /opt/ReScheme/stock/libstock_rl_helper.so \
             /opt/ReScheme/libtorch_std_helper.so \
             root@<ssh_domain>:/opt/stock/
```

### 5. 在远程实例上准备训练环境（SSH 登录后由人工/AI 执行）

`xgc_ctl.py` 只生成 SSH 命令，不自动完成后续训练部署。登录远程后通常执行：

```bash
# 把数据解压到 /dev/shm 以加速读取
rm -rf /dev/shm/stock_data_daily_10y
mkdir -p /dev/shm/stock_data_daily_10y
tar -xf ~/stock_data.tar -C /tmp/
cp -r /tmp/stock_data_daily_10y/* /dev/shm/stock_data_daily_10y/

# 确认 GPU 环境
nvidia-smi
```

### 6. 启动训练

```bash
export LD_LIBRARY_PATH=/opt/lib:/opt/nvidia/cu13/lib:/opt/nvidia/cublas/lib:/opt/nvidia/cuda_runtime/lib:/opt/nvidia/cudnn/lib:/opt/nvidia/cufft/lib:/opt/nvidia/cusparse/lib:/opt/nvidia/nccl/lib:/opt/nvidia/nvjitlink/lib:/opt/stock:$LD_LIBRARY_PATH
cd /opt/stock
mkdir -p /opt/ReScheme/stock/weights

nohup ./stock_rl_train 5000 240 1 /dev/shm/stock_data_daily_10y 3e-5 3 120 0.96 0.95 0.1 0.01 0.5 \
  > /tmp/train_daily_500.log 2>&1 < /dev/null &
```

### 7. 训练结束后销毁止损

```bash
python3 xgc_ctl.py shutdown_destroy <id>
```

确认销毁：

```bash
python3 xgc_ctl.py list
```

`list` 返回空列表即表示销毁完成。

---

## 注意事项

1. `deploy` 异步执行，创建后需要 `wait` 轮询等待 `status` 变为 `running`。
2. 实例 `running` 后才会显示 `ssh_domain`、`ssh_port`、`password`。
3. `shutdown_destroy` 会同时完成关机和销毁，停止按小时计费，适合训练结束后一次性清理。
4. 镜像 `image_type` 通常为 `private`（使用自定义镜像）或 `public`（使用官方镜像）。
5. 当前脚本默认使用 RTX 4090 + 1 卡 + 浙江一区（`data_center_id=1`），可在 `deploy` 命令中覆盖。

---

## 参考来源

- Xiangongyun API Playground：https://api-playground.xiangongyun.com/instance/0
- Xiangongyun API Base：https://api.xiangongyun.com
- 本文档中的 API 端点定义来自上述官方 API 文档。

---

## 扩展建议

- 增加 `cost` 子命令，根据 `start_timestamp` 和 `price_per_hour` 计算已运行成本。
- 增加 `sync` 子命令，自动把本地代码、二进制、数据压缩包上传到远程实例。
- 增加 `run` 子命令，SSH 登录后自动执行训练脚本。
---

## ComfyUI 适配（static_comfyui）

本项目已完全脱离 torch，产物是独立 ELF 二进制 `comfycli-bin` + `libsdcpp_adapter.so`。
`libsdcpp_adapter.so` 本身很小；sd.cpp / ggml 以及 CUDA 后端（`libggml-cuda.so`）作为独立共享库在运行时被加载。

### 新增脚本

- `comfycli_remote.sh`：端到端脚本，自动完成**编译 → 打包 → 部署实例 → 同步模型 → 上传二进制 → 远程运行 → 下载输出**。
- `xgc_ctl.py` 与 `.env` 已复用自 ReScheme，负责 Xiangongyun 实例生命周期。

### 前置准备

在项目根目录已有 `.env`（含 `XGC_API_TOKEN` / `XGC_IMAGE_ID`）。

### 典型工作流

#### 1. 工作流模式（推荐）

```bash
# 远程镜像已自带 CUDA Runtime（如 ReScheme 镜像）
bash comfycli_remote.sh \
  --workflow test_workflow.json \
  --output-dir ./output \
  --name comfy-4090 \
  --run

# 远程镜像没有 CUDA Runtime（需要把 CUDA Runtime 一起打包）
bash comfycli_remote.sh \
  --workflow test_workflow.json \
  --output-dir ./output \
  --name comfy-4090 \
  --with-cuda \
  --run
```

流程：
1. 本地执行 `./build.sh && ./deploy.sh` 打包（默认含 CUDA 后端插件，不含 CUDA Runtime）
2. 调用 `xgc_ctl.py deploy` 创建实例
3. 轮询等待 `running`
4. 从 workflow 提取所需模型文件并打印 scp 命令（加 `--sync-models` 会自动执行）
5. scp 上传部署包 + workflow JSON
6. 远程解压并执行 `bash run.sh /opt/comfycli/test_workflow.json --output-dir /opt/comfycli/output`
7. 下载输出图片到本地 `./output/`

#### 2. prompt-only 模式

```bash
bash comfycli_remote.sh \
  --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "a red apple" \
  --output /tmp/apple.png \
  --width 512 --height 512 --steps 5 \
  --name comfy-4090
```

#### 3. 已有实例模式

```bash
INSTANCE_ID=xxxxxx
bash comfycli_remote.sh \
  --instance-id $INSTANCE_ID \
  --workflow test_workflow.json \
  --output-dir ./output \
  --run
```

### 关于 CUDA Runtime

- 默认 `./deploy.sh` 只包含 `libggml-cuda.so` 后端插件，**不包含** CUDA Runtime。
  适合远程镜像已安装 CUDA Runtime 的场景（如 ReScheme 镜像）。
- `WITH_CUDA=1 ./deploy.sh` 或 `comfycli_remote.sh --with-cuda` 会把 `libcudart.so.12` / `libcublas.so.12` / `libcublasLt.so.12` 一起打包，适合远程只有 NVIDIA 驱动但无 CUDA Runtime 的场景。
- `WITH_CUDA_BACKEND=0 ./deploy.sh` 打包 CPU-only 版本，约 35MB。

### 模型文件说明

脚本会自动从 workflow 中解析以下字段：
`ckpt_name`、`clip_l_name`、`clip_g_name`、`vae_name`、`control_net_name`、`clip_name1`、`clip_name2`。

如果写的是相对路径，会自动补全为 `/data/models/image/`；如果写绝对路径则直接使用。
远程实例上也需要对应目录存在，脚本会把模型 scp 到远程同名路径下。

### 销毁止损

训练/推理结束后：

```bash
python3 xgc_ctl.py shutdown_destroy <instance_id>
```

确认销毁：

```bash
python3 xgc_ctl.py list
```
