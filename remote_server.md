# 远程 GPU 部署：Xiangongyun 一键推理

## 概述

ComfyCLI 编译为独立 ELF 二进制，零 Python 依赖。远程部署有三条路线：

| 路线 | 二进制 | 适用场景 | 章节 |
|------|--------|---------|------|
| **A：workflow 路线** | `comfycli-bin`（StaticPy/Chez 编排层） | 执行 ComfyUI workflow JSON / prompt 模式 | 全文 + `comfycli_remote.sh` 一键 |
| **B：出图管线路线** | `img_hires`（纯 C++） | 只跑 `backup.sh` / `backup_qwen.sh` 风格出图 | [方案 B：出图管线部署](#方案-b出图管线部署img_hires--backupsh--backup_qwensh) |
| **C：图片编辑路线** | `sd-cli`（sd.cpp 官方 CLI） | Qwen-Image-2.1 指令图片编辑（换背景/换装/缩放保真） | [方案 C：图片编辑部署](#方案-c图片编辑部署sd-cli--editsh) |

路线 A 流程（`comfycli_remote.sh` 一键完成）：
1. 本地编译出 `comfycli-bin` + `libsdcpp_adapter.so`
2. 打包所需依赖（sd.cpp / ggml / CUDA 后端）
3. SCP 到远程 GPU 实例
4. 远程执行推理

---

## 前置配置（.env）

```bash
XGC_API_TOKEN=your_api_token
XGC_IMAGE_ID=25802c4f-7939-4ff6-80c9-f385e608c80a
XGC_PASSWORD=your_password
```

---

## 核心脚本

| 脚本 | 职责 |
|------|------|
| `xgc_ctl.py` | Xiangongyun 实例生命周期管理（部署/等待/SSH/销毁） |
| `comfycli_remote.sh` | 端到端自动化：编译 → 打包 → 部署 → 同步模型 → 上传 → 运行 → 下载 |
| `deploy.sh` | 打包部署包（含 sd.cpp/ggml/CUDA 后端等运行时 .so） |
| `cpp/sd/edit.sh` | Qwen-Image-2.1 指令图片编辑（sd-cli `-r` 参考图 + FA/EasyCache 加速，见方案 C） |

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

## 方案 B：出图管线部署（img_hires + backup.sh / backup_qwen.sh）

只部署 C++ 出图二进制与两个风格出图脚本，不涉及 `comfycli-bin`。实测 RTX 4090 裸机
（glibc 2.39，已有 CUDA Runtime）：768×768 冒烟 `backup_qwen.sh` 21s、`backup.sh` 15s。

### B1. 本地编译

```bash
# ① sd.cpp 动态后端（产物 /opt/sd/build-dl/bin/*.so；已有则跳过）
bash cpp/sd/build_sd_dl.sh

# ② 出图二进制 + 适配层（完整重建 cpp/sd/build/）
bash cpp/sd/scripts/build.sh
```

路线 B 需要的产物（`libsdcpp_adapter.so` 是路线 A 的 FFI 后端，**不需要**）：

| 文件 | 本地路径 | 大小 |
|------|---------|------|
| `img_hires` | `cpp/sd/build/` | 275K |
| `libstable-diffusion.so` | `/opt/sd/build-dl/bin/` | 37M |
| `libggml.so.0` | `/opt/sd/build-dl/bin/` | 54K |
| `libggml-base.so.0` | `/opt/sd/build-dl/bin/` | 894K |
| `libggml-cuda.so` | `/opt/sd/build-dl/bin/` | 129M |
| **核心 5 件合计** | | **167M** |

### B2. 系统依赖闭包收集

远程非同一镜像时，还需 `.so` 系统闭包（opencv/gdal/tbb 等，本机 ldd 解析出 145 条路径，
排除 glibc 核心与驱动后 134 个实体，**231MB**）：

```bash
STAGE=/tmp/syslibs; rm -rf $STAGE; mkdir -p $STAGE
# 远程已有 CUDA Runtime → EXCLUDE_CUDART=1（省 573MB）；无则置 0
EXCLUDE_CUDART="${EXCLUDE_CUDART:-1}"

# 三库 ldd 并集（img_hires 为入口；ggml-cuda 运行时 dlopen，必须单独查）
ldd cpp/sd/build/img_hires \
    /opt/sd/build-dl/bin/libstable-diffusion.so \
    /opt/sd/build-dl/bin/libggml-cuda.so \
  | grep -oP '=> \K[^ ]+' | grep '^/' | sort -u > /tmp/union.txt

# 排除：glibc 核心（远程同版本自带，不能替换）+ libcuda.so.1（NVIDIA 驱动提供）
#       + 可选 CUDA Runtime（远程 ldconfig 已有时）
grep -vE '/(libc\.so\.6|libm\.so\.6|libpthread\.so\.0|libdl\.so\.2|librt\.so\.1|ld-linux-x86-64\.so\.2|libresolv\.so\.2|libgcc_s\.so\.1|libcuda\.so\.1)$' /tmp/union.txt \
| { [ "$EXCLUDE_CUDART" = 1 ] && grep -vE 'libcudart\.so\.12|libcublas\.so\.12|libcublasLt\.so\.12' || cat; } \
| while read -r f; do cp -n "$f" "$STAGE/"; done

# SONAME 兜底符号链接（cp 按 ldd 路径名落盘，一般 0 个需补）
cd $STAGE && for f in *; do
  s=$(readelf -d "$f" 2>/dev/null | grep -oP 'SONAME.*\[\K[^\]]+' | head -1)
  [ -n "$s" ] && [ "$s" != "$f" ] && [ ! -e "$s" ] && ln -s "$f" "$s"
done
```

两种 CUDA 模式实测体积（不含核心 5 件）：

| 模式 | 条件 | 系统闭包 | 总计（+核心 167M） |
|------|------|---------|------------------|
| A：远程已有 CUDA Runtime | `ldconfig -p \| grep cudart` 命中 | 134 个 / 231M | ~398M |
| B：远程无 CUDA Runtime | 同上未命中 | 137 个 / 804M（含 cublasLt 468M + cublas 103M） | ~971M |

> 仙宫云镜像自带 `/usr/local/cuda`（`ldconfig` 可见 libcudart/libcublas），走模式 A。
> 同 glibc（本地 2.39 = 远程 2.39）才能直接拷系统 so；glibc 不同见 [deploy.md](./deploy.md) GLIBC 兼容章节。

### B3. 远程目录布局

```
/root/
├── backup.sh               # 写实管线（z_image，默认 E1xMIN）
├── backup_qwen.sh          # 风格管线（Qwen-Image-2.1，预设库 22 组）
└── build/                  # 二进制 + 全部依赖（实测 246 项 / 376M）
    ├── img_hires
    ├── libstable-diffusion.so
    ├── libggml.so.0
    ├── libggml-base.so.0
    ├── libggml-cuda.so      # GGML_BACKEND_PATH 运行时加载
    └── <系统依赖闭包 ~140 个 .so>

/data/models/image/         # 模型 19G（见 B6）
```

脚本默认值（`SD_CLI=$SCRIPT_DIR/build/img_hires`、`GGML_BACKEND_PATH=$SD_BACKEND_DIR/libggml-cuda.so`）与该布局对齐；
若放别处，用 `SD_CLI=... SD_BACKEND_DIR=...` 覆盖。

### B4. scp 清单

```bash
H=<domain>; P=<port>   # python3 xgc_ctl.py info <id>

# 1) 两个出图脚本 → ~/
scp -P $P cpp/sd/backup.sh cpp/sd/backup_qwen.sh root@$H:/

# 2) 核心 5 件 → ~/build/
ssh -p $P root@$H 'mkdir -p ~/build'
scp -P $P cpp/sd/build/img_hires \
          /opt/sd/build-dl/bin/libstable-diffusion.so \
          /opt/sd/build-dl/bin/libggml.so.0 \
          /opt/sd/build-dl/bin/libggml-base.so.0 \
          /opt/sd/build-dl/bin/libggml-cuda.so \
          root@$H:~/build/

# 3) 系统依赖闭包 → ~/build/（tar 保留符号链接，比逐个 scp 快）
tar czf - -C $STAGE . | ssh -p $P root@$H 'tar xzf - -C ~/build'

# 4) 模型 → /data/models/image/（19G，单次 ~40 分钟；详见 B6）

# 5) 脚本默认后端目录改为 ~/build（一次性）
ssh -p $P root@$H 'sed -i "s|SD_BACKEND_DIR=\"\${SD_BACKEND_DIR:-/opt/sd/build-dl/bin}\"|SD_BACKEND_DIR=\"\${SD_BACKEND_DIR:-\$SCRIPT_DIR/build}\"|" ~/backup.sh ~/backup_qwen.sh'
```

### B5. 远程验证与运行

```bash
ssh -p $P root@$H
# ldd 三连必须 0 缺失
LD_LIBRARY_PATH=~/build ldd ~/build/img_hires            | grep 'not found' || echo OK
LD_LIBRARY_PATH=~/build ldd ~/build/libstable-diffusion.so | grep 'not found' || echo OK
LD_LIBRARY_PATH=~/build ldd ~/build/libggml-cuda.so       | grep 'not found' || echo OK

# 冒烟（768×768，应秒级出图）
PRESET=penguin bash ~/backup_qwen.sh ~/smoke.png 768 768
bash ~/backup.sh "a red apple" ~/smoke_z.png 768 768

# 正式出图
PRESET=lighthouse bash ~/backup_qwen.sh ~/out.png 2560 1440   # ~4.5min @4090
bash ~/backup.sh "portrait, soft light" ~/out_z.png 2560 1440
```

### B6. 模型文件（→ `/data/models/image/`，共 19G）

| 文件 | 大小 | 用于 |
|------|------|------|
| `z_image_turbo-Q5_K_M.gguf` | 5.2G | backup.sh 扩散 |
| `qwen-image-2.1-Q6_K.gguf` | 5.8G | backup_qwen.sh 扩散 |
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 4.7G | backup_qwen.sh LLM |
| `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.3G | backup.sh LLM |
| `qwen_image_2.1_vae_bf16.safetensors` | 644M | Qwen VAE |
| `ae.safetensors` | 335M | z_image VAE |
| `2x_ESRGAN.gguf` | 63M | 放大 |

```bash
scp -P $P /data/models/image/{z_image_turbo-Q5_K_M.gguf,qwen-image-2.1-Q6_K.gguf,Qwen3VL-8B-Instruct-Q4_K_M.gguf,Qwen3-4B-Instruct-2507-Q4_K_M.gguf,qwen_image_2.1_vae_bf16.safetensors,ae.safetensors,2x_ESRGAN.gguf} root@$H:/data/models/image/
# 校验
md5sum ... | ssh -p $P root@$H 'cd /data/models/image && md5sum -c -'
```

### B7. 裸机实例清理（可选，实测释放 12G）

镜像遗留的 RL 环境可删（删前确认 `ldconfig -p | grep -E "cu126|/opt/nvidia"` 与 profile 均无引用）：

```bash
# /opt 11G → 59K：CUDA 安装缓存 + 旧项目
rm -rf /opt/{nvidia,cu126,corpus,ReScheme,cs_core.tar.gz,ChezScheme,coderl,code_caches,luajit2,redis,stock,redis_src.tar.gz,obj,third_party,models}
rm -f  /opt/run_*.sh /opt/start_linux_vec.sh /opt/gen_refs_remote.py /opt/fix_algo_remote.py /opt/extract_parallel.sh /opt/batch_extract.sh
# /root 1.8G → 118M：Python 环境与缓存
rm -rf /root/{miniconda3,.cache,.npm,.triton,.launchpadlib}
# 保留：/usr/local/cuda（CUDA Runtime）、NVIDIA 驱动、/data/models、~/build、~/backup*.sh
```

---

## 方案 C：图片编辑部署（sd-cli + edit.sh）

基于 sd.cpp 官方 `sd-cli` 的 `-r` 参考图路线，跑 Qwen-Image-2.1 指令图片编辑，零 pip。
编辑脚本 `cpp/sd/edit.sh` 部署到远程 `~/edit.sh`，与路线 B 的 `backup*.sh` 同级共存。

### C1. 用法

```bash
bash edit.sh <输入图> <编辑指令> <输出图> [宽] [高]

# 换背景（身份保持 + 光照自动协调）
bash edit.sh in.jpg "Change the background to a sunset beach" out.png

# 省略宽高 → 按输入图比例自动: 最长边 <= MAX_SIDE(默认1024), 宽高 32 整除（缩小保真）
bash edit.sh in.jpg "Keep the image exactly the same" small.png

# 指定分辨率（需 32 整除）
bash edit.sh in.jpg "Change the dress to a red silk gown" out.png 768 1024
```

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `FA` / `CACHE` | `1` / `1` | 加速开关：FlashAttention（`--diffusion-fa`）/ EasyCache 步缓存；`FA=0`、`CACHE=0` 关闭 |
| `SAGE` | `0` | `SAGE=1` 额外启用 `--sage-attn` |
| `STEPS` / `CFG` / `METHOD` | `20` / `6.0` / `euler` | 采样参数 |
| `MAX_SIDE` | `1024` | 自动尺寸时的最长边上限（越小越快） |
| `SEED` | 随机 | 固定种子 |
| `SD_CLI` / `MODEL_DIR` | 自动查找 | sd-cli 路径 / 模型目录（见 C3 查找顺序） |

编辑是**整图参考式重绘**（非 mask 局部编辑）：换背景/换装/缩放保真效果好，
构图会轻微收紧、项链等配饰级细节会重绘，不保证像素级还原。

### C2. 本地编译 sd-cli

```bash
cmake --build /opt/sd/build-dl --target sd-cli -j$(nproc)
# 产物: /opt/sd/build-dl/bin/sd-cli（依赖同目录 libstable-diffusion.so / libggml*.so / libwebp*）
```

### C3. scp 清单

```bash
H=<domain>; P=<port>   # python3 xgc_ctl.py info <id>
STAGE=/tmp/sdcli_stage; rm -rf $STAGE; mkdir -p $STAGE/bin

# 1) sd-cli 核心 12 件 → ~/sdcli/（或并入路线 B 的 ~/build/，edit.sh 两处都找）
scp -P $P cpp/sd/edit.sh root@$H:/
ssh -p $P root@$H 'mkdir -p ~/sdcli'
B=/opt/sd/build-dl/bin
scp -P $P $B/sd-cli $B/libstable-diffusion.so \
          $B/libggml.so.0* $B/libggml-base.so.0* $B/libggml-cpu.so $B/libggml-cuda.so \
          $B/libwebp.so* $B/libwebpmux.so* $B/libwebm.so $B/libsharpyuv.so* \
          root@$H:~/sdcli/

# 2) 系统依赖闭包（B2 配方通用，入口换 sd-cli）
ldd $B/sd-cli $B/libstable-diffusion.so $B/libggml-cuda.so \
  | grep -oP '=> \K[^ ]+' | grep '^/' | sort -u > /tmp/union.txt
grep -vE '/(libc\.so\.6|libm\.so\.6|libpthread\.so\.0|libdl\.so\.2|librt\.so\.1|ld-linux-x86-64\.so\.2|libresolv\.so\.2|libgcc_s\.so\.1|libcuda\.so\.1)$' /tmp/union.txt \
| grep -vE 'libcudart\.so\.12|libcublas\.so\.12|libcublasLt\.so\.12|/opt/sd/build-dl' \
| while read -r f; do cp -rn "$f" "$STAGE/bin/"; done
tar czf - -C $STAGE . | ssh -p $P root@$H 'tar xzf - -C ~/sdcli'

# 3) mmproj 视觉权重 → /data/models/image/（1.1G，**编辑必需**；T2I 不需要）
#    扩散/VAE/LLM 三个模型与路线 B 相同，镜像已有
curl -L -o /data/models/image/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
  https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf
scp -P $P /data/models/image/mmproj-Qwen3VL-8B-Instruct-F16.gguf root@$H:/data/models/image/
```

模型四件（`MODEL_DIR=/data/models/image`）：

| 文件 | 大小 | 用于 |
|------|------|------|
| `qwen-image-2.1-Q6_K.gguf` | 5.8G | 扩散（同 B） |
| `qwen_image_2.1_vae_bf16.safetensors` | 644M | VAE（同 B） |
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 4.7G | LLM 文本编码（同 B） |
| `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1.1G | **视觉编码（编辑必需，需下载）** |

### C4. 远程运行与实测

```bash
ssh -p $P root@$H
# edit.sh 的 sd-cli 查找顺序: $SCRIPT_DIR/sd-cli → ~/sdcli/sd-cli → ~/build/sd-cli → $SCRIPT_DIR/build/sd-cli
bash ~/edit.sh /root/test.jpg "Change the background to a sunset beach" /root/out.png
```

4090 实测（编辑 2885×4325 输入）：

| 配置 | 采样 | 端到端 |
|------|------|--------|
| 调优前：30 步、无加速、1024×1536 | 316s（另 VAE OOM 触发 tiling 重试） | 330s |
| 调优后：20 步、FA + EasyCache、672×1024 | 17.8s（EasyCache 跳 11/20 步） | **24-27s** |

约 12× 提速；EasyCache 与 30 步无缓存画质无肉眼差异。质量表现见 C1 末段。

---

## CUDA Runtime 说明

| 模式 | 包大小 | 远程环境要求 | 命令 |
|------|--------|-------------|------|
| GPU（不含 CUDA Runtime） | ~57MB | 需已安装 CUDA 12.x | `./deploy.sh` |
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
