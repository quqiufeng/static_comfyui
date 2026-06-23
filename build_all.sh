#!/bin/bash
# build_all.sh — Compile ALL StaticPy modules into .so, then deliver ELF
# Usage: bash build_all.sh
# Pre-requisite: Chez Scheme + libtorch + CUDA
# 
# 部署时 libtorch_std_helper.so 通过 LD_LIBRARY_PATH 查找,
# STATICPY_TORCH_STD_SO 设为纯文件名（不含路径）

set -euo pipefail

# 部署模式下: load-shared-object 使用纯文件名, 运行时通过 LD_LIBRARY_PATH 查找
export STATICPY_TORCH_STD_SO="libtorch_std_helper.so"

FILES=" \
  sd_runtime/ops.static.py \
  sd_runtime/nn.static.py \
  sd_runtime/attention.static.py \
  sd_runtime/flux_attention.static.py \
  sd_runtime/clip_tokenizer.static.py \
  sd_runtime/clip_model.static.py \
  sd_runtime/sd1_clip.static.py \
  sd_runtime/sdxl_clip.static.py \
  sd_runtime/sd_unet.static.py \
  sd_runtime/sd_vae.static.py \
  sd_runtime/sd_samplers.static.py \
  sd_runtime/sd_samplers_extras.static.py \
  sd_runtime/sd_t5.static.py \
  sd_runtime/sd_lora.static.py \
  sd_runtime/sd_flux.static.py \
  sd_runtime/sd_controlnet.static.py \
  sd_runtime/sd_embed.static.py \
  sd_runtime/sd_utils.static.py \
  sd_runtime/sd_clip_vision.static.py \
  sd_runtime/sd_core.static.py \
  sd_runtime/sd_models.static.py \
  sd_runtime/sd_pipeline.static.py \
  sd_runtime/sd_loha_lokr.static.py \
  sd_runtime/sd_t2i_adapter.static.py \
  sd_runtime/sd_flux_controlnet.static.py \
  sd_runtime/sd_long_clip.static.py \
  sd_runtime/sd_t5_config.static.py \
  sd_runtime/sd_sd3.static.py"

./build.sh $FILES && ./deliver.sh
