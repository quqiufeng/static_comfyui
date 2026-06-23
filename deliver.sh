#!/usr/bin/env bash
# deliver.sh — 制作部署包：ELF（嵌入 scheme+boot+runtime.so）+ lib/（PyTorch + torch_std_helper）
#
# 前提: 先运行 build.sh，产生 build_out/runtime.so 和 build_out/libs/libtorch_std_helper.so
#
# 部署包结构:
#   sd_generate.elf       ← 单文件 ELF（嵌入 scheme + boot + runtime.so）
#   lib/
#   ├── libtorch_std_helper.so
#   ├── libtorch.so
#   ├── libc10.so
#   ├── libtorch_cpu.so
#   ├── libtorch_cuda.so    (如有 CUDA)
#   ├── libc10_cuda.so      (如有 CUDA)
#   ├── libcudart.so.12     (CUDA runtime)
#   └── ...                 (其他依赖)
#
# 部署机需求: NVIDIA 驱动 + libc，不需要安装 PyTorch，不需要 Python
#
# 用法:
#   ./deliver.sh                    打包到 build_out/deploy/
#   ./deliver.sh -o sd_generate.elf 指定输出文件名
#
# 环境变量:
#   BUILD_DIR=build_out        构建输出目录
#   OUTPUT=sd_generate.elf     输出 ELF 文件名
#   CHEZ_SCHEME=scheme         Chez Scheme 可执行文件路径

set -euo pipefail

# ====== 配置 ======
OUTPUT="${OUTPUT:-sd_generate.elf}"
BUILD_DIR="${BUILD_DIR:-build_out}"
DEPLOY_DIR="${BUILD_DIR}/deploy"
LIB_DIR="${DEPLOY_DIR}/lib"
LAUNCHER_DIR="${BUILD_DIR}/launcher"
CHEZ="${CHEZ_SCHEME:-scheme}"

rm -rf "${DEPLOY_DIR}" "${LAUNCHER_DIR}"
mkdir -p "${DEPLOY_DIR}" "${LIB_DIR}" "${LAUNCHER_DIR}"

# ====== 查找 Chez Scheme boot files ======
CHEZ_ROOT=$(dirname "$(command -v "${CHEZ}")" 2>/dev/null || echo "/usr/bin")
for try_lib in \
    "${CHEZ_ROOT}/../lib/csv" \
    "${CHEZ_ROOT}/../lib/csv9.5" \
    "${CHEZ_ROOT}/../lib/csv10.0" \
    "${CHEZ_ROOT}/../lib" \
; do
    if [ -f "${try_lib}/petite.boot" ]; then
        CHEZ_LIB="${try_lib}"
        break
    fi
done
if [ -z "${CHEZ_LIB:-}" ]; then
    CHEZ_LIB=$(find / -name "petite.boot" 2>/dev/null | head -1 | xargs dirname)
fi
echo "  Boot dir: ${CHEZ_LIB}"

# ====== 检查构建产物 ======
RUNTIME_SO="${BUILD_DIR}/runtime.so"
TORCH_SO="${BUILD_DIR}/libs/libtorch_std_helper.so"

if [ ! -f "${RUNTIME_SO}" ]; then
    echo "Error: ${RUNTIME_SO} not found. Run build.sh first." >&2
    exit 1
fi
if [ ! -f "${TORCH_SO}" ]; then
    echo "Error: ${TORCH_SO} not found. Run build.sh first." >&2
    exit 1
fi

echo "=== Build artifacts ==="
echo "  Runtime SO: ${RUNTIME_SO} ($(stat -c%s "${RUNTIME_SO}") bytes)"
echo "  Torch SO:   ${TORCH_SO} ($(stat -c%s "${TORCH_SO}") bytes)"

# ====== 收集部署库到 lib/ ======
echo "=== Collecting deployment libraries ==="

# 1) libtorch_std_helper.so
cp "${TORCH_SO}" "${LIB_DIR}/"
echo "  → libtorch_std_helper.so"

# 2) 从构建机收集 PyTorch + CUDA 运行时 .so
# 优先从 Python torch 包取，否则从标准路径取
TORCH_LIB_DIR=$(
    python3 -c "
import torch, os
p = os.path.join(torch.__path__[0], 'lib')
if os.path.isdir(p): print(p)
" 2>/dev/null || echo "/data/venv/lib/python3.12/site-packages/torch/lib"
)

echo "  PyTorch lib dir: ${TORCH_LIB_DIR}"

# 核心 .so 列表（这些必须带）
CORE_SOS=(
    libtorch.so
    libc10.so
    libtorch_cpu.so
)
# CUDA 相关（如果存在就带上）
CUDA_SOS=(
    libtorch_cuda.so
    libc10_cuda.so
    libcudart.so.12
    libcudart.so.11.0
    libcublas.so.12
    libcublasLt.so.12
    libcudnn.so.9
    libcudnn_ops.so.9
    libcudnn_cnn.so.9
    libcudnn_adv.so.9
    libcufft.so.10
    libcurand.so.10
    libnvrtc.so.12
    libnvToolsExt.so.1
)

COPIED=0
for so in "${CORE_SOS[@]}" "${CUDA_SOS[@]}"; do
    # 先找 TORCH_LIB_DIR
    src=""
    if [ -f "${TORCH_LIB_DIR}/${so}" ]; then
        src="${TORCH_LIB_DIR}/${so}"
    elif [ -f "/usr/local/cuda/lib64/${so}" ]; then
        src="/usr/local/cuda/lib64/${so}"
    elif [ -f "/usr/lib/x86_64-linux-gnu/${so}" ]; then
        src="/usr/lib/x86_64-linux-gnu/${so}"
    elif [ -f "/usr/lib/${so}" ]; then
        src="/usr/lib/${so}"
    fi
    if [ -n "${src}" ]; then
        # 使用硬链接或复制（不重复复制硬链接的相同文件）
        if [ ! -f "${LIB_DIR}/${so}" ]; then
            cp -L "${src}" "${LIB_DIR}/${so}" 2>/dev/null && COPIED=$((COPIED+1)) && echo "  → ${so}"
        fi
    fi
done

# 3) 解析 DT_NEEDED 补全依赖（自动追踪）
echo "  Resolving DT_NEEDED dependencies..."
for f in "${LIB_DIR}"/*.so*; do
    [ -f "$f" ] || continue
    for dep in $(objdump -p "$f" 2>/dev/null | grep NEEDED | awk '{print $2}'); do
        if [ ! -f "${LIB_DIR}/${dep}" ]; then
            # 查找依赖
            found=""
            if [ -f "${TORCH_LIB_DIR}/${dep}" ]; then
                found="${TORCH_LIB_DIR}/${dep}"
            elif [ -f "/usr/local/cuda/lib64/${dep}" ]; then
                found="/usr/local/cuda/lib64/${dep}"
            elif [ -f "/usr/lib/x86_64-linux-gnu/${dep}" ]; then
                found="/usr/lib/x86_64-linux-gnu/${dep}"
            elif [ -f "/usr/lib/${dep}" ]; then
                found="/usr/lib/${dep}"
            fi
            if [ -n "${found}" ]; then
                cp -L "${found}" "${LIB_DIR}/${dep}" 2>/dev/null && echo "  → ${dep} (dependency)"
            fi
        fi
    done
done

echo "  Collected $(find "${LIB_DIR}" -name '*.so*' | wc -l) shared libraries"
echo "  Total lib size: $(du -sh "${LIB_DIR}" | cut -f1)"

# ====== 写入 C launcher ======
echo "=== Creating C launcher ==="

cat > "${LAUNCHER_DIR}/launcher.c" << 'LAUNCHER_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>

/* objcopy 嵌入的符号 */
extern char _binary_scheme_start, _binary_scheme_end;
extern char _binary_petite_boot_start, _binary_petite_boot_end;
extern char _binary_scheme_boot_start, _binary_scheme_boot_end;
extern char _binary_runtime_so_start, _binary_runtime_so_end;

static void write_embedded(const char *path, const char *start, const char *end) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); exit(1); }
    size_t len = end - start;
    if (fwrite(start, 1, len, f) != len) { perror("fwrite"); exit(1); }
    fclose(f);
    chmod(path, 0755);
}

static void ensure_dir(const char *dir) {
    struct stat st;
    if (stat(dir, &st) == -1) mkdir(dir, 0755);
}

/* 获取 ELF 自身所在目录 */
static void get_self_dir(char *buf, size_t sz) {
    char exe[PATH_MAX];
    ssize_t r = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (r > 0) {
        exe[r] = '\0';
        char *d = dirname(exe);
        snprintf(buf, sz, "%s", d);
    } else {
        snprintf(buf, sz, ".");
    }
}

int main(int argc, char *argv[]) {
    /* 确定自身目录 */
    char self_dir[PATH_MAX];
    get_self_dir(self_dir, sizeof(self_dir));

    /* 创建临时运行目录 */
    const char *tmpdir = "/stock/.tmp";
    ensure_dir("/stock");
    ensure_dir(tmpdir);

    /* 写入嵌入的 scheme 二进制 */
    char scheme_path[PATH_MAX];
    snprintf(scheme_path, sizeof(scheme_path), "%s/scheme", tmpdir);
    write_embedded(scheme_path, &_binary_scheme_start, &_binary_scheme_end);

    /* 写入 boot 文件 */
    char petite_path[PATH_MAX];
    snprintf(petite_path, sizeof(petite_path), "%s/petite.boot", tmpdir);
    write_embedded(petite_path, &_binary_petite_boot_start, &_binary_petite_boot_end);

    char scheme_boot_path[PATH_MAX];
    snprintf(scheme_boot_path, sizeof(scheme_boot_path), "%s/scheme.boot", tmpdir);
    write_embedded(scheme_boot_path, &_binary_scheme_boot_start, &_binary_scheme_boot_end);

    /* 写入运行时 .so */
    char runtime_path[PATH_MAX];
    snprintf(runtime_path, sizeof(runtime_path), "%s/runtime.so", tmpdir);
    write_embedded(runtime_path, &_binary_runtime_so_start, &_binary_runtime_so_end);

    /* 构建 LD_LIBRARY_PATH:
       1) 自身目录下的 lib/（部署包的标准位置）
       2) 标准 CUDA 路径
       3) 标准系统路径
    */
    char lib_path[PATH_MAX * 4];
    snprintf(lib_path, sizeof(lib_path),
        "%s/lib:%s/lib/x86_64-linux-gnu:/usr/lib:/usr/local/cuda/lib64",
        self_dir, self_dir);
    setenv("LD_LIBRARY_PATH", lib_path, 1);

    /* exec scheme */
    chdir(tmpdir);
    execl("./scheme", "scheme", "--quiet",
          "--boot", "petite.boot",
          "--library-version", "runtime.so", NULL);

    perror("execl");
    return 1;
}
LAUNCHER_EOF

echo "  Launcher: ${LAUNCHER_DIR}/launcher.c"

# ====== 编译 launcher 目标文件 ======
gcc -c -o "${LAUNCHER_DIR}/launcher.o" "${LAUNCHER_DIR}/launcher.c" \
    -O2 -Wall -Wno-unused-parameter

# ====== 嵌入二进制数据 ======
echo "=== Embedding binary data ==="

EMBED_OBJ="${LAUNCHER_DIR}/embedded.o"
rm -f "${EMBED_OBJ}"

CHEZ_PATH=$(command -v "${CHEZ}" 2>/dev/null || echo "/usr/bin/scheme")
if [ -f "${CHEZ_PATH}" ]; then
    # 如果 scheme 是脚本 wrapper，找到真正的二进制
    if file "${CHEZ_PATH}" 2>/dev/null | grep -q "script"; then
        REAL_CHEZ=$(head -1 "${CHEZ_PATH}" | grep -oE '/[^ ]*scheme[^ ]*' | head -1 || echo "")
        if [ -n "${REAL_CHEZ}" ] && [ -f "${REAL_CHEZ}" ]; then
            CHEZ_PATH="${REAL_CHEZ}"
        fi
    fi
    echo "  Embedding: ${CHEZ_PATH}"
    echo "  Embedding: ${CHEZ_LIB}/petite.boot"
    echo "  Embedding: ${CHEZ_LIB}/scheme.boot"
    echo "  Embedding: ${RUNTIME_SO}"
    ld -r -b binary -o "${EMBED_OBJ}" \
        "${CHEZ_PATH}" \
        "${CHEZ_LIB}/petite.boot" \
        "${CHEZ_LIB}/scheme.boot" \
        "${RUNTIME_SO}"
else
    echo "Error: scheme binary not found at ${CHEZ_PATH}" >&2
    exit 1
fi

# ====== 链接最终 ELF ======
echo "=== Linking final ELF ==="

# RPATH 指向 $ORIGIN/lib/（与 ELF 同目录下的 lib/，包含所有 .so）
gcc -o "${DEPLOY_DIR}/${OUTPUT}" \
    "${LAUNCHER_DIR}/launcher.o" \
    "${EMBED_OBJ}" \
    -O2 -no-pie -ldl \
    -Wl,-rpath,'$ORIGIN/lib'

echo ""
echo "=== Deliver complete ==="
echo "  ELF:        ${DEPLOY_DIR}/${OUTPUT} ($(stat -c%s "${DEPLOY_DIR}/${OUTPUT}") bytes)"
echo "  Libraries:  ${LIB_DIR}/ ($(du -sh "${LIB_DIR}" | cut -f1))"
echo ""
echo "Deployment structure:"
echo "  ${DEPLOY_DIR}/"
echo "  ├── ${OUTPUT}"
echo "  └── lib/"
find "${LIB_DIR}" -name '*.so*' -printf "      ├── %f\n" | sort
echo ""
echo "Run: ${DEPLOY_DIR}/${OUTPUT}"
echo ""
echo "Deployment machine needs: NVIDIA driver + libc (glibc)"
echo "No Python, no PyTorch installation required."
echo ""

# ====== 创建部署 tarball ======
echo "=== Creating deployment tarball ==="
TARBALL="${BUILD_DIR}/deploy.tar.gz"
(cd "${BUILD_DIR}" && tar czf "../${TARBALL}" deploy/)
echo "  ${TARBALL} ($(stat -c%s "${TARBALL}") bytes)"
echo ""
