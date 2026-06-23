#!/usr/bin/env bash
# deliver.sh — 将编译产物打包为单文件 ELF
#
# 前提: 先运行 build.sh 产生 build_out/runtime.so 和 build_out/libs/libtorch_std_helper.so
#
# 用法:
#   ./deliver.sh                    打包默认产物
#   ./deliver.sh -o my_model.elf    指定输出文件名
#
# 环境变量:
#   CHEZ_SCHEME=scheme         Chez Scheme 可执行文件路径
#   CHEZ_BOOT_DIR=./boot       Chez boot 文件目录 (petite.boot, scheme.boot)
#   BUILD_DIR=build_out        构建输出目录
#   OUTPUT=sd_generate.elf     输出 ELF 文件名

set -euo pipefail

# ====== 配置 ======
OUTPUT="${OUTPUT:-sd_generate.elf}"
BUILD_DIR="${BUILD_DIR:-build_out}"
LIBS_DIR="${BUILD_DIR}/libs"
LAUNCHER_DIR="${BUILD_DIR}/launcher"
CHEZ="${CHEZ_SCHEME:-scheme}"

# 查找 Chez Scheme boot files
CHEZ_ROOT=$(dirname "$(command -v "${CHEZ}")" 2>/dev/null || echo "/usr/bin")
CHEZ_LIB="${CHEZ_ROOT}/../lib/csv"  # 常见安装路径
if [ ! -f "${CHEZ_LIB}/petite.boot" ]; then
    CHEZ_LIB="${CHEZ_ROOT}/../lib/csv9.5"  # 另一常见路径
fi
if [ ! -f "${CHEZ_LIB}/petite.boot" ]; then
    CHEZ_LIB="${CHEZ_ROOT}/../lib/csv10.0"  # 另一常见路径
fi
if [ ! -f "${CHEZ_LIB}/petite.boot" ]; then
    CHEZ_LIB="${CHEZ_ROOT}/../lib"  # 尝试直接 lib 目录
fi
if [ ! -f "${CHEZ_LIB}/petite.boot" ]; then
    echo "Warning: Cannot find petite.boot, searching..." >&2
    CHEZ_LIB=$(find / -name "petite.boot" 2>/dev/null | head -1 | xargs dirname)
fi
echo "  Boot dir: ${CHEZ_LIB}"

# ====== 检查构建产物 ======
echo "=== Checking build artifacts ==="

RUNTIME_SO="${BUILD_DIR}/runtime.so"
TORCH_SO="${LIBS_DIR}/libtorch_std_helper.so"

if [ ! -f "${RUNTIME_SO}" ]; then
    echo "Error: ${RUNTIME_SO} not found. Run build.sh first." >&2
    exit 1
fi

echo "  Runtime SO: ${RUNTIME_SO} ($(stat -c%s "${RUNTIME_SO}") bytes)"
if [ -f "${TORCH_SO}" ]; then
    echo "  Torch SO:   ${TORCH_SO} ($(stat -c%s "${TORCH_SO}") bytes)"
else
    echo "  Torch SO:   (will use system-installed libtorch)"
fi

# ====== 创建 C launcher ======
echo "=== Creating C launcher ==="

mkdir -p "${LAUNCHER_DIR}"

# 生成 C launcher 源码
# 它负责: 1) 创建临时目录 2) 写入 embedded 数据 3) exec scheme
cat > "${LAUNCHER_DIR}/launcher.c" << 'LAUNCHER_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <libgen.h>

/* 由 objcopy 提供的符号 */
extern char _binary_scheme_start;
extern char _binary_scheme_end;
extern char _binary_petite_boot_start;
extern char _binary_petite_boot_end;
extern char _binary_scheme_boot_start;
extern char _binary_scheme_boot_end;
extern char _binary_runtime_so_start;
extern char _binary_runtime_so_end;
extern char _binary_libtorch_std_helper_so_start;
extern char _binary_libtorch_std_helper_so_end;

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
    if (stat(dir, &st) == -1) {
        mkdir(dir, 0755);
    }
}

int main(int argc, char *argv[]) {
    /* 运行时不需要任何输出，除非有错误 */

    /* 创建临时目录 */
    const char *tmpdir = "/stock/.tmp";
    ensure_dir("/stock");
    ensure_dir(tmpdir);

    /* 工具目录 */
    char bindir[256];
    snprintf(bindir, sizeof(bindir), "%s/bin", tmpdir);
    ensure_dir(bindir);

    /* 写入方案二进制和 boot 文件 */
    char scheme_path[256];
    snprintf(scheme_path, sizeof(scheme_path), "%s/scheme", tmpdir);
    write_embedded(scheme_path, &_binary_scheme_start, &_binary_scheme_end);

    char petite_path[256];
    snprintf(petite_path, sizeof(petite_path), "%s/petite.boot", tmpdir);
    write_embedded(petite_path, &_binary_petite_boot_start, &_binary_petite_boot_end);

    char scheme_boot_path[256];
    snprintf(scheme_boot_path, sizeof(scheme_boot_path), "%s/scheme.boot", tmpdir);
    write_embedded(scheme_boot_path, &_binary_scheme_boot_start, &_binary_scheme_boot_end);

    /* 写入运行时 .so */
    char runtime_path[256];
    snprintf(runtime_path, sizeof(runtime_path), "%s/runtime.so", tmpdir);
    write_embedded(runtime_path, &_binary_runtime_so_start, &_binary_runtime_so_end);

    /* 写入 torch_std_helper .so (如果嵌入) */
    char torch_path[256];
    snprintf(torch_path, sizeof(torch_path), "%s/libtorch_std_helper.so", tmpdir);
    size_t torch_len = &_binary_libtorch_std_helper_so_end - &_binary_libtorch_std_helper_so_start;
    if (torch_len > 0) {
        write_embedded(torch_path, &_binary_libtorch_std_helper_so_start, &_binary_libtorch_std_helper_so_end);
    }

    /* 设置环境 */
    setenv("LD_LIBRARY_PATH",
           "/data/venv/lib/python3.12/site-packages/torch/lib:"
           "/usr/local/cuda/lib64:" tmpdir,
           1);

    /* exec scheme 运行 runtime.so */
    chdir(tmpdir);
    execl("./scheme", "scheme", "--quiet", "--boot", "petite.boot",
          "--library-version", "runtime.so", NULL);

    /* 如果 execl 返回，说明出错 */
    perror("execl");
    return 1;
}
LAUNCHER_EOF

echo "  Launcher source: ${LAUNCHER_DIR}/launcher.c"

# ====== 编译 launcher ======
echo "=== Compiling launcher ==="

gcc -c -o "${LAUNCHER_DIR}/launcher.o" "${LAUNCHER_DIR}/launcher.c" \
    -O2 -Wall -Wextra

# ====== 嵌入二进制文件 ======
echo "=== Embedding binary data ==="

# 使用 objcopy 将文件作为二进制符号嵌入
# 语法: objcopy --add-symbol <name>=<file> --rename-section .data=<section>

EMBED_OBJ="${LAUNCHER_DIR}/embedded.o"
rm -f "${EMBED_OBJ}"

# Embed scheme binary
CHEZ_PATH=$(command -v "${CHEZ}" 2>/dev/null || echo "/usr/bin/scheme")
if [ -f "${CHEZ_PATH}" ]; then
    # 检查是否为脚本 wrapper (Chez 有时安装为脚本)
    if file "${CHEZ_PATH}" | grep -q "script"; then
        # 尝试找到真正的二进制
        REAL_CHEZ=$(head -1 "${CHEZ_PATH}" | grep -o '/[^ ]*scheme[^ ]*' || echo "")
        if [ -n "${REAL_CHEZ}" ] && [ -f "${REAL_CHEZ}" ]; then
            CHEZ_PATH="${REAL_CHEZ}"
        fi
    fi
    echo "  Embedding scheme binary: ${CHEZ_PATH}"
    ld -r -b binary -o "${EMBED_OBJ}" \
        "${CHEZ_PATH}" \
        "${CHEZ_LIB}/petite.boot" \
        "${CHEZ_LIB}/scheme.boot" \
        "${RUNTIME_SO}"
    # Add torch .so if exists
    if [ -f "${TORCH_SO}" ]; then
        echo "  Embedding torch_std_helper: ${TORCH_SO}"
        # 我们需要额外添加一个段，或者用更简单的方法
        # 将 .so 作为额外目标文件加入
        TORCH_OBJ="${LAUNCHER_DIR}/torch_embedded.o"
        ld -r -b binary -o "${TORCH_OBJ}" "${TORCH_SO}"
        # 合并两个嵌入文件
        ld -r -o "${EMBED_OBJ}.merged" "${EMBED_OBJ}" "${TORCH_OBJ}"
        mv "${EMBED_OBJ}.merged" "${EMBED_OBJ}"
    fi
else
    echo "Warning: scheme binary not found at ${CHEZ_PATH}" >&2
    echo "  Will try to use a simpler embedding" >&2
    ld -r -b binary -o "${EMBED_OBJ}" \
        "${RUNTIME_SO}"
fi

# ====== 链接最终 ELF ======
echo "=== Linking final ELF ==="

# 查找 PyTorch 库路径
TORCH_LIB=$(${PYTHON:-python3} -c "import torch; import os; print(os.path.join(torch.__path__[0], 'lib'))" 2>/dev/null || echo "/data/venv/lib/python3.12/site-packages/torch/lib")

gcc -o "${OUTPUT}" \
    "${LAUNCHER_DIR}/launcher.o" \
    "${EMBED_OBJ}" \
    -O2 -no-pie -ldl \
    -Wl,-rpath,"${TORCH_LIB}"

echo ""
echo "=== Deliver complete ==="
echo "  Output: ${OUTPUT} ($(stat -c%s "${OUTPUT}") bytes)"
echo ""
echo "Run: ./${OUTPUT}"
