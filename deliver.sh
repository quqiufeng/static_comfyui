#!/bin/bash
# deliver.sh — static_comfyui 单文件 ELF 打包
# 编译 StaticPy → .so → 嵌入 scheme + boot + dgemm → 单文件 ELF
# 完全自包含，不依赖外部项目

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/data/venv/bin/python3}"
SCHEME="${SCHEME:-/opt/ChezScheme/ta6le/bin/ta6le/scheme}"
OUTPUT="${1:-sd_generate.elf}"

echo "=== static_comfyui — ELF Build ==="
echo "Output: $OUTPUT"

# Step 1: Compile .so
echo "  [1] Compiling StaticPy → .so..."
INPUT_PY="/tmp/sd_full_$$.py"
> "$INPUT_PY"
for f in sd_runtime/array_ops.static.py sd_runtime/torch_ops.static.py \
         sd_runtime/nn_ops.static.py sd_runtime/transformer_ops.static.py \
         sd_runtime/unet_forward.static.py \
         sd_runtime/samplers.static.py sd_runtime/clip.static.py \
         sd_runtime/model_loader.static.py sd_runtime/vae.static.py \
         sd_runtime/main.static.py; do
    [ -f "$f" ] && cat "$f" >> "$INPUT_PY" && echo "" >> "$INPUT_PY"
done

bash build.sh "$INPUT_PY" sd_runtime 2>&1

# Step 2: Build C runtime .so
echo "  [2] Building C runtime..."
gcc -O2 -shared -fPIC -o /tmp/libdgemm_row.so runtime/dgemm_wrapper.c -ldl

# Step 2b: Build PyTorch C wrapper
echo "  [2b] Building libstaticpy_torch.so..."
TORCH_INC=$($PYTHON -c 'from torch.utils.cpp_extension import include_paths; print(" ".join("-I"+p for p in include_paths()))')
TORCH_LIB=$($PYTHON -c 'import torch; print(torch.__path__[0]+"/lib")')
g++ -O2 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 \
  $TORCH_INC \
  -o /tmp/libstaticpy_torch.so runtime/staticpy_torch.cpp \
  -L"$TORCH_LIB" -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda -Wl,-rpath,"$TORCH_LIB"

# Step 3: Embed into ELF
echo "  [3] Packaging ELF..."
SO_FILE="build_out/sd_runtime.so"
BOOT_DIR="/opt/ChezScheme/boot/ta6le"
BUILD_DIR=$(mktemp -d /tmp/sd-elf-XXXXXX)

# Embed scheme binary
{ echo "#include <stddef.h>"; echo "const unsigned char _bin_scheme[] = {"; 
  xxd -i < "$SCHEME"; echo "};"; echo "const unsigned int _bin_scheme_len = sizeof(_bin_scheme);"; } > "$BUILD_DIR/s1.c"

# Embed boot files  
{ echo "#include <stddef.h>"; IDX=0
  for f in "$BOOT_DIR/petite.boot" "$BOOT_DIR/scheme.boot"; do
    echo "static const unsigned char _b${IDX}[] = {"; xxd -i < "$f" | grep -v unsigned | head -c -1
    echo "};"; echo "static const unsigned int _b${IDX}_len = sizeof(_b${IDX});"; IDX=$((IDX+1))
  done
  echo "const void* _boots[] = {_b0,_b1}; const unsigned int _boot_lens[] = {_b0_len,_b1_len};"
  echo "const char* _boot_names[] = {\"petite.boot\",\"scheme.boot\"};"
} > "$BUILD_DIR/boot.c"

# Embed .so
{ echo "#include <stddef.h>"; echo "const unsigned char _bin_so[] = {"; 
  xxd -i < "$SO_FILE" | grep -v unsigned | head -c -1; echo "};"
  echo "const unsigned int _bin_so_len = sizeof(_bin_so);"; } > "$BUILD_DIR/so.c"

# Embed dgemm .so
{ echo "#include <stddef.h>"; echo "const unsigned char _bin_dg[] = {";
  xxd -i < /tmp/libdgemm_row.so | grep -v unsigned | head -c -1; echo "};"
  echo "const unsigned int _bin_dg_len = sizeof(_bin_dg);"; } > "$BUILD_DIR/dg.c"

# Embed torch wrapper .so
{ echo "#include <stddef.h>"; echo "const unsigned char _bin_torch[] = {";
  xxd -i < /tmp/libstaticpy_torch.so | grep -v unsigned | head -c -1; echo "};"
  echo "const unsigned int _bin_torch_len = sizeof(_bin_torch);"; } > "$BUILD_DIR/torch.c"

# C launcher
cat > "$BUILD_DIR/main.c" << 'LAUNCHER'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
extern const unsigned char _bin_scheme[], _bin_so[], _bin_dg[], _bin_torch[];
extern const unsigned int _bin_scheme_len, _bin_so_len, _bin_dg_len, _bin_torch_len;
extern const void* _boots[2]; extern const unsigned int _boot_lens[2];
extern const char* _boot_names[2];

static void wf(const char *p, const unsigned char *d, unsigned int l, int m) {
    char dir[4096]; strncpy(dir,p,sizeof(dir)-1); dir[sizeof(dir)-1]=0;
    char *s = strrchr(dir,'/'); if(s){*s=0;char *pd=dir;while(*pd){if(*pd=='/'){*pd=0;mkdir(dir,0755);*pd='/';}pd++;}mkdir(dir,0755);}
    FILE *f=fopen(p,"wb"); if(!f){fprintf(stderr,"Error: %s\n",p);exit(1);} fwrite(d,1,l,f); fclose(f); if(m)chmod(p,m);
}
int main(int argc, char **argv) {
    const char *base="/stock/.tmp"; mkdir(base,0755);
    char sp[4096],sop[4096],lp[4096],bp[4096];
    snprintf(sp,sizeof(sp),"%s/scheme",base);
    snprintf(sop,sizeof(sop),"%s/runtime.so",base);
    snprintf(lp,sizeof(lp),"%s/lib",base); mkdir(lp,0755);
    snprintf(bp,sizeof(bp),"%s/boot",base); mkdir(bp,0755);
    wf(sp,_bin_scheme,_bin_scheme_len,0755);
    wf(sop,_bin_so,_bin_so_len,0644);
    { char p[4096]; snprintf(p,sizeof(p),"%s/libdgemm.so",lp); wf(p,_bin_dg,_bin_dg_len,0755); }
    { char p[4096]; snprintf(p,sizeof(p),"%s/libstaticpy_torch.so",lp); wf(p,_bin_torch,_bin_torch_len,0755); }
    // StaticPy translate.py hardcodes /tmp/libstaticpy_torch.so for dlopen
    { char cmd[8192]; snprintf(cmd,sizeof(cmd),"cp %s/libstaticpy_torch.so /tmp/libstaticpy_torch.so",lp); system(cmd); }
    for(int i=0;i<2;i++){char p[4096];snprintf(p,sizeof(p),"%s/%s",bp,_boot_names[i]);wf(p,_boots[i],_boot_lens[i],0644);}
    char ld[4096]; snprintf(ld,sizeof(ld),"%s:%s",lp,TORCH_LIB);
    char extra[4096]; snprintf(extra,sizeof(extra),"/data/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/data/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/data/venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/data/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/data/venv/lib/python3.12/site-packages/nvidia/nccl/lib");
    char final_ld[8192]; snprintf(final_ld,sizeof(final_ld),"%s:%s",ld,extra);
    setenv("LD_LIBRARY_PATH",final_ld,1); setenv("SCHEMEHEAPDIRS",bp,1);
    fprintf(stderr,"=== static_comfyui ELF ===\n");
    if(chdir(base)!=0)fprintf(stderr,"Warning: chdir failed\n");
    char *args[]={sp,"--quiet",sop,NULL};
    execv(sp,args);
    fprintf(stderr,"Error: exec failed\n"); return 1;
}
LAUNCHER

# 将 PyTorch 库路径作为 C 宏注入 launcher
sed -i "1i#define TORCH_LIB \"$TORCH_LIB\"" "$BUILD_DIR/main.c"

gcc -c -O2 "$BUILD_DIR/s1.c" -o "$BUILD_DIR/s1.o"
gcc -c -O2 "$BUILD_DIR/boot.c" -o "$BUILD_DIR/boot.o"
gcc -c -O2 "$BUILD_DIR/so.c" -o "$BUILD_DIR/so.o"
gcc -c -O2 "$BUILD_DIR/dg.c" -o "$BUILD_DIR/dg.o"
gcc -c -O2 "$BUILD_DIR/torch.c" -o "$BUILD_DIR/torch.o"
gcc -c -O2 "$BUILD_DIR/main.c" -o "$BUILD_DIR/main.o"
gcc -o "$OUTPUT" "$BUILD_DIR/main.o" "$BUILD_DIR/s1.o" "$BUILD_DIR/boot.o" \
    "$BUILD_DIR/so.o" "$BUILD_DIR/dg.o" "$BUILD_DIR/torch.o" -ldl -Wl,--export-dynamic

rm -rf "$BUILD_DIR" /tmp/sd-elf-* /tmp/sd_full_*.py
echo "  Done: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo "  ldd: $(ldd "$OUTPUT" 2>/dev/null | grep -c '=>' && ldd "$OUTPUT" | head -3)"
