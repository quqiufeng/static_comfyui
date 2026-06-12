#!/bin/bash
# sd_embed.sh — Build SD Runtime ELF
set -euo pipefail
SCHEME="/opt/ChezScheme/ta6le/bin/ta6le/scheme"
RESCHEME_DIR="/opt/ReScheme"

echo "=== static_comfyui — ELF Build ==="
echo "  [1] Merging sources..."
MERGED_PY="/tmp/sd_full.py"
> "$MERGED_PY"
for f in sd_runtime/array_ops.static.py sd_runtime/nn_ops.static.py sd_runtime/unet_blocks.static.py sd_runtime/samplers.static.py sd_runtime/clip.static.py sd_runtime/vae.static.py sd_runtime/model_loader.static.py sd_runtime/unet.static.py sd_runtime/main.static.py; do
    [ -f "$f" ] && cat "$f" >> "$MERGED_PY" && echo "" >> "$MERGED_PY"
done
echo "  [2] Translating..."
/data/venv/bin/python3 "$RESCHEME_DIR/static_translate.py" < "$MERGED_PY" > /tmp/sd_translated.ss
echo "  [3] Compiling..."
MERGED_SS="/tmp/sd_merged.ss"
cat "$RESCHEME_DIR/static_prelude.scm" > "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat "$RESCHEME_DIR/static_stdlib.scm" >> "$MERGED_SS"
echo "" >> "$MERGED_SS"
cat /tmp/sd_translated.ss >> "$MERGED_SS"
cat > /tmp/sd_compile.ss << EOF
(import (chezscheme))
(compile-file "$MERGED_SS")
EOF
$SCHEME --quiet /tmp/sd_compile.ss 2>&1
cp "${MERGED_SS}o" /tmp/sd_runtime.so 2>/dev/null || cp /tmp/sd_merged.so /tmp/sd_runtime.so 2>/dev/null || true

echo "  [4] Packaging ELF..."
BUILD_DIR=$(mktemp -d /tmp/sd-elf-XXXXXX)
# Embed scheme binary
{ echo "#include <stddef.h>"; echo "const unsigned char _binary_scheme[] = {"; xxd -i < "$SCHEME"; echo "};"; echo "const unsigned int _binary_scheme_len = sizeof(_binary_scheme);"; } > "$BUILD_DIR/scheme.c"
# Embed boot files
BOOT_FILES="/opt/ChezScheme/boot/ta6le/petite.boot /opt/ChezScheme/boot/ta6le/scheme.boot"
echo '#include <stddef.h>' > "$BUILD_DIR/boot.c"
IDX=0
for f in $BOOT_FILES; do
    BASE=$(basename "$f")
    echo "static const unsigned char _boot_${IDX}[] = {" >> "$BUILD_DIR/boot.c"
    xxd -i < "$f" | grep -v "unsigned" | head -c -1 >> "$BUILD_DIR/boot.c"
    echo "};" >> "$BUILD_DIR/boot.c"
    echo "static const unsigned int _boot_${IDX}_len = sizeof(_boot_${IDX});" >> "$BUILD_DIR/boot.c"
    IDX=$((IDX + 1))
done
echo "typedef struct { const char *name; const unsigned char *data; unsigned int len; } _boot_entry;" >> "$BUILD_DIR/boot.c"
echo "const _boot_entry _boot_table[] = { {\"petite.boot\", _boot_0, _boot_0_len}, {\"scheme.boot\", _boot_1, _boot_1_len}, {NULL, NULL, 0} };" >> "$BUILD_DIR/boot.c"
# Embed sd_runtime.so
{ echo "#include <stddef.h>"; echo "const unsigned char _binary_so[] = {"; xxd -i < /tmp/sd_runtime.so | grep -v "unsigned" | head -c -1; echo "};"; echo "const unsigned int _binary_so_len = sizeof(_binary_so);"; } > "$BUILD_DIR/so.c"
# Embed libdgemm_row.so
{ echo "#include <stddef.h>"; echo "const unsigned char _binary_dgemm[] = {"; xxd -i < /tmp/libdgemm_row.so | grep -v "unsigned" | head -c -1; echo "};"; echo "const unsigned int _binary_dgemm_len = sizeof(_binary_dgemm);"; } > "$BUILD_DIR/dgemm.c"
cat > "$BUILD_DIR/launcher.c" << 'LAUNCHER'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
extern const unsigned char _binary_scheme[]; extern const unsigned int _binary_scheme_len;
extern const unsigned char _binary_so[]; extern const unsigned int _binary_so_len;
extern const unsigned char _binary_dgemm[]; extern const unsigned int _binary_dgemm_len;
typedef struct { const char *name; const unsigned char *data; unsigned int len; } _boot_entry;
extern const _boot_entry _boot_table[];

static void write_file(const char *p, const unsigned char *d, unsigned int l, int m) {
    char dir[4096]; strncpy(dir, p, sizeof(dir)-1); dir[sizeof(dir)-1]=0;
    char *sep = strrchr(dir, '/');
    if (sep) { *sep=0; char *pd=dir; while(*pd){if(*pd=='/'){*pd=0;mkdir(dir,0755);*pd='/';}pd++;} mkdir(dir,0755); }
    FILE *f = fopen(p,"wb"); if(!f){fprintf(stderr,"Error: cannot write %s\n",p);exit(1);}
    fwrite(d,1,l,f); fclose(f); if(m) chmod(p,m);
}
int main(int argc, char **argv) {
    const char *base="/stock/.tmp"; mkdir(base,0755);
    char scheme_path[4096], so_path[4096], lib_dir[4096], boot_dir[4096];
    snprintf(scheme_path,sizeof(scheme_path),"%s/scheme",base);
    snprintf(so_path,sizeof(so_path),"%s/runtime.so",base);
    snprintf(lib_dir,sizeof(lib_dir),"%s/lib",base); mkdir(lib_dir,0755);
    snprintf(boot_dir,sizeof(boot_dir),"%s/boot",base); mkdir(boot_dir,0755);
    write_file(scheme_path,_binary_scheme,_binary_scheme_len,0755);
    write_file(so_path,_binary_so,_binary_so_len,0644);
    { char dgemm_path[4096]; snprintf(dgemm_path,sizeof(dgemm_path),"%s/libdgemm_row.so",lib_dir);
      write_file(dgemm_path,_binary_dgemm,_binary_dgemm_len,0755); }
    { const _boot_entry *e; for(e=_boot_table;e->name!=NULL;e++){
        char p[4096]; snprintf(p,sizeof(p),"%s/%s",boot_dir,e->name);
        write_file(p,e->data,e->len,0644);
    }}
    char ld[4096]; snprintf(ld,sizeof(ld),"%s",lib_dir);
    setenv("LD_LIBRARY_PATH",ld,1); setenv("SCHEMEHEAPDIRS",boot_dir,1);
    fprintf(stderr,"=== static_comfyui ===\n");
    if(chdir(base)!=0){fprintf(stderr,"Warning: chdir failed\n");}
    char *args[]={scheme_path,"--quiet",so_path,NULL};
    execv(scheme_path,args);
    fprintf(stderr,"Error: exec failed\n"); return 1;
}
LAUNCHER

gcc -c -O2 "$BUILD_DIR/scheme.c" -o "$BUILD_DIR/scheme.o"
gcc -c -O2 "$BUILD_DIR/boot.c" -o "$BUILD_DIR/boot.o"
gcc -c -O2 "$BUILD_DIR/so.c" -o "$BUILD_DIR/so.o"
gcc -c -O2 "$BUILD_DIR/dgemm.c" -o "$BUILD_DIR/dgemm.o"
gcc -c -O2 "$BUILD_DIR/launcher.c" -o "$BUILD_DIR/launcher.o"
gcc -o "${1:-sd_generate.elf}" "$BUILD_DIR/launcher.o" "$BUILD_DIR/scheme.o" "$BUILD_DIR/boot.o" "$BUILD_DIR/so.o" "$BUILD_DIR/dgemm.o" -ldl -Wl,--export-dynamic
rm -rf "$BUILD_DIR"
echo "  [5] Done: ${1:-sd_generate.elf} ($(du -h "${1:-sd_generate.elf}" | cut -f1))"
