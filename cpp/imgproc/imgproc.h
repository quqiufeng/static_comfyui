// imgproc.h — 独立图像处理底层库（C ABI）
//
// 供 StaticPy 通过 FFI 在运行时加载调用（见 imgproc/imgproc_ffi.scm）。
// 与推理栈（sdcpp_adapter / torch）无关：只做图像格式编解码 + 原始像素缓冲。
//
// 当前能力：GIF 多帧解码 / 编码（内部动态链接 libgif.so.7 运行时解析）。
// 像素格式统一为 RGBA8（w*h*4），帧延时为毫秒。
#ifndef IMGPROC_H
#define IMGPROC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 打开（解码）一个图像文件，返回不透明句柄；失败返回 NULL。
void* img_open(const char* path);

// 新建一个空白输出图像（w*h，n 帧，索引图全 0）。
void* img_new(int w, int h, int n);

// 按 src 的结构复制出空白输出图（同尺寸/帧数/bpp/调色板/延时/透明，像素清零）。
// 这是做逐像素变换（如镜像）的推荐入口：填好像素后 img_write 即可。
void* img_clone_blank(void* src);

int  img_width(void* h);
int  img_height(void* h);
int  img_count(void* h);

// 第 i 帧每像素字节数：1=调色板索引，4=RGBA。
int  img_frame_bpp(void* h, int i);

// 第 i 帧像素缓冲首地址（大小 = w*h*bpp 字节），越界返回 NULL。
unsigned char* img_frame_ptr(void* h, int i);
unsigned char* img_pixels(void* h, int i);  // 兼容旧名

// 第 i 帧延时（毫秒）。
int  img_delay(void* h, int i);
int  img_set_delay(void* h, int i, int ms);

// 写出文件（按扩展名派发：.gif → GIF 编码）。成功返回 0。
int  img_write(void* h, const char* path);

// 释放句柄（解码/新建得到的都适用）。
int  img_free(void* h);

// —— 原始内存字节读写（供 StaticPy 在缓冲上做像素级处理，如镜像）——
unsigned int  img_u32_get(void* p, long off);
void          img_u32_set(void* p, long off, unsigned int v);
unsigned char img_u8_get(void* p, long off);
int           img_u8_set(void* p, long off, unsigned char v);

#ifdef __cplusplus
}
#endif

#endif  // IMGPROC_H
