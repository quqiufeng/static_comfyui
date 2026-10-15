// quant.cpp — Pillow Quant.c 的 MEDIANCUT 精确移植（哈希聚合 + 3 链表排序 + 最大堆 median-cut + 逐盒平均 + 最近色映射）
#include "imgproc.h"
#include "imgproc_internal.h"
#include <unordered_map>
#include <functional>
#include <queue>

namespace {

struct Node {
    int v[3];
    int count;
    bool flag;
    int next[3], prev[3];
};
struct Box {
    int l = -1, r = -1;
    int head[3], tail[3], axis = -1, volume = -1;
    long pixelCount = 0;
};

struct Quant {
    std::vector<Node> nodes;
    std::vector<Box> boxes;
    // 1-indexed max-heap by pixelCount（对齐 QuantHeap）
    std::vector<int> heap;  // box indices, heap[0] unused
    int count_ = 0;
    int pixelTotal = 0;

    int cmp(int a, int b) { return (int)(boxes[a].pixelCount - boxes[b].pixelCount); }
    void heapAdd(int val) {
        if ((int)heap.size() <= count_ + 1) heap.resize(heap.size() * 2);
        int k = ++count_;
        while (k != 1) { if (cmp(val, heap[k / 2]) <= 0) break; heap[k] = heap[k / 2]; k >>= 1; }
        heap[k] = val;
    }
    bool heapRemove(int& out) {
        if (!count_) return false;
        out = heap[1];
        int v = heap[count_--];
        int k, l;
        for (k = 1; k * 2 <= count_; k = l) {
            l = k * 2;
            if (l < count_ && cmp(heap[l], heap[l + 1]) < 0) l++;
            if (cmp(v, heap[l]) > 0) break;
            heap[k] = heap[l];
        }
        heap[k] = v;
        return true;
    }

    int boxVolume(Box& b) {
        if (b.volume >= 0) return b.volume;
        if (b.head[0] < 0) b.volume = 0;
        else {
            int rh = nodes[b.head[0]].v[0], rl = nodes[b.tail[0]].v[0];
            int gh = nodes[b.head[1]].v[1], gl = nodes[b.tail[1]].v[1];
            int bh = nodes[b.head[2]].v[2], bl = nodes[b.tail[2]].v[2];
            b.volume = (rh - rl + 1) * (gh - gl + 1) * (bh - bl + 1);
        }
        return b.volume;
    }

    bool split(int ni) {
        Box& node = boxes[ni];
        int f[3];
        f[0] = (nodes[node.head[0]].v[0] - nodes[node.tail[0]].v[0]) * 77;
        f[1] = (nodes[node.head[1]].v[1] - nodes[node.tail[1]].v[1]) * 150;
        f[2] = (nodes[node.head[2]].v[2] - nodes[node.tail[2]].v[2]) * 29;
        int best = f[0], axis = 0;
        for (int i = 1; i < 3; ++i) if (best < f[i]) { best = f[i]; axis = i; }
        node.axis = axis;
        int leftheads[2][3], lefttails[2][3]; long nCount[2];
        splitlists_inner(ni, axis, leftheads, lefttails, nCount);
        int li = (int)boxes.size(); boxes.push_back(Box()); 
        int ri = (int)boxes.size(); boxes.push_back(Box());
        Box& L = boxes[li]; Box& R = boxes[ri];
        for (int i = 0; i < 3; ++i) { L.head[i] = leftheads[0][i]; L.tail[i] = lefttails[0][i]; R.head[i] = leftheads[1][i]; R.tail[i] = lefttails[1][i]; node.head[i] = -1; node.tail[i] = -1; }
        node.l = li; node.r = ri;
        L.pixelCount = nCount[0]; R.pixelCount = nCount[1];
        return true;
    }

    void splitlists_inner(int ni, int axis, int h[2][3], int t[2][3], long nCount[2]) {
        Box& node = boxes[ni];
        long left; int c;
        long nRight = 0; int splitColourVal;
        nCount[0] = nCount[1] = 0;
        for (left = 0, c = node.head[axis]; c >= 0;) {
            left += nodes[c].count; nCount[0] += nodes[c].count; nodes[c].flag = false;
            c = nodes[c].next[axis];
            if (left * 2 > node.pixelCount) break;
        }
        if (c >= 0) {
            splitColourVal = nodes[nodes[c].prev[axis]].v[axis];
            for (; c >= 0; c = nodes[c].next[axis]) { if (splitColourVal != nodes[c].v[axis]) break; nodes[c].flag = false; nCount[0] += nodes[c].count; }
        }
        for (; c >= 0; c = nodes[c].next[axis]) { nodes[c].flag = true; nRight++; nCount[1] += nodes[c].count; }
        if (!nRight) {
            c = node.tail[axis]; splitColourVal = nodes[c].v[axis];
            for (; c >= 0; c = nodes[c].prev[axis]) { if (splitColourVal != nodes[c].v[axis]) break; nodes[c].flag = true; nRight++; nCount[0] -= nodes[c].count; nCount[1] += nodes[c].count; }
        }
        for (int i = 0; i < 3; ++i) {
            int l = -1, r = -1, nh0 = -1, nt0 = -1, nh1 = -1, nt1 = -1;
            for (int cur = node.head[i]; cur >= 0;) {
                int nx = nodes[cur].next[i];
                if (nodes[cur].flag) { if (r >= 0) nodes[r].next[i] = cur; else nh1 = cur; nodes[cur].prev[i] = r; r = cur; }
                else { if (l >= 0) nodes[l].next[i] = cur; else nh0 = cur; nodes[cur].prev[i] = l; l = cur; }
                cur = nx;
            }
            if (l >= 0) nodes[l].next[i] = -1;
            if (r >= 0) nodes[r].next[i] = -1;
            nt0 = l; nt1 = r;
            h[0][i] = nh0; t[0][i] = nt0; h[1][i] = nh1; t[1][i] = nt1;
        }
    }
};

}  // namespace

extern "C" {

// 对齐 Pillow ImagingQuantize(mode=MEDIANCUT)。仅 RGB（忽略 alpha）。
void* img_quantize(void* p, int colors, int method) {
    (void)method;
    Image* im = (Image*)p; if (!im || colors < 1 || colors > 256) return nullptr;
    Image* rg = (im->frames[0].bpp == 4) ? im : to_rgba(im);
    int W = rg->w, H = rg->h;
    long nPixels = (long)W * H;
    uint8_t* data = rg->frames[0].data.data();

    // 1) 聚合唯一色 + 计数（scale 处理：唯一色 > 65536 时右移）
    std::unordered_map<uint32_t, int> agg;
    std::vector<uint32_t> order;
    int scale = 0;
    for (int attempt = 0; attempt < 8; ++attempt) {
        agg.clear(); order.clear();
        for (long i = 0; i < nPixels; ++i) {
            uint8_t* q = data + i * 4;
            uint32_t key = ((uint32_t)(q[0] >> scale) << 16) | ((uint32_t)(q[1] >> scale) << 8) | (q[2] >> scale);
            auto it = agg.find(key);
            if (it == agg.end()) { agg[key] = 1; order.push_back(key); }
            else it->second++;
        }
        if ((int)agg.size() <= 65536) break;
        scale++;
    }

    Quant Q; Q.pixelTotal = (int)nPixels;
    int n = (int)order.size();
    Q.nodes.resize(n);
    for (int i = 0; i < n; ++i) {
        uint32_t k = order[i];
        Node& nd = Q.nodes[i];
        nd.v[0] = (k >> 16) & 0xFF; nd.v[1] = (k >> 8) & 0xFF; nd.v[2] = k & 0xFF;
        nd.count = agg[k]; nd.flag = false;
        for (int a = 0; a < 3; ++a) { nd.next[a] = -1; nd.prev[a] = -1; }
    }
    // 2) 三个轴各自降序排序并建链
    int heads[3], tails[3];
    for (int a = 0; a < 3; ++a) {
        std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return Q.nodes[x].v[a] > Q.nodes[y].v[a]; });
        for (int i = 0; i < n; ++i) { Q.nodes[idx[i]].prev[a] = (i ? idx[i - 1] : -1); Q.nodes[idx[i]].next[a] = (i + 1 < n ? idx[i + 1] : -1); }
        heads[a] = n ? idx[0] : -1; tails[a] = n ? idx[n - 1] : -1;
    }
    // 3) median cut
    Q.boxes.reserve(2 * colors + 4);
    Q.boxes.push_back(Box());
    int root = 0;
    for (int a = 0; a < 3; ++a) { Q.boxes[root].head[a] = heads[a]; Q.boxes[root].tail[a] = tails[a]; }
    Q.boxes[root].pixelCount = nPixels;
    Q.heap.assign(512, 0); Q.count_ = 0;
    Q.heapAdd(root);
    for (int k = 1; k < colors; ++k) {
        int thisNode;
        do { if (!Q.heapRemove(thisNode)) { k = colors; break; } } while (Q.boxVolume(Q.boxes[thisNode]) == 1);
        if (k >= colors) break;
        if (!Q.split(thisNode)) break;
        Q.heapAdd(Q.boxes[thisNode].l);
        Q.heapAdd(Q.boxes[thisNode].r);
    }
    // 4) annotate：叶子盒分配索引（DFS 左→右）
    std::unordered_map<uint32_t, int> pixBox;
    int boxCounter = 0;
    // 递归（用显式栈）
    {
        std::vector<int> st; st.push_back(root);
        std::function<void(int)> dfs = [&](int ni) {
            Box& b = Q.boxes[ni];
            if (b.l >= 0 && b.r >= 0) { dfs(b.l); dfs(b.r); return; }
            if (b.l >= 0 || b.r >= 0) return;
            for (int c = b.head[0]; c >= 0; c = Q.nodes[c].next[0]) {
                uint32_t key = ((uint32_t)(Q.nodes[c].v[0] << scale) << 16) | ((uint32_t)(Q.nodes[c].v[1] << scale) << 8) | (Q.nodes[c].v[2] << scale);
                pixBox[key] = boxCounter;
            }
            if (b.head[0] >= 0) boxCounter++;
        };
        dfs(root);
    }
    int nPal = boxCounter;
    // 5) 逐盒平均（按图像像素出现加权）
    std::vector<long> sumR(nPal, 0), sumG(nPal, 0), sumB(nPal, 0), cnt(nPal, 0);
    for (long i = 0; i < nPixels; ++i) {
        uint8_t* q = data + i * 4;
        uint32_t key = ((uint32_t)q[0] << 16) | ((uint32_t)q[1] << 8) | q[2];
        auto it = pixBox.find(key);
        int pe = (it == pixBox.end()) ? 0 : it->second;
        sumR[pe] += q[0]; sumG[pe] += q[1]; sumB[pe] += q[2]; cnt[pe]++;
    }
    std::vector<RGB> pal(nPal);
    for (int i = 0; i < nPal; ++i) {
        if (cnt[i] == 0) { pal[i] = {0, 0, 0}; continue; }
        pal[i] = {(uint8_t)(int)(0.5 + (double)sumR[i] / cnt[i]), (uint8_t)(int)(0.5 + (double)sumG[i] / cnt[i]), (uint8_t)(int)(0.5 + (double)sumB[i] / cnt[i])};
    }
    // 6) 映射每个像素到最近调色板色
    Image* o = new Image(); o->w = W; o->h = H; o->n = 1; o->mode = "P";
    o->frames.resize(1);
    Frame& d = o->frames[0]; d.bpp = 1; d.hasPalette = true; d.palette = pal;
    d.data.assign(nPixels, 0);
    std::unordered_map<uint32_t, int> mem;
    for (long i = 0; i < nPixels; ++i) {
        uint8_t* q = data + i * 4;
        uint32_t key = ((uint32_t)q[0] << 16) | ((uint32_t)q[1] << 8) | q[2];
        auto it = mem.find(key);
        if (it != mem.end()) { d.data[i] = (uint8_t)it->second; continue; }
        int best = 0; long bd = 1L << 60;
        for (int j = 0; j < nPal; ++j) {
            int dr = q[0] - pal[j].r, dg = q[1] - pal[j].g, db = q[2] - pal[j].b;
            long dd = (long)dr * dr + (long)dg * dg + (long)db * db;
            if (dd < bd) { bd = dd; best = j; }
        }
        mem[key] = best; d.data[i] = (uint8_t)best;
    }
    if (rg != im) delete rg;
    return o;
}

}  // extern "C"
