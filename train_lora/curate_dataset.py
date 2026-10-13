#!/usr/bin/env python3
"""curate_dataset.py — 图片目录 → LoRA 数据集（检测筛 → 统一尺寸 → 自然语言打标）

两种模式（对应两个 LoRA；both 时先 style 再以 style 的图做 face）：
  style = 写真/整体 LoRA：缩放到合适大小(长边→--res，不裁) + 检测「是不是写真人像」
  face  = 人脸 LoRA：扣人脸(5点对齐裁 --crop) —— 输入可用第一遍处理完的图

流程（复用 static_comfyui 的 facecli / florence img2prompt，走批量=模型只加载一次）：
  1) 缩略图(1024, PIL draft 快) → 批量 facecli detect  → 人脸判据（有人脸/单脸/脸占比/脸分）
  2) 画质(Laplacian 方差)、近重复(dHash)
  3) 统一尺寸： style=长边缩到 --res（不裁）；face=5点对齐裁脸(--crop)
  4) 自然语言打标： florence img2prompt 批量（style=<DETAILED_CAPTION>，face=<CAPTION>）
     并据描述判「是否写真人像/照片」（排除 动漫/插画/截图/海报/logo…）→ 不合格剔除
  5) 产出 <out>/images/* + *.txt(触发词+描述) + dataset.toml + report.csv

用法:
  curate_dataset.py <src_dir> <out_dir> <trigger> --mode face|style [选项]
选项: --repo DIR --res N(style) --min-side --blur-min --min-face-ratio --min-face-score
      --max-faces(face默认1) --no-photo-check --dedup 0|1 --beams N --limit N
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NONPHOTO = re.compile(
    r"\b(anime|manga|cartoon|illustration|illustrated|drawing|drawn|sketch|painting|"
    r"figure|figurine|statue|doll|toy|logo|watermark|poster|advertisement|screenshot|"
    r"comic|graphic|vector|clipart|emoji|sticker|cd|dvd|cassette|album cover)\b", re.I)
PERSON = re.compile(r"\b(woman|women|man|men|girl|boy|person|people|portrait|face|model|female|male|lady|guy|child|kid)\b", re.I)


def log(*a): print(*a, file=sys.stderr, flush=True)


def thumb(src, dst, side=1024):
    try:
        im = Image.open(src); im.draft("RGB", (side, side))
        im = im.convert("RGB"); im.thumbnail((side, side)); im.save(dst, quality=90)
        return True
    except Exception:
        return False


def blur_of(p):
    try:
        g = np.asarray(Image.open(p).convert("L").resize((512, 512)), dtype=np.float64)
    except Exception:
        return -1.0
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def dhash(p, hs=8):
    try:
        g = np.asarray(Image.open(p).convert("L").resize((hs + 1, hs)), dtype=np.int16)
    except Exception:
        return None
    d = g[:, 1:] > g[:, :-1]
    return int("".join("1" if b else "0" for b in d.flatten()), 2)


def ham(a, b): return bin(a ^ b).count("1") if (a is not None and b is not None) else 999


def batch_faces(facecli, paths):
    """批量人脸检测；返回 {path: [faces...]}"""
    if not paths: return {}
    inp = "\n".join(str(p) for p in paths)
    try:
        out = subprocess.run([facecli, "detect", "--batch"], input=inp, capture_output=True, text=True)
        lines = out.stdout.splitlines()
    except Exception as e:
        log(f"facecli 失败: {e}"); return {}
    res = {}
    for p, l in zip(paths, lines):
        try: res[str(p)] = json.loads(l)
        except Exception: res[str(p)] = []
    return res


def batch_caption(img2prompt, model_dir, task, paths, beams=3):
    if not paths: return {}
    inp = "\n".join(str(p) for p in paths)
    try:
        out = subprocess.run([img2prompt, "--model-dir", model_dir, "--task", task,
                              "--beams", str(beams), "--batch"], input=inp, capture_output=True, text=True)
        lines = out.stdout.splitlines()
    except Exception as e:
        log(f"img2prompt 失败: {e}"); return {}
    return {str(p): (lines[i].strip() if i < len(lines) else "") for i, p in enumerate(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("trigger")
    ap.add_argument("--mode", choices=["style", "face"], default="face")
    ap.add_argument("--repo", default="/opt/static_comfyui")
    ap.add_argument("--res", type=int, default=1536)          # style 训练长边
    ap.add_argument("--train-width", type=int, default=0, help="style 输出统一宽度(=训练宽，0=用--res)；高度按原比例")
    ap.add_argument("--land-aspect", type=float, default=0.75, help="横图(宽>高)裁剪目标宽高比(默认0.75=3:4)")
    ap.add_argument("--face-scale", type=float, default=6.0, help="横图裁剪窗高 = 脸高 × 该倍数(默认6，人尽量大)")
    ap.add_argument("--no-land-crop", action="store_true", help="横图不裁，仅缩放")
    ap.add_argument("--crop", type=int, default=512)          # face 对齐脸尺寸
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--blur-min", type=float, default=30.0)
    ap.add_argument("--max-faces", type=int, default=-1)
    ap.add_argument("--min-face-ratio", type=float, default=-1)
    ap.add_argument("--min-face-score", type=float, default=0.60)
    ap.add_argument("--no-photo-check", action="store_true")
    ap.add_argument("--dedup", type=int, default=1)
    ap.add_argument("--rm-rejected", action="store_true", help="把不合格的原图（源文件）删掉")
    ap.add_argument("--rm-source", action="store_true", help="处理完后把源图全部删掉（只留处理结果）")
    ap.add_argument("--copy-to", default="", help="把合格图(+txt)另存到此目录")
    ap.add_argument("--beams", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    repo = Path(a.repo)
    facecli = str(repo / "cpp/face/facecli")
    img2prompt = str(repo / "cpp/florence2/img2prompt")
    model_dir = "/data/models/florence2"
    src = Path(a.src); out = Path(a.out)
    images_dir = out / "images"; rej = out / "rejected"
    for d in (images_dir, rej): d.mkdir(parents=True, exist_ok=True)

    max_faces = a.max_faces if a.max_faces >= 0 else 1   # 默认：只保留单张脸（多脸/无脸都剔）
    min_ratio = a.min_face_ratio if a.min_face_ratio >= 0 else (0.03 if a.mode == "face" else 0.0)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if a.limit: files = files[:a.limit]
    if not files: sys.exit(f"no images in {src}")
    log(f"[curate] {len(files)} 张 mode={a.mode} max_faces={max_faces or '不限'} min_ratio={min_ratio}")

    tmp = Path(tempfile.mkdtemp(prefix="curate_th_"))
    rows = []
    # ── 1) 缩略图 + 批量人脸检测 ────────────────────────────────
    thumbs = {}
    for i, p in enumerate(files):
        t = tmp / (p.stem + ".jpg")
        if thumb(p, t): thumbs[p] = t
    log(f"[curate] 缩略图 {len(thumbs)} 张；批量人脸检测 ...")
    fmap = batch_faces(facecli, list(thumbs.values()))

    cand = []   # 通过「人脸+画质+去重」的候选（原图路径）
    hashes = []
    for p in files:
        t = thumbs.get(p)
        if t is None:
            rows.append(dict(file=p.name, w=0, h=0, faces=-1, score=0, ratio=0, blur=0, dup=0, verdict="rejected", reason="unreadable", caption="")); continue
        try: w, h = Image.open(p).size
        except Exception: w = h = 0
        faces = fmap.get(str(t), [])
        nf = len(faces)
        best = max(faces, key=lambda f: f.get("score", 0)) if faces else None
        score = float(best.get("score", 0)) if best else 0.0
        ratio = 0.0
        if best:
            try: tw, th = Image.open(t).size
            except Exception: tw = th = 0
            x1, y1, x2, y2 = best["box"]   # 框在缩略图坐标系
            if tw and th: ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(tw * th)
        blur = blur_of(t)
        dh = dhash(t); dup = 0
        if a.dedup:
            for h2 in hashes:
                if ham(dh, h2) <= 6: dup = 1; break
            if not dup: hashes.append(dh)
        reasons = []
        if w and max(w, h) < a.min_side: reasons.append("small")
        if blur >= 0 and blur < a.blur_min: reasons.append("blur")
        if dup: reasons.append("dup")
        if nf < 1: reasons.append("no_face")
        if max_faces and nf > max_faces: reasons.append("many_face")
        if min_ratio and ratio < min_ratio: reasons.append("face_small")
        if faces and score < a.min_face_score: reasons.append("face_lowconf")
        row = dict(file=p.name, w=w, h=h, faces=nf, score=round(score, 3), ratio=round(ratio, 4),
                   blur=round(blur, 1), dup=dup, verdict="", reason="", caption="")
        rows.append(row)
        row["_p"] = p
        if not reasons: cand.append(p)
        else: row["verdict"] = "rejected"; row["reason"] = ",".join(reasons)
    log(f"[curate] 检测后候选 {len(cand)} / {len(files)}")

    # ── 2) 统一裁减 ────────────────────────────────────────────
    outimgs = {}
    if a.mode == "face":
        log(f"[curate] 对齐裁脸({a.crop}) 批量 ...")
        lines = []
        for p in cand:
            o = images_dir / (p.stem + ".png"); outimgs[p] = o
            lines.append(f"{p}\t{o} {a.crop}")
        r = subprocess.run([facecli, "crop", "--batch"], input="\n".join(lines), capture_output=True, text=True)
    else:
        log(f"[curate] 统一尺寸：长边={a.res}" + ("；横图以人脸为锚 crop 竖构图" if not a.no_land_crop else ""))
        for p in cand:
            o = images_dir / (p.stem + ".jpg")
            try:
                im = Image.open(p).convert("RGB"); W, H = im.size
                # 横图(宽>高)：以最大人脸为锚，按人脸的倍数定裁剪窗（左右留余量），保留人物大
                if W > H and not a.no_land_crop:
                    faces = fmap.get(str(thumbs.get(p)), [])
                    if faces:
                        try: tw, th = Image.open(thumbs[p]).size
                        except Exception: tw, th = W, H
                        b = max(faces, key=lambda f: f.get("score", 0))
                        sx = W / float(tw or W); sy = H / float(th or H)
                        x1, y1, x2, y2 = b["box"][0] * sx, b["box"][1] * sy, b["box"][2] * sx, b["box"][3] * sy
                        fw, fh = (x2 - x1), (y2 - y1); cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        ch = min(float(H), max(fh * a.face_scale, H * 0.3))   # 窗高≈脸高×倍数（人尽量大）
                        cw = min(float(W), ch * a.land_aspect)                 # 按目标宽高比 → 左右留出余量
                        x0 = int(max(0, min(cx - cw / 2.0, W - cw)))           # 横向以脸为中心
                        y0 = int(max(0, min(cy - fh * 1.2, H - ch)))           # 纵向留头顶空间
                        im = im.crop((x0, y0, x0 + int(cw), y0 + int(ch)))
                # 宽度对齐到训练宽（=train_width/--res）；高度按原比例（对齐到 16 的倍数）
                tw = a.train_width or a.res
                nh = int(round(im.height * tw / float(im.width) / 16.0) * 16)
                im = im.resize((tw, max(16, nh)), Image.LANCZOS)
                im.save(o, quality=95); outimgs[p] = o
            except Exception:
                pass

    # ── 3) 自然语言打标（批量）+ 写真判断 ──────────────────────
    task = "<CAPTION>" if a.mode == "face" else "<DETAILED_CAPTION>"
    ready = [p for p in cand if outimgs.get(p) and outimgs[p].exists()]
    log(f"[curate] florence 打标 {len(ready)} 张 (task={task}) ...")
    caps = batch_caption(img2prompt, model_dir, task, [outimgs[p] for p in ready], a.beams)

    nk = 0
    for p in cand:
        row = next(r for r in rows if r["_p"] == p)
        o = outimgs.get(p)
        if not (o and o.exists()):
            row["verdict"] = "rejected"; row["reason"] = (row["reason"] or "") + ",crop_failed"
            try: shutil.move(str(p), rej / p.name)
            except Exception: pass
            continue
        cap = caps.get(str(o), "")
        row["caption"] = cap
        reasons = [x for x in row["reason"].split(",") if x]
        if not a.no_photo_check and cap:
            if NONPHOTO.search(cap): reasons.append("not_photo")
            elif not PERSON.search(cap): reasons.append("no_person")
        if reasons:
            row["verdict"] = "rejected"; row["reason"] = ",".join(reasons)
            try: os.remove(o)
            except Exception: pass
        else:
            row["verdict"] = "kept"; row["reason"] = ""; nk += 1
            (images_dir / (o.stem + ".txt")).write_text(f"{a.trigger}, {cap}" if cap else a.trigger, encoding="utf-8")

    # ── 4) 合格图另存 / 删除不合格原图 / dataset.toml + report ──
    if a.copy_to:
        cd = Path(a.copy_to); cd.mkdir(parents=True, exist_ok=True)
        nc = 0
        for f in sorted(images_dir.iterdir()):
            shutil.copy2(f, cd / f.name); nc += 1
        log(f"[curate] 合格图另存 -> {a.copy_to}（{nc} 个文件）")
    if a.rm_rejected:
        nd = 0
        for r in rows:
            if r.get("verdict") == "rejected" and r.get("_p"):
                try:
                    os.remove(r["_p"]); nd += 1
                except Exception:
                    pass
        log(f"[curate] 已删除不合格原图 {nd} 张")
    if a.rm_source:
        nd = 0
        for p in files:
            try:
                os.remove(p); nd += 1
            except Exception:
                pass
        log(f"[curate] 已删除源图 {nd} 张")
    res = a.crop if a.mode == "face" else (a.train_width or a.res)
    (out / "dataset.toml").write_text(
        f'[general]\nresolution = [{res}, {res}]\ncaption_extension = ".txt"\nbatch_size = 1\n'
        f'enable_bucket = true\nbucket_no_upscale = true\n\n[[datasets]]\n'
        f'image_directory = "{images_dir}"\ncache_directory = "{out}/cache"\nnum_repeats = 1\n', encoding="utf-8")
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "w", "h", "faces", "score", "ratio", "blur", "dup", "verdict", "reason", "caption"])
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in w.fieldnames})
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"kept={nk} rejected={len(rows)-nk} -> {out}（images/ + dataset.toml + report.csv）")


if __name__ == "__main__":
    main()
